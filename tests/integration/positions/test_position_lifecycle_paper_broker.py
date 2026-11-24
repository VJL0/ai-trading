"""REAL Alpaca PAPER round trip of the position lifecycle (``@pytest.mark.paper_broker`` + explicit opt-in).

This test PLACES PAPER ORDERS: BUY 1 share of a low-priced allowlisted ETF, a broker-held protective STOP_LIMIT SELL,
a cancel of that stop and a marketable LIMIT SELL exit, all through the production risk engine, signed intents,
ledger, ``AlpacaPaperGateway`` and ``PositionManager``. Paper only (the gateway refuses every non-paper URL); no real
money. It runs only when ALL of these hold, otherwise it is skipped:

* Alpaca PAPER credentials are available (Keychain / environment / git-ignored .env);
* ``LAT_PAPER_ORDER_TESTS=1`` is set (a deliberate opt-in: a paper account is shared state);
* XNYS is in its regular session and not within 45 minutes of the close;
* the paper account has no positions and no open orders (the throwaway ledger must reconcile CLEAN).

Side effect: the orders appear in the paper account's history, so the *main* paper ledger's reconciliation will see
broker orders it does not know (``UNKNOWN_ORDER`` for filled orders in its 5-day lookback) and halt entries until
the operator reviews them. Use a dedicated paper account, or expect that review. A pass here means the lifecycle
ran against the real Alpaca PAPER broker; it is not evidence of profitability.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import (
    DecisionAction,
    FeedMode,
    GatewayKind,
    OrderState,
    OrderType,
    Proposer,
    ReconciliationStatus,
    Side,
    TradingMode,
)
from packages.contracts.models import MarketSnapshot, RiskContext, SystemHealth, TradeProposal
from packages.contracts.policy import RiskPolicy
from packages.core.clock import SystemClock
from packages.core.config import PROJECT_ROOT
from packages.core.db import Database
from packages.core.ids import new_causal_id, new_id
from packages.core.money import round_price
from packages.core.secrets import SecretStore
from packages.core.signing import IntentSigner
from services.market_data.snapshot import build_market_snapshot
from services.order_ledger.entries import EntriesLock
from services.order_ledger.execution import ExecutionService
from services.order_ledger.ledger import OrderLedger
from services.order_ledger.reconcile import Reconciler
from services.order_ledger.sessions import session_bounds
from services.positions.manager import PositionManager
from services.positions.policy import PositionPolicy
from services.positions.store import PositionStore
from services.risk.engine import RiskEngine
from services.risk.policy_store import PolicyStore

pytestmark = [pytest.mark.paper_broker, pytest.mark.slow]
SYMBOL = os.environ.get("LAT_PAPER_TEST_SYMBOL", "XLF")
D = Decimal


class PaperPort:
    mode = TradingMode.PAPER
    dataset_version = "live-iex_free"

    def __init__(self, h: dict[str, Any]) -> None:
        self.h = h

    @property
    def ledger(self) -> OrderLedger:
        return self.h["ledger"]

    @property
    def entries(self) -> EntriesLock:
        return self.h["entries"]

    def now(self) -> datetime:
        return datetime.now(UTC)

    def risk_policy(self) -> RiskPolicy:
        return self.h["policy"]

    async def snapshot(self, symbol: str) -> MarketSnapshot:
        snap = await build_market_snapshot(self.h["provider"], symbol, self.now(), FeedMode.IEX_FREE,
                                           dataset_version=self.dataset_version)
        self.ledger.record_market_snapshot(snap)
        return snap

    async def ctx(self, market: MarketSnapshot) -> RiskContext:
        gw = self.h["gateway"]
        now = self.now()
        account = await gw.get_account()
        positions = await gw.get_positions()
        locked, reasons = self.entries.is_locked()
        xlocked, xreasons = self.entries.exits_locked()
        return RiskContext(
            now=now, mode=TradingMode.PAPER, gateway=GatewayKind.ALPACA_PAPER, account=account,
            positions=tuple(positions), open_orders=tuple(self.ledger.open_orders()),
            recent_submission_times=tuple(self.ledger.recent_submission_times(now - timedelta(days=1))),
            recent_intent_fingerprints=tuple(self.ledger.recent_fingerprints(now - timedelta(seconds=300))),
            start_of_day_equity=account.last_equity, daily_realized_pl=self.ledger.daily_realized_pl(now),
            market=market, health=SystemHealth(
                feed_ok=market.quote is not None, broker_connected=True, reconciled=True, engine_ok=True,
                entries_locked=locked, lock_reasons=tuple(reasons), exits_locked=xlocked,
                exit_lock_reasons=tuple(xreasons)))

    async def submit_proposal(self, proposal: TradeProposal, snapshot: MarketSnapshot) -> dict[str, Any]:
        engine: RiskEngine = self.h["engine"]
        self.ledger.record_proposal(proposal)
        ctx = await self.ctx(snapshot)
        decision = engine.evaluate(proposal, ctx)
        self.ledger.record_decision(decision)
        if not decision.approved:
            return {"outcome": "REJECTED", "reason": "; ".join(decision.reasons)}
        intent = engine.authorize(proposal, decision, ctx)
        self.ledger.record_intent(intent)
        self.h["gate"] = await self.ctx(snapshot)
        await self.h["execution"].submit(intent)
        rec = self.ledger.get_record(intent.client_order_id)
        ok = rec is not None and rec.state != OrderState.REJECTED
        return {"outcome": "SUBMITTED" if ok else "BLOCKED", "client_order_id": intent.client_order_id,
                "reason": None if ok else (rec.state_reason if rec else "missing")}

    async def cancel_order(self, client_order_id: str) -> OrderState | None:
        return await self.h["execution"].cancel(client_order_id)

    async def request_reconcile(self) -> Any:
        return await self.h["reconciler"].reconcile()


@pytest.fixture
def secrets() -> SecretStore:
    store = SecretStore(PROJECT_ROOT)
    if not store.alpaca_paper_available():
        pytest.skip("Alpaca PAPER credentials absent: real paper lifecycle test not run")
    if os.environ.get("LAT_PAPER_ORDER_TESTS") != "1":
        pytest.skip("set LAT_PAPER_ORDER_TESTS=1 to place real PAPER orders (see module docstring)")
    now = datetime.now(UTC)
    bounds = session_bounds(now)
    if bounds is None or not (bounds[0] <= now < bounds[1] - timedelta(minutes=45)):
        pytest.skip("XNYS regular session closed or within 45 minutes of the close")
    return store


async def _poll(h: dict[str, Any], until: Any, timeout_s: float = 60.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        await h["reconciler"].reconcile()
        if until():
            return True
        await asyncio.sleep(2.0)
    return False


async def test_real_paper_entry_protect_exit_round_trip(tmp_path: Path, secrets: SecretStore) -> None:
    from services.market_data.alpaca_provider import AlpacaMarketDataProvider
    from services.order_ledger.alpaca_gateway import AlpacaPaperGateway

    clock = SystemClock()
    db = Database(tmp_path / "ledger.sqlite")
    signer = IntentSigner.load_or_create(tmp_path / "keys")
    ledger = OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier)
    entries = EntriesLock(db, TradingMode.PAPER, clock=clock)
    gateway = AlpacaPaperGateway(secrets, ledger=ledger, verifier=signer.verifier, clock=clock)
    try:
        if await gateway.get_positions() or await gateway.list_orders(status="open"):
            pytest.skip("paper account not flat (positions or open orders): would not reconcile CLEAN")
        reconciler = Reconciler(db, ledger, gateway, entries, clock)
        assert (await reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
        pstore = PolicyStore(db, TradingMode.PAPER)
        pstore.ensure_default()
        policy = pstore.active()
        h: dict[str, Any] = {"ledger": ledger, "entries": entries, "gateway": gateway, "reconciler": reconciler,
                             "policy": policy, "engine": RiskEngine(policy, signer, clock),
                             "provider": AlpacaMarketDataProvider(secrets, FeedMode.IEX_FREE, clock), "gate": None}

        def gate(intent: Any) -> list[str]:
            ctx, h["gate"] = h["gate"], None
            return ["no gate context"] if ctx is None else h["engine"].final_gate(
                intent, ctx, signer.verifier, pstore.active_version())

        h["execution"] = ExecutionService(ledger, gateway, gate, clock, entries_lock=entries)
        port = PaperPort(h)
        manager = PositionManager(port, PositionStore(db, "PAPER"), PositionPolicy())
        snap = await port.snapshot(SYMBOL)
        assert snap.quote is not None, "no IEX quote"
        limit = round_price(snap.quote.ask, side_buy=True)
        stop = round_price(limit * D("0.985"), side_buy=False)
        now = port.now()
        buy = TradeProposal(
            proposal_id=new_id("prop"), causal_id=new_causal_id(), created_at=now, data_asof=snap.data_asof,
            action=DecisionAction.TRADE, proposer=Proposer.TEST, strategy_id="paper-lifecycle-test", symbol=SYMBOL,
            side=Side.BUY, quantity=D(1), limit_price=limit, stop_price=stop, max_planned_loss=limit - stop,
            evidence_ids=("paper-lifecycle-test",), dataset_version=port.dataset_version)
        out = await port.submit_proposal(buy, snap)
        assert out["outcome"] == "SUBMITTED", out
        assert await _poll(h, lambda: ledger.position_quantities().get(SYMBOL) == D(1)), "entry did not fill in 60s"
        await manager.tick()
        assert await _poll(h, lambda: any(r.is_protective and r.state == OrderState.ACKNOWLEDGED
                                          for r in ledger.records())), "protective stop not acknowledged"
        plan = manager.store.open_plan(SYMBOL)
        assert plan is not None
        manager.store.update_plan(plan.plan_id, port.now(), exit_reason="TIME", exit_started_at=port.now())
        for _ in range(30):
            await manager.tick()
            await reconciler.reconcile()
            if ledger.position_quantities().get(SYMBOL) is None and manager.store.open_plan(SYMBOL) is None:
                break
            await asyncio.sleep(2.0)
        assert ledger.position_quantities().get(SYMBOL) is None, "position not flat"
        closed = manager.store.closed_plans(limit=1)[0]
        assert closed.close_reason == "TIME" and closed.exit_quantity == D(1)
        sells = [r for r in ledger.records() if r.side == Side.SELL]
        assert sells[0].order_type == OrderType.STOP_LIMIT and sells[0].state == OrderState.CANCELED
        assert sells[-1].order_type == OrderType.LIMIT and sells[-1].state == OrderState.FILLED
        assert (await reconciler.reconcile()).status == ReconciliationStatus.CLEAN
    finally:
        await gateway.aclose()
        db.close()
