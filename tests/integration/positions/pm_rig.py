"""Real-component rig for the position lifecycle (SIMULATED: deterministic in-process FakeBroker, no network).

Wiring mirrors ``services.api.runtime``: PolicyStore -> RiskEngine.evaluate/authorize -> OrderLedger.record_intent ->
fresh RiskContext (broker account/positions + ledger orders + EntriesLock entry *and* exit locks) ->
ExecutionService.submit with ``RiskEngine.final_gate`` as the boundary callback. The PositionManager talks to it only
through ``PositionPort`` (``RigPort``), exactly like the runtime adapter. Broker events are pumped into the ledger
explicitly so every test is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import (
    DataFeed,
    DecisionAction,
    FeedMode,
    GatewayKind,
    MarketSession,
    OrderState,
    OrderType,
    Proposer,
    Side,
    TradingMode,
)
from packages.contracts.models import ApprovedOrderIntent, MarketSnapshot, Quote, RiskContext, SystemHealth, TradeProposal
from packages.contracts.policy import RiskPolicy
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.ids import new_causal_id, new_id
from packages.core.signing import IntentSigner
from services.order_ledger.entries import EntriesLock
from services.order_ledger.execution import ExecutionService
from services.order_ledger.fake_broker import FakeBroker, QuoteBook
from services.order_ledger.ledger import OrderLedger, OrderRecord
from services.order_ledger.reconcile import Reconciler
from services.positions.manager import PositionManager
from services.positions.policy import PositionPolicy
from services.positions.store import PositionStore
from services.risk.engine import RiskEngine
from services.risk.operator_auth import OperatorAuth
from services.risk.policy_store import PolicyStore

# Thursday 2026-11-05 10:00 America/New_York (EST, UTC-5): inside the XNYS regular session 09:30-16:00.
OPEN_T = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)
CLOSE_T = datetime(2026, 11, 5, 21, 0, tzinfo=UTC)
NEXT_OPEN_T = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)


def ny(hh: int, mm: int, day: int = 5) -> datetime:
    """A New York wall time on 2026-11-<day> (EST, UTC-5)."""
    return datetime(2026, 11, day, hh + 5, mm, tzinfo=UTC)


@dataclass
class Rig:
    tmp: Path
    db: Database
    clock: ManualClock
    policy_store: PolicyStore
    auth: OperatorAuth
    signer: IntentSigner
    engine: RiskEngine
    ledger: OrderLedger
    entries: EntriesLock
    quotes: QuoteBook
    broker: FakeBroker
    execution: ExecutionService
    reconciler: Reconciler
    store: PositionStore
    manager: PositionManager
    port: RigPort
    sod_equity: Decimal
    bid: Decimal = Decimal("500.00")
    ask: Decimal = Decimal("500.04")
    size: Decimal = Decimal("500")
    feed_live: bool = True
    gate_results: list[list[str]] = field(default_factory=list)
    reconcile_calls: int = 0

    # ------------------------------------------------------------------ market
    def quote_now(self, symbol: str = "SPY") -> Quote:
        now = self.clock.now()
        return Quote(symbol=symbol, bid=self.bid, ask=self.ask, bid_size=self.size, ask_size=self.size, exchange_ts=now,
                     received_ts=now, available_at_ts=now, feed=DataFeed.SYNTHETIC, source_id="rig-quotes",
                     is_synthetic=True)

    def set_quote(self, bid: str, ask: str, *, size: str | None = None, symbol: str = "SPY") -> None:
        """New market quote: resting orders (protective stops, exits) may fill on it."""
        self.bid, self.ask = Decimal(bid), Decimal(ask)
        if size is not None:
            self.size = Decimal(size)
        q = self.quote_now(symbol)
        self.quotes.set(q)
        self.broker.advance(q)
        self.pump()

    def refresh_feed(self, symbol: str = "SPY") -> None:
        if self.feed_live:
            q = self.quote_now(symbol)
            self.quotes.set(q)
            self.broker.advance(q)
            self.pump()

    def snapshot(self, symbol: str = "SPY") -> MarketSnapshot:
        q = self.quotes(symbol)
        now = self.clock.now()
        return MarketSnapshot(
            snapshot_id=new_id("snap"), symbol=symbol, feed=DataFeed.SYNTHETIC, feed_mode=FeedMode.SIMULATED,
            session=MarketSession.REGULAR, quote=q, data_asof=now, source_id="rig", dataset_version="ds-test",
            is_simulated=True)

    # ------------------------------------------------------------------ plumbing
    def pump(self) -> int:
        n = 0
        for ev in self.broker.drain_events():
            self.execution.handle_event(ev)
            n += 1
        return n

    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self.broker.expire_day_orders()
        self.pump()

    def set_time(self, when: datetime) -> None:
        self.clock.set(when)
        self.broker.expire_day_orders()
        self.pump()

    async def health(self) -> SystemHealth:
        locked, reasons = self.entries.is_locked()
        xlocked, xreasons = self.entries.exits_locked()
        return SystemHealth(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True, entries_locked=locked,
                            lock_reasons=tuple(reasons), exits_locked=xlocked, exit_lock_reasons=tuple(xreasons))

    async def ctx(self, market: MarketSnapshot) -> RiskContext:
        now = self.clock.now()
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        policy = self.policy_store.active()
        return RiskContext(
            now=now, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, account=account, positions=tuple(positions),
            open_orders=tuple(self.ledger.open_orders()),
            recent_submission_times=tuple(self.ledger.recent_submission_times(now - timedelta(days=1))),
            recent_intent_fingerprints=tuple(self.ledger.recent_fingerprints(
                now - timedelta(seconds=policy.duplicate_window_seconds))),
            start_of_day_equity=self.sod_equity, daily_realized_pl=self.ledger.daily_realized_pl(now),
            daily_unrealized_pl=sum((p.unrealized_pl or Decimal(0) for p in positions), Decimal(0)),
            market=market, health=await self.health())

    _gate: RiskContext | None = None

    def final_gate(self, intent: ApprovedOrderIntent) -> list[str]:
        ctx, self._gate = self._gate, None
        if ctx is None:
            return ["no fresh risk context captured for the final gate"]
        row = self.db.query_one("SELECT payload_json FROM proposals WHERE proposal_id=?", (intent.proposal_id,))
        proposal = TradeProposal.model_validate_json(row["payload_json"]) if row else None
        result = self.engine.final_gate(intent, ctx, self.signer.verifier, self.policy_store.active_version(),
                                        proposal=proposal)
        self.gate_results.append(result)
        return result

    async def execute(self, proposal: TradeProposal, snapshot: MarketSnapshot) -> dict[str, Any]:
        """Same steps as ``Runtime.execute_proposal``."""
        self.ledger.record_market_snapshot(snapshot)
        self.ledger.record_proposal(proposal)
        out: dict[str, Any] = {"causal_id": proposal.causal_id}
        ctx = await self.ctx(snapshot)
        decision = self.engine.evaluate(proposal, ctx)
        self.ledger.record_decision(decision)
        out["decision"] = decision
        if not decision.approved:
            out.update(outcome="REJECTED", reason="; ".join(decision.reasons))
            return out
        intent = self.engine.authorize(proposal, decision, ctx)
        self.ledger.record_intent(intent)
        out["client_order_id"] = intent.client_order_id
        self._gate = await self.ctx(snapshot)
        await self.execution.submit(intent)
        self.pump()
        rec = self.ledger.get_record(intent.client_order_id)
        state = rec.state if rec else None
        out["outcome"] = "BLOCKED" if state in (None, OrderState.REJECTED) else "SUBMITTED"
        if out["outcome"] == "BLOCKED":
            out["reason"] = rec.state_reason if rec else "missing"
        return out

    # ------------------------------------------------------------------ actions
    def entry_proposal(self, qty: str = "3", *, stop: str | None = "495.00", symbol: str = "SPY",
                       limit: str | None = None, signal_id: str | None = None, holding_sessions: int | None = None,
                       strategy_id: str = "test-entry") -> TradeProposal:
        now = self.clock.now()
        lim = Decimal(limit) if limit is not None else self.ask
        st = Decimal(stop) if stop is not None else None
        return TradeProposal(
            proposal_id=new_id("prop"), causal_id=new_causal_id(), created_at=now, data_asof=now,
            action=DecisionAction.TRADE, proposer=Proposer.TEST, strategy_id=strategy_id, symbol=symbol, side=Side.BUY,
            quantity=Decimal(qty), order_type=OrderType.LIMIT, limit_price=lim, stop_price=st,
            max_planned_loss=Decimal(qty) * (lim - (st or lim * Decimal("0.985"))), evidence_ids=("ev_test",),
            signal_id=signal_id, dataset_version="ds-test", holding_sessions=holding_sessions)

    async def buy(self, qty: str = "3", **kw: Any) -> dict[str, Any]:
        self.refresh_feed()
        return await self.execute(self.entry_proposal(qty, **kw), self.snapshot())

    async def tick(self, n: int = 1) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(n):
            self.refresh_feed()
            out.extend(await self.manager.tick())
            self.pump()
        return out

    async def reconcile(self) -> Any:
        self.reconcile_calls += 1
        report = await self.reconciler.reconcile()
        self.pump()
        return report

    # ------------------------------------------------------------------ queries
    def position(self, symbol: str = "SPY") -> Decimal:
        return self.ledger.position_quantities().get(symbol, Decimal(0))

    def open_records(self, symbol: str = "SPY") -> list[OrderRecord]:
        from services.order_ledger.state_machine import TERMINAL

        return [r for r in self.ledger.records() if r.symbol == symbol and r.state not in TERMINAL]

    def protective(self, symbol: str = "SPY") -> list[OrderRecord]:
        return [r for r in self.open_records(symbol) if r.is_protective]

    def sells(self, symbol: str = "SPY") -> list[OrderRecord]:
        return [r for r in self.ledger.records() if r.symbol == symbol and r.side == Side.SELL]

    def safety(self) -> dict[str, str]:
        return {s["symbol"]: s["state"] for s in self.entries.position_safety()}

    def actions(self, symbol: str = "SPY") -> list[dict[str, Any]]:
        return list(reversed(self.store.actions(symbol, limit=500)))

    def reopen(self) -> Rig:
        """Process restart: new Database/ledger/lock/engine/execution/manager objects on the same files; the same
        (external) broker. The in-memory process-start guard therefore starts locked again."""
        self.db.close()
        return build_rig(self.tmp, clock=self.clock, broker=self.broker, quotes=self.quotes, signer=self.signer,
                         policy=self.manager.policy, sod_equity=self.sod_equity, bid=self.bid, ask=self.ask,
                         size=self.size, recover=False)


class RigPort:
    mode = TradingMode.PAPER
    dataset_version = "ds-test"

    def __init__(self, rig: Rig) -> None:
        self.rig = rig

    @property
    def ledger(self) -> OrderLedger:
        return self.rig.ledger

    @property
    def entries(self) -> EntriesLock:
        return self.rig.entries

    def now(self) -> datetime:
        return self.rig.clock.now()

    def risk_policy(self) -> RiskPolicy:
        return self.rig.policy_store.active()

    async def snapshot(self, symbol: str) -> MarketSnapshot:
        return self.rig.snapshot(symbol)

    async def submit_proposal(self, proposal: TradeProposal, snapshot: MarketSnapshot) -> dict[str, Any]:
        return await self.rig.execute(proposal, snapshot)

    async def cancel_order(self, client_order_id: str) -> OrderState | None:
        state = await self.rig.execution.cancel(client_order_id)
        self.rig.pump()
        return state

    async def request_reconcile(self) -> Any:
        return await self.rig.reconcile()


def build_rig(tmp: Path, *, clock: ManualClock | None = None, broker: FakeBroker | None = None,
              quotes: QuoteBook | None = None, signer: IntentSigner | None = None, policy: PositionPolicy | None = None,
              risk_policy: RiskPolicy | None = None, sod_equity: Decimal = Decimal("10000"),
              bid: Decimal = Decimal("500.00"), ask: Decimal = Decimal("500.04"), size: Decimal = Decimal("500"),
              recover: bool = True, **broker_kw: Any) -> Rig:
    var = tmp / "var" / "paper"
    db = Database(var / "ledger.sqlite")
    clock = clock or ManualClock(OPEN_T)
    auth = OperatorAuth(var / "keys")
    auth.ensure_token()
    signer = signer or IntentSigner.load_or_create(var / "keys")
    pstore = PolicyStore(db, TradingMode.PAPER, clock=clock)
    pstore.ensure_default()
    if risk_policy is not None and pstore.active() != risk_policy:
        pstore.apply_change(risk_policy, operator_token=auth.token_path.read_text().strip(), actor="victor",
                            reason="test policy")
    ledger = OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier)
    entries = EntriesLock(db, TradingMode.PAPER, clock=clock)
    quotes = quotes or QuoteBook()
    if broker is None:
        broker = FakeBroker(clock, quotes, starting_cash=sod_equity, verifier=signer.verifier,
                            claim_check=lambda coid: (r := ledger.get_record(coid)) is not None
                            and r.state == OrderState.SUBMITTING, **broker_kw)
    else:
        broker.claim_check = lambda coid: (r := ledger.get_record(coid)) is not None and r.state == OrderState.SUBMITTING
    reconciler = Reconciler(db, ledger, broker, entries, clock, confirm_delay_s=0, unknown_grace_s=0)
    engine = RiskEngine(pstore.active(), signer, clock)
    store = PositionStore(db, TradingMode.PAPER.value)
    rig = Rig(tmp=tmp, db=db, clock=clock, policy_store=pstore, auth=auth, signer=signer, engine=engine, ledger=ledger,
              entries=entries, quotes=quotes, broker=broker, execution=None, reconciler=reconciler,  # type: ignore[arg-type]
              store=store, manager=None, port=None, sod_equity=sod_equity, bid=bid, ask=ask, size=size)  # type: ignore[arg-type]
    rig.execution = ExecutionService(ledger, broker, rig.final_gate, clock, entries_lock=entries)
    rig.port = RigPort(rig)
    rig.manager = PositionManager(rig.port, store, policy or PositionPolicy())
    rig.quotes.set(rig.quote_now())
    return rig
