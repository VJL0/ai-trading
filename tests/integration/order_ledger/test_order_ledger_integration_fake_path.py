"""End-to-end order path on the deterministic FakeBroker built by ``build_gateway`` (SIMULATED).

Every test here ran against the deterministic in-process FakeBroker (GatewayKind.FAKE, SIMULATED fills, state
persisted to ``var/paper/fake_broker_state.json`` so it outlives a 'process' like a real broker) -- NOT against a
real Alpaca paper broker. The real XNYS calendar decides the regular session; the ledger is a real SQLite WAL file.

Wiring mirrors ``services/api/runtime.py``: signer keys -> ``build_gateway`` -> ``ExecutionService`` (final gate,
async ``run_event_consumer`` task) -> ``Reconciler`` (startup recovery + periodic runs); a restart is a fresh set of
objects on the same SQLite file and broker state file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from packages.contracts.enums import GatewayKind, OrderState, ReconciliationStatus, Side, TradingMode
from packages.contracts.models import ApprovedOrderIntent
from packages.core.audit import verify_chain
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretStore
from packages.core.signing import IntentSigner
from services.order_ledger.entries import PROCESS_GUARD_REASON, EntriesLock
from services.order_ledger.execution import ExecutionService
from services.order_ledger.fake_broker import FakeBroker, QuoteBook
from services.order_ledger.gateway_factory import build_gateway
from services.order_ledger.ledger import OrderLedger
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import T0, IntentFactory, make_quote

S = OrderState
NEXT_DAY = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)  # Thursday 10:00 New York
NEXT_DAY_AFTER_CLOSE = datetime(2026, 11, 5, 21, 0, 1, tzinfo=UTC)


def _no_secrets() -> SecretStore:
    return SecretStore(Path("/nonexistent"), use_keychain=False, env={}, dotenv_path=Path("/nonexistent/.env"))


async def until(pred: Callable[[], bool], within_s: float = 3.0) -> None:
    for _ in range(int(within_s / 0.005)):
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached in time")


@dataclass
class Process:
    """One 'process lifetime' of the order path (what the API runtime holds)."""

    db: Database
    ledger: OrderLedger
    entries: EntriesLock
    broker: FakeBroker
    svc: ExecutionService
    rec: Reconciler
    factory: IntentFactory
    stop: asyncio.Event
    consumer: asyncio.Task[None]

    @classmethod
    async def start(cls, settings: Settings, clock: ManualClock, quotes: QuoteBook, signer: IntentSigner) -> Process:
        db = Database(settings.db_path)
        ledger = OrderLedger(db, clock, settings.mode, verifier=signer.verifier)
        entries = EntriesLock(db, settings.mode, clock=clock)
        gw = build_gateway(settings, _no_secrets(), ledger=ledger, clock=clock, quote_source=quotes,
                           max_fill_per_match=Decimal(4), poll_interval_s=0.002)
        assert isinstance(gw, FakeBroker) and gw.verifier is not None

        def final_gate(intent: ApprovedOrderIntent) -> list[str]:
            out: list[str] = []
            if not signer.verifier.verify(intent.signing_string(), intent.signature):
                out.append("signature does not verify")
            if clock.now() >= intent.expires_at:
                out.append("intent expired")
            return out

        svc = ExecutionService(ledger, gw, final_gate, clock, entries_lock=entries, submit_timeout_s=2.0,
                               watchdog_interval_s=3600)
        rec = Reconciler(db, ledger, gw, entries, clock, confirm_delay_s=0)
        stop = asyncio.Event()
        consumer = asyncio.create_task(svc.run_event_consumer(stop))
        await asyncio.wait_for(svc.consumer_ready.wait(), 2)
        factory = IntentFactory(ledger, signer, clock, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, ttl_s=120)
        return cls(db, ledger, entries, gw, svc, rec, factory, stop, consumer)

    def state(self, coid: str) -> OrderState:
        rec = self.ledger.get_record(coid)
        assert rec is not None
        return rec.state

    async def shutdown(self) -> None:
        self.stop.set()
        await asyncio.wait_for(self.consumer, 2)
        await self.broker.aclose()
        self.db.close()


async def test_signed_intent_fill_reconcile_restart_and_day_expiry(tmp_path: Path) -> None:
    """FakeBroker (SIMULATED): locked until startup reconciliation; a resting BUY fills in three partials on later
    quotes through the async consumer; a SELL realizes P&L; a 'process restart' is locked again until its own CLEAN
    reconciliation (positions/cash carried at the broker and in the ledger); a next-day DAY order expires at the
    close with the position carried; every submission reached the broker exactly once."""
    settings = Settings(project_root=tmp_path, use_keychain=False, gateway=GatewayKind.FAKE)
    settings.ensure_dirs()
    signer = IntentSigner.load_or_create(settings.keys_dir)
    clock = ManualClock(T0)
    quotes = QuoteBook()
    quotes.set(make_quote("SPY", "100.00", "100.05", T0))

    p = await Process.start(settings, clock, quotes, signer)
    # ---- entries are locked until this process has a CLEAN broker reconciliation
    locked, reasons = p.entries.is_locked()
    assert locked and PROCESS_GUARD_REASON in reasons
    early = p.factory.make(limit="99.50")
    assert await p.svc.submit(early) is None
    assert p.state(early.client_order_id) == S.REJECTED and p.broker.submission_count() == 0
    report = await p.rec.startup_recovery()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert p.entries.is_locked() == (False, [])

    # ---- resting BUY, filled 4 + 4 + 2 on later quotes (fills at the ask, never above the limit)
    buy = p.factory.make(limit="99.50")
    order = await p.svc.submit(buy)
    assert order is not None and order.broker_order_id
    await until(lambda: p.state(buy.client_order_id) == S.ACKNOWLEDGED)
    for _ in range(3):
        clock.advance(1)
        q = make_quote("SPY", "99.40", "99.45", clock.now())
        quotes.set(q)
        p.broker.advance(q)
    await until(lambda: p.state(buy.client_order_id) == S.FILLED)
    fills = p.ledger.fills(buy.client_order_id)
    assert [f["quantity"] for f in fills] == ["4", "4", "2"] and {f["price"] for f in fills} == {"99.45"}
    assert len({f["broker_exec_id"] for f in fills}) == 3
    [pos] = p.ledger.positions()
    assert pos.quantity == Decimal(10) and pos.avg_entry_price == Decimal("99.45")
    assert (await p.rec.reconcile()).status == ReconciliationStatus.CLEAN

    # ---- SELL 4 against a higher bid: realized P&L 4 * (99.60 - 99.45) = 0.60
    clock.advance(1)
    quotes.set(make_quote("SPY", "99.60", "99.65", clock.now()))
    sell = p.factory.make(side=Side.SELL, qty="4", limit="99.40")
    assert await p.svc.submit(sell) is not None
    # the synchronous response already says 'filled'; the execution report (with the fill) arrives on the stream
    assert p.state(sell.client_order_id) == S.FILLED
    await until(lambda: len(p.ledger.fills(sell.client_order_id)) == 1)
    assert p.ledger.realized_pl_total("SPY") == Decimal("0.6")
    assert p.ledger.daily_realized_pl(clock.now()) == Decimal("0.6")
    assert (await p.rec.reconcile()).status == ReconciliationStatus.CLEAN
    assert p.broker.submission_count() == 2
    await p.shutdown()

    # ---- restart: same SQLite ledger, broker re-created from its own persisted state
    p = await Process.start(settings, clock, quotes, signer)
    locked, reasons = p.entries.is_locked()
    assert locked and PROCESS_GUARD_REASON in reasons
    blocked = p.factory.make(limit="99.50")
    assert await p.svc.submit(blocked) is None and p.broker.submission_count() == 0
    report = await p.rec.startup_recovery()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert not p.entries.is_locked()[0]
    [pos] = p.ledger.positions()
    [bpos] = await p.broker.get_positions()
    assert pos.quantity == bpos.quantity == Decimal(6)
    account = await p.broker.get_account()
    assert account.cash == Decimal("100000") - Decimal("994.50") + Decimal("398.40")

    # ---- next session: a resting DAY order expires at the close; the position carries over
    clock.set(NEXT_DAY)
    quotes.set(make_quote("SPY", "99.00", "99.05", clock.now()))
    day = p.factory.make(qty="5", limit="90.00")
    assert await p.svc.submit(day) is not None
    await until(lambda: p.state(day.client_order_id) == S.ACKNOWLEDGED)
    clock.set(NEXT_DAY_AFTER_CLOSE)
    p.broker.advance(None)
    await until(lambda: p.state(day.client_order_id) == S.EXPIRED)
    assert p.ledger.fills(day.client_order_id) == []
    assert p.ledger.position_quantities() == {"SPY": Decimal(6)}
    final = await p.rec.reconcile()
    assert final.status == ReconciliationStatus.CLEAN, final.mismatches
    assert p.broker.submission_count(day.client_order_id) == 1

    # ---- evidence: intact audit chain, every gateway decision recorded, re-deliveries never re-applied
    ok, rows, detail = verify_chain(p.db)
    assert ok and rows > 0, detail
    decisions = p.db.query("SELECT result, detail FROM mode_transitions ORDER BY seq")
    assert [d["result"] for d in decisions] == ["ACCEPTED", "ACCEPTED"]
    assert all("SIMULATED" in d["detail"] for d in decisions)
    # the only re-deliveries are identical broker snapshots seen again by later reconciliation runs (no-ops)
    dup_keys = [r["dedupe_key"] for r in p.db.query("SELECT dedupe_key FROM order_event_duplicates")]
    assert all(k.startswith("event:reconcile-") for k in dup_keys), dup_keys
    assert len(p.ledger.fills(buy.client_order_id)) == 3  # nothing booked twice by those re-deliveries
    await p.shutdown()
