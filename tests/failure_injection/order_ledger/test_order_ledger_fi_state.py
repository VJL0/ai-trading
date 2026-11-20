"""Failure injection around broker *state*: empty/malformed/rate-limited reads, cross-day expiry, event-log replay,
Mac sleep, unexpected positions and cancel/fill races.

Unless a docstring says otherwise, every test ran against the deterministic in-process FakeBroker
(GatewayKind.FAKE, SIMULATED fills) with a ManualClock -- NOT against a real Alpaca paper broker. Tests that
exercise the Alpaca PAPER reader ran it against an in-memory fake HTTP session (alpaca-py's real request
building and our mapping, no network) -- also NOT a real paper broker.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, OrderEventType, OrderState, ReconciliationStatus, Side
from packages.contracts.models import OrderEvent
from packages.core.clock import ManualClock
from services.order_ledger.alpaca_reader import AlpacaPaperStateReader
from services.order_ledger.bridge import LeanBridgeGateway, ingest_engine_event, record_heartbeat
from services.order_ledger.entries import PROCESS_GUARD_REASON, Flag
from services.order_ledger.errors import UnlockDenied
from services.order_ledger.execution import ExecutionService
from services.order_ledger.reconcile import Reconciler
from tests.failure_injection.order_ledger.conftest import System, build_system
from tests.unit.order_ledger.ol_support import (
    ACCOUNT,
    CLOSE_T0,
    T0,
    FakeSession,
    Harness,
    IntentFactory,
    alpaca_routes,
    claim,
    make_harness,
    make_quote,
    paper_secret_store,
    pass_gate,
    state_of,
    stream_event,
)

S = OrderState
E = OrderEventType


def _counts(h: Harness) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in ("orders", "order_events", "order_state_transitions", "fills", "ledger_positions"):
        row = h.db.query_one(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608 - fixed table names
        out[table] = int(row["n"]) if row else -1
    return out


def _book(h: Harness) -> list[tuple[str, str, str, str]]:
    """Order states + filled quantities + positions + realized P&L: what replay must not change."""
    orders = [(r.client_order_id, r.state.value, format(r.filled_quantity, "f"), "") for r in h.ledger.records()]
    pos = [(p.symbol, "", format(p.quantity, "f"), format(p.cost_basis, "f")) for p in h.ledger.positions()]
    return [*orders, *pos, ("realized", "", format(h.ledger.realized_pl_total(), "f"), "")]


# ------------------------------------------------------------------------------------------ Alpaca-backed rig


class AlpacaRig:
    """LEAN_ALPACA_PAPER gateway whose broker-state reads go through AlpacaPaperStateReader -> fake HTTP session."""

    def __init__(self, h: Harness, routes: dict[str, list[tuple[int, Any]]], **reader_kw: Any) -> None:
        self.h = h
        self.session = FakeSession(routes)
        self.sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.reader = AlpacaPaperStateReader(paper_secret_store(), http_session=self.session, sleep=fake_sleep,
                                             clock=h.clock, **reader_kw)
        self.gw = LeanBridgeGateway(h.ledger, GatewayKind.LEAN_ALPACA_PAPER, h.clock, state_reader=self.reader,
                                    wall_clock=h.clock)
        self.rec = Reconciler(h.db, h.ledger, self.gw, h.entries, h.clock, confirm_delay_s=0)

    def beat(self) -> None:
        assert record_heartbeat(self.h.db, {"engine_id": "lean-1", "mode": "PAPER", "gateway": "LEAN_ALPACA_PAPER",
                                            "ts": "2026-11-04T15:00:00Z", "status": "RUNNING"}, clock=self.h.clock)["ok"]


def _empty_alpaca_routes() -> dict[str, list[tuple[int, Any]]]:
    routes = alpaca_routes()
    routes["/positions"] = [(200, [])]
    routes["/orders"] = [(200, [])]
    return routes


# ------------------------------------------------------------------------------------------ empty account


async def test_empty_account_state(tmp_path: Path) -> None:
    """FakeBroker with a zero-cash, empty account: reconciliation is CLEAN (nothing to disagree about), a buy is
    rejected by the broker for buying power (REJECTED, never retried, nothing booked)."""
    h = make_harness(tmp_path, starting_cash=Decimal(0))
    system = build_system(h)
    try:
        report = await system.startup()
        assert report.status == ReconciliationStatus.CLEAN and not report.entries_locked_after
        acct = h.db.query_one("SELECT equity, payload_json FROM account_snapshots")
        assert acct is not None and Decimal(acct["equity"]) == 0
        assert h.db.query_one("SELECT COUNT(*) n FROM position_snapshots")["n"] == 0  # type: ignore[index]
        assert h.ledger.positions() == [] and h.ledger.open_orders() == []
        h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
        intent = h.factory.make()
        assert await system.svc.submit(intent) is None
        assert state_of(h, intent.client_order_id) == S.REJECTED
        assert h.broker.submission_count() == 1 and h.broker.accepted_submissions == []
        assert h.ledger.fills(intent.client_order_id) == []
        assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
    finally:
        h.db.close()


async def test_empty_broker_account_but_ledger_holds_position_locks(system: System) -> None:
    """FakeBroker reports an empty account while the ledger booked a position (e.g. broker state was reset):
    MISSING_POSITION => MISMATCH, operator halt, entries locked. The ledger position is not silently deleted."""
    h = system.h
    await system.ready()
    intent = h.factory.make()
    claim(h, intent)
    coid = intent.client_order_id
    h.ledger.apply_event(stream_event(coid, E.NEW, S.ACKNOWLEDGED, T0, event_id="x-new"))
    h.ledger.apply_event(stream_event(coid, E.FILL, S.FILLED, T0, event_id="x-fill", fill=("10", "100.00"), cum="10",
                                      exec_id="x-exec-1"))
    report = await system.rec.reconcile()
    assert report.status == ReconciliationStatus.MISMATCH and report.entries_locked_after
    assert "MISSING_POSITION" in {m.kind for m in report.mismatches}
    assert h.entries.halts() and h.entries.is_locked()[0]
    assert h.ledger.positions()[0].quantity == Decimal(10)  # never "fixed" by deleting local evidence


async def test_empty_alpaca_paper_account_is_clean(h: Harness) -> None:
    """AlpacaPaperStateReader over a fake HTTP session (not a real paper broker): an empty paper account (no
    positions, no orders) reconciles CLEAN and releases the startup lock."""
    rig = AlpacaRig(h, _empty_alpaca_routes())
    rig.beat()
    assert PROCESS_GUARD_REASON in h.entries.is_locked()[1]
    report = await rig.rec.startup_recovery()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert not h.entries.is_locked()[0]
    assert {m for m, _, _ in rig.session.calls} == {"GET"}


# ------------------------------------------------------------------------------------------ malformed payloads


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/positions", [{"symbol": "SPY", "qty": "abc", "avg_entry_price": "1", "cost_basis": "1"}]),
        ("/positions", {"not": "a list"}),
        ("/positions", [{"symbol": "BTCUSD", "asset_class": "crypto", "qty": "1", "avg_entry_price": "1",
                         "cost_basis": "1"}]),
        ("/account", {**ACCOUNT, "equity": None}),
        ("/account", "garbage"),
        ("/orders", [{"client_order_id": "x", "symbol": "SPY", "status": "new", "order_type": "stop", "type": "stop",
                      "side": "buy", "time_in_force": "day", "qty": "1"}]),
        ("/orders", [{"client_order_id": "x", "symbol": "SPY", "status": "replaced", "order_type": "limit",
                      "side": "buy", "time_in_force": "day", "qty": "1"}]),
    ],
)
async def test_malformed_broker_payloads_fail_closed(h: Harness, route: str, body: Any) -> None:
    """AlpacaPaperStateReader over a fake HTTP session (not a real paper broker): malformed or unsupported broker
    payloads make reconciliation ERROR (entries stay locked, broker_disconnected raised) instead of being
    guessed at or silently degraded. Once the broker answers sanely again, a CLEAN run unlocks."""
    routes = _empty_alpaca_routes()
    routes[route] = [(200, body), (200, [] if route != "/account" else dict(ACCOUNT))]
    rig = AlpacaRig(h, routes)
    rig.beat()
    report = await rig.rec.startup_recovery()
    assert report.status == ReconciliationStatus.ERROR and report.entries_locked_after
    assert h.entries.is_flag_active(Flag.BROKER_DISCONNECTED) and h.entries.is_locked()[0]
    failures = h.db.query("SELECT kind, detail FROM failures WHERE component='reconciler'")
    assert failures and "BrokerPayloadError" in failures[-1]["detail"]
    assert (await rig.rec.reconcile()).status == ReconciliationStatus.CLEAN
    assert not h.entries.is_locked()[0]


async def test_malformed_stream_and_engine_payloads_are_recorded_and_lock(system: System) -> None:
    """FakeBroker stream delivering a malformed object, plus engine-bridge payloads (scripted engine, no LEAN process)
    carrying binary floats: nothing malformed is applied, everything is recorded as a failure, and the stream case
    locks entries (ledger_anomaly). JSON *text* numbers are parsed exactly as Decimal (parse_float=Decimal)."""
    h = system.h
    await system.ready()
    h.broker.event_log.append({"event": "fill", "qty": 1.5})  # type: ignore[arg-type]
    system.pump()
    assert h.entries.is_flag_active(Flag.LEDGER_ANOMALY) and h.entries.is_locked()[0]
    assert h.db.query("SELECT 1 FROM failures WHERE component='execution' AND kind='event_apply_failed'")
    lean = IntentFactory(h.ledger, h.signer, h.clock, gateway=GatewayKind.LEAN_PAPER_SIM).make()
    claim(h, lean)
    base: dict[str, Any] = {
        "engine_id": "lean-1", "client_order_id": lean.client_order_id, "intent_id": lean.intent_id,
        "lean_order_id": 1, "status": "Filled", "event_ts": "2026-11-04T15:00:01Z", "lean_event_id": 1,
    }
    # an already-parsed body holding binary floats is refused before touching the ledger
    res = ingest_engine_event(h.ledger, {**base, "fill_quantity": "10", "fill_price": 100.5, "cum_filled_quantity": "10"},
                              gateway_kind=GatewayKind.LEAN_PAPER_SIM)
    assert res["accepted"] is False and "float" in res["note"]
    for broken in ({**base, "lean_order_id": "seven"}, {**base, "event_ts": "yesterday"}, {**base, "status": "Exploded"},
                   {k: v for k, v in base.items() if k != "client_order_id"}):
        assert ingest_engine_event(h.ledger, broken, gateway_kind=GatewayKind.LEAN_PAPER_SIM)["accepted"] is False
    assert ingest_engine_event(h.ledger, b"{not json", gateway_kind=GatewayKind.LEAN_PAPER_SIM)["accepted"] is False
    assert state_of(h, lean.client_order_id) == S.SUBMITTING and h.ledger.fills(lean.client_order_id) == []
    assert len(h.db.query("SELECT 1 FROM failures WHERE component='lean_bridge'")) >= 6
    # raw JSON text: numbers become exact Decimals, never binary floats
    raw = (f'{{"engine_id":"lean-1","client_order_id":"{lean.client_order_id}","intent_id":"{lean.intent_id}",'
           '"lean_order_id":1,"status":"Filled","fill_quantity":10,"fill_price":100.10,"cum_filled_quantity":10,'
           '"event_ts":"2026-11-04T15:00:01Z","lean_event_id":2}')
    ok = ingest_engine_event(h.ledger, raw, gateway_kind=GatewayKind.LEAN_PAPER_SIM)
    assert ok["accepted"] is True and ok["state"] == "FILLED"
    [fill] = h.ledger.fills(lean.client_order_id)
    assert fill["price"] == "100.10" and fill["quantity"] == "10"


# ------------------------------------------------------------------------------------------ rate limits


async def test_broker_rate_limits_on_reads_are_bounded_and_fail_closed(h: Harness) -> None:
    """AlpacaPaperStateReader over a fake HTTP session (not a real paper broker): HTTP 429 on reads is retried with
    bounded backoff (reads only); still limited after the bound => reconciliation ERROR, entries locked. A later
    run that gets through after two 429s is CLEAN."""
    limited = (429, {"code": 42910000, "message": "rate limit exceeded"})
    routes = _empty_alpaca_routes()
    routes["/account"] = [limited, limited, limited, limited, limited, limited, (200, dict(ACCOUNT))]
    rig = AlpacaRig(h, routes, max_attempts=3, backoff_s=(0.1, 0.2))
    rig.beat()
    first = await rig.rec.startup_recovery()
    assert first.status == ReconciliationStatus.ERROR and "BrokerRateLimited" in first.detail
    assert rig.sleeps == [0.1, 0.2] and h.entries.is_locked()[0]
    second = await rig.rec.reconcile()  # three more 429s: still ERROR, bounded again
    assert second.status == ReconciliationStatus.ERROR and rig.sleeps == [0.1, 0.2, 0.1, 0.2]
    third = await rig.rec.reconcile()  # 200 now
    assert third.status == ReconciliationStatus.CLEAN, third.mismatches
    assert not h.entries.is_locked()[0]
    assert {m for m, _, _ in rig.session.calls} == {"GET"}  # a rate limit never turns into a write


# ------------------------------------------------------------------------------------------ cross-day


async def test_cross_day_day_order_expiry_with_position_carry(system: System) -> None:
    """FakeBroker: a DAY limit partially fills, expires at the close (no fills invented for the rest), the partial
    position carries overnight, and the next day's sell realizes P&L on the next trading day only."""
    h = system.h
    await system.ready()
    h.broker.max_fill_per_match = Decimal(4)
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    buy = h.factory.make(qty="10", limit="100.02")
    await system.svc.submit(buy)
    h.clock.advance(60)
    h.broker.advance(make_quote("SPY", "99.99", "100.01", h.clock.now()))  # partial 4 @ 100.01
    system.pump()
    assert state_of(h, buy.client_order_id) == S.PARTIALLY_FILLED
    h.clock.set(CLOSE_T0 + timedelta(seconds=1))
    h.broker.advance()  # session close: DAY order expires at the broker
    system.pump()
    rec = h.ledger.get_record(buy.client_order_id)
    assert rec is not None and rec.state == S.EXPIRED and rec.filled_quantity == Decimal(4)
    assert len(h.ledger.fills(buy.client_order_id)) == 1
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
    # ---- next trading day: position carried, sell it
    day2 = T0 + timedelta(days=1)
    h.clock.set(day2)
    [pos] = h.ledger.positions()
    assert pos.quantity == Decimal(4) and pos.cost_basis == Decimal("400.04")
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
    h.quotes.set(make_quote("SPY", "100.60", "100.65", day2))
    sell = h.factory.make(side=Side.SELL, qty="4", limit="100.50")
    order = await system.svc.submit(sell)
    assert order is not None
    system.pump()
    assert state_of(h, sell.client_order_id) == S.FILLED and h.ledger.positions() == []
    assert h.ledger.daily_realized_pl(T0) == Decimal(0)
    assert h.ledger.daily_realized_pl(day2) == Decimal("2.36")  # 4 * (100.60 - 100.01)
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
    assert not h.entries.is_locked()[0]


# ------------------------------------------------------------------------------------------ replay


async def test_replay_of_the_whole_event_log_is_idempotent(tmp_path: Path) -> None:
    """FakeBroker: after fills, a partial fill + cancel and a rejection, replaying (a) every stored ledger event and
    (b) the broker's entire stream -- in this process and again after a restart -- changes nothing."""
    h = make_harness(tmp_path)
    system = build_system(h)
    await system.ready()
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    filled = h.factory.make(qty="5", limit="100.00")
    await system.svc.submit(filled)
    h.broker.max_fill_per_match = Decimal(2)
    h.quotes.set(make_quote("QQQ", "400.00", "400.10", T0))
    partial = h.factory.make(symbol="QQQ", qty="6", limit="400.10")
    await system.svc.submit(partial)
    await system.svc.cancel(partial.client_order_id)
    h.broker.injector.reject_next = 1
    rejected = h.factory.make(symbol="SPY", qty="1", limit="100.00")
    await system.svc.submit(rejected)
    system.pump()
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
    before_book, before_counts = _book(h), _counts(h)
    assert state_of(h, partial.client_order_id) == S.CANCELED
    for ev in h.ledger.event_log():
        system.svc.handle_event(ev)
    for ev in h.broker.event_log:
        system.svc.handle_event(ev)
    assert _book(h) == before_book and _counts(h) == before_counts
    assert h.ledger.duplicate_count() >= len(h.broker.event_log)
    # ---- restart and replay again
    h2 = h.reopen()
    try:
        s2 = build_system(h2)
        for ev in h2.ledger.event_log():
            assert h2.ledger.apply_event(ev).duplicate
        for ev in h2.broker.event_log:
            s2.svc.handle_event(ev)
        assert _book(h2) == before_book and _counts(h2) == before_counts
        report = await s2.rec.startup_recovery()
        assert report.status == ReconciliationStatus.CLEAN, report.mismatches
        assert _book(h2) == before_book
        assert h2.broker.submission_count() == 3  # replay never re-sent anything
    finally:
        h2.db.close()


# ------------------------------------------------------------------------------------------ Mac sleep


async def test_mac_sleep_clock_jump_locks_until_broker_reconciliation(system: System) -> None:
    """FakeBroker + ManualClock jump standing in for Mac sleep: the watchdog sees the wall clock jump, entries lock,
    a new order is refused locally (nothing sent), and only a broker-authoritative reconciliation -- which books
    the fill that happened at the broker while we slept -- unlocks. No fill is invented."""
    h = system.h
    wall = ManualClock(T0)
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries, wall_clock=wall,
                           watchdog_interval_s=5, suspend_gap_s=30)
    await system.ready()
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    resting = h.factory.make(limit="99.50")
    await svc.submit(resting)
    for ev in h.broker.drain_events():
        svc.handle_event(ev)
    assert state_of(h, resting.client_order_id) == S.ACKNOWLEDGED
    assert svc.watchdog_tick() == []
    # ---- the Mac sleeps 30 minutes; the broker keeps working; our process sees nothing
    wall.advance(1800)
    h.clock.advance(1800)
    h.broker.advance(make_quote("SPY", "99.40", "99.45", h.clock.now()))
    missed = h.broker.drain_events()
    assert [e.event_type for e in missed] == [E.FILL]
    # ---- wake up
    assert svc.watchdog_tick() == [Flag.PROCESS_SUSPENDED.value]
    locked, reasons = h.entries.is_locked()
    assert locked and any("process_suspended" in r for r in reasons)
    attempts = h.broker.submission_count()
    h.quotes.set(make_quote("SPY", "99.40", "99.45", h.clock.now()))
    blocked = h.factory.make(limit="99.45")
    assert await svc.submit(blocked) is None
    assert state_of(h, blocked.client_order_id) == S.REJECTED and h.broker.submission_count() == attempts
    assert state_of(h, resting.client_order_id) == S.ACKNOWLEDGED  # not guessed
    report = await system.rec.reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert state_of(h, resting.client_order_id) == S.FILLED
    assert h.ledger.positions()[0].quantity == Decimal(10)
    [fill] = h.ledger.fills(resting.client_order_id)
    assert fill["synthetic"] == 1 and Decimal(fill["price"]) == Decimal("99.45")  # broker-authoritative catch-up
    assert not h.entries.is_locked()[0]


async def test_mac_sleep_stale_engine_heartbeat_locks_until_reconcile(h: Harness) -> None:
    """LEAN_PAPER_SIM bridge with a scripted engine (no real LEAN process, no broker): after a 1h wall-clock jump the
    engine heartbeat is stale => engine_heartbeat_stale + process_suspended; a reconciliation while the engine is
    still silent keeps entries locked; a fresh heartbeat plus a CLEAN reconciliation unlocks."""
    wall = ManualClock(T0)
    gw = LeanBridgeGateway(h.ledger, GatewayKind.LEAN_PAPER_SIM, h.clock, wall_clock=wall)

    def beat() -> None:
        assert record_heartbeat(h.db, {"engine_id": "lean-1", "mode": "PAPER", "gateway": "LEAN_PAPER_SIM",
                                       "ts": "2026-11-04T15:00:00Z", "status": "RUNNING"}, clock=wall)["ok"]

    beat()
    svc = ExecutionService(h.ledger, gw, pass_gate, h.clock, entries_lock=h.entries, wall_clock=wall,
                           watchdog_interval_s=5, suspend_gap_s=30)
    rec = Reconciler(h.db, h.ledger, gw, h.entries, h.clock, confirm_delay_s=0)
    assert (await rec.startup_recovery()).status == ReconciliationStatus.CLEAN
    assert svc.watchdog_tick() == [] and not h.entries.is_locked()[0]
    wall.advance(3600)
    h.clock.advance(3600)
    raised = svc.watchdog_tick()
    assert set(raised) == {Flag.PROCESS_SUSPENDED.value, Flag.ENGINE_HEARTBEAT_STALE.value}
    assert h.entries.is_locked()[0]
    still = await rec.reconcile()
    assert still.entries_locked_after and h.entries.is_flag_active(Flag.ENGINE_HEARTBEAT_STALE)
    assert not h.entries.is_flag_active(Flag.PROCESS_SUSPENDED)  # broker side reconciled; engine still silent
    beat()
    assert svc.watchdog_tick() == []
    after = await rec.reconcile()
    assert after.status == ReconciliationStatus.CLEAN and not after.entries_locked_after
    assert not h.entries.is_locked()[0]


# ------------------------------------------------------------------------------------------ unexpected position


async def test_unexpected_broker_position_locks_entries_until_operator_clears(system: System) -> None:
    """FakeBroker holds a position the ledger never booked (placed outside this system): MISMATCH, halt, new orders
    refused locally (nothing sent). Flattening it makes reconciliation CLEAN again, but the operator halt still
    needs explicit operator clearance -- which is refused while the latest reconciliation is not CLEAN."""
    h = system.h
    await system.ready()
    h.broker.seed_position("AAPL", Decimal(5), Decimal("200"))
    report = await system.rec.reconcile()
    assert report.status == ReconciliationStatus.MISMATCH and report.entries_locked_after
    assert [m.kind for m in report.mismatches] == ["UNEXPECTED_POSITION"]
    [halt] = h.entries.halts()
    with pytest.raises(UnlockDenied):
        h.entries.clear_halt(halt["halt_id"], by="operator", reason="looks fine")
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    intent = h.factory.make()
    assert await system.svc.submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.REJECTED and h.broker.submission_count() == 0
    h.broker.positions["AAPL"].qty = Decimal(0)  # operator flattened it at the broker out-of-band
    h.broker.positions["AAPL"].cost = Decimal(0)
    clean = await system.rec.reconcile()
    assert clean.status == ReconciliationStatus.CLEAN and clean.entries_locked_after  # halt still active
    locked, reasons = h.entries.is_locked()
    assert locked and any(halt["halt_id"] in r for r in reasons)
    assert h.entries.clear_halt(halt["halt_id"], by="operator", reason="stray AAPL position flattened")
    assert not h.entries.is_locked()[0]


# ------------------------------------------------------------------------------------------ cancel/fill race


async def test_cancel_fill_race_with_reordered_events_books_the_fill(system: System) -> None:
    """FakeBroker: the order executes while our cancel is in flight and the broker's events arrive reversed (FILL
    before PENDING_CANCEL). The fill is booked, the order ends FILLED, the stale cancel event never regresses it."""
    h = system.h
    await system.ready()
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    intent = h.factory.make(limit="99.00")
    await system.svc.submit(intent)
    system.pump()
    h.broker.injector.fill_on_cancel = 1
    h.broker.injector.reorder_window = 2
    state = await system.svc.cancel(intent.client_order_id)
    assert state == S.CANCEL_PENDING
    evs: list[OrderEvent] = h.broker.drain_events()
    assert [e.event_type for e in evs] == [E.FILL, E.PENDING_CANCEL]
    for ev in evs:
        system.svc.handle_event(ev)
    rec = h.ledger.get_record(intent.client_order_id)
    assert rec is not None and rec.state == S.FILLED and rec.filled_quantity == Decimal(10)
    assert h.ledger.positions()[0].quantity == Decimal(10)
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
