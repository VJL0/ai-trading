"""Reconciler against the deterministic FakeBroker (SIMULATED; never a real broker)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from packages.contracts.enums import OrderEventType, ReconciliationStatus
from packages.contracts.enums import OrderState as S
from packages.contracts.models import AccountSnapshot
from packages.core.ids import new_id
from services.order_ledger.entries import Flag
from services.order_ledger.execution import ExecutionService
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import T0, Harness, flags, make_quote, pass_gate, state_of


def _rec(h: Harness, **kw: object) -> Reconciler:
    kw.setdefault("confirm_delay_s", 0)
    return Reconciler(h.db, h.ledger, h.broker, h.entries, h.clock, **kw)  # type: ignore[arg-type]


async def test_clean_run_persists_snapshots_and_report(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    rec = _rec(h)
    first = await rec.startup_recovery()
    assert first.status == ReconciliationStatus.CLEAN and not first.entries_locked_after
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries)
    intent = h.factory.make()
    await svc.submit(intent)
    for ev in h.broker.drain_events():
        svc.handle_event(ev)
    report = await rec.reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    run = h.db.query_one("SELECT * FROM reconciliation_runs WHERE run_id=?", (report.run_id,))
    assert run is not None and run["status"] == "CLEAN" and run["entries_locked_after"] == 0
    assert h.db.query_one("SELECT COUNT(*) n FROM position_snapshots WHERE snapshot_group=?", (report.run_id,))["n"] == 1  # type: ignore[index]
    assert h.db.query_one("SELECT COUNT(*) n FROM account_snapshots")["n"] == 2  # type: ignore[index]
    assert h.db.query_one("SELECT COUNT(*) n FROM audit_events WHERE event_type='RECONCILIATION_RUN'")["n"] == 2  # type: ignore[index]


async def test_unexpected_broker_position_locks_entries_with_halt(h: Harness) -> None:
    await _rec(h).startup_recovery()
    h.broker.seed_position("AAPL", Decimal(5), Decimal("200"))
    report = await _rec(h).reconcile()
    assert report.status == ReconciliationStatus.MISMATCH and report.entries_locked_after
    assert {m.kind for m in report.mismatches} == {"UNEXPECTED_POSITION"}
    assert Flag.RECONCILIATION_MISMATCH.value in flags(h)
    assert len(h.entries.halts()) == 1
    # still mismatched next time: one halt only, still locked
    again = await _rec(h).reconcile()
    assert again.status == ReconciliationStatus.MISMATCH and len(h.entries.halts()) == 1


async def test_broker_order_unknown_to_ledger_is_mismatch(h: Harness) -> None:
    await _rec(h).startup_recovery()
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    foreign = h.factory.make(limit="99.00", record=False)  # placed outside the signed-intent gateway path
    await h.broker.submit(foreign)
    report = await _rec(h).reconcile()
    assert report.status == ReconciliationStatus.MISMATCH
    assert [m.kind for m in report.mismatches] == ["UNKNOWN_ORDER"]
    assert h.entries.is_locked()[0]


async def test_unknown_order_found_at_broker_is_resolved(h: Harness) -> None:
    await _rec(h).startup_recovery()
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    h.broker.injector.timeout_after_accept = 1
    h.broker.injector.fail_lookups = 1  # the immediate lookup fails too: stays UNKNOWN
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries)
    intent = h.factory.make()
    assert await svc.submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.UNKNOWN
    report = await _rec(h).reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert report.resolved_unknown_orders == (intent.client_order_id,)
    assert state_of(h, intent.client_order_id) == S.FILLED
    assert h.ledger.positions()[0].quantity == Decimal(10)  # catch-up from broker cumulative qty
    assert not h.entries.is_locked()[0]
    assert h.broker.submission_count() == 1


async def test_unknown_order_absent_at_broker_becomes_reconciled_after_grace(h: Harness) -> None:
    await _rec(h).startup_recovery()
    h.broker.injector.partition_before_accept = 1
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries)
    intent = h.factory.make()
    await svc.submit(intent)
    within = await _rec(h, unknown_grace_s=60).reconcile()
    assert within.status == ReconciliationStatus.MISMATCH
    assert [m.kind for m in within.mismatches] == ["UNRESOLVED_UNKNOWN_ORDER"]
    assert state_of(h, intent.client_order_id) == S.UNKNOWN
    h.clock.advance(61)
    after = await _rec(h, unknown_grace_s=60).reconcile()
    assert after.status == ReconciliationStatus.CLEAN
    assert state_of(h, intent.client_order_id) == S.RECONCILED
    assert h.ledger.fills(intent.client_order_id) == []  # no fills invented
    assert h.broker.submission_count() == 1  # never resubmitted
    assert not h.entries.is_locked()[0]


async def test_missed_fill_events_are_caught_up_from_broker(h: Harness) -> None:
    await _rec(h).startup_recovery()
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    h.broker.injector.drop_event_types = frozenset({OrderEventType.FILL})
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries)
    intent = h.factory.make()
    await svc.submit(intent)
    for ev in h.broker.drain_events():
        svc.handle_event(ev)
    assert h.ledger.positions() == []
    report = await _rec(h).reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    [pos] = h.ledger.positions()
    assert pos.quantity == Decimal(10) and pos.cost_basis == Decimal("1000.00")
    # the dropped fill is redelivered later: absorbed, not double-booked
    for ev in h.broker.dropped_events:
        svc.handle_event(ev)
    assert h.ledger.positions()[0].quantity == Decimal(10)


async def test_broker_unavailable_is_error_and_locks(h: Harness) -> None:
    await _rec(h).startup_recovery()
    h.broker.injector.fail_reads = 1
    report = await _rec(h).reconcile()
    assert report.status == ReconciliationStatus.ERROR and report.entries_locked_after
    assert Flag.BROKER_DISCONNECTED.value in flags(h)
    assert (await _rec(h).reconcile()).status == ReconciliationStatus.CLEAN
    assert not h.entries.is_locked()[0]


async def test_blocked_or_non_paper_account_is_mismatch(h: Harness) -> None:
    await _rec(h).startup_recovery()
    real = h.broker.get_account

    async def live_account() -> AccountSnapshot:
        acct = await real()
        return acct.model_copy(update={"is_paper": False, "snapshot_id": new_id("acct")})

    h.broker.get_account = live_account  # type: ignore[method-assign]
    report = await _rec(h).reconcile()
    assert "ACCOUNT_NOT_PAPER" in {m.kind for m in report.mismatches}
    assert h.entries.halts()

    async def blocked() -> AccountSnapshot:
        acct = await real()
        return acct.model_copy(update={"trading_blocked": True, "snapshot_id": new_id("acct")})

    h.broker.get_account = blocked  # type: ignore[method-assign]
    assert "ACCOUNT_BLOCKED" in {m.kind for m in (await _rec(h).reconcile()).mismatches}


async def test_startup_recovery_treats_submitting_as_unknown(h: Harness) -> None:
    intent = h.factory.make()
    h.ledger.claim_for_submission(intent.client_order_id, "crashed-process")
    report = await _rec(h, unknown_grace_s=0).startup_recovery()
    assert intent.client_order_id in report.resolved_unknown_orders
    assert state_of(h, intent.client_order_id) == S.RECONCILED  # broker never saw it
    to_states = [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)]
    assert to_states[-2:] == ["UNKNOWN", "RECONCILED"]
    assert h.broker.submit_attempts == []


async def test_in_flight_submission_is_not_touched_by_periodic_reconcile(h: Harness) -> None:
    await _rec(h).startup_recovery()
    intent = h.factory.make()
    h.ledger.claim_for_submission(intent.client_order_id, "in-flight")
    report = await _rec(h, unknown_grace_s=60).reconcile()
    assert report.status == ReconciliationStatus.CLEAN
    assert state_of(h, intent.client_order_id) == S.SUBMITTING
    h.clock.advance(61)
    await _rec(h, unknown_grace_s=60).reconcile()
    assert state_of(h, intent.client_order_id) in (S.UNKNOWN, S.RECONCILED)


async def test_expired_intents_are_cleaned_up(h: Harness) -> None:
    intent = h.factory.make(ttl_s=10)
    h.clock.advance(timedelta(seconds=11).total_seconds())
    report = await _rec(h).startup_recovery()
    assert report.status == ReconciliationStatus.CLEAN
    assert state_of(h, intent.client_order_id) == S.EXPIRED


async def test_in_doubt_order_of_another_gateway_is_never_resolved_by_this_reader(h: Harness) -> None:
    """An UNKNOWN FakeBroker order must not be declared 'not present at broker' by a LEAN reader (wrong broker)."""
    from packages.contracts.enums import GatewayKind
    from services.order_ledger.bridge import LeanBridgeGateway

    await _rec(h).startup_recovery()
    h.broker.injector.partition_before_accept = 1
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries)
    intent = h.factory.make()
    await svc.submit(intent)
    assert state_of(h, intent.client_order_id) == S.UNKNOWN
    lean = LeanBridgeGateway(h.ledger, GatewayKind.LEAN_PAPER_SIM, h.clock, wall_clock=h.clock)
    h.clock.advance(600)
    report = await Reconciler(h.db, h.ledger, lean, h.entries, h.clock, confirm_delay_s=0).reconcile()
    assert report.status == ReconciliationStatus.MISMATCH and report.entries_locked_after
    assert "GATEWAY_MISMATCH" in {m.kind for m in report.mismatches}
    assert state_of(h, intent.client_order_id) == S.UNKNOWN  # not RECONCILED by the wrong broker
    # the right broker resolves it
    fixed = await _rec(h).reconcile()
    assert state_of(h, intent.client_order_id) == S.RECONCILED and fixed.status == ReconciliationStatus.CLEAN
