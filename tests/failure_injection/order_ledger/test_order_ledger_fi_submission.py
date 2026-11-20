"""Failure injection around order submission, the event stream and restarts.

Every test here ran against the deterministic in-process FakeBroker (GatewayKind.FAKE, SIMULATED fills) with a
ManualClock -- NOT against a real Alpaca paper broker.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from packages.contracts.enums import OrderEventType, OrderState, ReconciliationStatus
from packages.core.db import Database
from services.order_ledger.entries import PROCESS_GUARD_REASON, EntriesLock, Flag
from services.order_ledger.errors import SimulatedCrash
from services.order_ledger.fake_broker import FakeBroker
from services.order_ledger.ledger import OrderLedger
from tests.failure_injection.order_ledger.conftest import System, build_system
from tests.unit.order_ledger.ol_support import T0, Harness, always_open, make_harness, make_quote, state_of

S = OrderState
E = OrderEventType


async def test_network_partition_during_submit(system: System) -> None:
    """FakeBroker: the request never reaches the broker. UNKNOWN, entries locked, resolved only by reconciliation
    ("not present at broker" after the grace period); no fills invented; exactly one submission attempt."""
    h = system.h
    await system.ready()
    h.broker.injector.partition_before_accept = 1
    intent = h.factory.make()
    assert await system.svc.submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.UNKNOWN
    assert h.entries.is_locked()[0] and h.entries.is_flag_active(Flag.UNKNOWN_ORDERS_PRESENT)
    early = await system.rec.reconcile()
    assert early.status == ReconciliationStatus.MISMATCH and early.entries_locked_after
    h.clock.advance(61)
    late = await system.rec.reconcile()
    assert late.status == ReconciliationStatus.CLEAN and intent.client_order_id in late.resolved_unknown_orders
    assert state_of(h, intent.client_order_id) == S.RECONCILED
    assert h.ledger.fills(intent.client_order_id) == [] and h.ledger.positions() == []
    assert h.broker.submission_count() == 1 and h.broker.accepted_submissions == []
    assert not h.entries.is_locked()[0]


async def test_network_partition_after_accept_resolved_by_client_id(system: System) -> None:
    """FakeBroker: accepted at the broker, response lost (SubmitTimeout). Marked UNKNOWN, then resolved by the
    immediate client-order-id lookup; entries stay locked until a CLEAN reconciliation."""
    h = system.h
    await system.ready()
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    h.broker.injector.timeout_after_accept = 1
    intent = h.factory.make()
    found = await system.svc.submit(intent)
    assert found is not None and found.state == S.FILLED
    assert "UNKNOWN" in [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)]
    assert h.entries.is_locked()[0]
    system.pump()
    report = await system.rec.reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert h.ledger.positions()[0].quantity == Decimal(10)
    assert h.broker.submission_count() == 1
    assert not h.entries.is_locked()[0]


async def test_unknown_outcome_is_never_automatically_resubmitted(system: System) -> None:
    """FakeBroker: after an unknown outcome every further submit of the same intent is refused -- before,
    during and after reconciliation. The broker sees exactly one submission."""
    h = system.h
    await system.ready()
    h.broker.injector.timeout_before_accept = 1
    intent = h.factory.make()
    await system.svc.submit(intent)
    for _ in range(3):
        assert await system.svc.submit(intent) is None
    h.clock.advance(120)
    await system.rec.reconcile()
    assert await system.svc.submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.RECONCILED
    assert h.broker.submission_count(intent.client_order_id) == 1
    refused = h.db.query("SELECT payload_json FROM audit_events WHERE event_type='SUBMIT_REFUSED'")
    assert len(refused) == 4


async def test_broker_rate_limit_on_submit_is_not_retried(system: System) -> None:
    """FakeBroker: HTTP-429 style refusal on submit => REJECTED (the order was not accepted), never retried."""
    h = system.h
    await system.ready()
    h.broker.injector.rate_limit_next = 1
    intent = h.factory.make()
    assert await system.svc.submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.broker.submission_count() == 1 and h.broker.accepted_submissions == []
    audit = h.db.query("SELECT payload_json FROM audit_events WHERE event_type='SUBMIT_REJECTED'")
    assert len(audit) == 1 and '"retried":false' in audit[0]["payload_json"] and "429" in audit[0]["payload_json"]


async def test_crash_after_broker_accepts_then_restart(tmp_path: Path) -> None:
    """FakeBroker (state persisted to disk, i.e. it outlives our process like a real broker): the process
    'crashes' right after the broker accepted the order. On restart the SUBMITTING order is treated as UNKNOWN,
    resolved by client-order-id lookup, the fill is caught up from broker state, exactly one submission."""
    state = tmp_path / "broker_state.json"
    h = make_harness(tmp_path, state_path=state)
    sysm = build_system(h)
    await sysm.ready()
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    h.broker.injector.crash_after_accept = 1
    intent = h.factory.make()
    try:
        await sysm.svc.submit(intent)
        raise AssertionError("expected the simulated crash")
    except SimulatedCrash:
        pass
    assert state_of(h, intent.client_order_id) == S.SUBMITTING  # nothing recorded after the broker accepted
    # --- restart: new process objects on the same SQLite file; the broker is re-created from its own state
    h.db.close()
    db = Database(tmp_path / "ledger.sqlite")
    ledger = OrderLedger(db, h.clock, h.ledger.mode)
    entries = EntriesLock(db, h.ledger.mode, clock=h.clock)
    broker = FakeBroker(h.clock, h.quotes, session_bounds_fn=always_open, verifier=h.signer.verifier, state_path=state)
    restarted = build_system(Harness(tmp_path, db, h.clock, ledger, entries, h.signer, h.factory, h.quotes, broker))
    try:
        locked, reasons = entries.is_locked()
        assert locked and PROCESS_GUARD_REASON in reasons
        report = await restarted.rec.startup_recovery()
        assert report.status == ReconciliationStatus.CLEAN, report.mismatches
        assert intent.client_order_id in report.resolved_unknown_orders
        rec = ledger.get_record(intent.client_order_id)
        assert rec is not None and rec.state == S.FILLED
        to_states = [t["to_state"] for t in ledger.transitions(intent.client_order_id)]
        assert to_states[-2:] == ["UNKNOWN", "FILLED"]
        assert ledger.positions()[0].quantity == Decimal(10)
        assert broker.submission_count() == 0 and h.broker.submission_count() == 1  # never resubmitted
        assert broker.accepted_submissions == [] and len(broker.orders) == 1
        assert not entries.is_locked()[0]
    finally:
        db.close()


async def test_stream_disconnect_and_reconnect_keeps_entries_locked_until_reconcile(system: System) -> None:
    """FakeBroker: the order-update stream drops; a fill happens while disconnected (missed for good, as with a
    websocket). Entries are locked from the disconnect until a CLEAN reconciliation books the missed fill."""
    h = system.h
    await system.ready()
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    svc = system.svc
    stop = asyncio.Event()
    consumer = asyncio.create_task(svc.run_event_consumer(stop))
    await asyncio.wait_for(svc.consumer_ready.wait(), 1)
    intent = h.factory.make(limit="99.50")
    await svc.submit(intent)
    for _ in range(100):
        await asyncio.sleep(0.01)
        if state_of(h, intent.client_order_id) == S.ACKNOWLEDGED:
            break
    assert state_of(h, intent.client_order_id) == S.ACKNOWLEDGED
    h.broker.disconnect_stream()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if svc.disconnects:
            break
    assert h.entries.is_flag_active(Flag.BROKER_DISCONNECTED) and h.entries.is_locked()[0]
    h.clock.advance(5)
    h.broker.advance(make_quote("SPY", "99.40", "99.45", h.clock.now()))  # fills while we are disconnected
    h.broker.reconnect_stream()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if svc.consumer_ready.is_set():
            break
    await asyncio.sleep(0.05)
    assert state_of(h, intent.client_order_id) == S.ACKNOWLEDGED  # the missed fill is not replayed by the stream
    assert h.entries.is_locked()[0], "still locked after reconnect: missed events must be reconciled first"
    report = await system.rec.reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    assert state_of(h, intent.client_order_id) == S.FILLED
    assert h.ledger.positions()[0].quantity == Decimal(10)
    assert not h.entries.is_locked()[0]
    stop.set()
    await asyncio.wait_for(consumer, 2)


async def test_duplicate_events(system: System) -> None:
    """FakeBroker: every event delivered twice (and a full re-delivery). Booked once."""
    h = system.h
    await system.ready()
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    h.broker.injector.duplicate_next_events = 10
    intent = h.factory.make()
    await system.svc.submit(intent)
    evs = h.broker.drain_events()
    assert len(evs) == 6
    for ev in evs + evs:
        system.svc.handle_event(ev)
    assert system.svc.events_duplicate == 9 and system.svc.events_applied == 3
    assert h.ledger.positions()[0].quantity == Decimal(10)
    assert len(h.ledger.fills(intent.client_order_id)) == 1
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN


async def test_out_of_order_events(system: System) -> None:
    """FakeBroker: events delivered in reverse order (fill before new/accepted), and partial fills reordered."""
    h = system.h
    await system.ready()
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    h.broker.max_fill_per_match = Decimal(3)
    h.broker.injector.reorder_window = 6
    intent = h.factory.make(limit="99.90")
    await system.svc.submit(intent)  # accepted + new buffered
    for i in range(4):
        h.clock.advance(1)
        h.broker.advance(make_quote("SPY", "99.80", f"99.8{i}", h.clock.now()))
    h.broker.flush_reordered()
    evs = h.broker.drain_events()
    assert evs[0].event_type == E.FILL and evs[-1].event_type == E.ACCEPTED
    for ev in evs:
        system.svc.handle_event(ev)
    order = h.ledger.get_order(intent.client_order_id)
    assert order is not None and order.state == S.FILLED and order.filled_quantity == Decimal(10)
    assert [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)][-1] == "FILLED"
    stale = [e for e in h.ledger.events(intent.client_order_id) if not e["applied"]]
    assert {e["event_type"] for e in stale} >= {"new", "accepted"}
    assert (await system.rec.reconcile()).status == ReconciliationStatus.CLEAN
