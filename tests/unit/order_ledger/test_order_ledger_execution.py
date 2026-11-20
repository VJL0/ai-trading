"""ExecutionService: single order path, final gate, UNKNOWN on timeout, never blind retry, cancel, consumer.

All tests run against the deterministic in-process FakeBroker (SIMULATED), never a real broker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, OrderEventType, OrderState
from packages.contracts.models import ApprovedOrderIntent, BrokerOrder, OrderEvent
from packages.core.clock import ManualClock
from services.order_ledger.entries import Flag
from services.order_ledger.errors import CancelNotSupported
from services.order_ledger.execution import ExecutionService
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import T0, Harness, flags, make_quote, pass_gate, state_of

S = OrderState


async def _unlock(h: Harness) -> None:
    report = await Reconciler(h.db, h.ledger, h.broker, h.entries, h.clock, confirm_delay_s=0).reconcile()
    assert not report.entries_locked_after, report


def _svc(h: Harness, gate: Any = pass_gate, **kw: Any) -> ExecutionService:
    kw.setdefault("submit_timeout_s", 0.5)
    kw.setdefault("lookup_timeout_s", 0.5)
    return ExecutionService(h.ledger, h.broker, gate, h.clock, entries_lock=h.entries, **kw)


async def test_submit_happy_path(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    intent = h.factory.make()
    order = await _svc(h).submit(intent)
    assert order is not None and order.state == S.FILLED
    assert h.broker.submission_count(intent.client_order_id) == 1
    rec = h.ledger.get_record(intent.client_order_id)
    assert rec is not None and rec.state == S.FILLED and rec.broker_order_id == order.broker_order_id
    assert rec.submitted_at is not None


async def test_final_gate_violation_rejects_locally_and_sends_nothing(h: Harness) -> None:
    await _unlock(h)
    intent = h.factory.make()
    out = await _svc(h, gate=lambda _i: ["quote stale", "policy changed"]).submit(intent)
    assert out is None
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.broker.submit_attempts == []
    audit = h.db.query("SELECT payload_json FROM audit_events WHERE event_type='FINAL_GATE_REJECTED'")
    assert len(audit) == 1 and "quote stale" in audit[0]["payload_json"]


async def test_final_gate_exception_fails_closed(h: Harness) -> None:
    await _unlock(h)
    intent = h.factory.make()

    def boom(_i: ApprovedOrderIntent) -> list[str]:
        raise RuntimeError("gate crashed")

    assert await _svc(h, gate=boom).submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.broker.submit_attempts == []


async def test_entries_locked_blocks_submission(h: Harness) -> None:
    intent = h.factory.make()  # no reconciliation yet => locked
    assert await _svc(h).submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.broker.submit_attempts == []


async def test_tampered_or_unrecorded_intent_is_refused(h: Harness) -> None:
    await _unlock(h)
    intent = h.factory.make()
    tampered = intent.model_copy(update={"quantity": Decimal("1000")})
    assert await _svc(h).submit(tampered) is None
    assert state_of(h, intent.client_order_id) == S.REJECTED
    unrecorded = h.factory.make(record=False)
    assert await _svc(h).submit(unrecorded) is None
    assert h.broker.submit_attempts == []


async def test_same_intent_is_never_submitted_twice(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    intent = h.factory.make()
    svc = _svc(h)
    first = await svc.submit(intent)
    second = await svc.submit(intent)
    assert first is not None and second is None
    assert h.broker.submission_count() == 1


@pytest.mark.parametrize("hook", ["reject_next", "rate_limit_next"])
async def test_gateway_rejection_and_rate_limit_are_terminal_never_retried(h: Harness, hook: str) -> None:
    await _unlock(h)
    setattr(h.broker.injector, hook, 1)
    intent = h.factory.make()
    assert await _svc(h).submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.broker.submission_count(intent.client_order_id) == 1


async def test_timeout_after_accept_marks_unknown_then_resolves_by_lookup(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    h.broker.injector.timeout_after_accept = 1
    intent = h.factory.make()
    found = await _svc(h).submit(intent)
    assert found is not None and found.client_order_id == intent.client_order_id
    assert h.broker.submission_count(intent.client_order_id) == 1
    rec = h.ledger.get_record(intent.client_order_id)
    assert rec is not None and rec.state == S.FILLED  # broker truth applied via lookup
    to_states = [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)]
    assert "UNKNOWN" in to_states
    assert Flag.UNKNOWN_ORDERS_PRESENT.value in flags(h)  # entries stay locked until reconciliation
    assert h.entries.is_locked()[0]


async def test_hung_submit_times_out_to_unknown(h: Harness) -> None:
    await _unlock(h)
    h.broker.injector.hang_after_accept_s = 5.0
    intent = h.factory.make()
    out = await _svc(h, submit_timeout_s=0.05).submit(intent)
    assert out is not None  # accepted at the broker, found by client order id
    assert h.broker.submission_count() == 1


async def test_partition_before_accept_stays_unknown_and_is_not_resubmitted(h: Harness) -> None:
    await _unlock(h)
    h.broker.injector.partition_before_accept = 1
    intent = h.factory.make()
    svc = _svc(h)
    assert await svc.submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.UNKNOWN
    assert await svc.submit(intent) is None  # a retry attempt is refused by the ledger
    assert h.broker.submission_count() == 1
    audits = h.db.query("SELECT payload_json FROM audit_events WHERE event_type='SUBMIT_OUTCOME_UNKNOWN'")
    assert len(audits) == 1 and '"resubmitted":false' in audits[0]["payload_json"]


async def test_lookup_failure_after_timeout_stays_unknown(h: Harness) -> None:
    await _unlock(h)
    h.broker.injector.timeout_after_accept = 1
    h.broker.injector.fail_lookups = 1
    intent = h.factory.make()
    assert await _svc(h).submit(intent) is None
    assert state_of(h, intent.client_order_id) == S.UNKNOWN
    assert h.db.query_one("SELECT COUNT(*) n FROM failures WHERE kind='unknown_lookup_failed'")["n"] == 1  # type: ignore[index]


async def test_unrecordable_response_is_unknown_not_retry(h: Harness) -> None:
    await _unlock(h)

    class BadGateway:
        kind = GatewayKind.FAKE
        is_paper = True
        submits = 0

        async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder:
            self.submits += 1
            snap = await h.broker.submit(intent)
            return snap.model_copy(update={"client_order_id": "someone-else"})

        async def get_order_by_client_id(self, coid: str) -> BrokerOrder | None:
            return await h.broker.get_order_by_client_id(coid)

        def __getattr__(self, name: str) -> Any:
            return getattr(h.broker, name)

    gw = BadGateway()
    svc = ExecutionService(h.ledger, gw, pass_gate, h.clock, entries_lock=h.entries)  # type: ignore[arg-type]
    intent = h.factory.make()
    await svc.submit(intent)
    assert gw.submits == 1
    assert "UNKNOWN" in [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)]


async def test_cancel_goes_cancel_pending_then_broker_decides(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    intent = h.factory.make(limit="99.00")
    svc = _svc(h)
    await svc.submit(intent)
    for ev in h.broker.drain_events():
        svc.handle_event(ev)
    assert state_of(h, intent.client_order_id) == S.ACKNOWLEDGED
    await svc.cancel(intent.client_order_id)
    assert "CANCEL_PENDING" in [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)]
    for ev in h.broker.drain_events():
        svc.handle_event(ev)
    assert state_of(h, intent.client_order_id) == S.CANCELED


async def test_cancel_before_submission_is_local(h: Harness) -> None:
    intent = h.factory.make()
    assert await _svc(h).cancel(intent.client_order_id) == S.CANCELED
    assert h.broker.cancel_requests == []


async def test_cancel_refused_by_broker_restores_broker_state(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    intent = h.factory.make()
    svc = _svc(h)
    await svc.submit(intent)  # fills immediately
    h.db.write(lambda c: c.execute("UPDATE orders SET state='ACKNOWLEDGED' WHERE client_order_id=?",
                                   (intent.client_order_id,)))  # pretend the fill was not seen yet
    assert await svc.cancel(intent.client_order_id) == S.FILLED  # broker refused: order already filled


async def test_cancel_not_supported_reverts_local_cancel_pending(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    intent = h.factory.make(limit="99.00")
    svc = _svc(h)
    await svc.submit(intent)
    for ev in h.broker.drain_events():
        svc.handle_event(ev)

    async def no_cancel(_coid: str) -> None:
        raise CancelNotSupported("no cancel channel")

    h.broker.cancel = no_cancel  # type: ignore[method-assign]
    assert await svc.cancel(intent.client_order_id) == S.ACKNOWLEDGED
    assert [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)][-2:] == ["CANCEL_PENDING", "ACKNOWLEDGED"]


class _ScriptedGateway:
    """Order stream that yields scripted events, then disconnects (for consumer tests)."""

    kind = GatewayKind.FAKE
    is_paper = True

    def __init__(self, batches: list[list[OrderEvent]]) -> None:
        self.batches = batches
        self.subscriptions = 0

    def order_events(self) -> AsyncIterator[OrderEvent]:
        self.subscriptions += 1
        batch = self.batches.pop(0) if self.batches else []

        async def gen() -> AsyncIterator[OrderEvent]:
            for ev in batch:
                yield ev
            if not self.batches:
                await asyncio.sleep(3600)
            raise ConnectionError("stream dropped")

        return gen()


async def test_event_consumer_applies_dedupes_and_flags_disconnects(h: Harness) -> None:
    await _unlock(h)
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    intent = h.factory.make()
    h.broker.injector.duplicate_next_events = 2
    h.broker.injector.reorder_window = 3
    await _svc(h).submit(intent)
    evs = h.broker.drain_events()
    gw = _ScriptedGateway([evs[:2], evs[2:] + evs])
    svc = ExecutionService(h.ledger, gw, pass_gate, h.clock, entries_lock=h.entries,  # type: ignore[arg-type]
                           reconnect_backoff_s=(0.01,), watchdog_interval_s=60)
    stop = asyncio.Event()
    task = asyncio.create_task(svc.run_event_consumer(stop))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if gw.subscriptions >= 2 and svc.events_applied + svc.events_duplicate >= 2 * len(evs):
            break
    stop.set()
    await asyncio.wait_for(task, 2)
    assert svc.disconnects >= 1
    assert Flag.BROKER_DISCONNECTED.value in flags(h)
    assert svc.events_duplicate >= len(evs)
    assert state_of(h, intent.client_order_id) == S.FILLED
    assert h.ledger.positions()[0].quantity == intent.quantity


async def test_consumer_stops_promptly(h: Harness) -> None:
    svc = _svc(h, watchdog_interval_s=60)
    stop = asyncio.Event()
    task = asyncio.create_task(svc.run_event_consumer(stop))
    await asyncio.wait_for(svc.consumer_ready.wait(), 1)
    stop.set()
    await asyncio.wait_for(task, 1)


async def test_watchdog_flags_wall_clock_jump(h: Harness) -> None:
    wall = ManualClock(T0)
    svc = _svc(h, wall_clock=wall, watchdog_interval_s=5, suspend_gap_s=30)
    assert svc.watchdog_tick() == []
    wall.advance(5)
    assert svc.watchdog_tick() == []
    wall.advance(timedelta(hours=8).total_seconds())  # lid closed overnight
    assert svc.watchdog_tick() == [Flag.PROCESS_SUSPENDED.value]
    assert h.entries.is_flag_active(Flag.PROCESS_SUSPENDED)


async def test_handle_event_failure_locks_entries(h: Harness) -> None:
    svc = _svc(h)

    def broken(_ev: OrderEvent) -> Any:
        raise RuntimeError("disk full")

    h.ledger.apply_event = broken  # type: ignore[method-assign]
    from tests.unit.order_ledger.ol_support import stream_event

    svc.handle_event(stream_event("lat-sim-x", OrderEventType.NEW, S.ACKNOWLEDGED, T0))
    assert Flag.LEDGER_ANOMALY.value in flags(h)
    assert h.db.query_one("SELECT COUNT(*) n FROM failures WHERE kind='event_apply_failed'")["n"] == 1  # type: ignore[index]
