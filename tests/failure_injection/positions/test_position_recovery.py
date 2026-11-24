"""Restart, crash and in-doubt recovery of the position lifecycle (SIMULATED, deterministic FakeBroker).

A "restart" closes the SQLite handle and rebuilds every service object on the same files against the same broker,
so the process-start guard locks entries *and* exits until the new process completes its own CLEAN
broker-authoritative reconciliation. The invariants under test: nothing is resubmitted blindly, no protective
stop or exit is duplicated, and the lifecycle resumes from durable state (ledger + position plan).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import OrderState, OrderType, ReconciliationStatus, Side
from tests.integration.positions.pm_rig import Rig, build_rig

D = Decimal


@pytest.fixture
async def rig(tmp_path: Path) -> AsyncIterator[Rig]:
    r = build_rig(tmp_path)
    assert (await r.reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
    yield r
    r.db.close()


def _accepted_sells(rig: Rig, order_type: OrderType | None = None) -> list[str]:
    return [c for c in rig.broker.accepted_submissions
            if rig.broker.orders[c].side == Side.SELL
            and (order_type is None or rig.broker.orders[c].order_type == order_type)]


async def test_crash_after_entry_fill_before_protection_reprotects_after_restart(rig: Rig) -> None:
    await rig.buy("3")
    assert rig.position() == D(3)
    rig2 = rig.reopen()  # crash: the manager never ticked
    try:
        assert rig2.entries.exits_locked()[0]  # new process: locked until its own CLEAN reconciliation
        assert await rig2.tick() == []  # no order action while blind...
        assert rig2.reconcile_calls == 1  # ...but it asks for the broker-authoritative reconciliation
        acts = await rig2.tick()
        assert [a["action"] for a in acts] == ["PROTECT"] and acts[0]["outcome"] == "SUBMITTED"
        assert len(_accepted_sells(rig2, OrderType.STOP_LIMIT)) == 1
        plan = rig2.store.open_plan("SPY")
        assert plan is not None and plan.stop_price == D("495.00")
    finally:
        rig2.db.close()


async def test_protective_submit_outcome_unknown_is_resolved_never_duplicated(rig: Rig) -> None:
    await rig.buy("3")
    rig.broker.injector.timeout_after_accept = 1  # broker accepted, response lost
    rig.broker.injector.fail_lookups = 1  # and the immediate lookup by client order id fails
    rig.broker.injector.drop_next_events = 2  # and the stream updates (accepted, new) never arrive
    acts = await rig.tick()
    assert acts[0]["action"] == "PROTECT"
    unknown = [r for r in rig.ledger.records() if r.state == OrderState.UNKNOWN]
    assert len(unknown) == 1 and unknown[0].is_protective
    assert rig.entries.exits_locked()[0]  # UNKNOWN locks exits too: never act on doubt
    await rig.tick()  # requests reconciliation, which resolves the order by client order id
    assert rig.ledger.get_record(unknown[0].client_order_id).state == OrderState.ACKNOWLEDGED
    assert await rig.tick(3) == []
    assert len(_accepted_sells(rig, OrderType.STOP_LIMIT)) == 1  # exactly one protective stop at the broker
    assert rig.safety() == {}


async def test_restart_with_protection_in_place_changes_nothing(rig: Rig) -> None:
    await rig.buy("3")
    await rig.tick()
    before = rig.broker.submission_count()
    rig2 = rig.reopen()
    try:
        await rig2.tick(4)
        assert rig2.broker.submission_count() == before
        assert len(rig2.protective()) == 1 and rig2.safety() == {}
    finally:
        rig2.db.close()


async def test_restart_mid_exit_resumes_the_exit_not_the_protection(rig: Rig) -> None:
    await rig.buy("3")
    await rig.tick()
    rig.advance(61 * 60)
    acts = await rig.tick()  # TIME exit chosen (persisted) and protection canceled
    assert acts[0]["purpose"] == "EXIT_TIME" and rig.protective() == []
    rig2 = rig.reopen()  # crash before the exit order was sent
    try:
        await rig2.tick()  # reconcile
        acts = await rig2.tick()
        assert acts[0]["action"] == "EXIT" and acts[0]["order_type"] == "LIMIT"  # not a new protective stop
        await rig2.tick()
        assert rig2.position() == D(0)
        assert rig2.store.closed_plans(limit=1)[0].close_reason == "TIME"
        assert len(_accepted_sells(rig2, OrderType.STOP_LIMIT)) == 1  # the original stop only
    finally:
        rig2.db.close()


async def test_exit_outcome_unknown_across_a_restart_is_never_resubmitted(rig: Rig) -> None:
    await rig.buy("3")
    await rig.tick()
    rig.advance(61 * 60)
    await rig.tick()  # cancel protection
    rig.broker.injector.timeout_after_accept = 1
    rig.broker.injector.fail_lookups = 1
    rig.broker.injector.drop_next_events = 3  # accepted, new, fill: lost with the crashing process
    await rig.tick()  # exit accepted (and filled) at the broker; the ledger only knows UNKNOWN
    assert rig.broker.positions["SPY"].qty == D(0)
    assert [r.state for r in rig.ledger.records() if r.side == Side.SELL and r.order_type == OrderType.LIMIT] == [
        OrderState.UNKNOWN]
    rig2 = rig.reopen()
    try:
        await rig2.tick()  # startup reconciliation resolves the exit from broker truth (fill booked once)
        await rig2.tick(2)
        assert rig2.position() == D(0)
        assert len(_accepted_sells(rig2, OrderType.LIMIT)) == 1  # never a second exit
        plan = rig2.store.closed_plans(limit=1)[0]
        assert plan.close_reason == "TIME" and plan.exit_quantity == D(3)
        assert rig2.entries.is_locked() == (False, [])
    finally:
        rig2.db.close()


async def test_stop_fill_while_process_down_is_recovered_from_the_broker(rig: Rig) -> None:
    await rig.buy("3")
    await rig.tick()
    rig2 = rig.reopen()
    try:
        # while the process was down the market fell through the stop: the broker-held stop filled
        rig2.broker.advance(rig2.quote_now().model_copy(update={"bid": D("494.80"), "ask": D("494.85")}))
        rig2.bid, rig2.ask = D("494.80"), D("494.85")
        assert rig2.broker.positions["SPY"].qty == D(0)
        await rig2.tick()  # reconciliation books the stop fill from broker state
        await rig2.tick()
        assert rig2.position() == D(0)
        plan = rig2.store.closed_plans(limit=1)[0]
        assert plan.close_reason == "STOP" and plan.realized_pl == D("-15.72")
    finally:
        rig2.db.close()
