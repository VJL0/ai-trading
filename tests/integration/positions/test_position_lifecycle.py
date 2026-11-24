"""Entry -> protection -> exit -> flat, through the real risk engine, ledger, execution service and FakeBroker.

SIMULATED: every fill comes from the deterministic in-process fake broker. Each test drives one lifecycle path the
roadmap's Phase 1 completion criteria name: partial fills, cancellation races, disconnects, market close,
overnight, restart recovery, kill switch, and the explicit UNPROTECTED safety state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import OrderState, OrderType, ReconciliationStatus, Side
from services.order_ledger.entries import Flag
from services.positions.policy import PositionPolicy

from .pm_rig import NEXT_OPEN_T, Rig, build_rig, ny

D = Decimal


@pytest.fixture
async def rig(tmp_path: Path) -> AsyncIterator[Rig]:
    r = build_rig(tmp_path)
    report = await r.reconciler.startup_recovery()
    assert report.status == ReconciliationStatus.CLEAN, report.detail
    yield r
    r.db.close()


async def _open_protected(rig: Rig, qty: str = "3") -> None:
    out = await rig.buy(qty)
    assert out["outcome"] == "SUBMITTED", out
    assert rig.position() == D(qty)
    await rig.tick()
    prot = rig.protective()
    assert len(prot) == 1 and prot[0].state == OrderState.ACKNOWLEDGED, prot


# ------------------------------------------------------------------------------------------------- entry + protect
async def test_filled_entry_gets_a_broker_held_protective_stop_limit(rig: Rig) -> None:
    await _open_protected(rig)
    prot = rig.protective()[0]
    assert prot.order_type == OrderType.STOP_LIMIT and prot.side == Side.SELL and prot.quantity == D(3)
    assert prot.stop_price == D("495.00")  # the entry proposal's planned stop
    assert prot.limit_price == D("492.52")  # stop - 50 bps, floored to the tick
    assert prot.intent.signing_string().startswith("lat-intent-v2|")  # stop price is signed
    plan = rig.store.open_plan("SPY")
    assert plan is not None and plan.stop_source == "entry_proposal_stop" and plan.horizon_minutes == 60
    assert rig.safety() == {}
    assert rig.entries.is_locked() == (False, [])
    # a second tick changes nothing: no duplicate protection
    before = rig.broker.submission_count()
    assert await rig.tick(3) == []
    assert rig.broker.submission_count() == before


async def test_protection_failure_raises_unprotected_which_locks_entries_but_not_exits(rig: Rig) -> None:
    await rig.buy("3")
    rig.broker.injector.reject_next = 1  # the broker refuses the protective stop
    acts = await rig.tick()
    assert acts[0]["action"] == "PROTECT" and acts[0]["outcome"] == "BLOCKED"
    assert rig.safety() == {}  # inside the grace period
    rig.advance(5)
    await rig.tick()  # still cooling down after the refusal; grace not yet elapsed
    rig.advance(30)
    rig.broker.injector.reject_next = 1  # refused again
    await rig.tick()
    assert rig.safety() == {"SPY": "UNPROTECTED"}
    locked, reasons = rig.entries.is_locked()
    assert locked and any("UNPROTECTED" in r for r in reasons)
    assert rig.entries.exits_locked() == (False, [])
    # a new entry is refused by the independent risk engine while the position is unprotected
    blocked = await rig.buy("1")
    assert blocked["outcome"] == "REJECTED" and "health_entries_unlocked" in blocked["reason"]
    # protection succeeds once the broker accepts (after the 3-orders-per-minute rate window); the safety state
    # clears from ledger evidence only
    rig.advance(61)
    await rig.tick()
    assert len(rig.protective()) == 1
    assert rig.safety() == {}
    assert rig.entries.is_locked() == (False, [])


async def test_partial_entry_is_protected_topped_up_then_remainder_canceled(tmp_path: Path) -> None:
    rig = build_rig(tmp_path, max_fill_per_match=D(1))
    await rig.reconciler.startup_recovery()
    await rig.buy("3")
    assert rig.position() == D(1)  # one share filled, two still working
    rig.feed_live = False  # quotes only when the test says so
    acts = await rig.tick()
    assert [a["action"] for a in acts] == ["PROTECT"] and acts[0]["quantity"] == "1"
    rig.set_quote("500.00", "500.04")  # the resting entry fills one more share
    assert rig.position() == D(2)
    acts = await rig.tick()
    assert [a["action"] for a in acts] == ["TOP_UP"] and acts[0]["quantity"] == "1"
    assert sum(r.quantity for r in rig.protective()) == D(2)
    rig.advance(16)  # past entry_fill_timeout_s
    acts = await rig.tick()
    assert acts[0]["action"] == "CANCEL" and acts[0]["purpose"] == "ENTRY_TIMEOUT"
    entry = next(r for r in rig.ledger.records() if r.side == Side.BUY)
    assert entry.state == OrderState.CANCELED and entry.filled_quantity == D(2)
    assert rig.position() == D(2) and sum(r.quantity for r in rig.protective()) == D(2)
    rig.db.close()


async def test_cancel_race_entry_fills_during_cancel_then_protection_covers_everything(tmp_path: Path) -> None:
    rig = build_rig(tmp_path, max_fill_per_match=D(1))
    await rig.reconciler.startup_recovery()
    await rig.buy("3")
    rig.feed_live = False
    await rig.tick()  # protect the first share
    assert rig.position() == D(1) and sum(r.quantity for r in rig.protective()) == D(1)
    rig.advance(16)
    rig.broker.injector.fill_on_cancel = 1  # the remainder executes before the cancel takes effect
    await rig.tick()
    entry = next(r for r in rig.ledger.records() if r.side == Side.BUY)
    assert entry.state == OrderState.FILLED and rig.position() == D(3)
    rig.feed_live = True
    acts = await rig.tick()
    assert [a["action"] for a in acts] == ["TOP_UP"] and acts[0]["quantity"] == "2"
    assert sum(r.quantity - r.filled_quantity for r in rig.protective()) == D(3)
    rig.db.close()


# ------------------------------------------------------------------------------------------------- exits
async def test_time_exit_cancels_protection_first_then_sells_and_books_realized_pnl(rig: Rig) -> None:
    await _open_protected(rig)
    rig.advance(61 * 60)  # past the 60-minute horizon
    prot = rig.protective()[0]
    acts = await rig.tick()
    assert acts == [{"action": "CANCEL", "symbol": "SPY", "purpose": "EXIT_TIME",
                     "client_order_id": prot.client_order_id, "outcome": "CANCEL_PENDING"}]
    assert rig.ledger.get_record(prot.client_order_id).state == OrderState.CANCELED  # broker confirmed
    assert rig.safety() == {"SPY": "EXITING"} and rig.entries.is_locked()[0]
    acts = await rig.tick()
    assert acts[0]["action"] == "EXIT" and acts[0]["outcome"] == "SUBMITTED" and acts[0]["order_type"] == "LIMIT"
    assert acts[0]["limit_price"] == "500.00"  # marketable at the bid
    assert rig.position() == D(0)
    await rig.tick()
    plan = rig.store.closed_plans(limit=1)[0]
    assert plan.close_reason == "TIME"
    assert plan.entry_avg_price == D("500.04") and plan.exit_avg_price == D("500.00")
    assert plan.realized_pl == D("-0.12")
    assert rig.safety() == {} and rig.entries.is_locked() == (False, [])
    assert rig.store.open_plan("SPY") is None


async def test_broker_stop_fill_closes_the_plan_with_reason_stop(rig: Rig) -> None:
    await _open_protected(rig)
    rig.set_quote("494.90", "494.95")  # the stop trades: the broker-held stop-limit fills at the bid
    assert rig.position() == D(0)
    await rig.tick()
    plan = rig.store.closed_plans(limit=1)[0]
    assert plan.close_reason == "STOP" and plan.realized_pl == D("-15.42")
    assert rig.safety() == {}


async def test_gap_through_stop_limit_is_escalated_to_a_marketable_exit(rig: Rig) -> None:
    await _open_protected(rig)
    rig.set_quote("490.00", "490.10")  # gapped below the protective limit 492.52: the broker order cannot fill
    assert rig.position() == D(3)
    acts = await rig.tick()
    assert acts[0]["action"] == "CANCEL" and acts[0]["purpose"] == "EXIT_STOP_BREACH"
    assert rig.safety() == {"SPY": "BREACHED"}
    acts = await rig.tick()
    assert acts[0]["action"] == "EXIT" and acts[0]["outcome"] == "SUBMITTED" and acts[0]["limit_price"] == "490.00"
    await rig.tick()
    plan = rig.store.closed_plans(limit=1)[0]
    assert plan.close_reason == "STOP_BREACH" and plan.exit_avg_price == D("490.00")
    assert rig.safety() == {}


async def test_unfilled_exit_is_repriced_a_tick_lower_never_as_a_duplicate(tmp_path: Path) -> None:
    rig = build_rig(tmp_path, use_quote_size=True)
    await rig.reconciler.startup_recovery()
    await _open_protected(rig)
    rig.advance(61 * 60)
    await rig.tick()  # cancel protection
    rig.size = D(0)  # no bid size: the marketable exit rests unfilled
    acts = await rig.tick()
    first = acts[0]
    assert first["action"] == "EXIT" and first["limit_price"] == "500.00"
    rig.advance(11)  # past exit_reprice_s
    acts = await rig.tick()
    assert acts[0]["action"] == "CANCEL" and acts[0]["purpose"] == "EXIT_REPRICE"
    acts = await rig.tick()
    assert acts[0]["action"] == "EXIT" and acts[0]["outcome"] == "SUBMITTED" and acts[0]["limit_price"] == "499.99"
    rig.size = D(500)
    await rig.tick(2)
    assert rig.position() == D(0)
    sells = [r for r in rig.sells() if r.order_type == OrderType.LIMIT]
    assert [r.state for r in sells] == [OrderState.CANCELED, OrderState.FILLED]
    rig.db.close()


async def test_end_of_day_flatten_and_pre_close_entry_cutoff(rig: Rig) -> None:
    rig.set_time(ny(15, 20))
    await _open_protected(rig)
    rig.set_time(ny(15, 35))  # inside the 30-minute pre-close entry cutoff
    late = await rig.buy("1")
    assert late["outcome"] == "REJECTED" and "entry_window" in late["reason"]
    rig.set_time(ny(15, 51))  # 9 minutes before the close: flatten
    acts = await rig.tick()
    assert acts[0]["purpose"] == "EXIT_EOD"
    await rig.tick(2)
    assert rig.position() == D(0)
    assert rig.store.closed_plans(limit=1)[0].close_reason == "EOD"


async def test_position_left_at_the_close_is_overnight_then_liquidated_at_the_next_open(rig: Rig) -> None:
    rig.set_time(ny(15, 20))
    await _open_protected(rig)
    halt = rig.entries.lock("operator kill switch", "operator", scope="ALL")  # exits blocked: flatten cannot run
    rig.set_time(ny(15, 51))
    assert await rig.tick() == []
    rig.set_time(ny(16, 5))  # after the close: the DAY protective stop expired at the broker
    assert rig.protective() == []
    await rig.tick()
    assert rig.safety() == {"SPY": "OVERNIGHT"}
    assert rig.entries.is_locked()[0]
    # next session: operator clears the kill switch after a CLEAN reconciliation
    rig.set_time(NEXT_OPEN_T.replace(minute=31))
    assert (await rig.reconcile()).status == ReconciliationStatus.CLEAN
    assert rig.entries.clear_halt(halt, by="victor", reason="resolved")
    acts = await rig.tick()
    assert acts[0]["action"] == "EXIT" and acts[0]["outcome"] == "SUBMITTED"
    await rig.tick()
    assert rig.position() == D(0)
    assert rig.store.closed_plans(limit=1)[0].close_reason == "OVERNIGHT"
    assert rig.safety() == {}


# ------------------------------------------------------------------------------------------------- locks
async def test_entries_halt_does_not_block_exits_but_the_kill_switch_does(rig: Rig) -> None:
    await _open_protected(rig)
    rig.entries.lock("stop new entries", "operator")  # ENTRIES scope
    assert rig.entries.is_locked()[0] and rig.entries.exits_locked() == (False, [])
    rig.advance(61 * 60)
    await rig.tick(3)
    assert rig.position() == D(0)  # time exit completed under an entries halt
    await _reenter_after_halts_cleared(rig)
    kill = rig.entries.lock("kill switch", "operator", scope="ALL")
    rig.advance(61 * 60)
    before = rig.broker.submission_count()
    assert await rig.tick(3) == []
    assert rig.broker.submission_count() == before and rig.position() == D(3)
    assert rig.reconcile_calls == 0  # never spams reconciliation while the operator holds the kill switch
    assert rig.entries.clear_halt(kill, by="victor", reason="done")


async def _reenter_after_halts_cleared(rig: Rig) -> None:
    for h in rig.entries.halts():
        rig.entries.clear_halt(h["halt_id"], by="victor", reason="clear")
    rig.advance(400)  # leave the duplicate window
    await _open_protected(rig)


async def test_disconnect_blocks_exits_until_a_clean_reconciliation(rig: Rig) -> None:
    await _open_protected(rig)
    rig.entries.set_flag(Flag.BROKER_DISCONNECTED, "order-update stream error (test)", "execution")
    rig.broker.injector.fail_reads = 1  # the first reconciliation attempt cannot read the broker
    rig.advance(61 * 60)
    assert await rig.tick() == []
    assert rig.reconcile_calls == 1  # asked for a broker-authoritative reconciliation...
    assert rig.entries.is_flag_active(Flag.BROKER_DISCONNECTED)  # ...which failed: still locked
    assert len(rig.protective()) == 1  # nothing was canceled while blind; the broker stop keeps protecting
    rig.advance(16)
    await rig.tick()  # reconciliation succeeds now: flag cleared
    assert not rig.entries.is_flag_active(Flag.BROKER_DISCONNECTED)
    await rig.tick(3)
    assert rig.position() == D(0)
    assert rig.store.closed_plans(limit=1)[0].close_reason == "TIME"


async def test_working_external_sell_is_respected_and_never_oversold(rig: Rig) -> None:
    out = await rig.buy("3")
    assert out["outcome"] == "SUBMITTED"
    # an agent/manual LIMIT SELL for 1 share rests above the market
    from packages.contracts.enums import DecisionAction, Proposer
    from packages.contracts.models import TradeProposal
    from packages.core.ids import new_causal_id, new_id

    now = rig.clock.now()
    manual = TradeProposal(
        proposal_id=new_id("prop"), causal_id=new_causal_id(), created_at=now, data_asof=now,
        action=DecisionAction.TRADE, proposer=Proposer.TEST, strategy_id="manual", symbol="SPY", side=Side.SELL,
        quantity=D(1), limit_price=D("500.05"), max_planned_loss=D(0), evidence_ids=("ev",), dataset_version="ds-test")
    assert (await rig.execute(manual, rig.snapshot()))["outcome"] == "SUBMITTED"
    acts = await rig.tick()
    assert acts[0]["action"] == "PROTECT" and acts[0]["quantity"] == "2"  # net of the working SELL
    assert rig.safety() == {}


async def test_view_reports_position_risk_and_never_claims_guaranteed_protection(rig: Rig) -> None:
    await _open_protected(rig)
    view = rig.manager.view()
    row = view["open"][0]
    assert row["symbol"] == "SPY" and row["quantity"] == "3" and row["protected_quantity"] == "3"
    assert row["stop_price"] == "495.00" and D(row["risk_to_stop"]) == D("15.00")
    assert "not guaranteed" in row["risk_to_stop_note"] and "not a guaranteed" in view["note"]
    assert view["policy_version"] == PositionPolicy().version


@pytest.mark.parametrize(("day", "hour"), [(5, 15), (27, 12)])
@pytest.mark.parametrize("fill_race", [False, True])
async def test_flat_entry_canceled_at_flatten_boundary_before_timeout(
    tmp_path: Path, day: int, hour: int, fill_race: bool,
) -> None:
    """Normal and early closes: even a young unfilled entry cannot outlive the flatten cutoff."""
    rig = build_rig(tmp_path, policy=PositionPolicy(flatten_before_close_s=3600))
    try:
        await rig.reconciler.startup_recovery()
        cutoff = ny(hour, 0, day)
        rig.set_time(cutoff - timedelta(seconds=1))
        entry = await rig.buy("3", limit="500.00")
        assert entry["outcome"] == "SUBMITTED"
        assert rig.position() == D(0)
        assert await rig.tick() == []  # still before cutoff and younger than timeout
        rig.set_time(cutoff)
        if fill_race:
            rig.broker.injector.fill_on_cancel = 1
        acts = await rig.tick()
        assert len(acts) == 1 and acts[0]["purpose"] == "ENTRY_EOD"
        rec = rig.ledger.get_record(entry["client_order_id"])
        assert rec is not None
        assert rec.state == (OrderState.FILLED if fill_race else OrderState.CANCELED)
        await rig.tick(3)
        assert rig.position() == D(0) and rig.open_records() == []
        if fill_race:
            assert rig.store.closed_plans(limit=1)[0].close_reason == "EOD"
        else:
            assert rig.sells() == []
    finally:
        rig.db.close()


async def test_flat_pending_cancel_recovers_missing_broker_event(rig: Rig) -> None:
    entry = await rig.buy("3", limit="500.00")
    assert entry["outcome"] == "SUBMITTED" and rig.position() == D(0)
    rig.advance(16)
    rig.broker.injector.drop_next_events = 2  # pending cancel and terminal cancellation
    await rig.tick()
    rec = rig.ledger.get_record(entry["client_order_id"])
    assert rec is not None and rec.state == OrderState.CANCEL_PENDING
    assert await rig.tick() == []
    assert rig.reconcile_calls == 0
    rig.advance(31)
    await rig.tick()
    assert rig.reconcile_calls == 1
    rec = rig.ledger.get_record(entry["client_order_id"])
    assert rec is not None and rec.state == OrderState.CANCELED
    assert rig.broker.cancel_requests == [entry["client_order_id"]]
    assert rig.position() == D(0)
