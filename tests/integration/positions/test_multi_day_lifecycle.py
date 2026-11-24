"""Multi-day safety/recovery through the real risk engine and ledger. FakeBroker fills only; no network."""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import OrderState, OrderType, ReconciliationStatus
from packages.contracts.policy import RiskPolicy
from services.risk.engine import RiskEngine

from .pm_rig import Rig, build_rig, ny

D = Decimal
STRATEGY = "multi-day-test"


@pytest.fixture
async def rig(tmp_path: Path) -> AsyncIterator[Rig]:
    policy = RiskPolicy(allow_overnight_positions=True, multi_day_strategy_allowlist=(STRATEGY,))
    r = build_rig(tmp_path, risk_policy=policy)
    assert (await r.reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
    yield r
    r.db.close()


async def enter(rig: Rig, sessions: int = 2, *, protect: bool = True) -> str:
    out = await rig.buy("1", holding_sessions=sessions, strategy_id=STRATEGY)
    assert out["outcome"] == "SUBMITTED", out
    if protect:
        await rig.tick()
        assert len(rig.protective()) == 1
    return str(out["client_order_id"])


async def test_holds_across_holiday_rearms_day_stop_and_exits_before_early_close(rig: Rig) -> None:
    rig.set_time(ny(10, 0, 25))
    await enter(rig)
    original = rig.store.open_plan("SPY")
    assert original is not None and original.holding_sessions == 2
    assert original.exit_deadline == ny(12, 50, 27)  # Thanksgiving excluded, Friday closes at 13:00
    assert original.entry_risk_policy_version == rig.policy_store.active_version()
    stop_id = rig.protective()[0].client_order_id
    rig.set_time(ny(15, 51, 25))
    assert await rig.tick() == []  # neither the intraday horizon nor EOD liquidates this authorized holding
    rig.set_time(ny(16, 1, 25))
    await rig.tick()
    assert rig.protective() == [] and rig.safety() == {"SPY": "OVERNIGHT"}
    assert rig.entries.is_locked()[0] and not rig.entries.exits_locked()[0]
    rig.set_time(ny(10, 0, 26))
    assert await rig.tick() == []  # no orders during the exchange holiday
    rig.set_time(ny(9, 31, 27))
    actions = await rig.tick()
    assert actions[0]["action"] == "PROTECT" and actions[0]["outcome"] == "SUBMITTED"
    assert rig.protective()[0].client_order_id != stop_id and rig.protective()[0].stop_price == D("495.00")
    assert rig.store.open_plan("SPY").exit_deadline == original.exit_deadline
    assert rig.safety() == {}
    rig.set_time(ny(12, 50, 27))
    actions = await rig.tick()
    assert actions[0]["purpose"] == "EXIT_HOLDING_LIMIT"
    await rig.tick(2)
    assert rig.position() == 0 and rig.store.closed_plans(1)[0].close_reason == "HOLDING_LIMIT"


async def test_weekend_deadline_survives_restart_without_renewing_the_holding_period(rig: Rig) -> None:
    rig.set_time(ny(10, 0, 25))
    await enter(rig, 3)
    original = rig.store.open_plan("SPY")
    assert original is not None and original.exit_deadline == ny(15, 50, 30)
    rig.set_time(ny(16, 1, 25))
    await rig.tick()
    restarted = rig.reopen()
    try:
        restarted.set_time(ny(9, 31, 27))
        assert await restarted.tick() == []  # startup guard: reconcile first
        await restarted.tick()
        assert len(restarted.protective()) == 1 and restarted.store.open_plan("SPY") == original
        before = restarted.broker.submission_count()
        restarted.set_time(ny(10, 0, 28))
        assert await restarted.tick() == [] and restarted.broker.submission_count() == before
        restarted.set_time(ny(9, 31, 30))
        await restarted.tick()
        assert restarted.store.open_plan("SPY").exit_deadline == original.exit_deadline
        assert len(restarted.protective()) == 1
    finally:
        restarted.db.close()


async def test_opening_gap_exits_instead_of_installing_an_already_breached_stop(rig: Rig) -> None:
    await enter(rig)
    rig.set_time(ny(16, 1))
    await rig.tick()
    rig.set_time(ny(9, 31, 6))
    rig.set_quote("490.00", "490.10")
    actions = await rig.tick()
    assert actions[0]["action"] == "EXIT" and actions[0]["outcome"] == "SUBMITTED"
    await rig.tick()
    closed = rig.store.closed_plans(1)[0]
    assert closed.close_reason == "STOP" and closed.exit_avg_price == D("490.00")
    assert len([r for r in rig.sells() if r.order_type == OrderType.STOP_LIMIT]) == 1  # original day's stop


@pytest.mark.parametrize("change", ["disabled", "strategy_removed", "sessions_reduced"])
async def test_active_policy_revocation_triggers_risk_reducing_exit(rig: Rig, change: str) -> None:
    await enter(rig, 3)
    before = rig.store.open_plan("SPY")
    current = rig.policy_store.active()
    changes = {
        "disabled": {"allow_overnight_positions": False},
        "strategy_removed": {"multi_day_strategy_allowlist": ("another-strategy",)},
        "sessions_reduced": {"max_holding_sessions": 2},
    }
    policy = RiskPolicy.model_validate({**current.model_dump(), **changes[change]})
    rig.policy_store.apply_change(policy, operator_token=rig.auth.token_path.read_text().strip(), actor="test",
                                  reason="test revocation")
    rig.engine = RiskEngine(policy, rig.signer, rig.clock)
    actions = await rig.tick()
    assert actions[0]["purpose"] == "EXIT_POLICY_REVOKED"
    await rig.tick(2)
    closed = rig.store.closed_plans(1)[0]
    assert closed.close_reason == "POLICY_REVOKED" and closed.entry_risk_policy_version == before.entry_risk_policy_version
    assert rig.position() == 0


async def test_crash_before_plan_creation_recovers_original_authorized_holding(rig: Rig) -> None:
    rig.set_time(ny(15, 20))
    await enter(rig, protect=False)
    assert rig.store.open_plan("SPY") is None
    rig.set_time(ny(9, 31, 6))
    restarted = rig.reopen()
    try:
        await restarted.tick()  # recovery
        actions = await restarted.tick()
        assert actions[0]["action"] == "PROTECT"
        plan = restarted.store.open_plan("SPY")
        assert plan.holding_sessions == 2 and plan.exit_deadline == ny(15, 50, 6)
        assert plan.opened_at == ny(15, 20)
    finally:
        restarted.db.close()


async def test_lost_day_expiry_is_reconciled_before_rearming(rig: Rig) -> None:
    await enter(rig)
    old = rig.protective()[0]
    rig.broker.injector.drop_next_events = 1
    rig.set_time(ny(16, 1))  # broker expires the stop, but ledger misses its update
    assert rig.ledger.get_record(old.client_order_id).state == OrderState.ACKNOWLEDGED
    rig.set_time(ny(9, 31, 6))
    before = rig.broker.submission_count()
    assert await rig.tick() == []
    assert rig.reconcile_calls == 1 and rig.broker.submission_count() == before
    assert rig.ledger.get_record(old.client_order_id).state == OrderState.EXPIRED
    actions = await rig.tick()
    assert actions[0]["action"] == "PROTECT" and len(rig.protective()) == 1


async def test_enabling_overnight_does_not_change_entries_without_a_holding_request(rig: Rig) -> None:
    await rig.buy("3", strategy_id=STRATEGY)
    await rig.tick()
    assert rig.store.open_plan("SPY").holding_sessions == 1
    rig.advance(61 * 60)
    await rig.tick(3)
    assert rig.position() == 0 and rig.store.closed_plans(1)[0].close_reason == "TIME"


async def test_recovery_after_deadline_exits_without_starting_a_new_holding_period(rig: Rig) -> None:
    await enter(rig, protect=False)
    rig.set_time(ny(9, 31, 9))  # entry Thursday, deadline Friday, restart Monday
    restarted = rig.reopen()
    try:
        await restarted.tick()
        actions = await restarted.tick()
        assert actions[0]["action"] == "EXIT"
        await restarted.tick()
        closed = restarted.store.closed_plans(1)[0]
        assert closed.close_reason == "HOLDING_LIMIT" and closed.exit_deadline == ny(15, 50, 6)
        assert all(r.order_type == OrderType.LIMIT for r in restarted.sells())
    finally:
        restarted.db.close()


async def test_revocation_before_first_plan_keeps_original_provenance_and_exits(rig: Rig) -> None:
    await enter(rig, protect=False)
    original_version = rig.policy_store.active_version()
    revoked = RiskPolicy.model_validate({**rig.policy_store.active().model_dump(), "allow_overnight_positions": False})
    rig.policy_store.apply_change(revoked, operator_token=rig.auth.token_path.read_text().strip(), actor="test",
                                  reason="revoke before restart")
    restarted = rig.reopen()
    try:
        await restarted.tick()
        actions = await restarted.tick()
        assert actions[0]["action"] == "EXIT"
        await restarted.tick()
        closed = restarted.store.closed_plans(1)[0]
        assert closed.close_reason == "POLICY_REVOKED" and closed.entry_risk_policy_version == original_version
        assert closed.holding_sessions == 2
    finally:
        restarted.db.close()
