from __future__ import annotations

from decimal import Decimal

import pytest

from packages.contracts.enums import SessionConstraint, TimeInForce
from packages.contracts.policy import RiskPolicy
from services.risk.engine import CHECK_HOLDING_PERIOD
from tests.unit.risk.risk_factories import SIGNER, check, make_ctx, make_engine, make_position, make_proposal


def overnight_policy(**changes: object) -> RiskPolicy:
    return RiskPolicy.model_validate({
        "allow_overnight_positions": True, "multi_day_strategy_allowlist": ["test-strategy"],
        "max_daily_loss": "100", **changes,
    })


@pytest.mark.parametrize("holding", [2, 5, 20])
def test_disabled_policy_rejects_multiday_request(holding: int) -> None:
    result = make_engine().evaluate(make_proposal(holding_sessions=holding), make_ctx())
    assert not result.approved
    assert not check(result, CHECK_HOLDING_PERIOD).passed


@pytest.mark.parametrize("holding", [None, 1])
def test_default_intraday_contract_unchanged(holding: int | None) -> None:
    assert make_engine().evaluate(make_proposal(holding_sessions=holding), make_ctx()).approved


def test_overnight_requires_allowlisted_strategy_and_session_bound() -> None:
    engine = make_engine(overnight_policy())
    for overrides in ({"strategy_id": "unvalidated"}, {"holding_sessions": 6}):
        p = make_proposal(**{"holding_sessions": 3, "quantity": Decimal(1), **overrides})
        assert not check(engine.evaluate(p, make_ctx()), CHECK_HOLDING_PERIOD).passed
    assert engine.evaluate(make_proposal(holding_sessions=3, quantity=Decimal(1)), make_ctx()).approved


def test_overnight_gap_stress_counts_held_exposure_and_remaining_loss_budget() -> None:
    engine = make_engine(overnight_policy())
    p = make_proposal(holding_sessions=3, quantity=Decimal(1))
    assert engine.evaluate(p, make_ctx()).approved
    # A tight stop cannot make a 10% overnight gap fit the $100 daily loss budget.
    for ctx in (make_ctx(daily_realized_pl=Decimal(-60)), make_ctx(positions=(make_position(),))):
        result = engine.evaluate(p, ctx)
        assert not check(result, CHECK_HOLDING_PERIOD).passed


def test_final_gate_rechecks_immutable_holding_proposal_and_gap_budget() -> None:
    engine = make_engine(overnight_policy())
    p = make_proposal(holding_sessions=3, quantity=Decimal(1))
    ctx = make_ctx()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    assert engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy.version, proposal=p) == []
    assert any("originating proposal required" in x for x in
               engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy.version))
    later = make_ctx(daily_realized_pl=Decimal(-60))
    assert any("gap stress" in x for x in
               engine.final_gate(intent, later, SIGNER.verifier, engine.policy.version, proposal=p))
    wrong = p.model_copy(update={"proposal_id": "another"})
    assert any("does not match" in x for x in
               engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy.version, proposal=wrong))


def test_crypto_policy_flag_does_not_enable_unimplemented_execution() -> None:
    engine = make_engine(RiskPolicy(crypto_enabled=True))
    p = make_proposal(symbol="BTC/USD", time_in_force=TimeInForce.IOC,
                      session_constraint=SessionConstraint.CONTINUOUS)
    assert not engine.evaluate(p, make_ctx(symbol="BTC/USD")).approved
