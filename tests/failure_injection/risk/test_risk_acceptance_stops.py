"""Acceptance criterion 5: a risky or invalid proposed order is stopped by the deterministic risk policy.

Each scenario starts from the approvable baseline, makes it risky in one way, and proves (a) the decision
is REJECTED for the expected reason, (b) ``authorize`` refuses to sign it, and - where an attacker could
skip the evaluation step - (c) the final order-boundary gate refuses a crafted intent.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from packages.contracts.enums import GatewayKind, MarketSession, OrderType, RiskVerdict, Side, TradingMode
from packages.contracts.models import ApprovedOrderIntent, TradeProposal
from packages.contracts.policy import RiskPolicy
from services.risk.context import fingerprint
from services.risk.engine import RiskAuthorizationError, max_allowable_quantity
from tests.unit.risk.risk_factories import (
    NOW,
    PRE_MARKET,
    SIGNER,
    d,
    failed,
    healthy,
    make_account,
    make_ctx,
    make_engine,
    make_market,
    make_proposal,
    make_quote,
)

BASE = make_proposal()

SCENARIOS: dict[str, tuple[TradeProposal, dict[str, Any], str]] = {
    "oversized order": (make_proposal(quantity=d("40"), max_planned_loss=d("201.60")), {}, "max_order_notional"),
    "leveraged ETF TQQQ": (make_proposal(symbol="TQQQ"), {"market": make_market("TQQQ")}, "symbol_denylist"),
    "inverse ETF SQQQ": (make_proposal(symbol="SQQQ"), {"market": make_market("SQQQ")}, "symbol_denylist"),
    "volatility ETP UVXY": (make_proposal(symbol="UVXY"), {"market": make_market("UVXY")}, "symbol_denylist"),
    "short sale": (
        make_proposal(side=Side.SELL, quantity=d("3"), limit_price=d("500.00"), stop_price=None,
                      max_planned_loss=d("0")),
        {}, "long_only",
    ),
    "extended hours (pre-market clock)": (make_proposal(now=PRE_MARKET), {"now": PRE_MARKET}, "regular_session"),
    "extended hours (PRE session snapshot)": (BASE, {"market": make_market(session=MarketSession.PRE)},
                                              "regular_session"),
    "stale quote": (BASE, {"market": make_market(quote=make_quote(exchange_age_s=120, received_age_s=120))},
                    "max_quote_age"),
    "wide spread": (
        make_proposal(limit_price=d("501.50"), max_planned_loss=d("19.50")),
        {"market": make_market(quote=make_quote(bid="499.00", ask="501.50"))}, "max_spread",
    ),
    "market order": (BASE.model_copy(update={"order_type": OrderType.MARKET}), {}, "order_type_limit_only"),
    "model-inflated arithmetic (understated loss)": (
        make_proposal(quantity=d("3"), stop_price=d("450.00"), max_planned_loss=d("5.00")), {}, "proposal_arithmetic",
    ),
    "model-inflated arithmetic (huge size, tiny claimed loss)": (
        make_proposal(quantity=d("400"), stop_price=d("499.90"), max_planned_loss=d("1.00")), {}, "max_order_notional",
    ),
    "duplicate order": (BASE, {"recent_intent_fingerprints": (fingerprint(BASE),)}, "duplicate_order"),
    "entries locked": (BASE, {"health": healthy(entries_locked=True, lock_reasons=("startup_unreconciled",))},
                       "health_entries_unlocked"),
    "margin needed": (BASE, {"account": make_account(cash="100.00", buying_power="40000.00")}, "no_margin"),
    "account unknown": (BASE, {"account": None}, "account_present"),
    "broker disconnected": (BASE, {"health": healthy(broker_connected=False)}, "health_broker_connected"),
    "not reconciled": (BASE, {"health": healthy(reconciled=False)}, "health_reconciled"),
    "quote missing": (BASE, {"market": make_market(no_quote=True)}, "quote_present"),
    "research-only mode": (BASE, {"mode": TradingMode.RESEARCH_ONLY}, "mode_permits_trading"),
    "live mode via env/runtime switch": (
        BASE, {"mode": TradingMode.LIVE_APPROVAL, "gateway": GatewayKind.LEAN_ALPACA_LIVE}, "mode_permits_trading",
    ),
    "daily loss limit hit": (BASE, {"daily_realized_pl": d("-251.00")}, "max_daily_loss"),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_risky_or_invalid_order_is_stopped(scenario: str) -> None:
    proposal, ctx_kw, expected = SCENARIOS[scenario]
    when = ctx_kw.get("now", NOW)
    engine = make_engine(now=when)
    ctx = make_ctx(**ctx_kw)
    decision = engine.evaluate(proposal, ctx)
    assert decision.verdict == RiskVerdict.REJECTED, f"{scenario} was approved"
    assert expected in failed(decision), f"{scenario}: failed={sorted(failed(decision))}"
    with pytest.raises(RiskAuthorizationError):
        engine.authorize(proposal, decision, ctx)


def test_oversized_order_reports_but_never_applies_max_quantity() -> None:
    engine = make_engine()
    p = make_proposal(quantity=d("40"), max_planned_loss=d("201.60"))
    decision = engine.evaluate(p, make_ctx())
    assert decision.recomputed_quantity == d("40")
    assert max_allowable_quantity(decision) == d("3")


def test_model_inflated_arithmetic_is_recomputed_not_trusted() -> None:
    engine = make_engine()
    p = make_proposal(quantity=d("400"), stop_price=d("499.90"), max_planned_loss=d("1.00"))
    decision = engine.evaluate(p, make_ctx())
    assert decision.recomputed_notional == d("200016.00")  # 400 * 500.04, not anything the model said
    assert decision.recomputed_max_loss == d("400") * (d("500.04") - d("492.53"))  # stop too tight => default
    assert decision.recomputed_max_loss != p.max_planned_loss


def test_market_order_and_extended_hours_cannot_even_be_expressed() -> None:
    with pytest.raises(ValidationError):
        make_proposal(order_type=OrderType.MARKET)
    engine = make_engine()
    ctx = make_ctx()
    p = make_proposal()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    with pytest.raises(ValidationError, match="extended-hours"):
        ApprovedOrderIntent.model_validate({**intent.model_dump(), "extended_hours": True})


def _forge(engine: Any, **update: Any) -> ApprovedOrderIntent:
    """An attacker holding the signing key skips evaluate() and signs a risky intent directly."""
    ctx = make_ctx()
    p = make_proposal()
    good = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    unsigned = good.model_copy(update=update)
    return ApprovedOrderIntent.model_validate({**unsigned.model_dump(), "signature": SIGNER.sign(unsigned.signing_string())})


@pytest.mark.parametrize(
    ("update", "ctx_kw", "needle"),
    [
        ({"symbol": "TQQQ"}, {}, "denylisted"),
        ({"quantity": d("400")}, {}, "max_order_notional"),
        ({"side": Side.SELL}, {}, "long-only"),
        ({"expires_at": NOW + timedelta(hours=2)}, {}, "TTL"),
        ({"mode": TradingMode.LIVE_APPROVAL, "gateway": GatewayKind.LEAN_ALPACA_LIVE}, {}, "live execution is denied"),
        ({}, {"health": healthy(entries_locked=True, lock_reasons=("halt",))}, "entries locked"),
        ({}, {"now": PRE_MARKET}, "session:"),
    ],
)
def test_final_gate_stops_risky_intents_signed_outside_evaluate(update: dict[str, Any], ctx_kw: dict[str, Any],
                                                                needle: str) -> None:
    engine = make_engine()
    forged = _forge(engine, **update)
    when = ctx_kw.get("now")
    if when is not None:
        engine = make_engine(now=when)  # same signing key, clock moved outside regular hours
    fails = engine.final_gate(forged, make_ctx(**ctx_kw), SIGNER.verifier, engine.policy_version, self_recorded=False)
    assert any(needle in f for f in fails), fails


def test_policy_with_stricter_limits_stops_previously_fine_order() -> None:
    strict = RiskPolicy(max_order_notional=d("1000.00"))
    decision = make_engine(strict).evaluate(BASE, make_ctx())
    assert not decision.approved and "max_order_notional" in failed(decision)
