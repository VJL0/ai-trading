"""Hard ceilings / structural invariants (limits.py), fingerprints and context builders (context.py), and
the XNYS session helper (market_hours.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, MarketSession, OrderType, Side, TradingMode
from packages.contracts.policy import RiskPolicy
from services.risk import limits, market_hours
from services.risk.context import (
    ANY_STRATEGY,
    FingerprintEntry,
    build_risk_context,
    build_system_health,
    fingerprint,
    fingerprint_variants,
    intent_fingerprint,
    order_fingerprint,
    proposal_fingerprint,
    recent_fingerprints,
    unknown_health,
)
from tests.unit.risk.risk_factories import (
    AFTER_HOURS,
    NOW,
    PRE_MARKET,
    SATURDAY,
    THANKSGIVING,
    make_account,
    make_ctx,
    make_engine,
    make_market,
    make_no_trade,
    make_position,
    make_proposal,
)

# ------------------------------------------------------------------------------------------ limits


def test_default_policy_is_valid() -> None:
    assert limits.validate_policy(RiskPolicy()) == []


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"allow_margin": True}, "allow_margin"),
        ({"allow_short": True}, "allow_short"),
        ({"long_only": False}, "long_only"),
        ({"allow_extended_hours": True}, "allow_extended_hours"),
        ({"regular_session_only": False}, "regular_session_only"),
        ({"allowed_order_types": (OrderType.LIMIT, OrderType.MARKET)}, "allowed_order_types"),
        ({"max_order_notional": Decimal("25000.01")}, "max_order_notional"),
        ({"max_position_notional": Decimal("50001")}, "max_position_notional"),
        ({"max_concentration_pct": Decimal("0.51")}, "max_concentration_pct"),
        ({"max_daily_loss": Decimal("2501")}, "max_daily_loss"),
        ({"max_planned_loss_per_trade": Decimal("501")}, "max_planned_loss_per_trade"),
        ({"max_spread_bps": Decimal("101")}, "max_spread_bps"),
        ({"max_quote_age_seconds": Decimal("61")}, "max_quote_age_seconds"),
        ({"max_limit_deviation_bps": Decimal("101")}, "max_limit_deviation_bps"),
        ({"min_price": Decimal("1.00")}, "min_price"),
        ({"min_account_equity": Decimal("100")}, "min_account_equity"),
        ({"min_buying_power_buffer": Decimal("0.99")}, "min_buying_power_buffer"),
        ({"max_positions": 11}, "max_positions"),
        ({"max_pending_orders": 11}, "max_pending_orders"),
        ({"max_orders_per_minute": 11}, "max_orders_per_minute"),
        ({"max_orders_per_day": 201}, "max_orders_per_day"),
        ({"intent_ttl_seconds": 301}, "intent_ttl_seconds"),
        ({"duplicate_window_seconds": 30}, "duplicate_window_seconds"),
        ({"default_stop_loss_bps": Decimal("5")}, "default_stop_loss_bps"),
        ({"max_order_notional": Decimal("0")}, "max_order_notional"),
        ({"symbol_allowlist": ()}, "symbol_allowlist"),
        ({"symbol_allowlist": ("SPY", "SPY")}, "duplicates"),
        ({"symbol_allowlist": ("SPY", "BAD1")}, "invalid symbols"),
        ({"symbol_denylist": ("TQQQ",)}, "symbol_denylist must keep"),
        ({"symbol_allowlist": ("SPY", "SQQQ")}, "both allowlisted and denylisted"),
        ({"policy_schema": "risk-policy/0"}, "policy_schema"),
        ({"name": "  "}, "name"),
        ({"max_orders_per_minute": 5, "max_orders_per_day": 4}, "cannot exceed"),
    ],
)
def test_unsafe_or_out_of_bounds_policies_are_invalid(kwargs: dict[str, Any], needle: str) -> None:
    violations = limits.validate_policy(RiskPolicy(**kwargs))
    assert any(needle in v for v in violations), violations


def test_tightening_within_ceilings_is_valid() -> None:
    tighter = RiskPolicy(max_order_notional=Decimal("500"), max_spread_bps=Decimal("10"), max_positions=1)
    assert limits.validate_policy(tighter) == []


def test_hard_ceilings_view_is_strings_only() -> None:
    view = limits.hard_ceilings()
    assert view["max_order_notional"] == "25000"
    assert all(isinstance(v, str) for v in view.values())


# ------------------------------------------------------------------------------------------ fingerprints


def test_fingerprint_normalises_decimals() -> None:
    a = order_fingerprint("SPY", Side.BUY, Decimal("10"), Decimal("500.0"), "s")
    b = order_fingerprint("spy", "BUY", Decimal("10.00"), Decimal("500"), "s")
    assert a == b and a.startswith("fp_")


def test_fingerprint_distinguishes_every_component() -> None:
    base = order_fingerprint("SPY", Side.BUY, Decimal("10"), Decimal("500"), "s")
    assert base != order_fingerprint("QQQ", Side.BUY, Decimal("10"), Decimal("500"), "s")
    assert base != order_fingerprint("SPY", Side.SELL, Decimal("10"), Decimal("500"), "s")
    assert base != order_fingerprint("SPY", Side.BUY, Decimal("11"), Decimal("500"), "s")
    assert base != order_fingerprint("SPY", Side.BUY, Decimal("10"), Decimal("500.01"), "s")
    assert base != order_fingerprint("SPY", Side.BUY, Decimal("10"), Decimal("500"), "t")


def test_fingerprint_rejects_floats_and_empty_strategy() -> None:
    with pytest.raises(TypeError):
        order_fingerprint("SPY", Side.BUY, 10.0, Decimal("1"), "s")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strategy_id"):
        order_fingerprint("SPY", Side.BUY, Decimal("1"), Decimal("1"), " ")


def test_proposal_and_intent_fingerprints_agree() -> None:
    engine = make_engine()
    ctx = make_ctx()
    p = make_proposal()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    assert intent_fingerprint(intent, p.strategy_id) == proposal_fingerprint(p) == fingerprint(p)
    assert fingerprint(intent) == order_fingerprint("SPY", Side.BUY, Decimal("3"), Decimal("500.04"), ANY_STRATEGY)
    assert set(fingerprint_variants("SPY", Side.BUY, Decimal("3"), Decimal("500.04"), p.strategy_id)) == {
        fingerprint(p), fingerprint(intent),
    }
    assert engine.fingerprint(intent) == fingerprint(intent)


def test_no_trade_has_no_fingerprint() -> None:
    assert fingerprint(make_no_trade()) is None


def test_recent_fingerprints_window_and_self_exclusion() -> None:
    entries = [
        FingerprintEntry("fp_old", NOW - timedelta(seconds=400), "c1"),
        FingerprintEntry("fp_new", NOW - timedelta(seconds=10), "c2"),
        FingerprintEntry("fp_self", NOW - timedelta(seconds=5), "c3"),
        FingerprintEntry("fp_future", NOW + timedelta(seconds=5), "c4"),
        FingerprintEntry("fp_new", NOW - timedelta(seconds=1), "c5"),
    ]
    got = recent_fingerprints(entries, now=NOW, window_seconds=300, exclude_client_order_id="c3")
    assert got == ("fp_new", "fp_future")


# ------------------------------------------------------------------------------------------ health / context


def test_unknown_health_fails_every_gate() -> None:
    h = unknown_health("no data")
    assert not (h.feed_ok or h.broker_connected or h.reconciled or h.engine_ok or h.strategy_state_ok
                or h.permissions_ok)
    assert h.entries_locked and h.lock_reasons == ("no data",)


def test_build_system_health_halts_force_entries_lock() -> None:
    h = build_system_health(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True,
                            active_halts=[{"halt_id": "halt_1", "reason": "operator"}])
    assert h.entries_locked and h.lock_reasons == ("halt halt_1: operator",)
    h2 = build_system_health(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True,
                             entries_locked=True)
    assert h2.entries_locked and h2.lock_reasons == ("entries locked",)
    h3 = build_system_health(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True)
    assert not h3.entries_locked


def test_build_risk_context_defaults_fail_closed() -> None:
    ctx = build_risk_context(now=NOW, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, account=make_account(),
                             market=make_market(), health=None)
    assert ctx.health.entries_locked
    decision = make_engine().evaluate(make_proposal(), ctx)
    assert not decision.approved


def test_build_risk_context_sums_unrealized_and_rejects_floats() -> None:
    pos = make_position("SPY", "2", "500.00").model_copy(update={"unrealized_pl": Decimal("-12.50")})
    ctx = build_risk_context(now=NOW, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, account=make_account(),
                             positions=[pos], market=make_market(), health=unknown_health(),
                             daily_realized_pl="-3.25")
    assert ctx.daily_unrealized_pl == Decimal("-12.50") and ctx.daily_realized_pl == Decimal("-3.25")
    with pytest.raises(TypeError):
        build_risk_context(now=NOW, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, account=None,
                           market=None, health=None, daily_realized_pl=1.5)  # type: ignore[arg-type]


def test_built_context_with_full_health_is_approvable() -> None:
    health = build_system_health(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True)
    ctx = build_risk_context(now=NOW, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, account=make_account(),
                             market=make_market(), health=health, start_of_day_equity=Decimal("10000"))
    assert make_engine().evaluate(make_proposal(), ctx).approved


# ------------------------------------------------------------------------------------------ market hours


@pytest.mark.parametrize(
    ("when", "is_open"),
    [
        (datetime(2026, 11, 5, 14, 30, tzinfo=UTC), True),  # open instant (09:30 ET) counts as open
        (datetime(2026, 11, 5, 14, 29, 59, tzinfo=UTC), False),
        (datetime(2026, 11, 5, 20, 59, 59, tzinfo=UTC), True),
        (datetime(2026, 11, 5, 21, 0, tzinfo=UTC), False),  # close instant (16:00 ET) counts as closed
        (NOW, True),
        (PRE_MARKET, False),
        (AFTER_HOURS, False),
        (SATURDAY, False),
        (THANKSGIVING, False),
        (datetime(2026, 11, 27, 17, 30, tzinfo=UTC), True),  # 12:30 ET on the early-close day
        (datetime(2026, 11, 27, 18, 30, tzinfo=UTC), False),  # after the 13:00 ET early close
        (datetime(2026, 7, 6, 13, 45, tzinfo=UTC), True),  # summer (EDT): 09:45 ET
        (datetime(2099, 1, 5, 15, 0, tzinfo=UTC), False),  # beyond the calendar range => fail closed
    ],
)
def test_xnys_regular_session(when: datetime, is_open: bool) -> None:
    assert market_hours.is_regular_session_open(when)[0] is is_open


def test_market_session_classification() -> None:
    assert market_hours.market_session_at(NOW) == MarketSession.REGULAR
    assert market_hours.market_session_at(PRE_MARKET) == MarketSession.PRE
    assert market_hours.market_session_at(AFTER_HOURS) == MarketSession.POST
    assert market_hours.market_session_at(SATURDAY) == MarketSession.CLOSED
    assert market_hours.market_session_at(THANKSGIVING) == MarketSession.CLOSED
    assert market_hours.ny_date(datetime(2026, 11, 6, 2, 0, tzinfo=UTC)) == "2026-11-05"


def test_naive_datetime_is_not_open() -> None:
    assert market_hours.is_regular_session_open(datetime(2026, 11, 5, 15, 0))[0] is False
