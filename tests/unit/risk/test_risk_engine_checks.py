"""Per-check unit tests for RiskEngine.evaluate: the baseline passes every check, and each test breaks one
input and asserts the named check fails (and the verdict is REJECTED). No I/O, fixed clock."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from packages.contracts.enums import (
    DataFeed,
    DecisionAction,
    GatewayKind,
    MarketSession,
    OrderType,
    RiskVerdict,
    Side,
    TradingMode,
)
from packages.contracts.models import TradeProposal
from packages.contracts.policy import RiskPolicy
from services.risk import engine as eng
from services.risk.context import fingerprint, order_fingerprint
from services.risk.engine import REQUIRED_TRADE_CHECKS, max_allowable_quantity
from tests.unit.risk.risk_factories import (
    AFTER_HOURS,
    EARLY_CLOSE_AFTER,
    NOW,
    PRE_MARKET,
    SATURDAY,
    THANKSGIVING,
    check,
    d,
    failed,
    healthy,
    make_account,
    make_ctx,
    make_engine,
    make_market,
    make_no_trade,
    make_order,
    make_position,
    make_proposal,
    make_quote,
)


def evaluate(proposal: TradeProposal | None = None, policy: RiskPolicy | None = None, *, now: Any = NOW, **ctx: Any):
    engine = make_engine(policy, now=now)
    return engine.evaluate(proposal or make_proposal(), make_ctx(**ctx))


def assert_rejected_by(decision: Any, name: str) -> None:
    assert decision.verdict == RiskVerdict.REJECTED
    assert not decision.approved
    assert name in failed(decision), f"expected {name} to fail; failed={sorted(failed(decision))}"
    assert any(r.startswith(f"{name}:") for r in decision.reasons)


# ------------------------------------------------------------------------------------------ baseline


def test_baseline_is_approved_and_every_required_check_ran_and_passed() -> None:
    dec = evaluate()
    assert dec.verdict == RiskVerdict.APPROVED, dec.reasons
    assert dec.reasons == ()
    names = {c.name for c in dec.checks}
    assert names >= REQUIRED_TRADE_CHECKS
    assert all(c.passed for c in dec.checks)
    assert dec.policy_version == RiskPolicy().version
    assert dec.mode == TradingMode.PAPER
    assert dec.account_snapshot_id == "acct_test" and dec.market_snapshot_id == "snap_SPY"


def test_recomputes_reference_notional_and_loss_independently() -> None:
    dec = evaluate()
    assert dec.reference_price == d("500.04")  # ask for BUY
    assert dec.recomputed_quantity == d("3")
    assert dec.recomputed_notional == d("1500.12")
    assert dec.recomputed_max_loss == d("15.12")  # 3 * (500.04 - 495.00)


def test_sell_reference_price_is_bid() -> None:
    p = make_proposal(side=Side.SELL, quantity=d("2"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    dec = evaluate(p, positions=(make_position("SPY", "5"),))
    assert dec.approved, dec.reasons
    assert dec.reference_price == d("500.00")
    assert dec.recomputed_max_loss == d("0")


def test_missing_stop_uses_policy_default_stop_floored_to_tick() -> None:
    # default 150 bps below 500.04 = 492.5394 -> floored to 492.53; loss 3 * 7.51 = 22.53
    p = make_proposal(stop_price=None, max_planned_loss=d("22.53"))
    dec = evaluate(p)
    assert dec.approved, dec.reasons
    assert dec.recomputed_max_loss == d("22.53")
    assert "policy_default_stop" in check(dec, "max_planned_loss").detail


@pytest.mark.parametrize("bad_stop", ["500.04", "501.00", "499.99"])  # at/above limit, or < 10 bps away
def test_invalid_proposal_stop_is_replaced_by_default_stop(bad_stop: str) -> None:
    p = make_proposal(stop_price=d(bad_stop), max_planned_loss=d("22.53"))
    dec = evaluate(p)
    assert dec.recomputed_max_loss == d("22.53")
    assert "proposal stop invalid" in check(dec, "max_planned_loss").detail


# ------------------------------------------------------------------------------------------ action / mode


def test_no_trade_is_rejected_with_no_trade_requested() -> None:
    dec = evaluate(make_no_trade())
    assert dec.verdict == RiskVerdict.REJECTED
    assert any("no trade requested" in r for r in dec.reasons)
    assert dec.recomputed_notional is None


def test_research_only_rejects() -> None:
    assert_rejected_by(evaluate(mode=TradingMode.RESEARCH_ONLY), "mode_permits_trading")


def test_replay_with_fake_gateway_is_allowed() -> None:
    assert evaluate(mode=TradingMode.REPLAY, gateway=GatewayKind.FAKE).approved


@pytest.mark.parametrize(
    ("mode", "gateway"),
    [
        (TradingMode.REPLAY, GatewayKind.LEAN_ALPACA_PAPER),
        (TradingMode.REPLAY, GatewayKind.LEAN_PAPER_SIM),
        (TradingMode.PAPER, GatewayKind.LEAN_ALPACA_LIVE),
        (TradingMode.LIVE_APPROVAL, GatewayKind.LEAN_ALPACA_LIVE),
        (TradingMode.LIVE_APPROVAL, GatewayKind.FAKE),
        (TradingMode.BOUNDED_LIVE, GatewayKind.FAKE),
        (TradingMode.BOUNDED_LIVE, GatewayKind.LEAN_ALPACA_LIVE),
    ],
)
def test_disallowed_mode_gateway_pairs_reject(mode: TradingMode, gateway: GatewayKind) -> None:
    assert_rejected_by(evaluate(mode=mode, gateway=gateway), "mode_permits_trading")


def test_live_mode_rejection_explains_compiled_off() -> None:
    dec = evaluate(mode=TradingMode.LIVE_APPROVAL, gateway=GatewayKind.LEAN_ALPACA_LIVE)
    assert "LIVE_EXECUTION_COMPILED is False" in check(dec, "mode_permits_trading").detail


def test_policy_violating_hard_limits_blocks_everything() -> None:
    dec = evaluate(policy=RiskPolicy(allow_margin=True))
    assert_rejected_by(dec, "policy_valid")


# ------------------------------------------------------------------------------------------ symbol


def test_symbol_format_rejects_digits_and_long_tickers() -> None:
    p = make_proposal(symbol="AB1")
    assert_rejected_by(evaluate(p, market=make_market("AB1")), "symbol_format")
    p = make_proposal(symbol="ABCDEF")
    assert_rejected_by(evaluate(p, market=make_market("ABCDEF")), "symbol_format")


def test_symbol_not_on_allowlist_rejects() -> None:
    p = make_proposal(symbol="IBM")
    assert_rejected_by(evaluate(p, market=make_market("IBM")), "symbol_allowlist")


def test_leveraged_etf_is_denied_even_if_a_bad_policy_allowlists_it() -> None:
    bad = RiskPolicy(
        symbol_allowlist=(*RiskPolicy().symbol_allowlist, "TQQQ"),
        symbol_denylist=tuple(s for s in RiskPolicy().symbol_denylist if s != "TQQQ"),
    )
    p = make_proposal(symbol="TQQQ")
    dec = evaluate(p, policy=bad, market=make_market("TQQQ"))
    assert_rejected_by(dec, "symbol_denylist")
    assert "policy_valid" in failed(dec)


def test_min_price_rejects_low_priced_names() -> None:
    q = make_quote("SPY", bid="4.00", ask="4.01")
    p = make_proposal(quantity=d("10"), limit_price=d("4.01"), stop_price=d("3.90"), max_planned_loss=d("1.10"))
    assert_rejected_by(evaluate(p, market=make_market(quote=q)), "min_price")


# ------------------------------------------------------------------------------------------ order shape


def test_market_order_rejected_by_contract_and_engine() -> None:
    with pytest.raises(ValidationError):
        make_proposal(order_type=OrderType.MARKET)
    smuggled = make_proposal().model_copy(update={"order_type": OrderType.MARKET})  # bypasses validation
    assert_rejected_by(evaluate(smuggled), "order_type_limit_only")


def test_fractional_quantity_rejected() -> None:
    p = make_proposal(quantity=d("2.5"), max_planned_loss=d("12.60"))
    assert_rejected_by(evaluate(p), "whole_shares")


def test_sub_tick_limit_rejected() -> None:
    p = make_proposal(limit_price=d("500.035"))
    assert_rejected_by(evaluate(p), "limit_price_tick")


def test_limit_far_from_mid_rejected() -> None:
    p = make_proposal(limit_price=d("502.00"), max_planned_loss=d("21.00"))
    dec = evaluate(p)
    assert_rejected_by(dec, "limit_vs_mid")


def test_buy_limit_through_ask_rejected_even_within_mid_band() -> None:
    p = make_proposal(limit_price=d("500.79"), max_planned_loss=d("17.37"))  # ~15 bps above ask, ~15 bps from mid
    dec = evaluate(p)
    assert_rejected_by(dec, "limit_vs_touch")
    assert check(dec, "limit_vs_mid").passed


def test_sell_limit_below_bid_rejected() -> None:
    p = make_proposal(side=Side.SELL, quantity=d("2"), limit_price=d("499.20"), stop_price=None, max_planned_loss=d("0"))
    dec = evaluate(p, positions=(make_position("SPY", "5"),))
    assert_rejected_by(dec, "limit_vs_touch")


# ------------------------------------------------------------------------------------------ long-only / margin


def test_sell_without_position_is_a_short_sale_and_rejected() -> None:
    p = make_proposal(side=Side.SELL, quantity=d("1"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    assert_rejected_by(evaluate(p), "long_only")


def test_sell_more_than_held_rejected() -> None:
    p = make_proposal(side=Side.SELL, quantity=d("6"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    dec = evaluate(p, positions=(make_position("SPY", "5"),))
    assert_rejected_by(dec, "long_only")
    assert max_allowable_quantity(dec) == d("5")


def test_sell_net_of_pending_sells_rejected() -> None:
    p = make_proposal(side=Side.SELL, quantity=d("3"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    pending = make_order("SPY", Side.SELL, "3", "500.00")
    dec = evaluate(p, positions=(make_position("SPY", "5"),), open_orders=(pending,),
                   recent_submission_times=())
    assert_rejected_by(dec, "long_only")


def test_cash_shortfall_rejected_no_margin() -> None:
    dec = evaluate(account=make_account(cash="1000.00", equity="10000.00"))
    assert_rejected_by(dec, "no_margin")


def test_margin_buying_power_is_never_used() -> None:
    acct = make_account(cash="1000.00", buying_power="40000.00", equity="10000.00")
    assert_rejected_by(evaluate(account=acct), "no_margin")


def test_non_marginable_buying_power_caps_spend() -> None:
    acct = make_account(cash="10000.00", non_marginable_buying_power="1000.00")
    assert_rejected_by(evaluate(account=acct), "no_margin")


def test_buying_power_buffer_applies() -> None:
    # 1500.12 * 1.02 = 1530.12 > 1520 cash, although 1500.12 alone would fit
    acct = make_account(cash="1520.00", equity="10000.00")
    assert_rejected_by(evaluate(account=acct), "no_margin")


def test_negative_cash_margin_debit_rejected() -> None:
    acct = make_account(cash="-50.00", equity="10000.00", buying_power="20000.00")
    assert_rejected_by(evaluate(account=acct), "no_margin")


def test_pending_buy_cash_commitment_counts_against_cash() -> None:
    policy = RiskPolicy(max_pending_orders=2, max_positions=2)
    pending = make_order("QQQ", Side.BUY, "10", "400.00")  # commits 4000 of 5000 cash
    acct = make_account(cash="5000.00", equity="10000.00")
    dec = evaluate(policy=policy, account=acct, open_orders=(pending,))
    assert_rejected_by(dec, "no_margin")


# ------------------------------------------------------------------------------------------ positions / size


def test_one_position_at_a_time() -> None:
    dec = evaluate(positions=(make_position("QQQ", "1", "400.00"),))
    assert_rejected_by(dec, "max_positions")


def test_pending_buy_in_other_symbol_counts_as_position() -> None:
    dec = evaluate(policy=RiskPolicy(max_pending_orders=2), open_orders=(make_order("QQQ", Side.BUY, "1", "400.00"),))
    assert_rejected_by(dec, "max_positions")


def test_adding_to_the_same_position_is_not_a_second_position() -> None:
    dec = evaluate(make_proposal(quantity=d("1"), max_planned_loss=d("5.04")), positions=(make_position("SPY", "2"),))
    assert dec.approved, dec.reasons


def test_oversized_order_is_rejected_not_resized_and_max_qty_reported() -> None:
    p = make_proposal(quantity=d("10"), max_planned_loss=d("50.40"))
    dec = evaluate(p)
    assert_rejected_by(dec, "max_order_notional")
    assert_rejected_by(dec, "max_allowable_quantity")
    assert dec.recomputed_quantity == d("10")  # never silently resized
    assert dec.recomputed_notional == d("5000.40")
    assert max_allowable_quantity(dec) == d("3")
    assert "never resized" in check(dec, "max_allowable_quantity").detail


def test_post_trade_position_notional_limit() -> None:
    dec = evaluate(positions=(make_position("SPY", "2"),))  # 5 * 500.04 = 2500.20 > 2500
    assert_rejected_by(dec, "max_position_notional")
    assert max_allowable_quantity(dec) == d("2")


def test_position_notional_counts_pending_buys_in_symbol() -> None:
    policy = RiskPolicy(max_pending_orders=2)
    dec = evaluate(policy=policy, open_orders=(make_order("SPY", Side.BUY, "2", "500.00", coid="lat-sim-p2"),))
    assert_rejected_by(dec, "max_position_notional")


def test_concentration_limit() -> None:
    acct = make_account(equity="5000.00", cash="5000.00")
    dec = evaluate(account=acct, start_of_day_equity=d("5000.00"))  # 1500.12 / 5000 = 0.30 > 0.25
    assert_rejected_by(dec, "max_concentration")
    assert max_allowable_quantity(dec) == d("2")


def test_planned_loss_per_trade_limit() -> None:
    p = make_proposal(stop_price=d("480.00"), max_planned_loss=d("60.12"))
    dec = evaluate(p)
    assert_rejected_by(dec, "max_planned_loss")
    assert max_allowable_quantity(dec) == d("2")  # 50 / 20.04 = 2.49


def test_model_understated_planned_loss_is_not_trusted() -> None:
    p = make_proposal(max_planned_loss=d("1.00"))  # true loss is 15.12
    dec = evaluate(p)
    assert_rejected_by(dec, "proposal_arithmetic")
    assert dec.recomputed_max_loss == d("15.12")
    assert "claimed=1.00" in (check(dec, "proposal_arithmetic").observed or "")


def test_overstated_planned_loss_is_recorded_and_harmless() -> None:
    dec = evaluate(make_proposal(max_planned_loss=d("40.00")))
    assert dec.approved
    assert dec.recomputed_max_loss == d("15.12")


# ------------------------------------------------------------------------------------------ daily loss / equity


def test_daily_loss_limit_reached_blocks_entries() -> None:
    assert_rejected_by(evaluate(daily_realized_pl=d("-200.00"), daily_unrealized_pl=d("-50.00")), "max_daily_loss")


def test_daily_loss_from_equity_drop_even_if_ledger_pl_is_zero() -> None:
    acct = make_account(equity="9700.00", cash="9700.00")
    assert_rejected_by(evaluate(account=acct, start_of_day_equity=d("10000.00")), "max_daily_loss")


def test_daily_loss_uses_last_equity_when_start_of_day_missing() -> None:
    acct = make_account(equity="9700.00", cash="9700.00", last_equity="10000.00")
    assert_rejected_by(evaluate(account=acct, start_of_day_equity=None), "max_daily_loss")


def test_daily_pl_unknown_fails_closed() -> None:
    assert_rejected_by(evaluate(start_of_day_equity=None), "max_daily_loss")


def test_remaining_daily_loss_budget() -> None:
    dec = evaluate(daily_realized_pl=d("-240.00"))  # 10 left; planned loss 15.12
    assert_rejected_by(dec, "daily_loss_budget")
    assert check(dec, "max_daily_loss").passed


def test_account_equity_floor() -> None:
    acct = make_account(equity="1900.00", cash="1900.00")
    assert_rejected_by(evaluate(account=acct, start_of_day_equity=d("1900.00")), "account_equity_floor")


# ------------------------------------------------------------------------------------------ market data


def test_wide_spread_rejected() -> None:
    q = make_quote(bid="499.00", ask="500.50")  # ~30 bps
    p = make_proposal(limit_price=d("500.50"), max_planned_loss=d("16.50"))
    assert_rejected_by(evaluate(p, market=make_market(quote=q)), "max_spread")


def test_stale_exchange_timestamp_rejected() -> None:
    q = make_quote(exchange_age_s=20, received_age_s=1)
    assert_rejected_by(evaluate(market=make_market(quote=q)), "max_quote_age")


def test_stale_received_timestamp_rejected() -> None:
    q = make_quote(exchange_age_s=1, received_age_s=20)
    assert_rejected_by(evaluate(market=make_market(quote=q)), "max_quote_age")


def test_future_quote_rejected_as_look_ahead() -> None:
    q = make_quote(exchange_age_s=-30, received_age_s=-30)
    assert_rejected_by(evaluate(market=make_market(quote=q)), "max_quote_age")


def test_missing_quote_rejected() -> None:
    dec = evaluate(market=make_market(no_quote=True))
    assert_rejected_by(dec, "quote_present")
    assert_rejected_by(dec, "max_quote_age")
    assert dec.reference_price is None


def test_missing_market_snapshot_rejected() -> None:
    assert_rejected_by(evaluate(market=None), "market_snapshot_present")


def test_market_snapshot_for_other_symbol_rejected() -> None:
    assert_rejected_by(evaluate(market=make_market("QQQ")), "market_snapshot_present")


def test_stale_proposal_data_asof_rejected() -> None:
    p = make_proposal(data_asof=NOW - timedelta(minutes=10))
    assert_rejected_by(evaluate(p), "data_asof_fresh")


def test_stale_snapshot_data_asof_rejected() -> None:
    assert_rejected_by(evaluate(market=make_market(data_asof=NOW - timedelta(minutes=10))), "data_asof_fresh")


def test_future_data_asof_rejected() -> None:
    p = make_proposal(data_asof=NOW + timedelta(minutes=1))
    assert_rejected_by(evaluate(p), "data_asof_fresh")


def test_synthetic_data_cannot_drive_alpaca_paper_orders() -> None:
    dec = evaluate(gateway=GatewayKind.LEAN_ALPACA_PAPER)
    assert_rejected_by(dec, "data_provenance")
    ok = evaluate(gateway=GatewayKind.LEAN_ALPACA_PAPER, market=make_market(feed=DataFeed.IEX))
    assert ok.approved, ok.reasons


# ------------------------------------------------------------------------------------------ session


@pytest.mark.parametrize("session", [MarketSession.PRE, MarketSession.POST, MarketSession.CLOSED])
def test_extended_or_closed_snapshot_session_rejected(session: MarketSession) -> None:
    assert_rejected_by(evaluate(market=make_market(session=session)), "regular_session")


@pytest.mark.parametrize("when", [PRE_MARKET, AFTER_HOURS, SATURDAY, THANKSGIVING, EARLY_CLOSE_AFTER])
def test_calendar_says_closed_even_if_snapshot_claims_regular(when: Any) -> None:
    p = make_proposal(now=when)
    dec = make_engine(now=when).evaluate(p, make_ctx(now=when))
    assert_rejected_by(dec, "regular_session")


# ------------------------------------------------------------------------------------------ orders / rate


def test_max_pending_orders() -> None:
    dec = evaluate(open_orders=(make_order("SPY", Side.BUY, "1", "499.00"),))
    assert_rejected_by(dec, "max_pending_orders")


def test_terminal_orders_do_not_count_as_pending() -> None:
    from packages.contracts.enums import OrderState

    done = make_order("SPY", Side.BUY, "1", "499.00", state=OrderState.FILLED)
    assert evaluate(open_orders=(done,)).approved


def test_order_rate_per_minute() -> None:
    times = tuple(NOW - timedelta(seconds=s) for s in (5, 20, 40))
    assert_rejected_by(evaluate(recent_submission_times=times), "order_rate_per_minute")
    old = tuple(NOW - timedelta(seconds=s) for s in (61, 70, 80))
    assert evaluate(recent_submission_times=old).approved


def test_order_rate_per_day_counts_new_york_day_only() -> None:
    today = tuple(NOW - timedelta(minutes=2 + i) for i in range(20))
    assert_rejected_by(evaluate(recent_submission_times=today), "order_rate_per_day")
    yesterday = tuple(NOW - timedelta(days=1, minutes=i) for i in range(20))
    assert evaluate(recent_submission_times=yesterday).approved


# ------------------------------------------------------------------------------------------ account


@pytest.mark.parametrize(
    "kwargs",
    [{"trading_blocked": True}, {"account_blocked": True}, {"status": "ACCOUNT_CLOSED"}, {"is_paper": False}],
)
def test_account_status_problems_reject(kwargs: dict[str, Any]) -> None:
    assert_rejected_by(evaluate(account=make_account(**kwargs)), "account_status")


def test_missing_account_rejects() -> None:
    dec = evaluate(account=None)
    assert_rejected_by(dec, "account_present")
    assert_rejected_by(dec, "account_fresh")


def test_stale_account_snapshot_rejects() -> None:
    assert_rejected_by(evaluate(account=make_account(age_s=120)), "account_fresh")


def test_short_position_in_account_is_unexpected_state() -> None:
    assert_rejected_by(evaluate(positions=(make_position("QQQ", "-1", "400.00"),)), "positions_consistent")


def test_stale_risk_context_rejects() -> None:
    engine = make_engine(now=NOW + timedelta(seconds=30))
    dec = engine.evaluate(make_proposal(), make_ctx())
    assert_rejected_by(dec, "context_fresh")


# ------------------------------------------------------------------------------------------ health


@pytest.mark.parametrize(
    ("field", "check_name"),
    [
        ("feed_ok", "health_feed"),
        ("broker_connected", "health_broker_connected"),
        ("reconciled", "health_reconciled"),
        ("engine_ok", "health_engine"),
        ("strategy_state_ok", "health_strategy_state"),
        ("permissions_ok", "health_permissions"),
    ],
)
def test_health_gates(field: str, check_name: str) -> None:
    assert_rejected_by(evaluate(health=healthy(**{field: False})), check_name)


def test_entries_locked_blocks() -> None:
    dec = evaluate(health=healthy(entries_locked=True, lock_reasons=("startup_unreconciled",)))
    assert_rejected_by(dec, "health_entries_unlocked")
    assert "startup_unreconciled" in check(dec, "health_entries_unlocked").detail


# ------------------------------------------------------------------------------------------ duplicates / evidence


def test_duplicate_by_strategy_fingerprint() -> None:
    p = make_proposal()
    fp = fingerprint(p)
    assert fp is not None
    assert_rejected_by(evaluate(p, recent_intent_fingerprints=(fp,)), "duplicate_order")


def test_duplicate_by_strategy_agnostic_intent_fingerprint() -> None:
    fp = order_fingerprint("SPY", Side.BUY, d("3"), d("500.040"), "*")  # 500.040 == 500.04 after normalisation
    assert_rejected_by(evaluate(recent_intent_fingerprints=(fp,)), "duplicate_order")


def test_different_quantity_is_not_a_duplicate() -> None:
    fp = order_fingerprint("SPY", Side.BUY, d("2"), d("500.04"), "test-strategy")
    assert evaluate(recent_intent_fingerprints=(fp,)).approved


def test_identical_open_order_is_duplicate() -> None:
    twin = make_order("SPY", Side.BUY, "3", "500.04")
    dec = evaluate(policy=RiskPolicy(max_pending_orders=3), open_orders=(twin,))
    assert_rejected_by(dec, "duplicate_order")


def test_evidence_required() -> None:
    with pytest.raises(ValidationError):
        make_proposal(evidence_ids=())
    smuggled = make_proposal().model_copy(update={"evidence_ids": ()})
    assert_rejected_by(evaluate(smuggled), "evidence_present")
    blank = make_proposal().model_copy(update={"evidence_ids": ("  ",)})
    assert_rejected_by(evaluate(blank), "evidence_present")


def test_incomplete_trade_proposal_smuggled_past_contract_rejects() -> None:
    smuggled = TradeProposal.model_construct(**{**make_proposal().model_dump(), "quantity": None})
    dec = evaluate(smuggled)
    assert_rejected_by(dec, "proposal_complete")


def test_no_trade_action_value_other_than_trade_never_approves() -> None:
    smuggled = make_proposal().model_copy(update={"action": DecisionAction.NO_TRADE})
    dec = evaluate(smuggled)
    assert not dec.approved


def test_check_exception_fails_that_check_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("injected")

    monkeypatch.setattr(eng.RiskEngine, "_chk_spread", boom)
    dec = evaluate()
    assert_rejected_by(dec, "max_spread")
    assert "fail closed" in check(dec, "max_spread").detail


def test_float_quantities_cannot_enter_contracts() -> None:
    with pytest.raises(ValidationError):
        make_proposal(quantity=3.0)
    with pytest.raises(ValidationError):
        make_proposal(limit_price=500.04)
    with pytest.raises(ValidationError):
        make_account(equity=10000.0)  # type: ignore[arg-type]


def test_decimal_amounts_stay_decimal_in_decision() -> None:
    dec = evaluate()
    for value in (dec.recomputed_notional, dec.recomputed_max_loss, dec.reference_price, dec.recomputed_quantity):
        assert isinstance(value, Decimal)


def test_stale_position_snapshot_is_unknown_state() -> None:
    stale = make_position("SPY", "5").model_copy(update={"asof": NOW - timedelta(minutes=10)})
    p = make_proposal(side=Side.SELL, quantity=d("2"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    assert_rejected_by(evaluate(p, positions=(stale,)), "positions_consistent")


def test_smuggled_inputs_fail_contract_revalidation() -> None:
    market_order = make_proposal().model_copy(update={"order_type": OrderType.MARKET})
    assert_rejected_by(evaluate(market_order), "input_contracts_valid")
    float_cash = make_ctx().model_copy(update={"account": make_account().model_copy(update={"cash": 10000.0})})
    dec = make_engine().evaluate(make_proposal(), float_cash)
    assert_rejected_by(dec, "input_contracts_valid")
    naive = make_ctx().model_copy(update={"now": NOW.replace(tzinfo=None)})
    assert not make_engine().evaluate(make_proposal(), naive).approved


def test_lock_reason_without_flag_still_locks() -> None:
    dec = evaluate(health=healthy(entries_locked=False, lock_reasons=("reconciliation_mismatch",)))
    assert_rejected_by(dec, "health_entries_unlocked")


def test_evaluate_is_deterministic_given_ids_and_clock() -> None:
    from packages.core.clock import ManualClock
    from services.risk.engine import RiskEngine
    from tests.unit.risk.risk_factories import SIGNER

    def run() -> Any:
        counter = iter(range(100))
        engine = RiskEngine(RiskPolicy(), SIGNER, ManualClock(NOW), id_factory=lambda p: f"{p}_{next(counter)}")
        return [engine.evaluate(make_proposal(quantity=d(q), max_planned_loss=d(q) * d("5.04")), make_ctx())
                for q in ("1", "3", "10")]

    assert run() == run()


def test_evaluate_and_final_gate_do_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import socket

    engine = make_engine()  # calendar loaded/warmed at construction
    p = make_proposal()
    ctx = make_ctx()

    def no_io(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("I/O attempted inside the risk engine")

    monkeypatch.setattr(builtins, "open", no_io)
    monkeypatch.setattr(socket, "socket", no_io)
    decision = engine.evaluate(p, ctx)
    assert decision.approved, decision.reasons
    intent = engine.authorize(p, decision, ctx)
    from tests.unit.risk.risk_factories import SIGNER

    assert engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy_version, self_recorded=False) == []


@pytest.mark.parametrize(
    ("target", "field", "raw", "gate"),
    [
        ("health", "feed_ok", "no", "health_feed"),
        ("health", "broker_connected", "0", "health_broker_connected"),
        ("health", "reconciled", "false", "health_reconciled"),
        ("health", "engine_ok", "off", "health_engine"),
        ("health", "strategy_state_ok", "no", "health_strategy_state"),
        ("health", "permissions_ok", "n", "health_permissions"),
        ("account", "is_paper", "false", "account_status"),
    ],
)
def test_truthy_strings_smuggled_past_validation_fail_closed(target: str, field: str, raw: str, gate: str) -> None:
    """A raw string is truthy, so checks must read the re-validated (coerced) value, never the caller's object."""
    ctx = make_ctx()
    part = getattr(ctx, target)
    smuggled = ctx.model_copy(update={target: part.model_copy(update={field: raw})})  # model_copy skips validation
    dec = make_engine().evaluate(make_proposal(), smuggled)
    assert_rejected_by(dec, gate)
    assert_rejected_by(dec, "input_contracts_valid")


def test_quantity_string_smuggled_past_validation_is_not_canonical() -> None:
    smuggled = make_proposal().model_copy(update={"quantity": "3"})
    assert_rejected_by(evaluate(smuggled), "input_contracts_valid")


def test_test_factories_never_launder_floats() -> None:
    """Guards the guards: factories must hand raw values to the contracts, so float rejection stays real."""
    with pytest.raises(ValidationError, match="float"):
        make_quote(bid=500.0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="float"):
        make_order(qty=1.0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="float"):
        make_account(cash=1.0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
