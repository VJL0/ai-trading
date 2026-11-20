"""Property-based tests (hypothesis) for the risk engine.

The oracle below is an *independent* re-statement of the policy limits written directly from the spec
(simple formulas over the generated inputs). It does not call engine internals, so these properties are
not tautological: they would catch an engine that approves something the plain-language policy forbids
(soundness) or rejects a perfectly valid order (completeness, within the generated domain).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from packages.contracts.enums import DataFeed, GatewayKind, Side, TradingMode
from packages.contracts.models import INTENT_SIGNING_FIELDS, AccountSnapshot, ApprovedOrderIntent, Quote
from packages.contracts.policy import RiskPolicy
from packages.core import modes
from services.risk.engine import RiskAuthorizationError, max_allowable_quantity
from tests.unit.risk.risk_factories import (
    NOW,
    SIGNER,
    healthy,
    make_account,
    make_ctx,
    make_engine,
    make_market,
    make_no_trade,
    make_position,
    make_proposal,
    make_quote,
)

POLICY = RiskPolicy()
ENGINE = make_engine(POLICY)
BPS = Decimal(10_000)
CENT = Decimal("0.01")
PROPS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])

SIZE_CHECKS = frozenset({
    "max_order_notional", "max_position_notional", "max_concentration", "max_planned_loss",
    "daily_loss_budget", "no_margin", "max_allowable_quantity", "proposal_arithmetic",
})


def cents(n: int) -> Decimal:
    return Decimal(n).scaleb(-2)


@dataclass(frozen=True)
class Case:
    side: Side
    bid: Decimal
    ask: Decimal
    limit: Decimal
    qty: Decimal
    stop: Decimal | None
    claimed_extra: Decimal
    equity: Decimal
    cash: Decimal
    existing: Decimal
    daily_pl: Decimal
    ex_age: int
    rx_age: int

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    def stop_used(self) -> Decimal:
        s = self.stop
        if s is not None and Decimal(0) < s < self.limit and (self.limit - s) / self.limit * BPS >= 10:
            return s
        raw = self.limit * (1 - POLICY.default_stop_loss_bps / BPS)
        return raw.quantize(CENT, rounding=ROUND_FLOOR)

    def planned_loss(self) -> Decimal:
        return self.qty * (self.limit - self.stop_used()) if self.side == Side.BUY else Decimal(0)

    def claimed(self) -> Decimal:
        return self.planned_loss() + self.claimed_extra


@st.composite
def cases(draw: st.DrawFn, profile: str = "broad") -> Case:
    """Inputs drawn *relative* to the limits.

    ``broad``: every parameter independently straddles its boundary (mostly rejections).
    ``near_valid``: each parameter is drawn from its safe range with ~80% probability, so a good share of
    cases is approvable while single-boundary violations still occur.
    ``oversized``: safe market inputs, BUY, but size/cash/equity/loss inputs straddle their limits.
    """

    def pick(safe: st.SearchStrategy[Any], wide: st.SearchStrategy[Any]) -> Any:
        if profile == "broad":
            return draw(wide)
        return draw(st.one_of(safe, safe, safe, safe, wide))

    market_safe = profile in ("near_valid", "oversized")
    side = Side.BUY if profile == "oversized" else draw(st.sampled_from([Side.BUY, Side.BUY, Side.BUY, Side.SELL]))
    bid = cents(draw(st.integers(2_000, 80_000)))
    spread_bps = Decimal(draw(st.integers(0, 20) if market_safe else st.integers(0, 32)))
    ask = bid + max(CENT, (bid * spread_bps / BPS).quantize(CENT))
    ref = ask if side == Side.BUY else bid
    offset_bps = Decimal(draw(st.integers(-15, 8)) if market_safe else draw(st.integers(-36, 14)))
    signed_offset = offset_bps if side == Side.BUY else -offset_bps
    limit = (ref * (1 + signed_offset / BPS)).quantize(CENT)
    assume(limit > 0)
    affordable = max(1, int(POLICY.max_order_notional / limit))
    if profile == "oversized":
        qty = Decimal(draw(st.integers(1, affordable * 3 + 3)))
    else:
        qty = Decimal(pick(st.integers(1, max(1, affordable // 2)), st.integers(1, affordable + 2)))
    stop_kind = pick(st.sampled_from(["none", "valid"]), st.sampled_from(["none", "valid", "valid", "random"]))
    stop: Decimal | None
    if stop_kind == "none":
        stop = None
    elif stop_kind == "valid":
        stop = (limit * (1 - Decimal(pick(st.integers(20, 200), st.integers(5, 600))) / BPS)).quantize(CENT)
    else:
        stop = cents(draw(st.integers(1, 90_000)))
    equity = Decimal(pick(st.integers(10_000, 20_000), st.integers(1_800, 20_000)))
    cash_pct = pick(st.just(100), st.sampled_from([100, 100, 100, 60, 30, 10, 0]))
    cash = (equity * Decimal(cash_pct) / 100).quantize(CENT)
    if side == Side.SELL and profile != "broad":
        existing = qty + Decimal(pick(st.integers(0, 3), st.integers(-int(qty), 3)))
    else:
        existing = Decimal(pick(st.just(0), st.sampled_from([0, 0, 0, 1, 2, 5])))
    daily_pl = Decimal(pick(st.sampled_from([0, 50, -100]), st.sampled_from([0, 0, 50, -100, -240, -250, -300])))
    claimed_extra = cents(pick(st.just(0), st.sampled_from([0, 0, 0, 100, 2_000])))
    ages = st.sampled_from([0, 1, 3, 10, 15, 16, 30])
    ex_age = 1 if profile == "oversized" else pick(st.integers(0, 10), ages)
    rx_age = 1 if profile == "oversized" else pick(st.integers(0, 10), ages)
    return Case(side, bid, ask, limit, qty, stop, claimed_extra, equity, cash, existing, daily_pl, ex_age, rx_age)


def build(case: Case) -> tuple[Any, Any]:
    quote = make_quote("SPY", bid=str(case.bid), ask=str(case.ask), exchange_age_s=case.ex_age,
                       received_age_s=case.rx_age)
    positions = (make_position("SPY", str(case.existing), str(case.bid)),) if case.existing > 0 else ()
    ctx = make_ctx(
        account=make_account(equity=str(case.equity), cash=str(case.cash)),
        positions=positions,
        start_of_day_equity=case.equity,
        daily_realized_pl=case.daily_pl,
        market=make_market(quote=quote),
    )
    proposal = make_proposal(side=case.side, quantity=case.qty, limit_price=case.limit, stop_price=case.stop,
                             max_planned_loss=case.claimed())
    return proposal, ctx


def oracle_violations(c: Case) -> list[str]:
    """Plain-language policy, restated independently of the engine."""
    v: list[str] = []
    spread_bps = (c.ask - c.bid) / c.mid * BPS
    if spread_bps > POLICY.max_spread_bps:
        v.append("spread")
    if max(c.ex_age, c.rx_age) > POLICY.max_quote_age_seconds:
        v.append("quote_age")
    if abs(c.limit - c.mid) / c.mid * BPS > POLICY.max_limit_deviation_bps:
        v.append("limit_vs_mid")
    if c.side == Side.BUY:
        if (c.limit - c.ask) / c.ask * BPS > 10:
            v.append("through_ask")
        if min(c.ask, c.limit) < POLICY.min_price:
            v.append("min_price")
        notional = c.qty * c.limit
        if notional > POLICY.max_order_notional:
            v.append("order_notional")
        post = (c.existing + c.qty) * max(c.limit, c.ask)
        if post > POLICY.max_position_notional:
            v.append("position_notional")
        if post / c.equity > POLICY.max_concentration_pct:
            v.append("concentration")
        loss = c.planned_loss()
        if loss > POLICY.max_planned_loss_per_trade:
            v.append("planned_loss")
        worst = min(c.daily_pl, Decimal(0))
        if worst <= -POLICY.max_daily_loss:
            v.append("daily_loss")
        if loss > POLICY.max_daily_loss + worst:
            v.append("daily_budget")
        if c.cash <= 0 or notional * POLICY.min_buying_power_buffer > c.cash:
            v.append("margin")
        if c.equity < POLICY.min_account_equity:
            v.append("equity_floor")
    else:
        if (c.bid - c.limit) / c.bid * BPS > 10:
            v.append("through_bid")
        if min(c.bid, c.limit) < POLICY.min_price:
            v.append("min_price")
        if c.qty > c.existing:
            v.append("short_sale")
    return v


@PROPS
@given(st.one_of(cases("broad"), cases("near_valid"), cases("oversized")))
def test_engine_verdict_matches_independent_policy_oracle(case: Case) -> None:
    proposal, ctx = build(case)
    decision = ENGINE.evaluate(proposal, ctx)
    violations = oracle_violations(case)
    if decision.approved:
        # soundness: nothing the policy forbids is ever approved
        assert violations == [], f"engine approved despite {violations}"
        assert decision.recomputed_notional == case.qty * case.limit
        assert decision.recomputed_max_loss == case.planned_loss()
        assert decision.reference_price == (case.ask if case.side == Side.BUY else case.bid)
    else:
        # completeness (within this domain): a rejection always has an oracle reason
        assert violations, f"engine rejected a valid order: {decision.reasons}"


@PROPS
@given(cases("near_valid"))
def test_approved_decisions_respect_every_limit(case: Case) -> None:
    proposal, ctx = build(case)
    decision = ENGINE.evaluate(proposal, ctx)
    assume(decision.approved)
    assert ctx.market is not None and ctx.market.quote is not None and ctx.account is not None
    q = ctx.market.quote
    assert q.spread_bps <= POLICY.max_spread_bps
    assert (NOW - q.exchange_ts).total_seconds() <= float(POLICY.max_quote_age_seconds)
    assert (NOW - q.received_ts).total_seconds() <= float(POLICY.max_quote_age_seconds)
    assert decision.recomputed_max_loss is not None and decision.recomputed_notional is not None
    assert decision.recomputed_max_loss <= POLICY.max_planned_loss_per_trade
    if case.side == Side.BUY:
        held = sum((p.quantity for p in ctx.positions), Decimal(0))
        post = (held + case.qty) * max(case.limit, q.ask)
        assert decision.recomputed_notional <= POLICY.max_order_notional
        assert post <= POLICY.max_position_notional
        assert post / ctx.account.equity <= POLICY.max_concentration_pct
        assert decision.recomputed_notional * POLICY.min_buying_power_buffer <= ctx.account.cash
    else:
        assert case.qty <= sum((p.quantity for p in ctx.positions), Decimal(0))  # never a short sale


@PROPS
@given(cases("oversized"))
def test_reported_max_allowable_quantity_is_approvable(case: Case) -> None:
    """Never resized: an oversize order is rejected, and the reported max qty itself would pass."""
    assume(case.side == Side.BUY)
    proposal, ctx = build(case)
    decision = ENGINE.evaluate(proposal, ctx)
    failed = {c.name for c in decision.checks if not c.passed}
    assume(not decision.approved and failed <= SIZE_CHECKS and case.claimed_extra == 0)
    max_qty = max_allowable_quantity(decision)
    assert max_qty is not None and max_qty < case.qty
    assert decision.recomputed_quantity == case.qty  # the engine did not resize
    assume(max_qty >= 1)
    smaller = Case(**{**case.__dict__, "qty": max_qty})
    p2, ctx2 = build(smaller)
    again = ENGINE.evaluate(p2, ctx2)
    assert again.approved, again.reasons


money_floats = st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1e7)


@PROPS
@given(money_floats, st.sampled_from(["quantity", "limit_price", "max_planned_loss", "stop_price"]))
def test_any_float_in_proposal_money_fields_is_rejected(value: float, field: str) -> None:
    with pytest.raises(ValidationError, match="float"):
        make_proposal(**{field: value})


@PROPS
@given(money_floats, st.sampled_from(["bid", "ask", "bid_size", "ask_size"]))
def test_any_float_in_quote_is_rejected(value: float, field: str) -> None:
    kwargs: dict[str, Any] = {
        "symbol": "SPY", "bid": "1.00", "ask": "1.01", "bid_size": "1", "ask_size": "1", "exchange_ts": NOW,
        "received_ts": NOW, "available_at_ts": NOW, "feed": DataFeed.SYNTHETIC, "source_id": "t", "is_synthetic": True,
    }
    kwargs[field] = value
    with pytest.raises(ValidationError, match="float"):
        Quote(**kwargs)


@PROPS
@given(money_floats, st.sampled_from(["equity", "cash", "buying_power"]))
def test_any_float_in_account_is_rejected(value: float, field: str) -> None:
    base = make_account().model_dump()
    base[field] = value
    with pytest.raises(ValidationError, match="float"):
        AccountSnapshot(**base)


@PROPS
@given(money_floats, st.sampled_from(["max_order_notional", "max_daily_loss", "max_spread_bps", "min_price"]))
def test_any_float_in_policy_is_rejected(value: float, field: str) -> None:
    with pytest.raises(ValidationError, match="float"):
        RiskPolicy(**{field: value})  # pyright: ignore[reportArgumentType]  # the float is the point


@PROPS
@given(
    st.sampled_from(list(TradingMode)),
    st.sampled_from(list(GatewayKind)),
    st.booleans(),
    st.booleans(),
    st.text(min_size=1, max_size=40).filter(lambda s: s.strip() != ""),
)
def test_no_trade_never_yields_an_intent(mode: TradingMode, gateway: GatewayKind, locked: bool, has_account: bool,
                                         reason: str) -> None:
    ctx = make_ctx(mode=mode, gateway=gateway, health=healthy(entries_locked=locked),
                   account=make_account() if has_account else None)
    nt = make_no_trade(no_trade_reason=reason)
    decision = ENGINE.evaluate(nt, ctx)
    assert not decision.approved
    assert any("no trade requested" in r for r in decision.reasons)
    with pytest.raises(RiskAuthorizationError):
        ENGINE.authorize(nt, decision, ctx)


ALLOWED = {(m, g) for m in TradingMode for g in GatewayKind
           if g not in modes.LIVE_GATEWAYS and g in modes.allowed_gateways(m)}


@PROPS
@given(
    st.sampled_from(list(TradingMode)),
    st.sampled_from(list(GatewayKind)),
    st.sampled_from([DataFeed.SYNTHETIC, DataFeed.IEX]),
)
def test_approvals_only_in_allowed_modes(mode: TradingMode, gateway: GatewayKind, feed: DataFeed) -> None:
    ctx = make_ctx(mode=mode, gateway=gateway, market=make_market(feed=feed))
    proposal = make_proposal()
    decision = ENGINE.evaluate(proposal, ctx)
    if decision.approved:
        assert (mode, gateway) in ALLOWED
        assert mode in (TradingMode.REPLAY, TradingMode.PAPER)
        intent = ENGINE.authorize(proposal, decision, ctx)
        assert intent.mode == mode and intent.gateway not in modes.LIVE_GATEWAYS
    else:
        synthetic_for_paper = gateway in (GatewayKind.LEAN_ALPACA_PAPER, GatewayKind.ALPACA_PAPER) and feed == DataFeed.SYNTHETIC
        assert (mode, gateway) not in ALLOWED or synthetic_for_paper, decision.reasons


_INTENT_CACHE: dict[str, ApprovedOrderIntent] = {}


def _intent() -> ApprovedOrderIntent:
    if "i" not in _INTENT_CACHE:
        ctx = make_ctx()
        p = make_proposal()
        _INTENT_CACHE["i"] = ENGINE.authorize(p, ENGINE.evaluate(p, ctx), ctx)
    return _INTENT_CACHE["i"]


def _mutate(intent: ApprovedOrderIntent, field: str) -> ApprovedOrderIntent:
    value = getattr(intent, field)
    new: Any
    if isinstance(value, bool):
        new = not value
    elif isinstance(value, Decimal):
        new = value + Decimal("1")
    elif field in ("created_at", "expires_at"):
        new = value + timedelta(seconds=1)
    elif field == "mode":
        new = TradingMode.REPLAY if value != TradingMode.REPLAY else TradingMode.PAPER
    elif field == "gateway":
        new = GatewayKind.LEAN_PAPER_SIM
    elif field == "side":
        new = Side.SELL
    elif value is None:
        new = "hap_forged"
    else:
        new = str(value) + "X"
    return intent.model_copy(update={field: new})


@pytest.mark.filterwarnings("ignore::UserWarning")  # pydantic warns when a mutated enum field is a plain str
@PROPS
@given(st.sampled_from(INTENT_SIGNING_FIELDS))
def test_mutating_any_signed_field_fails_the_final_gate(field: str) -> None:
    intent = _intent()
    assert ENGINE.final_gate(intent, make_ctx(), SIGNER.verifier, ENGINE.policy_version, self_recorded=False) == []
    tampered = _mutate(intent, field)
    fails = ENGINE.final_gate(tampered, make_ctx(), SIGNER.verifier, ENGINE.policy_version, self_recorded=False)
    assert fails, f"tampering with {field} was not detected"
