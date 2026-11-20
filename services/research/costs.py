"""Transaction-cost model for U.S. equities (spec §3: "estimate spread, fees, and slippage").

All arithmetic is ``Decimal``. Components, per order leg:

* **Commission** — Alpaca equities are commission-free ($0); per-share/per-order/minimum knobs exist
  for sensitivity analysis.
* **Regulatory fees on SELLS only** (passed through by the broker):
  - SEC Section 31 fee: ``rate x sale proceeds``.
  - FINRA Trading Activity Fee (TAF): ``per-share x shares sold``, capped per trade.
  Each fee is rounded *up* to the next cent by default (conservative; brokers round pass-through fees
  to the cent). **The default rates below are as-of dated and must be verified** — the SEC resets the
  Section 31 rate at least annually and FINRA amends Schedule A periodically.
* **Spread** — crossing cost = *half* the quoted spread. Uses the live quote spread when supplied,
  otherwise a conservative estimate by liquidity bucket (ADV in dollars), otherwise a conservative
  unknown-liquidity default.
* **Slippage** — fixed bps + square-root participation impact ``coef x sqrt(qty / ADV)``; a
  conservative fixed impact is used when ADV is unknown.

The model is deliberately conservative: overstated costs suppress marginal candidates (fail closed).
It is *not* calibrated against live fills; Alpaca paper fills omit impact/queue effects entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal

from pydantic import BaseModel, ConfigDict

from packages.contracts.enums import Side
from packages.contracts.types import Dec
from packages.core.money import BPS, CENT, ZERO, D

BPS_Q = Decimal("0.000001")
USD_Q = Decimal("0.000001")

SEC_FEE_RATE_DEFAULT = Decimal("0.0000278")
SEC_FEE_ASOF = (
    "Default $27.80 per $1,000,000 of covered sale proceeds = SEC Section 31 rate effective 2024-05-22 "
    "(last non-zero rate known to this build, recorded 2026-11-05). The SEC resets the rate at least "
    "annually and via mid-year adjustments (a FY2025 mid-year adjustment set it to $0.00 per $1M from "
    "2025-05-14). VERIFY CURRENT RATE at sec.gov fee-rate advisories; this default is deliberately "
    "conservative (overstates sell costs if the current rate is lower)."
)
FINRA_TAF_PER_SHARE_DEFAULT = Decimal("0.000166")
FINRA_TAF_MAX_DEFAULT = Decimal("8.30")
FINRA_TAF_ASOF = (
    "$0.000166/share sold, max $8.30/trade; FINRA By-Laws Schedule A Sec.1(b) covered equity sales, "
    "effective 2024-01-01 (recorded 2026-11-05). VERIFY CURRENT RATE (FINRA amends Schedule A periodically)."
)


@dataclass(frozen=True)
class FeeSchedule:
    commission_per_share: Decimal = ZERO
    commission_per_order: Decimal = ZERO
    commission_min_per_order: Decimal = ZERO
    sec_fee_rate: Decimal = SEC_FEE_RATE_DEFAULT
    sec_fee_asof: str = SEC_FEE_ASOF
    finra_taf_per_share: Decimal = FINRA_TAF_PER_SHARE_DEFAULT
    finra_taf_max_per_trade: Decimal = FINRA_TAF_MAX_DEFAULT
    finra_taf_asof: str = FINRA_TAF_ASOF
    round_fees_up_to_cent: bool = True
    source: str = "Alpaca equities: $0 commission; SEC/FINRA pass-through on sells"

    def __post_init__(self) -> None:
        for name in (
            "commission_per_share",
            "commission_per_order",
            "commission_min_per_order",
            "sec_fee_rate",
            "finra_taf_per_share",
            "finra_taf_max_per_trade",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or value < 0:
                raise ValueError(f"{name} must be a non-negative Decimal")


@dataclass(frozen=True)
class LiquidityBucket:
    """Assumed *full* quoted spread for names whose ADV (USD) is at least ``min_adv_usd``."""

    min_adv_usd: Decimal
    spread_bps: Decimal
    label: str


DEFAULT_LIQUIDITY_BUCKETS: tuple[LiquidityBucket, ...] = (
    LiquidityBucket(Decimal(1_000_000_000), Decimal(2), "MEGA (ADV >= $1B)"),
    LiquidityBucket(Decimal(100_000_000), Decimal(5), "LARGE (ADV >= $100M)"),
    LiquidityBucket(Decimal(10_000_000), Decimal(15), "MID (ADV >= $10M)"),
    LiquidityBucket(ZERO, Decimal(50), "ILLIQUID (ADV < $10M)"),
)


@dataclass(frozen=True)
class SpreadModel:
    buckets: tuple[LiquidityBucket, ...] = DEFAULT_LIQUIDITY_BUCKETS
    unknown_liquidity_spread_bps: Decimal = Decimal(30)

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("at least one liquidity bucket required")
        ordered = sorted(self.buckets, key=lambda b: b.min_adv_usd, reverse=True)
        if list(self.buckets) != ordered:
            raise ValueError("liquidity buckets must be ordered by min_adv_usd descending")
        if self.buckets[-1].min_adv_usd != 0:
            raise ValueError("last liquidity bucket must start at 0 so every ADV is covered")

    def full_spread_bps(self, adv_usd: Decimal | None) -> tuple[Decimal, str]:
        if adv_usd is None:
            return self.unknown_liquidity_spread_bps, "UNKNOWN_LIQUIDITY_DEFAULT"
        for bucket in self.buckets:
            if adv_usd >= bucket.min_adv_usd:
                return bucket.spread_bps, f"LIQUIDITY_BUCKET:{bucket.label}"
        raise AssertionError("unreachable: last bucket starts at 0")  # pragma: no cover


@dataclass(frozen=True)
class SlippageModel:
    fixed_bps: Decimal = Decimal(2)
    impact_coef_bps: Decimal = Decimal(100)
    """impact_bps = impact_coef_bps * sqrt(quantity / ADV shares): 1% of ADV -> 10 bps."""
    unknown_adv_impact_bps: Decimal = Decimal(10)
    max_participation: Decimal = Decimal("0.05")


class CostEstimate(BaseModel):
    """Estimated cost of ONE order leg. ``*_bps`` are relative to ``notional``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: Side
    quantity: Dec
    price: Dec
    notional: Dec
    commission_usd: Dec
    sec_fee_usd: Dec
    finra_taf_usd: Dec
    fees_usd: Dec
    fees_bps: Dec
    quoted_spread_bps: Dec
    spread_source: str
    spread_cost_bps: Dec
    spread_cost_usd: Dec
    participation: Dec | None
    slippage_fixed_bps: Dec
    impact_bps: Dec
    slippage_bps: Dec
    slippage_usd: Dec
    total_usd: Dec
    total_bps: Dec
    notes: tuple[str, ...] = ()


class RoundTripEstimate(BaseModel):
    """BUY entry + SELL exit. A long-only position always pays the sell-side regulatory fees on exit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: CostEstimate
    exit: CostEstimate
    spread_bps: Dec
    fees_bps: Dec
    slippage_bps: Dec
    total_bps: Dec
    total_usd: Dec


def _q_bps(value: Decimal) -> Decimal:
    return value.quantize(BPS_Q, rounding=ROUND_HALF_EVEN)


def _q_usd(value: Decimal) -> Decimal:
    return value.quantize(USD_Q, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class CostModel:
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    spread: SpreadModel = field(default_factory=SpreadModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    version: str = "cost-model-v1"

    def _fee(self, amount: Decimal) -> Decimal:
        if self.fees.round_fees_up_to_cent and amount > 0:
            return amount.quantize(CENT, rounding=ROUND_CEILING)
        return _q_usd(amount)

    def estimate(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal | int | str,
        price: Decimal | int | str,
        *,
        spread_bps: Decimal | int | str | None = None,
        adv_shares: Decimal | int | str | None = None,
    ) -> CostEstimate:
        """Cost of one leg. ``spread_bps`` is the *full* quoted spread (e.g. ``Quote.spread_bps``)."""
        qty = D(quantity)
        px = D(price)
        if not (qty.is_finite() and px.is_finite()):
            raise ValueError("quantity and price must be finite")
        if qty <= 0:
            raise ValueError("quantity must be positive")
        if px <= 0:
            raise ValueError("price must be positive")
        if not isinstance(side, Side):
            raise TypeError("side must be a Side")
        notional = qty * px
        notes: list[str] = []

        # commission (both legs)
        commission = self.fees.commission_per_order + self.fees.commission_per_share * qty
        if commission < self.fees.commission_min_per_order:
            commission = self.fees.commission_min_per_order
        commission = _q_usd(commission)

        # regulatory pass-through fees (sells only)
        sec_fee = ZERO
        taf = ZERO
        if side == Side.SELL:
            sec_fee = self._fee(notional * self.fees.sec_fee_rate)
            raw_taf = qty * self.fees.finra_taf_per_share
            if raw_taf > self.fees.finra_taf_max_per_trade:
                raw_taf = self.fees.finra_taf_max_per_trade
                notes.append("FINRA TAF capped at max per trade")
            taf = self._fee(raw_taf)
        fees_usd = commission + sec_fee + taf

        # spread
        adv = None if adv_shares is None else D(adv_shares)
        if adv is not None and not adv.is_finite():
            raise ValueError("adv_shares must be finite")
        if adv is not None and adv <= 0:
            adv = None
            notes.append("non-positive ADV treated as unknown")
        if spread_bps is not None:
            quoted = D(spread_bps)
            if not quoted.is_finite():
                raise ValueError("spread_bps must be finite")
            if quoted < 0:
                raise ValueError("spread_bps must be non-negative")
            source = "QUOTE"
        else:
            quoted, source = self.spread.full_spread_bps(None if adv is None else adv * px)
            notes.append("spread estimated (no quote supplied)")
        spread_cost_bps = quoted / 2

        # slippage = fixed + sqrt participation impact
        participation: Decimal | None = None
        if adv is None:
            impact = self.slippage.unknown_adv_impact_bps
            notes.append("ADV unknown: conservative fixed impact applied")
        else:
            participation = qty / adv
            impact = self.slippage.impact_coef_bps * participation.sqrt()
            if participation > self.slippage.max_participation:
                notes.append(
                    f"participation {participation:.4f} exceeds max {self.slippage.max_participation}"
                )
        slippage_bps = self.slippage.fixed_bps + impact

        spread_usd = notional * spread_cost_bps / BPS
        slippage_usd = notional * slippage_bps / BPS
        total_usd = fees_usd + spread_usd + slippage_usd
        return CostEstimate(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=px,
            notional=notional,
            commission_usd=commission,
            sec_fee_usd=sec_fee,
            finra_taf_usd=taf,
            fees_usd=fees_usd,
            fees_bps=_q_bps(fees_usd / notional * BPS),
            quoted_spread_bps=_q_bps(quoted),
            spread_source=source,
            spread_cost_bps=_q_bps(spread_cost_bps),
            spread_cost_usd=_q_usd(spread_usd),
            participation=None if participation is None else participation.quantize(Decimal("1e-12")),
            slippage_fixed_bps=_q_bps(self.slippage.fixed_bps),
            impact_bps=_q_bps(impact),
            slippage_bps=_q_bps(slippage_bps),
            slippage_usd=_q_usd(slippage_usd),
            total_usd=_q_usd(total_usd),
            total_bps=_q_bps(total_usd / notional * BPS),
            notes=tuple(notes),
        )

    def estimate_round_trip(
        self,
        symbol: str,
        quantity: Decimal | int | str,
        price: Decimal | int | str,
        *,
        exit_price: Decimal | int | str | None = None,
        spread_bps: Decimal | int | str | None = None,
        adv_shares: Decimal | int | str | None = None,
    ) -> RoundTripEstimate:
        """Entry BUY at ``price`` + exit SELL at ``exit_price`` (default: same price). bps vs entry notional."""
        entry = self.estimate(symbol, Side.BUY, quantity, price, spread_bps=spread_bps, adv_shares=adv_shares)
        exit_leg = self.estimate(
            symbol,
            Side.SELL,
            quantity,
            price if exit_price is None else exit_price,
            spread_bps=spread_bps,
            adv_shares=adv_shares,
        )
        base = entry.notional

        def rel(usd: Decimal) -> Decimal:
            return _q_bps(usd / base * BPS)

        return RoundTripEstimate(
            entry=entry,
            exit=exit_leg,
            spread_bps=rel(entry.spread_cost_usd + exit_leg.spread_cost_usd),
            fees_bps=rel(entry.fees_usd + exit_leg.fees_usd),
            slippage_bps=rel(entry.slippage_usd + exit_leg.slippage_usd),
            total_bps=rel(entry.total_usd + exit_leg.total_usd),
            total_usd=_q_usd(entry.total_usd + exit_leg.total_usd),
        )

    def describe(self) -> dict[str, object]:
        """JSON-able description recorded in every experiment manifest."""
        return {
            "version": self.version,
            "fees": {
                "commission_per_share": str(self.fees.commission_per_share),
                "commission_per_order": str(self.fees.commission_per_order),
                "commission_min_per_order": str(self.fees.commission_min_per_order),
                "sec_fee_rate": str(self.fees.sec_fee_rate),
                "sec_fee_asof": self.fees.sec_fee_asof,
                "finra_taf_per_share": str(self.fees.finra_taf_per_share),
                "finra_taf_max_per_trade": str(self.fees.finra_taf_max_per_trade),
                "finra_taf_asof": self.fees.finra_taf_asof,
                "round_fees_up_to_cent": self.fees.round_fees_up_to_cent,
                "source": self.fees.source,
            },
            "spread": {
                "convention": "crossing cost = half of full quoted spread",
                "buckets": [
                    {"min_adv_usd": str(b.min_adv_usd), "full_spread_bps": str(b.spread_bps), "label": b.label}
                    for b in self.spread.buckets
                ],
                "unknown_liquidity_full_spread_bps": str(self.spread.unknown_liquidity_spread_bps),
            },
            "slippage": {
                "fixed_bps": str(self.slippage.fixed_bps),
                "impact": "impact_coef_bps * sqrt(quantity / adv_shares)",
                "impact_coef_bps": str(self.slippage.impact_coef_bps),
                "unknown_adv_impact_bps": str(self.slippage.unknown_adv_impact_bps),
                "max_participation": str(self.slippage.max_participation),
            },
            "caveats": [
                "Not calibrated against live fills.",
                "Alpaca paper fills omit market impact, latency slippage, queue position and fees.",
                "Regulatory fee defaults are as-of dated; verify current rates.",
            ],
        }


def spread_bps_from_quote(bid: Decimal | int | str, ask: Decimal | int | str) -> Decimal:
    """Full quoted spread in bps of the mid: ``(ask - bid) / mid * 10_000``. Crossed/empty quotes are rejected."""
    b, a = D(bid), D(ask)
    if b <= 0 or a <= 0:
        raise ValueError("bid and ask must be positive")
    if a < b:
        raise ValueError("crossed quote (ask < bid)")
    mid = (a + b) / 2
    return _q_bps((a - b) / mid * BPS)


def default_cost_model() -> CostModel:
    return CostModel()
