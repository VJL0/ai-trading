"""Vectorized execution-cost models per asset class (float bps; research only).

Equities reuse the runtime Decimal model's parameters (``services.research.costs``) so research and live
decisions charge the same spread buckets, slippage and regulatory fees. Crypto uses Alpaca's published tier-1
maker/taker schedule plus a spread/slippage allowance. Options use executable half-spreads plus per-contract
regulatory pass-through fees. All defaults are deliberately conservative and as-of dated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from services.research.costs import (
    DEFAULT_LIQUIDITY_BUCKETS,
    FINRA_TAF_PER_SHARE_DEFAULT,
    SEC_FEE_RATE_DEFAULT,
    SlippageModel,
)

_SLIP = SlippageModel()


@dataclass(frozen=True)
class EquityCosts:
    bucket_adv: tuple[float, ...] = tuple(float(b.min_adv_usd) for b in DEFAULT_LIQUIDITY_BUCKETS)
    bucket_spread_bps: tuple[float, ...] = tuple(float(b.spread_bps) for b in DEFAULT_LIQUIDITY_BUCKETS)
    unknown_spread_bps: float = 30.0
    slippage_fixed_bps: float = float(_SLIP.fixed_bps)
    impact_coef_bps: float = float(_SLIP.impact_coef_bps)
    sec_fee_rate: float = float(SEC_FEE_RATE_DEFAULT)
    finra_taf_per_share: float = float(FINRA_TAF_PER_SHARE_DEFAULT)
    open_spread_multiplier: float = 1.5  # opening prints are wider than the intraday average spread
    close_spread_multiplier: float = 1.0
    cost_multiplier: float = 1.0  # stress tests (e.g. 2.0)
    version: str = "quantlab-equity-costs-v1 (spread/slippage/fees from services.research.costs v1)"

    def full_spread_bps(self, adv_usd: np.ndarray) -> np.ndarray:
        adv = np.asarray(adv_usd, dtype=float)
        out = np.full(adv.shape, self.unknown_spread_bps)
        known = np.isfinite(adv)
        # buckets are ordered by min ADV descending; assign from the smallest threshold upward
        for thr, bps in sorted(zip(self.bucket_adv, self.bucket_spread_bps, strict=True)):
            out = np.where(known & (adv >= thr), bps, out)
        return out

    def one_way_bps(self, adv_usd: np.ndarray, trade_notional: np.ndarray, price: np.ndarray, *, sell: np.ndarray,
                    timing: str = "open") -> np.ndarray:
        """Cost of one leg in bps of traded notional (half spread + fixed slippage + impact + sell-side fees)."""
        mult = self.open_spread_multiplier if timing == "open" else self.close_spread_multiplier
        half = 0.5 * self.full_spread_bps(adv_usd) * mult
        adv = np.asarray(adv_usd, dtype=float)
        part = np.where(np.isfinite(adv) & (adv > 0), np.abs(trade_notional) / np.where(adv > 0, adv, 1.0), np.nan)
        impact = np.where(np.isfinite(part), self.impact_coef_bps * np.sqrt(np.clip(part, 0, None)), 10.0)
        px = np.where(np.asarray(price, dtype=float) > 0, price, np.nan)
        fees = np.where(sell, self.sec_fee_rate * 1e4 + np.nan_to_num(self.finra_taf_per_share / px * 1e4), 0.0)
        return (half + self.slippage_fixed_bps + impact + fees) * self.cost_multiplier


@dataclass(frozen=True)
class CryptoCosts:
    """Alpaca crypto spot, volume tier 1 (0-100k USD 30-day volume): 0.15% maker / 0.25% taker (docs, 2025-09)."""

    taker_fee_bps: float = 25.0
    maker_fee_bps: float = 15.0
    spread_bps: dict[str, float] = field(default_factory=dict)  # measured full spreads; default below
    default_spread_bps: float = 10.0
    slippage_bps: float = 2.0
    cost_multiplier: float = 1.0
    use_maker: bool = False
    asof: str = "Alpaca crypto fee schedule (tier 1), docs.alpaca.markets/docs/crypto-fees, retrieved 2026-11-23"

    def one_way_bps(self, pair: str) -> float:
        fee = self.maker_fee_bps if self.use_maker else self.taker_fee_bps
        spread = self.spread_bps.get(pair, self.default_spread_bps)
        half = 0.0 if self.use_maker else 0.5 * spread
        return (fee + half + self.slippage_bps) * self.cost_multiplier


@dataclass(frozen=True)
class OptionCosts:
    """Per-contract pass-through fees (VERIFY: rates change). Alpaca option commissions are $0."""

    commission_per_contract: float = 0.0
    orf_per_contract: float = 0.02685  # Options Regulatory Fee (charged on all contracts), as-of 2025 schedules
    occ_per_contract: float = 0.02  # OCC clearing fee pass-through (conservative)
    finra_taf_per_contract_sold: float = 0.00279
    sec_fee_rate_on_sells: float = float(SEC_FEE_RATE_DEFAULT)
    half_spread_fill_fraction: float = 1.0  # 1.0 = pay the full half-spread (cross to the far touch)
    asof: str = "as-of 2025 published ORF/OCC/TAF schedules; VERIFY CURRENT RATES before relying on them"

    def per_contract_fees(self, *, sell: bool, premium: float, multiplier: float = 100.0) -> float:
        fees = self.commission_per_contract + self.orf_per_contract + self.occ_per_contract
        if sell:
            fees += self.finra_taf_per_contract_sold + self.sec_fee_rate_on_sells * premium * multiplier
        return fees

    def leg_cost_usd(self, bid: float, ask: float, *, sell: bool, multiplier: float = 100.0) -> float:
        mid = 0.5 * (bid + ask)
        half = max(ask - bid, 0.0) * 0.5 * self.half_spread_fill_fraction
        return half * multiplier + self.per_contract_fees(sell=sell, premium=mid, multiplier=multiplier)
