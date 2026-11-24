"""Unified cross-asset opportunity contract and incremental-portfolio-utility ranking.

Every candidate — a stock, an ETF, a leveraged ETF, a crypto pair or an option structure — is expressed per unit of
*capital committed* over its own holding period, with its own execution model already applied:

* expected gross / cost / net return (fractions of capital), forecast std and downside CVaR(95) over the horizon,
* capital required, buying-power effect, exposure multiple (delta-/leverage-adjusted notional per $ of capital),
  defined max loss (options/spreads), liquidity capacity, execution probability, expected slippage,
* evidence (validation status, DSR, walk-forward Sharpe, holdout result) and execution eligibility (capability
  layers + risk policy).

Ranking converts each candidate to per-day moments, shrinks the expected return by evidence quality, and computes
the mean-variance-CVaR utility increment of adding it to the current book at the policy-capped size:

    dU(w) = w*mu - (gamma/2) * (w^2 s^2 + 2 w rho s sigma_book w_book),   s = max(sigma, tail * |CVaR95| / 2.063)

(2.063 = CVaR95/sigma of a normal: fat-tailed payoffs such as short options get a larger effective volatility).

Raw percentage returns are never compared directly. The selector returns the executable candidate with the
largest positive dU, or ``NO_TRADE`` with the reason each candidate was not chosen.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(StrEnum):
    VALIDATED = "VALIDATED"  # passed development gate AND the one-use protected holdout
    SHADOW = "SHADOW"  # validated, accumulating prospective shadow evidence
    DEV_PASSED = "DEV_PASSED"  # passed the development gate, holdout not yet evaluated
    RESEARCH_ONLY = "RESEARCH_ONLY"  # informative scan / model output without a passing validation record
    REJECTED = "REJECTED"  # failed validation (kept for attribution)


EVIDENCE_SHRINK: dict[EvidenceStatus, float] = {
    EvidenceStatus.VALIDATED: 1.0,
    EvidenceStatus.SHADOW: 0.75,
    EvidenceStatus.DEV_PASSED: 0.25,
    EvidenceStatus.RESEARCH_ONLY: 0.0,
    EvidenceStatus.REJECTED: 0.0,
}
"""Multiplier on expected return: unvalidated forecasts carry no expected edge in the utility."""


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: EvidenceStatus
    family: str | None = None
    report: str | None = None
    dsr: float | None = None
    wf_oos_sharpe: float | None = None
    holdout_passed: bool | None = None
    trades: int | None = None
    notes: list[str] = Field(default_factory=list)


class Opportunity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opportunity_id: str
    asof: datetime
    asset_class: str  # us_equity | etf | leveraged_etf | crypto | us_option
    instrument: str
    legs: list[dict[str, Any]] = Field(default_factory=list)
    strategy_family: str
    direction: str  # LONG | SHORT | DEBIT_SPREAD | CREDIT_SPREAD | ...
    horizon_days: float = Field(gt=0)
    expected_gross_return: float
    expected_cost: float = Field(ge=0)
    forecast_std: float = Field(gt=0)
    downside_cvar95: float = Field(le=0)
    max_loss_fraction: float | None = None  # defined-risk: max loss / capital (<= 1)
    capital_required: float = Field(ge=0)
    buying_power_effect: float = Field(ge=0)
    exposure_multiple: float = 1.0
    liquidity_capacity_usd: float | None = None
    execution_probability: float = Field(default=1.0, ge=0, le=1)
    expected_slippage_bps: float = 0.0
    evidence: Evidence
    executable: bool = False
    execution_blockers: list[str] = Field(default_factory=list)
    correlation_to_book: float = 0.0
    notes: list[str] = Field(default_factory=list)

    @property
    def expected_net_return(self) -> float:
        return self.expected_gross_return - self.expected_cost


class RankedOpportunity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opportunity: Opportunity
    expected_net_return: float
    mu_daily_effective: float
    sigma_daily: float
    weight: float
    utility: float
    selectable: bool
    reasons: list[str]


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asof: datetime
    action: str  # TRADE | NO_TRADE
    selected: RankedOpportunity | None
    ranked: list[RankedOpportunity]
    reason: str
    params: dict[str, float]


NORMAL_CVAR95_PER_SIGMA = 2.063


def rank(opps: list[Opportunity], *, gamma: float = 4.0, tail: float = 1.0, max_weight: float = 0.25,
         book_sigma_daily: float = 0.0, book_weight: float = 0.0, min_utility: float = 0.0) -> list[RankedOpportunity]:
    out: list[RankedOpportunity] = []
    for o in opps:
        reasons: list[str] = []
        h = o.horizon_days
        shrink = EVIDENCE_SHRINK[o.evidence.status]
        net = o.expected_net_return * o.execution_probability - (1 - o.execution_probability) * 0.0
        mu = net / h * shrink
        sigma = max(o.forecast_std, tail * abs(o.downside_cvar95) / NORMAL_CVAR95_PER_SIGMA) / math.sqrt(h)
        cap = max_weight
        if o.max_loss_fraction is not None and o.max_loss_fraction > 0:
            cap = min(cap, max_weight / max(o.max_loss_fraction, 1e-9) * 0.25)  # max-loss budget: 25% of cap
        w_star = (mu - gamma * o.correlation_to_book * sigma * book_sigma_daily * book_weight) / \
            (gamma * sigma * sigma) if sigma > 0 else 0.0
        w = min(max(w_star, 0.0), cap)
        u = (w * mu - 0.5 * gamma * (w * w * sigma * sigma + 2 * w * o.correlation_to_book * sigma
                                     * book_sigma_daily * book_weight))
        if shrink == 0:
            reasons.append(f"evidence {o.evidence.status}: no validated edge (expected return shrunk to 0)")
        if net <= 0:
            reasons.append("expected net return <= 0 after costs")
        if not o.executable:
            reasons.append("not executable: " + "; ".join(o.execution_blockers or ["blocked"]))
        if w <= 0:
            reasons.append("optimal size is zero (edge does not cover risk/tail penalty)")
        selectable = o.executable and w > 0 and u > min_utility and shrink > 0
        out.append(RankedOpportunity(opportunity=o, expected_net_return=net, mu_daily_effective=mu,
                                     sigma_daily=sigma, weight=w, utility=u, selectable=selectable, reasons=reasons))
    out.sort(key=lambda r: (r.selectable, r.utility), reverse=True)
    return out


def select(opps: list[Opportunity], *, asof: datetime | None = None, **kw: float) -> Selection:
    ranked = rank(opps, **kw)
    best = next((r for r in ranked if r.selectable), None)
    now = asof or datetime.now(UTC)
    params = {k: float(v) for k, v in kw.items()}
    if best is None:
        why = "no candidate" if not ranked else "no executable candidate with positive evidence-weighted utility"
        return Selection(asof=now, action="NO_TRADE", selected=None, ranked=ranked, reason=why, params=params)
    return Selection(asof=now, action="TRADE", selected=best, ranked=ranked,
                     reason=f"max incremental utility {best.utility:.3g}/day at weight {best.weight:.3f}",
                     params=params)


def leveraged_etf_expected_return(underlying_mu_daily: float, underlying_sigma_daily: float, leverage: float,
                                  days: float, expense_ratio_annual: float = 0.0095) -> tuple[float, float]:
    """Expected horizon return and std of a daily-rebalanced L-times ETF (volatility drag included)."""
    drag = 0.5 * (leverage ** 2 - leverage) * underlying_sigma_daily ** 2
    mu = (leverage * underlying_mu_daily - drag - expense_ratio_annual / 252) * days
    sd = abs(leverage) * underlying_sigma_daily * math.sqrt(days)
    return mu, sd
