"""Economic mechanism catalog (mechanism-first research, TASK.md §2 and §5).

Every hypothesis task targets one mechanism. Only mechanisms that the closed factor DSL
(``services.quantlab.factors``: daily price and volume features only) can express may yield factor specs.
The others can only yield ``data_requests`` for the operator: the crew refuses any spec for them, because a
price-only proxy of, say, insider buying tests a different hypothesis than the one stated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Mechanism:
    key: str
    title: str
    description: str
    expressible_now: bool
    dsl_functions: tuple[str, ...] = ()
    data_needed: str = ""


CATALOG: Final[tuple[Mechanism, ...]] = (
    Mechanism("momentum", "Momentum / underreaction",
              "Slow diffusion of information and herding: past winners keep outperforming for weeks to months "
              "(price, residual and 52-week-high momentum, sector leadership via relative strength).",
              True, ("mom", "resid_mom", "high52", "dist_sma")),
    Mechanism("reversal", "Short-term reversal / overreaction",
              "Liquidity provision and overreaction: short-horizon losers rebound as temporary price pressure "
              "fades (1-week to 1-month reversal, stretched distance from a moving average, RSI extremes).",
              True, ("mom", "rsi", "dist_sma")),
    Mechanism("attention", "Attention and volume shocks",
              "Investor attention: abnormal trading volume raises visibility and subsequent demand (high-volume "
              "return premium); attention-driven overpricing later reverses. Daily volume is the only proxy here.",
              True, ("relvol", "mom", "high52"),
              "news counts, search trends and social attention for the non-volume variants"),
    Mechanism("volatility_risk", "Volatility and risk anomalies",
              "Leverage constraints and lottery preferences: low-volatility and low-beta stocks earn more per unit "
              "of risk than high-volatility ones; volatility regimes change how other signals behave.",
              True, ("vol", "beta", "rank", "zscore")),
    Mechanism("earnings", "Earnings surprises and drift",
              "Post-earnings-announcement drift: prices underreact to earnings, revenue and guidance surprises.",
              False, data_needed="point-in-time earnings dates, consensus estimates, actuals and guidance"),
    Mechanism("fundamentals", "Fundamentals (quality, value, investment)",
              "Profitability, quality, value, investment, accruals and free-cash-flow improvement priced slowly.",
              False, data_needed="point-in-time financial statements with filing timestamps"),
    Mechanism("insider", "Insider activity",
              "Open-market insider purchases, clusters and seniority carry private information.",
              False, data_needed="SEC Form 4 filings with public acceptance timestamps"),
    Mechanism("analyst", "Analyst information",
              "Estimate and recommendation revisions, target changes and forecast dispersion.",
              False, data_needed="point-in-time analyst estimate and recommendation history"),
    Mechanism("market_structure", "Market structure / microstructure",
              "Order imbalance, spreads, adverse selection, liquidity withdrawal, opening and closing pressure.",
              False, data_needed="intraday quotes, trades and auction imbalance data"),
    Mechanism("etf_mechanics", "ETF mechanics and index demand",
              "ETF flows, creations/redemptions and index reconstitution create non-fundamental price pressure.",
              False, data_needed="ETF flows / shares outstanding history and index change announcements"),
    Mechanism("positioning", "Positioning",
              "Short interest, borrow costs, options-implied expectations, institutional holdings changes.",
              False, data_needed="short interest, borrow fees, options surfaces, 13F holdings"),
    Mechanism("relationships", "Economic relationships",
              "Sector lead-lag, supplier/customer links, ETF-constituent and pairs relationships.",
              False, data_needed="industry classification history, supply-chain links, constituent weights"),
    Mechanism("macro", "Macroeconomic sensitivity",
              "Rates, inflation and employment surprises and their differential effect across stocks.",
              False, data_needed="point-in-time macro release calendars, surprises and Treasury yields"),
    Mechanism("calendar", "Calendar effects",
              "Month-end flows, earnings-season patterns, index rebalance dates, holiday and announcement timing.",
              False, data_needed="date features in the factor language and an event calendar"),
)

BY_KEY: Final[dict[str, Mechanism]] = {m.key: m for m in CATALOG}
KEYS: Final[tuple[str, ...]] = tuple(m.key for m in CATALOG)
EXPRESSIBLE: Final[tuple[str, ...]] = tuple(m.key for m in CATALOG if m.expressible_now)


def get(key: str) -> Mechanism | None:
    return BY_KEY.get(key)


def is_expressible(key: str) -> bool:
    m = BY_KEY.get(key)
    return bool(m and m.expressible_now)
