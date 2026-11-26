"""Economic mechanism catalog (mechanism-first research, TASK.md §2 and §5).

Every hypothesis task targets one mechanism. An experiment needs both a closed factor DSL primitive and
point-in-time development data. Mechanisms with missing inputs yield data requests for the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from services.market_data.edgar.store import EdgarStore


@dataclass(frozen=True)
class Mechanism:
    key: str
    title: str
    description: str
    expressible_now: bool
    dsl_functions: tuple[str, ...] = ()
    data_needed: str = ""
    requires_edgar: bool = False


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
              "Post-earnings-announcement drift: prices underreact to earnings announcements and surprises "
              "relative to a seasonal EPS baseline.",
              True, ("sue", "earnings_announcement", "earnings_window", "earnings_drift_age"),
              "point-in-time SEC 8-K item 2.02 events, periodic filings and quarterly XBRL earnings history", True),
    Mechanism("fundamentals", "Fundamentals (quality, value, investment)",
              "Profitability, quality, value, investment, accruals and free-cash-flow improvement priced slowly.",
              True, ("accruals", "asset_growth", "gross_profitability", "net_share_issuance"),
              "point-in-time XBRL statements with filing acceptance timestamps", True),
    Mechanism("insider", "Insider activity",
              "Open-market insider purchases, clusters and seniority carry private information.",
              True, ("insider_net_buy_ratio", "insider_cluster_buy"),
              "SEC Form 4 filings with public acceptance timestamps", True),
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
EXPRESSIBLE: Final[tuple[str, ...]] = tuple(m.key for m in CATALOG if m.expressible_now and not m.requires_edgar)


def _edgar_store(root: Path | None) -> EdgarStore | None:
    from services.market_data.edgar.store import EdgarStore, store_path

    if root is None:
        from .config import default_project_root

        root = default_project_root()
    store = EdgarStore(store_path(root))
    return store if store.path.is_file() else None


def _development_cutoff() -> datetime:
    from services.market_data.calendar import get_calendar
    from services.quantlab.gates import HOLDOUT_START

    cal = get_calendar()
    return cal.session_open(cal.previous_session(HOLDOUT_START))


def _edgar_coverage(root: Path | None) -> frozenset[str]:
    store = _edgar_store(root)
    return store.coverage(_development_cutoff()) if store is not None else frozenset()


def edgar_evidence(root: Path | None = None) -> str | None:
    """Development-period EDGAR content, so a data extension is new evidence even when coverage is unchanged."""
    store = _edgar_store(root)
    return store.digest(_development_cutoff()) if store is not None else None


def available_mechanisms(root: Path | None = None) -> tuple[str, ...]:
    coverage = _edgar_coverage(root)
    return tuple(m.key for m in CATALOG if m.expressible_now and (not m.requires_edgar or m.key in coverage))


def get(key: str, root: Path | None = None) -> Mechanism | None:
    m = BY_KEY.get(key)
    if m is not None and m.requires_edgar and key not in _edgar_coverage(root):
        return replace(m, expressible_now=False)
    return m


def is_expressible(key: str, root: Path | None = None) -> bool:
    m = get(key, root)
    return bool(m and m.expressible_now)
