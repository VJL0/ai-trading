"""Point-in-time cross-sectional features on an :class:`EquityPanel` (all trailing, all computed at the close)."""

from __future__ import annotations

import numpy as np

from ..panel import EquityPanel, compound, shift, trailing_mean, trailing_std


def momentum(panel: EquityPanel, lookback: int, skip: int = 0) -> np.ndarray:
    """Compounded adjusted return from t-lookback to t-skip (classic 12-1 momentum: lookback=252, skip=21)."""
    r = panel.f["ret_cc"]
    m = compound(r, lookback - skip, min_periods=int((lookback - skip) * 0.8))
    return shift(m, skip) if skip else m


def realized_vol(panel: EquityPanel, window: int = 63) -> np.ndarray:
    return trailing_std(panel.f["ret_cc"], window, min_periods=int(window * 0.8)) * np.sqrt(252)


def market_return(panel: EquityPanel, symbol: str = "SPY") -> np.ndarray:
    return panel.f["ret_cc"][:, panel.col(symbol)]


def rolling_beta(panel: EquityPanel, window: int = 252, symbol: str = "SPY") -> np.ndarray:
    r = panel.f["ret_cc"]
    m = market_return(panel, symbol)[:, None]
    rz = np.where(np.isfinite(r), r, np.nan)
    mm = np.where(np.isfinite(rz), m, np.nan)
    cov = trailing_mean(rz * mm, window, int(window * 0.8)) - trailing_mean(rz, window, int(window * 0.8)) * \
        trailing_mean(mm, window, int(window * 0.8))
    var = trailing_mean(mm * mm, window, int(window * 0.8)) - trailing_mean(mm, window, int(window * 0.8)) ** 2
    return cov / np.where(var > 0, var, np.nan)


def residual_momentum(panel: EquityPanel, lookback: int = 252, skip: int = 21, beta_window: int = 252) -> np.ndarray:
    """Momentum of market-beta-adjusted returns, scaled by residual volatility (Blitz, Huij & Martens 2011)."""
    r = panel.f["ret_cc"]
    m = market_return(panel)[:, None]
    beta = shift(rolling_beta(panel, beta_window), 1)
    resid = r - beta * m
    s = pd_sum(resid, lookback - skip)
    sd = trailing_std(resid, lookback - skip, int((lookback - skip) * 0.8))
    score = s / np.where(sd > 0, sd, np.nan)
    return shift(score, skip) if skip else score


def pd_sum(a: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    return np.asarray(pd.DataFrame(a).rolling(window, min_periods=int(window * 0.8)).sum(), dtype=float)


def sma(panel: EquityPanel, window: int, field: str = "adj_close") -> np.ndarray:
    return trailing_mean(panel.f[field], window, int(window * 0.9))


def rsi(panel: EquityPanel, window: int = 2) -> np.ndarray:
    r = panel.f["ret_cc"]
    up = np.where(r > 0, r, 0.0)
    dn = np.where(r < 0, -r, 0.0)
    up = np.where(np.isfinite(r), up, np.nan)
    dn = np.where(np.isfinite(r), dn, np.nan)
    au = _ewm(up, window)
    ad = _ewm(dn, window)
    rs = au / np.where(ad > 0, ad, np.nan)
    out = 100 - 100 / (1 + rs)
    return np.where((ad == 0) & (au > 0), 100.0, out)


def _ewm(a: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    return np.asarray(pd.DataFrame(a).ewm(alpha=1.0 / window, adjust=False, ignore_na=True).mean(), dtype=float)


def high_52w_ratio(panel: EquityPanel) -> np.ndarray:
    import pandas as pd

    hi = np.asarray(pd.DataFrame(panel.f["adj_close"]).rolling(252, min_periods=200).max(), dtype=float)
    return panel.f["adj_close"] / hi


def cross_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] among ``mask`` names per row; NaN elsewhere."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    import pandas as pd

    return np.asarray(pd.DataFrame(v).rank(axis=1, pct=True), dtype=float)
