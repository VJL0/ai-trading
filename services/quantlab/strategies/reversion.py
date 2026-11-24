"""Strategy C — regime-conditioned short-term mean reversion (daily bars).

Families tested (all long-only executable forms; costs charged on every leg):
* C1 weekly-loser reversal (Lehmann 1990; Jegadeesh 1990) among liquid large caps, optionally only in high-volatility
  regimes (Nagel 2012: reversal profits are compensation for liquidity provision and rise with volatility).
* C2 short-horizon oversold pullback in an uptrend (RSI(2) < threshold while above the 200-day average; exit on a
  close above the 5-day average or after ``max_hold`` sessions) — a well-known practitioner rule, re-tested here.
* C3 the same pullback rule on index/sector ETFs only (directly executable under the 1-position policy).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..panel import EquityPanel, trailing_std
from . import features as F
from .momentum import ROTATION_ETFS

FAMILY_REVERSAL = "C1_weekly_reversal"
FAMILY_PULLBACK = "C2_pullback"
FAMILY_ETF_PULLBACK = "C3_etf_pullback"


def event_positions(entry: np.ndarray, exit_: np.ndarray, rank_score: np.ndarray, *, max_positions: int,
                    max_hold: int) -> np.ndarray:
    """Equal-weight (1/max_positions) positions: enter on ``entry`` (best ``rank_score`` first, lower = better),
    exit on ``exit_`` or after ``max_hold`` sessions. Decisions at the close of each row."""
    T, N = entry.shape
    w = np.zeros((T, N))
    held: dict[int, int] = {}
    for t in range(T):
        for j in list(held):
            held[j] += 1
            if exit_[t, j] or held[j] >= max_hold:
                del held[j]
        free = max_positions - len(held)
        if free > 0:
            cand = np.where(entry[t])[0]
            cand = [int(j) for j in cand if j not in held]
            if cand:
                order = np.argsort(rank_score[t, cand])
                for k in order[:free]:
                    held[cand[k]] = 0
        for j in held:
            w[t, j] = 1.0 / max_positions
    return w


def high_vol_regime(panel: EquityPanel, pct: float = 0.7, window: int = 21) -> np.ndarray:
    """SPY realized vol above its own expanding ``pct`` quantile (known at t)."""
    spy = panel.col("SPY")
    vol = trailing_std(panel.f["ret_cc"][:, [spy]], window, int(window * 0.8))[:, 0]
    import pandas as pd

    thr = pd.Series(vol).expanding(min_periods=252).quantile(pct).to_numpy()
    return np.nan_to_num(vol) > np.nan_to_num(thr, nan=np.inf)


def reversal_variants() -> Iterator[dict[str, Any]]:
    for lb, n, hold, reg in itertools.product((5, 10), (10, 20), (5, 10), ("none", "highvol", "uptrend")):
        yield {"lookback": lb, "n": n, "hold": hold, "regime": reg}


def reversal_targets(panel: EquityPanel, universe: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    ret = F.momentum(panel, int(p["lookback"]), 0)
    vol = F.realized_vol(panel, 63)
    z = ret / np.where(vol > 0, vol * np.sqrt(int(p["lookback"]) / 252), np.nan)
    ok = universe & np.isfinite(z)
    rank = F.cross_rank(np.where(ok, z, np.nan), ok)
    entry = ok & (rank <= 0.05)
    reg = np.ones(len(panel.dates), dtype=bool)
    if p["regime"] == "highvol":
        reg = high_vol_regime(panel)
    elif p["regime"] == "uptrend":
        spy = panel.col("SPY")
        s = F.sma(panel, 200)[:, spy]
        reg = np.isfinite(s) & (panel.f["adj_close"][:, spy] > s)
    entry &= reg[:, None]
    never = np.zeros_like(entry)
    return event_positions(entry, never, np.nan_to_num(z, nan=np.inf), max_positions=int(p["n"]),
                           max_hold=int(p["hold"]))


def pullback_variants() -> Iterator[dict[str, Any]]:
    for thr, n, hold, trend in itertools.product((5, 10, 20), (5, 10), (5, 10), ("sma200", "sma100")):
        yield {"rsi_below": thr, "n": n, "max_hold": hold, "trend": trend}


def pullback_targets(panel: EquityPanel, universe: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    r2 = F.rsi(panel, 2)
    trend_w = 200 if p["trend"] == "sma200" else 100
    s_tr = F.sma(panel, trend_w)
    s5 = F.sma(panel, 5)
    px = panel.f["adj_close"]
    entry = universe & np.isfinite(r2) & (r2 < float(p["rsi_below"])) & np.isfinite(s_tr) & (px > s_tr)
    exit_ = np.isfinite(s5) & (px > s5)
    return event_positions(entry, exit_, np.nan_to_num(r2, nan=np.inf), max_positions=int(p["n"]),
                           max_hold=int(p["max_hold"]))


def etf_pullback_variants() -> Iterator[dict[str, Any]]:
    for thr, hold, trend in itertools.product((5, 10, 20), (5, 10), ("sma200", "none")):
        yield {"rsi_below": thr, "n": 1, "max_hold": hold, "trend": trend}


def etf_universe(panel: EquityPanel, etfs: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")) -> np.ndarray:
    T, N = panel.shape
    u = np.zeros((T, N), dtype=bool)
    for s in etfs:
        if s in set(panel.symbols):
            j = panel.col(s)
            u[:, j] = np.isfinite(panel.f["adj_close"][:, j])
    return u


def etf_pullback_targets(full: EquityPanel, p: dict[str, Any]) -> np.ndarray:
    panel, cols = full.subset(("SPY", "QQQ", "IWM", "DIA"))
    return full.expand(_etf_pullback_targets(panel, p), cols)


def _etf_pullback_targets(panel: EquityPanel, p: dict[str, Any]) -> np.ndarray:
    u = etf_universe(panel)
    r2 = F.rsi(panel, 2)
    s200 = F.sma(panel, 200)
    s5 = F.sma(panel, 5)
    px = panel.f["adj_close"]
    entry = u & np.isfinite(r2) & (r2 < float(p["rsi_below"]))
    if p["trend"] == "sma200":
        entry &= np.isfinite(s200) & (px > s200)
    exit_ = np.isfinite(s5) & (px > s5)
    return event_positions(entry, exit_, np.nan_to_num(r2, nan=np.inf), max_positions=1,
                           max_hold=int(p["max_hold"]))


_ = ROTATION_ETFS
