"""Family J — volatility-managed index exposure (long-only, never levered).

Moreira & Muir (2017, JF) "Volatility-Managed Portfolios": scaling market exposure by the inverse of recent realized
variance raised Sharpe ratios historically; Cederburg, O'Doherty, Wang & Yan (2020, JFE) show the gains are fragile
out of sample once the scaling is estimated in real time. Here the scale is capped at 1 (no margin — the policy
forbids borrowing), so the strategy can only de-risk in turbulent periods, optionally combined with a 200-day trend
filter (Faber 2007). It is the simplest credible "defensive positioning" benchmark for the ETF families and is
re-tested after costs, not assumed to work.

Decision at the close of t from returns through t; trades at the next open (``execution="open"``). Small weight
changes inside ``band`` are not traded (turnover control).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..backtest import apply_rebalance_schedule
from ..panel import EquityPanel, trailing_mean, trailing_std

FAMILY = "J_vol_managed"


def variants() -> Iterator[dict[str, Any]]:
    for asset, tv, lb, trend, band in itertools.product(("SPY", "QQQ"), (0.10, 0.15, 0.20), (10, 21, 63),
                                                        ("none", "sma200"), (0.10, 0.25)):
        yield {"asset": asset, "target_vol": tv, "lookback": lb, "trend": trend, "band": band}


def targets(panel: EquityPanel, p: dict[str, Any]) -> np.ndarray:
    T, N = panel.shape
    j = panel.col(str(p["asset"]))
    r = panel.f["ret_cc"][:, [j]]
    lb = int(p["lookback"])
    vol = trailing_std(r, lb, max(5, int(lb * 0.8)))[:, 0] * np.sqrt(252)
    scale = np.clip(float(p["target_vol"]) / np.where(vol > 0, vol, np.nan), 0.0, 1.0)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    if p["trend"] == "sma200":
        px = panel.f["adj_close"][:, j]
        sma = trailing_mean(panel.f["adj_close"][:, [j]], 200, 180)[:, 0]
        scale = np.where(np.isfinite(sma) & (px > sma), scale, 0.0)
    band = float(p["band"])
    held = 0.0
    trade = np.zeros(T, dtype=bool)
    out = np.zeros((T, N))
    for t in range(T):
        want = float(scale[t])
        if t == 0 or abs(want - held) >= band or (want == 0.0) != (held == 0.0):
            held = want
            trade[t] = True
        out[t, j] = held
    return apply_rebalance_schedule(out, trade)
