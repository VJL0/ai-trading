"""Strategy A — adaptive cross-sectional relative-strength momentum (stocks/ETFs) and ETF rotation.

Literature: Jegadeesh & Titman (1993) 3-12 month momentum; skip-month 12-1 (Fama-French UMD); residual momentum
(Blitz, Huij & Martens 2011) reduces crash exposure; volatility scaling (Barroso & Santa-Clara 2015) and bear-market
crash protection (Daniel & Moskowitz 2016); dual/absolute momentum rotation for asset-class ETFs (Antonacci 2014).
All of these are re-tested here after costs on Alpaca SIP data — none is assumed to work.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..backtest import apply_rebalance_schedule, hold_schedule
from ..panel import EquityPanel, shift, trailing_std
from . import features as F

FAMILY = "A_momentum"
ROTATION_FAMILY = "A2_etf_rotation"

ROTATION_ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD")
"""Asset-class/sector ETFs on the current execution allowlist (rotation is directly executable at 1 position)."""


def score(panel: EquityPanel, kind: str) -> np.ndarray:
    if kind == "mom_12_1":
        return F.momentum(panel, 252, 21)
    if kind == "mom_6_1":
        return F.momentum(panel, 126, 21)
    if kind == "mom_3":
        return F.momentum(panel, 63, 0)
    if kind == "resid_12_1":
        return F.residual_momentum(panel, 252, 21)
    if kind == "blend":
        u = np.isfinite(panel.f["ret_cc"])
        ranks = [F.cross_rank(F.momentum(panel, lb, 0), u) for lb in (21, 63, 126, 252)]
        return np.nanmean(np.stack(ranks), axis=0)
    raise ValueError(kind)


def regime_exposure(panel: EquityPanel, kind: str, target_vol: float = 0.15) -> np.ndarray:
    """Gross exposure multiplier per row (T,) decided at the close."""
    spy = panel.col("SPY")
    px = panel.f["adj_close"][:, spy]
    T = len(px)
    if kind == "none":
        return np.ones(T)
    if kind == "sma200":
        s = F.sma(panel, 200)[:, spy]
        return np.where(np.isfinite(s) & (px > s), 1.0, 0.0)
    if kind == "volscale":
        vol = trailing_std(panel.f["ret_cc"][:, [spy]], 21, 18)[:, 0] * np.sqrt(252)
        return np.clip(target_vol / np.where(vol > 0, vol, np.nan), 0.0, 1.0)
    raise ValueError(kind)


def top_n_weights(sc: np.ndarray, universe: np.ndarray, n: int, *, buffer: float = 2.0,
                  weighting: str = "equal", vol: np.ndarray | None = None) -> np.ndarray:
    """Hold the top ``n`` by score; a holding is kept while it stays inside the top ``n*buffer`` (turnover control)."""
    T, N = sc.shape
    out = np.zeros((T, N))
    held: set[int] = set()
    s = np.where(universe & np.isfinite(sc), sc, -np.inf)
    for t in range(T):
        row = s[t]
        valid = np.isfinite(row)
        if valid.sum() < n:
            held = set()
            continue
        order = np.argsort(-row)
        rank = np.empty(N, dtype=int)
        rank[order] = np.arange(N)
        keep = {j for j in held if valid[j] and rank[j] < int(n * buffer)}
        for j in order:
            if len(keep) >= n:
                break
            if valid[j]:
                keep.add(int(j))
        held = keep
        idx = np.fromiter(held, dtype=int)
        if weighting == "invvol" and vol is not None:
            iv = 1.0 / np.where(np.isfinite(vol[t, idx]) & (vol[t, idx] > 0), vol[t, idx], np.nan)
            fill = float(np.nanmean(iv)) if np.isfinite(iv).any() else 1.0
            iv = np.where(np.isfinite(iv), iv, fill)
            out[t, idx] = iv / iv.sum()
        else:
            out[t, idx] = 1.0 / len(idx)
    return out


def variants() -> Iterator[dict[str, Any]]:
    for uni, sc, n, reb, reg in itertools.product(("liquid1000", "large300"),
                                                  ("mom_12_1", "mom_6_1", "mom_3", "resid_12_1", "blend"),
                                                  (10, 30), (5, 21), ("none", "sma200", "volscale")):
        yield {"universe": uni, "score": sc, "n": n, "rebalance": reb, "regime": reg}


def universe_mask(panel: EquityPanel, name: str) -> np.ndarray:
    if name == "liquid1000":
        return panel.universe(min_price=5.0, min_adv=20e6, min_history=252, kinds=("COMMON_STOCK",), top_n=1000)
    if name == "large300":
        return panel.universe(min_price=5.0, min_adv=50e6, min_history=252, kinds=("COMMON_STOCK",), top_n=300)
    raise ValueError(name)


class ScoreCache:
    def __init__(self, panel: EquityPanel) -> None:
        self.panel = panel
        self._s: dict[str, np.ndarray] = {}
        self._u: dict[str, np.ndarray] = {}
        self._r: dict[str, np.ndarray] = {}

    def score(self, kind: str) -> np.ndarray:
        if kind not in self._s:
            self._s[kind] = score(self.panel, kind)
        return self._s[kind]

    def universe(self, name: str) -> np.ndarray:
        if name not in self._u:
            self._u[name] = universe_mask(self.panel, name)
        return self._u[name]

    def regime(self, kind: str) -> np.ndarray:
        if kind not in self._r:
            self._r[kind] = regime_exposure(self.panel, kind)
        return self._r[kind]


def targets(cache: ScoreCache, p: dict[str, Any]) -> np.ndarray:
    panel = cache.panel
    w = top_n_weights(cache.score(p["score"]), cache.universe(p["universe"]), int(p["n"]))
    w = w * cache.regime(p["regime"])[:, None]
    mask = hold_schedule(len(panel.dates), int(p["rebalance"]))
    # regime changes act immediately (risk reduction is not delayed to the next rebalance)
    reg = cache.regime(p["regime"])
    changed = np.r_[False, np.abs(np.diff(reg)) > 0.25]
    return apply_rebalance_schedule(w, mask | changed)


# ------------------------------------------------------------------------------------------- ETF rotation
def rotation_variants() -> Iterator[dict[str, Any]]:
    for lb, n, reb, absf in itertools.product(("blend", "mom_6_1", "mom_3", "mom_12_1"), (1, 2, 3), (5, 21),
                                              ("none", "positive", "sma200")):
        yield {"score": lb, "n": n, "rebalance": reb, "absolute": absf}


def rotation_targets(full: EquityPanel, p: dict[str, Any], etfs: tuple[str, ...] = ROTATION_ETFS) -> np.ndarray:
    panel, sub_cols = full.subset(etfs)
    return full.expand(_rotation_targets(panel, p), sub_cols)


def _rotation_targets(panel: EquityPanel, p: dict[str, Any]) -> np.ndarray:
    cols = list(range(panel.shape[1]))
    T, N = panel.shape
    uni = np.zeros((T, N), dtype=bool)
    uni[:, cols] = np.isfinite(panel.f["adj_close"][:, cols]) & (np.nan_to_num(panel.f["obs_count"][:, cols]) >= 252)
    if p["score"] == "blend":
        ranks = [F.momentum(panel, lb, 0) for lb in (21, 63, 126, 252)]
        sc = np.nanmean(np.stack([F.cross_rank(x, uni) for x in ranks]), axis=0)
        absolute = F.momentum(panel, 252, 0)
    else:
        sc = score(panel, p["score"])
        absolute = sc
    w = top_n_weights(sc, uni, int(p["n"]), buffer=1.0)
    if p["absolute"] == "positive":
        w = np.where(np.nan_to_num(absolute) > 0, w, 0.0)
    elif p["absolute"] == "sma200":
        s = F.sma(panel, 200)
        w = np.where(np.isfinite(s) & (panel.f["adj_close"] > s), w, 0.0)
    w = w / np.maximum(1.0, w.sum(axis=1, keepdims=True))
    return apply_rebalance_schedule(w, hold_schedule(T, int(p["rebalance"])))


def buy_and_hold(panel: EquityPanel, symbol: str) -> np.ndarray:
    T, N = panel.shape
    w = np.zeros((T, N))
    j = panel.col(symbol)
    w[:, j] = np.where(np.isfinite(panel.f["adj_close"][:, j]), 1.0, 0.0)
    return apply_rebalance_schedule(w, np.r_[True, np.diff(w[:, j]) != 0])


_ = shift  # re-export for strategy authors
