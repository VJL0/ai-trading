"""Score -> expected-edge calibration with honest uncertainty, fitted only on earlier out-of-sample predictions.

* :class:`BinnedEdge` maps a cross-sectional score percentile to the expected *excess* executable forward return of
  a name in that percentile bin, Fama-MacBeth style: per decision date the bin's mean realized excess return, then
  the mean over dates and its standard error over dates (dates, not names, are the independent observations —
  same-day returns are cross-sectionally correlated). With overlapping daily decisions on an ``h``-session label
  the effective sample is ``n_dates / h``. Bin means are smoothed by weighted isotonic regression (monotone
  non-decreasing in the score), which pools bins that violate the ranking and is conservative when the score has
  no or negative skill.
* :class:`ConformalBands` adds Mondrian (per-bin) split-conformal intervals for a single name's realized excess
  return. They answer a different question — how wide is the uncertainty of ONE position — and are reported with
  their realized coverage on later data, because time-series dependence can break the exchangeability that
  conformal guarantees rely on.

Both are refitted per test period on predictions made before it; nothing here sees the period it is applied to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _bin_index(pct: np.ndarray, n_bins: int) -> np.ndarray:
    p = np.asarray(pct, dtype=float)
    b = np.floor(np.where(np.isfinite(p), p, 0.0) * n_bins - 1e-12).astype(np.int64)
    return np.where(np.isfinite(p), np.clip(b, 0, n_bins - 1), -1)


def _isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators (non-decreasing); NaN entries keep NaN and are skipped."""
    ok = np.isfinite(y) & (w > 0)
    out = np.full(len(y), np.nan)
    if not ok.any():
        return out
    vals = list(y[ok].astype(float))
    wts = list(w[ok].astype(float))
    sizes = [1] * len(vals)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            tw = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tw
            wts[i] = tw
            sizes[i] += sizes[i + 1]
            del vals[i + 1], wts[i + 1], sizes[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.repeat(np.asarray(vals), sizes)
    out[np.where(ok)[0]] = fitted
    return out


@dataclass(frozen=True)
class BinnedEdge:
    n_bins: int
    mean: np.ndarray  # smoothed expected excess return per bin (fraction over the label horizon)
    raw_mean: np.ndarray
    se: np.ndarray
    n_dates: np.ndarray
    overlap: float
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = _bin_index(pct, self.n_bins)
        ok = b >= 0
        bi = np.where(ok, b, 0)
        mu = np.where(ok, self.mean[bi], np.nan)
        se = np.where(ok, self.se[bi], np.nan)
        return mu, se

    def lower_bound(self, pct: np.ndarray, z: float) -> np.ndarray:
        mu, se = self.predict(pct)
        return mu - z * se

    def summary(self, scale: float = 1e4) -> dict[str, Any]:
        return {"n_bins": self.n_bins, "overlap": self.overlap,
                "mean_bps": [None if not math.isfinite(x) else round(float(x) * scale, 3) for x in self.mean],
                "raw_mean_bps": [None if not math.isfinite(x) else round(float(x) * scale, 3) for x in self.raw_mean],
                "se_bps": [None if not math.isfinite(x) else round(float(x) * scale, 3) for x in self.se],
                "n_dates": [int(x) for x in self.n_dates], **self.meta}


def fit_binned_edge(pct: np.ndarray, realized: np.ndarray, row_id: np.ndarray, *, n_bins: int = 20,
                    overlap: float = 1.0, min_dates: int = 5, isotonic: bool = True) -> BinnedEdge:
    """Per-bin Fama-MacBeth mean and standard error of ``realized`` (one value per observation) grouped by
    ``row_id`` (decision date). ``overlap``: label length / decision spacing (effective dates = n / overlap)."""
    p = np.asarray(pct, dtype=float)
    y = np.asarray(realized, dtype=float)
    rid = np.asarray(row_id)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y, rid = p[ok], y[ok], rid[ok]
    b = _bin_index(p, n_bins)
    rows, inv = np.unique(rid, return_inverse=True)
    R = len(rows)
    sums = np.zeros((R, n_bins))
    cnts = np.zeros((R, n_bins))
    np.add.at(sums, (inv, b), y)
    np.add.at(cnts, (inv, b), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_date = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    have = np.isfinite(per_date)
    n_dates = have.sum(axis=0)
    raw = np.full(n_bins, np.nan)
    se = np.full(n_bins, np.nan)
    for k in range(n_bins):
        v = per_date[have[:, k], k]
        if len(v) >= min_dates:
            raw[k] = float(v.mean())
            n_eff = max(1.0, len(v) / max(1.0, overlap))
            se[k] = float(v.std(ddof=1) / math.sqrt(n_eff)) if len(v) > 1 else float("nan")
    w = np.where(np.isfinite(se) & (se > 0), 1.0 / np.where(se > 0, se, 1.0) ** 2, 0.0)
    smooth = _isotonic(raw, w) if isotonic else raw.copy()
    return BinnedEdge(n_bins, smooth, raw, se, n_dates.astype(np.int64), float(overlap),
                      {"observations": len(y), "dates": R})


@dataclass(frozen=True)
class ConformalBands:
    n_bins: int
    alpha: float
    lo: np.ndarray  # residual quantile per bin (added to the point forecast)
    hi: np.ndarray
    n: np.ndarray

    def interval(self, pct: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = _bin_index(pct, self.n_bins)
        ok = b >= 0
        bi = np.where(ok, b, 0)
        return (np.where(ok, point + self.lo[bi], np.nan), np.where(ok, point + self.hi[bi], np.nan))


def _conformal_quantile(e: np.ndarray, q: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1) q)-th smallest residual (inf when n is too small)."""
    n = len(e)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * q)
    if k > n:
        return float("inf")
    if k < 1:
        return float("-inf")
    return float(np.sort(e)[k - 1])


def fit_conformal(pct: np.ndarray, residual: np.ndarray, *, n_bins: int = 20, alpha: float = 0.2,
                  min_obs: int = 30) -> ConformalBands:
    """Two-sided Mondrian split-conformal bands at level 1 - ``alpha`` from calibration residuals."""
    p = np.asarray(pct, dtype=float)
    e = np.asarray(residual, dtype=float)
    ok = np.isfinite(p) & np.isfinite(e)
    p, e = p[ok], e[ok]
    b = _bin_index(p, n_bins)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    n = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        ek = e[b == k]
        n[k] = len(ek)
        if len(ek) >= min_obs:
            # lower bound: the floor((n+1) a/2)-th smallest; upper: the ceil((n+1)(1 - a/2))-th smallest
            lo[k] = -_conformal_quantile(-ek, 1 - alpha / 2)
            hi[k] = _conformal_quantile(ek, 1 - alpha / 2)
    return ConformalBands(n_bins, alpha, lo, hi, n)


def coverage(lo: np.ndarray, hi: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(realized)
    if not ok.any():
        return {"n": 0.0, "coverage": float("nan"), "median_width": float("nan")}
    inside = (realized[ok] >= lo[ok]) & (realized[ok] <= hi[ok])
    return {"n": float(ok.sum()), "coverage": float(inside.mean()), "median_width": float(np.median(hi[ok] - lo[ok]))}
