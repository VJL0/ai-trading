"""Cross-sectional ranking primitives: executable forward labels, relevance grades, rank IC and decile spreads.

Every array is (T, N) on the daily panel grid. A decision at the close of ``t`` executes at the open of ``t+1``
(the backtester's convention), so the *executable* forward return over ``h`` sessions runs from the open of
``t+1`` to the open of ``t+1+h``:

    sum_{d=t+1}^{t+h} log(1 + ret_oc[d])  +  sum_{d=t+2}^{t+h+1} log(1 + ret_co[d])

The close-to-close label used by family G includes the overnight gap into ``t+1``'s open, which no order placed
after the close of ``t`` can earn. Labels need at least 80 % of their legs (a name that stops trading inside the
window has a NaN label; the portfolio backtest still books its delisting through the backtester).

Diagnostics are computed on decision rows sampled every ``h`` sessions so consecutive labels do not overlap and
plain t-statistics apply (no overlap correction needed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .panel import trailing_sum


def executable_forward_return(f: dict[str, np.ndarray], horizon: int) -> np.ndarray:
    """Open(t+1) -> open(t+1+h) adjusted return per decision row ``t`` (NaN where unknown)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    lr_oc = np.log1p(np.where(np.isfinite(f["ret_oc"]), f["ret_oc"], np.nan))
    lr_co = np.log1p(np.where(np.isfinite(f["ret_co"]), f["ret_co"], np.nan))
    need = max(1, math.floor(horizon * 0.8))
    s_oc = trailing_sum(lr_oc, horizon, need)  # row r sums r-h+1..r
    s_co = trailing_sum(lr_co, horizon, need)
    out = np.full(lr_oc.shape, np.nan)
    T = lr_oc.shape[0]
    if horizon + 1 < T:
        out[: T - horizon - 1] = s_oc[horizon: T - 1] + s_co[horizon + 1:]  # oc rows t+1..t+h, co rows t+2..t+h+1
    return np.expm1(out)


def group_demean(x: np.ndarray, mask: np.ndarray, groups: np.ndarray | None = None,
                 min_group: int = 5) -> np.ndarray:
    """``x`` minus the mean of eligible names per row — within ``groups`` (int ids, -1 = none) when the group has at
    least ``min_group`` eligible names with a finite value, else the row mean. NaN outside ``mask``."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    fin = np.isfinite(v)
    cnt_all = fin.sum(axis=1, keepdims=True)
    row_mean = np.where(cnt_all > 0, np.where(fin, v, 0.0).sum(axis=1, keepdims=True) / np.maximum(cnt_all, 1), np.nan)
    out = v - row_mean
    if groups is None:
        return out
    g = np.where(np.isfinite(v), groups, -1).astype(np.int64)
    ids = [int(i) for i in np.unique(g) if i >= 0]
    for gid in ids:
        sel = g == gid
        cnt = sel.sum(axis=1)
        s = np.where(sel, v, 0.0).sum(axis=1)
        mean = np.where(cnt >= min_group, s / np.maximum(cnt, 1), np.nan)
        rows = np.where(np.isfinite(mean))[0]
        if len(rows):
            sub = sel[rows]
            out[rows] = np.where(sub, v[rows] - mean[rows, None], out[rows])
    return out


def pct_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] among eligible finite names per row (ties averaged); NaN elsewhere."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    return np.asarray(pd.DataFrame(v).rank(axis=1, pct=True), dtype=float)


def relevance_grades(label: np.ndarray, mask: np.ndarray, levels: int = 10) -> np.ndarray:
    """Integer relevance 0..levels-1 from the per-row percentile of ``label`` (-1 where not gradable)."""
    if levels < 2:
        raise ValueError("levels must be >= 2")
    p = pct_rank(label, mask)
    g = np.floor(np.where(np.isfinite(p), p, 0.0) * levels - 1e-12)
    return np.where(np.isfinite(p), np.clip(g, 0, levels - 1), -1).astype(np.int32)


def decision_rows(dates: pd.DatetimeIndex, horizon: int, start: Any = None,
                  end_row: int | None = None) -> np.ndarray:
    """Non-overlapping decision rows (every ``horizon`` sessions from ``start``) whose label is fully known by the
    last panel row (``end_row`` exclusive bound defaults to T - horizon - 1)."""
    T = len(dates)
    stop = (T - horizon - 1) if end_row is None else min(end_row, T - horizon - 1)
    first = 0 if start is None else int(np.searchsorted(np.asarray(dates), np.datetime64(str(start)[:10], "D")))
    return np.arange(first, max(first, stop), max(1, horizon))


def _spearman_rows(a: np.ndarray, b: np.ndarray, min_names: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        ok = np.isfinite(a[i]) & np.isfinite(b[i])
        if ok.sum() < min_names:
            continue
        ra = pd.Series(a[i, ok]).rank().to_numpy()
        rb = pd.Series(b[i, ok]).rank().to_numpy()
        sa, sb = ra.std(), rb.std()
        if sa > 0 and sb > 0:
            out[i] = float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))
    return out


def rank_ic(score: np.ndarray, label: np.ndarray, mask: np.ndarray, rows: np.ndarray,
            min_names: int = 30) -> np.ndarray:
    """Spearman correlation between score and realized label across eligible names, per decision row."""
    s = np.where(mask[rows], score[rows], np.nan)
    y = np.where(mask[rows], label[rows], np.nan)
    return _spearman_rows(s, y, min_names)


def quantile_means(score: np.ndarray, label: np.ndarray, mask: np.ndarray, rows: np.ndarray, q: int = 10,
                   min_names: int = 50) -> np.ndarray:
    """Mean label per score quantile (bottom .. top) per decision row: (len(rows), q); NaN rows when thin."""
    out = np.full((len(rows), q), np.nan)
    for i, r in enumerate(rows):
        ok = mask[r] & np.isfinite(score[r]) & np.isfinite(label[r])
        if ok.sum() < min_names:
            continue
        s, y = score[r, ok], label[r, ok]
        order = np.argsort(s, kind="stable")
        for k, chunk in enumerate(np.array_split(order, q)):
            out[i, k] = float(y[chunk].mean())
    return out


def t_stat(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0.0
    sd = v.std(ddof=1)
    return float(v.mean() / (sd / math.sqrt(len(v)))) if sd > 0 else 0.0


def ic_summary(ic: np.ndarray, dates: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> dict[str, Any]:
    v = np.asarray(ic, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    years = np.asarray(pd.DatetimeIndex(dates).year)
    yearly = {int(y): float(np.nanmean(v[(years == y) & ok])) for y in np.unique(years[ok])}
    sd = float(v[ok].std(ddof=1))
    return {"n": int(ok.sum()), "mean": float(v[ok].mean()), "std": sd,
            "ir_per_decision": float(v[ok].mean() / sd) if sd > 0 else 0.0, "t": t_stat(v),
            "hit_rate": float((v[ok] > 0).mean()), "yearly": yearly,
            "pct_positive_years": float(np.mean([x > 0 for x in yearly.values()])) if yearly else None}


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean and t-statistic of ``a - b`` over rows where both are finite (paired by decision date)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    ok = np.isfinite(d)
    if ok.sum() < 3:
        return {"n": float(ok.sum()), "mean_diff": 0.0, "t": 0.0}
    return {"n": float(ok.sum()), "mean_diff": float(d[ok].mean()), "t": t_stat(d[ok])}


def spread_summary(qmeans: np.ndarray, dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
                   scale: float = 1e4) -> dict[str, Any]:
    """Top-minus-bottom quantile spread and monotonicity of the average quantile profile (labels x ``scale``)."""
    ok = np.isfinite(qmeans).all(axis=1)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    qm = qmeans[ok] * scale
    spread = qm[:, -1] - qm[:, 0]
    profile = qm.mean(axis=0)
    mono = float(pd.Series(profile).rank().corr(pd.Series(np.arange(len(profile), dtype=float))))
    years = np.asarray(pd.DatetimeIndex(dates).year)[ok]
    return {"n": int(ok.sum()), "top_minus_bottom_mean": float(spread.mean()), "top_minus_bottom_t": t_stat(spread),
            "top_mean": float(qm[:, -1].mean()), "top_t": t_stat(qm[:, -1]), "bottom_mean": float(qm[:, 0].mean()),
            "profile": [float(x) for x in profile], "monotonicity": mono,
            "top_yearly": {int(y): float(qm[years == y, -1].mean()) for y in np.unique(years)}}


def top_set_turnover(score: np.ndarray, mask: np.ndarray, rows: np.ndarray, n: int) -> float:
    """Average fraction of the top-``n`` set replaced between consecutive decision rows."""
    prev: set[int] | None = None
    changes = []
    for r in rows:
        s = np.where(mask[r] & np.isfinite(score[r]), score[r], -np.inf)
        if np.isfinite(s).sum() < n:
            prev = None
            continue
        top = set(np.argsort(-s, kind="stable")[:n].tolist())
        if prev is not None:
            changes.append(len(top - prev) / n)
        prev = top
    return float(np.mean(changes)) if changes else float("nan")
