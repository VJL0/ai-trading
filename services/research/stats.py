"""Evaluation statistics for research experiments (floats: these are statistics, not accounting).

* ``spearman_ic`` — rank information coefficient between a score and realised forward returns.
* ``ic_pvalue`` — one-sided p-value for H0: IC <= 0 using the t approximation with an *effective*
  sample size (overlapping H-bar labels are serially dependent, so callers pass ``n / H``).
* ``benjamini_hochberg`` — multiple-testing control across experiment variants: controls the false
  discovery rate at level ``q`` (Benjamini & Hochberg, 1995) and reports adjusted q-values.
* ``brier_score`` / ``reliability_table`` / ``expected_calibration_error`` — probability calibration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats as _st


def spearman_ic(score: np.ndarray, ret: np.ndarray) -> float:
    """Spearman rank correlation; NaN when undefined (fewer than 3 points or a constant input)."""
    s = np.asarray(score, dtype=np.float64)
    r = np.asarray(ret, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(r)
    s, r = s[mask], r[mask]
    if len(s) < 3 or np.ptp(s) == 0 or np.ptp(r) == 0:
        return float("nan")
    rs = _st.rankdata(s)
    rr = _st.rankdata(r)
    return float(np.corrcoef(rs, rr)[0, 1])


def ic_pvalue(ic: float, n_eff: float) -> float:
    """One-sided p-value of H0: rho <= 0 vs H1: rho > 0 (t = r*sqrt((n-2)/(1-r^2)), df = n-2)."""
    if not math.isfinite(ic) or n_eff < 4:
        return 1.0
    r = min(max(ic, -0.999999), 0.999999)
    df = n_eff - 2.0
    t = r * math.sqrt(df / (1.0 - r * r))
    return float(_st.t.sf(t, df))


def benjamini_hochberg(pvalues: Mapping[str, float], q: float = 0.10) -> dict[str, dict[str, Any]]:
    """Benjamini-Hochberg step-up procedure over a family of hypotheses.

    Sort p ascending (p_(1) <= ... <= p_(m)); find the largest k with p_(k) <= k/m * q and reject the
    hypotheses ranked 1..k. Adjusted q-value: q_(i) = min_{j >= i} min(1, m/j * p_(j)). Non-finite p-values
    are treated as 1. Returns ``{name: {"p_value", "rank", "threshold", "q_value", "reject"}}``.
    """
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    names = sorted(pvalues, key=lambda k: (_finite_p(pvalues[k]), k))
    m = len(names)
    if m == 0:
        return {}
    ps = [_finite_p(pvalues[k]) for k in names]
    k_max = 0
    for i, p in enumerate(ps, start=1):
        if p <= i / m * q:
            k_max = i
    adjusted = [0.0] * m
    running = 1.0
    for i in range(m, 0, -1):
        running = min(running, min(1.0, ps[i - 1] * m / i))
        adjusted[i - 1] = running
    return {
        name: {
            "p_value": ps[i],
            "rank": i + 1,
            "threshold": (i + 1) / m * q,
            "q_value": adjusted[i],
            "reject": (i + 1) <= k_max,
        }
        for i, name in enumerate(names)
    }


def _finite_p(p: float) -> float:
    if not isinstance(p, int | float) or not math.isfinite(p):
        return 1.0
    return min(max(float(p), 0.0), 1.0)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    pa = np.asarray(p, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(pa) == 0:
        return float("nan")
    return float(np.mean((pa - ya) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    pa = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    ya = np.asarray(y, dtype=np.float64)
    if len(pa) == 0:
        return float("nan")
    return float(-np.mean(ya * np.log(pa) + (1 - ya) * np.log(1 - pa)))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width probability bins: count, mean predicted probability, observed frequency of y=1."""
    pa = np.asarray(p, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pa, edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": n,
                "mean_predicted": float(pa[mask].mean()) if n else None,
                "observed_frequency": float(ya[mask].mean()) if n else None,
            }
        )
    return rows


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    pa = np.asarray(p, dtype=np.float64)
    if len(pa) == 0:
        return float("nan")
    total = 0.0
    for row in reliability_table(pa, y, n_bins):
        if row["count"]:
            total += row["count"] * abs(row["mean_predicted"] - row["observed_frequency"])
    return total / len(pa)


def mean_tstat(x: np.ndarray, n_eff: float | None = None) -> tuple[float, float]:
    """(mean, t-statistic) with an optional effective sample size for dependent observations."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    sd = float(a.std(ddof=1))
    n = float(n_eff) if n_eff is not None else float(len(a))
    if sd == 0 or n < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(a.mean() / (sd / math.sqrt(n)))
