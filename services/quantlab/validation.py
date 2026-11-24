"""Statistical validation: walk-forward selection, multiple-testing corrections, overfitting diagnostics.

* Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado 2012, 2014): the observed Sharpe is tested against
  the expected maximum Sharpe of ``n_trials`` unskilled strategies (using the variance of trial Sharpes), with the
  non-normality (skew/kurtosis) correction.
* Hansen (2005) Superior Predictive Ability test (studentized, consistent p-value) with the Politis-Romano
  stationary bootstrap — "does any variant beat the benchmark after accounting for the whole search?".
* Probability of Backtest Overfitting via combinatorially-symmetric cross-validation (Bailey et al. 2017).
* Walk-forward parameter selection: each fold uses the variant with the best *past* Sharpe (expanding window).
* Benjamini-Hochberg FDR (re-used from ``services.research.stats``).

All functions operate on daily return arrays and are deterministic given a seed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from services.research.stats import benjamini_hochberg

EULER_GAMMA = 0.5772156649015329
__all__ = ["benjamini_hochberg", "deflated_sharpe", "expected_max_sharpe", "pbo_cscv", "probabilistic_sharpe",
           "spa_test", "walk_forward_select"]


def _sr(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > ``sr_benchmark``) given the sample (per-period, not annualized)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return 0.0
    sr = _sr(r)
    g3 = float(sps.skew(r))
    g4 = float(sps.kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    return float(sps.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period SR among ``n_trials`` independent zero-skill trials with SR variance ``var_sr``."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(r: np.ndarray, n_trials: int, var_sr: float) -> dict[str, float]:
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return {"dsr": probabilistic_sharpe(r, sr0), "sr0_per_period": sr0, "sr_per_period": _sr(np.asarray(r)),
            "n_trials": float(n_trials), "var_sr": float(var_sr)}


# ------------------------------------------------------------------------------------------- bootstrap / SPA
def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(1.0, mean_block)
    idx = np.empty((n_boot, n), dtype=np.int64)
    starts = rng.integers(0, n, size=(n_boot, n))
    newblock = rng.random((n_boot, n)) < p
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(newblock[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def spa_test(variants: np.ndarray, benchmark: np.ndarray, *, n_boot: int = 1000, mean_block: float = 10.0,
             seed: int = 7) -> dict[str, float]:
    """Hansen SPA (consistent). ``variants`` (T, K) daily returns; H0: no variant outperforms ``benchmark``."""
    X = np.asarray(variants, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    b = np.asarray(benchmark, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(b)
    d = X[ok] - b[ok, None]  # performance differentials (higher = better)
    n, k = d.shape
    if n < 50:
        return {"p_value": 1.0, "n": float(n), "k": float(k), "stat": 0.0}
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    dbar = d.mean(axis=0)
    boot_means = np.stack([d[i].mean(axis=0) for i in idx])  # (B, k)
    omega = np.sqrt(n) * boot_means.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    stat = max(0.0, float(np.max(np.sqrt(n) * dbar / omega)))
    thresh = -np.sqrt(2 * np.log(np.log(n))) * omega / np.sqrt(n)
    mu_c = np.where(dbar >= thresh, dbar, 0.0)
    centered = np.sqrt(n) * (boot_means - mu_c) / omega  # Hansen: Z* = d*_b - g_c(dbar)
    boot_stat = np.maximum(0.0, centered.max(axis=1))
    p = float((boot_stat >= stat).mean())
    return {"p_value": p, "stat": stat, "n": float(n), "k": float(k), "best_variant": int(np.argmax(dbar)),
            "best_mean_diff_daily": float(dbar.max())}


# ------------------------------------------------------------------------------------------- PBO
def pbo_cscv(variants: np.ndarray, n_splits: int = 16) -> dict[str, float]:
    """Probability of backtest overfitting: share of CSCV splits whose in-sample best variant is below the
    out-of-sample median (logit <= 0)."""
    X = np.asarray(variants, dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    T, K = X.shape
    if K < 2 or n_splits * 10 > T:
        return {"pbo": float("nan"), "splits": 0.0}
    n_splits -= n_splits % 2
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > 2000:
        rng = np.random.default_rng(11)
        combos = [combos[i] for i in rng.choice(len(combos), 2000, replace=False)]
    for is_blocks in combos:
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_blocks])
        sr_is = np.array([_sr(X[is_idx, j]) for j in range(K)])
        sr_oos = np.array([_sr(X[oos_idx, j]) for j in range(K)])
        best = int(np.argmax(sr_is))
        rank = (sps.rankdata(sr_oos)[best]) / (K + 1)
        logits.append(math.log(rank / (1 - rank)))
    lg = np.array(logits)
    return {"pbo": float((lg <= 0).mean()), "splits": float(len(lg)), "median_logit": float(np.median(lg))}


# ------------------------------------------------------------------------------------------- walk-forward
@dataclass(frozen=True)
class WalkForwardResult:
    oos: pd.Series
    choices: list[dict[str, Any]]


def walk_forward_select(variants: Mapping[str, pd.Series], *, first_test_year: int, min_train_days: int = 504,
                        metric: str = "sharpe", period: str = "year") -> WalkForwardResult:
    """Expanding-window selection: in each test period (calendar year, or half-year with ``period='half'``) use
    the variant with the best Sharpe on all data before that period."""
    frame = pd.DataFrame(dict(variants)).sort_index()
    ix = pd.DatetimeIndex(frame.index)
    yr = np.asarray(ix.year) * 2 + (np.asarray(ix.month) > 6) if period == "half" else np.asarray(ix.year)
    start_key = first_test_year * 2 if period == "half" else first_test_year
    years = sorted({int(y) for y in yr if y >= start_key})
    cols = [str(c) for c in frame.columns]
    values = frame.to_numpy(dtype=float)
    pieces: list[pd.Series] = []
    choices: list[dict[str, Any]] = []
    for y in years:
        tr = yr < y
        te = yr == y
        if int(tr.sum()) < min_train_days or not te.any():
            continue
        scores = {c: _annual_sharpe(values[tr, k]) for k, c in enumerate(cols)}
        best = max(scores, key=lambda c: (scores[c], c))
        pieces.append(pd.Series(values[te, cols.index(best)], index=ix[te]))
        choices.append({"year": y, "selected": best, "train_sharpe": scores[best], "train_days": int(tr.sum())})
    oos = pd.Series(np.concatenate([p.to_numpy() for p in pieces]),
                    index=pd.DatetimeIndex(np.concatenate([p.index.to_numpy() for p in pieces]))) if pieces \
        else pd.Series(dtype=float)
    return WalkForwardResult(oos, choices)


def _annual_sharpe(r: np.ndarray) -> float:
    return _sr(r) * math.sqrt(252)


def alpha_beta(r: np.ndarray, bench: np.ndarray, periods_per_year: float = 252.0) -> dict[str, float]:
    """OLS of strategy on benchmark daily returns with Newey-West (5 lags) t-stat for the intercept."""
    ok = np.isfinite(r) & np.isfinite(bench)
    y, x = r[ok], bench[ok]
    if len(y) < 60:
        return {"alpha_annual": 0.0, "beta": 0.0, "alpha_t": 0.0}
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n = len(y)
    lags = 5
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    se_a = math.sqrt(max(cov[0, 0], 1e-18))
    return {"alpha_annual": float(coef[0] * periods_per_year), "beta": float(coef[1]),
            "alpha_t": float(coef[0] / se_a), "n": float(n)}


def trial_sharpe_variance(sharpes_per_period: Sequence[float]) -> float:
    s = np.asarray([x for x in sharpes_per_period if np.isfinite(x)], dtype=float)
    return float(s.var(ddof=1)) if len(s) > 1 else 0.0
