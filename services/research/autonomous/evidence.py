"""Conservative research screening. Passing this screen is NOT deployment authorization."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def block_p_value(daily_bps: np.ndarray, seed: int = 42, draws: int = 4000) -> float:
    """One-sided centered circular moving-block bootstrap; resample whole five-session blocks.

    This is an approximate dependence-aware test of positive mean, not a proof of stationarity.
    Sparse samples fail closed. Reported uncertainty is limited by the bootstrap resolution.
    """
    values = np.asarray(daily_bps, dtype=float)
    if len(values) < 60 or not np.isfinite(values).all() or values.mean() <= 0:
        return 1.0
    centered = values - values.mean()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(draws, math.ceil(len(values) / 5)))
    ids = (starts[..., None] + np.arange(5)) % len(values)
    null = centered[ids.reshape(draws, -1)[:, :len(values)]].mean(axis=1)
    return float((1 + np.sum(null >= values.mean())) / (draws + 1))


def screen(result: dict[str, Any], total_trials: int) -> dict[str, Any]:
    """Bonferroni across all recorded trials, including failed trials, and both required tests."""
    alpha = 0.05 / (2 * max(1, total_trials))
    reasons = list(result.get('promotion_blockers', []))
    if result.get('is_simulated') is not False:
        reasons.append('synthetic or unknown data provenance')
    metrics = result.get('metrics') or {}
    if metrics.get('n_trades', 0) < 100:
        reasons.append('fewer than 100 development trades')
    if result.get('sessions', 0) < 120:
        reasons.append('fewer than 120 development sessions')
    for field in ('p_positive', 'p_incremental'):
        p = result.get(field)
        if p is None or not math.isfinite(p) or not 0 <= p <= alpha:
            reasons.append(f'{field} fails all-trial correction')
    if (result.get('stress_2x_costs') or {}).get('net_return', 0) <= 0:
        reasons.append('nonpositive return at double modeled costs')
    if sum(f.get('net_return', 0) > 0 for f in result.get('folds', [])) < 3:
        reasons.append('fewer than three profitable development folds')
    if (result.get('risk') or {}).get('max_drawdown', 1) > .15:
        reasons.append('development drawdown exceeds research threshold')
    return {'eligible_for_independent_validation': not reasons, 'reasons': reasons,
            'alpha_per_test': alpha, 'total_trials': total_trials, 'execution_authority': False}
