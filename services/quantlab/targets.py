"""Resolve scheduled portfolio decisions without treating an explicit exit as a missing signal."""

from __future__ import annotations

import numpy as np


def latest_decision(targets: np.ndarray, t: int) -> tuple[int | None, np.ndarray]:
    """Return the last explicit decision at/before ``t`` (zero means cash; all-NaN means HOLD).

    These are desired weights, not drift-adjusted holdings. Actual prospective holdings are reconstructed by
    the execution simulator from timestamped decisions. A future row can never supply today's allocation.
    """
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or t < 0 or t >= len(values):
        raise ValueError("target matrix and row index are invalid")
    if np.isinf(values[:t + 1]).any():
        raise ValueError("infinite target weight")
    rows = np.flatnonzero(~np.isnan(values[:t + 1]).all(axis=1))
    if not len(rows):
        return None, np.zeros(values.shape[1])
    i = int(rows[-1])
    weights = np.nan_to_num(values[i], nan=0.0)
    if (weights < 0).any() or weights.sum() > 1.0 + 1e-9:
        raise ValueError("prospective targets must be long-only and unleveraged")
    return i, weights
