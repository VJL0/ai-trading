"""Causal, dimensionless OHLCV histories shared by training and paper inference."""
from __future__ import annotations

import numpy as np


def normalize_history(values: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(values[-length:], dtype=np.float32).copy()
    if len(x) != length or x.shape[1] != 6 or not np.isfinite(x).all() or np.any(x[:, :4] <= 0):
        raise ValueError('complete finite positive-price history required')
    # All normalizers use this decision's observed history, never future observations.
    x[:, :4] = np.log(x[:, :4]/x[-1, 3])
    x[:, 4:] = np.log1p(np.maximum(x[:, 4:], 0))
    x[:, 4:] -= x[:, 4:].mean(axis=0)
    return x


def model_inputs(spec, arrays, indices):
    from ..autonomous.learners import transform
    if not spec.family.startswith('temporal_'):
        return transform(arrays['X'][indices], spec.transform)
    output = []
    for row in indices:
        symbol = int(arrays['symbol_id'][row])
        stop = np.searchsorted(arrays[f'avail_{symbol}'], arrays['decision'][row], side='right')
        output.append(normalize_history(arrays[f'bars_{symbol}'][:stop], spec.sequence_length))
    return np.asarray(output, dtype=np.float32)
