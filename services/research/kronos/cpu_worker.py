"""CPU comparator subprocess. Do not import torch: macOS wheels may carry incompatible OpenMP runtimes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..autonomous.learners import Hypothesis, fit
from ..models.estimators import ZeroModel, fit_lgbm_regressor, fit_ridge
from ..models.features import VolSpec
from ..models.vol import fit_har


def run(directory: Path) -> None:
    meta = json.loads((directory / 'input.json').read_text())
    with np.load(directory / 'input.npz', allow_pickle=False) as archive:
        a = dict(archive)
    output = {}
    for family in ('lightgbm', 'catboost', 'ridge'):
        model = fit(Hypothesis(family, seed=meta['seed']), a['X_train'], a['y_train'])
        output[family] = model.predict(a['X_predict'])
    if meta['registry'] is not None:
        reg = meta['registry']
        values = []
        for column, horizon in enumerate((5, 15, 60)):
            mid = reg['selected'][f'ret_{horizon}m']
            cfg = reg['config']
            if mid == 'zero_baseline':
                model = ZeroModel()
            elif mid == f'ridge_{horizon}':
                model = fit_ridge(mid, a['X_train'], a['y_train'][:, column], cfg['ridge_alpha'])
            elif mid == f'lgbm_reg_{horizon}':
                model = fit_lgbm_regressor(mid, a['X_train'], a['y_train'][:, column], cfg['lgbm_params'], cfg['seed'])
            else:
                raise ValueError(f'unsupported selected return model {mid}; comparison cannot be substituted')
            values.append(model.predict(a['X_predict']))
        output['selected_registry'] = np.column_stack(values)
    har = fit_har(a['vol_train'], a['rv_train'], VolSpec(**meta['vol_spec']))
    output['har_rv'] = har.forecast(a['vol_test'])
    np.savez_compressed(directory / 'output.npz', **output)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
