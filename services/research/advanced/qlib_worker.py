# pyright: reportMissingImports=false
"""Fit Microsoft's actual Qlib DoubleEnsemble in the existing isolated Qlib environment."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.contrib.model.double_ensemble import DEnsembleModel

root = Path(sys.argv[1])
config = json.loads((root/'config.json').read_text())
with np.load(root/'train.npz', allow_pickle=False) as a:
    X, y = a['X'], a['y']
np.random.seed(config['seed'])
result = []
for target in range(y.shape[1]):
    frame = pd.concat({'feature': pd.DataFrame(X), 'label': pd.DataFrame(y[:, target])}, axis=1)
    class Dataset:
        def __init__(self, training):
            self.training = training

        def prepare(self, segments, **kwargs):
            # The second frame is only a logging set. Early stopping is disabled; no outer calibration/test supplied.
            return [self.training, self.training]
    model = DEnsembleModel(num_models=3, epochs=config['rounds'], decay=1., num_threads=2,
                           verbosity=-1, seed=config['seed'], early_stopping_rounds=None)
    model.fit(Dataset(frame))
    members = []
    for k, booster in enumerate(model.ensemble):
        name = f'target-{target}-member-{k}.txt'
        booster.save_model(str(root/name))
        members.append({'file': name, 'columns': list(map(int, model.sub_features[k])),
                        'weight': float(model.sub_weights[k]/sum(model.sub_weights))})
    result.append(members)
(root/'ensemble.json').write_text(json.dumps(result))
