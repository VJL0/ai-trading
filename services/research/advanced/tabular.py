"""Native DoubleEnsemble and trained TabM adapters."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class DoubleEnsemble:
    def __init__(self, rounds=32, seed=42):
        self.rounds, self.seed = rounds, seed

    def fit(self, X, y):
        import lightgbm as lgb

        from ..qlib_bridge import QLIB_PYTHON
        with tempfile.TemporaryDirectory(prefix='lat-doubleensemble-') as temporary:
            root = Path(temporary)
            np.savez(root/'train.npz', X=X, y=y)
            (root/'config.json').write_text(json.dumps({'rounds': self.rounds, 'seed': self.seed}))
            env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG')}
            env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            worker = Path(__file__).with_name('qlib_worker.py')
            done = subprocess.run([str(QLIB_PYTHON), str(worker), str(root)], env=env,
                                  capture_output=True, text=True, timeout=600, check=False)
            if done.returncode:
                raise RuntimeError(f'Qlib DoubleEnsemble failed: {done.stderr[-2000:]}')
            definitions = json.loads((root/'ensemble.json').read_text())
            self.targets = [[(m['columns'], m['weight'], lgb.Booster(model_file=str(root/m['file'])))
                             for m in members] for members in definitions]
        return self

    def predict(self, X):
        return np.column_stack([sum(weight*model.predict(X[:, columns]) for columns, weight, model in members)
                                for members in self.targets])


class TabMLearner:
    def __init__(self, width=64, epochs=8, seed=42):
        self.width, self.epochs, self.seed = width, epochs, seed

    def fit(self, X, y):
        import tabm
        import torch
        from sklearn.preprocessing import StandardScaler
        torch.manual_seed(self.seed)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.xscale, self.yscale = StandardScaler().fit(X), StandardScaler().fit(y)
        self.model = tabm.TabM.make(n_num_features=X.shape[1], d_out=y.shape[1], d_block=self.width,
                                    n_blocks=2, k=8, dropout=.1).to(self.device)
        x = torch.tensor(self.xscale.transform(X), dtype=torch.float32)
        target = torch.tensor(self.yscale.transform(y), dtype=torch.float32)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.model.train()
        for _ in range(self.epochs):
            for indices in torch.randperm(len(x)).split(128):
                pred = self.model(x[indices].to(self.device))
                loss = ((pred-target[indices, None].to(self.device))**2).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
        self.model.eval()
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.inference_mode():
            x = torch.tensor(self.xscale.transform(X), dtype=torch.float32)
            result = np.concatenate([self.model(b.to(self.device)).mean(dim=1).cpu().numpy() for b in x.split(128)])
        return self.yscale.inverse_transform(result)


class MeanEnsemble:
    """Equal-weight diversity baseline; weights are a predeclared hypothesis, never fitted on test outcomes."""
    def __init__(self, rounds=32, seed=42):
        self.rounds, self.seed = rounds, seed

    def fit(self, X, y):
        from ..autonomous.learners import Hypothesis, fit
        self.members = [fit(Hypothesis(name, capacity=self.rounds, seed=self.seed), X, y)
                        for name in ('ridge', 'lightgbm', 'catboost')]
        return self

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.members], axis=0)
