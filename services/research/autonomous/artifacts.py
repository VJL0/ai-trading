"""Allowlisted inference formats: JSON, numeric NPZ and native boosters. Never unpickle.

No class names, Python modules or callables are imported from an artifact. A promotion binds
its complete file digest in the validation database, not merely in a mutable manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .learners import RegimeExperts, TorchLearner, make_transformer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_model(model: Any, directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    counter = 0

    from ..advanced.calibration import CalibratedTargets
    from ..advanced.discovery import GeneticFactors
    from ..advanced.tabular import DoubleEnsemble, MeanEnsemble, TabMLearner
    from ..advanced.temporal import TemporalLearner

    def encode(m: Any) -> dict[str, Any]:
        nonlocal counter
        from sklearn.decomposition import PCA
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.linear_model import BayesianRidge, MultiTaskElasticNet, Ridge, SGDRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        counter += 1
        stem = f'model_{counter}'
        if isinstance(m, CalibratedTargets):
            return {'kind': 'calibrated', 'model': encode(m.model),
                    'maps': {str(c): {'x': iso.X_thresholds_.tolist(), 'y': iso.y_thresholds_.tolist()}
                             for c, iso in m.calibrators.items()}}
        if isinstance(m, MeanEnsemble):
            return {'kind': 'mean_ensemble', 'members': [encode(member) for member in m.members]}
        if isinstance(m, GeneticFactors):
            return {'kind': 'symbolic', 'trees': m.trees, 'scaler': encode(m.scaler), 'head': encode(m.head)}
        if isinstance(m, DoubleEnsemble):
            return {'kind': 'column_ensemble', 'targets': [[{'columns': columns, 'weight': weight, 'model': encode(model)}
                    for columns, weight, model in members] for members in m.targets]}
        if isinstance(m, TemporalLearner):
            path = directory / f'{stem}.npz'
            np.savez(path, **{k: v.detach().cpu().numpy() for k, v in m.model.state_dict().items()})
            return {'kind': 'temporal', 'family': m.family, 'width': m.width, 'shape': list(m.shape),
                    'mean': m.mean.tolist(), 'scale': m.scale.tolist(), 'ymean': m.ymean.tolist(),
                    'yscale': m.yscale.tolist(), 'file': path.name}
        if isinstance(m, TabMLearner):
            path = directory / f'{stem}.npz'
            np.savez(path, **{k: v.detach().cpu().numpy() for k, v in m.model.state_dict().items()})
            return {'kind': 'tabm', 'width': m.width, 'xscale': encode(m.xscale),
                    'yscale': encode(m.yscale), 'file': path.name}
        if isinstance(m, Pipeline):
            return {'kind': 'pipeline', 'steps': [encode(step) for _, step in m.steps]}
        if isinstance(m, StandardScaler):
            return {'kind': 'scale', 'mean': np.asarray(m.mean_).tolist(), 'scale': np.asarray(m.scale_).tolist()}
        if isinstance(m, PCA):
            return {'kind': 'pca', 'mean': np.asarray(m.mean_).tolist(), 'components': np.asarray(m.components_).tolist()}
        if isinstance(m, Ridge | BayesianRidge | SGDRegressor | MultiTaskElasticNet):
            return {'kind': 'linear', 'coef': np.asarray(m.coef_).tolist(), 'intercept': np.asarray(m.intercept_).tolist()}
        if isinstance(m, MultiOutputRegressor):
            return {'kind': 'multi', 'models': [encode(e) for e in m.estimators_]}
        if isinstance(m, ExtraTreesRegressor):
            return {'kind': 'forest', 'trees': [{'left': e.tree_.children_left.tolist(),
                    'right': e.tree_.children_right.tolist(), 'feature': e.tree_.feature.tolist(),
                    'threshold': e.tree_.threshold.tolist(), 'value': e.tree_.value[:, :, 0].tolist()}
                    for e in m.estimators_]}
        if isinstance(m, RegimeExperts):
            return {'kind': 'regime', 'scaler': encode(m.scaler), 'means': np.asarray(m.gmm.means_).tolist(),
                    'covariances': np.asarray(m.gmm.covariances_).tolist(), 'weights': np.asarray(m.gmm.weights_).tolist(),
                    'experts': [encode(e) for e in m.experts]}
        if isinstance(m, TorchLearner):
            path = directory / f'{stem}.npz'
            np.savez(path, **{k: v.detach().cpu().numpy() for k, v in m.model.state_dict().items()})
            return {'kind': 'torch', 'family': m.spec.family, 'width': m.spec.capacity,
                    'xscale': encode(m.xscale), 'yscale': encode(m.yscale), 'file': path.name}
        # Only explicitly supported native APIs. No dynamic imports from serialized names.
        if type(m).__module__.startswith('lightgbm.'):
            path = directory / f'{stem}.lgb.txt'
            getattr(m, 'booster_', m).save_model(str(path))
            return {'kind': 'lightgbm', 'file': path.name}
        if type(m).__module__.startswith('xgboost.'):
            path = directory / f'{stem}.json'
            m.save_model(path)
            return {'kind': 'xgboost', 'file': path.name}
        if type(m).__module__.startswith('catboost.'):
            path = directory / f'{stem}.cbm'
            m.save_model(str(path))
            return {'kind': 'catboost', 'file': path.name}
        raise ValueError(f'unsupported artifact type {type(m).__name__}')

    (directory / 'model.json').write_text(json.dumps(encode(model), allow_nan=False))
    return {p.name: digest(p) for p in directory.iterdir() if p.is_file()}


class PortableModel:
    def __init__(self, directory: Path, files: dict[str, str]) -> None:
        if 'model.json' not in files:
            raise ValueError('missing model definition digest')
        for name, expected in files.items():
            if Path(name).name != name or (directory / name).is_symlink() or digest(directory / name) != expected:
                raise ValueError('artifact integrity mismatch')
        self.directory, self.files = directory, files
        self.definition = json.loads((directory / 'model.json').read_text())
        self.loaded: dict[str, Any] = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = np.asarray(self._apply(self.definition, np.asarray(X, dtype=float)), dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('non-finite artifact prediction')
        return result

    def _apply(self, d: dict[str, Any], X: np.ndarray) -> np.ndarray:
        kind = d['kind']
        if kind == 'calibrated':
            pred = self._apply(d['model'], X).copy()
            for col, mapping in d['maps'].items():
                pred[:, int(col)] = np.interp(pred[:, int(col)], mapping['x'], mapping['y'])
            if pred.shape[1] >= 8:
                pred[:, 5:8] = np.clip(pred[:, 5:8], 0, 1)
            return pred
        if kind == 'mean_ensemble':
            return np.mean([self._apply(member, X) for member in d['members']], axis=0)
        if kind == 'symbolic':
            from ..advanced.discovery import expression
            z = self._apply(d['scaler'], X)
            return self._apply(d['head'], np.column_stack([z, *[expression(t, z) for t in d['trees']]]))
        if kind == 'column_ensemble':
            return np.column_stack([sum(m['weight']*self._apply(m['model'], X[:, m['columns']]) for m in members)
                                    for members in d['targets']])
        if kind == 'pipeline':
            for step in d['steps']:
                X = self._apply(step, X)
            return X
        if kind == 'scale':
            return (X - np.asarray(d['mean'])) / np.asarray(d['scale'])
        if kind == 'pca':
            return (X - np.asarray(d['mean'])) @ np.asarray(d['components']).T
        if kind == 'linear':
            return X @ np.asarray(d['coef']).T + np.asarray(d['intercept'])
        if kind == 'multi':
            return np.column_stack([self._apply(m, X) for m in d['models']])
        if kind == 'forest':
            predictions = []
            # sklearn's tree predictor casts inputs to float32 before comparison.
            z = X.astype(np.float32)
            for tree in d['trees']:
                left, right, feature = (np.asarray(tree[k], dtype=int) for k in ('left', 'right', 'feature'))
                threshold, value = np.asarray(tree['threshold']), np.asarray(tree['value'])
                node = np.zeros(len(X), dtype=int)
                for _ in range(len(left)):
                    active = np.flatnonzero(left[node] != -1)
                    if not len(active):
                        break
                    n = node[active]
                    node[active] = np.where(z[active, feature[n]] <= threshold[n], left[n], right[n])
                else:
                    raise ValueError('cyclic tree artifact')
                predictions.append(value[node])
            return np.mean(predictions, axis=0)
        if kind == 'regime':
            z = self._apply(d['scaler'], X)
            means, cov = np.asarray(d['means']), np.asarray(d['covariances'])
            logp = -.5 * (np.log(cov).sum(axis=1) + ((z[:, None] - means)**2 / cov).sum(axis=2))
            logp += np.log(np.asarray(d['weights']))
            weights = np.exp(logp - logp.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            return np.sum([self._apply(e, z) * weights[:, k, None] for k, e in enumerate(d['experts'])], axis=0)
        name = d.get('file')
        if not isinstance(name, str) or name not in self.files:
            raise ValueError('unverified native artifact reference')
        if name not in self.loaded:
            path = self.directory / name
            if path.is_symlink() or digest(path) != self.files[name]:
                raise ValueError('native artifact changed before loading')
            if kind == 'lightgbm':
                import lightgbm as lgb
                self.loaded[name] = lgb.Booster(model_file=str(path))
            elif kind == 'xgboost':
                from xgboost import XGBRegressor
                self.loaded[name] = XGBRegressor(n_jobs=1)
                self.loaded[name].load_model(path)
            elif kind == 'catboost':
                from catboost import CatBoostRegressor
                self.loaded[name] = CatBoostRegressor(thread_count=1)
                self.loaded[name].load_model(str(path))
            elif kind == 'temporal':
                import torch

                from ..advanced.temporal import network
                net = network(d['family'], d['shape'][0], d['shape'][1], len(d['ymean']), d['width'])
                with np.load(path, allow_pickle=False) as weights:
                    net.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files}, strict=True)
                self.loaded[name] = net.eval()
            elif kind == 'tabm':
                import tabm
                import torch
                net = tabm.TabM.make(n_num_features=len(d['xscale']['mean']), d_out=len(d['yscale']['mean']),
                                    d_block=d['width'], n_blocks=2, k=8, dropout=.1)
                with np.load(path, allow_pickle=False) as weights:
                    net.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files}, strict=True)
                self.loaded[name] = net.eval()
            elif kind == 'torch':
                import torch
                from torch import nn
                inputs, outputs, width = len(d['xscale']['mean']), len(d['yscale']['mean']), int(d['width'])
                if d['family'] == 'torch_transformer':
                    net = make_transformer(inputs, outputs, width)
                elif d['family'] in ('torch_mlp', 'torch_masked'):
                    net = nn.Sequential(nn.Sequential(nn.Linear(inputs, width), nn.GELU(),
                                        nn.Linear(width, width), nn.GELU()), nn.Linear(width, outputs))
                else:
                    raise ValueError('unregistered neural architecture')
                with np.load(path, allow_pickle=False) as weights:
                    net.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files}, strict=True)
                self.loaded[name] = net.eval()
            else:
                raise ValueError(f'unknown artifact kind {kind}')
        model = self.loaded[name]
        if kind == 'temporal':
            import torch
            if X.ndim != 3 or tuple(X.shape[1:]) != tuple(d['shape']):
                raise ValueError('temporal artifact input shape mismatch')
            with torch.inference_mode():
                z = torch.tensor((X-np.asarray(d['mean']))/np.asarray(d['scale']), dtype=torch.float32)
                result = np.concatenate([model(b).numpy() for b in z.split(64)])
            return result*np.asarray(d['yscale'])+np.asarray(d['ymean'])
        if kind in ('torch', 'tabm'):
            import torch
            with torch.no_grad():
                z = torch.tensor(self._apply(d['xscale'], X), dtype=torch.float32)
                result = np.concatenate([(model(b).mean(dim=1) if kind == 'tabm' else model(b)).numpy() for b in z.split(512)])
            return result * np.asarray(d['yscale']['scale']) + np.asarray(d['yscale']['mean'])
        return np.asarray(model.predict(X))
