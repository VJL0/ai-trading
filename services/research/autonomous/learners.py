"""Extensible local learner adapters. All preprocessing is fitted inside each training fold.

The research contract is multi-output prediction in target units; column zero is net return.
Optional dependencies are discovered, never installed or downloaded by the worker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Hypothesis:
    family: str
    seed: int = 42
    capacity: int = 64
    alpha: float = 10.0
    transform: str = "identity"
    diagnostic_rows: int = 0
    diagnostic_test_rows: int = 0
    sequence_length: int = 64
    epochs: int = 8
    pretrain_epochs: int = 0
    meta_steps: int = 0
    foundation_artifact_hash: str = ""
    kronos_context: int = 128
    kronos_samples: int = 8
    kronos_rows_per_fold: int = 64
    kronos_tune_steps: int = 0
    kronos_artifact_hash: str = ""
    memory_mode: str = ""

    def describe(self) -> dict[str, Any]:
        return asdict(self)


FAMILIES = {
    **{f"foundation_{name}": f"Local probabilistic {name} feature challenger; research-only"
       for name in ("fincast", "timestral", "t0_gamma", "simons", "chronos2", "tirex2", "timesfm25", "fintext")},
    **{name: f"Local {name} tabular foundation challenger; research-only" for name in ("tabiclv2", "kumo", "mitra")},
    "ensemble_mean": "Equal-weight Ridge/LightGBM/CatBoost ensemble, independently validated as one candidate",
    "symbolic_gp": "Training-only genetic symbolic factor discovery with regularized multi-task heads",
    **{f"temporal_{name}": f"Trainable OHLCV {name} multi-task sequence model"
       for name in ("patchtst", "itransformer", "moderntcn", "multitask_transformer")},
    "kronos_base": "Local Kronos-base 102.3M sequence challenger (research only)",
    "kronos_small": "Optional local Kronos-small 24.7M efficiency baseline (research only)",
    "ridge": "Regularized linear multi-task baseline",
    "double_ensemble": "Qlib DoubleEnsemble with learning-curve sample reweighting and feature selection",
    "tabm": "Trained parameter-efficient TabM ensemble for joint return and risk targets",
    "elastic_net": "Sparse regularized multi-task return/volatility/cost regression",
    "bayesian": "Bayesian return regression with posterior uncertainty",
    "extra_trees": "Nonlinear multi-task randomized ensemble",
    "lightgbm": "Gradient-boosted cost-aware return regression",
    "catboost": "Ordered boosting return regression",
    "xgboost": "Histogram boosting return regression",
    "online": "Chronological incremental SGD with adaptive learning rate",
    "regime": "Training-only Gaussian mixture with regime-specific ridge experts",
    "pca": "Self-supervised linear representation with multi-task ridge head",
    "torch_mlp": "Joint return, volatility and cost neural representation",
    "torch_masked": "Masked feature pretraining followed by joint supervised fine-tuning",
    "torch_transformer": "Feature-token transformer with joint regression heads",
}


def available() -> dict[str, dict[str, Any]]:
    deps = {"tabm": "tabm", "lightgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost"}
    from ..kronos.adapter import DEFAULT_ROOT
    result = {}
    for k, v in FAMILIES.items():
        dep = "torch" if k.startswith(("torch_", "kronos_", "temporal_")) else deps.get(k, "sklearn")
        ready = find_spec(dep) is not None
        if k.startswith('foundation_') or k in ('tabiclv2', 'kumo', 'mitra'):
            from ..advanced.foundation import PYTHON, WEIGHTS
            ready = PYTHON.is_file() and (WEIGHTS/k.removeprefix('foundation_')/'manifest.json').is_file()
        if k == 'ensemble_mean':
            ready = all(find_spec(dep) is not None for dep in ('lightgbm', 'catboost'))
        if k == 'double_ensemble':
            from ..qlib_bridge import QLIB_PYTHON
            ready = QLIB_PYTHON.is_file()
        if k.startswith('kronos_'):
            ready = ready and (DEFAULT_ROOT / k.removeprefix('kronos_') / 'model.safetensors').is_file()
            ready = ready and (DEFAULT_ROOT / 'manifest.json').is_file() and find_spec('safetensors') is not None
        result[k] = {"description": v, "available": ready}
    return result


class RegimeExperts:
    def fit(self, X: np.ndarray, y: np.ndarray) -> RegimeExperts:
        from sklearn.linear_model import Ridge
        from sklearn.mixture import GaussianMixture
        self.scaler = StandardScaler().fit(X)
        z = self.scaler.transform(X)
        self.gmm = GaussianMixture(3, covariance_type="diag", random_state=42).fit(z)
        weights = self.gmm.predict_proba(z)
        self.experts = [Ridge(alpha=10).fit(z, y, sample_weight=weights[:, k]) for k in range(3)]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(X)
        weights = self.gmm.predict_proba(z)
        return np.sum([m.predict(z) * weights[:, k, None] for k, m in enumerate(self.experts)], axis=0)


class TorchLearner:
    def __init__(self, spec: Hypothesis) -> None:
        self.spec = spec

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchLearner:
        import torch
        from torch import nn
        torch.manual_seed(self.spec.seed)
        torch.set_num_threads(2)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.xscale, self.yscale = StandardScaler().fit(X), StandardScaler().fit(y)
        x = torch.tensor(self.xscale.transform(X), dtype=torch.float32)
        target = torch.tensor(self.yscale.transform(y), dtype=torch.float32)
        width = self.spec.capacity
        if self.spec.family == "torch_transformer":
            self.model = make_transformer(X.shape[1], y.shape[1], width).to(self.device)
        else:
            encoder = nn.Sequential(nn.Linear(X.shape[1], width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            if self.spec.family == "torch_masked":
                autoencoder = nn.Sequential(encoder, nn.Linear(width, X.shape[1])).to(self.device)
                opt = torch.optim.AdamW(autoencoder.parameters(), lr=0.001)
                for _ in range(5):
                    for batch in x.split(512):
                        batch = batch.to(self.device)
                        masked = batch * (torch.rand_like(batch) > 0.25)
                        opt.zero_grad()
                        loss = nn.functional.mse_loss(autoencoder(masked), batch)
                        loss.backward()
                        opt.step()
            self.model = nn.Sequential(encoder, nn.Linear(width, y.shape[1])).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.model.train()
        for _ in range(20):
            for ids in torch.randperm(len(x)).split(512):
                optimizer.zero_grad()
                loss = nn.functional.huber_loss(self.model(x[ids].to(self.device)), target[ids].to(self.device))
                if not torch.isfinite(loss):
                    raise ValueError("non-finite neural training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        x = torch.tensor(self.xscale.transform(X), dtype=torch.float32)
        with torch.no_grad():
            result = np.concatenate([self.model(b.to(self.device)).cpu().numpy() for b in x.split(512)])
        return self.yscale.inverse_transform(result)


def fit(spec: Hypothesis, X: np.ndarray, y: np.ndarray) -> Any:
    if spec.family not in FAMILIES:
        raise ValueError(f"unregistered learner {spec.family}")
    if not available()[spec.family]["available"]:
        raise ImportError(f"optional dependency unavailable for {spec.family}")
    if spec.family.startswith('foundation_') or spec.family in ('tabiclv2', 'kumo', 'mitra'):
        raise ValueError('foundation challengers require the isolated paired evaluator')
    if spec.family.startswith('kronos_'):
        raise ValueError('Kronos requires the point-in-time sequence evaluator, not tabular fit')
    from sklearn.linear_model import BayesianRidge, MultiTaskElasticNet, Ridge, SGDRegressor
    from sklearn.multioutput import MultiOutputRegressor
    if spec.family.startswith('temporal_'):
        from ..advanced.temporal import TemporalLearner
        return TemporalLearner(spec.family.removeprefix('temporal_'), width=spec.capacity, epochs=spec.epochs,
                               seed=spec.seed, pretrain_epochs=spec.pretrain_epochs, meta_steps=spec.meta_steps).fit(X, y)
    if spec.family == 'ensemble_mean':
        from ..advanced.tabular import MeanEnsemble
        model = MeanEnsemble(rounds=spec.capacity, seed=spec.seed)
    elif spec.family == 'symbolic_gp':
        from ..advanced.discovery import GeneticFactors
        model = GeneticFactors(population=spec.capacity, seed=spec.seed)
    elif spec.family == 'double_ensemble':
        from ..advanced.tabular import DoubleEnsemble
        model = DoubleEnsemble(rounds=spec.capacity, seed=spec.seed)
    elif spec.family == 'tabm':
        from ..advanced.tabular import TabMLearner
        model = TabMLearner(width=spec.capacity, seed=spec.seed)
    elif spec.family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=spec.alpha))
    elif spec.family == "elastic_net":
        model = make_pipeline(StandardScaler(), MultiTaskElasticNet(alpha=spec.alpha, l1_ratio=.5, max_iter=5000,
                                                                    random_state=spec.seed))
    elif spec.family == "bayesian":
        model = make_pipeline(StandardScaler(), MultiOutputRegressor(BayesianRidge()))
    elif spec.family == "online":
        model = make_pipeline(StandardScaler(), MultiOutputRegressor(
            SGDRegressor(shuffle=False, random_state=spec.seed, alpha=0.001, max_iter=500)))
    elif spec.family == "pca":
        from sklearn.decomposition import PCA
        model = make_pipeline(StandardScaler(), PCA(n_components=0.95), Ridge(alpha=spec.alpha))
    elif spec.family == "regime":
        model = RegimeExperts()
    elif spec.family == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=spec.capacity, min_samples_leaf=20, n_jobs=2, random_state=spec.seed)
    elif spec.family == "lightgbm":
        from lightgbm import LGBMRegressor
        model = MultiOutputRegressor(LGBMRegressor(n_estimators=spec.capacity, num_leaves=15,
                                                   verbosity=-1, n_jobs=2, random_state=spec.seed))
    elif spec.family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=spec.capacity, depth=5, loss_function="MultiRMSE", thread_count=2,
                                  random_seed=spec.seed, verbose=False, allow_writing_files=False)
    elif spec.family == "xgboost":
        from xgboost import XGBRegressor
        model = MultiOutputRegressor(XGBRegressor(n_estimators=spec.capacity, max_depth=4, tree_method="hist",
                                                  n_jobs=2, random_state=spec.seed))
    else:
        model = TorchLearner(spec)
    return model.fit(X, y)


def transform(X: np.ndarray, name: str) -> np.ndarray:
    """Bounded symbolic feature grammar; no evaluation of generated Python or future data."""
    if name == "identity":
        return X
    z = np.clip(X, -10, 10)
    if name == "signed_log":
        return np.concatenate([X, np.sign(z) * np.log1p(np.abs(z))], axis=1)
    if name == "interactions":
        return np.concatenate([X, z * np.roll(z, 1, axis=1)], axis=1)
    raise ValueError(f"unknown feature transform {name}")


def make_transformer(inputs: int, outputs: int, width: int) -> Any:
    """Fixed local architecture shared by training and safe numeric-weight loading."""
    import torch
    from torch import nn

    class FeatureTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, width)
            self.position = nn.Parameter(torch.randn(1, inputs, width) * 0.02)
            layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(width, outputs)

        def forward(self, a: Any) -> Any:
            return self.head(self.encoder(self.embed(a.unsqueeze(-1)) + self.position).mean(dim=1))

    return FeatureTransformer()
