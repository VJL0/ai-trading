"""Configuration, stable model ids and version hashes for the local quant model suite.

Every model has a stable ``model_id`` (e.g. ``lgbm_reg_60``) and a ``model_version`` = sha256 over the model id,
its canonical parameters and the source code of this subpackage, so a registry entry pins exactly what produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..features import FeatureSpec

MODELS_DIR = Path(__file__).resolve().parent

# ---- stable ids -------------------------------------------------------------------------------------------
ZERO_BASELINE = "zero_baseline"  # predicts 0 net return -> never trades (NO_TRADE)
CLIMATOLOGY = "climatology"  # success probability = training base rate (no skill) -> no probability served
MOMENTUM_RANK = "momentum_rank"  # deterministic momentum t-stat, no fitting
LGBM_RANK = "lgbm_rank"
CATBOOST_CLF = "catboost_clf"
HAR_RV = "har_rv"
EWMA = "ewma"
VOL_BLEND = "blend"
MARKOV_2STATE = "markov_2state"
UNCONDITIONAL_VAR = "unconditional_baseline"
SPREAD_BASELINE = "spread_baseline"
SPREAD_PLUS_RESIDUAL = "spread_plus_lgbm_residual"
COST_MODEL_FULL = "cost_model_full"  # existing CostModel one-way total (half spread + fees + fixed + sqrt impact)


def lgbm_reg_id(h_min: int) -> str:
    return f"lgbm_reg_{h_min}"


def ridge_id(h_min: int) -> str:
    return f"ridge_{h_min}"


def ret_target(h_min: int) -> str:
    return f"ret_{h_min}m"


TRADE_TARGET_BASELINE = ZERO_BASELINE


@lru_cache(maxsize=1)
def models_code_hash() -> str:
    """sha256 over this subpackage's ``.py`` sources (path + bytes)."""
    h = hashlib.sha256()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        files.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
    for path in sorted(files):
        h.update(path.relative_to(MODELS_DIR).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return "models-src-sha256:" + h.hexdigest()[:24]


RESEARCH_DIR = MODELS_DIR.parent
# Source files that define how a deployed model's INPUTS are computed at prediction time (bar preparation, features,
# vol inputs, Hamilton filter, estimator loading). A registry trained with different versions of these files is
# refused by ``load_registry`` (fail closed: a feature-definition change must never silently feed old models).
INFERENCE_SOURCES: tuple[Path, ...] = (
    RESEARCH_DIR / "frames.py",
    RESEARCH_DIR / "features.py",
    MODELS_DIR / "features.py",
    MODELS_DIR / "estimators.py",
    MODELS_DIR / "vol.py",
    MODELS_DIR / "regime.py",
)


@lru_cache(maxsize=1)
def inference_code_hash() -> str:
    """sha256 over the inference-defining sources (``INFERENCE_SOURCES``)."""
    h = hashlib.sha256()
    for path in INFERENCE_SOURCES:
        h.update(path.relative_to(RESEARCH_DIR).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return "inference-src-sha256:" + h.hexdigest()[:24]


def _canon(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return format(o, "f")
        if isinstance(o, tuple | set | frozenset):
            return list(o)
        raise TypeError(type(o).__name__)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


def model_version(model_id: str, params: dict[str, Any]) -> str:
    """Stable version hash of (model id, canonical params, model-suite source code)."""
    payload = _canon({"model_id": model_id, "params": params, "code": models_code_hash()})
    return "mv_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _lgbm_default() -> dict[str, Any]:
    return {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "num_boost_round": 150,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
    }


def _rank_default() -> dict[str, Any]:
    return {
        "num_leaves": 7,
        "learning_rate": 0.05,
        "num_boost_round": 100,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
    }


def _cost_default() -> dict[str, Any]:
    return {
        "num_leaves": 7,
        "learning_rate": 0.05,
        "num_boost_round": 100,
        "min_data_in_leaf": 100,
        "feature_fraction": 1.0,
        "lambda_l2": 5.0,
    }


def _catboost_default() -> dict[str, Any]:
    return {"iterations": 200, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0}


def _default_feature_spec() -> FeatureSpec:
    return FeatureSpec(return_horizons=(1, 5, 15, 60), vol_window=30, volume_window=30, ma_window=60)


@dataclass(frozen=True)
class SuiteConfig:
    """All walk-forward / model parameters. ``describe()`` is hashed into the registry."""

    return_horizons_minutes: tuple[int, ...] = (5, 15, 60)
    success_horizon_minutes: int = 60
    rank_horizon_minutes: int = 60
    vol_horizon_minutes: int = 60
    feature_spec: FeatureSpec = field(default_factory=_default_feature_spec)
    # volatility
    har_base_minutes: int | None = None  # None = auto: one session if history allows, else the vol horizon (vol.py)
    har_lags: tuple[int, int, int] = (1, 5, 22)
    har_min_fit_sessions: int = 10  # sessions beyond the monthly window needed before the literal session-based HAR
    ewma_lambda: float = 0.94
    blend_weights: tuple[float, ...] = (0.25, 0.5, 0.75)
    # regime
    regime_block_minutes: int = 5
    regime_max_fit_obs: int = 2000
    # walk-forward
    n_folds: int = 4
    final_test_fraction: float = 0.2
    embargo_steps: int | None = None  # default: longest label horizon in bars
    min_train_fraction: float = 0.4
    calib_fraction: float = 0.25
    min_unique_times: int = 200
    # selection
    fdr_q: float = 0.10
    min_trades: int = 10
    entry_threshold_bps: float = 0.0
    require_frozen_confirmation: bool = True
    # costs
    reference_notional: Decimal = Decimal(10_000)
    max_quote_age_s: float = 300.0
    # models
    seed: int = 20261107
    lgbm_params: dict[str, Any] = field(default_factory=_lgbm_default)
    rank_params: dict[str, Any] = field(default_factory=_rank_default)
    cost_params: dict[str, Any] = field(default_factory=_cost_default)
    catboost_params: dict[str, Any] = field(default_factory=_catboost_default)
    ridge_alpha: float = 10.0
    rank_levels: int = 5
    reliability_bins: int = 10

    def __post_init__(self) -> None:
        hs = self.return_horizons_minutes
        if not hs or min(hs) <= 0 or len(set(hs)) != len(hs):
            raise ValueError("return horizons must be distinct positive minutes")
        for name in ("success_horizon_minutes", "rank_horizon_minutes"):
            if getattr(self, name) not in hs:
                raise ValueError(f"{name} must be one of the return horizons {hs}")
        if self.vol_horizon_minutes <= 0:
            raise ValueError("vol_horizon_minutes must be positive")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1)")
        if any(not 0.0 < w < 1.0 for w in self.blend_weights):
            raise ValueError("blend weights must be in (0, 1) (0 and 1 are the pure EWMA/HAR candidates)")
        if not (len(self.har_lags) == 3 and 1 <= self.har_lags[0] < self.har_lags[1] < self.har_lags[2]):
            raise ValueError("har_lags must be three increasing positive integers (daily < weekly < monthly)")
        if not 0 < self.fdr_q < 1:
            raise ValueError("fdr_q must be in (0, 1)")
        if self.min_trades < 2:
            raise ValueError("min_trades must be >= 2")
        if not isinstance(self.reference_notional, Decimal) or self.reference_notional <= 0:
            raise ValueError("reference_notional must be a positive Decimal")
        if self.har_base_minutes is not None and self.har_base_minutes <= 0:
            raise ValueError("har_base_minutes must be positive (or None for auto)")
        if self.har_min_fit_sessions < 1:
            raise ValueError("har_min_fit_sessions must be >= 1")
        if self.regime_block_minutes <= 0:
            raise ValueError("regime_block_minutes must be positive")
        if self.rank_levels < 2:
            raise ValueError("rank_levels must be >= 2")

    @property
    def max_horizon_minutes(self) -> int:
        return max(*self.return_horizons_minutes, self.vol_horizon_minutes)

    def describe(self) -> dict[str, Any]:
        return {
            "return_horizons_minutes": list(self.return_horizons_minutes),
            "success_horizon_minutes": self.success_horizon_minutes,
            "rank_horizon_minutes": self.rank_horizon_minutes,
            "vol_horizon_minutes": self.vol_horizon_minutes,
            "feature_spec": self.feature_spec.describe(),
            "har_base_minutes": self.har_base_minutes,
            "har_lags": list(self.har_lags),
            "har_min_fit_sessions": self.har_min_fit_sessions,
            "ewma_lambda": self.ewma_lambda,
            "blend_weights": list(self.blend_weights),
            "regime_block_minutes": self.regime_block_minutes,
            "regime_max_fit_obs": self.regime_max_fit_obs,
            "n_folds": self.n_folds,
            "final_test_fraction": self.final_test_fraction,
            "embargo_steps": self.embargo_steps,
            "min_train_fraction": self.min_train_fraction,
            "calib_fraction": self.calib_fraction,
            "min_unique_times": self.min_unique_times,
            "fdr_q": self.fdr_q,
            "min_trades": self.min_trades,
            "entry_threshold_bps": self.entry_threshold_bps,
            "require_frozen_confirmation": self.require_frozen_confirmation,
            "reference_notional": format(self.reference_notional, "f"),
            "max_quote_age_s": self.max_quote_age_s,
            "seed": self.seed,
            "lgbm_params": dict(self.lgbm_params),
            "rank_params": dict(self.rank_params),
            "cost_params": dict(self.cost_params),
            "catboost_params": dict(self.catboost_params),
            "ridge_alpha": self.ridge_alpha,
            "rank_levels": self.rank_levels,
            "reliability_bins": self.reliability_bins,
        }


def feature_spec_to_json(fs: FeatureSpec) -> dict[str, Any]:
    return {
        "return_horizons": list(fs.return_horizons),
        "vol_window": fs.vol_window,
        "volume_window": fs.volume_window,
        "ma_window": fs.ma_window,
    }


def feature_spec_from_json(d: dict[str, Any]) -> FeatureSpec:
    return FeatureSpec(
        return_horizons=tuple(int(h) for h in d["return_horizons"]),
        vol_window=int(d["vol_window"]),
        volume_window=int(d["volume_window"]),
        ma_window=int(d["ma_window"]),
    )
