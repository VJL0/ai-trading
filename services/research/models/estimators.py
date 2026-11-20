"""Fitted estimators of the suite with deterministic fitting and pickle-free persistence.

* ``BoosterModel`` - LightGBM regressor (``lgbm_reg_{h}``), lambdarank ranker (``lgbm_rank``) or the execution-cost
  residual regressor; persisted in LightGBM's native text format.
* ``LinearModel`` - Ridge regression on standardised features (``ridge_{h}``, scikit-learn fit); persisted as JSON
  (scaler mean/scale, coefficients, intercept) so loading never unpickles anything.
* ``ZeroModel`` - the zero-forecast baseline (never trades).
* ``MomentumModel`` - deterministic momentum t-stat ranker (no fitting).
* ``SuccessClassifier`` - CatBoost classifier (native ``.cbm``) + held-out calibrators (isotonic and Platt/sigmoid from
  ``services.research.ml``); the calibrator is chosen by Brier score on validation folds.

All fits are single-threaded with fixed seeds (LightGBM ``deterministic=True``; CatBoost ``random_seed``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..features import FeatureSpec
from ..ml import Calibrator, fit_calibrator
from ..stats import brier_score
from .features import momentum_score

MIN_FIT_ROWS = 50


class FitError(ValueError):
    """The sample cannot support this fit (the candidate is skipped, never silently replaced)."""


def lgbm_train(X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int, objective: str,
               group: np.ndarray | None = None) -> Any:
    import lightgbm as lgb

    if len(X) < MIN_FIT_ROWS:
        raise FitError(f"only {len(X)} rows (< {MIN_FIT_ROWS})")
    p = dict(params)
    rounds = int(p.pop("num_boost_round", 100))
    full: dict[str, Any] = {
        "objective": objective,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbosity": -1,
        **p,
    }
    if objective == "lambdarank":
        full.setdefault("metric", "ndcg")
    ds = lgb.Dataset(X, label=y, group=group, free_raw_data=True)
    return lgb.train(full, ds, num_boost_round=rounds)


@dataclass
class BoosterModel:
    model_id: str
    booster: Any

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(X), dtype=np.float64)

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model_id}.lgb.txt"
        self.booster.save_model(str(path))
        return [path]

    @classmethod
    def load(cls, model_id: str, directory: Path) -> BoosterModel:
        import lightgbm as lgb

        return cls(model_id, lgb.Booster(model_file=str(directory / f"{model_id}.lgb.txt")))


def fit_lgbm_regressor(model_id: str, X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int) -> BoosterModel:
    ok = np.isfinite(y)
    return BoosterModel(model_id, lgbm_train(X[ok], y[ok], params, seed, "regression"))


def fit_lgbm_ranker(model_id: str, X: np.ndarray, rel: np.ndarray, decision_ns: np.ndarray, params: dict[str, Any],
                    seed: int) -> BoosterModel:
    """Lambdarank with one query group per decision timestamp (rows must be sorted by decision time)."""
    keep = rel >= 0
    Xk, rk, dk = X[keep], rel[keep], decision_ns[keep]
    if len(dk) and (np.diff(dk) < 0).any():
        raise ValueError("ranker rows must be sorted by decision time")
    _, counts = np.unique(dk, return_counts=True)
    if len(counts) < 10:
        raise FitError(f"only {len(counts)} ranking groups")
    return BoosterModel(model_id, lgbm_train(Xk, rk, params, seed, "lambdarank", group=counts))


@dataclass
class LinearModel:
    model_id: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(((X - self.mean) / self.scale) @ self.coef + self.intercept, dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "alpha": self.alpha}

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model_id}.ridge.json"
        path.write_text(json.dumps(self.to_json(), sort_keys=True, indent=1) + "\n", encoding="utf-8")
        return [path]

    @classmethod
    def load(cls, model_id: str, directory: Path) -> LinearModel:
        d = json.loads((directory / f"{model_id}.ridge.json").read_text(encoding="utf-8"))
        return cls(model_id, np.asarray(d["mean"], dtype=np.float64), np.asarray(d["scale"], dtype=np.float64),
                   np.asarray(d["coef"], dtype=np.float64), float(d["intercept"]), float(d["alpha"]))


def fit_ridge(model_id: str, X: np.ndarray, y: np.ndarray, alpha: float) -> LinearModel:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ok = np.isfinite(y)
    if ok.sum() < MIN_FIT_ROWS:
        raise FitError(f"only {int(ok.sum())} rows (< {MIN_FIT_ROWS})")
    scaler = StandardScaler().fit(X[ok])
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X[ok] - np.asarray(scaler.mean_, dtype=np.float64)) / scale
    reg = Ridge(alpha=alpha).fit(Z, y[ok])
    intercept = float(np.asarray(reg.intercept_, dtype=np.float64).ravel()[0])
    return LinearModel(model_id, np.asarray(scaler.mean_, dtype=np.float64), scale,
                       np.asarray(reg.coef_, dtype=np.float64).ravel(), intercept, alpha)


@dataclass
class ZeroModel:
    model_id: str = "zero_baseline"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)


@dataclass
class MomentumModel:
    feature_columns: tuple[str, ...]
    spec: FeatureSpec
    model_id: str = "momentum_rank"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return momentum_score(X, self.feature_columns, self.spec)


# --------------------------------------------------------------------------------------------
# success classifier + calibration
# --------------------------------------------------------------------------------------------


@dataclass
class SuccessClassifier:
    model_id: str
    clf: Any
    base_rate: float
    calibrators: dict[str, Calibrator] = field(default_factory=lambda: {})
    method: str | None = None  # chosen calibrator ("isotonic" | "sigmoid") or None (raw)

    def raw(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(X)[:, 1], dtype=np.float64)

    def calibrated(self, X: np.ndarray, method: str | None = None) -> np.ndarray:
        m = self.method if method is None else method
        p = self.raw(X)
        if m is None or m == "raw":
            return p
        return self.calibrators[m].transform(p)

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        cbm = directory / f"{self.model_id}.cbm"
        self.clf.save_model(str(cbm))
        meta = directory / f"{self.model_id}.calibration.json"
        meta.write_text(json.dumps({
            "base_rate": self.base_rate,
            "method": self.method,
            "calibrators": {k: v.to_json() for k, v in sorted(self.calibrators.items())},
        }, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        return [cbm, meta]

    @classmethod
    def load(cls, model_id: str, directory: Path) -> SuccessClassifier:
        from catboost import CatBoostClassifier

        clf = CatBoostClassifier()
        clf.load_model(str(directory / f"{model_id}.cbm"))
        meta = json.loads((directory / f"{model_id}.calibration.json").read_text(encoding="utf-8"))
        cals = {k: Calibrator.from_json(v) for k, v in meta["calibrators"].items()}
        return cls(model_id, clf, float(meta["base_rate"]), cals, meta["method"])


def fit_success_classifier(model_id: str, X_fit: np.ndarray, y_fit: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray,
                           params: dict[str, Any], seed: int) -> tuple[SuccessClassifier, dict[str, str]]:
    """CatBoost on the fit block; isotonic AND sigmoid calibrators on the held-out calibration block.

    Returns the classifier and per-method calibration notes (a method missing from ``calibrators`` could not be fitted).
    """
    from catboost import CatBoostClassifier

    if len(X_fit) < MIN_FIT_ROWS:
        raise FitError(f"only {len(X_fit)} fit rows (< {MIN_FIT_ROWS})")
    if len(np.unique(y_fit)) < 2:
        raise FitError("single-class training labels")
    clf = CatBoostClassifier(loss_function="Logloss", random_seed=seed, thread_count=1, verbose=False,
                             allow_writing_files=False, **params)
    clf.fit(X_fit, y_fit)
    model = SuccessClassifier(model_id, clf, float(np.mean(y_fit)))
    notes: dict[str, str] = {}
    if len(X_cal):
        raw_cal = model.raw(X_cal)
        for method in ("isotonic", "sigmoid"):
            cal, note = fit_calibrator(raw_cal, y_cal, method)
            notes[method] = note
            if cal is not None:
                model.calibrators[method] = cal
    else:
        notes = {"isotonic": "no calibration block", "sigmoid": "no calibration block"}
    return model, notes


def choose_calibrator(p_raw: np.ndarray, p_by_method: dict[str, np.ndarray], y: np.ndarray
                      ) -> tuple[str | None, dict[str, float]]:
    """Pick the calibrator with the lowest validation Brier; ``None`` when no calibrator beats the raw scores."""
    briers: dict[str, float] = {"raw": brier_score(p_raw, y)}
    for m, p in sorted(p_by_method.items()):
        briers[m] = brier_score(p, y)
    best: str | None = None
    best_b = briers["raw"]
    for m in ("isotonic", "sigmoid"):
        b = briers.get(m)
        if b is not None and math.isfinite(b) and (not math.isfinite(best_b) or b < best_b):
            best, best_b = m, b
    return best, briers
