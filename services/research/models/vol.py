"""Volatility forecasts: HAR-RV (OLS), RiskMetrics EWMA and their blend, compared by out-of-sample QLIKE.

HAR-RV (Corsi 2009): ``RV_next = b0 + b_d RV_d + b_w RV_w + b_m RV_m`` estimated by ordinary least squares on PAST
data only (each walk-forward training window). Components are trailing realised variances of intra-session one-bar
log returns averaged over 1, 5 and 22 base periods ("daily", "weekly", "monthly"). The base period is chosen by
``targets.choose_har_base_minutes`` and recorded in the registry: by default ONE REGULAR SESSION (390 trading
minutes: the literal daily/weekly/monthly HAR on a rolling trading-minute clock) whenever the data has at least
22 + ``har_min_fit_sessions`` sessions; on shorter samples (e.g. the 20-session synthetic fixture) it falls back
to horizon-length periods (60 one-minute bars) and every report says FALLBACK.

EWMA (RiskMetrics): per-bar variance ``s2_t = lam s2_{t-1} + (1-lam) r_t^2`` (``lam`` configurable, default 0.94);
horizon forecast ``H * s2_t``.

Blend: ``w * HAR + (1 - w) * EWMA`` for ``w`` in the configured grid. The model is chosen by mean QLIKE
``RV/F - ln(RV/F) - 1`` (Patton 2011, robust to noisy RV proxies) on the development folds; MSE of the volatility
(sqrt) forecast is reported as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EWMA, HAR_RV, VOL_BLEND
from .estimators import MIN_FIT_ROWS, FitError
from .features import VolSpec

VAR_FLOOR = 1e-12


def qlike(rv: np.ndarray, f: np.ndarray) -> float:
    r = np.maximum(np.asarray(rv, dtype=np.float64), VAR_FLOOR)
    ff = np.maximum(np.asarray(f, dtype=np.float64), VAR_FLOOR)
    ok = np.isfinite(r) & np.isfinite(ff)
    if not ok.any():
        return float("nan")
    x = r[ok] / ff[ok]
    return float(np.mean(x - np.log(x) - 1.0))


def vol_rmse_bps(rv: np.ndarray, f: np.ndarray, scale: float = 10_000.0) -> float:
    r = np.asarray(rv, dtype=np.float64)
    ff = np.asarray(f, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(ff) & (r >= 0) & (ff >= 0)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((np.sqrt(r[ok]) - np.sqrt(ff[ok])) ** 2)) * scale)


@dataclass
class HarModel:
    coef: np.ndarray  # [b0, b_d, b_w, b_m] in variance per base period
    floor: float
    base_bars: int
    horizon_bars: int

    def forecast(self, vol_X: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(vol_X)), vol_X[:, 0], vol_X[:, 1], vol_X[:, 2]])
        per_base = Z @ self.coef
        return np.maximum(per_base, self.floor) * (self.horizon_bars / self.base_bars)

    def to_json(self) -> dict[str, Any]:
        return {"coef": self.coef.tolist(), "floor": self.floor, "base_bars": self.base_bars,
                "horizon_bars": self.horizon_bars, "terms": ["const", "rv_daily", "rv_weekly", "rv_monthly"]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> HarModel:
        return cls(np.asarray(d["coef"], dtype=np.float64), float(d["floor"]), int(d["base_bars"]), int(d["horizon_bars"]))


def fit_har(vol_X: np.ndarray, rv_target: np.ndarray, vs: VolSpec) -> HarModel:
    ok = np.isfinite(vol_X[:, :3]).all(axis=1) & np.isfinite(rv_target) & (rv_target > 0)
    if ok.sum() < MIN_FIT_ROWS:
        raise FitError(f"only {int(ok.sum())} HAR rows (< {MIN_FIT_ROWS})")
    y = rv_target[ok] * (vs.base_bars / vs.horizon_bars)
    Z = np.column_stack([np.ones(int(ok.sum())), vol_X[ok, 0], vol_X[ok, 1], vol_X[ok, 2]])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    floor = max(VAR_FLOOR, 0.01 * float(np.median(y)))
    return HarModel(np.asarray(coef, dtype=np.float64), floor, vs.base_bars, vs.horizon_bars)


def ewma_forecast(vol_X: np.ndarray, vs: VolSpec) -> np.ndarray:
    return np.maximum(vol_X[:, 3], VAR_FLOOR) * vs.horizon_bars


@dataclass
class VolModel:
    """Deployed vol forecaster: ``selected`` in {har_rv, ewma, blend}; ``weight`` = HAR weight."""

    selected: str
    weight: float
    har: HarModel | None
    vs: VolSpec

    def forecast(self, vol_X: np.ndarray) -> np.ndarray:
        ew = ewma_forecast(vol_X, self.vs)
        if self.weight <= 0 or self.har is None:
            return ew
        return self.weight * self.har.forecast(vol_X) + (1.0 - self.weight) * ew

    def to_json(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "har_weight": self.weight,
            "har": self.har.to_json() if self.har is not None else None,
            "ewma_lambda": self.vs.ewma_lambda,
            "horizon_bars": self.vs.horizon_bars,
            "base_bars": self.vs.base_bars,
            "lags": list(self.vs.lags),
            "ewma_init_bars": self.vs.ewma_init_bars,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> VolModel:
        vs = VolSpec(int(d["horizon_bars"]), int(d["base_bars"]), (int(d["lags"][0]), int(d["lags"][1]), int(d["lags"][2])),
                     float(d["ewma_lambda"]), int(d["ewma_init_bars"]))
        har = HarModel.from_json(d["har"]) if d.get("har") else None
        return cls(str(d["selected"]), float(d["har_weight"]), har, vs)


def candidate_forecasts(har: HarModel | None, vol_X: np.ndarray, vs: VolSpec, weights: tuple[float, ...]
                        ) -> dict[str, np.ndarray]:
    """Forecasts of every vol candidate: ``har_rv``, ``ewma`` and ``blend_<w>`` for each grid weight."""
    out: dict[str, np.ndarray] = {EWMA: ewma_forecast(vol_X, vs)}
    if har is not None:
        h = har.forecast(vol_X)
        out[HAR_RV] = h
        for w in weights:
            out[blend_name(w)] = w * h + (1.0 - w) * out[EWMA]
    return out


def blend_name(w: float) -> str:
    return f"{VOL_BLEND}_{w:.2f}"


def parse_candidate(name: str) -> tuple[str, float]:
    """(family, HAR weight) of a candidate name."""
    if name == HAR_RV:
        return HAR_RV, 1.0
    if name == EWMA:
        return EWMA, 0.0
    if name.startswith(VOL_BLEND + "_"):
        return VOL_BLEND, float(name.split("_", 1)[1])
    raise ValueError(f"unknown vol candidate {name}")


def vol_metrics(rv: np.ndarray, f: np.ndarray, *, scale: float = 10_000.0) -> dict[str, Any]:
    """QLIKE and RMSE of the volatility (sqrt) forecast in bps (``scale`` = 1 when inputs are already in bps^2)."""
    q = qlike(rv, f)
    rm = vol_rmse_bps(rv, f, scale)
    return {"qlike": q if math.isfinite(q) else None, "vol_rmse_bps": rm if math.isfinite(rm) else None,
            "n_test": len(rv)}
