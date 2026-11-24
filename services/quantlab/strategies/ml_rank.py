"""Family G — machine-learned cross-sectional ranking vs a linear baseline on identical point-in-time features.

Controlled comparison: LightGBM (non-linear, interactions) and ridge regression (linear) are trained on the same
features and the same purged, expanding walk-forward folds; both feed the same top-N long-only portfolio rule as
Strategy A. The advanced model earns a place only if it beats the simple one after costs and multiple testing.

* Features at the close of t (all trailing): momentum 21/63/126/252 (and 12-1), realized vol 21/63, RSI(2),
  RSI(14), relative volume, 52-week-high ratio, distance from the 50/200-day averages, market beta — each
  cross-sectionally ranked within the liquid universe (robust to scale and regime drift).
* Label: cross-sectionally ranked forward 21-session close-to-close return (t -> t+21).
* Folds: for test year Y, train on decision dates whose label window ends before Y (21-session purge),
  sampling every 5th date to limit overlap; predictions for year Y come only from models trained before Y.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..backtest import apply_rebalance_schedule, hold_schedule
from ..panel import EquityPanel, compound, shift
from . import features as F
from .momentum import ScoreCache, top_n_weights

FAMILY = "G_ml_rank"
HORIZON = 21


def feature_stack(panel: EquityPanel, mask: np.ndarray) -> tuple[list[str], np.ndarray]:
    raw = {
        "mom21": F.momentum(panel, 21), "mom63": F.momentum(panel, 63), "mom126": F.momentum(panel, 126),
        "mom252": F.momentum(panel, 252), "mom12_1": F.momentum(panel, 252, 21),
        "vol21": F.realized_vol(panel, 21), "vol63": F.realized_vol(panel, 63), "rsi2": F.rsi(panel, 2),
        "rsi14": F.rsi(panel, 14), "high52": F.high_52w_ratio(panel),
        "dist50": panel.f["adj_close"] / F.sma(panel, 50) - 1, "dist200": panel.f["adj_close"] / F.sma(panel, 200) - 1,
        "beta": F.rolling_beta(panel, 252),
    }
    v = panel.f["volume"]
    from ..panel import trailing_mean

    raw["relvol"] = v / np.where(trailing_mean(v, 20, 15) > 0, trailing_mean(v, 20, 15), np.nan)
    names = sorted(raw)
    X = np.stack([F.cross_rank(raw[k], mask) - 0.5 for k in names], axis=-1)  # (T, N, K)
    return names, X


def forward_label(panel: EquityPanel, mask: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    fwd = shift(compound(panel.f["ret_cc"], horizon, int(horizon * 0.8)), -horizon)
    return F.cross_rank(fwd, mask & np.isfinite(fwd)) - 0.5


class WalkForwardScores:
    """Out-of-sample scores per model kind, produced once and cached for every portfolio variant."""

    def __init__(self, panel: EquityPanel, cache: ScoreCache, *, universe: str = "liquid1000",
                 first_test_year: int = 2019, sample_every: int = 5, seed: int = 7) -> None:
        self.panel = panel
        self.mask = cache.universe(universe)
        self.cols = np.where(self.mask.any(axis=0))[0]  # only names that are ever eligible (memory)
        names, X = feature_stack(panel, self.mask)
        self.names, self.X = names, X[:, self.cols, :].astype(np.float32)
        del X
        self.y = forward_label(panel, self.mask)[:, self.cols].astype(np.float32)
        self.sub_mask = self.mask[:, self.cols]
        self.first_test_year = first_test_year
        self.sample_every = sample_every
        self.seed = seed
        self._scores: dict[str, np.ndarray] = {}
        self.fold_log: list[dict[str, Any]] = []

    def scores(self, kind: str) -> np.ndarray:
        if kind not in self._scores:
            self._scores[kind] = self._fit_predict(kind)
        return self._scores[kind]

    def _fit_predict(self, kind: str) -> np.ndarray:
        T, N, K = self.X.shape
        years = np.asarray(self.panel.dates.year)
        out = np.full((T, N), np.nan)
        full = np.full((T, self.panel.shape[1]), np.nan)
        for y in sorted({int(v) for v in years if v >= self.first_test_year}):
            test_rows = np.where(years == y)[0]
            first_test = int(test_rows[0])
            train_end = first_test - HORIZON - 1  # purge: label windows end before the test year
            train_rows = np.arange(252, max(252, train_end), self.sample_every)
            if len(train_rows) < 50:
                continue
            Xtr = self.X[train_rows].reshape(-1, K)
            ytr = self.y[train_rows].reshape(-1)
            ok = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
            Xtr, ytr = Xtr[ok], ytr[ok]
            model = self._model(kind)
            model.fit(Xtr, ytr)
            Xte = self.X[test_rows].reshape(-1, K)
            ok_te = np.isfinite(Xte).all(axis=1) & self.sub_mask[test_rows].reshape(-1)
            pred = np.full(len(Xte), np.nan)
            if ok_te.any():
                pred[ok_te] = model.predict(Xte[ok_te])
            out[test_rows] = pred.reshape(len(test_rows), N)
            self.fold_log.append({"model": kind, "test_year": y, "train_rows": len(ytr),
                                  "train_dates": len(train_rows)})
        full[:, self.cols] = out
        return full

    def _model(self, kind: str) -> Any:
        if kind == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=10.0)
        if kind == "lgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                                     subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                     random_state=self.seed, n_jobs=4, deterministic=True, force_row_wise=True,
                                     verbose=-1)
        raise ValueError(kind)


def variants() -> Iterator[dict[str, Any]]:
    for model, n, reb, reg in itertools.product(("ridge", "lgbm"), (10, 30), (5, 21), ("none", "volscale")):
        yield {"model": model, "n": n, "rebalance": reb, "regime": reg}


def targets(wf: WalkForwardScores, cache: ScoreCache, p: dict[str, Any]) -> np.ndarray:
    sc = wf.scores(str(p["model"]))
    w = top_n_weights(sc, wf.mask & np.isfinite(sc), int(p["n"]))
    w = w * cache.regime(str(p["regime"]))[:, None]
    return apply_rebalance_schedule(w, hold_schedule(len(wf.panel.dates), int(p["rebalance"])))
