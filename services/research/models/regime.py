"""Two-state Markov-switching regime model (statsmodels ``MarkovRegression``, switching variance).

The observation is the equal-weighted market proxy return (bps) over complete session-anchored blocks
(``features.market_blocks``). Parameters are estimated by maximum likelihood on PAST blocks only (each walk-forward
training window). At decision time only **filtered** probabilities ``P(S_t | y_1..y_t)`` are used: they are produced
by running the Hamilton filter forward with the fixed fitted parameters over blocks completed by the decision time.
Smoothed probabilities ``P(S_t | y_1..y_T)`` condition on data after ``t`` and are FORBIDDEN here (see the
filtered-vs-smoothed unit test). States are mapped to ``MarketRegime.TURBULENT`` (larger variance) and
``MarketRegime.CALM``.

Out-of-sample check: the regime-implied variance of the next block, ``sum_j P(S_{t+1}=j | F_t) sigma2_j``, is scored
by QLIKE against the realised squared block return and compared with the training window's unconditional variance
(``unconditional_baseline``). If the Markov model does not beat that baseline the regime is reported as UNKNOWN.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from packages.contracts.enums import MarketRegime

from .estimators import FitError

MIN_REGIME_OBS = 60


def _model(y: np.ndarray) -> Any:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    return MarkovRegression(np.asarray(y, dtype=np.float64), k_regimes=2, trend="c", switching_variance=True)


@dataclass
class RegimeModel:
    params: np.ndarray
    param_names: list[str]
    turbulent_state: int
    n_fit: int
    loglike: float

    @property
    def sigma2(self) -> np.ndarray:
        return np.asarray([self.params[self.param_names.index(f"sigma2[{k}]")] for k in (0, 1)], dtype=np.float64)

    def transition(self) -> np.ndarray:
        """``T[i, j] = P(S_t = i | S_{t-1} = j)``."""
        p00 = float(self.params[self.param_names.index("p[0->0]")])
        p10 = float(self.params[self.param_names.index("p[1->0]")])
        return np.asarray([[p00, p10], [1.0 - p00, 1.0 - p10]], dtype=np.float64)

    def filtered(self, y_bps: np.ndarray) -> np.ndarray:
        """Filtered marginal state probabilities (n, 2) with the FIXED fitted parameters (causal)."""
        y = np.asarray(y_bps, dtype=np.float64)
        if len(y) < 2:
            raise FitError("need at least two blocks to filter")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _model(y).filter(self.params)
        return np.asarray(res.filtered_marginal_probabilities, dtype=np.float64)

    def p_turbulent(self, y_bps: np.ndarray) -> np.ndarray:
        return self.filtered(y_bps)[:, self.turbulent_state]

    def next_variance(self, y_bps: np.ndarray) -> np.ndarray:
        """One-block-ahead variance forecast made at each block from the filtered probabilities."""
        f = self.filtered(y_bps)
        nxt = f @ self.transition().T  # row t: P(S_{t+1} = i | F_t)
        return np.asarray(nxt @ self.sigma2, dtype=np.float64)

    def classify(self, p_turb: float) -> tuple[MarketRegime, float]:
        if not math.isfinite(p_turb):
            return MarketRegime.UNKNOWN, float("nan")
        if p_turb > 0.5:
            return MarketRegime.TURBULENT, p_turb
        return MarketRegime.CALM, 1.0 - p_turb

    def to_json(self) -> dict[str, Any]:
        return {"params": self.params.tolist(), "param_names": list(self.param_names),
                "turbulent_state": self.turbulent_state, "n_fit": self.n_fit, "loglike": self.loglike,
                "spec": "MarkovRegression(k_regimes=2, trend='c', switching_variance=True); filtered probabilities only"}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RegimeModel:
        return cls(np.asarray(d["params"], dtype=np.float64), [str(x) for x in d["param_names"]],
                   int(d["turbulent_state"]), int(d["n_fit"]), float(d["loglike"]))


def fit_regime(y_bps: np.ndarray, *, max_obs: int) -> RegimeModel:
    """MLE fit on the most recent ``max_obs`` blocks of the (past-only) training series."""
    y = np.asarray(y_bps, dtype=np.float64)
    y = y[np.isfinite(y)][-max_obs:]
    if len(y) < MIN_REGIME_OBS:
        raise FitError(f"only {len(y)} regime blocks (< {MIN_REGIME_OBS})")
    if float(np.std(y)) <= 0:
        raise FitError("constant regime series")
    mod = _model(y)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mod.fit(disp=False)
    except Exception as exc:  # numerical failure: no regime model (reported UNKNOWN), never a guess
        raise FitError(f"Markov fit failed: {type(exc).__name__}: {exc}") from exc
    params = np.asarray(res.params, dtype=np.float64)
    names = [str(n) for n in mod.param_names]
    if not np.isfinite(params).all():
        raise FitError("non-finite Markov parameters")
    s2 = [params[names.index(f"sigma2[{k}]")] for k in (0, 1)]
    if min(s2) <= 0:
        raise FitError("non-positive regime variance")
    return RegimeModel(params, names, int(np.argmax(s2)), len(y), float(res.llf))


def latest_block_index(block_avail_ns: np.ndarray, decision_ns: np.ndarray) -> np.ndarray:
    """Index of the last block knowable at each decision time (-1 when none)."""
    return np.searchsorted(np.asarray(block_avail_ns, dtype=np.int64), np.asarray(decision_ns, dtype=np.int64),
                           side="right") - 1
