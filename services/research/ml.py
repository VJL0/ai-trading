"""Supervised walk-forward research: LightGBM / CatBoost vs a deterministic baseline (spec §3).

Pipeline (``run_walk_forward``):

1. **Panel** — causal features (``features.FeatureSpec``: multi-horizon log returns, realised vol, volume
   z-score, range, distance from the rolling mean) and forward-return labels over ``H`` bars that follow
   the execution convention of the backtester (enter at the first open at/after the decision time, exit
   ``H`` bars later). Rows with incomplete features/labels are dropped, never filled.
2. **Splits** — ``splits.plan_walk_forward``: a frozen final test interval (last fraction of decision
   times), expanding walk-forward folds over the development period, purging of training samples whose
   label window overlaps the test window, an embargo of ``E`` decision steps, and a held-out calibration
   block at the end of every training set.
3. **Variants** — the deterministic momentum t-stat baseline (no fitting) plus LightGBM and CatBoost
   classifiers (P(fwd_ret > 0)) and regressors (E[fwd_ret]); fixed seeds, single-threaded, deterministic.
4. **Calibration** — classifier probabilities are calibrated (isotonic or sigmoid/Platt) on the held-out
   calibration block of each training set only; Brier score (raw, calibrated and climatology), log loss and
   a reliability table are computed out of sample.
5. **Multiple testing** — every variant is a hypothesis "out-of-sample rank IC > 0". One-sided p-values use
   an effective sample size ``n_eff = unique OOS decision times / H`` (overlapping labels and cross-sectional
   dependence make naive ``n`` far too optimistic). **Benjamini-Hochberg** controls the false discovery
   rate across the variant family at level ``q``; adjusted q-values are reported.
6. **Selection** uses development-fold metrics only (``select_variant``); the frozen final test is then
   evaluated **once** for the selected variant and the pre-registered baseline. The sha256 of the
   final-test sample keys is recorded, and a registry (``<out_dir>/frozen_test_registry.json``) refuses a
   second evaluation of the same interval unless reuse is explicitly allowed (then labelled CONTAMINATED).

Nothing here is evidence of edge on synthetic or small samples; results say so explicitly.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from packages.core.money import floor_shares

from .costs import CostModel
from .features import FeatureSpec, build_panel, latest_feature_row
from .frames import (
    SegmentPolicy,
    bar_trading_minutes,
    dataset_version_of,
    is_intraday,
    is_simulated_of,
    last_clean_segment,
    prepare_bars,
)
from .provenance import (
    NO_EDGE_NOTE,
    code_version_hash,
    config_hash,
    library_versions,
    new_run_id,
    relpath,
    utc_now_iso,
    write_json,
)
from .signals import ModelPrediction
from .splits import Fold, SplitPlan, plan_walk_forward
from .stats import (
    benjamini_hochberg,
    brier_score,
    expected_calibration_error,
    ic_pvalue,
    log_loss,
    mean_tstat,
    reliability_table,
    spearman_ic,
)

Library = Literal["baseline", "lightgbm", "catboost"]
Task = Literal["rule", "classifier", "regressor"]
MIN_FIT_SAMPLES = 40
MIN_CALIB_SAMPLES = 20


class FrozenTestReuseError(RuntimeError):
    """The frozen final test interval was already evaluated (in this run or a previous one)."""


class FrozenTestRegistryError(FrozenTestReuseError):
    """The frozen-test registry cannot be read, so prior evaluations cannot be ruled out (fail closed)."""


@dataclass(frozen=True)
class VariantSpec:
    name: str
    library: Library
    task: Task
    params: dict[str, Any] = field(default_factory=lambda: {})

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "library": self.library, "task": self.task, "params": dict(self.params)}


DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("baseline_momentum_tstat", "baseline", "rule", {"rule": "ret_L / (rv * sqrt(L)), L = longest return horizon"}),
    VariantSpec(
        "lgbm_classifier",
        "lightgbm",
        "classifier",
        {"num_leaves": 7, "learning_rate": 0.03, "num_boost_round": 150, "min_data_in_leaf": 20,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0},
    ),
    VariantSpec(
        "lgbm_regressor",
        "lightgbm",
        "regressor",
        {"num_leaves": 7, "learning_rate": 0.03, "num_boost_round": 150, "min_data_in_leaf": 20,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0},
    ),
    VariantSpec("catboost_classifier", "catboost", "classifier", {"iterations": 200, "depth": 4, "learning_rate": 0.05,
                                                                 "l2_leaf_reg": 3.0}),
    VariantSpec("catboost_regressor", "catboost", "regressor", {"iterations": 200, "depth": 4, "learning_rate": 0.05,
                                                               "l2_leaf_reg": 3.0}),
)


@dataclass(frozen=True)
class WalkForwardConfig:
    horizon_bars: int = 5
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    n_folds: int = 4
    final_test_fraction: float = 0.2
    embargo_steps: int | None = None  # default: horizon_bars decision steps
    min_train_fraction: float = 0.4
    calib_fraction: float = 0.25
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"
    seed: int = 20261105
    fdr_q: float = 0.10
    variants: tuple[VariantSpec, ...] = DEFAULT_VARIANTS
    reference_notional: Decimal = Decimal(10_000)
    same_session: bool | None = None
    min_unique_times: int = 40
    reliability_bins: int = 10

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.calibration not in ("isotonic", "sigmoid"):
            raise ValueError("calibration must be 'isotonic' or 'sigmoid'")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names) or not names:
            raise ValueError("variant names must be unique and non-empty")
        if not isinstance(self.reference_notional, Decimal) or self.reference_notional <= 0:
            raise ValueError("reference_notional must be a positive Decimal")

    @property
    def embargo(self) -> int:
        return self.horizon_bars if self.embargo_steps is None else self.embargo_steps

    def describe(self) -> dict[str, Any]:
        return {
            "horizon_bars": self.horizon_bars,
            "features": self.feature_spec.describe(),
            "n_folds": self.n_folds,
            "final_test_fraction": self.final_test_fraction,
            "embargo_steps": self.embargo,
            "min_train_fraction": self.min_train_fraction,
            "calib_fraction": self.calib_fraction,
            "calibration": self.calibration,
            "seed": self.seed,
            "fdr_q": self.fdr_q,
            "variants": [v.describe() for v in self.variants],
            "reference_notional": format(self.reference_notional, "f"),
            "same_session": self.same_session,
            "min_unique_times": self.min_unique_times,
        }


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def baseline_score(X: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Deterministic momentum t-stat from the feature matrix (no fitting)."""
    cols = list(spec.columns)
    lag = max(spec.return_horizons)
    ret = X[:, cols.index(f"ret_{lag}")]
    rv = X[:, cols.index(f"rv_{spec.vol_window}")]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(rv > 0, ret / (rv * math.sqrt(lag)), 0.0)
    return np.asarray(s, dtype=np.float64)


@dataclass
class FittedModel:
    spec: VariantSpec
    feature_spec: FeatureSpec
    booster: Any = None  # lightgbm.Booster | catboost model | None (rule)
    mu_up: float = 0.0
    mu_dn: float = 0.0
    base_rate: float = 0.5

    def raw(self, X: np.ndarray) -> np.ndarray:
        """Classifier: P(up) uncalibrated; regressor: E[fwd_ret]; rule: baseline score."""
        if self.spec.task == "rule":
            return baseline_score(X, self.feature_spec)
        if self.spec.library == "lightgbm":
            return np.asarray(self.booster.predict(X), dtype=np.float64)
        if self.spec.task == "classifier":
            return np.asarray(self.booster.predict_proba(X)[:, 1], dtype=np.float64)
        return np.asarray(self.booster.predict(X), dtype=np.float64)

    def expected_return(self, raw_or_prob: np.ndarray) -> np.ndarray:
        """Expected fwd return (fraction). Classifier: p*mu_up - (1-p)*mu_dn with mu from the fit set."""
        if self.spec.task == "regressor":
            return raw_or_prob
        if self.spec.task == "classifier":
            p = raw_or_prob
            return p * self.mu_up - (1.0 - p) * self.mu_dn
        return np.full(len(raw_or_prob), np.nan)  # the rule baseline makes no return forecast


def fit_variant(spec: VariantSpec, fs: FeatureSpec, X: np.ndarray, fwd: np.ndarray, seed: int) -> FittedModel:
    """Fit one variant deterministically. Raises ValueError when the sample cannot support a fit."""
    y = (fwd > 0).astype(np.int64)
    ups = fwd[fwd > 0]
    dns = -fwd[fwd <= 0]
    model = FittedModel(
        spec, fs, mu_up=float(ups.mean()) if len(ups) else 0.0, mu_dn=float(dns.mean()) if len(dns) else 0.0,
        base_rate=float(y.mean()) if len(y) else 0.5,
    )
    if spec.task == "rule":
        return model
    if len(X) < MIN_FIT_SAMPLES:
        raise ValueError(f"only {len(X)} fit samples (< {MIN_FIT_SAMPLES})")
    if spec.task == "classifier" and len(np.unique(y)) < 2:
        raise ValueError("single-class training labels")
    if spec.library == "lightgbm":
        import lightgbm as lgb

        p = dict(spec.params)
        rounds = int(p.pop("num_boost_round", 100))
        params: dict[str, Any] = {
            "objective": "binary" if spec.task == "classifier" else "regression",
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbosity": -1,
            **p,
        }
        target = y if spec.task == "classifier" else fwd
        ds = lgb.Dataset(X, label=target, free_raw_data=True)
        model.booster = lgb.train(params, ds, num_boost_round=rounds)
        return model
    if spec.library == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        common: dict[str, Any] = {"random_seed": seed, "thread_count": 1, "verbose": False,
                                  "allow_writing_files": False, **spec.params}
        if spec.task == "classifier":
            clf = CatBoostClassifier(loss_function="Logloss", **common)
            clf.fit(X, y)
            model.booster = clf
        else:
            reg = CatBoostRegressor(loss_function="RMSE", **common)
            reg.fit(X, fwd)
            model.booster = reg
        return model
    raise ValueError(f"unknown library {spec.library}")


@dataclass
class Calibrator:
    """Monotone map raw P(up) -> calibrated probability, fitted on a held-out calibration block only."""

    method: str
    n_fit: int
    x_thresholds: list[float] = field(default_factory=lambda: [])
    y_thresholds: list[float] = field(default_factory=lambda: [])
    coef: float = 0.0
    intercept: float = 0.0

    def transform(self, p: np.ndarray) -> np.ndarray:
        pa = np.asarray(p, dtype=np.float64)
        if self.method == "isotonic":
            out = np.interp(pa, self.x_thresholds, self.y_thresholds)
        else:
            z = _logit(pa)
            out = 1.0 / (1.0 + np.exp(-(self.coef * z + self.intercept)))
        return np.clip(out, 1e-4, 1 - 1e-4)

    def to_json(self) -> dict[str, Any]:
        return {"method": self.method, "n_fit": self.n_fit, "x_thresholds": self.x_thresholds,
                "y_thresholds": self.y_thresholds, "coef": self.coef, "intercept": self.intercept}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Calibrator:
        return cls(method=str(d["method"]), n_fit=int(d["n_fit"]), x_thresholds=list(d["x_thresholds"]),
                   y_thresholds=list(d["y_thresholds"]), coef=float(d["coef"]), intercept=float(d["intercept"]))


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def fit_calibrator(raw_p: np.ndarray, y: np.ndarray, method: str) -> tuple[Calibrator | None, str]:
    """Fit on held-out (raw_p, y). Returns (None, reason) when the block cannot support calibration."""
    if len(raw_p) < MIN_CALIB_SAMPLES:
        return None, f"calibration block too small ({len(raw_p)} < {MIN_CALIB_SAMPLES})"
    if len(np.unique(y)) < 2:
        return None, "calibration block has a single class"
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_p, y)
        xs = np.asarray(iso.X_thresholds_, dtype=np.float64)
        ys = np.asarray(iso.y_thresholds_, dtype=np.float64)
        return Calibrator("isotonic", len(raw_p), xs.tolist(), ys.tolist()), "ok"
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(raw_p).reshape(-1, 1), y)
    coef = float(np.asarray(lr.coef_, dtype=np.float64).ravel()[0])
    intercept = float(np.asarray(lr.intercept_, dtype=np.float64).ravel()[0])
    return Calibrator("sigmoid", len(raw_p), coef=coef, intercept=intercept), "ok"


# --------------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------------


@dataclass
class Prepared:
    panel: pd.DataFrame
    X: np.ndarray
    fwd: np.ndarray
    y: np.ndarray
    decision_ns: np.ndarray
    label_end_ns: np.ndarray
    keys: list[str]
    cost_bps: np.ndarray


def _round_trip_cost_bps(panel: pd.DataFrame, cost_model: CostModel, notional: Decimal) -> np.ndarray:
    """Point-in-time round-trip cost (bps) per sample from the Decimal cost model (diagnostic use only).

    Uses each sample's own close and trailing ADV (both knowable at its decision time) - never statistics of
    the whole panel, which would leak frozen-test prices into development metrics. Inputs are quantised
    (price to the cent, ADV to 3 significant figures) and memoised to keep this fast.
    """
    cache: dict[tuple[str, str, str], float] = {}
    out = np.empty(len(panel), dtype=np.float64)
    syms = panel["symbol"].astype(str).to_numpy()
    closes = panel["close"].to_numpy(dtype=np.float64)
    advs = panel["adv_shares"].to_numpy(dtype=np.float64)
    for i in range(len(panel)):
        price = Decimal(f"{closes[i]:.2f}")
        adv_f = advs[i]
        adv_key = f"{adv_f:.3g}" if math.isfinite(adv_f) and adv_f > 0 else "none"
        key = (syms[i], format(price, "f"), adv_key)
        cost = cache.get(key)
        if cost is None:
            adv = None if adv_key == "none" else Decimal(f"{float(adv_key):.0f}")
            qty = max(Decimal(1), floor_shares(notional / price))
            cost = float(cost_model.estimate_round_trip(syms[i], qty, price, adv_shares=adv).total_bps)
            cache[key] = cost
        out[i] = cost
    return out


def prepare_supervised(bars: pd.DataFrame, cfg: WalkForwardConfig, cost_model: CostModel) -> Prepared:
    tf = str(bars["timeframe"].iloc[0])
    same_session = cfg.same_session if cfg.same_session is not None else is_intraday(tf)
    panel = build_panel(bars, cfg.feature_spec, cfg.horizon_bars, same_session=same_session)
    if panel.empty:
        raise ValueError("no complete feature/label rows")
    X = panel[list(cfg.feature_spec.columns)].to_numpy(dtype=np.float64)
    fwd = panel["fwd_ret"].to_numpy(dtype=np.float64)
    keys = [f"{s}|{d}" for s, d in zip(panel["symbol"], panel["decision_ns"], strict=True)]
    return Prepared(
        panel=panel,
        X=X,
        fwd=fwd,
        y=(fwd > 0).astype(np.int64),
        decision_ns=panel["decision_ns"].to_numpy(dtype=np.int64),
        label_end_ns=panel["label_end_ns"].to_numpy(dtype=np.int64),
        keys=keys,
        cost_bps=_round_trip_cost_bps(panel, cost_model, cfg.reference_notional),
    )


def _n_eff(decision_ns: np.ndarray, horizon: int) -> float:
    return len(np.unique(decision_ns)) / float(max(1, horizon))


def evaluate_predictions(
    model: FittedModel,
    calibrator: Calibrator | None,
    X: np.ndarray,
    fwd: np.ndarray,
    cost_bps: np.ndarray,
    decision_ns: np.ndarray,
    horizon: int,
    bins: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Out-of-sample metrics for one evaluation block. Returns (metrics, arrays)."""
    raw = model.raw(X)
    y = (fwd > 0).astype(np.int64)
    arrays: dict[str, np.ndarray] = {"score": raw}
    metrics: dict[str, Any] = {"n": len(fwd), "unique_times": len(np.unique(decision_ns))}
    ic = spearman_ic(raw, fwd)
    n_eff = _n_eff(decision_ns, horizon)
    metrics.update({"rank_ic": ic, "n_eff": n_eff, "ic_p_value_one_sided": ic_pvalue(ic, n_eff)})
    if model.spec.task == "classifier":
        p_cal = calibrator.transform(raw) if calibrator is not None else raw
        arrays["p_raw"] = raw
        arrays["p_cal"] = p_cal
        clim = np.full(len(y), model.base_rate)
        metrics.update(
            {
                "brier_raw": brier_score(raw, y),
                "brier_calibrated": brier_score(p_cal, y) if calibrator is not None else None,
                "brier_climatology": brier_score(clim, y),
                "log_loss_raw": log_loss(raw, y),
                "log_loss_calibrated": log_loss(p_cal, y) if calibrator is not None else None,
                "ece_calibrated": expected_calibration_error(p_cal, y, bins) if calibrator is not None else None,
                "hit_rate": float(np.mean((p_cal > 0.5) == (y == 1))) if len(y) else None,
                "calibrated": calibrator is not None,
                "reliability": reliability_table(p_cal, y, bins) if calibrator is not None else None,
            }
        )
        exp_ret = model.expected_return(p_cal)
    elif model.spec.task == "regressor":
        exp_ret = raw
        metrics.update(
            {
                "rmse": float(np.sqrt(np.mean((raw - fwd) ** 2))) if len(fwd) else None,
                "hit_rate": float(np.mean((raw > 0) == (fwd > 0))) if len(fwd) else None,
            }
        )
    else:
        exp_ret = np.where(raw > 0, np.inf, -np.inf)  # sign rule: long when the t-stat is positive
    arrays["expected_return"] = exp_ret
    # trading proxy: take the long only when expected return exceeds the modelled round-trip cost
    take = exp_ret * 10_000.0 > cost_bps
    net = fwd[take] * 10_000.0 - cost_bps[take]
    mean_net, t_net = mean_tstat(net, n_eff=min(float(take.sum()), n_eff) if take.any() else None)
    metrics.update(
        {
            "selected_fraction": float(take.mean()) if len(take) else 0.0,
            "selected_n": int(take.sum()),
            "mean_net_bps_when_selected": mean_net if take.any() else None,
            "t_net_when_selected": t_net if take.any() else None,
            "mean_cost_bps": float(np.mean(cost_bps)) if len(cost_bps) else None,
        }
    )
    return metrics, arrays


def select_variant(dev_summary: dict[str, dict[str, Any]]) -> str | None:
    """Pick the variant with the best mean development-fold rank IC (ties -> name). Dev metrics ONLY."""
    best: tuple[float, str] | None = None
    for name in sorted(dev_summary):
        mean_ic = dev_summary[name].get("mean_fold_ic")
        if mean_ic is None or not math.isfinite(float(mean_ic)):
            continue
        cand = (-float(mean_ic), name)
        if best is None or cand < best:
            best = cand
    return None if best is None else best[1]


# --------------------------------------------------------------------------------------------
# Frozen final test
# --------------------------------------------------------------------------------------------


class FrozenTest:
    """Holds the frozen final-test indices; ``evaluate_once`` may run exactly once per interval.

    A prior evaluation is detected either by the exact sample-key hash, or - so that changing the horizon,
    folds or features cannot buy a fresh look at the same data - by any registry entry for the same
    ``dataset_version`` whose evaluated time interval overlaps this one and shares at least one symbol.
    """

    def __init__(
        self,
        idx: np.ndarray,
        test_hash: str,
        *,
        registry: Path | None,
        allow_reuse: bool = False,
        interval_ns: tuple[int, int] | None = None,
        symbols: Sequence[str] = (),
    ) -> None:
        self._idx = np.asarray(idx, dtype=np.int64)
        self.hash = test_hash
        self.registry = registry
        self.allow_reuse = allow_reuse
        self.interval_ns = interval_ns
        self.symbols = sorted(set(symbols))
        self._consumed = False
        self.reused_from: dict[str, Any] | None = None

    @property
    def size(self) -> int:
        return len(self._idx)

    def _registry_entries(self) -> dict[str, Any]:
        if self.registry is None or not self.registry.exists():
            return {}
        try:
            data = json.loads(self.registry.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenTestRegistryError(f"frozen-test registry unreadable ({exc}); refusing to evaluate") from exc
        return data if isinstance(data, dict) else {}

    def previously_evaluated(self, dataset_version: str | None = None) -> dict[str, Any] | None:
        # Coordinate with autonomous research's application-wide exposure book. The
        # validator reserves while holding both legacy registry locks, so neither
        # pipeline can race the other's first look at the same market interval.
        book = self.registry.parent.parent / "autonomous-validation.sqlite3" if self.registry else None
        if book is not None and book.exists() and self.interval_ns is not None:
            import sqlite3

            try:
                with contextlib.closing(sqlite3.connect(f"file:{book}?mode=ro", uri=True)) as db:
                    lo, hi = self.interval_ns
                    for symbol in self.symbols:
                        prior = db.execute(
                            "SELECT owner,kind FROM exposures WHERE symbol=? AND lo<=? AND hi>=?"
                            " AND kind!='DEVELOPMENT_SIMULATED' LIMIT 1",
                            (symbol, hi, lo),
                        ).fetchone()
                        if prior:
                            return {"run_id": prior[0], "match": "autonomous_research_exposure", "hash": self.hash}
            except sqlite3.Error as exc:
                raise FrozenTestRegistryError("autonomous exposure book unreadable; refusing evaluation") from exc
        entries = self._registry_entries()
        entry = entries.get(self.hash)
        if isinstance(entry, dict):
            return {**entry, "match": "exact_sample_hash", "hash": self.hash}
        if dataset_version is None or self.interval_ns is None:
            return None
        lo, hi = self.interval_ns
        for key in sorted(entries):
            e = entries[key]
            if not isinstance(e, dict) or e.get("dataset_version") != dataset_version:
                continue
            iv = e.get("interval_ns")
            if not (isinstance(iv, list) and len(iv) == 2):
                continue
            shared = set(self.symbols).intersection(e.get("symbols") or [])
            if int(iv[0]) <= hi and int(iv[1]) >= lo and (shared or not self.symbols):
                return {**e, "match": "overlapping_interval_same_dataset", "hash": key}
        return None

    def evaluate_once(self, fn: Callable[[np.ndarray], dict[str, Any]], *, run_id: str, dataset_version: str) -> dict[str, Any]:
        """Run ``fn`` on the frozen-test indices at most once (per run and across runs sharing the registry).

        Check-and-reserve happens under an exclusive file lock, and the registry entry is written BEFORE ``fn``
        looks at the data: two concurrent runs cannot both get a "first" look, and an evaluation that crashes
        half-way still counts as a look (its entry stays ``STARTED``, so later runs are refused - fail closed).
        """
        if self._consumed:
            raise FrozenTestReuseError("frozen final test already evaluated in this run")
        with _registry_lock(self.registry):
            prior = self.previously_evaluated(dataset_version)
            if prior is not None and not self.allow_reuse:
                raise FrozenTestReuseError(
                    f"frozen final test {self.hash} was already evaluated by run {prior.get('run_id')} "
                    f"({prior.get('match')}: {prior.get('hash')}); re-evaluation refused (pass allow_reuse to "
                    "override; results would be CONTAMINATED)"
                )
            self._consumed = True
            self.reused_from = prior
            self._record(run_id, dataset_version, prior, status="STARTED")
            result = fn(self._idx)
            self._record(run_id, dataset_version, prior, status="COMPLETED")
        return result

    def _record(self, run_id: str, dataset_version: str, prior: dict[str, Any] | None, *, status: str) -> None:
        if self.registry is None:
            return
        entries = self._registry_entries()
        own = entries.get(self.hash)
        first = own if isinstance(own, dict) else None
        history: list[dict[str, Any]] = list(first.get("history", [])) if first else []
        now = utc_now_iso()
        if status == "STARTED":
            history.append({"run_id": run_id, "evaluated_at": now, "status": "STARTED", "reuse": prior is not None,
                            "reuse_of": prior.get("hash") if prior else None})
        else:
            for h in reversed(history):
                if h.get("run_id") == run_id and h.get("status") == "STARTED":
                    h.update({"status": status, "completed_at": now})
                    break
        entries[self.hash] = {
            "run_id": first.get("run_id") if first else run_id,
            "first_evaluated_at": first.get("first_evaluated_at") if first else now,
            "dataset_version": dataset_version,
            "interval_ns": list(self.interval_ns) if self.interval_ns is not None else None,
            "symbols": self.symbols,
            "history": history,
        }
        write_json(self.registry, entries)


@contextlib.contextmanager
def _registry_lock(registry: Path | None) -> Iterator[None]:
    """Exclusive advisory lock (``fcntl.flock``) serialising frozen-test check-and-reserve across processes."""
    if registry is None:
        yield
        return
    registry.parent.mkdir(parents=True, exist_ok=True)
    with (registry.parent / f".{registry.name}.lock").open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def _assert_disjoint_from_final(plan: SplitPlan) -> None:
    final = set(plan.final_test_idx.tolist())
    for f in plan.folds:
        for name, idx in (("fit", f.fit_idx), ("calib", f.calib_idx), ("test", f.test_idx)):
            if final.intersection(idx.tolist()):
                raise AssertionError(f"fold {f.fold} {name} indices intersect the frozen final test")
    for name, idx in (("final_fit", plan.final_fit_idx), ("final_calib", plan.final_calib_idx)):
        if final.intersection(idx.tolist()):
            raise AssertionError(f"{name} indices intersect the frozen final test")


def _fit_and_eval(
    spec: VariantSpec,
    data: Prepared,
    fit_idx: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    cfg: WalkForwardConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], FittedModel | None, Calibrator | None]:
    try:
        model = fit_variant(spec, cfg.feature_spec, data.X[fit_idx], data.fwd[fit_idx], cfg.seed)
    except ValueError as exc:
        return {"status": "SKIPPED", "reason": str(exc)}, {}, None, None
    calibrator: Calibrator | None = None
    calib_note = "n/a"
    if spec.task == "classifier":
        calibrator, calib_note = fit_calibrator(model.raw(data.X[calib_idx]), data.y[calib_idx], cfg.calibration)
    metrics, arrays = evaluate_predictions(
        model, calibrator, data.X[eval_idx], data.fwd[eval_idx], data.cost_bps[eval_idx], data.decision_ns[eval_idx],
        cfg.horizon_bars, cfg.reliability_bins,
    )
    metrics.update({"status": "OK", "fit_n": len(fit_idx), "calib_n": len(calib_idx), "calibration": calib_note})
    return metrics, arrays, model, calibrator


def _summarise_variant(spec: VariantSpec, fold_metrics: list[dict[str, Any]], pooled: dict[str, list[np.ndarray]],
                       cfg: WalkForwardConfig) -> dict[str, Any]:
    ok = [m for m in fold_metrics if m.get("status") == "OK"]
    ics = [float(m["rank_ic"]) for m in ok if m.get("rank_ic") is not None and math.isfinite(float(m["rank_ic"]))]
    out: dict[str, Any] = {
        "folds_ok": len(ok),
        "folds_total": len(fold_metrics),
        "fold_ics": ics,
        "mean_fold_ic": float(np.mean(ics)) if ics else None,
        "std_fold_ic": float(np.std(ics, ddof=1)) if len(ics) > 1 else None,
    }
    if not pooled.get("score"):
        out.update({"pooled_rank_ic": None, "p_value": 1.0, "n_oos": 0})
        return out
    score = np.concatenate(pooled["score"])
    fwd = np.concatenate(pooled["fwd"])
    dns = np.concatenate(pooled["decision_ns"])
    ic = spearman_ic(score, fwd)
    n_eff = _n_eff(dns, cfg.horizon_bars)
    out.update({"pooled_rank_ic": ic, "n_oos": len(fwd), "n_eff": n_eff, "p_value": ic_pvalue(ic, n_eff)})
    if spec.task == "classifier" and pooled.get("p_cal"):
        p_cal = np.concatenate(pooled["p_cal"])
        p_raw = np.concatenate(pooled["p_raw"])
        y = (fwd > 0).astype(np.int64)
        out.update(
            {
                "pooled_brier_raw": brier_score(p_raw, y),
                "pooled_brier_calibrated": brier_score(p_cal, y),
                "pooled_ece_calibrated": expected_calibration_error(p_cal, y, cfg.reliability_bins),
                "pooled_reliability": reliability_table(p_cal, y, cfg.reliability_bins),
                "all_folds_calibrated": all(bool(m.get("calibrated")) for m in ok) and bool(ok),
            }
        )
    return out


@dataclass
class WalkForwardOutput:
    payload: dict[str, Any]
    run_dir: Path
    final_model: FittedModel | None
    final_calibrator: Calibrator | None
    plan: SplitPlan


def run_walk_forward(
    bars_df: pd.DataFrame,
    cost_model: CostModel,
    out_dir: Path | str,
    *,
    config: WalkForwardConfig | None = None,
    run_id: str | None = None,
    allow_final_test_reuse: bool = False,
    evaluate_final: bool = True,
    data_adjustments: dict[str, Any] | None = None,
    segment_policy: SegmentPolicy | None = SegmentPolicy(),  # frozen dataclass: safe default
) -> WalkForwardOutput:
    """Run the full pipeline (module docstring) and write ``<out_dir>/<run_id>/``.

    ``out_dir`` is the base directory (normally ``experiments/ml``); the frozen-test registry lives there.
    """
    cfg = config or WalkForwardConfig()
    bars = prepare_bars(bars_df)
    data_quality: dict[str, Any] = {"applied": "none (segment_policy=None)"}
    if segment_policy is not None:  # features/labels never span a multi-session hole or a ticker change
        bars, data_quality = last_clean_segment(bars, segment_policy)
    tf = str(bars["timeframe"].iloc[0])
    run_id = run_id or new_run_id("wf")
    base = Path(out_dir)
    run_dir = base / run_id
    dataset_version = dataset_version_of(bars)
    simulated = is_simulated_of(bars)
    started = utc_now_iso()
    data = prepare_supervised(bars, cfg, cost_model)
    plan = plan_walk_forward(
        data.decision_ns, data.label_end_ns, keys=data.keys, n_folds=cfg.n_folds,
        final_test_fraction=cfg.final_test_fraction, embargo_steps=cfg.embargo,
        min_train_fraction=cfg.min_train_fraction, calib_fraction=cfg.calib_fraction,
        min_unique_times=cfg.min_unique_times,
    )
    _assert_disjoint_from_final(plan)

    # ---- development walk-forward (the frozen final test is never touched here)
    fold_results: dict[str, list[dict[str, Any]]] = {}
    dev_summary: dict[str, dict[str, Any]] = {}
    oos_rows: list[pd.DataFrame] = []
    for spec in cfg.variants:
        per_fold: list[dict[str, Any]] = []
        pooled: dict[str, list[np.ndarray]] = {"score": [], "fwd": [], "decision_ns": [], "p_cal": [], "p_raw": []}
        for fold in plan.folds:
            m, arrays, _, _ = _fit_and_eval(spec, data, fold.fit_idx, fold.calib_idx, fold.test_idx, cfg)
            m["fold"] = fold.fold
            per_fold.append(m)
            if m.get("status") != "OK":
                continue
            pooled["score"].append(arrays["score"])
            pooled["fwd"].append(data.fwd[fold.test_idx])
            pooled["decision_ns"].append(data.decision_ns[fold.test_idx])
            if "p_cal" in arrays and m.get("calibrated"):
                pooled["p_cal"].append(arrays["p_cal"])
                pooled["p_raw"].append(arrays["p_raw"])
            oos_rows.append(_prediction_frame(data, fold.test_idx, spec.name, f"fold{fold.fold}", arrays))
        fold_results[spec.name] = per_fold
        dev_summary[spec.name] = _summarise_variant(spec, per_fold, pooled, cfg)

    bh = benjamini_hochberg({k: float(v.get("p_value", 1.0)) for k, v in dev_summary.items()}, q=cfg.fdr_q)
    for name, row in bh.items():
        dev_summary[name]["bh"] = row
    selected = select_variant(dev_summary)
    selected_spec = next((v for v in cfg.variants if v.name == selected), None)
    baseline_spec = next((v for v in cfg.variants if v.task == "rule"), None)

    # ---- frozen final test: evaluated once, for the selected variant + the pre-registered baseline
    registry = base / "frozen_test_registry.json"
    ft_idx = plan.final_test_idx
    frozen = FrozenTest(
        ft_idx, plan.final_test_hash, registry=registry, allow_reuse=allow_final_test_reuse,
        interval_ns=(int(data.decision_ns[ft_idx].min()), int(data.label_end_ns[ft_idx].max())) if len(ft_idx) else None,
        symbols=sorted(set(data.panel["symbol"].to_numpy()[ft_idx].astype(str))) if len(ft_idx) else (),
    )
    final_block: dict[str, Any] = {
        "final_test_hash": plan.final_test_hash,
        "final_test_start_ns": plan.final_test_start_ns,
        "final_test_samples": frozen.size,
        "selected_variant": selected,
        "status": "NOT_EVALUATED",
    }
    final_model: FittedModel | None = None
    final_cal: Calibrator | None = None
    final_rows: list[pd.DataFrame] = []
    if evaluate_final and selected_spec is not None and frozen.size > 0:

        def _evaluate(idx: np.ndarray) -> dict[str, Any]:
            nonlocal final_model, final_cal
            out: dict[str, Any] = {}
            specs = [selected_spec] + ([baseline_spec] if baseline_spec and baseline_spec is not selected_spec else [])
            for spec in specs:
                m, arrays, model, cal = _fit_and_eval(spec, data, plan.final_fit_idx, plan.final_calib_idx, idx, cfg)
                out[spec.name] = m
                if arrays:
                    final_rows.append(_prediction_frame(data, idx, spec.name, "final_test", arrays))
                if spec is selected_spec:
                    final_model, final_cal = model, cal
            return out

        try:
            final_block["metrics"] = frozen.evaluate_once(_evaluate, run_id=run_id, dataset_version=dataset_version)
            final_block["status"] = "EVALUATED_REUSED_CONTAMINATED" if frozen.reused_from else "EVALUATED_ONCE"
            if frozen.reused_from:
                final_block["reused_from"] = frozen.reused_from
        except FrozenTestRegistryError as exc:
            final_block["status"] = "REFUSED_REGISTRY_UNREADABLE"
            final_block["reason"] = str(exc)
        except FrozenTestReuseError as exc:
            final_block["status"] = "REFUSED_ALREADY_EVALUATED"
            final_block["reason"] = str(exc)
    elif selected_spec is None:
        final_block["reason"] = "no variant produced valid development metrics"
    elif not evaluate_final:
        final_block["reason"] = "final evaluation disabled by caller"

    # ---- write results
    run_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    if oos_rows:
        p = run_dir / "oos_predictions.csv"
        pd.concat(oos_rows, ignore_index=True).to_csv(p, index=False)
        files["oos_predictions.csv"] = _sha256_file(p)
    if final_rows:
        p = run_dir / "final_test_predictions.csv"
        pd.concat(final_rows, ignore_index=True).to_csv(p, index=False)
        files["final_test_predictions.csv"] = _sha256_file(p)
    if final_model is not None:
        files.update(_save_model(run_dir, final_model, final_cal))

    cfg_desc = cfg.describe()
    cfg_h = config_hash({"config": cfg_desc, "cost_model": cost_model.describe(), "dataset_version": dataset_version})
    calibration_evaluated = _calibration_passed(final_cal, final_block, selected_spec)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "kind": "ml_walk_forward",
        "created_at": started,
        "dataset_version": dataset_version,
        "is_simulated": simulated,
        "data_label": "SIMULATED (synthetic fixture)" if simulated else "HISTORICAL (non-synthetic bars)",
        "timeframe": tf,
        "horizon_bars": cfg.horizon_bars,
        "horizon_minutes": cfg.horizon_bars * bar_trading_minutes(tf),
        "bar_count": len(bars),
        "panel_rows": len(data.panel),
        "symbols": sorted(set(data.panel["symbol"])),
        "data_adjustments": data_adjustments or {"applied": "none (caller supplied no corporate-action handling)"},
        "data_quality": data_quality,
        "label_note": "price-only forward returns (dividends excluded); bars must be split-adjusted",
        "code_version": code_version_hash(),
        "config_hash": cfg_h,
        "config": cfg_desc,
        "cost_model": cost_model.describe(),
        "splits": plan.describe(),
        "development": {"folds": fold_results, "summary": dev_summary},
        "multiple_testing": {
            "method": "Benjamini-Hochberg FDR across variants",
            "q": cfg.fdr_q,
            "hypothesis": "H0: out-of-sample rank IC <= 0 (one-sided)",
            "p_value": "t-approximation with n_eff = unique OOS decision times / horizon_bars",
            "rejections": sorted(k for k, v in bh.items() if v["reject"]),
            "family_size": len(bh),
        },
        "selection": {
            "rule": "max mean development-fold rank IC (frozen final test never consulted)",
            "selected_variant": selected,
            "selected_bh_reject": bool(bh.get(selected, {}).get("reject")) if selected else False,
        },
        "final_test": final_block,
        "probability_calibrated_for_signals": calibration_evaluated,
        "probability_calibrated_rule": (
            "True only for a classifier with a held-out calibrator whose calibrated Brier score on the frozen "
            "final test (evaluated once) is <= the climatology Brier score"
        ),
        "no_edge_note": NO_EDGE_NOTE,
        "run_dir": relpath(run_dir),
    }
    write_json(run_dir / "results.json", payload)
    files["results.json"] = _sha256_file(run_dir / "results.json")
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "kind": "ml_walk_forward",
            "created_at": started,
            "dataset_version": dataset_version,
            "is_simulated": simulated,
            "code_version": code_version_hash(),
            "config_hash": cfg_h,
            "final_test_hash": plan.final_test_hash,
            "final_test_status": final_block["status"],
            "files_sha256": files,
            "libraries": library_versions("lightgbm", "catboost", "scikit-learn", "scipy", "numpy", "pandas"),
            "no_edge_note": NO_EDGE_NOTE,
        },
    )
    return WalkForwardOutput(payload=payload, run_dir=run_dir, final_model=final_model, final_calibrator=final_cal,
                             plan=plan)


def _calibration_passed(cal: Calibrator | None, final_block: dict[str, Any], spec: VariantSpec | None) -> bool:
    """Probabilities may be labelled calibrated only if: a held-out calibrator exists, the frozen test was
    evaluated exactly once (not reused), and the calibrated Brier score there is no worse than climatology."""
    if cal is None or spec is None or spec.task != "classifier" or final_block.get("status") != "EVALUATED_ONCE":
        return False
    m = final_block.get("metrics", {}).get(spec.name, {})
    b_cal, b_clim = m.get("brier_calibrated"), m.get("brier_climatology")
    if b_cal is None or b_clim is None or not (math.isfinite(b_cal) and math.isfinite(b_clim)):
        return False
    return bool(b_cal <= b_clim)


def _prediction_frame(data: Prepared, idx: np.ndarray, variant: str, block: str, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "variant": variant,
            "block": block,
            "symbol": data.panel["symbol"].to_numpy()[idx],
            "decision_ts": pd.to_datetime(data.decision_ns[idx], utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "fwd_ret": data.fwd[idx],
            "score": arrays["score"],
        }
    )
    if "p_cal" in arrays:
        df["p_raw"] = arrays["p_raw"]
        df["p_cal"] = arrays["p_cal"]
    return df


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_model(run_dir: Path, model: FittedModel, cal: Calibrator | None) -> dict[str, str]:
    files: dict[str, str] = {}
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "variant": model.spec.describe(),
        "features": list(model.feature_spec.columns),
        "feature_spec": {
            "return_horizons": list(model.feature_spec.return_horizons),
            "vol_window": model.feature_spec.vol_window,
            "volume_window": model.feature_spec.volume_window,
            "ma_window": model.feature_spec.ma_window,
        },
        "mu_up": model.mu_up,
        "mu_dn": model.mu_dn,
        "base_rate": model.base_rate,
        "calibrator": cal.to_json() if cal is not None else None,
    }
    if model.booster is not None:
        if model.spec.library == "lightgbm":
            path = model_dir / "model.lgb.txt"
            model.booster.save_model(str(path))
        else:
            path = model_dir / "model.cbm"
            model.booster.save_model(str(path))
        files[f"model/{path.name}"] = _sha256_file(path)
        meta["model_file"] = path.name
    write_json(model_dir / "model_meta.json", meta)
    files["model/model_meta.json"] = _sha256_file(model_dir / "model_meta.json")
    return files


# --------------------------------------------------------------------------------------------
# Signal-engine adapter
# --------------------------------------------------------------------------------------------


@dataclass
class TrainedSignalModel:
    """Adapter so a walk-forward model can score candidates in ``QuantSignalEngine``.

    ``probability_calibrated`` is True only when the probabilities pass through a calibrator fitted on a
    held-out block *and* the calibrated Brier score was evaluated on the frozen final test.
    """

    model: FittedModel
    calibrator: Calibrator | None
    horizon_bars: int
    model_version: str
    strategy_id: str
    calibration_evaluated: bool = False

    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction | None:
        row = latest_feature_row(sym_bars, self.model.feature_spec)
        if row is None:
            return None
        X = row.reshape(1, -1)
        raw = float(self.model.raw(X)[0])
        if not math.isfinite(raw):
            return None
        if self.model.spec.task == "classifier":
            p = float(self.calibrator.transform(np.array([raw]))[0]) if self.calibrator is not None else raw
            exp = float(self.model.expected_return(np.array([p]))[0]) * 10_000.0
            calibrated = self.calibrator is not None and self.calibration_evaluated
            return ModelPrediction(expected_return_bps=exp, score=p, probability_up=p, probability_calibrated=calibrated)
        if self.model.spec.task == "regressor":
            return ModelPrediction(expected_return_bps=raw * 10_000.0, score=raw, probability_up=None,
                                   probability_calibrated=False)
        return None  # the rule baseline is already the engine's default score

    @classmethod
    def from_output(cls, out: WalkForwardOutput) -> TrainedSignalModel | None:
        if out.final_model is None or out.final_model.spec.task == "rule":
            return None
        p = out.payload
        return cls(
            model=out.final_model,
            calibrator=out.final_calibrator,
            horizon_bars=int(p["horizon_bars"]),
            model_version=f"ml:{out.final_model.spec.name}:{p['run_id']}:{str(p['config_hash'])[:12]}",
            strategy_id=f"ml_{out.final_model.spec.name}",
            calibration_evaluated=bool(p.get("probability_calibrated_for_signals")),
        )


def load_trained_model(run_dir: Path | str) -> TrainedSignalModel:
    """Load the selected final model of a walk-forward run (LightGBM text / CatBoost cbm + JSON meta)."""
    rd = Path(run_dir)
    payload = json.loads((rd / "results.json").read_text())
    meta = json.loads((rd / "model" / "model_meta.json").read_text())
    v = meta["variant"]
    spec = VariantSpec(v["name"], v["library"], v["task"], dict(v["params"]))
    fs_d = meta["feature_spec"]
    fs = FeatureSpec(
        return_horizons=tuple(int(h) for h in fs_d["return_horizons"]), vol_window=int(fs_d["vol_window"]),
        volume_window=int(fs_d["volume_window"]), ma_window=int(fs_d["ma_window"]),
    )
    booster: Any = None
    if spec.task != "rule":
        path = rd / "model" / str(meta["model_file"])
        if spec.library == "lightgbm":
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(path))
        elif spec.task == "classifier":
            from catboost import CatBoostClassifier

            booster = CatBoostClassifier()
            booster.load_model(str(path))
        else:
            from catboost import CatBoostRegressor

            booster = CatBoostRegressor()
            booster.load_model(str(path))
    model = FittedModel(spec, fs, booster=booster, mu_up=float(meta["mu_up"]), mu_dn=float(meta["mu_dn"]),
                        base_rate=float(meta["base_rate"]))
    cal = Calibrator.from_json(meta["calibrator"]) if meta.get("calibrator") else None
    return TrainedSignalModel(
        model=model,
        calibrator=cal,
        horizon_bars=int(payload["horizon_bars"]),
        model_version=f"ml:{spec.name}:{payload['run_id']}:{str(payload['config_hash'])[:12]}",
        strategy_id=f"ml_{spec.name}",
        calibration_evaluated=bool(payload.get("probability_calibrated_for_signals")),
    )


def fold_overlap_report(plan: SplitPlan, decision_ns: np.ndarray, label_end_ns: np.ndarray) -> list[dict[str, Any]]:
    """Diagnostic used by tests/CLI: max training label end vs test start per fold (must be < start)."""
    rows: list[dict[str, Any]] = []
    folds: Sequence[Fold] = plan.folds
    for f in folds:
        tr = f.train_idx
        rows.append(
            {
                "fold": f.fold,
                "max_train_label_end_ns": int(label_end_ns[tr].max()) if len(tr) else None,
                "test_start_ns": f.test_start_ns,
                "max_train_decision_ns": int(decision_ns[tr].max()) if len(tr) else None,
            }
        )
    return rows

