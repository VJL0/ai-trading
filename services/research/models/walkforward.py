"""Leakage-free walk-forward training, performance-based model selection and the frozen final test.

``train_suite`` runs, on one intraday bar frame:

1. **Dataset** - ``targets.build_suite_dataset`` (point-in-time features; forward NET return / success / rank / vol /
   execution-cost labels; rows with unknown liquidity or incomplete labels dropped).
2. **Splits** - ``services.research.splits.plan_walk_forward`` (reused): a frozen final test interval (last fraction of
   decision times), expanding walk-forward folds on the development period, purging of every training row whose label
   interval (the LONGEST label of the row) overlaps a test window, an embargo of ``E`` decision steps (default: the
   longest horizon in bars) and a held-out calibration block at the end of every training set.
3. **Development folds** - every candidate is fitted on the fold's training rows only and scored on its test window:
   ``lgbm_reg_h`` / ``ridge_h`` / zero forecast per return horizon, ``lgbm_rank`` / ``momentum_rank`` / zero for
   ranking, ``catboost_clf`` (+ isotonic and sigmoid calibrators fitted on the calibration block) vs climatology,
   HAR-RV / EWMA / blends for volatility, the 2-state Markov model vs the unconditional variance for the regime, the
   spread baseline vs spread baseline + LightGBM residual for execution cost.
4. **Selection** (development OOS only; the frozen test is never consulted): trade targets by NET Sharpe of the
   long-only one-position strategy each candidate implies (``strategy.py``, net of the cost model). Each candidate is a
   hypothesis "mean per-trade net return > 0" (= beats the zero baseline); Benjamini-Hochberg controls the FDR across
   the whole family (``services.research.stats.benjamini_hochberg``, reused). A candidate is eligible only when it
   survives BH, has a positive net Sharpe and at least ``min_trades`` trades; otherwise the selection is the zero
   baseline and the signal adapter returns a zero expected return -> NO_TRADE. The calibrator is chosen by validation
   Brier; vol by QLIKE; regime by QLIKE of the next-block second moment; cost by MAE.
5. **Final models** are refitted on the whole development period (never on the frozen test).
6. **Frozen final test** - evaluated ONCE through ``services.research.ml.FrozenTest`` (reused; registry
   ``<base_dir>/frozen_test_registry.json``) for the selected models and their baselines. With
   ``require_frozen_confirmation`` (default) a selected trading model whose frozen-test net Sharpe is not > 0, or that
   could not be confirmed because the frozen test was refused or only re-evaluated (``--allow-final-test-reuse``:
   CONTAMINATED, reported but never confirming), is demoted to the zero baseline (fail closed).
   ``probability_calibrated`` is True only for a held-out calibrator chosen on validation folds whose calibrated Brier
   on the once-evaluated frozen test is <= climatology.

Nothing here is evidence of edge on synthetic or small samples; every registry says so.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..costs import CostModel
from ..frames import SegmentPolicy, is_intraday, last_clean_segment, prepare_bars
from ..ml import FrozenTest, FrozenTestRegistryError, FrozenTestReuseError
from ..provenance import NO_EDGE_NOTE, code_version_hash, config_hash, new_run_id, utc_now_iso
from ..splits import SplitPlan, plan_walk_forward
from ..stats import benjamini_hochberg, brier_score, reliability_table, spearman_ic
from .config import (
    CATBOOST_CLF,
    CLIMATOLOGY,
    COST_MODEL_FULL,
    EWMA,
    HAR_RV,
    LGBM_RANK,
    MARKOV_2STATE,
    MOMENTUM_RANK,
    SPREAD_BASELINE,
    SPREAD_PLUS_RESIDUAL,
    UNCONDITIONAL_VAR,
    ZERO_BASELINE,
    SuiteConfig,
    feature_spec_to_json,
    inference_code_hash,
    lgbm_reg_id,
    model_version,
    models_code_hash,
    ret_target,
    ridge_id,
)
from .estimators import (
    BoosterModel,
    FitError,
    LinearModel,
    MomentumModel,
    SuccessClassifier,
    ZeroModel,
    choose_calibrator,
    fit_lgbm_ranker,
    fit_lgbm_regressor,
    fit_ridge,
    fit_success_classifier,
)
from .regime import RegimeModel, fit_regime, latest_block_index
from .strategy import ANNUALISATION_NOTE, StrategyStats, evaluate_score
from .targets import QuoteSpreads, SuiteDataset, build_suite_dataset, describe_targets
from .vol import VolModel, candidate_forecasts, fit_har, parse_candidate, vol_metrics

COST_RESIDUAL_ID = "cost_residual_lgbm"
NO_TRADE_THRESHOLD = -math.inf  # rankers always hold the top-ranked symbol


@dataclass
class FittedSuite:
    """The deployed (selected) estimators. ``None`` / ZeroModel entries mean "baseline"."""

    returns: dict[int, tuple[str, Any]] = field(default_factory=lambda: {})
    success: SuccessClassifier | None = None
    rank: tuple[str, Any] | None = None
    vol: VolModel | None = None
    regime: RegimeModel | None = None
    cost_residual: BoosterModel | None = None
    cost_selected: str = SPREAD_BASELINE


@dataclass
class TrainResult:
    meta: dict[str, Any]
    fitted: FittedSuite
    dataset: SuiteDataset
    plan: SplitPlan


@dataclass
class _Pool:
    """Out-of-sample predictions of one candidate across development folds."""

    stats: StrategyStats = field(default_factory=StrategyStats)
    preds: list[np.ndarray] = field(default_factory=lambda: [])
    idx: list[np.ndarray] = field(default_factory=lambda: [])
    targets: list[np.ndarray] = field(default_factory=lambda: [])
    skipped: list[str] = field(default_factory=lambda: [])
    folds_ok: int = 0

    def add(self, idx: np.ndarray, pred: np.ndarray, target: np.ndarray | None = None) -> None:
        self.idx.append(np.asarray(idx, dtype=np.int64))
        self.preds.append(np.asarray(pred, dtype=np.float64))
        if target is not None:
            self.targets.append(np.asarray(target, dtype=np.float64))
        self.folds_ok += 1

    def cat(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.preds:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return np.concatenate(self.idx), np.concatenate(self.preds)

    def cat_target(self) -> np.ndarray:
        return np.concatenate(self.targets) if self.targets else np.empty(0)


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ic(pred: np.ndarray, target: np.ndarray) -> float | None:
    return _f(spearman_ic(pred, target)) if len(pred) else None


def _assert_disjoint(plan: SplitPlan) -> None:
    final = set(plan.final_test_idx.tolist())
    for f in plan.folds:
        for name, idx in (("fit", f.fit_idx), ("calib", f.calib_idx), ("test", f.test_idx)):
            if final.intersection(idx.tolist()):
                raise AssertionError(f"fold {f.fold} {name} rows intersect the frozen final test")
    for name, idx in (("final_fit", plan.final_fit_idx), ("final_calib", plan.final_calib_idx)):
        if final.intersection(idx.tolist()):
            raise AssertionError(f"{name} rows intersect the frozen final test")


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat().replace("+00:00", "Z")


def _reliability(p: np.ndarray, y: np.ndarray, bins: int) -> list[dict[str, Any]]:
    return [
        {"bin": r["bin"], "lower": r["lower"], "upper": r["upper"], "pred": r["mean_predicted"],
         "obs": r["observed_frequency"], "n": r["count"]}
        for r in reliability_table(p, y, bins)
    ]


# --------------------------------------------------------------------------------------------
# fold-level fitting
# --------------------------------------------------------------------------------------------


class _Suite:
    """Holds the dataset/config and implements fit + predict for every candidate on given row sets."""

    def __init__(self, ds: SuiteDataset, cfg: SuiteConfig) -> None:
        self.ds = ds
        self.cfg = cfg
        self.momentum = MomentumModel(ds.feature_columns, cfg.feature_spec)

    # ---- return regressors
    def fit_return(self, model_id: str, h: int, train: np.ndarray) -> Any:
        ds, cfg = self.ds, self.cfg
        if model_id == ZERO_BASELINE:
            return ZeroModel()
        if model_id == lgbm_reg_id(h):
            return fit_lgbm_regressor(model_id, ds.X[train], ds.net_bps[h][train], cfg.lgbm_params, cfg.seed)
        if model_id == ridge_id(h):
            return fit_ridge(model_id, ds.X[train], ds.net_bps[h][train], cfg.ridge_alpha)
        raise ValueError(model_id)

    # ---- ranker
    def fit_rank(self, model_id: str, train: np.ndarray) -> Any:
        ds, cfg = self.ds, self.cfg
        if model_id == ZERO_BASELINE:
            return ZeroModel()
        if model_id == MOMENTUM_RANK:
            return self.momentum
        if model_id == LGBM_RANK:
            return fit_lgbm_ranker(model_id, ds.X[train], ds.rank_rel[train], ds.decision_ns[train], cfg.rank_params,
                                   cfg.seed)
        raise ValueError(model_id)

    # ---- classifier
    def fit_success(self, fit: np.ndarray, cal: np.ndarray) -> tuple[SuccessClassifier, dict[str, str]]:
        ds, cfg = self.ds, self.cfg
        return fit_success_classifier(CATBOOST_CLF, ds.X[fit], ds.success[fit], ds.X[cal], ds.success[cal],
                                      cfg.catboost_params, cfg.seed)

    # ---- vol
    def fit_vol(self, train: np.ndarray) -> Any:
        ds = self.ds
        tr = train[ds.vol_valid[train]]
        return fit_har(ds.vol_X[tr], ds.rv_target[tr], ds.vol_spec)

    # ---- regime
    def fit_regime_on(self, train: np.ndarray) -> RegimeModel:
        b = self.ds.blocks
        if len(b) == 0 or len(train) == 0:
            raise FitError("no regime blocks")
        end = int(self.ds.decision_ns[train].max())
        mask = b.avail_ns <= end
        return fit_regime(b.ret_bps[mask], max_obs=self.cfg.regime_max_fit_obs)

    def regime_eval(self, model: RegimeModel | None, train: np.ndarray, start_ns: int, end_ns: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(realised next-block y^2, Markov forecast, unconditional forecast) for blocks inside [start, end]."""
        b = self.ds.blocks
        y = b.ret_bps
        end_train = int(self.ds.decision_ns[train].max())
        y_train = y[b.avail_ns <= end_train]
        uncond = float(np.mean(y_train ** 2)) if len(y_train) else float("nan")
        sel = np.flatnonzero((b.avail_ns >= start_ns) & (b.avail_ns <= end_ns))
        sel = sel[sel + 1 < len(y)]
        # the realised target (next block) must itself be published inside the window: a development fold never
        # scores against data from the embargo gap / frozen test that follows it
        sel = sel[b.avail_ns[sel + 1] <= end_ns]
        realised = y[sel + 1] ** 2
        mk = np.full(len(sel), np.nan) if model is None or len(y) < 2 else next_second_moment(model, y)[sel]
        return realised, mk, np.full(len(sel), uncond)

    # ---- cost
    def fit_cost(self, train: np.ndarray) -> BoosterModel:
        """LightGBM on realised slippage = realised one-way cost - (half spread + fees)."""
        ds, cfg = self.ds, self.cfg
        slip = ds.realized_cost_bps[train] - ds.cost_floor_bps[train]
        return fit_lgbm_regressor(COST_RESIDUAL_ID, ds.cost_X[train], slip, cfg.cost_params, cfg.seed)

    def cost_pred(self, kind: str, model: BoosterModel | None, rows: np.ndarray) -> np.ndarray:
        """One-way cost forecast (bps). Predicted slippage is clamped at >= 0 (never forecast price improvement)."""
        ds = self.ds
        if kind == SPREAD_BASELINE:
            return ds.cost_floor_bps[rows]
        if kind == COST_MODEL_FULL:
            return ds.baseline_cost_bps[rows]
        if kind == SPREAD_PLUS_RESIDUAL and model is not None:
            return ds.cost_floor_bps[rows] + np.maximum(model.predict(ds.cost_X[rows]), 0.0)
        raise ValueError(kind)


def next_second_moment(model: RegimeModel, y: np.ndarray) -> np.ndarray:
    """One-block-ahead ``E[y^2]`` from filtered probabilities: ``sum_j P(S_{t+1}=j|F_t) (sigma2_j + const_j^2)``."""
    f = model.filtered(y)
    nxt = f @ model.transition().T
    const = np.asarray([model.params[model.param_names.index(f"const[{k}]")] for k in (0, 1)], dtype=np.float64)
    return np.asarray(nxt @ (model.sigma2 + const ** 2), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------------


def _trade_metrics(stats: StrategyStats, pred: np.ndarray, target: np.ndarray, h: int) -> dict[str, Any]:
    m = stats.metrics(h)
    m["ic"] = _ic(pred, target)
    return m


def train_suite(
    bars_df: pd.DataFrame,
    cost_model: CostModel,
    base_dir: Any,
    *,
    config: SuiteConfig | None = None,
    quotes: QuoteSpreads | None = None,
    allow_final_test_reuse: bool = False,
    data_adjustments: dict[str, Any] | None = None,
    segment_policy: SegmentPolicy | None = SegmentPolicy(),  # frozen dataclass: safe default
    data_note: str = "",
    run_id: str | None = None,
) -> TrainResult:
    """Train + select the whole suite (module docstring). ``base_dir`` is normally ``experiments/models``."""
    from pathlib import Path

    cfg = config or SuiteConfig()
    base = Path(base_dir)
    run_id = run_id or new_run_id("mtrain")
    started = utc_now_iso()
    bars = prepare_bars(bars_df)
    tf = str(bars["timeframe"].iloc[0])
    if not is_intraday(tf):
        raise ValueError(f"the model suite forecasts minute horizons and needs intraday bars, got {tf}")
    dq: dict[str, Any] = {"applied": "none (segment_policy=None)"}
    if segment_policy is not None:
        bars, dq = last_clean_segment(bars, segment_policy)
    ds = build_suite_dataset(bars, cost_model, cfg, quotes=quotes)
    hb = ds.horizons_bars
    embargo = cfg.embargo_steps if cfg.embargo_steps is not None else max(*hb.values(), ds.vol_spec.horizon_bars)
    plan = plan_walk_forward(
        ds.decision_ns, ds.label_end_ns, keys=ds.keys, n_folds=cfg.n_folds, final_test_fraction=cfg.final_test_fraction,
        embargo_steps=embargo, min_train_fraction=cfg.min_train_fraction, calib_fraction=cfg.calib_fraction,
        min_unique_times=cfg.min_unique_times,
    )
    _assert_disjoint(plan)
    suite = _Suite(ds, cfg)
    H = list(cfg.return_horizons_minutes)
    ret_cands = {h: [lgbm_reg_id(h), ridge_id(h), ZERO_BASELINE] for h in H}
    rank_cands = [LGBM_RANK, MOMENTUM_RANK, ZERO_BASELINE]
    pools: dict[str, _Pool] = {}

    def pool(name: str) -> _Pool:
        return pools.setdefault(name, _Pool())

    success_folds: list[dict[str, Any]] = []
    fold_log: list[dict[str, Any]] = []
    rh = cfg.rank_horizon_minutes
    for fold in plan.folds:
        train = np.union1d(fold.fit_idx, fold.calib_idx)
        test = fold.test_idx
        flog: dict[str, Any] = fold.describe()
        dec = ds.decision_ns[test]
        # -- return regressors (+ zero baseline)
        for h in H:
            for mid in ret_cands[h]:
                key = f"{ret_target(h)}:{mid}"
                try:
                    model = suite.fit_return(mid, h, train)
                except FitError as exc:
                    pool(key).skipped.append(f"fold {fold.fold}: {exc}")
                    continue
                pred = model.predict(ds.X[test])
                evaluate_score(dec, pred, ds.exit_ns[h][test], ds.net_bps[h][test], ds.bar_minutes, pool(key).stats,
                               threshold=cfg.entry_threshold_bps)
                pool(key).add(test, pred, ds.net_bps[h][test])
        # -- rankers (always hold the top-ranked symbol) + zero baseline (never trades)
        for mid in rank_cands:
            key = f"rank:{mid}"
            try:
                model = suite.fit_rank(mid, train)
            except FitError as exc:
                pool(key).skipped.append(f"fold {fold.fold}: {exc}")
                continue
            pred = model.predict(ds.X[test])
            thr = cfg.entry_threshold_bps if mid == ZERO_BASELINE else NO_TRADE_THRESHOLD
            evaluate_score(dec, pred, ds.exit_ns[rh][test], ds.net_bps[rh][test], ds.bar_minutes, pool(key).stats,
                           threshold=thr)
            pool(key).add(test, pred, ds.net_bps[rh][test])
        # -- success classifier: CatBoost on fit block, calibrators on the held-out calibration block
        try:
            clf, notes = suite.fit_success(fold.fit_idx, fold.calib_idx)
            raw = clf.raw(ds.X[test])
            success_folds.append({
                "fold": fold.fold, "idx": test, "raw": raw,
                "cal": {m: c.transform(raw) for m, c in clf.calibrators.items()},
                "base_rate": float(np.mean(ds.success[train])), "notes": notes,
            })
        except FitError as exc:
            pool(f"success:{CATBOOST_CLF}").skipped.append(f"fold {fold.fold}: {exc}")
        # -- volatility (candidates compared on the folds where HAR could be fitted)
        vtest = test[ds.vol_valid[test]]
        try:
            har = suite.fit_vol(train)
        except FitError as exc:
            har = None
            flog["vol"] = f"HAR unavailable ({exc}); fold excluded from the HAR/EWMA comparison"
        if len(vtest):
            fc_all = candidate_forecasts(har, ds.vol_X[vtest], ds.vol_spec, cfg.blend_weights)
            pool(f"vol_all:{EWMA}").add(vtest, fc_all[EWMA], ds.rv_target[vtest])
            if har is not None:
                for name, fc in fc_all.items():
                    pool(f"vol:{name}").add(vtest, fc, ds.rv_target[vtest])
        # -- regime (Markov vs unconditional compared on the folds where the Markov model could be fitted)
        try:
            reg: RegimeModel | None = suite.fit_regime_on(train)
        except FitError as exc:
            reg = None
            flog["regime"] = f"Markov fit unavailable ({exc})"
        realised, mk, unc = suite.regime_eval(reg, train, fold.test_start_ns, fold.test_end_ns)
        if len(realised) and reg is not None:
            pool(f"regime:{MARKOV_2STATE}").add(np.arange(len(realised)), mk, realised)
            pool(f"regime:{UNCONDITIONAL_VAR}").add(np.arange(len(realised)), unc, realised)
        # -- execution cost
        for kind in (SPREAD_BASELINE, COST_MODEL_FULL):
            pool(f"cost:{kind}").add(test, suite.cost_pred(kind, None, test), ds.realized_cost_bps[test])
        try:
            cm = suite.fit_cost(train)
            pool(f"cost:{SPREAD_PLUS_RESIDUAL}").add(test, suite.cost_pred(SPREAD_PLUS_RESIDUAL, cm, test),
                                                     ds.realized_cost_bps[test])
        except FitError as exc:
            pool(f"cost:{SPREAD_PLUS_RESIDUAL}").skipped.append(f"fold {fold.fold}: {exc}")
        fold_log.append(flog)

    n_folds = len(plan.folds)
    targets: dict[str, dict[str, Any]] = {}

    # ---------------- trade targets: metrics + BH family
    trade_metrics: dict[str, dict[str, Any]] = {}
    for h in H:
        for mid in ret_cands[h]:
            key = f"{ret_target(h)}:{mid}"
            p = pool(key)
            _, pred = p.cat()
            trade_metrics[key] = _trade_metrics(p.stats, pred, p.cat_target(), h) | {
                "folds_ok": p.folds_ok, "folds_total": n_folds, "skipped": p.skipped}
    for mid in rank_cands:
        key = f"rank:{mid}"
        p = pool(key)
        _, pred = p.cat()
        trade_metrics[key] = _trade_metrics(p.stats, pred, p.cat_target(), rh) | {
            "folds_ok": p.folds_ok, "folds_total": n_folds, "skipped": p.skipped}
    family = {k: float(v.get("p_value") or 1.0) for k, v in trade_metrics.items()
              if not k.endswith(f":{ZERO_BASELINE}") and v["folds_ok"] == n_folds}
    bh = benjamini_hochberg(family, q=cfg.fdr_q) if family else {}
    survivors = sorted(k for k, v in bh.items() if v["reject"])

    def select_trade(target: str, cands: list[str]) -> tuple[str, dict[str, str]]:
        reasons: dict[str, str] = {}
        eligible: list[tuple[float, str]] = []
        for mid in cands:
            if mid == ZERO_BASELINE:
                continue
            key = f"{target}:{mid}"
            m = trade_metrics[key]
            sharpe = _f(m.get("net_sharpe"))
            if m["folds_ok"] < n_folds:
                reasons[mid] = f"fit failed in {n_folds - m['folds_ok']} fold(s): {'; '.join(m['skipped'])[:300]}"
            elif m["n_trades"] < cfg.min_trades:
                reasons[mid] = f"only {m['n_trades']} development trades (< {cfg.min_trades})"
            elif sharpe is None or sharpe <= 0:
                reasons[mid] = f"net Sharpe {sharpe} <= 0: does not beat the zero baseline net of costs"
            elif not bh.get(key, {}).get("reject"):
                reasons[mid] = (f"not significant after Benjamini-Hochberg (p={m['p_value']:.4g}, "
                                f"q={bh.get(key, {}).get('q_value', 1.0):.4g}, FDR {cfg.fdr_q})")
            else:
                eligible.append((-sharpe, mid))
        if not eligible:
            return ZERO_BASELINE, reasons
        eligible.sort()
        winner = eligible[0][1]
        for _, mid in eligible[1:]:
            reasons[mid] = f"eligible but lower development net Sharpe than {winner}"
        return winner, reasons

    selections: dict[str, str] = {}
    sel_reasons: dict[str, dict[str, str]] = {}
    for h in H:
        selections[ret_target(h)], sel_reasons[ret_target(h)] = select_trade(ret_target(h), ret_cands[h])
    selections["rank"], sel_reasons["rank"] = select_trade("rank", rank_cands)

    # ---------------- success classifier: choose calibrator on validation folds
    success_metrics: dict[str, dict[str, Any]] = {}
    sh = cfg.success_horizon_minutes
    calib_choice: str | None = None
    calib_briers: dict[str, float] = {}
    if len(success_folds) == n_folds:
        idx_all = np.concatenate([f["idx"] for f in success_folds])
        y_all = ds.success[idx_all]
        raw_all = np.concatenate([f["raw"] for f in success_folds])
        methods = [m for m in ("isotonic", "sigmoid") if all(m in f["cal"] for f in success_folds)]
        cal_all = {m: np.concatenate([f["cal"][m] for f in success_folds]) for m in methods}
        calib_choice, calib_briers = choose_calibrator(raw_all, cal_all, y_all)
        p_used = cal_all[calib_choice] if calib_choice else raw_all
        clim_all = np.concatenate([np.full(len(f["idx"]), f["base_rate"]) for f in success_folds])
        st = StrategyStats()
        for f in success_folds:
            pf = f["cal"][calib_choice] if calib_choice else f["raw"]
            ix = f["idx"]
            evaluate_score(ds.decision_ns[ix], pf - 0.5, ds.exit_ns[sh][ix], ds.net_bps[sh][ix], ds.bar_minutes, st)
        sm = st.metrics(sh)
        sm.update({
            "brier": _f(brier_score(p_used, y_all)), "brier_raw": _f(calib_briers.get("raw")),
            "brier_isotonic": _f(calib_briers.get("isotonic")), "brier_sigmoid": _f(calib_briers.get("sigmoid")),
            "brier_climatology": _f(brier_score(clim_all, y_all)), "ic": _ic(p_used, ds.net_bps[sh][idx_all]),
            "classification_hit_rate": _f(np.mean((p_used > 0.5) == (y_all == 1))),
            "calibration_method": calib_choice, "reliability_dev": _reliability(p_used, y_all, cfg.reliability_bins),
            "calibration_notes": [f["notes"] for f in success_folds], "folds_ok": n_folds,
        })
        success_metrics[CATBOOST_CLF] = sm
        success_metrics[CLIMATOLOGY] = {"brier": sm["brier_climatology"], "n_test": len(idx_all), "n_trades": 0,
                                        "net_sharpe": 0.0, "net_return": 0.0, "turnover": 0.0}
        b_model, b_clim = _f(sm["brier"]), _f(sm["brier_climatology"])
        if b_model is not None and b_clim is not None and b_model < b_clim:
            selections["success"] = CATBOOST_CLF
            sel_reasons["success"] = {}
        else:
            selections["success"] = CLIMATOLOGY
            sel_reasons["success"] = {CATBOOST_CLF: f"development Brier {b_model} not below climatology {b_clim}"}
    else:
        selections["success"] = CLIMATOLOGY
        reason = "; ".join(pool(f"success:{CATBOOST_CLF}").skipped)[:300] or "classifier fit failed"
        sel_reasons["success"] = {CATBOOST_CLF: reason}
        success_metrics[CATBOOST_CLF] = {"status": "SKIPPED", "reason": reason}
        success_metrics[CLIMATOLOGY] = {}

    # ---------------- vol / regime / cost (forecast-quality selection)
    min_common = max(1, math.ceil(n_folds / 2))
    vol_metrics_d: dict[str, dict[str, Any]] = {}
    for name, p in sorted((k.split(":", 1)[1], v) for k, v in pools.items() if k.startswith("vol:")):
        _, f_ = p.cat()
        vol_metrics_d[name] = vol_metrics(p.cat_target(), f_) | {"folds_ok": p.folds_ok, "basis": "HAR-available folds"}
    common = {k: v for k, v in vol_metrics_d.items() if v["qlike"] is not None and v["folds_ok"] >= min_common}
    sel_reasons["vol"] = {}
    if common:
        selections["vol"] = min(common, key=lambda k: (common[k]["qlike"], k))
        for k in vol_metrics_d:
            if k != selections["vol"]:
                sel_reasons["vol"][k] = f"higher development QLIKE than {selections['vol']}"
    elif pool(f"vol_all:{EWMA}").folds_ok:
        selections["vol"] = EWMA
        p = pool(f"vol_all:{EWMA}")
        _, f_ = p.cat()
        vol_metrics_d[EWMA] = vol_metrics(p.cat_target(), f_) | {"folds_ok": p.folds_ok, "basis": "all folds"}
        for k in vol_metrics_d:
            if k != EWMA:
                sel_reasons["vol"][k] = f"HAR could be fitted in fewer than {min_common} development folds"
        sel_reasons["vol"].setdefault(HAR_RV, f"HAR could be fitted in fewer than {min_common} development folds")
    else:
        selections["vol"] = "none"

    regime_metrics: dict[str, dict[str, Any]] = {}
    for name in (MARKOV_2STATE, UNCONDITIONAL_VAR):
        p = pool(f"regime:{name}")
        _, f_ = p.cat()
        regime_metrics[name] = vol_metrics(p.cat_target(), f_, scale=1.0) | {"folds_ok": p.folds_ok}
    mk_q, un_q = regime_metrics[MARKOV_2STATE]["qlike"], regime_metrics[UNCONDITIONAL_VAR]["qlike"]
    if (mk_q is not None and un_q is not None and mk_q < un_q
            and regime_metrics[MARKOV_2STATE]["folds_ok"] >= min_common):
        selections["regime"], sel_reasons["regime"] = MARKOV_2STATE, {}
    else:
        selections["regime"] = UNCONDITIONAL_VAR
        sel_reasons["regime"] = {MARKOV_2STATE: f"development QLIKE {mk_q} not below the unconditional variance {un_q} "
                                 f"on {regime_metrics[MARKOV_2STATE]['folds_ok']} fold(s) (need >= {min_common})"}

    cost_metrics: dict[str, dict[str, Any]] = {}
    for name in (SPREAD_BASELINE, COST_MODEL_FULL, SPREAD_PLUS_RESIDUAL):
        p = pool(f"cost:{name}")
        _, pr = p.cat()
        cost_metrics[name] = _cost_metrics(pr, p.cat_target()) | {"folds_ok": p.folds_ok, "skipped": p.skipped}
    cost_ok = {k: v for k, v in cost_metrics.items() if v["mae_bps"] is not None and v["folds_ok"] == n_folds}
    selections["cost"] = min(cost_ok, key=lambda k: (cost_ok[k]["mae_bps"], k)) if cost_ok else SPREAD_BASELINE
    sel_reasons["cost"] = {k: (f"higher development MAE than {selections['cost']}" if k in cost_ok else
                               f"fit failed: {'; '.join(cost_metrics[k]['skipped'])[:200]}")
                           for k in cost_metrics if k != selections["cost"]}

    # ---------------- final models on the whole development period
    final_train = np.union1d(plan.final_fit_idx, plan.final_calib_idx)
    fitted = FittedSuite()
    demoted: dict[str, str] = {}
    for h in H:
        t = ret_target(h)
        mid = selections[t]
        try:
            fitted.returns[h] = (mid, suite.fit_return(mid, h, final_train))
        except FitError as exc:
            demoted[t] = f"final fit of {mid} failed: {exc}"
            selections[t] = ZERO_BASELINE
            fitted.returns[h] = (ZERO_BASELINE, ZeroModel())
    if selections["rank"] != ZERO_BASELINE:
        try:
            fitted.rank = (selections["rank"], suite.fit_rank(selections["rank"], final_train))
        except FitError as exc:
            demoted["rank"] = f"final fit failed: {exc}"
            selections["rank"] = ZERO_BASELINE
    final_clf: SuccessClassifier | None = None
    if selections["success"] == CATBOOST_CLF:
        try:
            final_clf, final_cal_notes = suite.fit_success(plan.final_fit_idx, plan.final_calib_idx)
            if calib_choice is not None and calib_choice not in final_clf.calibrators:
                demoted["success_calibration"] = (f"chosen calibrator {calib_choice} could not be fitted on the final "
                                                  f"calibration block: {final_cal_notes.get(calib_choice)}")
                final_clf.method = None
            else:
                final_clf.method = calib_choice
            fitted.success = final_clf
        except FitError as exc:
            demoted["success"] = f"final fit failed: {exc}"
            selections["success"] = CLIMATOLOGY
    if selections["vol"] != "none":
        fam, w = parse_candidate(selections["vol"])
        try:
            har_final = suite.fit_vol(final_train) if fam != EWMA else None
            fitted.vol = VolModel(selections["vol"], w, har_final, ds.vol_spec)
        except FitError as exc:
            demoted["vol"] = f"final HAR fit failed ({exc}); falling back to EWMA"
            selections["vol"] = EWMA
            fitted.vol = VolModel(EWMA, 0.0, None, ds.vol_spec)
    regime_final: RegimeModel | None = None
    if selections["regime"] == MARKOV_2STATE:
        try:
            regime_final = suite.fit_regime_on(final_train)
            fitted.regime = regime_final
        except FitError as exc:
            demoted["regime"] = f"final fit failed: {exc}"
            selections["regime"] = UNCONDITIONAL_VAR
    if selections["cost"] == SPREAD_PLUS_RESIDUAL:
        try:
            fitted.cost_residual = suite.fit_cost(final_train)
        except FitError as exc:
            demoted["cost"] = f"final fit failed: {exc}"
            selections["cost"] = COST_MODEL_FULL
    fitted.cost_selected = selections["cost"]

    # ---------------- frozen final test: evaluated once for the selected models + baselines
    ft_idx = plan.final_test_idx
    registry_path = base / "frozen_test_registry.json"
    frozen = FrozenTest(
        ft_idx, plan.final_test_hash, registry=registry_path, allow_reuse=allow_final_test_reuse,
        interval_ns=(int(ds.decision_ns[ft_idx].min()), int(ds.label_end_ns[ft_idx].max())) if len(ft_idx) else None,
        symbols=sorted(set(ds.panel["symbol"].to_numpy()[ft_idx].astype(str))) if len(ft_idx) else (),
    )
    frozen_block: dict[str, Any] = {
        "hash": plan.final_test_hash,
        "start": _iso(int(ds.decision_ns[ft_idx].min())) if len(ft_idx) else None,
        "end": _iso(int(ds.decision_ns[ft_idx].max())) if len(ft_idx) else None,
        "label_end": _iso(int(ds.label_end_ns[ft_idx].max())) if len(ft_idx) else None,
        "samples": len(ft_idx),
        "registry": "frozen_test_registry.json (in the registry base directory)",
        "status": "NOT_EVALUATED",
        "used": False,
    }

    def _frozen_eval(idx: np.ndarray) -> dict[str, Any]:
        return _evaluate_frozen(suite, fitted, selections, idx, final_train, cfg)

    frozen_metrics: dict[str, Any] = {}
    if len(ft_idx):
        try:
            frozen_metrics = frozen.evaluate_once(_frozen_eval, run_id=run_id, dataset_version=ds.dataset_version)
            frozen_block["status"] = "EVALUATED_REUSED_CONTAMINATED" if frozen.reused_from else "EVALUATED_ONCE"
            frozen_block["used"] = True
            if frozen.reused_from:
                frozen_block["reused_from"] = {k: frozen.reused_from.get(k) for k in ("run_id", "match", "hash")}
        except FrozenTestRegistryError as exc:
            frozen_block.update(status="REFUSED_REGISTRY_UNREADABLE", reason=str(exc))
        except FrozenTestReuseError as exc:
            frozen_block.update(status="REFUSED_ALREADY_EVALUATED", reason=str(exc))
    else:
        frozen_block["reason"] = "empty frozen test"

    # ---------------- frozen-test confirmation gate (fail closed) + calibration label
    evaluated = frozen_block["status"] in ("EVALUATED_ONCE", "EVALUATED_REUSED_CONTAMINATED")
    # only a FIRST look at the frozen test can confirm a trading model; a contaminated re-evaluation is reported but
    # can never promote anything (fail closed)
    confirmable = frozen_block["status"] == "EVALUATED_ONCE"
    gate_targets: dict[str, int | None] = {ret_target(h): h for h in H} | {"rank": None}
    for t, gate_h in gate_targets.items():
        mid = selections[t]
        if mid == ZERO_BASELINE or not cfg.require_frozen_confirmation:
            continue
        fm = frozen_metrics.get(t, {}).get(mid, {})
        sharpe = _f(fm.get("net_sharpe"))
        if not confirmable:
            demoted[t] = f"{mid} unconfirmed: frozen test {frozen_block['status']} (fail closed -> zero baseline)"
        elif sharpe is None or sharpe <= 0:
            demoted[t] = f"{mid} failed frozen-test confirmation (net Sharpe {sharpe} <= 0) -> zero baseline"
        else:
            continue
        sel_reasons[t][mid] = demoted[t]
        selections[t] = ZERO_BASELINE
        if gate_h is None:
            fitted.rank = None
        else:
            fitted.returns[gate_h] = (ZERO_BASELINE, ZeroModel())
    prob_calibrated = False
    if selections["success"] == CATBOOST_CLF and fitted.success is not None:
        fm = frozen_metrics.get("success", {}).get(CATBOOST_CLF, {})
        b_cal, b_clim = _f(fm.get("brier_calibrated")), _f(fm.get("brier_climatology"))
        b_used = _f(fm.get("brier"))
        if evaluated and b_used is not None and b_clim is not None and b_used > b_clim:
            demoted["success"] = f"frozen-test Brier {b_used:.5f} worse than climatology {b_clim:.5f} -> climatology"
            sel_reasons["success"][CATBOOST_CLF] = demoted["success"]
            selections["success"] = CLIMATOLOGY
            fitted.success = None
        else:
            prob_calibrated = bool(
                fitted.success.method is not None and frozen_block["status"] == "EVALUATED_ONCE"
                and b_cal is not None and b_clim is not None and b_cal <= b_clim
            )

    # ---------------- assemble registry metadata
    def cand(target: str, mid: str, metrics: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        sel = selections[target] == mid
        fm = frozen_metrics.get(target, {}).get(mid)
        return {
            "model_id": mid,
            "model_version": model_version(mid, params),
            "params": params,
            "net_sharpe": _f(metrics.get("net_sharpe")),
            "net_return": _f(metrics.get("net_return")),
            "turnover": _f(metrics.get("turnover")),
            "hit_rate": _f(metrics.get("hit_rate")),
            "brier": _f(metrics.get("brier")),
            "ic": _f(metrics.get("ic")),
            "n_test": metrics.get("n_test"),
            "selected": sel,
            "rejected_reason": None if sel else sel_reasons.get(target, {}).get(mid, "baseline / not selected"),
            "development": metrics,
            "frozen_test": fm,
        }

    lgbm_p = dict(cfg.lgbm_params) | {"seed": cfg.seed, "objective": "regression", "target": "net_bps"}
    for h in H:
        t = ret_target(h)
        targets[t] = {
            "target": t, "kind": "return", "horizon_minutes": h, "horizon_bars": hb[h],
            "selected_model": selections[t],
            "candidates": [
                cand(t, lgbm_reg_id(h), trade_metrics[f"{t}:{lgbm_reg_id(h)}"], lgbm_p | {"horizon_minutes": h}),
                cand(t, ridge_id(h), trade_metrics[f"{t}:{ridge_id(h)}"],
                     {"alpha": cfg.ridge_alpha, "standardize": True, "horizon_minutes": h, "target": "net_bps"}),
                cand(t, ZERO_BASELINE, trade_metrics[f"{t}:{ZERO_BASELINE}"], {"forecast": 0}),
            ],
        }
    targets["rank"] = {
        "target": "rank", "kind": "rank", "horizon_minutes": rh, "selected_model": selections["rank"],
        "candidates": [
            cand("rank", LGBM_RANK, trade_metrics[f"rank:{LGBM_RANK}"],
                 dict(cfg.rank_params) | {"seed": cfg.seed, "objective": "lambdarank", "groups": "decision timestamps",
                                          "levels": cfg.rank_levels}),
            cand("rank", MOMENTUM_RANK, trade_metrics[f"rank:{MOMENTUM_RANK}"],
                 {"rule": "ret_L / (rv * sqrt(L))", "L": max(cfg.feature_spec.return_horizons)}),
            cand("rank", ZERO_BASELINE, trade_metrics[f"rank:{ZERO_BASELINE}"], {"forecast": 0}),
        ],
    }
    succ_frozen = frozen_metrics.get("success", {}).get(CATBOOST_CLF) or {}
    dev_s = success_metrics.get(CATBOOST_CLF, {})
    targets["success"] = {
        "target": "success", "kind": "probability", "horizon_minutes": sh, "selected_model": selections["success"],
        "candidates": [
            cand("success", CATBOOST_CLF, dev_s, dict(cfg.catboost_params) | {"seed": cfg.seed,
                                                                             "calibrators": ["isotonic", "sigmoid"]}),
            cand("success", CLIMATOLOGY, success_metrics.get(CLIMATOLOGY, {}), {"forecast": "training base rate"}),
        ],
        "calibration": {
            "method": fitted.success.method if fitted.success is not None else calib_choice,
            "deployed": fitted.success is not None,
            "selection_rule": "lowest pooled validation (development-fold) Brier among raw / isotonic / sigmoid",
            "validation_briers": {k: _f(v) for k, v in calib_briers.items()},
            "brier_raw": _f(succ_frozen.get("brier_raw") if succ_frozen else dev_s.get("brier_raw")),
            "brier_calibrated": _f(succ_frozen.get("brier_calibrated") if succ_frozen else
                                   (dev_s.get("brier") if calib_choice else None)),
            "brier_climatology": _f(succ_frozen.get("brier_climatology") if succ_frozen else
                                    dev_s.get("brier_climatology")),
            "reliability": succ_frozen.get("reliability") or dev_s.get("reliability_dev") or [],
            "evaluated_on": "frozen_test" if succ_frozen else "development_folds",
            "probability_calibrated": prob_calibrated,
            "rule": "probability_calibrated=True only for a held-out calibrator chosen on validation folds whose "
                    "calibrated Brier on the once-evaluated frozen test is <= climatology",
        },
    }
    vs = ds.vol_spec
    targets["vol"] = {
        "target": "vol", "kind": "volatility", "horizon_minutes": cfg.vol_horizon_minutes,
        "selected_model": selections["vol"], "selection_metric": "qlike",
        "har": {"base_bars": vs.base_bars, "lags": list(vs.lags), "ewma_lambda": vs.ewma_lambda,
                "interpretation": f"HAR daily/weekly/monthly = {vs.lags} base periods of {vs.base_bars} bars",
                "base_period": ds.har_base_reason},
        "candidates": [
            {"model_id": k, "model_version": model_version(k, {"har_lags": list(vs.lags), "base_bars": vs.base_bars,
                                                                "ewma_lambda": vs.ewma_lambda}),
             "net_sharpe": None, "net_return": None, "turnover": None, "hit_rate": None, "brier": None, "ic": None,
             "n_test": v.get("n_test"), "qlike": v.get("qlike"), "vol_rmse_bps": v.get("vol_rmse_bps"),
             "selected": k == selections["vol"],
             "rejected_reason": None if k == selections["vol"] else sel_reasons["vol"].get(k),
             "development": v, "frozen_test": frozen_metrics.get("vol", {}).get(k)}
            for k, v in sorted(vol_metrics_d.items())
        ],
    }
    targets["regime"] = {
        "target": "regime", "kind": "regime", "selected_model": selections["regime"], "selection_metric": "qlike",
        "block_minutes": cfg.regime_block_minutes, "probabilities": "FILTERED only (smoothed probabilities forbidden)",
        "observation": "equal-weighted mean intra-session 1-bar log return of the proxy symbols, summed over complete "
                       "session-anchored blocks (bps)",
        "proxy_symbols": ds.regime_symbols,
        "params": regime_final.to_json() if regime_final is not None else None,
        "candidates": [
            {"model_id": k, "model_version": model_version(k, {"block_minutes": cfg.regime_block_minutes,
                                                                "max_fit_obs": cfg.regime_max_fit_obs}),
             "net_sharpe": None, "net_return": None, "turnover": None, "hit_rate": None, "brier": None, "ic": None,
             "n_test": v.get("n_test"), "qlike": v.get("qlike"), "selected": k == selections["regime"],
             "rejected_reason": None if k == selections["regime"] else sel_reasons["regime"].get(k),
             "development": v, "frozen_test": frozen_metrics.get("regime", {}).get(k)}
            for k, v in regime_metrics.items()
        ],
    }
    targets["cost"] = {
        "target": "cost", "kind": "execution_cost", "selected_model": selections["cost"], "selection_metric": "mae_bps",
        "realized_slippage": ("SIMULATED (derived from synthetic bars/quotes; no broker fills)" if ds.is_simulated else
                              "PROXY from historical bars (arrival close vs next open; no broker fills)"),
        "candidates": [
            {"model_id": k, "model_version": model_version(k, dict(cfg.cost_params) if k == SPREAD_PLUS_RESIDUAL else
                                                            {"cost_model": cost_model.version, "kind": k}),
             "net_sharpe": None, "net_return": None, "turnover": None, "hit_rate": None, "brier": None, "ic": None,
             "n_test": v.get("n_test"), "mae_bps": v.get("mae_bps"), "selected": k == selections["cost"],
             "rejected_reason": None if k == selections["cost"] else sel_reasons["cost"].get(k),
             "development": v, "frozen_test": frozen_metrics.get("cost", {}).get(k)}
            for k, v in cost_metrics.items()
        ],
    }

    purge_minutes = max(cfg.max_horizon_minutes, cfg.vol_horizon_minutes)
    notes = [
        NO_EDGE_NOTE,
        "Results on synthetic data are not evidence of edge." if ds.is_simulated else
        "Historical sample data; small samples are not evidence of edge.",
        ANNUALISATION_NOTE,
        "Return models predict NET returns (gross open-to-open return minus the CostModel round trip).",
        "Deployed models are fitted on the development period only; the frozen test is evaluated once and is never "
        "used for fitting or selection (only as a fail-closed confirmation gate).",
        f"HAR-RV components: daily/weekly/monthly = {vs.lags[0]}/{vs.lags[1]}/{vs.lags[2]} base periods of "
        f"{vs.base_bars} bars ({ds.har_base_reason}).",
        "Execution-cost 'realised slippage' is an arrival-price shortfall proxy"
        + (" and is SIMULATED on this synthetic dataset." if ds.is_simulated else " from bars (no broker fills)."),
        "Regime probabilities are FILTERED (Hamilton filter with fixed parameters); smoothed probabilities are never used.",
    ]
    if demoted:
        notes.append("Demotions: " + "; ".join(f"{k}: {v}" for k, v in sorted(demoted.items())))
    if data_note:
        notes.append(f"Data: {data_note}")
    cfg_desc = cfg.describe()
    meta: dict[str, Any] = {
        "run_id": run_id,
        "trained_at": started,
        "dataset_version": ds.dataset_version,
        "is_simulated": ds.is_simulated,
        "data_label": "SIMULATED (synthetic data)" if ds.is_simulated else "HISTORICAL (non-synthetic bars)",
        "timeframe": ds.timeframe,
        "bar_minutes": ds.bar_minutes,
        "symbols": ds.symbols,
        "regime_proxy_symbols": ds.regime_symbols,
        "rows": len(ds),
        "code_hash": code_version_hash(),
        "models_code_hash": models_code_hash(),
        "inference_code_hash": inference_code_hash(),
        "config": cfg_desc,
        "config_hash": config_hash({"config": cfg_desc, "cost_model": cost_model.describe(),
                                    "dataset_version": ds.dataset_version}),
        "cost_model": cost_model.describe(),
        "feature_columns": list(ds.feature_columns),
        "feature_spec": feature_spec_to_json(cfg.feature_spec),
        "targets_definition": describe_targets(cfg, ds),
        "data_quality": dq,
        "data_adjustments": data_adjustments or {"applied": "none supplied"},
        "selection_metric": "net_sharpe",
        "selected": dict(sorted(selections.items())),
        "targets": targets,
        "walk_forward": {
            "folds": n_folds,
            "purge_minutes": purge_minutes,
            "embargo_steps": embargo,
            "embargo_minutes": embargo * ds.bar_minutes,
            "scheme": plan.spec["scheme"],
            "splits": plan.describe(),
            "fold_log": fold_log,
            "frozen_test": frozen_block,
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg FDR",
            "alpha": cfg.fdr_q,
            "hypothesis": "H0: mean per-trade net return <= 0 (net Sharpe <= zero baseline), one-sided t-test",
            "family": sorted(family),
            "survivors": survivors,
            "details": bh,
        },
        "probability_calibrated": prob_calibrated,
        "notes": notes,
    }
    return TrainResult(meta=meta, fitted=fitted, dataset=ds, plan=plan)


def _cost_metrics(pred: np.ndarray, realised: np.ndarray) -> dict[str, Any]:
    if len(pred) == 0:
        return {"mae_bps": None, "rmse_bps": None, "bias_bps": None, "n_test": 0}
    err = pred - realised
    return {"mae_bps": _f(np.mean(np.abs(err))), "rmse_bps": _f(np.sqrt(np.mean(err ** 2))),
            "bias_bps": _f(np.mean(err)), "n_test": len(pred)}


def _evaluate_frozen(suite: _Suite, fitted: FittedSuite, selections: dict[str, str], idx: np.ndarray,
                     final_train: np.ndarray, cfg: SuiteConfig) -> dict[str, Any]:
    """Metrics on the frozen final test for the SELECTED models and their baselines (called exactly once)."""
    ds = suite.ds
    out: dict[str, Any] = {}
    dec = ds.decision_ns[idx]

    def trade(pred_fn: Callable[[np.ndarray], np.ndarray], h: int, thr: float) -> dict[str, Any]:
        st = StrategyStats()
        pred = pred_fn(ds.X[idx])
        evaluate_score(dec, pred, ds.exit_ns[h][idx], ds.net_bps[h][idx], ds.bar_minutes, st, threshold=thr)
        return _trade_metrics(st, pred, ds.net_bps[h][idx], h)

    for h, (mid, model) in fitted.returns.items():
        t = ret_target(h)
        res = {ZERO_BASELINE: trade(ZeroModel().predict, h, cfg.entry_threshold_bps)}
        if mid != ZERO_BASELINE:
            res[mid] = trade(model.predict, h, cfg.entry_threshold_bps)
        out[t] = res
    rh = cfg.rank_horizon_minutes
    res_r = {ZERO_BASELINE: trade(ZeroModel().predict, rh, cfg.entry_threshold_bps)}
    if fitted.rank is not None:
        res_r[fitted.rank[0]] = trade(fitted.rank[1].predict, rh, NO_TRADE_THRESHOLD)
    out["rank"] = res_r
    # success
    y = ds.success[idx]
    clim = float(np.mean(ds.success[final_train])) if len(final_train) else 0.5
    succ: dict[str, Any] = {CLIMATOLOGY: {"brier": _f(brier_score(np.full(len(y), clim), y)), "n_test": len(y)}}
    if fitted.success is not None:
        raw = fitted.success.raw(ds.X[idx])
        p = fitted.success.calibrated(ds.X[idx])
        sh = cfg.success_horizon_minutes
        st = StrategyStats()
        evaluate_score(dec, p - 0.5, ds.exit_ns[sh][idx], ds.net_bps[sh][idx], ds.bar_minutes, st)
        succ[CATBOOST_CLF] = st.metrics(sh) | {
            "brier": _f(brier_score(p, y)),
            "brier_raw": _f(brier_score(raw, y)),
            "brier_calibrated": _f(brier_score(p, y)) if fitted.success.method else None,
            "brier_climatology": _f(brier_score(np.full(len(y), clim), y)),
            "calibration_method": fitted.success.method,
            "reliability": _reliability(p, y, cfg.reliability_bins),
            "ic": _ic(p, ds.net_bps[sh][idx]),
            "n_test": len(y),
        }
    out["success"] = succ
    # vol: selected + EWMA reference
    v_idx = idx[ds.vol_valid[idx]]
    vres: dict[str, Any] = {}
    if fitted.vol is not None and len(v_idx):
        vres[fitted.vol.selected] = vol_metrics(ds.rv_target[v_idx], fitted.vol.forecast(ds.vol_X[v_idx]))
        ew = VolModel(EWMA, 0.0, None, ds.vol_spec)
        vres.setdefault(EWMA, vol_metrics(ds.rv_target[v_idx], ew.forecast(ds.vol_X[v_idx])))
    out["vol"] = vres
    # regime
    start, end = int(dec.min()), int(dec.max())
    realised, mk, unc = suite.regime_eval(fitted.regime, final_train, start, end)
    rres: dict[str, Any] = {UNCONDITIONAL_VAR: vol_metrics(realised, unc, scale=1.0)}
    if fitted.regime is not None and len(realised):
        rres[MARKOV_2STATE] = vol_metrics(realised, mk, scale=1.0)
        b = ds.blocks
        bi = latest_block_index(b.avail_ns, dec)
        pt = fitted.regime.p_turbulent(b.ret_bps) if len(b) >= 2 else np.full(len(b), np.nan)
        known = bi >= 0
        rres[MARKOV_2STATE]["fraction_turbulent"] = _f(np.mean(pt[bi[known]] > 0.5)) if known.any() else None
    out["regime"] = rres
    # cost
    out["cost"] = {SPREAD_BASELINE: _cost_metrics(suite.cost_pred(SPREAD_BASELINE, None, idx), ds.realized_cost_bps[idx])}
    if fitted.cost_selected != SPREAD_BASELINE:
        out["cost"][fitted.cost_selected] = _cost_metrics(
            suite.cost_pred(fitted.cost_selected, fitted.cost_residual, idx), ds.realized_cost_bps[idx])
    return out


__all__ = [
    "COST_RESIDUAL_ID",
    "FittedSuite",
    "LinearModel",
    "TrainResult",
    "next_second_moment",
    "train_suite",
]
