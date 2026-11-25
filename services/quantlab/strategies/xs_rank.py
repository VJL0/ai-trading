"""Family K — cross-sectional learning-to-rank tournament, holding horizons, and cost-aware selection.

Pre-registered controlled comparison (one grid, recorded in full, judged by the same gates as every family):

* **Models** on identical point-in-time features and identical purged walk-forward folds: ridge regression and
  LightGBM regression (the existing baselines of family G) against three explicit ranking objectives — LightGBM
  ``lambdarank``, XGBoost ``rank:pairwise`` and CatBoost ``YetiRank`` (one query group per decision date).
* **Label**: risk-adjusted return relative to the market and the data-inferred sector — the executable forward
  return (open ``t+1`` -> open ``t+1+h``, :func:`services.quantlab.ranking.executable_forward_return`) minus its
  sector-group mean, divided by trailing 63-session volatility, cross-sectionally ranked. Rankers see integer
  relevance grades (deciles) of the same quantity.
* **Horizons** ``h`` of 1, 5 and 20 sessions (label, purge and rebalance interval all follow ``h``).
* **Construction**: ``top_n`` (hold the 20 best-ranked names, keep holdings while inside the top 40 — family G's
  rule) vs ``cost_aware`` (:func:`services.quantlab.portfolio.select_cost_aware`: a name is bought only when the
  lower bound of its calibrated excess edge exceeds its round-trip cost at the intended size; unused slots hold
  SPY). The calibration for each test year is fitted on earlier out-of-sample predictions only
  (:class:`services.quantlab.calibration.BinnedEdge`, Fama-MacBeth standard errors, isotonic in the score).

Features at the close of ``t`` (all trailing, cross-sectionally ranked within the eligible universe): family G's
14 plus short-term reversal, return relative to the inferred sector ETF (21/63), Amihud illiquidity, ADV,
overnight and intraday return components, idiosyncratic volatility, maximum daily return and the last opening
gap. The inferred sector is the sector ETF with the highest trailing 126-session return correlation (> 0.3),
refreshed every 21 sessions from past data only.

Timeline: models predict from ``FIRST_PRED_YEAR`` (each test year trained on labels that completed before it);
portfolios trade from ``TRADE_START_YEAR`` (so cost-aware calibration always has a prior out-of-sample year and
both constructions start together); walk-forward variant selection starts at ``FIRST_TEST_YEAR``.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import BacktestConfig, apply_rebalance_schedule, hold_schedule, performance_metrics, run_weights
from ..calibration import BinnedEdge, ConformalBands, coverage, fit_binned_edge, fit_conformal
from ..costs import EquityCosts
from ..panel import EquityPanel, shift, trailing_mean, trailing_std, trailing_sum
from ..portfolio import round_trip_cost_bps, select_cost_aware
from ..ranking import (
    decision_rows,
    executable_forward_return,
    group_demean,
    ic_summary,
    paired_difference,
    pct_rank,
    quantile_means,
    rank_ic,
    relevance_grades,
    spread_summary,
    top_set_turnover,
)
from ..scanner import SECTOR_ETFS, _corr_last  # pyright: ignore[reportPrivateUsage]
from . import features as F
from .momentum import top_n_weights, universe_mask

FAMILY = "K_xs_rank"
MODELS = ("ridge", "lgbm_reg", "lgbm_lambdarank", "xgb_pairwise", "catboost_yetirank")
RANKERS = ("lgbm_lambdarank", "xgb_pairwise", "catboost_yetirank")
HORIZONS = (1, 5, 20)
CONSTRUCTIONS = ("top_n", "cost_aware")
UNIVERSE = "liquid1000"
N_HOLD = 20
FIRST_PRED_YEAR = 2018
TRADE_START_YEAR = 2019
FIRST_TEST_YEAR = 2020
SAMPLE_EVERY = 5
WARMUP_ROWS = 252
MIN_TRAIN_DATES = 40
"""The first test year (2018) has ~46-50 sampled training dates (one year after the 252-session warm-up)."""
REL_LEVELS = 10
EDGE_BINS = 20
EDGE_Z = 1.0
MIN_EDGE_BPS = 0.0
CONFORMAL_ALPHA = 0.2
PARK_SYMBOL = "SPY"
CAPITAL = 100_000.0
SECTOR_WINDOW = 126
SECTOR_REFRESH = 21
SECTOR_MIN_CORR = 0.3


def variants() -> Iterator[dict[str, Any]]:
    for model, h, cons in itertools.product(MODELS, HORIZONS, CONSTRUCTIONS):
        yield {"model": model, "horizon": h, "construction": cons}


# ------------------------------------------------------------------------------------------- PIT inputs
def inferred_sector_groups(panel: EquityPanel, etfs: tuple[str, ...] = SECTOR_ETFS, *, window: int = SECTOR_WINDOW,
                           every: int = SECTOR_REFRESH, min_corr: float = SECTOR_MIN_CORR) -> tuple[np.ndarray, list[str]]:
    """(T, N) int group id = index into the returned ETF list (-1 = unassigned), refreshed every ``every`` rows
    from the trailing ``window`` returns up to and including that row, carried forward until the next refresh."""
    names = [s for s in etfs if s in set(panel.symbols.tolist())]
    T, N = panel.shape
    out = np.full((T, N), -1, dtype=np.int16)
    if not names:
        return out, names
    r = panel.f["ret_cc"]
    etf_cols = [panel.col(s) for s in names]
    current = np.full(N, -1, dtype=np.int16)
    for t in range(T):
        if t >= window - 1 and (t - (window - 1)) % every == 0:
            seg = r[t - window + 1: t + 1]
            cors = np.stack([_corr_last(seg, seg[:, c], window) for c in etf_cols])
            filled = np.where(np.isfinite(cors), cors, -9.0)
            best = np.argmax(filled, axis=0)
            bestc = filled[best, np.arange(N)]
            current = np.where(bestc > min_corr, best, -1).astype(np.int16)
        out[t] = current
    return out, names


def feature_stack(panel: EquityPanel, mask: np.ndarray, groups: np.ndarray,
                  group_etfs: list[str]) -> tuple[list[str], np.ndarray]:
    """(T, N, K) float32 cross-sectional ranks (centred at 0) of point-in-time features at each close."""
    f = panel.f
    r = f["ret_cc"]
    spy = panel.col("SPY")
    v = f["volume"]
    adv_sh = trailing_mean(v, 20, 15)
    mom21, mom63 = F.momentum(panel, 21), F.momentum(panel, 63)
    etf_cols = np.array([panel.col(s) for s in group_etfs] + [spy], dtype=int)
    gidx = np.where(groups >= 0, groups, len(group_etfs)).astype(int)  # unassigned -> SPY
    ref21 = np.take_along_axis(mom21[:, etf_cols], gidx, axis=1) if len(etf_cols) else mom21[:, [spy]]
    ref63 = np.take_along_axis(mom63[:, etf_cols], gidx, axis=1) if len(etf_cols) else mom63[:, [spy]]
    beta = F.rolling_beta(panel, 252)
    resid = r - shift(beta, 1) * r[:, [spy]]
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud = np.abs(r) / np.where(f["dollar_vol"] > 0, f["dollar_vol"], np.nan)
    raw: dict[str, np.ndarray] = {
        "mom21": mom21, "mom63": mom63, "mom126": F.momentum(panel, 126), "mom252": F.momentum(panel, 252),
        "mom12_1": F.momentum(panel, 252, 21), "vol21": F.realized_vol(panel, 21), "vol63": F.realized_vol(panel, 63),
        "rsi2": F.rsi(panel, 2), "rsi14": F.rsi(panel, 14), "high52": F.high_52w_ratio(panel),
        "dist50": f["adj_close"] / F.sma(panel, 50) - 1, "dist200": f["adj_close"] / F.sma(panel, 200) - 1,
        "beta": beta, "relvol": v / np.where(adv_sh > 0, adv_sh, np.nan),
        "rev5": F.momentum(panel, 5), "sec_rel21": mom21 - ref21, "sec_rel63": mom63 - ref63,
        "illiq21": trailing_mean(amihud, 21, 15), "adv": f["adv20"],
        "overnight21": trailing_sum(np.log1p(f["ret_co"]), 21, 15),
        "intraday21": trailing_sum(np.log1p(f["ret_oc"]), 21, 15),
        "idio_vol63": trailing_std(resid, 63, 50), "max21": np.asarray(
            pd.DataFrame(r).rolling(21, min_periods=15).max(), dtype=float), "gap1": f["ret_co"],
    }
    names = sorted(raw)
    X = np.empty((*panel.shape, len(names)), dtype=np.float32)
    for k, name in enumerate(names):
        X[:, :, k] = (pct_rank(raw[name], mask) - 0.5).astype(np.float32)
    return names, X


# ------------------------------------------------------------------------------------------- models
def _fit(kind: str, X: np.ndarray, y: np.ndarray, rel: np.ndarray, group_sizes: np.ndarray, qid: np.ndarray,
         seed: int, n_jobs: int) -> Any:
    tree = {"n_estimators": 300, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.8,
            "reg_lambda": 5.0}
    if kind == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=10.0).fit(X, y)
    if kind in ("lgbm_reg", "lgbm_lambdarank"):
        import lightgbm as lgb

        common = {**tree, "num_leaves": 31, "min_child_samples": 500, "subsample_freq": 1, "random_state": seed,
                  "n_jobs": n_jobs, "deterministic": True, "force_row_wise": True, "verbose": -1}
        if kind == "lgbm_reg":
            return lgb.LGBMRegressor(**common).fit(X, y)
        return lgb.LGBMRanker(objective="lambdarank", **common).fit(X, rel, group=group_sizes)
    if kind == "xgb_pairwise":
        import xgboost as xgb

        return xgb.XGBRanker(objective="rank:pairwise", max_depth=6, tree_method="hist", random_state=seed,
                             n_jobs=n_jobs, **tree).fit(X, rel, qid=qid)
    if kind == "catboost_yetirank":
        import catboost as cb

        model = cb.CatBoostRanker(loss_function="YetiRank", iterations=300, learning_rate=0.05, depth=6,
                                  l2_leaf_reg=5.0, random_seed=seed, thread_count=n_jobs, verbose=False,
                                  allow_writing_files=False)
        return model.fit(cb.Pool(X, label=rel, group_id=qid))
    raise ValueError(f"unknown model {kind!r}")


@dataclass
class _Cal:
    edge: BinnedEdge
    bands: ConformalBands
    calib_rows: tuple[int, int]


@dataclass
class XsRankLab:
    """Lazily prepared walk-forward scores, calibrated edges and targets for family K (cached per model/horizon)."""

    panel: EquityPanel
    mask_full: np.ndarray | None = None
    n_hold: int = N_HOLD
    first_pred_year: int = FIRST_PRED_YEAR
    trade_start_year: int = TRADE_START_YEAR
    sample_every: int = SAMPLE_EVERY
    warmup_rows: int = WARMUP_ROWS
    seed: int = 7
    n_jobs: int = 4
    costs: EquityCosts = field(default_factory=EquityCosts)
    capital: float = CAPITAL
    fold_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ready = False
        self._scores: dict[tuple[str, int], np.ndarray] = {}
        self._edges: dict[tuple[str, int], tuple[np.ndarray, dict[int, _Cal]]] = {}
        self._targets: dict[str, np.ndarray] = {}
        self.selection_stats: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ preparation
    def prepare(self) -> None:
        if self._ready:
            return
        t0 = time.time()
        full = self.panel
        mask_full = self.mask_full if self.mask_full is not None else universe_mask(full, UNIVERSE)
        ever = set(np.where(mask_full.any(axis=0))[0].tolist())
        extra = {full.col(s) for s in (PARK_SYMBOL, *SECTOR_ETFS) if s in set(full.symbols.tolist())}
        cols = np.array(sorted(ever | extra), dtype=int)
        self.cols = cols
        self.sub = EquityPanel(full.dates, full.symbols[cols], {k: v[:, cols] for k, v in full.f.items()},
                               full.kind[cols], dict(full.meta))
        self.mask = mask_full[:, cols]
        self.park = self.sub.col(PARK_SYMBOL)
        self.mask[:, self.park] = False
        self.groups, self.group_etfs = inferred_sector_groups(self.sub)
        self.names, self.X = feature_stack(self.sub, self.mask, self.groups, self.group_etfs)
        vol63 = F.realized_vol(self.sub, 63) / np.sqrt(252)
        self.excess: dict[int, np.ndarray] = {}
        self.label: dict[int, np.ndarray] = {}
        self.rel: dict[int, np.ndarray] = {}
        for h in HORIZONS:
            fwd = executable_forward_return(self.sub.f, h)
            self.excess[h] = group_demean(fwd, self.mask).astype(np.float32)
            sector_rel = group_demean(fwd, self.mask, self.groups)
            with np.errstate(divide="ignore", invalid="ignore"):
                risk_adj = sector_rel / np.where(vol63 > 0, vol63 * np.sqrt(h), np.nan)
            self.label[h] = (pct_rank(risk_adj, self.mask) - 0.5).astype(np.float32)
            self.rel[h] = relevance_grades(risk_adj, self.mask, REL_LEVELS)
        self.years = np.asarray(self.sub.dates.year)
        self.cost_rt = round_trip_cost_bps(self.sub.f, self.costs, self.capital / self.n_hold).astype(np.float32)
        self._ready = True
        self.prep_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ walk-forward scores
    def _test_years(self) -> list[int]:
        return sorted({int(y) for y in self.years if y >= self.first_pred_year})

    def scores(self, model: str, h: int) -> np.ndarray:
        """Out-of-sample scores (T, M) on the reduced width: each calendar year predicted by a model trained only
        on decision rows whose label window (to the open of t+1+h) closed before that year began."""
        key = (model, h)
        if key in self._scores:
            return self._scores[key]
        self.prepare()
        T, M, K = self.X.shape
        out = np.full((T, M), np.nan, dtype=np.float32)
        for y in self._test_years():
            test_rows = np.where(self.years == y)[0]
            first_test = int(test_rows[0])
            last_train = first_test - h - 2  # label of row t uses the open of t+1+h <= session first_test - 1
            train_rows = np.arange(self.warmup_rows, last_train + 1, self.sample_every)
            if len(train_rows) < MIN_TRAIN_DATES:
                continue
            Xs, ys, rs, qs = [], [], [], []
            for r in train_rows:  # row order = decision time order (required by the ranking objectives)
                ok = self.mask[r] & np.isfinite(self.label[h][r]) & (self.rel[h][r] >= 0) & \
                    np.isfinite(self.X[r]).all(axis=1)
                if ok.sum() < 10:
                    continue
                Xs.append(self.X[r, ok])
                ys.append(self.label[h][r, ok])
                rs.append(self.rel[h][r, ok])
                qs.append(np.full(int(ok.sum()), r, dtype=np.int64))
            if len(Xs) < 20:
                continue
            Xtr = np.concatenate(Xs)
            qid = np.concatenate(qs)
            sizes = np.array([len(q) for q in qs], dtype=np.int64)
            t0 = time.time()
            fitted = _fit(model, Xtr, np.concatenate(ys), np.concatenate(rs), sizes, qid, self.seed, self.n_jobs)
            fit_s = time.time() - t0
            Xte = self.X[test_rows].reshape(-1, K)
            ok_te = np.isfinite(Xte).all(axis=1) & self.mask[test_rows].reshape(-1)
            pred = np.full(len(Xte), np.nan, dtype=np.float32)
            if ok_te.any():
                pred[ok_te] = np.asarray(fitted.predict(Xte[ok_te]), dtype=np.float32)
            out[test_rows] = pred.reshape(len(test_rows), M)
            self.fold_log.append({"model": model, "horizon": h, "test_year": y, "train_rows": len(Xtr),
                                  "train_dates": len(sizes), "last_train_row_date": str(self.sub.dates[
                                      int(train_rows[-1])])[:10], "fit_seconds": round(fit_s, 1)})
        self._scores[key] = out
        return out

    # ------------------------------------------------------------------ calibration
    def calibrated_edge(self, model: str, h: int) -> tuple[np.ndarray, dict[int, _Cal]]:
        """Lower confidence bound (bps over ``h`` sessions) of each name's expected excess executable return, from
        a calibrator fitted on out-of-sample predictions whose labels completed before the year it is applied in."""
        key = (model, h)
        if key in self._edges:
            return self._edges[key]
        sc = self.scores(model, h)
        pct = pct_rank(sc, self.mask)
        ex = self.excess[h]
        T, M = sc.shape
        lcb = np.full((T, M), np.nan, dtype=np.float32)
        cals: dict[int, _Cal] = {}
        first_pred_row = int(np.searchsorted(self.years, self.first_pred_year))
        for y in sorted({int(v) for v in self.years if v >= self.trade_start_year}):
            rows = np.where(self.years == y)[0]
            last = int(rows[0]) - h - 2
            if last <= first_pred_row:
                continue
            cr = np.arange(first_pred_row, last + 1)
            ok = self.mask[cr] & np.isfinite(pct[cr]) & np.isfinite(ex[cr])
            ri, ci = np.where(ok)
            if len(ri) < 100:
                continue
            p_obs, y_obs = pct[cr][ri, ci], ex[cr][ri, ci].astype(float)
            edge = fit_binned_edge(p_obs, y_obs, cr[ri], n_bins=EDGE_BINS, overlap=float(h))
            mu_obs, _ = edge.predict(p_obs)
            bands = fit_conformal(p_obs, y_obs - mu_obs, n_bins=EDGE_BINS, alpha=CONFORMAL_ALPHA)
            lcb[rows] = (edge.lower_bound(pct[rows], EDGE_Z) * 1e4).astype(np.float32)
            cals[y] = _Cal(edge, bands, (int(cr[0]), int(cr[-1])))
        self._edges[key] = (lcb, cals)
        return lcb, cals

    # ------------------------------------------------------------------ targets
    def sub_targets(self, p: dict[str, Any]) -> np.ndarray:
        name = f"{p['model']}|{p['horizon']}|{p['construction']}"
        if name in self._targets:
            return self._targets[name]
        model, h, cons = str(p["model"]), int(p["horizon"]), str(p["construction"])
        sc = self.scores(model, h)
        T = len(self.sub.dates)
        active = self.years >= self.trade_start_year
        elig = self.mask & active[:, None] & np.isfinite(sc)
        reb = hold_schedule(T, h)
        if cons == "top_n":
            # the keep-while-in-top-2n buffer is applied on rebalance rows only, so it protects what is actually held
            w = np.full((T, sc.shape[1]), np.nan)
            w[reb] = top_n_weights(sc[reb], elig[reb], self.n_hold)
        elif cons == "cost_aware":
            lcb, _ = self.calibrated_edge(model, h)
            w, stats = select_cost_aware(lcb, self.cost_rt, elig, self.n_hold, reb, min_edge_bps=MIN_EDGE_BPS,
                                         park_col=self.park, active=active & np.isfinite(lcb).any(axis=1))
            self.selection_stats[name] = stats
        else:
            raise ValueError(f"unknown construction {cons!r}")
        self._targets[name] = w
        return w

    def targets(self, p: dict[str, Any]) -> np.ndarray:
        self.prepare()
        return self.panel.expand(self.sub_targets(p), self.cols)


# ------------------------------------------------------------------------------------------- diagnostics
def _benchmarks(xs: XsRankLab, cfg: BacktestConfig) -> dict[str, np.ndarray]:
    """Passive comparators on the same panel: SPY, equal-weight eligible universe (monthly), top-3 sector-ETF 6-1
    momentum (monthly), cash."""
    sub = xs.sub
    T, M = sub.shape
    spy = np.zeros((T, M))
    spy[:, xs.park] = np.where(np.isfinite(sub.f["adj_close"][:, xs.park]), 1.0, 0.0)
    spy = np.where(np.r_[True, np.diff(spy[:, xs.park]) != 0][:, None], spy, np.nan)
    active = xs.years >= xs.trade_start_year
    elig = xs.mask & active[:, None]
    cnt = elig.sum(axis=1, keepdims=True)
    ew = apply_rebalance_schedule(np.where(elig, 1.0 / np.maximum(cnt, 1), 0.0), hold_schedule(T, 21))
    etf_cols = [sub.col(s) for s in xs.group_etfs]
    sect = np.zeros((T, M))
    if etf_cols:
        mom = F.momentum(sub, 126, 21)
        uni = np.zeros((T, M), dtype=bool)
        uni[:, etf_cols] = np.isfinite(mom[:, etf_cols]) & active[:, None]
        sect = top_n_weights(mom, uni, 3, buffer=1.0)
    sect = apply_rebalance_schedule(sect, hold_schedule(T, 21))
    out = {}
    for name, w in (("SPY buy-and-hold", spy), ("equal-weight universe (monthly)", ew),
                    ("sector-ETF 6-1 momentum top 3 (monthly)", sect)):
        out[name] = run_weights(sub.f, sub.dates, w, cfg).net
    out["cash"] = np.zeros(T - 1)
    return out


def _period_metrics(net: np.ndarray, dates: pd.DatetimeIndex, start_year: int) -> dict[str, Any]:
    keep = np.asarray(dates.year) >= start_year
    m = performance_metrics(np.asarray(net)[keep], dates[keep])
    return {k: m.get(k) for k in ("sharpe", "cagr", "ann_vol", "max_drawdown", "cvar95_daily", "start", "end")}


def diagnostics(xs: XsRankLab, cfg: BacktestConfig | None = None, *, oos_start_year: int = FIRST_TEST_YEAR,
                models: tuple[str, ...] = MODELS, horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
    """Descriptive evidence behind the family verdict (no trials are recorded): rank IC and decile spreads of each
    model's out-of-sample scores, ranking-objective vs regression paired tests, edge calibration and conformal
    coverage per year, cost-aware vs top-N portfolios, and passive benchmarks over the same period."""
    xs.prepare()
    cfg = cfg or BacktestConfig(capital=CAPITAL, execution="open")
    dates = xs.sub.dates
    start = pd.Timestamp(f"{xs.first_pred_year}-01-01")
    per: dict[str, Any] = {}
    ics: dict[tuple[str, int], np.ndarray] = {}
    for h in horizons:
        rows = decision_rows(dates, h, start)
        for model in models:
            sc = xs.scores(model, h)
            ic = rank_ic(sc, xs.excess[h], xs.mask, rows)
            ics[(model, h)] = ic
            qm = quantile_means(sc, xs.excess[h], xs.mask, rows, q=10)
            top_cost = []
            for r in rows:
                ok = xs.mask[r] & np.isfinite(sc[r])
                if ok.sum() >= 50:
                    cut = np.quantile(sc[r, ok], 0.9)
                    top_cost.append(float(np.nanmedian(xs.cost_rt[r, ok & (sc[r] >= cut)])))
            spread = spread_summary(qm, dates[rows])
            cost_med = float(np.median(top_cost)) if top_cost else float("nan")
            _, cals = xs.calibrated_edge(model, h)
            calib = {}
            for y, cal in cals.items():
                yrows = np.where((xs.years == y) & (np.arange(len(dates)) < len(dates) - h - 1))[0]
                pct = pct_rank(sc[yrows], xs.mask[yrows])
                mu, _se = cal.edge.predict(pct)
                lo, hi = cal.bands.interval(pct, mu)
                realized = np.where(xs.mask[yrows], xs.excess[h][yrows], np.nan).astype(float)
                top = np.isfinite(pct) & (pct > 1 - 1.0 / EDGE_BINS)
                calib[y] = {"top_bin_pred_bps": float(cal.edge.mean[-1] * 1e4) if np.isfinite(cal.edge.mean[-1])
                            else None, "top_bin_se_bps": float(cal.edge.se[-1] * 1e4)
                            if np.isfinite(cal.edge.se[-1]) else None,
                            "top_bin_realized_bps": float(np.nanmean(realized[top]) * 1e4) if top.any() else None,
                            "conformal_80": coverage(lo, hi, realized),
                            "calibration_rows": [str(dates[cal.calib_rows[0]])[:10],
                                                 str(dates[cal.calib_rows[1]])[:10]]}
            per[f"{model}|h{h}"] = {
                "model": model, "horizon": h, "rank_ic": ic_summary(ic, dates[rows]),
                "decile_excess_bps": spread, "top_decile_median_round_trip_cost_bps": cost_med,
                "top_decile_minus_cost_bps": (spread.get("top_mean", float("nan")) - cost_med)
                if "top_mean" in spread else None,
                "top20_turnover_per_rebalance": top_set_turnover(sc, xs.mask, rows, xs.n_hold),
                "calibration_by_year": calib,
            }
    paired: dict[str, Any] = {}
    for h in horizons:
        if ("lgbm_reg", h) in ics and ("ridge", h) in ics:
            paired[f"lgbm_reg-ridge|h{h}"] = paired_difference(ics[("lgbm_reg", h)], ics[("ridge", h)])
        for rk in RANKERS:
            if (rk, h) in ics and ("lgbm_reg", h) in ics:
                paired[f"{rk}-lgbm_reg|h{h}"] = paired_difference(ics[(rk, h)], ics[("lgbm_reg", h)])
    portfolios: dict[str, Any] = {}
    for model, h, cons in itertools.product(models, horizons, CONSTRUCTIONS):
        name = f"{model}|{h}|{cons}"
        res = run_weights(xs.sub.f, dates, xs.sub_targets({"model": model, "horizon": h, "construction": cons}), cfg)
        m = res.metrics()
        portfolios[name] = {"oos": _period_metrics(res.net, res.dates, oos_start_year),
                            "full_dev": {k: m.get(k) for k in ("sharpe", "gross_sharpe", "turnover_annual",
                                                               "cost_drag_annual", "avg_exposure", "avg_positions",
                                                               "trades", "max_drawdown")},
                            "selection": xs.selection_stats.get(name)}
    bench = {k: _period_metrics(v, dates[1:], oos_start_year) for k, v in _benchmarks(xs, cfg).items()}
    return {"family": FAMILY, "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "dev_period": [str(dates[0])[:10], str(dates[-1])[:10]], "oos_start_year": oos_start_year,
            "first_pred_year": xs.first_pred_year, "trade_start_year": xs.trade_start_year,
            "universe": UNIVERSE, "n_hold": xs.n_hold, "features": xs.names, "sector_etfs": xs.group_etfs,
            "eligible_names_ever": int(xs.mask.any(axis=0).sum()), "edge_z": EDGE_Z,
            "edge_bins": EDGE_BINS, "conformal_alpha": CONFORMAL_ALPHA, "per_model_horizon": per,
            "paired_ic_tests": paired, "portfolios": portfolios, "benchmarks_oos": bench,
            "fold_log": xs.fold_log, "prep_seconds": getattr(xs, "prep_seconds", None),
            "note": ("Development data only (the protected holdout is never loaded). Rank IC and decile spreads use "
                     "non-overlapping decision dates and the executable open->open excess return; costs are the "
                     "backtester's model at the intended position size. Descriptive, not a validated edge.")}
