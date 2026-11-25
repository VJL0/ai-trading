"""Family K (learning-to-rank x horizon x cost-aware construction) on a SIMULATED panel: walk-forward-only scores,
no future leakage into features/labels/calibration, executable targets, trial recording and JSON-safe diagnostics.
Synthetic results are never evidence of profitability."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from services.quantlab.panel import EquityPanel
from services.quantlab.pipeline import Lab
from services.quantlab.research import clean_json
from services.quantlab.strategies import xs_rank as K
from tests.unit.quantlab import synth

ETFS = ("SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV")
N_STOCKS = 36


def _panel() -> EquityPanel:
    rng = np.random.default_rng(11)
    syms = {s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 5e7} for s in ETFS}
    for i in range(N_STOCKS):
        syms[f"S{i:02d}"] = {"drift": float(rng.normal(0.0002, 0.0006)), "vol": 0.02, "price": 50, "volume": 3e6}
    raw, adj = synth.bars(syms, n=2600, start="2016-01-04")
    return EquityPanel.build(raw, adj, synth.assets(list(syms), etfs=ETFS), min_obs=50)


@pytest.fixture(scope="module")
def dev() -> EquityPanel:
    return _panel()


def _mask(panel: EquityPanel) -> np.ndarray:
    return panel.universe(min_adv=1e6, min_history=60, kinds=("COMMON_STOCK",))


@pytest.fixture(scope="module")
def xs(dev: EquityPanel) -> K.XsRankLab:
    lab = K.XsRankLab(dev, mask_full=_mask(dev), n_hold=5, n_jobs=1)
    lab.prepare()
    return lab


def test_scores_are_walk_forward_only(xs: K.XsRankLab) -> None:
    sc = xs.scores("ridge", 5)
    years = xs.years
    assert sc.shape == xs.sub.shape
    assert np.isnan(sc[years < K.FIRST_PRED_YEAR]).all()
    assert np.isfinite(sc[years == 2020]).any()
    first_row = {int(y): int(np.searchsorted(years, y)) for y in np.unique(years)}
    for fold in (f for f in xs.fold_log if f["model"] == "ridge" and f["horizon"] == 5):
        last = int(np.searchsorted(xs.sub.dates.values, np.datetime64(fold["last_train_row_date"])))
        assert last + 5 + 1 <= first_row[fold["test_year"]] - 1  # the label's last open precedes the test year


def test_future_data_cannot_change_past_scores(dev: EquityPanel, xs: K.XsRankLab) -> None:
    base = xs.scores("lgbm_lambdarank", 5)
    years = np.asarray(dev.dates.year)
    later = replace(dev, f={k: v.copy() for k, v in dev.f.items()})
    rng = np.random.default_rng(3)
    cut = years >= 2021
    for k in ("ret_cc", "ret_oc", "ret_co"):
        later.f[k][cut] = later.f[k][cut] * rng.uniform(-3, 3, size=later.f[k][cut].shape)
    other = K.XsRankLab(later, mask_full=_mask(dev), n_hold=5, n_jobs=1)
    changed = other.scores("lgbm_lambdarank", 5)
    np.testing.assert_allclose(np.nan_to_num(changed[years == 2020]), np.nan_to_num(base[years == 2020]), atol=1e-6)


def test_sector_groups_and_features_are_point_in_time(dev: EquityPanel, xs: K.XsRankLab) -> None:
    assert xs.group_etfs == ["XLK", "XLF", "XLE", "XLV"]
    assert xs.groups.min() >= -1 and xs.groups.max() < len(xs.group_etfs)
    assert len(xs.names) == 24 and xs.X.dtype == np.float32
    t = 1500
    g2, _ = K.inferred_sector_groups(
        EquityPanel(xs.sub.dates[: t + 1], xs.sub.symbols, {k: v[: t + 1] for k, v in xs.sub.f.items()}, xs.sub.kind, {}))
    np.testing.assert_array_equal(g2[t], xs.groups[t])  # identical with no data after t


def test_calibration_uses_only_earlier_out_of_sample_predictions(xs: K.XsRankLab) -> None:
    for h in (1, 20):
        lcb, cals = xs.calibrated_edge("ridge", h)
        assert cals, "a calibrator per trading year"
        for y, cal in cals.items():
            first = int(np.searchsorted(xs.years, y))
            assert cal.calib_rows[1] + h + 1 <= first - 1
            assert int(xs.years[cal.calib_rows[0]]) >= K.FIRST_PRED_YEAR
        assert np.isnan(lcb[xs.years < K.TRADE_START_YEAR]).all()


@pytest.mark.parametrize("construction", ["top_n", "cost_aware"])
def test_targets_are_long_only_capped_and_start_together(xs: K.XsRankLab, construction: str) -> None:
    w = xs.sub_targets({"model": "ridge", "horizon": 5, "construction": construction})
    rows = ~np.isnan(w).all(axis=1)
    assert rows.sum() == len(range(0, len(w), 5))  # rebalances every h sessions, holds in between
    filled = np.nan_to_num(w)
    assert filled.min() >= 0 and filled.sum(axis=1).max() <= 1 + 1e-9
    assert np.all(filled[xs.years < K.TRADE_START_YEAR] == 0)
    if construction == "cost_aware":
        park = filled[:, xs.park]
        assert park.max() > 0  # unused slots hold SPY
        stocks = np.delete(filled, xs.park, axis=1)
        assert (stocks > 0).sum(axis=1).max() <= xs.n_hold
    full = xs.targets({"model": "ridge", "horizon": 5, "construction": construction})
    assert full.shape == xs.panel.shape


def test_prohibitive_costs_mean_market_only(dev: EquityPanel) -> None:
    from services.quantlab.costs import EquityCosts

    lab = K.XsRankLab(dev, mask_full=_mask(dev), n_hold=5, n_jobs=1, costs=EquityCosts(cost_multiplier=1e4))
    w = np.nan_to_num(lab.sub_targets({"model": "ridge", "horizon": 5, "construction": "cost_aware"}))
    stocks = np.delete(w, lab.park, axis=1)
    assert stocks.max() == 0  # no edge survives costs: never trade a stock
    assert w[lab.years >= 2020, lab.park].max() == pytest.approx(1.0)


def test_all_models_fit_with_query_groups() -> None:
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    groups, size = 80, 50  # >= 2 x min_child_samples (500) after 70 % row subsampling
    X = rng.normal(size=(groups * size, 4)).astype(np.float32)
    y = 0.5 * X[:, 0] + rng.normal(size=len(X))
    rel = np.concatenate([np.floor(pd.Series(y[i * size:(i + 1) * size]).rank(pct=True).to_numpy() * 10 - 1e-12)
                          for i in range(groups)]).astype(np.int32)
    qid = np.repeat(np.arange(groups), size)
    sizes = np.full(groups, size)
    for model in K.MODELS:
        fitted = K._fit(model, X, (rel / 10.0 - 0.5).astype(np.float32), rel, sizes, qid, 7, 1)  # pyright: ignore[reportPrivateUsage]
        pred = np.asarray(fitted.predict(X))
        assert np.isfinite(pred).all()
        assert np.corrcoef(pred, X[:, 0])[0, 1] > 0.3, model  # learns the planted ordering


def test_family_records_trials_and_diagnostics_are_strict_json(tmp_path, dev: EquityPanel, xs: K.XsRankLab) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "datasets").mkdir()
    lab = Lab(tmp_path)
    spec = lab.xs_rank_family(dev, xs)
    assert len(list(K.variants())) == 30 and spec.family == K.FAMILY
    small = replace(spec, variants=[v for v in spec.variants if v["model"] in ("ridge", "lgbm_lambdarank")
                                    and v["horizon"] == 5])
    summary = lab.run_research([small], dev, "synthetic-fp", run_id="ql_k_test")
    fam = next(f for f in summary["families"] if f["family"] == K.FAMILY)
    assert fam["n_trials"] == 4 and isinstance(fam["fail_reasons"], list)
    assert lab.ledger.count(K.FAMILY) == 4
    diag = clean_json(K.diagnostics(xs, models=("ridge", "lgbm_lambdarank"), horizons=(5,)))
    json.dumps(diag, allow_nan=False)
    assert set(diag["per_model_horizon"]) == {"ridge|h5", "lgbm_lambdarank|h5"}
    assert "lgbm_lambdarank-lgbm_reg|h5" not in diag["paired_ic_tests"]  # baseline not computed in this subset
    assert {"SPY buy-and-hold", "cash"} <= set(diag["benchmarks_oos"])
    row = diag["per_model_horizon"]["ridge|h5"]
    assert row["rank_ic"]["n"] > 50 and "conformal_80" in next(iter(row["calibration_by_year"].values()))
    assert lab.equity_family_by_name(K.FAMILY, dev).family == K.FAMILY
