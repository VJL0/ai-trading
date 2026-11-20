"""ML walk-forward: frozen final test isolation, determinism, calibration, multiple testing, adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from services.research.costs import CostModel
from services.research.features import FeatureSpec, build_panel, symbol_features, symbol_labels
from services.research.frames import prepare_bars
from services.research.ml import (
    DEFAULT_VARIANTS,
    FrozenTest,
    FrozenTestReuseError,
    TrainedSignalModel,
    WalkForwardConfig,
    fit_calibrator,
    load_trained_model,
    run_walk_forward,
    select_variant,
)
from services.research.signals import QuantSignalEngine
from services.research.stats import benjamini_hochberg, brier_score, ic_pvalue, reliability_table

from .bar_factory import make_bars, to_dt


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return make_bars(("AAA", "BBB", "CCC"), timeframe="1Day", n=420, seed=17, dataset_version="synthetic-ml-v1")


def _strip(payload: dict[str, object]) -> str:
    keep = {k: v for k, v in payload.items() if k not in {"run_id", "created_at", "run_dir", "final_test"}}
    return json.dumps(keep, sort_keys=True, default=str)


def test_features_are_causal_and_labels_follow_execution_convention(daily: pd.DataFrame) -> None:
    bars = prepare_bars(daily)
    sb = bars.loc[bars["symbol"] == "AAA"].reset_index(drop=True)
    spec = FeatureSpec()
    full = symbol_features(sb, spec)
    k = 250
    part = symbol_features(sb.iloc[:k].reset_index(drop=True), spec)
    pd.testing.assert_frame_equal(full.iloc[:k].reset_index(drop=True), part)
    lab = symbol_labels(sb, 5, same_session=False)
    i = 100
    opens = sb["open_f"].to_numpy()
    assert lab.loc[i, "entry_idx"] == i + 1  # first bar opening at/after bar i's availability
    assert lab.loc[i, "fwd_ret"] == pytest.approx(opens[i + 6] / opens[i + 1] - 1)
    assert int(lab["label_end_ns"].iloc[i]) > int(lab["decision_ns"].iloc[i])
    panel = build_panel(bars, spec, 5)
    assert np.isfinite(panel[list(spec.columns)].to_numpy()).all()
    assert bool((panel["label_end_ns"].to_numpy() > panel["decision_ns"].to_numpy()).all())


def test_walk_forward_end_to_end_records_frozen_test_and_bh(daily: pd.DataFrame, tmp_path: Path) -> None:
    out = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_a")
    p = out.payload
    assert p["dataset_version"] == "synthetic-ml-v1" and p["is_simulated"] is True
    assert "NO EDGE CLAIMED" in p["no_edge_note"]
    ft = p["final_test"]
    assert ft["status"] == "EVALUATED_ONCE"
    assert ft["final_test_hash"].startswith("sha256:") and ft["final_test_hash"] == p["splits"]["final_test_hash"]
    assert ft["selected_variant"] == p["selection"]["selected_variant"]
    assert set(ft["metrics"]) >= {ft["selected_variant"], "baseline_momentum_tstat"}
    summary = p["development"]["summary"]
    assert set(summary) == {v.name for v in DEFAULT_VARIANTS}
    for row in summary.values():
        assert 0.0 <= row["p_value"] <= 1.0
        assert {"q_value", "reject", "rank"} <= set(row["bh"])
    assert p["multiple_testing"]["family_size"] == len(DEFAULT_VARIANTS)
    # classifier variants carry Brier scores + a reliability table from held-out calibration
    clf = summary["lgbm_classifier"]
    assert clf["pooled_brier_calibrated"] is not None
    assert sum(r["count"] for r in clf["pooled_reliability"]) == clf["n_oos"]
    manifest = json.loads((out.run_dir / "manifest.json").read_text())
    assert manifest["final_test_hash"] == ft["final_test_hash"]
    assert "results.json" in manifest["files_sha256"]
    assert (out.run_dir / "oos_predictions.csv").exists()
    oos = pd.read_csv(out.run_dir / "oos_predictions.csv")
    final_start = pd.Timestamp(ft["final_test_start_ns"], tz="UTC")
    assert (pd.to_datetime(oos["decision_ts"], utc=True) < final_start).all()
    # dev-fold metrics are computed on dev data only
    assert p["splits"]["dev_samples"] + p["splits"]["final_test_samples"] <= p["panel_rows"]


def test_final_test_can_only_be_evaluated_once(daily: pd.DataFrame, tmp_path: Path) -> None:
    run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_1")
    second = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_2")
    assert second.payload["final_test"]["status"] == "REFUSED_ALREADY_EVALUATED"
    assert "wf_1" in second.payload["final_test"]["reason"]
    assert "metrics" not in second.payload["final_test"]
    assert second.payload["probability_calibrated_for_signals"] is False
    third = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_3", allow_final_test_reuse=True)
    assert third.payload["final_test"]["status"] == "EVALUATED_REUSED_CONTAMINATED"
    assert third.payload["probability_calibrated_for_signals"] is False
    registry = json.loads((tmp_path / "frozen_test_registry.json").read_text())
    entry = registry[second.payload["splits"]["final_test_hash"]]
    assert entry["run_id"] == "wf_1"
    assert [h["run_id"] for h in entry["history"]] == ["wf_1", "wf_3"]


def test_changing_the_config_does_not_buy_a_fresh_look_at_the_frozen_period(daily: pd.DataFrame,
                                                                           tmp_path: Path) -> None:
    variants = DEFAULT_VARIANTS[:2]
    first = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_h5",
                             config=WalkForwardConfig(horizon_bars=5, variants=variants))
    assert first.payload["final_test"]["status"] == "EVALUATED_ONCE"
    other = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_h7",
                             config=WalkForwardConfig(horizon_bars=7, variants=variants, n_folds=3))
    # a different horizon gives different sample keys (different hash) over the same data period
    assert other.payload["splits"]["final_test_hash"] != first.payload["splits"]["final_test_hash"]
    ft = other.payload["final_test"]
    assert ft["status"] == "REFUSED_ALREADY_EVALUATED"
    assert "overlapping_interval_same_dataset" in ft["reason"] and "wf_h5" in ft["reason"]
    assert other.final_model is None and other.payload["probability_calibrated_for_signals"] is False
    # a different dataset version over the same calendar is a different frozen test
    relabelled = daily.assign(dataset_version="synthetic-ml-v2")
    fresh = run_walk_forward(relabelled, CostModel(), tmp_path, run_id="wf_v2",
                             config=WalkForwardConfig(horizon_bars=7, variants=variants, n_folds=3))
    assert fresh.payload["final_test"]["status"] == "EVALUATED_ONCE"


def test_frozen_test_object_refuses_second_use(tmp_path: Path) -> None:
    ft = FrozenTest(np.arange(5), "sha256:x", registry=None)
    assert ft.evaluate_once(lambda idx: {"n": len(idx)}, run_id="r", dataset_version="v") == {"n": 5}
    with pytest.raises(FrozenTestReuseError):
        ft.evaluate_once(lambda idx: {}, run_id="r", dataset_version="v")
    bad = tmp_path / "reg.json"
    bad.write_text("{not json")
    with pytest.raises(FrozenTestReuseError, match="unreadable"):
        FrozenTest(np.arange(2), "h", registry=bad).evaluate_once(lambda idx: {}, run_id="r", dataset_version="v")


def test_crashed_evaluation_still_counts_as_a_look(tmp_path: Path) -> None:
    reg = tmp_path / "reg.json"

    def boom(idx: np.ndarray) -> dict[str, object]:
        raise RuntimeError("crash after looking at the frozen test")

    with pytest.raises(RuntimeError):
        FrozenTest(np.arange(4), "sha256:crash", registry=reg).evaluate_once(boom, run_id="r1", dataset_version="v")
    entry = json.loads(reg.read_text())["sha256:crash"]
    assert [h["status"] for h in entry["history"]] == ["STARTED"]  # reserved BEFORE the data was touched
    with pytest.raises(FrozenTestReuseError, match="r1"):
        FrozenTest(np.arange(4), "sha256:crash", registry=reg).evaluate_once(lambda i: {}, run_id="r2",
                                                                              dataset_version="v")
    ok = FrozenTest(np.arange(4), "sha256:other", registry=reg)
    ok.evaluate_once(lambda i: {"n": len(i)}, run_id="r3", dataset_version="v")
    assert json.loads(reg.read_text())["sha256:other"]["history"][0]["status"] == "COMPLETED"


def test_concurrent_runs_cannot_both_get_the_first_look(tmp_path: Path) -> None:
    import threading
    import time

    reg = tmp_path / "reg.json"
    outcomes: list[str] = []
    lock = threading.Lock()

    def slow(idx: np.ndarray) -> dict[str, object]:
        time.sleep(0.3)  # the second contender arrives while the first is still evaluating
        return {"n": len(idx)}

    def contender(run_id: str) -> None:
        ft = FrozenTest(np.arange(3), "sha256:race", registry=reg)
        try:
            ft.evaluate_once(slow, run_id=run_id, dataset_version="v")
            res = "evaluated"
        except FrozenTestReuseError:
            res = "refused"
        with lock:
            outcomes.append(res)

    threads = [threading.Thread(target=contender, args=(f"r{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(outcomes) == ["evaluated", "refused"]


def test_model_selection_never_sees_frozen_test(daily: pd.DataFrame, tmp_path: Path) -> None:
    """Rewriting every bar inside the frozen-test period changes nothing about development or selection."""
    a = run_walk_forward(daily, CostModel(), tmp_path / "a", run_id="wf_sel_a")
    final_start = pd.Timestamp(a.payload["final_test"]["final_test_start_ns"], tz="UTC")
    mutated = daily.copy()
    later = pd.to_datetime(mutated["start_ts"], utc=True) >= final_start
    assert later.sum() > 30
    rng = np.random.default_rng(0)
    factors = np.exp(np.cumsum(rng.normal(0, 0.05, size=int(later.sum()))))
    for col in ("open", "high", "low", "close"):
        vals = list(mutated.loc[later, col])
        mutated.loc[later, col] = [type(v)(f"{float(v) * f:.2f}") for v, f in zip(vals, factors, strict=True)]
    ohlc = [list(mutated.loc[later, x]) for x in ("open", "high", "low", "close")]
    mutated.loc[later, "high"] = [max(row) for row in zip(*ohlc, strict=True)]
    mutated.loc[later, "low"] = [min(row) for row in zip(*ohlc, strict=True)]
    b = run_walk_forward(mutated, CostModel(), tmp_path / "b", run_id="wf_sel_b")
    assert a.payload["splits"]["final_test_hash"] == b.payload["splits"]["final_test_hash"]
    assert json.dumps(a.payload["development"], sort_keys=True) == json.dumps(b.payload["development"], sort_keys=True)
    assert a.payload["selection"] == b.payload["selection"]
    # ... while the frozen-test evaluation itself does depend on that period (the check is not vacuous)
    sel = a.payload["selection"]["selected_variant"]
    assert a.payload["final_test"]["metrics"][sel]["rank_ic"] != b.payload["final_test"]["metrics"][sel]["rank_ic"]


def test_select_variant_uses_dev_summary_only() -> None:
    dev = {"x": {"mean_fold_ic": 0.01}, "y": {"mean_fold_ic": 0.03}, "z": {"mean_fold_ic": None}}
    assert select_variant(dev) == "y"
    assert select_variant({"a": {"mean_fold_ic": 0.02}, "b": {"mean_fold_ic": 0.02}}) == "a"
    assert select_variant({"a": {"mean_fold_ic": None}}) is None


def test_deterministic_with_fixed_seeds(daily: pd.DataFrame, tmp_path: Path) -> None:
    a = run_walk_forward(daily, CostModel(), tmp_path / "a", run_id="wf_det")
    b = run_walk_forward(daily, CostModel(), tmp_path / "b", run_id="wf_det")
    assert _strip(a.payload) == _strip(b.payload)
    assert (a.run_dir / "oos_predictions.csv").read_bytes() == (b.run_dir / "oos_predictions.csv").read_bytes()
    assert json.dumps(a.payload["final_test"]["metrics"], sort_keys=True) == json.dumps(
        b.payload["final_test"]["metrics"], sort_keys=True
    )
    c = run_walk_forward(daily, CostModel(), tmp_path / "c", run_id="wf_det",
                         config=WalkForwardConfig(seed=1))
    lgb_a = a.payload["development"]["summary"]["lgbm_classifier"]["fold_ics"]
    lgb_c = c.payload["development"]["summary"]["lgbm_classifier"]["fold_ics"]
    assert lgb_a != lgb_c  # the seed is actually wired into the learners (bagging / feature fraction)


def test_isotonic_and_sigmoid_calibration_improve_brier_on_miscalibrated_scores() -> None:
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.2, 0.8, size=4000)
    y = (rng.uniform(size=4000) < true_p).astype(np.int64)
    raw = np.clip(true_p**3, 1e-3, 1 - 1e-3)  # monotone but badly miscalibrated
    for method in ("isotonic", "sigmoid"):
        cal, why = fit_calibrator(raw[:2000], y[:2000], method)
        assert cal is not None, why
        p = cal.transform(raw[2000:])
        assert brier_score(p, y[2000:]) < brier_score(raw[2000:], y[2000:]) - 0.01
    cal_none, why = fit_calibrator(raw[:10], y[:10], "isotonic")
    assert cal_none is None and "too small" in why
    cal_one, why1 = fit_calibrator(raw[:100], np.ones(100, dtype=np.int64), "sigmoid")
    assert cal_one is None and "single class" in why1
    table = reliability_table(raw, y, n_bins=5)
    assert sum(r["count"] for r in table) == len(raw)


def test_benjamini_hochberg_matches_scipy_and_textbook_example() -> None:
    ps = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05, "f": 0.5}
    res = benjamini_hochberg(ps, q=0.05)
    assert not any(r["reject"] for r in res.values())  # every p_(k) > k/m * q
    assert all(res[k]["q_value"] == pytest.approx(0.06) for k in "abcde")
    ref = false_discovery_control(np.array(list(ps.values())), method="bh")
    assert [res[k]["q_value"] for k in ps] == pytest.approx(ref.tolist())
    res2 = benjamini_hochberg({"x": 0.001, "y": 0.012, "z": 0.03, "w": 0.9}, q=0.05)
    assert [k for k, r in res2.items() if r["reject"]] == ["x", "y", "z"]  # 0.03 <= 3/4*0.05
    rng = np.random.default_rng(5)
    rand = {f"v{i}": float(p) for i, p in enumerate(rng.uniform(size=25))}
    ref2 = false_discovery_control(np.array(list(rand.values())), method="bh")
    got = benjamini_hochberg(rand, q=0.1)
    assert [got[k]["q_value"] for k in rand] == pytest.approx(ref2.tolist())
    assert [k for k in rand if got[k]["reject"]] == [k for k, q in zip(rand, ref2, strict=True) if q <= 0.1]


def test_ic_pvalue_behaviour() -> None:
    assert ic_pvalue(0.0, 100) == pytest.approx(0.5)
    assert ic_pvalue(0.3, 100) < 0.01
    assert ic_pvalue(0.3, 3) == 1.0  # too few effective observations
    assert ic_pvalue(float("nan"), 100) == 1.0


def test_trained_model_adapter_and_reload(daily: pd.DataFrame, tmp_path: Path) -> None:
    # a one-variant family makes the classifier the selected model deterministically
    cfg = WalkForwardConfig(variants=tuple(v for v in DEFAULT_VARIANTS if v.name == "lgbm_classifier"))
    out = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_adapt", config=cfg)
    assert out.payload["selection"]["selected_variant"] == "lgbm_classifier"
    model = TrainedSignalModel.from_output(out)
    assert model is not None
    assert model.calibration_evaluated == out.payload["probability_calibrated_for_signals"]
    bars = prepare_bars(daily)
    sb = bars.loc[bars["symbol"] == "AAA"].reset_index(drop=True)
    pred = model.predict_latest(sb)
    assert pred is not None and pred.probability_up is not None and 0 < pred.probability_up < 1
    reloaded = load_trained_model(out.run_dir)
    pred2 = reloaded.predict_latest(sb)
    assert pred2 is not None
    assert pred2.probability_up == pytest.approx(pred.probability_up)
    assert pred2.expected_return_bps == pytest.approx(pred.expected_return_bps)
    asof = to_dt(daily["available_at_ts"].max())
    eng = QuantSignalEngine(CostModel(), horizon_minutes=5 * 390, model=model, min_net_edge_bps=0)
    sigs = eng.generate(daily, asof, ["AAA", "BBB", "CCC"], "cz_ml")
    assert sigs and all(s.model_version == model.model_version for s in sigs)
    assert all(s.probability_calibrated == model.calibration_evaluated for s in sigs)
    mismatch = QuantSignalEngine(CostModel(), horizon_minutes=390, model=model, min_net_edge_bps=0)
    sigs2 = mismatch.generate(daily, asof, ["AAA"], "cz_ml")
    assert sigs2 and not sigs2[0].is_candidate and "horizon" in sigs2[0].reason


def test_rule_baseline_selection_yields_no_model_adapter(daily: pd.DataFrame, tmp_path: Path) -> None:
    cfg = WalkForwardConfig(variants=DEFAULT_VARIANTS[:1])
    out = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_rule", config=cfg)
    assert out.payload["selection"]["selected_variant"] == "baseline_momentum_tstat"
    assert TrainedSignalModel.from_output(out) is None
    assert out.payload["probability_calibrated_for_signals"] is False


def test_regressor_probability_never_labelled_calibrated(daily: pd.DataFrame, tmp_path: Path) -> None:
    cfg = WalkForwardConfig(variants=tuple(v for v in DEFAULT_VARIANTS if v.name == "catboost_regressor"))
    out = run_walk_forward(daily, CostModel(), tmp_path, run_id="wf_reg", config=cfg)
    model = TrainedSignalModel.from_output(out)
    assert model is not None
    bars = prepare_bars(daily)
    pred = model.predict_latest(bars.loc[bars["symbol"] == "BBB"].reset_index(drop=True))
    assert pred is not None and pred.probability_up is None and pred.probability_calibrated is False
    assert out.payload["probability_calibrated_for_signals"] is False


def test_intraday_walk_forward_runs(tmp_path: Path) -> None:
    bars = make_bars(("AAA", "BBB"), timeframe="5Min", n=40, seed=8)
    cfg = WalkForwardConfig(horizon_bars=6, variants=DEFAULT_VARIANTS[:2], n_folds=3)
    out = run_walk_forward(bars, CostModel(), tmp_path, run_id="wf_intraday", config=cfg)
    assert out.payload["horizon_minutes"] == 30
    assert out.payload["final_test"]["status"] == "EVALUATED_ONCE"
