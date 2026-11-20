"""Walk-forward suite: purge/embargo, selection (noise -> baseline, planted edge -> model), registry, predictions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from packages.contracts.models import PredictionSet
from services.research.costs import CostModel
from services.research.models.predict import PredictionService
from services.research.models.registry import (
    TARGET_ORDER,
    ModelRegistry,
    list_registry_versions,
    load_registry,
    load_registry_dir,
)
from services.research.models.walkforward import TrainResult
from services.research.signals import QuantSignalEngine

from .test_research_models_support import noise_bars, planted_bars, strip_ids, train_and_save

TRADE_TARGETS = ("ret_5m", "ret_15m", "ret_60m", "rank")


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]:
    exp = tmp_path_factory.mktemp("exp_planted")
    bars = planted_bars()
    res, reg = train_and_save(bars, exp / "models")
    return res, reg, bars, exp


@pytest.fixture(scope="module")
def noise(tmp_path_factory: pytest.TempPathFactory) -> tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]:
    exp = tmp_path_factory.mktemp("exp_noise")
    bars = noise_bars(seed=11)
    res, reg = train_and_save(bars, exp / "models")
    return res, reg, bars, exp


def _asof_in_window(bars: pd.DataFrame, minute: int) -> datetime:
    """A decision time ``minute`` bars after the open of the last session (inside the frozen test)."""
    sb = bars.loc[bars["symbol"] == bars["symbol"].iloc[0]].reset_index(drop=True)
    day = sb["start_ts"].dt.tz_convert("America/New_York").dt.date
    last = sb.loc[day == day.iloc[-1]].reset_index(drop=True)
    out = pd.Timestamp(last["available_at_ts"].iloc[minute]).to_pydatetime()
    assert isinstance(out, datetime)
    return out


# ---------------------------------------------------------------- splits


def test_purge_and_embargo_on_suite_dataset(planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]) -> None:
    res = planted[0]
    ds, plan = res.dataset, res.plan
    final = set(plan.final_test_idx.tolist())
    uniq = np.unique(ds.decision_ns)
    embargo = res.meta["walk_forward"]["embargo_steps"]
    assert embargo == 60
    for f in plan.folds:
        train = np.union1d(f.fit_idx, f.calib_idx)
        assert not final.intersection(train.tolist()) and not final.intersection(f.test_idx.tolist())
        # purge: every training label (the LONGEST label of the row) ends before the test window starts
        assert ds.label_end_ns[train].max() < f.test_start_ns
        # embargo: at least E distinct decision times between the last training decision and the test window
        gap = np.searchsorted(uniq, f.test_start_ns) - np.searchsorted(uniq, ds.decision_ns[train].max()) - 1
        assert gap >= embargo
        # calibration block is after (and purged against) the fit block
        if len(f.calib_idx):
            assert ds.label_end_ns[f.fit_idx].max() < ds.decision_ns[f.calib_idx].min()
    dev = np.union1d(plan.final_fit_idx, plan.final_calib_idx)
    assert ds.label_end_ns[dev].max() < ds.decision_ns[plan.final_test_idx].min()


# ---------------------------------------------------------------- selection


def test_pure_noise_selects_zero_baseline_and_signals_are_no_trade(
    noise: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path],
) -> None:
    res, reg, bars, _ = noise
    for t in TRADE_TARGETS:
        assert res.meta["selected"][t] == "zero_baseline", (t, res.meta["targets"][t]["candidates"])
    assert res.meta["multiple_testing"]["survivors"] == []
    assert res.meta["multiple_testing"]["family"], "candidates were tested, none survived"
    svc = PredictionService(reg, CostModel())
    asof = _asof_in_window(bars, 150)
    preds = svc.predict(bars, asof, ["AAA", "BBB", "CCC"])
    assert preds
    for p in preds:
        assert all(f.model_id == "zero_baseline" and f.expected_return_bps == 0 for f in p.return_forecasts)
        assert p.rank is None
    report = QuantSignalEngine(CostModel(), model=svc).generate_report(bars, asof, ["AAA", "BBB", "CCC"], "cz_noise")
    assert report.outcome == "NO_TRADE" and report.candidate is None
    assert all("expected return not positive" in s.reason for s in report.signals)


def test_planted_edge_is_selected_and_confirmed(planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]) -> None:
    res, reg, bars, _ = planted
    sel = res.meta["selected"]
    chosen = [t for t in ("ret_15m", "ret_60m") if sel[t] != "zero_baseline"]
    assert chosen, sel
    for t in chosen:
        cand = next(c for c in res.meta["targets"][t]["candidates"] if c["selected"])
        assert cand["net_sharpe"] > 0 and f"{t}:{cand['model_id']}" in res.meta["multiple_testing"]["survivors"]
        assert cand["frozen_test"]["net_sharpe"] > 0, "frozen-test confirmation"
    assert res.meta["walk_forward"]["frozen_test"]["status"] == "EVALUATED_ONCE"
    # inside the planted up-drift window the engine finds a candidate; outside it the forecast is negative
    svc = PredictionService(reg, CostModel())
    engine = QuantSignalEngine(CostModel(), model=svc)
    syms = ["AAA", "BBB", "CCC"]
    inside = engine.generate_report(bars, _asof_in_window(bars, 100), syms, "cz_in")
    outside = engine.generate_report(bars, _asof_in_window(bars, 300), syms, "cz_out")
    assert sel["ret_60m"] != "zero_baseline", "the planted 60-minute drift is learnable (deterministic fixture)"
    assert inside.outcome == "CANDIDATE", inside.no_trade_reason
    assert inside.candidate is not None and inside.candidate.strategy_id == "quant_model_suite_v1"
    assert inside.candidate.expected_return_bps > inside.candidate.est_total_cost_bps
    assert outside.outcome == "NO_TRADE"
    assert engine.model_version.startswith(f"msuite:{reg.registry_version}:")


def test_success_probability_calibration_label(planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]) -> None:
    res, reg, bars, _ = planted
    cal = res.meta["targets"]["success"]["calibration"]
    assert res.meta["selected"]["success"] == "catboost_clf"
    assert res.meta["probability_calibrated"] is True and reg.probability_calibrated is True
    assert cal["method"] in ("isotonic", "sigmoid") and cal["evaluated_on"] == "frozen_test"
    assert cal["brier_calibrated"] <= cal["brier_climatology"]
    assert cal["validation_briers"][cal["method"]] <= cal["validation_briers"]["raw"]
    assert cal["reliability"] and sum(r["n"] for r in cal["reliability"]) == res.meta["walk_forward"]["frozen_test"]["samples"]
    p = PredictionService(reg, CostModel()).predict(bars, _asof_in_window(bars, 100), ["AAA"])[0]
    assert p.probability_calibrated and p.calibration_method == cal["method"] and p.success_probability is not None
    assert p.success_horizon_minutes == 60


def test_noise_success_probability_is_not_served(noise: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]) -> None:
    res, reg, bars, _ = noise
    assert res.meta["selected"]["success"] == "climatology"
    assert reg.probability_calibrated is False
    p = PredictionService(reg, CostModel()).predict(bars, _asof_in_window(bars, 100), ["AAA"])[0]
    assert p.success_probability is None and p.probability_calibrated is False


# ---------------------------------------------------------------- registry


def test_registry_report_matches_api_shape(planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]) -> None:
    _, reg, _, exp = planted
    loaded = load_registry(exp)
    assert loaded is not None and loaded.registry_version == reg.registry_version
    rep = loaded.report()
    for k in ("registry_version", "trained_at", "dataset_version", "is_simulated", "selection_metric", "targets",
              "walk_forward", "multiple_testing", "notes"):
        assert k in rep
    assert rep["is_simulated"] is True and rep["selection_metric"] == "net_sharpe"
    assert [t["target"] for t in rep["targets"]] == list(TARGET_ORDER)
    for t in rep["targets"]:
        assert sum(c["selected"] for c in t["candidates"]) == 1
        for c in t["candidates"]:
            assert set(c) >= {"model_id", "net_sharpe", "net_return", "turnover", "hit_rate", "brier", "ic", "n_test",
                              "selected", "rejected_reason"}
    succ = next(t for t in rep["targets"] if t["target"] == "success")
    assert set(succ["calibration"]) >= {"method", "brier_raw", "brier_calibrated", "reliability"}
    assert set(rep["walk_forward"]) >= {"folds", "purge_minutes", "embargo_minutes", "frozen_test"}
    assert set(rep["walk_forward"]["frozen_test"]) >= {"start", "end", "used"}
    assert set(rep["multiple_testing"]) >= {"method", "alpha", "survivors"}
    assert any("not evidence of edge" in n for n in rep["notes"])
    json.dumps(rep, allow_nan=False)  # strict JSON


def _copy_registry(reg: ModelRegistry, dst_exp: Path) -> Path:
    import shutil

    dst = dst_exp / "models" / reg.registry_version
    shutil.copytree(reg.path, dst)
    return dst


@pytest.mark.parametrize("target", ["artifact", "metric", "selection", "missing"])
def test_registry_tamper_returns_none(planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path], tmp_path: Path,
                                      target: str) -> None:
    reg = planted[1]
    d = _copy_registry(reg, tmp_path)
    assert load_registry(tmp_path) is not None
    if target == "artifact":
        f = next(p for p in sorted((d / "artifacts").iterdir()) if p.suffix in (".txt", ".cbm", ".json"))
        data = bytearray(f.read_bytes())
        data[len(data) // 2] ^= 0x01
        f.write_bytes(bytes(data))
    elif target in ("metric", "selection"):
        payload = json.loads((d / "registry.json").read_text())
        if target == "metric":
            payload["targets"]["ret_60m"]["candidates"][0]["net_sharpe"] = 99.0
        else:
            payload["selected"]["ret_5m"] = "lgbm_reg_5"
        (d / "registry.json").write_text(json.dumps(payload))
    else:
        next(p for p in sorted((d / "artifacts").iterdir())).unlink()
    assert load_registry(tmp_path) is None, "tampered/corrupt registry must fail closed"


def test_registry_selection_of_newest_and_incomplete_dirs(
    planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path], tmp_path: Path,
) -> None:
    reg = planted[1]
    _copy_registry(reg, tmp_path)
    # a newer directory WITHOUT registry.json is an interrupted training run: ignored
    (tmp_path / "models" / "mreg_29991231T235959Z_ffffffffffff" / "artifacts").mkdir(parents=True)
    assert list_registry_versions(tmp_path) == [reg.registry_version]
    assert load_registry(tmp_path) is not None
    # a newer COMPLETE but corrupt registry is not silently replaced by an older one
    bad = tmp_path / "models" / "mreg_29991231T235959Z_eeeeeeeeeeee"
    bad.mkdir()
    (bad / "registry.json").write_text("{not json")
    assert load_registry(tmp_path) is None
    assert load_registry(tmp_path, reg.registry_version) is not None
    assert load_registry(tmp_path, "../../etc") is None
    assert load_registry(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------- predictions


def test_predictions_validate_as_prediction_sets(planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path]) -> None:
    _, reg, bars, _ = planted
    svc = PredictionService(reg, CostModel())
    asof = _asof_in_window(bars, 120)
    preds = svc.predict(bars, asof, ["AAA", "BBB", "CCC"], quote_spreads_bps={"AAA": Decimal("3.5")})
    assert len(preds) == 3
    for p in preds:
        again = PredictionSet.model_validate_json(p.model_dump_json())
        assert again == p
        assert p.registry_version == reg.registry_version and p.is_simulated
        assert p.features_asof <= p.asof == pd.Timestamp(asof).to_pydatetime()
        assert {f.horizon_minutes for f in p.return_forecasts} == {5, 15, 60}
        assert all(isinstance(f.expected_return_bps, Decimal) for f in p.return_forecasts)
        assert isinstance(p.est_cost_bps, Decimal) and p.est_cost_bps > 0
        assert p.vol_forecast_bps is not None and p.vol_forecast_bps > 0
        assert p.universe_size == 3
    ranks = [p.rank for p in preds]
    assert (reg.fitted.rank is None) == all(r is None for r in ranks)
    if reg.fitted.rank is not None:
        assert sorted(r for r in ranks if r is not None) == [1, 2, 3]
    with pytest.raises(ValueError, match="binary float"):
        PredictionSet.model_validate(preds[0].model_dump() | {"est_cost_bps": 1.5})


def test_predictions_unchanged_when_future_data_is_altered(
    planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path],
) -> None:
    _, reg, bars, _ = planted
    svc = PredictionService(reg, CostModel())
    asof = _asof_in_window(bars, 150)
    cutoff = pd.Timestamp(asof)
    base = [strip_ids(p.model_dump(mode="json")) for p in svc.predict(bars, asof, ["AAA", "BBB", "CCC"])]
    assert base
    fut = (bars["available_at_ts"] > cutoff).to_numpy()
    assert fut.sum() > 100
    altered = bars.copy()
    rng = np.random.RandomState(0)
    perm = rng.permutation(int(fut.sum()))
    for col in ("open", "high", "low", "close", "volume"):
        vals = altered.loc[fut, col].to_numpy()[perm]
        altered.loc[fut, col] = vals
    altered.loc[fut, "close"] = [Decimal("0.01")] * int(fut.sum())  # even nonsensical (OHLC-violating) future rows
    junk = altered.loc[fut].head(5).copy()
    junk["open"] = [float("nan")] * len(junk)  # malformed rows published after asof must not break or change anything
    altered = pd.concat([altered, junk], ignore_index=True)
    got = [strip_ids(p.model_dump(mode="json")) for p in svc.predict(altered, asof, ["AAA", "BBB", "CCC"])]
    assert got == base
    # sanity (the test is not vacuous): changing data BEFORE asof does change the predictions
    past = bars.copy()
    last_known = int(np.flatnonzero((past["available_at_ts"] <= cutoff).to_numpy() & (past["symbol"] == "AAA").to_numpy())[-1])
    past.loc[last_known, "close"] = past.loc[last_known, "close"] * Decimal("1.004")
    past.loc[last_known, "high"] = max(past.loc[last_known, "high"], past.loc[last_known, "close"])
    changed = [strip_ids(p.model_dump(mode="json")) for p in svc.predict(past, asof, ["AAA", "BBB", "CCC"])]
    assert changed != base


def test_predict_fails_closed_on_stale_short_or_mismatched_data(
    planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path],
) -> None:
    _, reg, bars, _ = planted
    svc = PredictionService(reg, CostModel())
    asof = _asof_in_window(bars, 150)
    # stale: one symbol stops publishing 30 minutes before asof
    stale = bars.loc[~((bars["symbol"] == "BBB") & (bars["available_at_ts"] > pd.Timestamp(asof) - pd.Timedelta(minutes=30)))]
    got = {p.symbol for p in svc.predict(stale, asof, ["AAA", "BBB", "CCC"])}
    assert got == {"AAA", "CCC"}
    # insufficient history
    short = bars.loc[bars["symbol"] == "AAA"].tail(30)
    assert svc.predict(short, asof, ["AAA"]) == []
    # timeframe mismatch
    other = bars.copy()
    other["timeframe"] = "5Min"
    assert svc.predict(other, asof, ["AAA"]) == []
    # nothing knowable
    assert svc.predict(bars, datetime(2000, 1, 3, 15, 0, tzinfo=UTC), ["AAA"]) == []
    # knowable-but-malformed data fails closed (no prediction), never repaired
    bad = bars.copy()
    bad.loc[10, "open"] = float("nan")
    assert svc.predict(bad, asof, ["AAA", "BBB", "CCC"]) == []
    with pytest.raises(ValueError, match="no 30-minute"):
        PredictionService(reg, CostModel(), signal_horizon_minutes=30)


def test_training_is_deterministic(tmp_path: Path) -> None:
    bars = planted_bars(sessions=8, seed=9)
    r1, g1 = train_and_save(bars, tmp_path / "a" / "models")
    r2, g2 = train_and_save(bars, tmp_path / "b" / "models")
    volatile = {"run_id", "trained_at"}

    def stable(meta: dict[str, Any]) -> str:
        m = {k: v for k, v in meta.items() if k not in volatile}
        return json.dumps(m, sort_keys=True, default=str)

    assert stable(r1.meta) == stable(r2.meta)
    asof = _asof_in_window(bars, 110)
    p1 = [strip_ids(p.model_dump(mode="json")) for p in PredictionService(g1, CostModel()).predict(bars, asof, ["AAA", "BBB"])]
    p2 = [strip_ids(p.model_dump(mode="json")) for p in PredictionService(g2, CostModel()).predict(bars, asof, ["AAA", "BBB"])]
    for a in p1:
        a.pop("registry_version")
    for b in p2:
        b.pop("registry_version")
    assert p1 == p2 and p1


def test_loaded_registry_predicts_like_the_trained_models(
    planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path],
) -> None:
    res, reg, _, _ = planted
    again = load_registry_dir(reg.path)
    assert again is not None
    X = res.dataset.X[res.plan.final_test_idx[:200]]
    for h, (mid, model) in res.fitted.returns.items():
        mid2, model2 = again.fitted.returns[h]
        assert mid == mid2
        np.testing.assert_allclose(model.predict(X), model2.predict(X), rtol=1e-9, atol=1e-9)
    if res.fitted.success is not None:
        assert again.fitted.success is not None
        np.testing.assert_allclose(res.fitted.success.calibrated(X), again.fitted.success.calibrated(X), atol=1e-9)


@pytest.mark.parametrize("edit", ["inference_code", "deployed_vs_selected", "calibration_method"])
def test_registry_semantic_checks_survive_a_recomputed_self_hash(
    planted: tuple[TrainResult, ModelRegistry, pd.DataFrame, Path], tmp_path: Path, edit: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Even a consistently re-hashed registry.json is refused when it no longer matches the code or its artifacts."""
    from services.research.models.registry import _canonical_hash

    reg = planted[1]
    d = _copy_registry(reg, tmp_path)
    payload = json.loads((d / "registry.json").read_text())
    payload.pop("registry_sha256")
    if edit == "inference_code":
        payload["inference_code_hash"] = "inference-src-sha256:000000000000000000000000"
    elif edit == "deployed_vs_selected":
        sel = payload["selected"]
        h = next(h for h in ("15", "60", "5") if sel[f"ret_{h}m"] != "zero_baseline")
        payload["selected"][f"ret_{h}m"] = "zero_baseline"  # claim baseline while a model is deployed
    else:
        assert payload["deployed"]["success"] is not None
        payload["deployed"]["success"]["method"] = "isotonic" if payload["deployed"]["success"]["method"] != "isotonic" \
            else "sigmoid"
    payload["registry_sha256"] = _canonical_hash(payload)
    (d / "registry.json").write_text(json.dumps(payload))
    with caplog.at_level("WARNING", logger="services.research.models.registry"):
        assert load_registry(tmp_path) is None
    expected = {"inference_code": "trained with inference code", "deployed_vs_selected": "disagrees with the recorded",
                "calibration_method": "deployed calibrator disagrees"}[edit]
    assert expected in caplog.text, caplog.text
