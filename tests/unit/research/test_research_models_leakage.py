"""Walk-forward leakage controls: the frozen test never influences fitting/selection and is evaluated only once."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from services.research.models.walkforward import TrainResult

from .test_research_models_support import planted_bars, train_and_save

DEV_TARGETS = ("ret_5m", "ret_15m", "ret_60m", "rank", "success", "vol", "regime", "cost")


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    bars = planted_bars(sessions=8, seed=17)
    base_a = tmp_path_factory.mktemp("leak_a") / "models"
    res_a, _ = train_and_save(bars, base_a)
    # rewrite everything published at/after the frozen-test start (mild, OHLC-consistent: no data-quality cut)
    cut = pd.Timestamp(res_a.plan.final_test_start_ns, tz="UTC")
    fut = (pd.to_datetime(bars["available_at_ts"], utc=True) >= cut).to_numpy()
    assert 1000 < fut.sum() < len(bars) // 2
    altered = bars.copy()
    rng = np.random.RandomState(1)
    factor = 1.0 + rng.uniform(-0.003, 0.003, int(fut.sum()))
    for col in ("open", "high", "low", "close"):
        vals = altered.loc[fut, col].to_numpy()
        altered.loc[fut, col] = [Decimal(f"{float(v) * f:.2f}") for v, f in zip(vals, factor, strict=True)]
    hi = altered.loc[fut, ["open", "high", "low", "close"]].max(axis=1)
    lo = altered.loc[fut, ["open", "high", "low", "close"]].min(axis=1)
    altered.loc[fut, "high"] = hi.to_numpy()
    altered.loc[fut, "low"] = lo.to_numpy()
    altered.loc[fut, "volume"] = altered.loc[fut, "volume"].to_numpy()[rng.permutation(int(fut.sum()))]
    res_b, _ = train_and_save(altered, tmp_path_factory.mktemp("leak_b") / "models")
    return {"bars": bars, "a": res_a, "b": res_b, "base_a": base_a}


def _dev(res: TrainResult, target: str) -> dict[str, Any]:
    return {c["model_id"]: c["development"] for c in res.meta["targets"][target]["candidates"]}


def test_frozen_test_data_cannot_change_development_results(runs: dict[str, Any]) -> None:
    a, b = runs["a"], runs["b"]
    np.testing.assert_array_equal(a.plan.final_test_idx, b.plan.final_test_idx)
    for t in DEV_TARGETS:
        assert _dev(a, t) == _dev(b, t), f"development metrics of {t} depend on frozen-test data (leak)"
    assert a.meta["multiple_testing"] == b.meta["multiple_testing"]
    assert a.meta["targets"]["success"]["calibration"]["validation_briers"] == \
        b.meta["targets"]["success"]["calibration"]["validation_briers"]
    # deployed models are fitted on the development period only: identical parameters -> identical predictions
    # (the frozen-test gate may only DEMOTE to the zero baseline, so compare where both deployed the same model)
    X = a.dataset.X[a.plan.dev_idx[:500]]
    compared = 0
    for h, (mid, model) in a.fitted.returns.items():
        mid_b, model_b = b.fitted.returns[h]
        if mid == mid_b and mid != "zero_baseline":
            np.testing.assert_array_equal(model.predict(X), model_b.predict(X))
            compared += 1
    assert compared >= 1, (a.meta["selected"], b.meta["selected"])
    # the test is not vacuous: the frozen-test evaluation itself did see the altered data
    fa = a.meta["targets"]["cost"]["candidates"][0]["frozen_test"]
    fb = b.meta["targets"]["cost"]["candidates"][0]["frozen_test"]
    assert fa is not None and fb is not None and fa != fb


def test_frozen_test_is_evaluated_once_then_fails_closed(runs: dict[str, Any]) -> None:
    a: TrainResult = runs["a"]
    assert a.meta["walk_forward"]["frozen_test"]["status"] == "EVALUATED_ONCE"
    assert any(a.meta["selected"][t] != "zero_baseline" for t in ("ret_15m", "ret_60m")), "positive control"
    again, reg = train_and_save(runs["bars"], runs["base_a"])  # same dataset, same registry base
    ft = again.meta["walk_forward"]["frozen_test"]
    assert ft["status"] == "REFUSED_ALREADY_EVALUATED" and ft["used"] is False
    for t in ("ret_5m", "ret_15m", "ret_60m", "rank"):
        assert again.meta["selected"][t] == "zero_baseline", "unconfirmed models are demoted (fail closed)"
    assert all(reg.fitted.returns[h][0] == "zero_baseline" for h in (5, 15, 60))
    assert reg.fitted.rank is None
    assert again.meta["probability_calibrated"] is False, "calibration was not evaluated on the frozen test"
    assert any("unconfirmed" in n for n in again.meta["notes"])
    base: Path = runs["base_a"]
    assert reg.path.parent == base
    assert len([d for d in base.iterdir() if d.is_dir() and d.name.startswith("mreg_")]) == 2, "both runs registered"
    assert (base / "frozen_test_registry.json").is_file()
    # an explicit re-evaluation is labelled CONTAMINATED and can never confirm (promote) a trading model
    contaminated, _ = train_and_save(runs["bars"], base, allow_final_test_reuse=True)
    ft = contaminated.meta["walk_forward"]["frozen_test"]
    assert ft["status"] == "EVALUATED_REUSED_CONTAMINATED" and ft["used"] is True
    assert all(contaminated.meta["selected"][t] == "zero_baseline" for t in ("ret_5m", "ret_15m", "ret_60m", "rank"))
    assert contaminated.meta["probability_calibrated"] is False
    reported = [c["frozen_test"] for c in contaminated.meta["targets"]["ret_60m"]["candidates"] if c["frozen_test"]]
    assert reported, "contaminated frozen-test metrics are still reported for transparency"


def test_training_rows_never_postdate_their_test_window(runs: dict[str, Any]) -> None:
    a: TrainResult = runs["a"]
    ds = a.dataset
    for f in a.plan.folds:
        for part in (f.fit_idx, f.calib_idx):
            if len(part):
                assert ds.decision_ns[part].max() < f.test_start_ns
                assert ds.label_end_ns[part].max() < f.test_start_ns
    dev = a.plan.dev_idx
    assert ds.label_end_ns[dev].max() < a.plan.final_test_start_ns
    # every row's label ends after its decision (labels are strictly forward-looking from the next open)
    assert (ds.label_end_ns > ds.decision_ns).all()
    for h, ex in ds.exit_ns.items():
        assert (ex > ds.decision_ns).all() and (ex <= ds.label_end_ns).all(), h
