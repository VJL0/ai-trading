"""``train-models`` / ``predict`` CLI commands on the explicitly SYNTHETIC in-memory fixture (no store, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from packages.contracts.models import PredictionSet
from services.research.cli import main
from services.research.models.registry import load_registry


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, Any, str]:
    code = main(argv)
    cap = capsys.readouterr()
    out = json.loads(cap.out) if cap.out.strip() else None
    return code, out, cap.err


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("cli_exp")
    mp = pytest.MonkeyPatch()
    import io
    import sys

    buf = io.StringIO()
    mp.setattr(sys, "stdout", buf)
    try:
        code = main(["train-models", "--source", "synthetic", "--out", str(out), "--folds", "3"])
    finally:
        mp.undo()
    assert code == 0
    return out, json.loads(buf.getvalue())


def test_train_models_writes_a_verified_registry(trained: tuple[Path, dict[str, Any]]) -> None:
    out, rep = trained
    assert rep["command"] == "train-models" and rep["is_simulated"] is True
    assert rep["registry_version"].startswith("mreg_")
    assert (out / "models" / rep["registry_version"] / "registry.json").is_file()
    reg = load_registry(out)
    assert reg is not None and reg.registry_version == rep["registry_version"]
    assert set(rep["selected"]) == {"ret_5m", "ret_15m", "ret_60m", "success", "rank", "vol", "regime", "cost"}
    assert rep["walk_forward"]["frozen_test"]["used"] is True
    assert any("not evidence of edge" in n for n in rep["notes"])
    assert rep["multiple_testing"]["method"] == "Benjamini-Hochberg FDR"
    assert "SYNTHETIC" in rep["data"]


def test_predict_emits_point_in_time_prediction_sets(trained: tuple[Path, dict[str, Any]],
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    out, rep = trained
    asof = pd.Timestamp(rep["walk_forward"]["frozen_test"]["end"]) - pd.Timedelta(minutes=90)
    code, res, err = _run(capsys, ["predict", "--source", "synthetic", "--out", str(out), "--asof", asof.isoformat()])
    assert code == 0, err
    assert res["registry_version"] == rep["registry_version"] and res["is_simulated"] is True
    preds = [PredictionSet.model_validate(p) for p in res["predictions"]]
    assert preds and {p.symbol for p in preds} | set(res["no_prediction"]) == {"AAA", "BBB", "CCC"}
    for p in preds:
        assert p.asof == asof.to_pydatetime() and p.features_asof <= p.asof
        assert p.registry_version == rep["registry_version"]
    assert res["signal_engine"]["model_version"].startswith(f"msuite:{rep['registry_version']}:")
    f60 = [f for p in preds for f in p.return_forecasts if f.horizon_minutes == 60]
    assert {f.model_id for f in f60} == {rep["selected"]["ret_60m"]}
    if rep["selected"]["ret_60m"] == "zero_baseline":
        assert res["signal_engine"]["outcome"] == "NO_TRADE"
        assert all(f.expected_return_bps == 0 for f in f60)
    else:
        assert any(f.expected_return_bps != 0 for f in f60)


def test_predict_fails_closed_without_a_valid_registry(trained: tuple[Path, dict[str, Any]], tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    import shutil

    out, rep = trained
    asof = pd.Timestamp(rep["walk_forward"]["frozen_test"]["end"]).isoformat()
    code, res, err = _run(capsys, ["predict", "--source", "synthetic", "--out", str(tmp_path), "--asof", asof])
    assert code == 2 and res is None and "no valid model registry" in err
    shutil.copytree(out / "models", tmp_path / "models")
    reg_json = tmp_path / "models" / rep["registry_version"] / "registry.json"
    payload = json.loads(reg_json.read_text())
    payload["selected"]["ret_60m"] = "lgbm_reg_60"  # attempt to switch a baseline to a model by editing the file
    reg_json.write_text(json.dumps(payload))
    code, res, err = _run(capsys, ["predict", "--source", "synthetic", "--out", str(tmp_path), "--asof", asof])
    assert code == 2 and "fail closed" in err


def test_cli_rejects_bad_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                trained: tuple[Path, dict[str, Any]]) -> None:
    out, _ = trained
    code, _, err = _run(capsys, ["predict", "--source", "synthetic", "--out", str(out), "--asof", "2023-01-13T15:00:00"])
    assert code == 2 and "timezone-aware" in err
    empty = tmp_path / "datasets"
    empty.mkdir()
    code, _, err = _run(capsys, ["train-models", "--source", "store", "--datasets-dir", str(empty),
                                 "--out", str(tmp_path)])
    assert code == 2 and "no stored dataset" in err, err
    code, _, err = _run(capsys, ["train-models", "--source", "synthetic", "--timeframe", "1Day", "--out", str(tmp_path)])
    assert code == 2 and "intraday" in err


class _FakeStore:
    def __init__(self, manifests: list[dict[str, Any]]) -> None:
        self._m = manifests

    def list_manifests(self, include_defective: bool = True) -> list[dict[str, Any]]:
        return [m for m in self._m if include_defective or not m.get("defects_injected")]


def test_default_dataset_for_models_skips_the_lean_sample_and_defects(monkeypatch: pytest.MonkeyPatch,
                                                                       tmp_path: Path,
                                                                       capsys: pytest.CaptureFixture[str]) -> None:
    import services.research.cli as cli_mod

    syn = {"dataset_version": "ds_syn", "kind": "synthetic_fixture", "timeframes": ["1Day", "1Min"]}
    lean = {"dataset_version": "ds_lean", "kind": "lean_sample_import", "timeframes": ["1Day", "1Min"]}
    daily = {"dataset_version": "ds_daily", "kind": "backfill", "timeframes": ["1Day"]}
    bad = {"dataset_version": "ds_bad", "kind": "synthetic_fixture_defects", "timeframes": ["1Min"],
           "defects_injected": 12}
    assert cli_mod.pick_models_manifest(_FakeStore([syn, lean, daily, bad])) == syn
    newer = {"dataset_version": "ds_alpaca", "kind": "backfill", "timeframes": ["1Min"]}
    assert cli_mod.pick_models_manifest(_FakeStore([syn, lean, newer])) == newer
    assert cli_mod.pick_models_manifest(_FakeStore([lean, daily])) is None
    # only unsuitable datasets stored: refuse (never silently train on the LEAN sample or fall back to synthetic)
    monkeypatch.setattr(cli_mod, "_store", lambda d=None: _FakeStore([lean, daily]))
    code, _, err = _run(capsys, ["train-models", "--out", str(tmp_path)])
    assert code == 2 and "no contiguous 1Min dataset" in err
