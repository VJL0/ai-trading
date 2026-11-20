"""Failure injection for research: broken data, broken isolated env, corrupted artefacts -> fail closed, loudly."""

from __future__ import annotations

import json
import stat
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import services.research.qlib_bridge as qb
from services.research.costs import CostModel
from services.research.frames import BarDataError
from services.research.ml import FrozenTestReuseError, WalkForwardConfig, load_trained_model, run_walk_forward
from services.research.signals import QuantSignalEngine
from services.research.synthetic import make_synthetic_bars


def to_dt(value: object) -> datetime:
    out = pd.Timestamp(value).to_pydatetime()  # type: ignore[arg-type]
    assert isinstance(out, datetime) and out.tzinfo is not None
    return out


def _fake_python(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fakepython"
    exe.write_text("#!/bin/sh\n" + body + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_isolated_env_import_failure_is_blocked_with_exact_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _fake_python(tmp_path, 'echo "ModuleNotFoundError: No module named qlib" >&2; exit 1')
    monkeypatch.setattr(qb, "QLIB_PYTHON", exe)
    man = qb.run_qlib_workflow(make_synthetic_bars(("AAA", "BBB"), n_sessions=200), tmp_path / "out", run_id="q1")
    assert man["status"] == "BLOCKED"
    assert "No module named qlib" in man["reason"]


def test_wrong_qlib_version_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _fake_python(tmp_path, 'echo "0.9.7 4.7.0"')
    monkeypatch.setattr(qb, "QLIB_PYTHON", exe)
    man = qb.run_qlib_workflow(make_synthetic_bars(("AAA", "BBB"), n_sessions=200), tmp_path / "out", run_id="q2")
    assert man["status"] == "BLOCKED" and "0.9.8" in man["reason"]


def test_workflow_crash_is_failed_not_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # passes the version probe, then the "workflow" dies without writing results
    exe = _fake_python(tmp_path, 'case "$2" in -c) echo "0.9.8 4.7.0";; *) echo "Segmentation fault" >&2; exit 139;; esac')
    monkeypatch.setattr(qb, "QLIB_PYTHON", exe)
    man = qb.run_qlib_workflow(make_synthetic_bars(("AAA", "BBB"), n_sessions=200), tmp_path / "out", run_id="q3")
    assert man["status"] == "FAILED" and man["returncode"] == 139
    assert "Segmentation fault" in man["reason"]
    saved = json.loads((tmp_path / "out" / "q3" / "manifest.json").read_text())
    assert saved["status"] == "FAILED"


def test_nan_and_out_of_order_bars_fail_closed() -> None:
    df = make_synthetic_bars(("AAA",), timeframe="1Min", n_sessions=2)
    asof = to_dt(df["available_at_ts"].iloc[500])
    nan = df.copy()
    nan.loc[100, "close"] = float("nan")
    with pytest.raises(BarDataError):
        QuantSignalEngine(CostModel()).generate(nan, asof, ["AAA"], "cz")
    shuffled_avail = df.copy()
    shuffled_avail.loc[200, "available_at_ts"] = shuffled_avail.loc[50, "available_at_ts"]
    with pytest.raises(BarDataError):
        QuantSignalEngine(CostModel()).generate(shuffled_avail, asof, ["AAA"], "cz")


def test_mixed_timeframes_rejected() -> None:
    a = make_synthetic_bars(("AAA",), timeframe="1Min", n_sessions=1)
    b = make_synthetic_bars(("AAA",), timeframe="1Day", n_sessions=5)
    asof = to_dt(b["available_at_ts"].max())
    with pytest.raises(BarDataError, match="timeframe"):
        QuantSignalEngine(CostModel()).generate(pd.concat([a, b], ignore_index=True), asof, ["AAA"], "cz")


def test_corrupted_model_artifacts_raise(tmp_path: Path) -> None:
    bars = make_synthetic_bars(("AAA", "BBB"), n_sessions=300, seed=12)
    cfg = WalkForwardConfig(variants=tuple(v for v in WalkForwardConfig().variants if v.name == "lgbm_classifier"))
    out = run_walk_forward(bars, CostModel(), tmp_path, run_id="wf_c", config=cfg)
    (out.run_dir / "model" / "model.lgb.txt").write_text("garbage")
    with pytest.raises(Exception):  # noqa: B017 - any loader error is acceptable; silently scoring is not
        load_trained_model(out.run_dir)
    (out.run_dir / "model" / "model_meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_trained_model(out.run_dir)


def test_corrupted_frozen_test_registry_refuses_evaluation(tmp_path: Path) -> None:
    bars = make_synthetic_bars(("AAA", "BBB"), n_sessions=300, seed=13)
    (tmp_path / "frozen_test_registry.json").write_text("{corrupt")
    cfg = WalkForwardConfig(variants=WalkForwardConfig().variants[:2])
    out = run_walk_forward(bars, CostModel(), tmp_path, run_id="wf_r", config=cfg)
    assert out.payload["final_test"]["status"] == "REFUSED_REGISTRY_UNREADABLE"
    assert "unreadable" in out.payload["final_test"]["reason"]
    assert out.final_model is None
    assert issubclass(FrozenTestReuseError, RuntimeError)
