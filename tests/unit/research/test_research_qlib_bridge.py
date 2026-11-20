"""Qlib provider conversion (binary format) and bridge failure handling - no Qlib import in the root venv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import services.research.qlib_bridge as qb
from services.research.frames import aggregate_daily, prepare_bars

from .bar_factory import make_bars


def test_convert_writes_qlib_file_storage_layout(tmp_path: Path) -> None:
    df = make_bars(("AAA", "BBB"), timeframe="1Day", n=30, seed=1)
    df = df.drop(index=df.index[(df["symbol"] == "BBB").to_numpy()][:5])  # BBB starts 5 sessions later
    man = qb.convert_to_qlib(df, tmp_path)
    cal = (tmp_path / "calendars" / "day.txt").read_text().split()
    assert len(cal) == 30 and cal == sorted(cal)
    inst = [line.split("\t") for line in (tmp_path / "instruments" / "all.txt").read_text().splitlines()]
    assert [i[0] for i in inst] == ["AAA", "BBB"]
    assert inst[1][1] == cal[5] and inst[1][2] == cal[-1]
    bars = prepare_bars(df)
    for sym, first in (("AAA", 0), ("BBB", 5)):
        start, vals = qb.read_qlib_feature(tmp_path, sym, "close")
        assert start == first
        expected = bars.loc[bars["symbol"] == sym, "close_f"].to_numpy()
        np.testing.assert_allclose(vals, expected.astype(np.float32), rtol=0, atol=0)
        _, factor = qb.read_qlib_feature(tmp_path, sym, "factor")
        assert (factor == 1.0).all()
    assert man["sessions"] == 30 and man["is_simulated"] is True
    assert set(man["files_sha256"]) >= {"calendars/day.txt", "instruments/all.txt", "features/aaa/close.day.bin"}


def test_intraday_bars_are_aggregated_to_point_in_time_daily(tmp_path: Path) -> None:
    df = make_bars(("AAA",), timeframe="5Min", n=3, seed=2)
    bars = prepare_bars(df)
    daily = aggregate_daily(bars)
    assert len(daily) == 3
    first_day = bars.loc[bars["session_date"] == daily["session_date"].iloc[0]]
    assert daily["open"].iloc[0] == first_day["open"].iloc[0]
    assert daily["close"].iloc[0] == first_day["close"].iloc[-1]
    assert daily["volume"].iloc[0] == sum(first_day["volume"])
    assert daily["available_at_ts"].iloc[0] == first_day["available_at_ts"].iloc[-1]
    man = qb.convert_to_qlib(df, tmp_path)
    assert man["aggregated_to_daily"] is True and man["sessions"] == 3


def test_segments_are_chronological_with_purge_gaps() -> None:
    cal = [f"d{i:04d}" for i in range(300)]
    seg = qb.plan_segments(cal, horizon=5)
    idx = {k: (cal.index(v[0]), cal.index(v[1])) for k, v in seg.items()}
    assert idx["train"][0] == 0
    assert idx["valid"][0] - idx["train"][1] - 1 >= 6
    assert idx["test"][0] - idx["valid"][1] - 1 >= 6
    assert idx["test"][1] <= 300 - 1 - 6  # last test label (H+1 sessions ahead) still inside the data
    with pytest.raises(ValueError, match="too short"):
        qb.plan_segments(cal[:50], horizon=5)


def test_missing_isolated_env_is_reported_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qb, "QLIB_PYTHON", tmp_path / "nope" / "python")
    df = make_bars(("AAA", "BBB"), timeframe="1Day", n=200, seed=3)
    man = qb.run_qlib_workflow(df, tmp_path / "out", run_id="qlib_blocked")
    assert man["status"] == "BLOCKED"
    assert "uv sync --python 3.11" in man["reason"]
    saved = json.loads((tmp_path / "out" / "qlib_blocked" / "manifest.json").read_text())
    assert saved["status"] == "BLOCKED"
    assert (tmp_path / "out" / ".gitignore").read_text().count("qlib_data") == 1


def test_child_environment_carries_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    env = qb._child_env()
    assert "ALPACA_PAPER_SECRET_KEY" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_symbols_not_representable_are_rejected(tmp_path: Path) -> None:
    df = make_bars(("BRK.B",), timeframe="1Day", n=10)
    with pytest.raises(ValueError, match="Qlib instrument"):
        qb.convert_to_qlib(df, tmp_path)
