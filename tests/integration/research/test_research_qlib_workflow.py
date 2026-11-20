"""Runs the REAL isolated Qlib v0.9.8 environment (subprocess, no network). Skips when it is not installed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import services.research.qlib_bridge as qb
from services.research.frames import prepare_bars
from services.research.synthetic import make_synthetic_bars

pytestmark = pytest.mark.skipif(not qb.QLIB_PYTHON.exists(), reason="isolated qlib env not installed")


def test_qlib_reads_our_binary_provider_exactly(tmp_path: Path) -> None:
    df = make_synthetic_bars(("AAA", "BBB"), timeframe="1Day", n_sessions=40, seed=9)
    qb.convert_to_qlib(df, tmp_path / "prov")
    script = (
        "import json,sys,qlib\n"
        "from qlib.constant import REG_US\n"
        "from qlib.data import D\n"
        "qlib.init(provider_uri=sys.argv[1], region=REG_US, kernels=1)\n"
        "f = D.features(['AAA','BBB'], ['$close','$volume'], freq='day')\n"
        "inst = sorted(set(f.index.get_level_values(0)))\n"
        "print(json.dumps({'n': len(f), 'close': f['$close'].round(4).tolist(), 'inst': inst}))\n"
    )
    proc = subprocess.run([str(qb.QLIB_PYTHON), "-I", "-c", script, str(tmp_path / "prov")], capture_output=True,
                          text=True, timeout=300, check=False, env=qb._child_env(), cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    bars = prepare_bars(df)
    expected = np.concatenate([bars.loc[bars["symbol"] == s, "close_f"].to_numpy() for s in ("AAA", "BBB")])
    assert out["n"] == 80 and out["inst"] == ["AAA", "BBB"]
    np.testing.assert_allclose(out["close"], np.round(expected.astype(np.float32), 4), atol=1e-3)


def test_minimal_lightgbm_workflow_runs_in_isolated_env(tmp_path: Path) -> None:
    df = make_synthetic_bars(("AAA", "BBB", "CCC"), timeframe="1Day", n_sessions=360, seed=4)
    man = qb.run_qlib_workflow(df, tmp_path, run_id="qlib_it", horizon=5)
    assert man["status"] == "OK", man.get("reason")
    assert man["qlib_version"] == "0.9.8"
    assert man["is_simulated"] is True and "NO EDGE CLAIMED" in man["no_edge_note"]
    res = json.loads((tmp_path / "qlib_it" / "qlib_results.json").read_text())
    assert res["model"]["class"] == "qlib.contrib.model.gbdt.LGBModel"
    assert res["segment_rows_after_dropna_label"]["train"] > 0 and res["test_rows_scored"] > 0
    assert set(res["recorder_metrics"]) >= {"IC", "Rank IC"}
    seg = res["segments"]
    assert seg["train"][1] < seg["valid"][0] < seg["valid"][1] < seg["test"][0]
    assert (tmp_path / "qlib_it" / "pred_test.csv").exists()
    assert not (tmp_path / "qlib_it" / "private").exists()
    # deterministic: same data + seed -> same predictions
    man2 = qb.run_qlib_workflow(df, tmp_path, run_id="qlib_it2", horizon=5)
    assert man2["status"] == "OK"
    a = (tmp_path / "qlib_it" / "pred_test.csv").read_text()
    b = (tmp_path / "qlib_it2" / "pred_test.csv").read_text()
    assert a == b
