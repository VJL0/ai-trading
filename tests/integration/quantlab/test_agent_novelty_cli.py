"""Agent factor experiments through the CLI on a SIMULATED development panel: an existing signal in disguise is
refused before any trial is recorded; an original factor is evaluated and reports its novelty."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

from services.quantlab import cli
from services.quantlab.factors import FAMILY
from services.quantlab.ledger import TrialLedger
from services.quantlab.panel import EquityPanel
from tests.unit.quantlab import synth


def _save_panel(root) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(9)
    syms = {"SPY": {"drift": 0.0003, "vol": 0.01, "price": 300, "volume": 8e7}}
    for i in range(30):
        syms[f"S{i:02d}"] = {"drift": float(rng.normal(0.0002, 0.0006)), "vol": float(rng.uniform(0.01, 0.03)),
                             "price": 50, "volume": 3e6}
    raw, adj = synth.bars(syms, n=1300, start="2019-01-02")
    panel = EquityPanel.build(raw, adj, synth.assets(list(syms)), min_obs=50)
    panel.save(root / "datasets" / "quantlab" / "panels" / "equity_synthetic.npz")


def _run(settings: SimpleNamespace, expression: str, capsys) -> tuple[int, dict]:  # type: ignore[no-untyped-def,type-arg]
    args = argparse.Namespace(spec=json.dumps({"expression": expression, "n": 5}), actor="test-agent")
    code = cli.cmd_experiment(args, settings)  # pyright: ignore[reportArgumentType]
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    return code, json.loads(lines[-1])


def test_redundant_factor_is_refused_and_original_factor_runs(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    _save_panel(tmp_path)
    settings = SimpleNamespace(project_root=tmp_path)
    code, out = _run(settings, "2 * mom(252, 21)", capsys)
    assert code == 3 and out["reason"] == "REDUNDANT_HYPOTHESIS"
    assert out["prior"]["nearest"] == "A:mom_12_1"
    ledger = TrialLedger(tmp_path / "experiments" / "quantlab" / "trials.sqlite3")
    assert ledger.count(FAMILY) == 0  # refused before evaluation: no trial, no deflated-Sharpe cost
    code, out = _run(settings, "rank(vol(21)) - rank(vol(126))", capsys)
    assert code == 0 and out["novelty"]["max_abs_corr"] < 0.9
    assert ledger.count(FAMILY) == 1
    # a later factor that merely re-spells the accepted one is redundant with that earlier agent factor
    code, out = _run(settings, "rank(vol(20)) - rank(vol(126))", capsys)
    assert code == 3 and str(out["prior"]["nearest"]).startswith("agent:")
    assert ledger.count(FAMILY) == 1
