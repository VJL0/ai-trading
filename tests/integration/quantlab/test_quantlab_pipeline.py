"""End-to-end quantlab pipeline on SIMULATED panels: family evaluation, ledger, FDR, registry, one-use holdout,
scan and NO_TRADE-capable selection. Synthetic results are never evidence of profitability."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from services.quantlab import gates
from services.quantlab.ledger import HoldoutLookRefused
from services.quantlab.panel import EquityPanel
from services.quantlab.pipeline import Lab, equity_opportunities, run_selection
from services.quantlab.registry import State
from services.quantlab.scanner import scan
from tests.unit.quantlab import synth

ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD")


def _panel() -> EquityPanel:
    rng = np.random.default_rng(5)
    syms = {s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 5e7} for s in ETFS}
    for i in range(40):
        syms[f"S{i:02d}"] = {"drift": float(rng.normal(0.0002, 0.0006)), "vol": 0.02, "price": 50, "volume": 3e6}
    raw, adj = synth.bars(syms, n=2740, start="2016-01-04")
    return EquityPanel.build(raw, adj, synth.assets(list(syms), etfs=ETFS), min_obs=50)


@pytest.fixture
def lab(tmp_path):  # type: ignore[no-untyped-def]
    (tmp_path / "datasets").mkdir()
    return Lab(tmp_path)


def test_pipeline_end_to_end(lab: Lab) -> None:
    panel = _panel()
    dev = lab.dev_slice(panel)
    assert dev.dates[-1] < np.datetime64(gates.HOLDOUT_START)
    fams = lab.equity_families(dev)
    small = []
    for f in fams:
        if f.family in ("A2_etf_rotation", "C3_etf_pullback"):
            small.append(replace(f, variants=list(f.variants)[:6]))
    summary = lab.run_research(small, dev, "synthetic-fp", run_id="ql_test")
    assert {f["family"] for f in summary["families"]} == {"A2_etf_rotation", "C3_etf_pullback"}
    for f in summary["families"]:
        assert f["dsr"] is not None and f["n_trials"] >= 6
        assert isinstance(f["fail_reasons"], list)
    assert lab.ledger.count("A2_etf_rotation") == 6
    assert (lab.out / "runs" / "ql_test" / "summary.json").is_file()
    latest = json.loads((lab.out / "latest_research.json").read_text())
    assert "A2_etf_rotation" in latest["families"]
    json.dumps(latest, allow_nan=False)  # strict JSON for the API/dashboard

    # holdout looks are refused for anything not DEV_PASSED
    spec = next(f for f in lab.equity_families(panel) if f.family == "A2_etf_rotation")
    with pytest.raises(HoldoutLookRefused):
        lab.evaluate_holdout(spec, panel, spec.variants[0], {})

    # force a DEV_PASSED registration to exercise the one-use holdout path
    from services.quantlab.ledger import spec_hash
    p = spec.variants[0]
    sid = f"{spec.family}:{spec_hash(p)}"
    lab.registry.register(sid, family=spec.family, spec_hash=spec_hash(p), spec=p, asset_class="etf",
                          holding="multi_day")
    lab.registry.transition(sid, State.DEV_PASSED, reason="test", evidence={"dev_passed": True,
                                                                             "spec_hash": spec_hash(p)})
    res = lab.evaluate_holdout(spec, panel, p, {"walk_forward": {"metrics": {"max_drawdown": -0.1}}})
    assert res["metrics"]["start"] >= str(gates.HOLDOUT_START)
    assert lab.registry.state(sid) in (State.SHADOW, State.REJECTED)
    with pytest.raises(HoldoutLookRefused):
        lab.vault.claim(gates.HOLDOUT_NAME, spec.family, p, sid)

    sc = scan(panel)
    assert sc["regime"]["label"] in ("RISK_ON", "RISK_OFF", "MIXED")
    from services.quantlab.report import render

    text = render(lab.root)
    assert "A2_etf_rotation" in text and "Protected holdout looks" in text
    opps = equity_opportunities(panel, lab)
    sel = run_selection(opps, lab.out)
    # nothing is executable today: multi-day holds are blocked by the lifecycle, so the engine must say NO_TRADE
    assert sel["action"] == "NO_TRADE"
    assert all(not r["selectable"] for r in sel["ranked"])


def test_shadow_journal_and_promotion_blockers(lab: Lab) -> None:
    from datetime import date, timedelta

    from services.market_data.calendar import get_calendar
    from services.quantlab import shadow
    from services.quantlab.ledger import spec_hash

    panel = _panel()
    spec = next(f for f in lab.equity_families(panel) if f.family == "A2_etf_rotation")
    p = spec.variants[0]
    sid = f"{spec.family}:{spec_hash(p)}"
    lab.registry.register(sid, family=spec.family, spec_hash=spec_hash(p), spec=p, asset_class="etf",
                          holding="multi_day")
    lab.registry.transition(sid, State.DEV_PASSED, reason="t", evidence={"dev_passed": True, "spec_hash": spec_hash(p)})
    lab.registry.transition(sid, State.VALIDATED, reason="t", evidence={"holdout_passed": True,
                                                                        "spec_hash": spec_hash(p), "look_id": 1})
    lab.registry.transition(sid, State.SHADOW, reason="t", evidence={})
    cal = get_calendar()
    first_time = cal.session_close(date.fromisoformat(str(panel.dates[-2])[:10])) + timedelta(minutes=1)
    second_time = cal.session_close(date.fromisoformat(str(panel.dates[-1])[:10])) + timedelta(minutes=1)
    first = shadow.update(lab, panel.slice_dates(None, panel.dates[-2]), now=first_time)
    assert first["recorded"] and not first["settled"]
    again = shadow.update(lab, panel, now=second_time)
    assert again["settled"], "the earlier journal entry is settled once the next session exists"
    check = again["promotion_checks"][0]
    assert check["promoted"] is False
    assert any("multi_day" in b for b in check["blockers"])  # lifecycle gap blocks PAPER
    assert lab.registry.state(sid) == State.SHADOW


def test_ml_rank_scores_are_walk_forward_only(lab: Lab) -> None:
    from services.quantlab.strategies import ml_rank as G
    from services.quantlab.strategies import momentum as A

    panel = _panel()
    dev = lab.dev_slice(panel)
    cache = A.ScoreCache(dev)
    u = dev.universe(min_adv=1e6, min_history=60)  # synthetic names are small
    u[:, : len(ETFS)] = False  # some columns never eligible: exercises the reduced-width mapping
    cache._u["liquid1000"] = u
    wf = G.WalkForwardScores(dev, cache, first_test_year=2019)
    sc = wf.scores("ridge")
    assert sc.shape == dev.shape  # mapped back to full panel width
    years = np.asarray(dev.dates.year)
    assert np.isnan(sc[years < 2019]).all()  # no in-sample scores before the first test year
    assert np.isfinite(sc[years == 2020]).any()
    # every fold trained strictly before its test year (purged by the label horizon)
    assert all(f["test_year"] >= 2019 for f in wf.fold_log)
    # changing the future (labels after 2021) cannot change 2020 scores
    wf2 = G.WalkForwardScores(dev, cache, first_test_year=2019)
    wf2.y = wf2.y.copy()
    wf2.y[years >= 2021] = 0.0
    np.testing.assert_allclose(np.nan_to_num(wf2.scores("ridge")[years == 2020]), np.nan_to_num(sc[years == 2020]))


def test_subset_computation_matches_full_width() -> None:
    from services.quantlab.strategies import momentum as A
    from services.quantlab.strategies import reversion as C

    panel = _panel()
    p = {"score": "blend", "n": 2, "rebalance": 5, "absolute": "sma200"}
    sub, cols = panel.subset(A.ROTATION_ETFS)
    full_width = A._rotation_targets(panel, p)  # pyright: ignore[reportPrivateUsage]
    # computing on the whole panel restricted to the ETF universe gives the same weights as the subset path
    np.testing.assert_allclose(np.nan_to_num(A.rotation_targets(panel, p))[:, cols],
                               np.nan_to_num(A._rotation_targets(sub, p)))  # pyright: ignore[reportPrivateUsage]
    assert full_width.shape == panel.shape
    w = C.etf_pullback_targets(panel, {"rsi_below": 10, "n": 1, "max_hold": 5, "trend": "sma200"})
    assert w.shape == panel.shape and np.nanmax(np.nansum(w, axis=1)) <= 1.0
