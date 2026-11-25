"""Session-timing research end to end on a SIMULATED temporary project: dataset directory -> intraday panel ->
families -> trial ledger -> summary; a re-run on the same data and code is skipped as carrying no new evidence.
Synthetic results are never evidence of profitability."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from services.market_data.calendar import get_calendar
from services.quantlab.panel import EquityPanel
from services.quantlab.pipeline import Lab
from services.quantlab.strategies import session as S
from tests.unit.quantlab import synth
from tests.unit.quantlab.test_session_families import _intraday_bars

SYMS = ("SPY", "QQQ", "IWM", "DIA")


@pytest.fixture(scope="module")
def project(tmp_path_factory):  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("proj")
    (root / "datasets").mkdir()
    cal = get_calendar()
    days = cal.sessions_in_range(date(2016, 9, 1), date(2019, 6, 28))
    src = root / "datasets" / "quantlab" / "intraday" / "session5m_all_20160901_20190628"
    src.mkdir(parents=True)
    _intraday_bars(days, SYMS, extended=False).to_parquet(src / "y2017_b000.parquet", index=False)
    (src / "manifest.json").write_text(json.dumps({"params": {"kind": "intraday_range", "synthetic": True},
                                                   "batches": {}}))
    raw, adj = synth.bars({s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 5e7} for s in SYMS},
                          n=len(cal.sessions_in_range(date(2016, 1, 4), date(2019, 6, 28))), start="2016-01-04")
    daily = EquityPanel.build(raw, adj, synth.assets(list(SYMS), etfs=SYMS), min_obs=50)
    return root, daily


def test_session_research_end_to_end_and_no_new_evidence_skip(project) -> None:  # type: ignore[no-untyped-def]
    root, daily = project
    lab = Lab(root)
    path = lab.build_session_panel()
    assert path.is_file() and path.with_suffix(".json").is_file()
    ip, fp = lab.load_session_panel()
    assert fp.startswith("session:") and len(ip.days) > 500
    dev = lab.dev_slice(daily)
    specs = [replace(s, variants=list(s.variants)[:4]) for s in lab.session_families(ip, dev, families=(S.H1, S.I1))]
    summary = lab.run_research(specs, dev, fp, run_id="ql_session_test")
    fams = {f["family"]: f for f in summary["families"]}
    assert set(fams) == {S.H1, S.I1}
    for f in fams.values():
        assert f["n_trials"] >= 4 and isinstance(f["fail_reasons"], list)
        assert f["policy_tier"]["status"] == ("OK" if f["family"] == S.H1 else "NOT_EXECUTABLE")
        assert f["benchmark"] == "SPY buy-and-hold (close-to-close)"
        json.dumps(f, allow_nan=False)
    assert lab.ledger.count(S.H1) == 4
    note_before = json.loads((lab.out / "latest_research.json").read_text())
    again = lab.run_research(specs, dev, fp, run_id="ql_session_test2")
    assert again["skipped_no_new_evidence"] == [S.H1, S.I1] and again["families"] == []
    assert lab.ledger.count(S.H1) == 4  # nothing appended
    assert json.loads((lab.out / "latest_research.json").read_text()) == note_before  # untouched, disclaimer kept
    assert note_before["no_edge_note"]
    forced = lab.run_research(specs[:1], dev, fp, run_id="ql_session_test3", force=True)
    assert forced["families"] and lab.ledger.count(S.H1) == 8
    assert S.I1 in forced["fdr"]["peers_from_earlier_runs"]  # a lone re-run is still corrected against its peers
    assert forced["fdr"]["families_in_correction"] == 2
    latest = json.loads((lab.out / "latest_research.json").read_text())
    assert S.H1 in latest["families"] and S.I1 in latest["families"]
    assert np.isfinite(latest["families"][S.H1]["dsr"])


def test_session_holdout_requires_dev_passed_and_a_committed_look(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pandas as pd

    from services.quantlab import gates
    from services.quantlab.ledger import HoldoutLookRefused, spec_hash
    from services.quantlab.registry import State
    from tests.unit.quantlab.test_session_families import _daily

    (tmp_path / "datasets").mkdir()
    lab = Lab(tmp_path)
    cal = get_calendar()
    days = cal.sessions_in_range(date(2025, 6, 2), date(2026, 1, 30))  # straddles HOLDOUT_START
    from services.quantlab.intraday import IntradayPanel

    ip = IntradayPanel.build(_intraday_bars(days, SYMS, extended=False))
    daily = _daily(days, SYMS)
    params = next(iter(S.VARIANTS[S.I1]()))
    with pytest.raises(HoldoutLookRefused):
        lab.claim_holdout_look(S.I1, params)  # not DEV_PASSED
    with pytest.raises(ValueError, match="development data only"):
        lab.session_families(ip, daily, families=(S.I1,))
    with pytest.raises(ValueError, match="development data only"):
        lab.session_families(ip, daily, families=(S.I1,), holdout_look=12345)  # an uncommitted look id
    # another spec's open look can never be used to score this spec
    other = list(S.VARIANTS[S.I1]())[1]
    osid = f"{S.I1}:{spec_hash(other)}"
    lab.registry.register(osid, family=S.I1, spec_hash=spec_hash(other), spec=other, asset_class="etf",
                          holding="multi_day")
    lab.registry.transition(osid, State.DEV_PASSED, reason="test", evidence={"dev_passed": True,
                                                                               "spec_hash": spec_hash(other)})
    other_look = lab.claim_holdout_look(S.I1, other)
    assert lab.claim_holdout_look(S.I1, other, resume=True) == other_look  # unresolved own look: resumable
    sid = f"{S.I1}:{spec_hash(params)}"
    lab.registry.register(sid, family=S.I1, spec_hash=spec_hash(params), spec=params, asset_class="etf",
                          holding="multi_day")
    lab.registry.transition(sid, State.DEV_PASSED, reason="test", evidence={"dev_passed": True,
                                                                             "spec_hash": spec_hash(params)})
    look = lab.claim_holdout_look(S.I1, params)
    spec = lab.session_families(ip, daily, families=(S.I1,), holdout_look=look)[0]
    with pytest.raises(HoldoutLookRefused, match="own unresolved look"):
        lab.evaluate_holdout(spec, daily, params, {}, look=other_look)
    with pytest.raises(ValueError, match="development data only"):
        lab.session_families(ip, daily, families=(S.I1, S.I2), holdout_look=look)  # one family per look
    res = lab.evaluate_holdout(spec, daily, params, {"walk_forward": {"metrics": {"max_drawdown": -0.1}}}, look=look)
    assert res["look_id"] == look and res["metrics"]["start"] >= str(gates.HOLDOUT_START)
    assert lab.registry.state(sid) in (State.SHADOW, State.REJECTED)
    with pytest.raises(HoldoutLookRefused):
        lab.claim_holdout_look(S.I1, params)  # one look per spec, permanently
    with pytest.raises(HoldoutLookRefused):
        lab.claim_holdout_look(S.I1, params, resume=True)  # a RESOLVED look is never reopened
    assert pd.Timestamp(gates.HOLDOUT_START) in set(ip.days) or ip.days.max() > pd.Timestamp(gates.HOLDOUT_START)
