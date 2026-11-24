"""Prospective evidence must be forward-only, after costs, and preserve explicit exits."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from services.market_data.calendar import get_calendar
from services.quantlab import shadow
from services.quantlab.backtest import BacktestConfig
from services.quantlab.ledger import code_hash, spec_hash
from services.quantlab.panel import EquityPanel
from services.quantlab.pipeline import Lab, crypto_opportunities, equity_opportunities
from services.quantlab.registry import State, TransitionRefused
from services.quantlab.research import FamilySpec
from services.quantlab.targets import latest_decision


def _register(lab: Lab, asset_class: str = "us_equity") -> dict:
    lab.registry.register("test", family="test", spec_hash=spec_hash({}), spec={}, asset_class=asset_class,
                          holding="multi_day")
    lab.registry.transition("test", State.DEV_PASSED, reason="synthetic", evidence={"dev_passed": True,
                                                                                  "spec_hash": spec_hash({})})
    lab.registry.transition("test", State.VALIDATED, reason="synthetic", evidence={"holdout_passed": True,
                                                                                 "spec_hash": spec_hash({}), "look_id": 1})
    lab.registry.transition("test", State.SHADOW, reason="synthetic", evidence={})
    return lab.registry.get("test") or {}


def _panel(*, crypto: bool = False) -> EquityPanel:
    days = ["2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"]
    if crypto:
        days = ["2026-11-22", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27"]
    op = np.array([100, 101, 102, 104, 105, 106.0])[:, None]
    cl = op + 1.0
    prev = np.r_[cl[:1], cl[:-1]]
    f = {"open": op, "close": cl, "ret_co": op / prev - 1, "ret_oc": cl / op - 1,
         "ret_cc": cl / prev - 1, "adv20": np.full_like(op, 1e9)}
    return EquityPanel(pd.DatetimeIndex(days), np.array(["BTC/USD" if crypto else "SPY"]), f,
                       np.array(["CRYPTO" if crypto else "ETF"]), {"freq": "1D"})


def _install_spec(monkeypatch: pytest.MonkeyPatch, panel: EquityPanel, weights: list[float]) -> None:
    by_date = dict(zip(panel.dates, weights, strict=True))

    def families(_lab, p, _asset):  # type: ignore[no-untyped-def]
        raw = np.array([by_date[d] for d in p.dates])[:, None]
        return {"test": FamilySpec("test", "us_equity", "synthetic", [{}], lambda _: raw,
                                    "cash", np.zeros(p.shape),
                                    BacktestConfig(capital=1000, flat_cost_bps=(100.0,)))}

    monkeypatch.setattr(shadow, "_family_specs", families)


def _after_close(day: str) -> datetime:
    return get_calendar().session_close(date.fromisoformat(day)) + timedelta(minutes=1)


@pytest.mark.parametrize("targets,t,expected", [
    ([[1.0], [0.0], [np.nan]], 2, [0.0]),
    ([[1.0], [np.nan], [0.0]], 1, [1.0]),
    ([[np.nan], [np.nan], [1.0]], 1, [0.0]),
])
def test_target_resolution_respects_cash_and_never_reads_future(targets, t, expected) -> None:  # type: ignore[no-untyped-def]
    _, weights = latest_decision(np.array(targets), t)
    np.testing.assert_array_equal(weights, expected)


def test_shadow_crops_future_bars_and_charges_costs_and_marks_missed_days(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    lab, panel = Lab(tmp_path), _panel()
    _register(lab)
    _install_spec(monkeypatch, panel, [1, 1, 0, np.nan, 1, np.nan])
    first = shadow.update(lab, panel, now=_after_close("2026-11-23"))
    assert first["recorded"][0]["session"] == "2026-11-23"  # later bars in the supplied panel were ignored
    assert first["recorded"][0]["targets"] == {"SPY": 1.0}
    assert not first["settled"]
    next_day = shadow.update(lab, panel, now=_after_close("2026-11-24"))
    assert next_day["recorded"][0]["targets"] == {}  # explicit EXIT cannot resurrect the older long target
    mark = next_day["settled"][0]
    assert mark["cost"] == pytest.approx(0.01)
    assert mark["gross"] == pytest.approx(103 / 102 - 1)
    assert mark["net"] == pytest.approx(0.99 * 103 / 102 - 1)
    assert mark["replayed"] == pytest.approx(mark["net"])
    # Monitor misses two sessions. The committed exit still executes on Nov 25 and cash stays cash. The Nov 27
    # signal was never journaled before its open, so it cannot be filled retroactively on Nov 30.
    later = shadow.update(lab, panel, now=_after_close("2026-11-30"))
    marks = {r["session"]: r for r in lab.registry.shadow_marks("test")}
    assert set(marks) == {"2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"}
    assert marks["2026-11-25"]["cost"] == pytest.approx(0.01)
    assert marks["2026-11-27"]["net"] == 0 and marks["2026-11-30"]["net"] == 0
    assert later["recorded"][0]["execution_session"] == "2026-12-01"
    assert lab.registry.shadow_sessions("test") == 4
    with sqlite3.connect(lab.registry.path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE shadow SET targets='{}'")
    with sqlite3.connect(lab.registry.path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE shadow_marks SET net=1")


def test_historical_or_late_decision_cannot_earn_prospective_evidence(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    lab, panel = Lab(tmp_path), _panel()
    _register(lab)
    _install_spec(monkeypatch, panel, [1] * 6)
    stale = panel.slice_dates(None, pd.Timestamp("2026-11-23"))
    late = shadow.update(lab, stale, now=datetime(2026, 11, 24, 15, tzinfo=UTC))
    assert not late["recorded"] and not late["settled"]
    protected = panel.slice_dates(None, pd.Timestamp("2026-11-20"))
    assert not shadow.update(lab, protected, now=_after_close("2026-11-20"))["recorded"]
    # The legacy API remains readable for provenance, but gross/backdated rows do not count as forward evidence.
    lab.registry.record_shadow("test", "2026-11-19", {"SPY": 1})
    lab.registry.settle_shadow("test", "2026-11-19", 0.01, 0.01)
    assert lab.registry.shadow_sessions("test") == 0


def test_crypto_daily_shadow_waits_for_a_future_open(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    lab, panel = Lab(tmp_path), _panel(crypto=True)
    _register(lab, "crypto")
    _install_spec(monkeypatch, panel, [1] * 6)
    first = shadow.update(lab, panel, asset_class="crypto", now=datetime(2026, 11, 24, 10, tzinfo=UTC))
    assert first["recorded"][0]["session"] == "2026-11-23"
    assert first["recorded"][0]["execution_session"] == "2026-11-25"
    assert not first["settled"]  # the already-passed Nov 24 open cannot be claimed
    second = shadow.update(lab, panel, asset_class="crypto", now=datetime(2026, 11, 26, 10, tzinfo=UTC))
    assert [m["session"] for m in second["settled"]] == ["2026-11-25"]
    assert shadow.lifecycle_supported(lab, {"asset_class": "crypto"})[0] is False


def _mark_evidence(lab: Lab, count: int = 20) -> tuple[dict, list[date]]:
    cfg = BacktestConfig()
    days = get_calendar().sessions_in_range(date(2026, 11, 23), date(2027, 1, 29))
    op = get_calendar().session_open(days[1])
    meta = {"model_version": shadow.MODEL_VERSION, "code_hash": code_hash(), "config_hash": spec_hash(asdict(cfg)),
            "decision_kind": "REBALANCE", "execution_session": str(days[1]),
            "data_complete_at": get_calendar().session_close(days[0]).isoformat(), "execution_open": op.isoformat()}
    lab.registry.record_shadow("test", str(days[0]), {"SPY": 1}, metadata=meta,
                               now=get_calendar().session_close(days[0]) + timedelta(minutes=1))
    row = lab.registry.shadow_rows("test")[0]
    for day in days[1:count + 1]:
        _add_mark(lab, row, day, 0.001)
    return row, days


def _add_mark(lab: Lab, row: dict, day: date, net: float) -> None:
    cal = get_calendar()
    prov = {k: row["metadata"][k] for k in ("model_version", "code_hash", "config_hash")}
    prov.update({"decision_id": row["id"], "execution_open": cal.session_open(day).isoformat(),
                 "data_complete_at": cal.session_close(day).isoformat(),
                 "observed_at": (cal.session_close(day) + timedelta(minutes=1)).isoformat()})
    lab.registry.record_shadow_mark("test", str(day), net=net, gross=net, cost=0, replayed=net, provenance=prov)


@pytest.mark.parametrize("gap", [float("nan"), float("inf"), -1.0])
def test_registry_rejects_invalid_tracking_even_with_enough_sessions(tmp_path, gap) -> None:  # type: ignore[no-untyped-def]
    lab = Lab(tmp_path)
    _register(lab)
    _mark_evidence(lab)
    with pytest.raises(TransitionRefused, match="tracking"):
        lab.registry.transition("test", State.PAPER, reason="invalid", evidence={
            "execution_eligible": True, "lifecycle_supported": True, "tracking_gap_bps": gap})
    with pytest.raises(ValueError, match="non-finite"):
        lab.registry.settle_shadow("test", "2026-11-23", float("nan"), 0)


def test_drawdown_surveillance_suspends_paper_strategy(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    lab = Lab(tmp_path)
    strategy = _register(lab)
    row, days = _mark_evidence(lab)
    lab.registry.transition("test", State.PAPER, reason="synthetic", evidence={
        "execution_eligible": True, "lifecycle_supported": True, "tracking_gap_bps": 0})
    (lab.out / "latest_research.json").write_text(json.dumps({"families": {"test": {
        "selected_spec_hash": strategy["spec_hash"], "wf_oos": {"max_drawdown": -0.05}}}}))
    monkeypatch.setattr(shadow, "lifecycle_supported", lambda *_: (True, "synthetic"))
    _add_mark(lab, row, days[21], -0.20)
    check = shadow.promotion_check(lab, lab.registry.get("test") or {},
                                   {"us_equity_long": {"execution_eligible": True}})
    assert check["state"] == State.SUSPENDED
    assert "prospective drawdown breach" in check["blockers"]


@pytest.mark.parametrize("crypto", [False, True])
def test_opportunities_do_not_resurrect_explicit_exit(tmp_path, monkeypatch, crypto) -> None:  # type: ignore[no-untyped-def]
    lab, panel = Lab(tmp_path), _panel(crypto=crypto)
    raw = np.array([[1], [0], [np.nan], [np.nan], [np.nan], [np.nan]])
    spec = FamilySpec("test", "crypto" if crypto else "etf", "test", [{"rebalance": 1}], lambda _: raw,
                       "cash", np.zeros(panel.shape))
    monkeypatch.setattr(lab, "crypto_families" if crypto else "equity_families", lambda _: [spec])
    (lab.out / "latest_research.json").write_text(json.dumps({"families": {"test": {
        "selected_variant": "rebalance=1", "selected_spec_hash": "test", "wf_oos": {"cagr": 1}}}}))
    results = crypto_opportunities(panel, lab) if crypto else equity_opportunities(panel, lab)
    assert results == []
