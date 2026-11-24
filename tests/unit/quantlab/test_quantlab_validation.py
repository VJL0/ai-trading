from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.quantlab.factors import SpecError, validate_expression, validate_spec
from services.quantlab.ledger import HoldoutLookRefused, HoldoutVault, TrialLedger
from services.quantlab.opportunity import Evidence, EvidenceStatus, Opportunity, leveraged_etf_expected_return, select
from services.quantlab.options import Leg, bs_price, crr_american, implied_vol, vertical_spread
from services.quantlab.registry import State, StrategyRegistry, TransitionRefused
from services.quantlab.validation import (
    deflated_sharpe,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe,
    spa_test,
    walk_forward_select,
)


def test_psr_dsr_behave() -> None:
    rng = np.random.default_rng(0)
    good = 0.001 + 0.01 * rng.standard_normal(2000)
    noise = 0.01 * rng.standard_normal(2000)
    assert probabilistic_sharpe(good) > 0.99
    assert 0.05 < probabilistic_sharpe(noise) < 0.95 or probabilistic_sharpe(noise) < 0.99
    assert expected_max_sharpe(100, 0.0004) > expected_max_sharpe(10, 0.0004) > 0
    d1 = deflated_sharpe(good, 1, 0.0004)["dsr"]
    d1000 = deflated_sharpe(good, 1000, 0.0004)["dsr"]
    assert d1 >= d1000


def test_spa_detects_edge_but_not_noise() -> None:
    rng = np.random.default_rng(1)
    bench = 0.01 * rng.standard_normal(1500)
    noise = bench[:, None] + 0.005 * rng.standard_normal((1500, 20))
    assert spa_test(noise, bench, n_boot=300)["p_value"] > 0.05
    edge = noise.copy()
    edge[:, 3] += 0.0015
    assert spa_test(edge, bench, n_boot=300)["p_value"] < 0.05


def test_pbo_noise_near_half_and_persistent_skill_low() -> None:
    rng = np.random.default_rng(2)
    noise = 0.01 * rng.standard_normal((1600, 12))
    assert 0.2 < pbo_cscv(noise, 8)["pbo"] < 0.8
    skill = noise + np.linspace(0, 0.002, 12)[None, :]
    assert pbo_cscv(skill, 8)["pbo"] < 0.2


def test_walk_forward_uses_only_past_years() -> None:
    idx = pd.bdate_range("2016-01-01", "2021-12-31")
    a = pd.Series(np.where(idx.year < 2019, 0.002, -0.002), index=idx)
    b = pd.Series(np.where(idx.year < 2019, -0.001, 0.003), index=idx) + 1e-6 * np.arange(len(idx)) % 3
    wf = walk_forward_select({"a": a, "b": b}, first_test_year=2019)
    assert wf.choices[0]["selected"] == "a"  # chosen on 2016-2018 only, even though b wins later
    assert set(wf.oos.index.year) == {2019, 2020, 2021}


def test_trial_ledger_append_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    led = TrialLedger(tmp_path / "t.sqlite3")
    led.record(family="X", variant="v", spec={"a": 1}, data_fingerprint="f", stage="DEV",
               metrics={"sharpe": 1.0, "days": 10}, status="OK")
    assert led.count("X") == 1
    with sqlite3.connect(tmp_path / "t.sqlite3") as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("DELETE FROM trials")


def test_holdout_vault_one_look_budget_and_immutable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    v = HoldoutVault(tmp_path / "h.sqlite3")
    assert v.register("h", date(2025, 11, 21), date(2026, 11, 20)) == (date(2025, 11, 21), date(2026, 11, 20))
    assert v.register("h", date(2020, 1, 1), date(2020, 2, 1)) == (date(2025, 11, 21), date(2026, 11, 20))
    look = v.claim("h", "F", {"p": 1}, "c1")
    with pytest.raises(HoldoutLookRefused):
        v.claim("h", "F", {"p": 1}, "c1-again")
    v.claim("h", "F", {"p": 2}, "c2")
    v.claim("h", "F", {"p": 3}, "c3")
    with pytest.raises(HoldoutLookRefused):
        v.claim("h", "F", {"p": 4}, "c4")  # family budget spent
    v.record_result(look, {"passed": False})
    with pytest.raises(HoldoutLookRefused):
        v.record_result(look, {"passed": True})  # cannot overwrite
    with sqlite3.connect(tmp_path / "h.sqlite3") as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE holdout SET start='2020-01-01'")


def test_registry_gates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    r = StrategyRegistry(tmp_path / "r.sqlite3")
    r.register("s1", family="A", spec_hash="h1", spec={"x": 1}, asset_class="us_equity", holding="multi_day")
    with pytest.raises(TransitionRefused):
        r.transition("s1", State.VALIDATED, reason="skip", evidence={})
    with pytest.raises(TransitionRefused):
        r.transition("s1", State.DEV_PASSED, reason="wrong spec", evidence={"dev_passed": True, "spec_hash": "zz"})
    r.transition("s1", State.DEV_PASSED, reason="ok", evidence={"dev_passed": True, "spec_hash": "h1"})
    r.transition("s1", State.VALIDATED, reason="ok", evidence={"holdout_passed": True, "spec_hash": "h1",
                                                               "look_id": 1})
    r.transition("s1", State.SHADOW, reason="auto", evidence={})
    with pytest.raises(TransitionRefused):
        r.transition("s1", State.PAPER, reason="too early", evidence={"execution_eligible": True,
                                                                      "lifecycle_supported": True,
                                                                      "tracking_gap_bps": 1})
    r.transition("s1", State.SUSPENDED, reason="operator", evidence={})
    assert r.state("s1") == State.SUSPENDED
    assert len(r.transitions("s1")) == 5


@pytest.mark.parametrize("bad", ["__import__('os')", "open('x')", "mom(21).real", "mom(x)", "mom(lb=21)",
                                 "[mom(21)]", "lambda: 1", "mom(21) if 1 else 2", "rank(mom(21))**2"])
def test_factor_dsl_rejects_code(bad: str) -> None:
    with pytest.raises(SpecError):
        validate_expression(bad)


def test_factor_dsl_accepts_and_validates_spec() -> None:
    validate_expression("rank(mom(126, 21)) - 0.5 * rank(vol(63))")
    spec = validate_spec({"expression": "zscore(resid_mom(252, 21))", "n": 20})
    assert spec["universe"] == "liquid1000" and spec["rebalance"] == 21
    with pytest.raises(SpecError):
        validate_spec({"expression": "rank(mom(21))", "holdout": True})


def _opp(status: EvidenceStatus, executable: bool, gross: float = 0.02, **kw) -> Opportunity:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime
    return Opportunity(opportunity_id=f"{status}-{executable}-{gross}", asof=datetime.now(UTC), asset_class="us_equity",
                       instrument="SPY", strategy_family="A", direction="LONG", horizon_days=21,
                       expected_gross_return=gross, expected_cost=0.001, forecast_std=0.05, downside_cvar95=-0.08,
                       capital_required=1000, buying_power_effect=1000, evidence=Evidence(status=status),
                       executable=executable, **kw)


def test_selection_requires_evidence_and_executability() -> None:
    sel = select([_opp(EvidenceStatus.RESEARCH_ONLY, True, 0.2), _opp(EvidenceStatus.VALIDATED, False, 0.2)])
    assert sel.action == "NO_TRADE"
    sel = select([_opp(EvidenceStatus.RESEARCH_ONLY, True, 0.2), _opp(EvidenceStatus.VALIDATED, True, 0.03)])
    assert sel.action == "TRADE" and sel.selected is not None
    assert sel.selected.opportunity.evidence.status == EvidenceStatus.VALIDATED
    assert select([_opp(EvidenceStatus.VALIDATED, True, 0.0005)]).action == "NO_TRADE"  # edge below costs


def test_leveraged_etf_drag() -> None:
    mu1, _ = leveraged_etf_expected_return(0.0004, 0.02, 1.0, 21, 0.0)
    mu3, sd3 = leveraged_etf_expected_return(0.0004, 0.02, 3.0, 21, 0.0)
    assert mu3 < 3 * mu1  # volatility drag
    assert sd3 == pytest.approx(3 * 0.02 * 21 ** 0.5)


def test_option_pricing_core() -> None:
    c = bs_price(100, 105, 0.25, 0.04, 0.01, 0.25, True)
    assert implied_vol(c, 100, 105, 0.25, 0.04, 0.01, True) == pytest.approx(0.25, abs=1e-5)
    am, _ = crr_american(100, 110, 1.0, 0.05, 0.0, 0.3, False)
    assert am >= bs_price(100, 110, 1.0, 0.05, 0.0, 0.3, False) - 1e-6
    exp = date(2026, 12, 18)
    res = vertical_spread(100, Leg("L", 100, True, 1, 3.0, 3.2, exp), Leg("S", 105, True, -1, 1.0, 1.2, exp),
                          sigma_physical=0.25, asof=date(2026, 11, 18))
    assert res["max_loss_per_spread"] == pytest.approx((3.2 - 1.0) * 100 + res["fees_per_share"] * 100)
    assert res["max_gain_per_spread"] < 500
    assert res["cvar95_frac"] >= -1.0 - 1e-6


def test_walk_forward_half_years() -> None:
    idx = pd.bdate_range("2021-01-01", "2023-12-31")
    a = pd.Series(0.001, index=idx)
    b = pd.Series(np.where(idx < pd.Timestamp("2022-07-01"), -0.001, 0.002), index=idx)
    wf = walk_forward_select({"a": a, "b": b}, first_test_year=2022, period="half", min_train_days=250)
    assert [c["year"] for c in wf.choices] == [4044, 4045, 4046, 4047]  # year*2 + (H2)
    assert wf.choices[0]["selected"] == "a"
    assert wf.oos.index[0] == pd.Timestamp("2022-01-03")


def test_fhs_keeps_fat_left_tail_and_zero_drift() -> None:
    from services.quantlab.options import fhs_terminal

    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_normal(2000)
    r[rng.choice(2000, 20, replace=False)] = -0.08  # crash days
    st, kind = fhs_terminal(100.0, 21, 0.2, r, 40000, np.random.default_rng(1))
    assert kind == "filtered_historical_simulation"
    assert abs(st.mean() - 100.0) < 0.2
    ln, _ = fhs_terminal(100.0, 21, 0.2, None, 40000, np.random.default_rng(1))
    assert np.quantile(st, 0.005) < np.quantile(ln, 0.005)  # heavier left tail than lognormal


def test_parity_spot_recovers_the_quotes_own_underlying() -> None:
    from services.quantlab.options import bs_price, parity_spot

    asof = date(2026, 11, 23)
    S, r, T = 101.3, 0.04, 31 / 365
    snaps = {}
    for k in (95, 98, 100, 102, 105):
        for call, cp in ((True, "C"), (False, "P")):
            mid = bs_price(S, k, T, r, 0.0, 0.25, call)
            snaps[f"XYZ261224{cp}{int(k * 1000):08d}"] = {"latestQuote": {"bp": mid - 0.05, "ap": mid + 0.05}}
    assert parity_spot({"snapshots": snaps}, r=r, asof=asof) == pytest.approx(S, abs=1e-6)


def test_prospective_journal_forward_only_and_append_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from services.quantlab.prospective import DEFAULT_ORB_SPECS, ProspectiveJournal

    j = ProspectiveJournal(tmp_path / "p.sqlite3")
    reg = datetime(2026, 11, 24, 15, 0, tzinfo=UTC)
    hashes = j.register("B_orb", DEFAULT_ORB_SPECS, now=reg)
    assert j.register("B_orb", DEFAULT_ORB_SPECS) == hashes  # idempotent; registration time is not reset
    spec = j.specs()[0]
    assert spec["registered_at"] == reg.isoformat()
    assert not j.eligible(spec, datetime(2026, 11, 24, 14, 30, tzinfo=UTC))  # opened before registration
    assert j.eligible(spec, datetime(2026, 11, 25, 14, 30, tzinfo=UTC))
    news_h = next(s["spec_hash"] for s in j.specs() if s["spec"]["catalyst"] == "news")
    none_h = next(s["spec_hash"] for s in j.specs() if s["spec"]["catalyst"] == "none")
    for i in range(5):
        assert j.record(news_h, date(2026, 12, 1 + i), 0.002 + 0.001 * i, 0.003, 4)
        assert j.record(none_h, date(2026, 12, 1 + i), 0.001, 0.002, 4)
    assert not j.record(news_h, date(2026, 12, 1), 9.9, 9.9, 1)  # one result per session, never overwritten
    s = j.summary()
    pair = s["paired_news_vs_none"][0]
    assert pair["sessions"] == 5 and pair["mean_daily_diff"] > 0
    assert pair["news_adds_value"] is False  # far below the 60-session evidence minimum
    with sqlite3.connect(tmp_path / "p.sqlite3") as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE sessions SET net = 1.0")
