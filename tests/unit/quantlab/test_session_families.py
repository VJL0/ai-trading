"""Session-timing research (5-minute intraday panel, lifecycle-faithful simulator, families H1-H3 / I1-I2) and the
hypothesis memory. Data is SIMULATED; nothing here is evidence of profitability."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from services.market_data.calendar import get_calendar
from services.quantlab import gates
from services.quantlab.costs import EquityCosts
from services.quantlab.intraday import (
    N_SLOTS,
    DailyContext,
    IntradayPanel,
    Leg,
    SimConfig,
    decision_slot,
    flatten_slot,
    latest_entry_slot,
    session_profile,
    simulate,
    simulate_leg,
    slot_of,
)
from services.quantlab.ledger import TrialLedger, duplicate_on_same_evidence
from services.quantlab.panel import EquityPanel
from services.quantlab.strategies import session as S
from tests.unit.quantlab import synth

NY = "America/New_York"
SYMS = ("SPY", "QQQ", "IWM", "DIA")


def _intraday_bars(days: list[date], symbols: tuple[str, ...] = SYMS, seed: int = 3,
                   extended: bool = True) -> pd.DataFrame:
    cal = get_calendar()
    rng = np.random.default_rng(seed)
    rows = []
    for sym in symbols:
        px = 100.0
        for d in days:
            open_ts = cal.session_open(d)
            n = int((cal.session_close(d) - open_ts).total_seconds() // 300)
            px *= float(np.exp(0.002 * rng.standard_normal()))  # overnight move
            for s in range(n):
                r = 0.0008 * rng.standard_normal()
                o, c = px, px * float(np.exp(r))
                rows.append((sym, open_ts + timedelta(minutes=5 * s), o, max(o, c) * 1.0004, min(o, c) * 0.9996, c,
                             1e5, 100, (o + c) / 2))
                px = c
            if extended:  # pre- and post-market bars must be dropped by the grid
                rows.append((sym, open_ts - timedelta(minutes=60), px, px, px, px, 10.0, 1, px))
                rows.append((sym, cal.session_close(d) + timedelta(minutes=30), px, px, px, px, 10.0, 1, px))
    df = pd.DataFrame(rows, columns=["symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def _daily(days: list[date], symbols: tuple[str, ...] = SYMS) -> EquityPanel:
    first = pd.Timestamp(days[0]) - pd.Timedelta(days=420)
    raw, adj = synth.bars({s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 5e7} for s in symbols},
                          n=len(get_calendar().sessions_in_range(first.date(), days[-1])), start=str(first.date()))
    return EquityPanel.build(raw, adj, synth.assets(list(symbols), etfs=symbols), min_obs=50)


@pytest.fixture(scope="module")
def world() -> tuple[IntradayPanel, DailyContext, list[date]]:
    cal = get_calendar()
    days = cal.sessions_in_range(date(2018, 10, 1), date(2019, 1, 31))  # includes 2018-11-23 and 12-24 early closes
    ip = IntradayPanel.build(_intraday_bars(days))
    ctx = DailyContext.align(ip, _daily(days))
    return ip, ctx, days


def _day_index(ip: IntradayPanel, d: date) -> int:
    return int(np.where(ip.days == pd.Timestamp(d))[0][0])


def test_grid_keeps_regular_session_and_early_closes(world) -> None:  # type: ignore[no-untyped-def]
    ip, _ctx, _ = world
    assert ip.f["c"].shape[1] == N_SLOTS
    early = _day_index(ip, date(2018, 11, 23))
    normal = _day_index(ip, date(2018, 11, 26))
    assert ip.last_slot[normal] == 77 and ip.last_slot[early] == 41
    assert np.isnan(ip.f["c"][early, 42:]).all()  # nothing after a 13:00 close
    assert np.isfinite(ip.f["c"][normal, :, 0]).all()
    assert ip.meta["rows"] == sum(int(n + 1) for n in ip.last_slot) * len(SYMS)  # extended-hours bars dropped
    assert slot_of("09:30") == 0 and slot_of("15:55") == 77 and decision_slot("10:00") == 5
    assert latest_entry_slot(77) == slot_of("15:25") and flatten_slot(77) == slot_of("15:50")
    assert latest_entry_slot(41) == slot_of("12:25") and flatten_slot(41) == slot_of("12:50")


def test_entry_cutoff_is_enforced_including_early_close(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    d = _day_index(ip, date(2018, 11, 26))
    cfg = SimConfig(weight=1.0, max_concurrent=1)
    ok = simulate_leg(ip, ctx, Leg(d, 0, slot_of("15:20"), d, flatten_slot(77)), cfg, 1.0)
    late = simulate_leg(ip, ctx, Leg(d, 0, slot_of("15:25"), d, flatten_slot(77)), cfg, 1.0)
    assert ok.filled and ok.entry_slot == slot_of("15:25")
    assert not late.filled and late.exit_reason == "ENTRY_WINDOW"
    e = _day_index(ip, date(2018, 11, 23))
    early = simulate_leg(ip, ctx, Leg(e, 0, slot_of("12:30"), e, 77), cfg, 1.0)
    assert not early.filled and early.exit_reason == "ENTRY_WINDOW"
    eod = simulate_leg(ip, ctx, Leg(e, 0, slot_of("10:00"), e, 77), cfg, 1.0)
    assert eod.filled and eod.exit_reason == "EOD"  # requested 15:55 is clamped to the 12:50 flatten bar


def test_non_fill_when_entry_bar_opens_through_the_limit(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    p2 = replace(ip, f={k: v.copy() for k, v in ip.f.items()})
    d = _day_index(p2, date(2018, 11, 27))
    dec = slot_of("11:00")
    p2.f["o"][d, dec + 1, 0] = p2.f["c"][d, dec, 0] * 1.002  # 20 bps above the decision close; limit is 10 bps
    res = simulate_leg(p2, ctx, Leg(d, 0, dec, d, flatten_slot(77)), SimConfig(limit_bps=10.0), 0.5)
    assert not res.filled and res.exit_reason == "NON_FILL"
    res_wide = simulate_leg(p2, ctx, Leg(d, 0, dec, d, flatten_slot(77)), SimConfig(limit_bps=30.0), 0.5)
    assert res_wide.filled


def test_stop_touch_and_overnight_gap_through_stop(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    p2 = replace(ip, f={k: v.copy() for k, v in ip.f.items()})
    d = _day_index(p2, date(2018, 11, 28))
    dec = slot_of("10:00")
    entry = float(p2.f["o"][d, dec + 1, 0])
    p2.f["l"][d, dec + 4, 0] = entry * 0.982  # touches a 150 bps stop, stays inside the 50 bps limit gap
    res = simulate_leg(p2, ctx, Leg(d, 0, dec, d, flatten_slot(77), stop_bps=150.0), SimConfig(), 0.5)
    assert res.exit_reason == "STOP" and res.exit_px == pytest.approx(entry * 0.985, rel=1e-6)
    # trading through the STOP_LIMIT's limit: the broker order cannot fill; marketable escalation, never at the stop
    p2.f["l"][d, dec + 4, 0] = entry * 0.95
    p2.f["c"][d, dec + 4, 0] = entry * 0.96
    br = simulate_leg(p2, ctx, Leg(d, 0, dec, d, flatten_slot(77), stop_bps=150.0), SimConfig(), 0.5)
    limit = entry * 0.985 * (1 - 50 / 1e4)
    assert br.exit_reason == "STOP_BREACH" and br.exit_px == pytest.approx(min(limit, entry * 0.96), rel=1e-6)
    assert br.exit_px < res.exit_px
    # overnight: the next session opens 4% below the entry -> exits at that open, not at the stop
    p3 = replace(ip, f={k: v.copy() for k, v in ip.f.items()})
    dec2 = slot_of("15:25")
    entry2 = float(p3.f["o"][d, dec2 + 1, 0])
    p3.f["o"][d + 1, 0, 0] = entry2 * 0.96
    over = simulate_leg(p3, ctx, Leg(d, 0, dec2 - 1, d + 1, slot_of("09:35"), stop_bps=150.0), SimConfig(), 0.5)
    assert over.exit_reason == "STOP_GAP" and over.exit_px == pytest.approx(entry2 * 0.96, rel=1e-6)
    assert over.exit_day_actual == d + 1


def test_cost_charged_exactly_once_per_leg(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    d = _day_index(ip, date(2018, 11, 29))
    leg = Leg(d, 0, slot_of("12:00"), d, flatten_slot(77))
    costs = EquityCosts()
    res = simulate_leg(ip, ctx, leg, SimConfig(capital=100_000, costs=costs), 0.25)
    adv, px = np.array([ctx.adv_usd[d, 0]]), np.array([ctx.raw_prev_close[d, 0]])
    notional = np.array([25_000.0])
    c_in = costs.one_way_bps(adv, notional, px, sell=np.array([False]), timing="close")[0]
    c_out = costs.one_way_bps(adv, notional, px, sell=np.array([True]), timing="close")[0]
    assert res.cost == pytest.approx((c_in + c_out) / 1e4)
    out, _ = simulate(ip, ctx, [leg], SimConfig(capital=100_000, weight=0.25, costs=costs))
    i = list(out.dates).index(ip.days[d])
    assert out.net[i] == pytest.approx(0.25 * (res.gross - res.cost))
    assert out.gross[i] - out.net[i] == pytest.approx(out.cost[i])  # cost appears once in the series
    free, _ = simulate(ip, ctx, [leg], SimConfig(weight=0.25, costs=EquityCosts(cost_multiplier=0.0)))
    assert free.net[i] == pytest.approx(free.gross[i])


def test_overnight_pnl_booked_on_exit_session_and_one_position_per_symbol(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    d = _day_index(ip, date(2018, 12, 3))
    over = Leg(d, 1, slot_of("15:20"), d + 1, slot_of("09:35"))
    dup = Leg(d + 1, 1, slot_of("09:30"), d + 1, flatten_slot(77))  # same symbol while still held
    other = Leg(d + 1, 2, slot_of("09:30"), d + 1, flatten_slot(77))
    res, outs = simulate(ip, ctx, [over, dup, other], SimConfig(weight=0.5, max_concurrent=2,
                                                                costs=EquityCosts(cost_multiplier=0.0)))
    i0, i1 = list(res.dates).index(ip.days[d]), list(res.dates).index(ip.days[d + 1])
    o_over = next(o for o in outs if o.leg == over)
    close_d = float(ip.f["c"][d, ip.last_slot[d], 1])
    assert o_over.marks == [(d, close_d)]  # marked to market at the entry session's close ...
    assert res.net[i0] == pytest.approx(0.5 * (close_d - o_over.entry_px) / o_over.entry_px)
    assert res.net[i0] + res.net[i1] - 0.5 * next(o for o in outs if o.leg == other).gross \
        == pytest.approx(0.5 * o_over.gross)  # ... and the segments sum to the round trip
    assert [o.leg for o in outs] == [over, other]  # the duplicate-symbol leg was never simulated
    capped, outs2 = simulate(ip, ctx, [over, other], SimConfig(weight=0.5, max_concurrent=1))
    assert [o.leg for o in outs2] == [over] and capped.trades == 2


def test_signals_use_only_bars_closed_by_the_decision_time(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    prm = {"entry": "12:00", "signal": "to_entry", "threshold_sd": 0.0, "universe": "index4", "exit": "eod"}
    base = S.h1_legs(S.SessionFeatures(ip, ctx), prm)
    p2 = replace(ip, f={k: v.copy() for k, v in ip.f.items()})
    dstar = 40
    after = decision_slot("12:00") + 1
    for k in ("o", "h", "l", "c", "vw"):
        p2.f[k][dstar, after:, :] *= 0.5  # bars after the decision on one session change ...
    changed = S.h1_legs(S.SessionFeatures(p2, ctx), prm)
    upto = [(lg.d, lg.j, lg.score) for lg in base if lg.d <= dstar]
    assert upto == [(lg.d, lg.j, lg.score) for lg in changed if lg.d <= dstar]  # ... decisions up to it do not
    assert any(lg.d == dstar for lg in base) or any(lg.d == dstar for lg in changed) or len(base) > 0
    for v in S.VARIANTS[S.H1]():
        assert v["entry"] in ("10:00", "12:00", "14:00", "15:00", "15:25")
        assert decision_slot(v["entry"]) + 1 <= latest_entry_slot(77)


def test_prev_close_and_trailing_scale_are_point_in_time(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    fe = S.SessionFeatures(ip, ctx)
    d = 40
    assert fe.pc[d, 0] == pytest.approx(ip.f["c"][d - 1, ip.last_slot[d - 1], 0])
    sig = fe.at(decision_slot("10:00")) / fe.pc - 1
    sd = fe.trailing_sd(("t", 1), sig)
    expect = pd.Series(sig[:, 0]).iloc[d - 20:d].std()
    assert sd[d, 0] == pytest.approx(expect)  # rows strictly before d


def test_month_positions_and_turn_of_month_holding_cap(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    last_k, first_m = S.month_positions(ip)
    i = _day_index(ip, date(2018, 11, 30))
    assert last_k[i] == 1 and first_m[i + 1] == 1 and last_k[i - 1] == 2
    for v in S.VARIANTS[S.I2]():
        assert S.holding_sessions(S.I2, v) <= 5  # RiskPolicy.max_holding_sessions default
    legs = S.i2_legs(S.SessionFeatures(ip, ctx), {"enter_k_last": 2, "exit_m_first": 1, "universe": "SPY",
                                                   "regime": "none", "stop_bps": 300.0}, (last_k, first_m))
    assert legs and all(last_k[lg.d] == 2 and first_m[lg.exit_day] == 1 and lg.exit_day - lg.d == 2 for lg in legs)
    assert S.holding_sessions(S.I1, {}) == 2 and S.holding_sessions(S.H1, {}) == 1


def test_every_family_produces_valid_legs(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    fe = S.SessionFeatures(ip, ctx)
    pos = S.month_positions(ip)
    for fam in S.FAMILIES:
        variants = list(S.VARIANTS[fam]())
        assert len({tuple(sorted(v.items())) for v in variants}) == len(variants)  # no duplicate variants
        for v in variants[:4]:
            for lg in S.legs_for(fam, fe, v, pos):
                assert lg.decision_slot + 1 <= latest_entry_slot(int(ip.last_slot[lg.d]))
                assert set(S.UNIVERSES[v["universe"]]) >= {str(ip.symbols[lg.j])}
                assert (lg.exit_day == lg.d) == (fam in S.INTRADAY_FAMILIES)
    prof = session_profile(ip, ("SPY",))
    assert "overnight" in prof["SPY"] and "to_16:00" in prof["SPY"]


def test_session_families_refuse_holdout_rows(tmp_path, world) -> None:  # type: ignore[no-untyped-def]
    from services.quantlab.pipeline import Lab

    ip, _ctx, days = world
    (tmp_path / "datasets").mkdir()
    lab = Lab(tmp_path)
    shifted = replace(ip, days=ip.days + pd.Timedelta(days=(gates.HOLDOUT_START - days[0]).days + 1))
    with pytest.raises(ValueError, match="development data only"):
        lab.session_families(shifted, _daily(days))
    specs = lab.session_families(ip, _daily(days), families=(S.H1,))
    spec = replace(specs[0], variants=list(specs[0].variants)[:2])
    res = spec.run(None, spec.variants[0])  # type: ignore[arg-type]
    assert len(res.dates) == len(ip.days)
    tier = lab.policy_tier(spec, None, spec.variants[0])  # type: ignore[arg-type]
    assert tier["status"] == "OK" and tier["max_positions"] == 1


# ------------------------------------------------------------------------------------------- hypothesis memory
def test_duplicate_hypothesis_on_same_evidence_is_detected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    led = TrialLedger(tmp_path / "t.sqlite3")
    spec = {"expression": "rank(mom(252, 21))", "universe": "liquid1000", "n": 20, "rebalance": 21, "regime": "none"}
    assert led.prior_outcomes("F_agent", spec) == []
    from services.quantlab.ledger import variant_name

    led.record(family="F_agent", variant="WALK_FORWARD:" + variant_name(spec), spec={"wf_of": "F_agent"},
               data_fingerprint="full1|dev:abc", stage="WF", metrics={"sharpe": 0.4}, status="FAIL", note="DSR low")
    prior = led.prior_outcomes("F_agent", spec)
    assert len(prior) == 1 and prior[0]["status"] == "FAIL"
    # same development data inside a newer full panel (a delta past the holdout) is NOT new evidence
    assert duplicate_on_same_evidence(prior, fingerprint="full2", dev_fingerprint="abc") is not None
    # corrected / extended development data is new evidence
    assert duplicate_on_same_evidence(prior, fingerprint="full3", dev_fingerprint="def") is None
    other = {**spec, "n": 30}
    assert led.prior_outcomes("F_agent", other) == []


def test_family_rerun_on_same_data_and_code_is_skipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from services.quantlab.ledger import code_hash

    led = TrialLedger(tmp_path / "t.sqlite3")
    specs = [{"a": 1}, {"a": 2}]
    assert not led.evaluated_with("X", specs, "fp", code_hash())
    led.record(family="X", variant="a=1", spec=specs[0], data_fingerprint="fp", stage="DEV", metrics={}, status="OK")
    assert not led.evaluated_with("X", specs, "fp", code_hash())
    led.record(family="X", variant="a=2", spec=specs[1], data_fingerprint="fp", stage="DEV", metrics={}, status="OK")
    assert not led.evaluated_with("X", specs, "fp", code_hash())  # trials exist but the run never completed
    led.record(family="X", variant="RUN_COMPLETE:r1", spec={"run_of": "X"}, data_fingerprint="fp", stage="RUN",
               metrics={}, status="DONE")
    assert led.evaluated_with("X", specs, "fp", code_hash())
    assert led.count("X") == 2  # completion markers never count as trials (deflated Sharpe unaffected)
    assert not led.evaluated_with("X", specs, "fp-new", code_hash())
    assert not led.evaluated_with("X", specs, "fp", "other-code")


def test_dev_content_fingerprint_ignores_rows_after_the_slice_and_rebasing() -> None:
    days = get_calendar().sessions_in_range(date(2018, 1, 2), date(2019, 6, 28))
    first = pd.Timestamp(days[0])
    raw, adj = synth.bars({s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 5e7} for s in SYMS},
                          n=len(days), start=str(first.date()))
    assets = synth.assets(list(SYMS), etfs=SYMS)
    base = EquityPanel.build(raw, adj, assets, min_obs=50)
    cut = pd.Timestamp(days[-60])
    fp = base.slice_dates(None, cut).content_fingerprint()
    rebased = adj.copy()  # a later split/dividend re-bases a symbol's WHOLE adjusted history
    m = rebased["symbol"] == "SPY"
    for col in ("o", "h", "l", "c", "vw"):
        rebased.loc[m, col] = rebased.loc[m, col] * 0.987
    assert EquityPanel.build(raw, rebased, assets, min_obs=50).slice_dates(None, cut).content_fingerprint() == fp
    later = raw.copy()  # rows after the slice change ...
    late = later["t"] > pd.Timestamp(cut).tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(days=1)
    later.loc[late, "c"] = later.loc[late, "c"] * 1.5
    assert EquityPanel.build(later, adj, assets, min_obs=50).slice_dates(None, cut).content_fingerprint() == fp
    fixed = raw.copy()  # ... but a corrected development bar is new evidence
    fixed.loc[fixed.index[5], "c"] = fixed.loc[fixed.index[5], "c"] * 1.01
    assert EquityPanel.build(fixed, adj, assets, min_obs=50).slice_dates(None, cut).content_fingerprint() != fp


def test_duplicate_guard_respects_methodology_and_canonical_expressions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from services.quantlab.factors import validate_spec

    a = validate_spec({"expression": "rank(mom(126,21)) - rank(vol(63))"})
    b = validate_spec({"expression": "(rank(mom(126, 21.0)))-rank( vol(63) )"})
    assert a == b  # one spelling per hypothesis
    prior = [{"data_fingerprint": "full|dev:abc|m:m1", "status": "FAIL"}]
    assert duplicate_on_same_evidence(prior, fingerprint="full2", dev_fingerprint="abc", methodology="m1")
    assert duplicate_on_same_evidence(prior, fingerprint="full2", dev_fingerprint="abc", methodology="m2") is None
    legacy = [{"data_fingerprint": "full", "status": "FAIL"}]  # recorded before dev/methodology keys existed
    assert duplicate_on_same_evidence(legacy, fingerprint="full", dev_fingerprint="abc", methodology="m1")
    assert duplicate_on_same_evidence(legacy, fingerprint="other", dev_fingerprint="abc", methodology="m1") is None


def test_multi_session_exposure_and_policy_tier_gating(tmp_path, world) -> None:  # type: ignore[no-untyped-def]
    from services.quantlab.pipeline import Lab

    ip, ctx, days = world
    d = _day_index(ip, date(2018, 12, 3))
    leg = Leg(d, 0, slot_of("15:20"), d + 2, flatten_slot(77), stop_bps=5000.0)
    res, outs = simulate(ip, ctx, [leg], SimConfig(weight=1.0, max_concurrent=1))
    i = list(res.dates).index(ip.days[d])
    assert res.exposure[i + 1] == pytest.approx(1.0) and 0 < res.exposure[i] < 0.1  # held all of d+1
    assert len(outs[0].marks) == 2
    nxt = S.i1_legs(S.SessionFeatures(ip, ctx), {"exit": "next_eod", "filter": "always", "universe": "SPY"})
    assert nxt and all(lg.exit_slot == flatten_slot(int(ip.last_slot[lg.exit_day])) for lg in nxt)
    assert S.lifecycle_executable(S.I1, {"exit": "09:35"})[0] is False
    assert S.lifecycle_executable(S.I1, {"exit": "next_eod"})[0] is True
    (tmp_path / "datasets").mkdir()
    lab = Lab(tmp_path)
    specs = {f.family: f for f in lab.session_families(ip, _daily(days), families=(S.I1, S.I2, S.H1))}
    early = lab.policy_tier(specs[S.I1], None, {"exit": "09:35", "filter": "always", "universe": "SPY"})  # type: ignore[arg-type]
    late = lab.policy_tier(specs[S.I1], None, {"exit": "next_eod", "filter": "always", "universe": "SPY"})  # type: ignore[arg-type]
    tom = lab.policy_tier(specs[S.I2], None, next(iter(S.VARIANTS[S.I2]())))  # type: ignore[arg-type]
    assert early["status"] == "NOT_EXECUTABLE" and late["status"] == "NOT_PERMITTED" and tom["status"] == "NOT_PERMITTED"
    assert lab.policy_tier(specs[S.H1], None, next(iter(S.VARIANTS[S.H1]())))["status"] == "OK"  # type: ignore[arg-type]


def test_fingerprint_ignores_columns_without_dev_rows_and_column_order() -> None:
    days = get_calendar().sessions_in_range(date(2018, 1, 2), date(2019, 6, 28))
    raw, adj = synth.bars({s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 5e7} for s in SYMS},
                          n=len(days), start=str(days[0]))
    cut = pd.Timestamp(days[-60])
    base = EquityPanel.build(raw, adj, synth.assets(list(SYMS), etfs=SYMS), min_obs=50)
    fp = base.slice_dates(None, cut).content_fingerprint()
    # a listing that only appears after the slice (it crossed min_obs through a later delta) adds a column
    new_raw, new_adj = synth.bars({"NEWCO": {"price": 20, "volume": 1e6}}, n=55, start=str(days[-56]), seed=9)
    syms = (*SYMS, "NEWCO")
    wider = EquityPanel.build(pd.concat([raw, new_raw]), pd.concat([adj, new_adj]),
                              synth.assets(list(syms), etfs=SYMS), min_obs=50)
    assert "NEWCO" in set(wider.symbols.tolist())
    assert wider.slice_dates(None, cut).content_fingerprint() == fp
    sub, _ = base.subset(list(reversed(SYMS)))  # column order does not matter
    assert sub.slice_dates(None, cut).content_fingerprint() == fp


def test_forced_exit_never_drops_a_held_multi_session_leg(world) -> None:  # type: ignore[no-untyped-def]
    ip, ctx, _ = world
    p2 = replace(ip, f={k: v.copy() for k, v in ip.f.items()})
    d = _day_index(p2, date(2018, 12, 3))
    for k in ("o", "h", "l", "c", "vw"):
        p2.f[k][d + 2, :, 0] = np.nan  # no bars at all on the planned exit session
    leg = Leg(d, 0, slot_of("15:20"), d + 2, flatten_slot(77), stop_bps=5000.0)
    res, outs = simulate(p2, ctx, [leg], SimConfig(weight=1.0, max_concurrent=1, costs=EquityCosts(cost_multiplier=0)))
    o = outs[0]
    assert o.filled and o.exit_reason == "FORCED" and o.exit_day_actual == d + 1  # last close before the gap
    assert sum(res.gross) == pytest.approx(o.gross) and o.gross != 0.0


def test_holdout_lease_refuses_a_concurrent_evaluator_and_result_is_written_once(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import os

    from services.quantlab.ledger import HoldoutLookRefused, HoldoutVault

    v = HoldoutVault(tmp_path / "h.sqlite3")
    v.register("h", date(2025, 11, 21), date(2026, 11, 20))
    spec = {"a": 1}
    look = v.claim("h", "X", spec, "X:1")
    assert v.unresolved_look("h", "X", spec) == look  # own process may resume its own open look
    import sqlite3
    with sqlite3.connect(tmp_path / "h.sqlite3") as db:  # pretend a different, LIVE process owns it
        db.execute("UPDATE looks SET pid=? WHERE id=?", (os.getppid(), look))
    with pytest.raises(HoldoutLookRefused, match="live process"):
        v.unresolved_look("h", "X", spec)
    with sqlite3.connect(tmp_path / "h.sqlite3") as db:  # owner gone (pid that cannot exist)
        db.execute("UPDATE looks SET pid=? WHERE id=?", (2 ** 22 + 12345, look))
    assert v.unresolved_look("h", "X", spec) == look
    v.record_result(look, {"ok": True})
    with pytest.raises(HoldoutLookRefused):
        v.record_result(look, {"ok": False})  # first writer wins, atomically
    assert v.unresolved_look("h", "X", spec) is None


def test_legacy_spelling_of_a_failed_factor_is_still_recognized(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from services.quantlab.factors import validate_spec
    from services.quantlab.ledger import variant_name

    led = TrialLedger(tmp_path / "t.sqlite3")
    legacy = {"expression": "rank(mom(252,21)) - rank(vol(63))", "universe": "large300", "n": 20, "rebalance": 21,
              "regime": "none"}  # whitespace-collapsed spelling recorded before canonicalization existed
    led.record(family="F_agent", variant=variant_name(legacy), spec=legacy, data_fingerprint="full", stage="DEV",
               metrics={}, status="OK")
    led.record(family="F_agent", variant="WALK_FORWARD:" + variant_name(legacy), spec={"wf_of": "F_agent"},
               data_fingerprint="full", stage="WF", metrics={}, status="FAIL")
    now = validate_spec(legacy)
    assert now["expression"] != legacy["expression"]
    assert led.prior_outcomes("F_agent", now) == []
    assert [r["status"] for r in led.prior_outcomes("F_agent", now, canonicalize=validate_spec)] == ["FAIL"]


def test_canonicalization_never_produces_an_invalid_spec() -> None:
    from services.quantlab.factors import SpecError, validate_spec

    assert validate_spec({"expression": "rank(mom(126)) * 1e300"})["expression"] == "rank(mom(126)) * 1e+300"
    with pytest.raises(SpecError):
        validate_spec({"expression": "rank(mom(126)) * " + "9" * 290})


def test_policy_tier_uses_the_policy_and_equity_it_is_given(tmp_path, world) -> None:  # type: ignore[no-untyped-def]
    from packages.contracts.policy import RiskPolicy
    from services.quantlab.ledger import spec_hash
    from services.quantlab.pipeline import Lab

    ip, _ctx, days = world
    (tmp_path / "datasets").mkdir()
    lab = Lab(tmp_path)
    spec = lab.session_families(ip, _daily(days), families=(S.I1,))[0]
    prm = {"exit": "next_eod", "filter": "always", "universe": "SPY"}
    assert lab.policy_tier(spec, None, prm)["status"] == "NOT_PERMITTED"  # type: ignore[arg-type]
    pol = RiskPolicy(allow_overnight_positions=True, multi_day_strategy_allowlist=(f"{S.I1}:{spec_hash(prm)}",))
    ok = lab.policy_tier(spec, None, prm, policy=pol, equity=20_000.0)  # type: ignore[arg-type]
    assert ok["status"] == "OK" and ok["equity"] == 20_000.0 and ok["metrics"]["trades"] > 0


def test_holdout_rebase_removes_the_fake_boundary_gap_and_refuses_without_overlap() -> None:
    from services.quantlab.data import rebase_onto

    cal = get_calendar()
    dev_days = cal.sessions_in_range(date(2025, 10, 1), date(2025, 11, 20))
    new_days = cal.sessions_in_range(date(2025, 11, 7), date(2025, 12, 31))  # overlaps the last dev sessions
    old = _intraday_bars(dev_days + [d for d in new_days if d > dev_days[-1]], ("SPY",), extended=False)
    dev = old[old["t"] < pd.Timestamp("2025-11-21", tz="UTC")]
    newer = old[old["t"] >= pd.Timestamp("2025-11-07", tz="UTC")].copy()
    for col in ("o", "h", "l", "c", "vw"):  # a dividend between the two downloads re-bases the newer one
        newer[col] = newer[col] * 1.01
    naive = IntradayPanel.build(pd.concat([dev, newer[newer["t"] >= pd.Timestamp("2025-11-21", tz="UTC")]]))
    good, missing = rebase_onto(dev, newer, ["SPY"])
    assert missing == []
    rebased = IntradayPanel.build(good)
    i = int(np.where(rebased.days == pd.Timestamp("2025-11-21"))[0][0])
    def gap(p: IntradayPanel) -> float:
        return float(p.first_open()[i, 0] / p.prev_close()[i, 0] - 1)

    truth = gap(IntradayPanel.build(old))
    assert abs(gap(naive) - truth) > 0.009  # joining without re-basing invents a ~1% gap
    assert gap(rebased) == pytest.approx(truth, abs=1e-6)  # float32 panel storage
    empty, missing2 = rebase_onto(dev, newer, ["SPY", "QQQ"])
    assert missing2 == ["QQQ"] and empty.empty
