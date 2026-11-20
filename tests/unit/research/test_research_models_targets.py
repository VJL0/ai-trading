"""Model-suite features/targets: point-in-time features, label conventions, net-of-cost targets, rank targets."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from packages.core.money import floor_shares
from services.research.costs import CostModel
from services.research.frames import prepare_bars, trailing_adv_shares, ts_ns
from services.research.models.config import SuiteConfig
from services.research.models.features import (
    intra_session_r1,
    market_blocks,
    symbol_model_features,
    vol_inputs,
    vol_spec_for,
)
from services.research.models.targets import (
    QuoteSpreads,
    build_suite_dataset,
    cross_sectional_rank,
    describe_targets,
)

from .test_research_models_support import fast_config, noise_bars


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return prepare_bars(noise_bars(("AAA", "BBB"), sessions=6, seed=21, latency_s=30.0))


@pytest.fixture(scope="module")
def cfg() -> SuiteConfig:
    return fast_config()


def test_features_at_t_use_only_rows_available_by_t(bars: pd.DataFrame, cfg: SuiteConfig) -> None:
    sb = bars.loc[bars["symbol"] == "AAA"].reset_index(drop=True)
    full = symbol_model_features(sb, cfg.feature_spec)
    vs = vol_spec_for("1Min", 60, None, cfg.har_lags, cfg.ewma_lambda)
    vfull = vol_inputs(sb, vs)
    avail = ts_ns(sb["available_at_ts"])
    for t in (150, 700, 1500, len(sb) - 2):
        decision = avail[t]
        visible = sb.loc[avail <= decision].reset_index(drop=True)  # exactly what is knowable at the decision
        assert len(visible) == t + 1
        trunc = symbol_model_features(visible, cfg.feature_spec)
        np.testing.assert_array_equal(full.iloc[t].to_numpy(), trunc.iloc[-1].to_numpy())
        vt = vol_inputs(visible, vs)
        np.testing.assert_array_equal(vfull.iloc[t].to_numpy(), vt.iloc[-1].to_numpy())


def test_intra_session_returns_exclude_overnight_gaps(bars: pd.DataFrame) -> None:
    sb = bars.loc[bars["symbol"] == "AAA"].reset_index(drop=True)
    r1 = intra_session_r1(sb)
    days = list(sb["session_date"])
    first = [i for i in range(1, len(sb)) if days[i] != days[i - 1]]
    assert first, "fixture must span several sessions"
    for i in first:
        assert r1[i] == pytest.approx(np.log(sb["close_f"].iloc[i] / sb["open_f"].iloc[i]))
    assert r1[5] == pytest.approx(np.log(sb["close_f"].iloc[5] / sb["close_f"].iloc[4]))


def test_labels_follow_next_open_convention_net_of_costs(bars: pd.DataFrame, cfg: SuiteConfig) -> None:
    ds = build_suite_dataset(bars, CostModel(), cfg)
    assert len(ds) > 1000
    panel = ds.panel
    sb = bars.loc[bars["symbol"] == "BBB"].reset_index(drop=True)
    start = ts_ns(sb["start_ts"])
    bbb_rows = np.flatnonzero(panel["symbol"].to_numpy() == "BBB")
    for r in (int(bbb_rows[0]), int(bbb_rows[400]), int(bbb_rows[900])):
        bi = int(panel.at[r, "bar_index"])
        decision = int(ds.decision_ns[r])
        e = int(np.searchsorted(start, decision, side="left"))
        assert e == bi + 2, "30 s publication latency: the next bar has already opened, enter one bar later"
        assert start[e] >= decision
        for h, H in ds.horizons_bars.items():
            gross = (sb["open_f"].iloc[e + H] / sb["open_f"].iloc[e] - 1) * 10_000
            assert ds.gross_bps[h][r] == pytest.approx(gross)
            assert ds.net_bps[h][r] == pytest.approx(gross - ds.cost_rt_bps[r])
            assert sb["session_date"].iloc[e + H] == sb["session_date"].iloc[bi], "label stays inside the session"
        # round-trip cost recomputed independently with the Decimal cost model (PIT close + trailing ADV)
        price = sb["close"].iloc[bi].quantize(Decimal("0.01"))
        adv = float(trailing_adv_shares(sb)[bi])
        qty = max(Decimal(1), floor_shares(cfg.reference_notional / price))
        rt = CostModel().estimate_round_trip("BBB", qty, price, adv_shares=Decimal(f"{float(f'{adv:.3g}'):.0f}"))
        assert ds.cost_rt_bps[r] == pytest.approx(float(rt.total_bps))
        assert ds.label_end_ns[r] > decision
    sh = cfg.success_horizon_minutes
    np.testing.assert_array_equal(ds.success, (ds.net_bps[sh] > 0).astype(np.int64))


def test_rows_with_unknown_liquidity_are_dropped(bars: pd.DataFrame, cfg: SuiteConfig) -> None:
    ds = build_suite_dataset(bars, CostModel(), cfg)
    first_day = min(bars["session_date"])
    days = pd.to_datetime(ds.panel["decision_ns"], utc=True).dt.tz_convert("America/New_York").dt.date
    assert first_day not in set(days), "no completed-session ADV on the first session: fail closed, never filled"


def test_cross_sectional_rank_target() -> None:
    t = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64)
    v = np.array([5.0, -1.0, 2.0, 0.0, 3.0, 9.0])
    pct, rel = cross_sectional_rank(t, v, levels=5)
    np.testing.assert_allclose(pct[:3], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(pct[3:5], [0.0, 1.0])
    assert np.isnan(pct[5]) and rel[5] == -1, "a lone symbol has no cross-sectional rank"
    assert list(rel[:5]) == [4, 0, 2, 0, 4]


def test_quote_spreads_are_point_in_time() -> None:
    q = QuoteSpreads.from_frame(pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA"], "available_at_ns": [100, 200, 300], "spread_bps": [1.0, 2.0, 3.0]}))
    got = q.lookup("AAA", np.array([50, 100, 250, 299, 10_000]), max_age_ns=500)
    assert np.isnan(got[0]), "nothing published yet"
    assert list(got[1:4]) == [1.0, 2.0, 2.0], "never a quote published after the query time"
    assert np.isnan(got[4]), "stale quote ignored"
    assert np.isnan(q.lookup("ZZZ", np.array([1]), 10)[0])


def test_quotes_change_costs_and_targets(bars: pd.DataFrame, cfg: SuiteConfig) -> None:
    syms = ["AAA", "BBB"]
    avail = ts_ns(bars["available_at_ts"])
    q = QuoteSpreads.from_frame(pd.DataFrame({"symbol": bars["symbol"].to_numpy(), "available_at_ns": avail - 5_000_000_000,
                                              "spread_bps": np.full(len(bars), 20.0)}))
    wide = build_suite_dataset(bars, CostModel(), cfg, quotes=q)
    base = build_suite_dataset(bars, CostModel(), cfg)
    assert set(wide.symbols) == set(syms)
    assert np.median(wide.cost_rt_bps) > np.median(base.cost_rt_bps) + 10, "20 bps quoted spread costs more"
    assert "quotes" in wide.spread_source and "estimate" in base.spread_source


def test_market_blocks_are_complete_and_ordered(bars: pd.DataFrame) -> None:
    b = market_blocks(bars, 5)
    assert len(b) > 300
    assert (np.diff(b.avail_ns) > 0).all()
    assert (b.avail_ns >= b.end_ns).all()


def test_forward_labels_match_research_labels_at_own_availability(bars: pd.DataFrame) -> None:
    from services.research.features import symbol_labels
    from services.research.models.targets import forward_labels

    sb = bars.loc[bars["symbol"] == "AAA"].reset_index(drop=True)
    ref = symbol_labels(sb, 15, same_session=True)
    got = forward_labels(sb, ts_ns(sb["available_at_ts"]), 15, same_session=True)
    for col in ("entry_idx", "exit_idx", "label_start_ns", "label_end_ns", "valid"):
        np.testing.assert_array_equal(got[col], ref[col].to_numpy())
    np.testing.assert_array_equal(got["fwd_ret"], ref["fwd_ret"].to_numpy())


def test_aligned_decisions_wait_for_the_whole_cross_section(cfg: SuiteConfig) -> None:
    from services.research.models.targets import aligned_decisions

    from .test_research_models_support import _minute_spans, bars_from_log_returns

    starts, ends, mos = _minute_spans(5)
    rng = np.random.RandomState(3)
    bodies = {s: rng.normal(0, 5e-4, len(mos)) for s in ("AAA", "BBB", "CCC")}
    # live-recorded style: per-symbol publication latency; CCC arrives 70 s late (after the next bar opened)
    raw = bars_from_log_returns(bodies, starts, ends, dataset_version="synthetic-latency-test", seed=3,
                                latency_s={"AAA": 1.0, "BBB": 20.0, "CCC": 70.0})
    b = prepare_bars(raw)
    dec = aligned_decisions(ts_ns(b["start_ts"]), ts_ns(b["available_at_ts"]))
    assert (dec >= ts_ns(b["available_at_ts"])).all(), "never decide before a row's own data is published"
    slot = ts_ns(b["start_ts"])
    for s in np.unique(slot)[:50]:
        m = slot == s
        assert len(set(dec[m].tolist())) == 1, "one decision time per bar slot"
        assert dec[m][0] == ts_ns(b["available_at_ts"])[m].max(), "the cross-section is complete only then"
    uniq = np.unique(dec)
    assert len(uniq) == len(np.unique(slot)), "decisions strictly increasing across slots"
    ds = build_suite_dataset(b, CostModel(), cfg)
    groups = ds.panel.groupby("decision_ns")["symbol"].nunique().to_numpy()
    assert float(np.mean(groups == 3)) > 0.99, "rank targets compare all three symbols at one decision time"
    assert np.isfinite(ds.rank_pct).mean() > 0.99
    # entry = first bar opening at/after the COMMON decision: 70 s latency -> AAA/BBB enter one bar later as well
    aaa = b.loc[b["symbol"] == "AAA"].reset_index(drop=True)
    r = int(np.flatnonzero(ds.panel["symbol"].to_numpy() == "AAA")[100])
    bi = int(ds.panel.at[r, "bar_index"])
    e = int(np.searchsorted(ts_ns(aaa["start_ts"]), ds.decision_ns[r], side="left"))
    assert e == bi + 3, "decision at t_end + 70 s: bars t+1 and t+2 already opened"
    h60 = ds.horizons_bars[60]
    gross = (aaa["open_f"].iloc[e + h60] / aaa["open_f"].iloc[e] - 1) * 10_000
    assert ds.gross_bps[60][r] == pytest.approx(gross)
    delayed = describe_targets(cfg, ds)["decisions_delayed_by_cross_section"]
    assert 0.6 * len(ds) < delayed < len(ds), "AAA and BBB rows wait for CCC; CCC rows do not"


def test_har_base_period_is_literal_only_with_enough_history() -> None:
    from services.research.models.targets import choose_har_base_minutes

    cfg = fast_config()
    base, why = choose_har_base_minutes(20, cfg)
    assert base == cfg.vol_horizon_minutes and why.startswith("FALLBACK")
    base, why = choose_har_base_minutes(cfg.har_lags[2] + cfg.har_min_fit_sessions, cfg)
    assert base == 390 and why.startswith("literal HAR")
    base, why = choose_har_base_minutes(5, fast_config(har_base_minutes=120))
    assert base == 120 and why.startswith("configured")
    with pytest.raises(ValueError):
        fast_config(har_base_minutes=0)


def test_literal_har_components_are_session_sums() -> None:
    from services.research.models.features import intra_session_r1

    long = prepare_bars(noise_bars(("AAA",), sessions=33, seed=4))
    cfg = fast_config()
    ds = build_suite_dataset(long, CostModel(), cfg)
    assert ds.vol_spec.base_bars == 390 and ds.har_base_reason.startswith("literal HAR")
    sb = long.reset_index(drop=True)
    sq = intra_session_r1(sb) ** 2
    r = int(np.flatnonzero(ds.vol_valid)[-1])
    bi = int(ds.panel.at[r, "bar_index"])
    np.testing.assert_allclose(ds.vol_X[r, 0], sq[bi - 389: bi + 1].sum(), rtol=1e-10)
    np.testing.assert_allclose(ds.vol_X[r, 1], sq[bi - 5 * 390 + 1: bi + 1].sum() / 5, rtol=1e-10)
    np.testing.assert_allclose(ds.vol_X[r, 2], sq[bi - 22 * 390 + 1: bi + 1].sum() / 22, rtol=1e-10)
