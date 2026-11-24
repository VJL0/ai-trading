from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.quantlab.backtest import BacktestConfig, apply_rebalance_schedule, hold_schedule, run_weights
from services.quantlab.costs import CryptoCosts, EquityCosts
from services.quantlab.panel import EquityPanel, infer_missing_kinds
from services.quantlab.strategies import momentum as A

from . import synth

SYMS = {"SPY": {"drift": 0.0004, "vol": 0.01, "price": 300, "volume": 8e7},
        "AAA": {"drift": 0.0010, "vol": 0.02, "price": 40, "volume": 5e6},
        "BBB": {"drift": -0.0005, "vol": 0.02, "price": 60, "volume": 5e6},
        "CCC": {"drift": 0.0002, "vol": 0.015, "price": 20, "volume": 5e6}}


def _panel(**kw) -> EquityPanel:  # type: ignore[no-untyped-def]
    raw, adj = synth.bars(SYMS, **kw)
    return EquityPanel.build(raw, adj, synth.assets(list(SYMS)), min_obs=10)


def test_panel_returns_adv_and_segments() -> None:
    p = _panel(n=300)
    j = p.col("AAA")
    c = p.f["adj_close"][:, j]
    np.testing.assert_allclose(p.f["ret_cc"][1:, j], c[1:] / c[:-1] - 1, rtol=1e-9)
    # ADV at t uses sessions < t only
    dv = p.f["close"][:, j] * p.f["volume"][:, j]
    assert p.f["adv20"][30, j] == pytest.approx(dv[10:30].mean())
    assert p.f["obs_count"][0, j] == 1 and p.f["obs_count"][-1, j] == 300


def test_gap_starts_new_segment_and_no_return_across_it() -> None:
    p = _panel(n=200, drop={"AAA": list(range(100, 115))})
    j = p.col("AAA")
    assert np.isnan(p.f["ret_cc"][115, j])  # first bar after a 15-session hole
    assert p.f["obs_count"][115, j] == 1
    assert p.f["segment"][114, j] != p.f["segment"][115, j] or np.isnan(p.f["segment"][114, j])


def test_delisted_kind_inference() -> None:
    known = pd.Series({"AAPL": "COMMON_STOCK"})
    out = infer_missing_kinds(np.array(["AAPL", "SIVB", "BRK.B", "ABCDW", "ABCDE"], dtype=object), known)
    assert list(out) == ["COMMON_STOCK", "COMMON_STOCK", "COMMON_STOCK", "OTHER", "COMMON_STOCK"]


def test_backtest_matches_hand_computation_and_costs_reduce() -> None:
    p = _panel(n=120)
    T, N = p.shape
    w = np.zeros((T, N))
    j = p.col("SPY")
    w[:, j] = 1.0
    free = run_weights(p.f, p.dates, w, BacktestConfig(costs=EquityCosts(cost_multiplier=0.0)))
    # fully invested from the first open: day 1 = intraday only, later days = close-to-close
    exp = np.r_[p.f["ret_oc"][1, j], p.f["ret_cc"][2:, j]]
    np.testing.assert_allclose(free.net, exp, rtol=1e-9, atol=1e-12)
    costly = run_weights(p.f, p.dates, w, BacktestConfig())
    assert costly.net[0] < free.net[0]
    np.testing.assert_allclose(costly.net[1:], free.net[1:], atol=1e-12)  # no trades after entry
    assert free.trades == 1


def test_no_lookahead_future_prices_do_not_change_past() -> None:
    p = _panel(n=200)
    q = _panel(n=200)
    k = 150
    for key in ("close", "adj_close", "open", "adj_open", "high", "low"):
        q.f[key][k:] *= 1.5
    q._derive(max_gap_sessions=10, max_abs_log_ret=5.0)
    p._derive(max_gap_sessions=10, max_abs_log_ret=5.0)
    cache_p, cache_q = A.ScoreCache(p), A.ScoreCache(q)
    par = {"universe": "liquid1000", "score": "mom_3", "n": 1, "rebalance": 5, "regime": "none"}
    for c in (cache_p, cache_q):
        c._u["liquid1000"] = np.isfinite(c.panel.f["close"]) & (np.arange(c.panel.shape[1]) != c.panel.col("SPY"))
    tp, tq = A.targets(cache_p, par), A.targets(cache_q, par)
    np.testing.assert_array_equal(np.nan_to_num(tp[:k]), np.nan_to_num(tq[:k]))
    rp = run_weights(p.f, p.dates, tp).net
    rq = run_weights(q.f, q.dates, tq).net
    np.testing.assert_allclose(rp[: k - 2], rq[: k - 2])


def test_non_fill_when_no_bar_and_hold_rows() -> None:
    p = _panel(n=60, drop={"AAA": [10]})
    T, N = p.shape
    j = p.col("AAA")
    w = np.zeros((T, N))
    w[9:, j] = 1.0  # decided at close 9, AAA has no bar at session 10 -> cannot buy until 11
    res = run_weights(p.f, p.dates, w, BacktestConfig(costs=EquityCosts(cost_multiplier=0.0)))
    assert res.non_fills >= 1
    assert res.exposure[9] == 0.0 and res.exposure[10] == 1.0
    held = apply_rebalance_schedule(w, hold_schedule(T, 20))
    assert np.isnan(held[1]).all() and not np.isnan(held[0]).all()


def test_long_only_clips_and_gross_cap() -> None:
    p = _panel(n=40)
    T, N = p.shape
    w = np.full((T, N), 0.5)
    w[:, 0] = -1.0
    res = run_weights(p.f, p.dates, w, BacktestConfig())
    assert res.exposure.max() <= 1.0 + 1e-9


def test_equity_cost_buckets_and_crypto_fees() -> None:
    c = EquityCosts()
    spread = c.full_spread_bps(np.array([2e9, 2e8, 2e7, 1e6, np.nan]))
    assert list(spread) == [2.0, 5.0, 15.0, 50.0, 30.0]
    buy = c.one_way_bps(np.array([2e9]), np.array([10_000.0]), np.array([100.0]), sell=np.array([False]),
                        timing="close")
    sell = c.one_way_bps(np.array([2e9]), np.array([10_000.0]), np.array([100.0]), sell=np.array([True]),
                         timing="close")
    assert sell[0] > buy[0] > 2.0
    assert CryptoCosts().one_way_bps("BTC/USD") == pytest.approx(25 + 5 + 2)


def test_delta_merge_rescales_adjusted_basis_after_a_split() -> None:
    from services.quantlab.data import merge_with_deltas

    t = pd.date_range("2026-01-05", periods=6, freq="B", tz="UTC")
    base = pd.DataFrame({"symbol": "X", "t": t[:4], "o": 100.0, "h": 100.0, "l": 100.0, "c": [100.0, 101, 102, 103],
                         "v": 1e6, "n": 1, "vw": 100.0})
    # a 2:1 split after the base fetch: the new back-adjusted basis halves every old price
    delta = pd.DataFrame({"symbol": "X", "t": t[2:], "o": 51.0, "h": 51.0, "l": 51.0, "c": [51.0, 51.5, 52, 52.5],
                          "v": 2e6, "n": 1, "vw": 51.0})
    out = merge_with_deltas(base, [delta], adjusted=True)
    assert list(out["c"]) == pytest.approx([50.0, 50.5, 51.0, 51.5, 52.0, 52.5])
    raw = merge_with_deltas(base, [delta], adjusted=False)
    assert len(raw) == 6 and raw["c"].iloc[0] == 100.0


def test_exlisted_candidates_recover_failed_listed_tickers() -> None:
    from services.quantlab.data import exlisted_candidates

    assets = pd.DataFrame({
        "symbol": ["FRCB", "SBNY", "ABCDQ", "AAPL", "ZZZZF"],
        "asset_class": "us_equity",
        "exchange": ["OTC", "OTC", "NASDAQ", "NASDAQ", "OTC"],
        "status": ["active", "active", "inactive", "active", "active"],
    })
    delisted = pd.DataFrame({"symbol": ["SIVBQ"], "event": ["worthless_removals"], "date": ["2024-11-08"],
                             "new_symbol": [None]})
    c = exlisted_candidates(assets, known={"AAPL"}, delisted=delisted)
    assert {"FRC", "FRCB", "SBNY", "ABCD", "SIVB", "ZZZZ"} <= set(c)
    assert "AAPL" not in c  # already in the base universe
