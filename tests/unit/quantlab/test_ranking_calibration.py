"""Ranking labels/diagnostics, edge calibration, conformal bands and the cost-aware selector on SIMULATED arrays.

Synthetic inputs only: these tests check arithmetic, point-in-time behaviour and decision rules, never evidence
of any edge."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from services.quantlab import calibration as C
from services.quantlab import ranking as R
from services.quantlab.costs import EquityCosts
from services.quantlab.portfolio import round_trip_cost_bps, select_cost_aware


def _returns(T: int = 40, N: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"ret_oc": rng.normal(0, 0.01, (T, N)), "ret_co": rng.normal(0, 0.005, (T, N))}


def test_executable_forward_return_runs_open_to_open_after_the_decision() -> None:
    f = _returns()
    for h in (1, 3):
        out = R.executable_forward_return(f, h)
        T = out.shape[0]
        for t in (0, 5, T - h - 2):
            want = np.prod(1 + f["ret_oc"][t + 1: t + h + 1], axis=0) * np.prod(1 + f["ret_co"][t + 2: t + h + 2], axis=0)
            np.testing.assert_allclose(out[t], want - 1, rtol=1e-10)
        assert np.isnan(out[T - h - 1:]).all()  # the last open is not observed


def test_executable_label_ignores_the_decision_day_and_the_gap_into_t_plus_1() -> None:
    f = _returns()
    base = R.executable_forward_return(f, 2)
    g = {k: v.copy() for k, v in f.items()}
    t = 10
    g["ret_oc"][t] += 0.5  # the decision day's own session is history
    g["ret_co"][t + 1] += 0.5  # the overnight gap into t+1's open cannot be earned by an order placed after t's close
    np.testing.assert_allclose(R.executable_forward_return(g, 2)[t], base[t])


def test_group_demean_uses_group_means_and_falls_back_to_row_mean() -> None:
    x = np.array([[1.0, 2.0, 3.0, 10.0, 20.0, np.nan]])
    mask = np.ones_like(x, dtype=bool)
    groups = np.array([[0, 0, 0, 1, 1, 1]])
    out = R.group_demean(x, mask, groups, min_group=3)
    np.testing.assert_allclose(out[0, :3], [-1.0, 0.0, 1.0])
    row_mean = np.nanmean(x)
    np.testing.assert_allclose(out[0, 3:5], [10 - row_mean, 20 - row_mean])  # group 1 has only 2 finite names
    assert np.isnan(out[0, 5])


def test_relevance_grades_and_rank_ic() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=(6, 200))
    mask = np.ones_like(y, dtype=bool)
    mask[:, :5] = False
    g = R.relevance_grades(y, mask, levels=10)
    assert g.min() == -1 and g[:, 5:].min() == 0 and g.max() == 9
    assert (g[:, :5] == -1).all()
    ic = R.rank_ic(y, y, mask, np.arange(6))
    np.testing.assert_allclose(ic, 1.0)
    ic_neg = R.rank_ic(-y, y, mask, np.arange(6))
    np.testing.assert_allclose(ic_neg, -1.0)


def test_quantile_spread_and_turnover() -> None:
    rng = np.random.default_rng(2)
    score = rng.normal(size=(30, 100))
    label = 0.001 * score + rng.normal(0, 1e-4, size=score.shape)
    mask = np.ones_like(score, dtype=bool)
    rows = np.arange(30)
    qm = R.quantile_means(score, label, mask, rows, q=5)
    s = R.spread_summary(qm, pd.date_range("2020-01-01", periods=30, freq="B"))
    assert s["top_minus_bottom_mean"] > 0 and s["monotonicity"] > 0.99
    assert R.top_set_turnover(np.tile(score[:1], (5, 1)), mask[:5], np.arange(5), 10) == 0.0
    assert R.paired_difference(np.ones(10) * 0.1, np.zeros(10))["mean_diff"] == pytest.approx(0.1)


def test_decision_rows_leave_room_for_the_label() -> None:
    dates = pd.bdate_range("2020-01-01", periods=50)
    rows = R.decision_rows(dates, 5, pd.Timestamp("2020-01-10"))
    assert rows[0] == int(np.searchsorted(dates.values, np.datetime64("2020-01-10")))
    assert np.all(np.diff(rows) == 5) and rows[-1] + 5 + 1 < 50


def test_isotonic_matches_pool_adjacent_violators() -> None:
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(3)
    y = rng.normal(size=25)
    w = rng.uniform(0.5, 2.0, size=25)
    ref = IsotonicRegression().fit(np.arange(25), y, sample_weight=w).predict(np.arange(25))
    np.testing.assert_allclose(C._isotonic(y, w), ref, atol=1e-12)  # pyright: ignore[reportPrivateUsage]


def test_binned_edge_recovers_a_monotone_edge_with_date_level_errors() -> None:
    rng = np.random.default_rng(4)
    dates, names = 400, 100
    pct = np.tile(np.arange(1, names + 1) / names, dates)
    row = np.repeat(np.arange(dates), names)
    day_shock = np.repeat(rng.normal(0, 0.002, dates), names)  # common within a date: dates are the sample
    realized = 0.002 * (pct - 0.5) + day_shock + rng.normal(0, 0.01, len(pct))
    e1 = C.fit_binned_edge(pct, realized, row, n_bins=10, overlap=1.0)
    assert np.all(np.diff(e1.mean) >= -1e-15)
    assert e1.mean[-1] > e1.mean[0]
    assert e1.mean[-1] == pytest.approx(0.002 * 0.45, abs=0.0006)
    e5 = C.fit_binned_edge(pct, realized, row, n_bins=10, overlap=5.0)
    np.testing.assert_allclose(e5.se / e1.se, math.sqrt(5), rtol=1e-9)  # overlapping labels: fewer effective dates
    mu, se = e1.predict(np.array([0.99, 0.01, np.nan]))
    assert mu[0] > mu[1] and np.isnan(mu[2]) and np.isnan(se[2])
    assert e1.lower_bound(np.array([0.99]), 1.0)[0] == pytest.approx(mu[0] - se[0])


def test_binned_edge_is_flat_without_skill() -> None:
    rng = np.random.default_rng(5)
    pct = np.tile(np.arange(1, 51) / 50, 300)
    row = np.repeat(np.arange(300), 50)
    realized = rng.normal(0, 0.01, len(pct))
    e = C.fit_binned_edge(pct, realized, row, n_bins=10)
    lcb = e.lower_bound(np.array([0.99]), 1.0)[0]
    assert lcb < 0.0005  # no skill: the lower bound never promises an edge worth trading


def test_conformal_bands_cover_iid_residuals() -> None:
    rng = np.random.default_rng(6)
    pct = rng.uniform(0, 1, 20000)
    res = rng.standard_t(4, size=20000) * 0.01
    bands = C.fit_conformal(pct[:10000], res[:10000], n_bins=5, alpha=0.2)
    lo, hi = bands.interval(pct[10000:], np.zeros(10000))
    cov = C.coverage(lo, hi, res[10000:])
    assert 0.77 < cov["coverage"] < 0.83
    tiny = C.fit_conformal(pct[:20], res[:20], n_bins=5, alpha=0.2, min_obs=30)
    assert np.isnan(tiny.lo).all()  # too few calibration residuals: no band rather than a false one


def test_round_trip_cost_is_higher_for_illiquid_names() -> None:
    f = {"adv20": np.array([[5e9, 2e7, np.nan]]), "close": np.array([[100.0, 20.0, 10.0]])}
    c = round_trip_cost_bps(f, EquityCosts(), 5000.0)
    assert c[0, 0] < c[0, 1] < c[0, 2]
    assert np.all(c > 0)


def test_cost_aware_selector_rules() -> None:
    T, N = 4, 5
    park = 4
    edge = np.array([[30.0, 20.0, 5.0, 1.0, 0.0]] * T)
    cost = np.full((T, N), 10.0)
    elig = np.ones((T, N), dtype=bool)
    reb = np.array([True, False, True, True])
    w, stats = select_cost_aware(edge, cost, elig, 2, reb, park_col=park)
    np.testing.assert_allclose(w[0], [0.5, 0.5, 0, 0, 0])  # both qualifying names; nothing to park
    assert np.isnan(w[1]).all()  # not a rebalance row: hold and drift
    # name 2's edge (5) never beats its round-trip cost (10): it is not bought
    edge2 = edge.copy()
    edge2[2:, 1] = 8.0  # held name 1 decays below its cost but stays positive: kept (no new entry cost)
    edge2[2:, 2] = 12.0  # a new name with edge 12 nets only 2 after costs: beats nothing held (name 1 nets 8)
    w2, _ = select_cost_aware(edge2, cost, elig, 2, reb, park_col=park)
    np.testing.assert_allclose(w2[2], [0.5, 0.5, 0, 0, 0])
    edge3 = edge.copy()
    edge3[2:, 1] = -1.0  # held name turns negative: dropped; its slot parks in the market ETF
    w3, _ = select_cost_aware(edge3, cost, elig, 2, reb, park_col=park)
    np.testing.assert_allclose(w3[2], [0.5, 0, 0, 0, 0.5])
    none, st = select_cost_aware(np.zeros((T, N)), cost, elig, 2, reb, park_col=park)
    np.testing.assert_allclose(none[0], [0, 0, 0, 0, 1.0])  # no edge beats costs: hold the market
    assert st["share_rebalances_with_no_name"] == 1.0
    flat, _ = select_cost_aware(edge, cost, elig, 2, reb, park_col=park, active=np.array([False, True, True, True]))
    np.testing.assert_allclose(flat[0], 0.0)  # inactive rows are flat, not parked
    assert stats["rebalances"] == 3
    with pytest.raises(ValueError):
        select_cost_aware(edge, cost, elig, 0, reb)
