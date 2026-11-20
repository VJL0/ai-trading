"""Model-suite components: filtered-vs-smoothed regime, calibration, HAR/EWMA/QLIKE, one-position strategy."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from packages.contracts.enums import MarketRegime
from services.research.models.estimators import FitError, choose_calibrator, fit_success_classifier
from services.research.models.features import VolSpec, ewma_variance
from services.research.models.regime import fit_regime
from services.research.models.strategy import StrategyStats, simulate_one_position
from services.research.models.vol import fit_har, qlike
from services.research.stats import brier_score


def _two_regime_series(seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.normal(0, 4, 300), rng.normal(0, 16, 150), rng.normal(0, 4, 300), rng.normal(0, 16, 100)])


# ---------------------------------------------------------------- regime


def test_regime_uses_filtered_probabilities_only_smoothed_would_leak() -> None:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = _two_regime_series()
    model = fit_regime(y, max_obs=5000)
    assert model.sigma2[model.turbulent_state] > model.sigma2[1 - model.turbulent_state]
    t = 500
    altered = y.copy()
    altered[t + 1:] = np.random.RandomState(99).normal(0, 40, len(y) - t - 1)  # rewrite the future
    p_full = model.p_turbulent(y)
    p_alt = model.p_turbulent(altered)
    p_trunc = model.p_turbulent(y[: t + 1])
    np.testing.assert_allclose(p_full[: t + 1], p_alt[: t + 1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_full[: t + 1], p_trunc, rtol=0, atol=1e-12)
    # the smoothed probabilities at the SAME t change when only future data changes: they look ahead (forbidden)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_full = np.asarray(MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
                             .smooth(model.params).smoothed_marginal_probabilities)[:, model.turbulent_state]
        sm_alt = np.asarray(MarkovRegression(altered, k_regimes=2, trend="c", switching_variance=True)
                            .smooth(model.params).smoothed_marginal_probabilities)[:, model.turbulent_state]
    assert np.max(np.abs(sm_full[: t + 1] - sm_alt[: t + 1])) > 1e-3
    # and the filtered series identifies the high-variance segments
    assert np.mean(p_full[320:450] > 0.5) > 0.8
    assert np.mean(p_full[20:290] > 0.5) < 0.2
    regime, prob = model.classify(float(p_full[400]))
    assert regime == MarketRegime.TURBULENT and prob > 0.5


def test_regime_fit_fails_closed_on_degenerate_series() -> None:
    with pytest.raises(FitError):
        fit_regime(np.zeros(500), max_obs=1000)
    with pytest.raises(FitError):
        fit_regime(np.ones(10), max_obs=1000)


# ---------------------------------------------------------------- calibration


def _logistic_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, n)
    p_true = 1 / (1 + np.exp(-x))
    y = (rng.uniform(size=n) < p_true).astype(np.int64)
    return x, p_true, y


def test_calibration_improves_brier_on_a_miscalibrated_problem() -> None:
    from services.research.ml import fit_calibrator

    x, _, y = _logistic_problem(6000, seed=1)
    raw = 1 / (1 + np.exp(-3.0 * x))  # overconfident scores (right ranking, wrong probabilities)
    cal_s, val_s, test_s = slice(0, 2000), slice(2000, 4000), slice(4000, 6000)
    cals = {}
    for m in ("isotonic", "sigmoid"):
        c, note = fit_calibrator(raw[cal_s], y[cal_s], m)
        assert c is not None, note
        cals[m] = c
    method, briers = choose_calibrator(raw[val_s], {m: c.transform(raw[val_s]) for m, c in cals.items()}, y[val_s])
    assert method in ("isotonic", "sigmoid")
    assert briers[method] < briers["raw"]
    b_raw = brier_score(raw[test_s], y[test_s])
    b_cal = brier_score(cals[method].transform(raw[test_s]), y[test_s])
    assert b_cal < b_raw - 0.01, (b_raw, b_cal)


def test_choose_calibrator_keeps_raw_when_calibration_does_not_help() -> None:
    _, p_true, y = _logistic_problem(3000, seed=2)
    worse = {"isotonic": np.clip(p_true + 0.2, 0, 1), "sigmoid": np.clip(p_true - 0.2, 0, 1)}
    method, briers = choose_calibrator(p_true, worse, y)
    assert method is None and briers["raw"] < min(briers["isotonic"], briers["sigmoid"])


def test_catboost_success_classifier_with_held_out_calibrators() -> None:
    rng = np.random.RandomState(4)
    n = 3000
    X = rng.normal(0, 1, (n, 4))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1])))).astype(np.int64)
    clf, notes = fit_success_classifier("catboost_clf", X[:1500], y[:1500], X[1500:2200], y[1500:2200],
                                        {"iterations": 300, "depth": 6, "learning_rate": 0.2}, seed=7)
    assert notes == {"isotonic": "ok", "sigmoid": "ok"}
    raw = clf.raw(X[2200:])
    assert raw.min() >= 0.0 and raw.max() <= 1.0
    for m in ("isotonic", "sigmoid"):
        p = clf.calibrated(X[2200:], m)
        assert np.isfinite(p).all() and p.min() > 0 and p.max() < 1
    assert brier_score(raw, y[2200:]) < 0.25, "classifier must beat a coin flip on a learnable problem"
    with pytest.raises(FitError):
        fit_success_classifier("c", X[:100], np.zeros(100, dtype=np.int64), X[:10], y[:10], {"iterations": 5}, 1)


# ---------------------------------------------------------------- volatility


def test_ewma_recursion_matches_definition() -> None:
    r = np.array([0.01, -0.02, 0.005, 0.0, 0.03])
    out = ewma_variance(r, 0.9, 2)
    s2 = (0.01 ** 2 + 0.02 ** 2) / 2
    assert np.isnan(out[0]) and out[1] == pytest.approx(s2)
    for i in range(2, 5):
        s2 = 0.9 * s2 + 0.1 * r[i] ** 2
        assert out[i] == pytest.approx(s2)


def test_har_ols_recovers_coefficients_and_qlike_prefers_truth() -> None:
    rng = np.random.RandomState(5)
    n = 4000
    d, w, m = (np.abs(rng.normal(1, 0.3, n)) * 1e-4 for _ in range(3))
    target = (0.2e-4 + 0.5 * d + 0.3 * w + 0.1 * m) * np.exp(rng.normal(0, 0.05, n))
    vs = VolSpec(horizon_bars=60, base_bars=60, lags=(1, 5, 22), ewma_lambda=0.94)
    har = fit_har(np.column_stack([d, w, m, d]), target, vs)
    np.testing.assert_allclose(har.coef[1:], [0.5, 0.3, 0.1], atol=0.03)
    truth = 0.2e-4 + 0.5 * d + 0.3 * w + 0.1 * m
    assert qlike(target, truth) < qlike(target, 2 * truth)
    assert qlike(target, truth) < qlike(target, 0.5 * truth)


# ---------------------------------------------------------------- one-position strategy


def test_one_position_strategy_rules() -> None:
    d = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    score = np.array([1.0, 2.0, 5.0, 0.0, 3.0, 1.0, -1.0, -2.0, 0.5, 0.7])
    exit_ = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64)
    taken = simulate_one_position(d, score, exit_, threshold=0.0)
    # t=0 takes the best (idx 1, busy until 2); t=1 blocked; t=2 takes idx 4; t=3 blocked; t=4 takes idx 9
    assert taken.tolist() == [1, 4, 9]
    assert simulate_one_position(d, np.zeros(len(d)), exit_, threshold=0.0).size == 0, "zero forecast never trades"
    with pytest.raises(ValueError, match="sorted"):
        simulate_one_position(d[::-1], score, exit_)


def test_strategy_stats_annualisation_and_pvalue() -> None:
    st = StrategyStats()
    trades = np.array([10.0, -5.0, 8.0, 3.0])
    st.add(trades, span_minutes=390.0 * 2, n_rows=100)
    m = st.metrics(60)
    years = 780.0 / (252 * 390)
    expect = trades.mean() / trades.std(ddof=1) * math.sqrt(4 / years)
    assert m["net_sharpe"] == pytest.approx(expect)
    assert m["net_return"] == pytest.approx(16.0 / 10_000)
    assert m["hit_rate"] == pytest.approx(0.75) and m["turnover"] == pytest.approx(2.0)
    assert 0 < m["p_value"] < 0.5
    empty = StrategyStats()
    empty.add(np.empty(0), 390.0, 10)
    assert empty.metrics(5)["net_sharpe"] == 0.0 and empty.metrics(5)["p_value"] == 1.0
