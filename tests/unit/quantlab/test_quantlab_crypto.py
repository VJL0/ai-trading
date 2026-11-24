from __future__ import annotations

import numpy as np
import pandas as pd

from services.quantlab.backtest import BacktestConfig, run_weights
from services.quantlab.strategies import crypto as D


def _hourly(n_days: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2022-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rows = []
    for sym, drift in (("BTC/USD", 0.0002), ("ETH/USD", 0.0001), ("SOL/USD", -0.0001)):
        c = 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(len(t))))
        o = np.r_[c[0], c[:-1]]
        rows.append(pd.DataFrame({"symbol": sym, "t": t, "o": o, "h": np.maximum(o, c) * 1.001,
                                  "l": np.minimum(o, c) * 0.999, "c": c, "v": 10.0, "n": 1, "vw": c}))
    return pd.concat(rows, ignore_index=True)


def test_daily_resample_uses_utc_days_and_last_close() -> None:
    bars = _hourly(30)
    p = D.build_crypto_panel(bars, freq="1D")
    assert p.shape == (30, 3)
    btc = bars[bars["symbol"] == "BTC/USD"]
    day0 = btc[btc["t"] < pd.Timestamp("2022-01-02", tz="UTC")]
    j = p.col("BTC/USD")
    assert p.f["close"][0, j] == day0["c"].iloc[-1]
    assert p.f["open"][0, j] == day0["o"].iloc[0]
    assert p.f["high"][0, j] == day0["h"].max()


def test_tsmom_targets_do_not_look_ahead() -> None:
    p = D.build_crypto_panel(_hourly(400), freq="1D")
    q = D.build_crypto_panel(_hourly(400), freq="1D")
    k = 300
    for key in ("close", "adj_close", "open", "adj_open"):
        q.f[key][k:] *= 2.0
    prm = {"rule": "sma", "lookback": 50, "target_vol": 0.8, "rebalance": 1, "assets": "core3"}
    tp, tq = D.tsmom_targets(p, prm), D.tsmom_targets(q, prm)
    np.testing.assert_array_equal(np.nan_to_num(tp[:k]), np.nan_to_num(tq[:k]))
    assert np.nanmax(np.nansum(tp, axis=1)) <= 1.0 + 1e-9  # never levered


def test_crypto_flat_costs_charged() -> None:
    p = D.build_crypto_panel(_hourly(120), freq="1D")
    w = D.buy_and_hold(p, ("BTC/USD",))
    free = run_weights(p.f, p.dates, w, BacktestConfig(flat_cost_bps=(0.0, 0.0, 0.0), periods_per_year=365.0))
    paid = run_weights(p.f, p.dates, w, BacktestConfig(flat_cost_bps=(32.0, 32.0, 32.0), periods_per_year=365.0))
    assert paid.net.sum() < free.net.sum()
    assert paid.cost.sum() > 0


def test_relisting_gap_breaks_segment_and_forces_exit() -> None:
    bars = _hourly(200)
    btc = bars["symbol"] == "BTC/USD"
    gap = (bars["t"] >= pd.Timestamp("2022-03-01", tz="UTC")) & (bars["t"] < pd.Timestamp("2022-04-01", tz="UTC"))
    bars = bars[~(btc & gap)].copy()
    first = btc & (bars["t"] == pd.Timestamp("2022-04-01", tz="UTC"))
    bars.loc[first, "o"] = bars.loc[first, "c"] / 9.0  # stray relisting print
    p = D.build_crypto_panel(bars, freq="1D")
    j = p.col("BTC/USD")
    t = [str(d)[:10] for d in p.dates].index("2022-04-01")
    assert np.isnan(p.f["ret_oc"][t, j]) and np.isnan(p.f["ret_cc"][t, j])
    assert p.f["obs_count"][t, j] == 1
    w = D.buy_and_hold(p, ("BTC/USD",))
    res = run_weights(p.f, p.dates, w, BacktestConfig(flat_cost_bps=(0.0, 0.0, 0.0), periods_per_year=365.0,
                                                      delist_after=D.MAX_GAP_DAYS))
    assert res.forced_exits >= 1
    assert res.net.max() < 0.5  # the stray print never becomes a return
