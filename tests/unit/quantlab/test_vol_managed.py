"""Family J (volatility-managed index exposure) on SIMULATED data: never levered, point in time, band-limited."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from services.quantlab.panel import EquityPanel
from services.quantlab.strategies import defensive as J
from tests.unit.quantlab import synth


def _panel() -> EquityPanel:
    raw, adj = synth.bars({"SPY": {"drift": 0.0004, "vol": 0.012, "price": 300, "volume": 8e7},
                           "QQQ": {"drift": 0.0005, "vol": 0.016, "price": 200, "volume": 5e7}}, n=700)
    return EquityPanel.build(raw, adj, synth.assets(["SPY", "QQQ"], etfs=("SPY", "QQQ")), min_obs=50)


def test_exposure_is_capped_and_only_on_the_chosen_asset() -> None:
    panel = _panel()
    p = {"asset": "SPY", "target_vol": 0.20, "lookback": 21, "trend": "none", "band": 0.10}
    w = J.targets(panel, p)
    filled = np.nan_to_num(w)
    assert filled.max() <= 1.0 + 1e-12 and filled.min() >= 0.0
    assert np.all(filled[:, panel.col("QQQ")] == 0)
    held = np.where(~np.isnan(w).all(axis=1))[0]
    assert len(held) < len(w)  # band: not every day trades


def test_point_in_time_future_returns_do_not_change_past_targets() -> None:
    panel = _panel()
    p = {"asset": "SPY", "target_vol": 0.10, "lookback": 10, "trend": "sma200", "band": 0.25}
    base = J.targets(panel, p)
    later = replace(panel, f={k: v.copy() for k, v in panel.f.items()})
    later.f["ret_cc"][500:] *= 3.0
    later.f["adj_close"][500:] *= 0.5
    changed = J.targets(later, p)
    np.testing.assert_array_equal(np.nan_to_num(base[:500], nan=-1), np.nan_to_num(changed[:500], nan=-1))


def test_trend_filter_goes_flat_below_the_200_day_average() -> None:
    panel = _panel()
    j = panel.col("SPY")
    down = replace(panel, f={k: v.copy() for k, v in panel.f.items()})
    down.f["adj_close"][400:, j] = down.f["adj_close"][399, j] * np.linspace(1.0, 0.5, len(panel.dates) - 400)
    w = J.targets(down, {"asset": "SPY", "target_vol": 0.15, "lookback": 21, "trend": "sma200", "band": 0.10})
    last_decision = np.where(~np.isnan(w[:, j]))[0][-1]
    assert w[last_decision, j] == 0.0
    assert len(list(J.variants())) == 72
