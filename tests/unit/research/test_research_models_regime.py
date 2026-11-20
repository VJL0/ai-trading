"""Regime model end to end: Markov selection on a switching-volatility market, filtered regime in predictions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.contracts.enums import MarketRegime
from services.research.costs import CostModel
from services.research.models.predict import PredictionService
from services.research.models.registry import ModelRegistry
from services.research.models.walkforward import TrainResult

from .test_research_models_support import regime_bars, strip_ids, train_and_save

SYMS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[TrainResult, ModelRegistry, pd.DataFrame, np.ndarray]:
    bars, state = regime_bars(sessions=10, seed=13)
    res, reg = train_and_save(bars, tmp_path_factory.mktemp("exp_regime") / "models")
    return res, reg, bars, state


def _last_session_asofs(bars: pd.DataFrame, step: int = 13) -> list[tuple[int, datetime]]:
    sb = bars.loc[bars["symbol"] == "AAA"].reset_index(drop=True)
    day = sb["start_ts"].dt.tz_convert("America/New_York").dt.date
    idx = np.flatnonzero((day == day.iloc[-1]).to_numpy())
    out: list[tuple[int, datetime]] = []
    for i in idx[30:-5:step]:
        ts = pd.Timestamp(sb["available_at_ts"].iloc[int(i)]).to_pydatetime()
        assert isinstance(ts, datetime)
        out.append((int(i), ts))
    return out


def test_markov_regime_is_selected_on_a_switching_market(
    trained: tuple[TrainResult, ModelRegistry, pd.DataFrame, np.ndarray],
) -> None:
    res, reg, _, _ = trained
    assert res.meta["selected"]["regime"] == "markov_2state", res.meta["targets"]["regime"]
    cands = {c["model_id"]: c for c in res.meta["targets"]["regime"]["candidates"]}
    assert cands["markov_2state"]["qlike"] < cands["unconditional_baseline"]["qlike"]
    assert res.meta["targets"]["regime"]["proxy_symbols"] == SYMS == res.meta["regime_proxy_symbols"]
    assert reg.fitted.regime is not None
    s2 = reg.fitted.regime.sigma2
    assert s2[reg.fitted.regime.turbulent_state] > 4 * s2[1 - reg.fitted.regime.turbulent_state]


def test_filtered_regime_in_predictions_tracks_the_hidden_state(
    trained: tuple[TrainResult, ModelRegistry, pd.DataFrame, np.ndarray],
) -> None:
    _, reg, bars, state = trained
    svc = PredictionService(reg, CostModel())
    hits = 0
    checked = 0
    seen: set[MarketRegime] = set()
    for i, asof in _last_session_asofs(bars):
        preds = svc.predict(bars, asof, SYMS)
        assert len(preds) == 3
        regimes = {p.regime for p in preds}
        assert len(regimes) == 1, "one market regime per decision time"
        p = preds[0]
        assert p.regime in (MarketRegime.CALM, MarketRegime.TURBULENT)
        assert isinstance(p.regime_probability, Decimal) and Decimal("0.5") <= p.regime_probability <= 1
        truth = MarketRegime.TURBULENT if state[i] == 1 else MarketRegime.CALM
        hits += p.regime == truth
        checked += 1
        seen |= {truth, p.regime}
    assert checked >= 20 and hits / checked >= 0.7, (hits, checked)
    assert seen == {MarketRegime.CALM, MarketRegime.TURBULENT}, "both regimes occur in the checked window"


def test_regime_needs_every_fresh_proxy_symbol(trained: tuple[TrainResult, ModelRegistry, pd.DataFrame, np.ndarray]) -> None:
    _, reg, bars, _ = trained
    svc = PredictionService(reg, CostModel())
    asof = _last_session_asofs(bars)[5][1]
    # a single-symbol request still uses the TRAINING proxy (all symbols present in the frame)
    one = svc.predict(bars, asof, ["AAA"])
    assert len(one) == 1 and one[0].regime != MarketRegime.UNKNOWN
    # the frame lacks a proxy symbol -> UNKNOWN (never a regime computed from a different basket)
    only_a = bars.loc[bars["symbol"] == "AAA"]
    p = svc.predict(only_a, asof, ["AAA"])[0]
    assert p.regime == MarketRegime.UNKNOWN and p.regime_probability is None
    assert any("proxy symbols without data ['BBB', 'CCC']" in n for n in p.notes)
    # a stale proxy symbol -> UNKNOWN for everyone (and no prediction for the stale symbol itself)
    cut = pd.Timestamp(asof) - pd.Timedelta(minutes=20)
    stale = bars.loc[~((bars["symbol"] == "CCC") & (bars["available_at_ts"] > cut))]
    preds = svc.predict(stale, asof, SYMS)
    assert {q.symbol for q in preds} == {"AAA", "BBB"}
    assert all(q.regime == MarketRegime.UNKNOWN and any("stale proxy" in n for n in q.notes) for q in preds)


def test_regime_predictions_ignore_data_after_asof(
    trained: tuple[TrainResult, ModelRegistry, pd.DataFrame, np.ndarray],
) -> None:
    _, reg, bars, _ = trained
    svc = PredictionService(reg, CostModel())
    asof = _last_session_asofs(bars)[8][1]
    base = [strip_ids(p.model_dump(mode="json")) for p in svc.predict(bars, asof, SYMS)]
    fut = (bars["available_at_ts"] > pd.Timestamp(asof)).to_numpy()
    altered = bars.copy()
    for col in ("open", "high", "low", "close"):
        altered.loc[fut, col] = [v * Decimal(3) for v in altered.loc[fut, col]]  # a wild future regime
    got = [strip_ids(p.model_dump(mode="json")) for p in svc.predict(altered, asof, SYMS)]
    assert got == base and base[0]["regime"] in ("CALM", "TURBULENT")


def test_regime_registry_roundtrip(trained: tuple[TrainResult, ModelRegistry, pd.DataFrame, np.ndarray],
                                   tmp_path: Path) -> None:
    from services.research.models.registry import load_registry_dir

    _, reg, _, _ = trained
    again = load_registry_dir(reg.path)
    assert again is not None and again.fitted.regime is not None and reg.fitted.regime is not None
    y = np.random.RandomState(0).normal(0, 10, 300)
    np.testing.assert_allclose(again.fitted.regime.p_turbulent(y), reg.fitted.regime.p_turbulent(y), atol=1e-12)
