"""Phase 3: research and paper runtime produce the same predictions and trade decisions from the same inputs.

* ``decide_edge`` charges costs exactly once (GROSS: live cost; NET: re-price the training cost convention, then the
  live cost) and keeps the uncertainty buffer and threshold explicit;
* the runtime signal engine no longer double-charges NET forecasts (the autonomous paper model and the suite
  adapter both declare ``basis="NET"``);
* feature/preprocessing parity: for real decision times, the model applied to the training panel's feature row
  (research path) equals ``PredictionService.predict`` on point-in-time truncated bars (runtime path), and the
  engine's net edge equals the research net forecast when the order is sized at the training reference notional.

Documented differences that remain by design: live quotes vs the last quote published by the decision time, and the
live order notional vs ``reference_notional`` (only the impact term moves).
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.research.costs import CostModel
from services.research.decision import GROSS, NET, decide_edge
from services.research.frames import prepare_bars
from services.research.models.predict import PredictionService
from services.research.models.registry import ModelRegistry
from services.research.models.targets import build_suite_dataset
from services.research.signals import ModelPrediction, QuantSignalEngine

from .test_research_models_support import fast_config, planted_bars, train_and_save


# ------------------------------------------------------------------------------------------------ contract
def test_gross_forecast_is_charged_once() -> None:
    e = decide_edge(20.0, basis=GROSS, live_cost_bps=6.0, horizon_minutes=60, threshold_bps=5.0)
    assert e.gross_bps == 20.0 and e.net_bps == 14.0 and e.trade
    with pytest.raises(ValueError, match="deducted no cost"):
        decide_edge(20.0, basis=GROSS, live_cost_bps=6.0, assumed_cost_bps=6.0, horizon_minutes=60)


def test_net_forecast_is_repriced_not_double_charged() -> None:
    same = decide_edge(9.0, basis=NET, live_cost_bps=6.0, assumed_cost_bps=6.0, horizon_minutes=60, threshold_bps=5.0)
    assert same.net_bps == 9.0 and same.gross_bps == 15.0 and same.trade  # equal sizing: net == the model's net
    bigger = decide_edge(9.0, basis=NET, live_cost_bps=7.5, assumed_cost_bps=6.0, horizon_minutes=60, threshold_bps=5.0)
    assert bigger.net_bps == 7.5  # only the cost difference (impact of a larger order) moves the edge
    buffered = decide_edge(9.0, basis=NET, live_cost_bps=6.0, assumed_cost_bps=6.0, uncertainty_bps=4.5,
                           horizon_minutes=60, threshold_bps=5.0)
    assert buffered.net_lower_bps == 4.5 and not buffered.trade
    assert not decide_edge(float("nan"), basis=NET, live_cost_bps=6.0, horizon_minutes=60).trade
    with pytest.raises(ValueError):
        decide_edge(9.0, basis=NET, live_cost_bps=6.0, uncertainty_bps=-1.0, horizon_minutes=60)


class _FixedNetModel:
    strategy_id = "fixed-net-test"
    model_version = "fixed-net:1"
    horizon_bars = 60
    calibration_evaluated = False

    def __init__(self, net: float, *, uncertainty: float = 0.0, notional: float = 10_000.0) -> None:
        self.net, self.uncertainty, self.notional = net, uncertainty, notional

    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction:
        return ModelPrediction(self.net, self.net, None, False, basis="NET", reference_notional=self.notional,
                               uncertainty_bps=self.uncertainty)


def _asof(bars: pd.DataFrame, minute: int) -> pd.Timestamp:
    sb = bars.loc[bars["symbol"] == bars["symbol"].iloc[0]].reset_index(drop=True)
    day = sb["start_ts"].dt.tz_convert("America/New_York").dt.date
    return pd.Timestamp(sb.loc[day == day.iloc[-1]].reset_index(drop=True)["available_at_ts"].iloc[minute])


def test_engine_charges_a_net_forecast_exactly_once() -> None:
    bars = planted_bars(sessions=4)
    asof = _asof(bars, 120).to_pydatetime()
    engine = QuantSignalEngine(CostModel(), model=_FixedNetModel(9.0), order_notional=Decimal(10_000))
    sig = next(s for s in engine.generate(bars, asof, ["AAA"], "cz_parity"))
    assert float(sig.net_edge_bps) == pytest.approx(9.0, abs=1e-3)  # previously 9 - round trip (double charge)
    assert float(sig.expected_return_bps) == pytest.approx(9.0 + float(sig.est_total_cost_bps), abs=1e-3)
    buffered = QuantSignalEngine(CostModel(), model=_FixedNetModel(9.0, uncertainty=4.5), order_notional=Decimal(10_000))
    sig2 = next(s for s in buffered.generate(bars, asof, ["AAA"], "cz_parity"))
    assert float(sig2.net_edge_bps) == pytest.approx(4.5, abs=1e-3) and float(sig2.uncertainty_bps) == 4.5
    assert not sig2.is_candidate and "threshold" in sig2.reason


# ------------------------------------------------------------------------------------------------ research vs runtime
@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelRegistry, pd.DataFrame]:
    bars = planted_bars()
    _, reg = train_and_save(bars, Path(tmp_path_factory.mktemp("parity")) / "models")
    return reg, bars


def test_research_and_runtime_predictions_and_decisions_are_identical(trained: tuple[ModelRegistry, pd.DataFrame]) -> None:
    reg, bars = trained
    cfg = fast_config()
    ds = build_suite_dataset(prepare_bars(bars), CostModel(), cfg)
    cols = [ds.feature_columns.index(c) for c in reg.feature_columns]
    mid, model = reg.fitted.returns[60]
    svc = PredictionService(reg, CostModel())
    engine = QuantSignalEngine(CostModel(), model=svc, order_notional=cfg.reference_notional, horizon_within_session=False)
    syms = ds.panel["symbol"].astype(str).to_numpy()
    checked = 0
    for minute in (60, 120, 180, 240):
        asof = _asof(bars, minute)
        rows = np.flatnonzero(ds.decision_ns == asof.value)
        assert len(rows) == 3, "one research row per symbol at this decision time"
        preds = {p.symbol: p for p in svc.predict(bars, asof.to_pydatetime(), ["AAA", "BBB", "CCC"])}
        report = engine.generate_report(bars, asof.to_pydatetime(), ["AAA", "BBB", "CCC"], "cz_parity")
        signals = {s.symbol: s for s in report.signals}
        for i in rows:
            sym = str(syms[i])
            research_net = float(model.predict(ds.X[i:i + 1, cols])[0]) if mid != "zero_baseline" else 0.0
            runtime_net = float(next(f.expected_return_bps for f in preds[sym].return_forecasts if f.horizon_minutes == 60))
            assert runtime_net == pytest.approx(research_net, abs=1e-3), (sym, minute)  # same features, same model
            research = decide_edge(research_net, basis=NET, live_cost_bps=float(ds.cost_rt_bps[i]),
                                   assumed_cost_bps=float(ds.cost_rt_bps[i]), threshold_bps=float(engine.min_net_edge_bps),
                                   horizon_minutes=60)
            sig = signals[sym]
            if mid == "zero_baseline":
                assert not sig.is_candidate
                continue
            assert float(sig.net_edge_bps) == pytest.approx(research.net_bps, abs=0.05), (sym, minute)
            # the decision itself: eligible-by-edge on both paths (the engine may still pick one of several)
            edge_ok = float(sig.net_edge_bps) > float(engine.min_net_edge_bps)
            assert edge_ok == research.trade or math.isclose(research.net_bps, research.threshold_bps, abs_tol=0.05)
            checked += 1
    assert checked or mid == "zero_baseline"
