"""QuantSignalEngine: point-in-time scoring, cost netting, at-most-one candidate, honest labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from packages.contracts.models import QuantSignal
from services.research.costs import CostModel
from services.research.frames import BarDataError
from services.research.signals import ModelPrediction, QuantSignalEngine

from .bar_factory import make_bars, to_dt, ts

UNIVERSE = ["AAA", "BBB", "CCC"]


def _bars(**kw: object) -> pd.DataFrame:
    params: dict[str, object] = {"timeframe": "1Min", "n": 3, "seed": 4, "drifts": {"AAA": 0.03, "BBB": -0.02}}
    params.update(kw)
    return make_bars(("AAA", "BBB", "CCC"), **params)  # type: ignore[arg-type]


def _asof(df: pd.DataFrame, i: int) -> datetime:
    return to_dt(sorted(df["available_at_ts"].unique())[i])


def _key(s: QuantSignal) -> tuple[object, ...]:
    return (s.symbol, s.rank, s.score, s.expected_return_bps, s.net_edge_bps, s.est_spread_bps, s.est_fees_bps,
            s.est_slippage_bps, s.is_candidate, s.features_asof)


def test_candidate_selected_only_above_threshold_and_at_most_one() -> None:
    df = _bars()
    asof = _asof(df, 700)
    sigs = QuantSignalEngine(CostModel(), min_net_edge_bps=0).generate(df, asof, UNIVERSE, "cz_1")
    assert len(sigs) == 3
    assert [s.rank for s in sigs] == [1, 2, 3]
    cands = [s for s in sigs if s.is_candidate]
    assert len(cands) == 1 and cands[0].rank == 1 and cands[0].net_edge_bps > 0
    for s in sigs:
        total = s.est_spread_bps + s.est_fees_bps + s.est_slippage_bps
        assert abs(s.expected_return_bps - total - s.net_edge_bps) <= Decimal("0.001")
        assert isinstance(s.net_edge_bps, Decimal)
        assert s.probability_up is None and s.probability_calibrated is False
    # same data, prohibitive threshold -> nobody is a candidate (caller emits NO_TRADE)
    none = QuantSignalEngine(CostModel(), min_net_edge_bps=100_000).generate(df, asof, UNIVERSE, "cz_2")
    assert none and not any(s.is_candidate for s in none)
    assert all("threshold" in s.reason for s in none)


def test_never_uses_data_after_asof() -> None:
    df = _bars()
    asof = _asof(df, 650)
    eng = QuantSignalEngine(CostModel(), min_net_edge_bps=0)
    base = [_key(s) for s in eng.generate(df, asof, UNIVERSE, "cz")]
    truncated = df.loc[df["available_at_ts"] <= pd.Timestamp(asof)]
    assert [_key(s) for s in eng.generate(truncated, asof, UNIVERSE, "cz")] == base
    # corrupt everything published after asof, including values that would fail validation
    future = df["available_at_ts"] > pd.Timestamp(asof)
    corrupt = df.copy()
    corrupt.loc[future, "close"] = Decimal("-5")
    corrupt.loc[future, "high"] = Decimal("0.01")
    corrupt.loc[future, "volume"] = Decimal("-1")
    assert [_key(s) for s in eng.generate(corrupt, asof, UNIVERSE, "cz")] == base
    for s in eng.generate(df, asof, UNIVERSE, "cz"):
        assert s.features_asof <= asof
    # a bar that ends exactly at asof but becomes available 1 us later is NOT visible
    late = df.copy()
    late["available_at_ts"] = late["available_at_ts"] + pd.Timedelta(microseconds=1)
    shifted = [_key(s) for s in eng.generate(late, asof, UNIVERSE, "cz")]
    assert all(to_dt(k[-1]) < asof for k in shifted)


def test_knowable_malformed_data_fails_closed() -> None:
    df = _bars()
    asof = _asof(df, 650)
    bad = df.copy()
    idx = bad.index[(bad["available_at_ts"] <= pd.Timestamp(asof)).to_numpy()][10]
    bad.loc[idx, "high"] = Decimal("0.01")
    with pytest.raises(BarDataError):
        QuantSignalEngine(CostModel()).generate(bad, asof, UNIVERSE, "cz")


def test_malformed_row_of_a_symbol_outside_the_universe_does_not_block() -> None:
    df = _bars()
    asof = _asof(df, 650)
    eng = QuantSignalEngine(CostModel(), min_net_edge_bps=0)
    base = [_key(s) for s in eng.generate(df, asof, ["AAA", "BBB"], "cz")]
    bad = df.copy()
    idx = bad.index[((bad["symbol"] == "CCC") & (bad["available_at_ts"] <= pd.Timestamp(asof))).to_numpy()][10]
    bad.loc[idx, "high"] = Decimal("0.01")
    assert [_key(s) for s in eng.generate(bad, asof, ["AAA", "BBB"], "cz")] == base
    with pytest.raises(BarDataError):  # ...but it still fails closed when CCC itself is being decided on
        eng.generate(bad, asof, UNIVERSE, "cz")


def test_stale_data_blocks_candidate() -> None:
    df = _bars()
    last = to_dt(df["available_at_ts"].max())
    sigs = QuantSignalEngine(CostModel(), min_net_edge_bps=0).generate(df, last + timedelta(hours=2), UNIVERSE, "cz")
    assert sigs and not any(s.is_candidate for s in sigs)
    assert all("stale" in s.reason for s in sigs)


def test_is_simulated_mirrors_dataset() -> None:
    real = _bars(is_synthetic=False, dataset_version="hist-v1")
    asof = _asof(real, 700)
    sigs = QuantSignalEngine(CostModel()).generate(real, asof, UNIVERSE, "cz")
    assert sigs and all(s.is_simulated is False and s.dataset_version == "hist-v1" for s in sigs)
    synth = _bars()
    sigs2 = QuantSignalEngine(CostModel()).generate(synth, asof, UNIVERSE, "cz")
    assert sigs2 and all(s.is_simulated is True for s in sigs2)


def test_quote_spread_used_when_supplied() -> None:
    df = _bars()
    asof = _asof(df, 700)
    sigs = QuantSignalEngine(CostModel()).generate(df, asof, ["AAA"], "cz", quote_spreads_bps={"AAA": Decimal("8")})
    assert len(sigs) == 1
    # round trip crosses half the 8 bps spread twice -> 8 bps (relative to entry notional, exit at same price)
    assert sigs[0].est_spread_bps == Decimal("8")
    assert "QUOTE" in sigs[0].reason


def test_intraday_horizon_past_the_session_close_blocks_candidate() -> None:
    df = _bars()
    model = _PerSymbolModel({"AAA": 400.0, "BBB": 200.0, "CCC": 100.0})
    eng = QuantSignalEngine(CostModel(), model=model, min_net_edge_bps=0)
    # session 2 of the fixture: index 390 + 300 -> 14:30 ET (horizon ends 15:30 <= 16:00 close)
    early = eng.generate(df, _asof(df, 690), UNIVERSE, "cz")
    assert sum(s.is_candidate for s in early) == 1
    # 15:20 ET: a 60-minute horizon would end after the 16:00 ET close -> nobody is a candidate
    late_asof = _asof(df, 740)
    late = eng.generate(df, late_asof, UNIVERSE, "cz")
    assert late and not any(s.is_candidate for s in late)
    assert all("after the scheduled session close" in s.reason for s in late)
    # the guard is the only reason: disabling it (research use) restores a candidate at the same asof
    off = QuantSignalEngine(CostModel(), model=model, min_net_edge_bps=0, horizon_within_session=False)
    assert sum(s.is_candidate for s in off.generate(df, late_asof, UNIVERSE, "cz")) == 1
    # a shorter horizon that fits before the close is eligible again (model horizon must match: 30 bars)
    model30 = _PerSymbolModel({"AAA": 400.0, "BBB": 200.0, "CCC": 100.0})
    model30.horizon_bars = 30
    short = QuantSignalEngine(CostModel(), model=model30, min_net_edge_bps=0, horizon_minutes=30)
    assert sum(s.is_candidate for s in short.generate(df, late_asof, UNIVERSE, "cz")) == 1


def test_illiquid_symbol_never_candidate() -> None:
    df = _bars(base_volume=1_000)
    asof = _asof(df, 700)
    sigs = QuantSignalEngine(CostModel(), min_net_edge_bps=0).generate(df, asof, UNIVERSE, "cz")
    assert sigs and not any(s.is_candidate for s in sigs)
    assert any("liquidity floor" in s.reason for s in sigs)


def test_unknown_liquidity_blocks_candidate() -> None:
    # a single intraday session has no completed prior session, so ADV is unknown -> never a candidate
    df = make_bars(("AAA",), timeframe="1Min", n=1, seed=4, drifts={"AAA": 0.2})
    asof = _asof(df, 300)
    sigs = QuantSignalEngine(CostModel(), min_net_edge_bps=0).generate(df, asof, ["AAA"], "cz")
    assert len(sigs) == 1 and not sigs[0].is_candidate
    assert "liquidity unknown" in sigs[0].reason


def test_empty_and_unknown_inputs_produce_no_signals() -> None:
    eng = QuantSignalEngine(CostModel())
    asof = datetime(2024, 3, 4, 15, 0, tzinfo=UTC)
    assert eng.generate(pd.DataFrame(), asof, UNIVERSE, "cz") == []
    df = _bars()
    rep = eng.generate_report(df, _asof(df, 700), ["ZZZ", "bad symbol"], "cz")
    assert rep.signals == [] and rep.outcome == "NO_TRADE"
    assert set(rep.skipped) == {"ZZZ", "bad symbol"}
    assert rep.no_trade_reason and "no eligible symbols" in rep.no_trade_reason
    early = eng.generate_report(df, _asof(df, 5), UNIVERSE, "cz")
    assert early.signals == [] and all("insufficient history" in v for v in early.skipped.values())


def test_naive_asof_rejected() -> None:
    df = _bars()
    with pytest.raises(BarDataError, match="naive"):
        QuantSignalEngine(CostModel()).generate(df, datetime(2024, 3, 4, 15, 0), UNIVERSE, "cz")


class _FakeModel:
    def __init__(self, pred: ModelPrediction | None, *, calibration_evaluated: bool = False) -> None:
        self.pred = pred
        self.model_version = "fake-model-v1"
        self.strategy_id = "fake_strategy"
        self.horizon_bars = 60
        self.calibration_evaluated = calibration_evaluated
        self.seen_max_ts: pd.Timestamp | None = None

    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction | None:
        latest = ts(sym_bars["available_at_ts"].max())
        self.seen_max_ts = latest if self.seen_max_ts is None else max(self.seen_max_ts, latest)
        return self.pred


def test_model_probability_calibration_flag_is_passed_through_honestly() -> None:
    df = _bars()
    asof = _asof(df, 700)
    unc = _FakeModel(ModelPrediction(expected_return_bps=80.0, score=1.0, probability_up=0.7, probability_calibrated=False))
    sigs = QuantSignalEngine(CostModel(), model=unc).generate(df, asof, UNIVERSE, "cz")
    assert all(s.probability_up == Decimal("0.700000") and s.probability_calibrated is False for s in sigs)
    assert all(s.strategy_id == "fake_strategy" and s.model_version == "fake-model-v1" for s in sigs)
    assert unc.seen_max_ts is not None and unc.seen_max_ts <= pd.Timestamp(asof)
    claim = ModelPrediction(expected_return_bps=80.0, score=1.0, probability_up=0.7, probability_calibrated=True)
    # a prediction that merely CLAIMS calibration, from a model without an evaluated calibrator, is not trusted
    boast = QuantSignalEngine(CostModel(), model=_FakeModel(claim)).generate(df, asof, UNIVERSE, "cz")
    assert all(s.probability_calibrated is False for s in boast)
    assert all("labelled UNCALIBRATED" in s.reason for s in boast)
    cal = _FakeModel(claim, calibration_evaluated=True)
    sigs2 = QuantSignalEngine(CostModel(), model=cal).generate(df, asof, UNIVERSE, "cz")
    assert all(s.probability_calibrated is True for s in sigs2)
    assert sum(s.is_candidate for s in sigs2) == 1  # ties broken deterministically by symbol


class _PerSymbolModel(_FakeModel):
    def __init__(self, expected: dict[str, float]) -> None:
        super().__init__(None)
        self.expected = expected

    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction | None:
        sym = str(sym_bars["symbol"].iloc[0])
        return ModelPrediction(expected_return_bps=self.expected[sym], score=self.expected[sym], probability_up=None,
                               probability_calibrated=False)


def test_stale_top_ranked_symbol_neither_blocks_nor_becomes_the_candidate() -> None:
    df = _bars()
    asof = _asof(df, 700)
    model = _PerSymbolModel({"AAA": 400.0, "BBB": 200.0, "CCC": -50.0})
    eng = QuantSignalEngine(CostModel(), model=model, min_net_edge_bps=0)
    fresh = eng.generate(df, asof, UNIVERSE, "cz")
    assert [s.symbol for s in fresh if s.is_candidate] == ["AAA"]
    # AAA's feed stops 30 minutes before asof: it still ranks first on its (stale) edge, but is blocked; the
    # fresh, liquid, positive-edge BBB at rank 2 becomes the single candidate
    stale_rows = (df["symbol"] == "AAA") & (df["available_at_ts"] > pd.Timestamp(asof) - pd.Timedelta(minutes=30))
    sigs = eng.generate(df.loc[~stale_rows], asof, UNIVERSE, "cz")
    by = {s.symbol: s for s in sigs}
    assert by["AAA"].rank == 1 and not by["AAA"].is_candidate and "stale" in by["AAA"].reason
    assert by["BBB"].rank == 2 and by["BBB"].is_candidate
    assert not by["CCC"].is_candidate and "not positive" in by["CCC"].reason
    assert sum(s.is_candidate for s in sigs) == 1
    # with every symbol eligible only the best one is selected; the runner-up says why it was not
    runner_up = next(s for s in fresh if s.symbol == "BBB")
    assert not runner_up.is_candidate and "higher-ranked eligible AAA selected" in runner_up.reason


def test_model_without_prediction_is_not_eligible() -> None:
    df = _bars()
    sigs = QuantSignalEngine(CostModel(), model=_FakeModel(None), min_net_edge_bps=0).generate(
        df, _asof(df, 700), UNIVERSE, "cz"
    )
    assert sigs and not any(s.is_candidate for s in sigs)
    assert all("no expected return" in s.reason for s in sigs)


@pytest.mark.parametrize(
    "pred",
    [
        ModelPrediction(expected_return_bps=float("nan"), score=1.0, probability_up=None, probability_calibrated=False),
        ModelPrediction(expected_return_bps=90.0, score=float("inf"), probability_up=None, probability_calibrated=False),
        ModelPrediction(expected_return_bps=90.0, score=1.0, probability_up=1.5, probability_calibrated=True),
    ],
)
def test_invalid_model_output_is_not_eligible_and_never_raises(pred: ModelPrediction) -> None:
    df = _bars()
    sigs = QuantSignalEngine(CostModel(), model=_FakeModel(pred), min_net_edge_bps=0).generate(
        df, _asof(df, 700), UNIVERSE, "cz"
    )
    assert len(sigs) == 3 and not any(s.is_candidate for s in sigs)
    assert all("model output invalid" in s.reason for s in sigs)
    assert all(s.probability_up is None and s.probability_calibrated is False for s in sigs)


def test_deterministic_given_same_inputs() -> None:
    df = _bars()
    asof = _asof(df, 720)
    a = QuantSignalEngine(CostModel()).generate(df, asof, UNIVERSE, "cz")
    b = QuantSignalEngine(CostModel()).generate(df.sample(frac=1.0, random_state=1), asof, UNIVERSE, "cz")
    assert [_key(s) for s in a] == [_key(s) for s in b]
    assert len({s.signal_id for s in a + b}) == 6
    assert np.all([s.generated_at == asof for s in a])


def test_engine_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError):
        QuantSignalEngine(CostModel(), min_net_edge_bps=-1)
    with pytest.raises(ValueError):
        QuantSignalEngine(CostModel(), horizon_minutes=0)
    with pytest.raises(TypeError):
        QuantSignalEngine(CostModel(), min_net_edge_bps=1.5)  # type: ignore[arg-type]
