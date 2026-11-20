"""Data-quality segmentation: multi-session holes and impossible one-bar jumps cut research series."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from services.research.baselines import BuyAndHold, run_all_baselines
from services.research.costs import CostModel
from services.research.frames import (
    BarDataError,
    SegmentPolicy,
    last_clean_segment,
    missing_sessions_between,
    prepare_bars,
)
from services.research.signals import QuantSignalEngine

from .bar_factory import make_bars, to_dt


def test_missing_sessions_uses_the_exchange_calendar() -> None:
    assert missing_sessions_between(date(2024, 3, 1), date(2024, 3, 4)) == 0  # Fri -> Mon
    assert missing_sessions_between(date(2024, 3, 28), date(2024, 4, 1)) == 0  # Good Friday is not a session
    assert missing_sessions_between(date(2024, 3, 4), date(2024, 3, 8)) == 3
    assert missing_sessions_between(date(2024, 3, 8), date(2024, 3, 8)) == 0


def test_hole_cuts_series_and_only_the_last_segment_is_kept() -> None:
    df = make_bars(("AAA", "BBB"), timeframe="1Day", n=80, seed=2)
    hole = (df["symbol"] == "AAA") & df.index.isin(df.index[(df["symbol"] == "AAA").to_numpy()][30:40])
    bars = prepare_bars(df.loc[~hole])
    kept, rep = last_clean_segment(bars)
    a = kept.loc[kept["symbol"] == "AAA"]
    assert len(a) == 40 and rep["symbols"]["AAA"]["dropped_before_cut"] == 30
    assert "10 missing sessions" in rep["symbols"]["AAA"]["last_cut"]
    assert rep["symbols"]["BBB"] == {"rows_in": 80, "rows_kept": 80, "dropped_before_cut": 0}
    assert rep["rows_dropped"] == 30
    # a short hole (<= max_missing_sessions) is tolerated
    small = (df["symbol"] == "AAA") & df.index.isin(df.index[(df["symbol"] == "AAA").to_numpy()][30:33])
    _kept2, rep2 = last_clean_segment(prepare_bars(df.loc[~small]))
    assert rep2["symbols"]["AAA"]["rows_kept"] == 77 and "last_cut" not in rep2["symbols"]["AAA"]


def test_impossible_jump_cuts_series_like_a_ticker_change() -> None:
    df = make_bars(("AAA",), timeframe="1Day", n=60, seed=5)
    early = df.index < 20
    for col in ("open", "high", "low", "close"):
        df.loc[early, col] = [v / 10 for v in df.loc[early, col]]  # a different, 10x cheaper security before
    kept, rep = last_clean_segment(prepare_bars(df))
    assert len(kept) == 40 and "one-bar |ln r|" in rep["symbols"]["AAA"]["last_cut"]


def test_intraday_overnight_is_not_a_hole_and_too_short_segments_are_excluded() -> None:
    df = make_bars(("AAA", "BBB"), timeframe="5Min", n=3, seed=1)
    kept, rep = last_clean_segment(prepare_bars(df))
    assert len(kept) == len(df) and rep["rows_dropped"] == 0
    with pytest.raises(BarDataError, match="clean segment"):
        last_clean_segment(prepare_bars(df), SegmentPolicy(min_bars=10_000))


def test_baselines_never_trade_across_a_hole(tmp_path: Path) -> None:
    df = make_bars(("AAA",), timeframe="1Day", n=80, seed=2)
    holed = df.drop(index=df.index[30:45])
    out = run_all_baselines(holed, CostModel(), tmp_path, strategies=[BuyAndHold()], run_id="bl_hole")
    run = out["runs"][0]
    assert run["diagnostics"]["bars"] == 35
    assert out["data_quality"]["symbols"]["AAA"]["dropped_before_cut"] == 30
    first_trade = pd.Timestamp(run["trades"][0]["ts"])
    assert first_trade > pd.Timestamp(holed["start_ts"].iloc[30])
    off = run_all_baselines(holed, CostModel(), tmp_path, strategies=[BuyAndHold()], run_id="bl_hole_off",
                            segment_policy=None)
    assert off["runs"][0]["diagnostics"]["bars"] == 65 and off["data_quality"]["applied"].startswith("none")
    assert Decimal(off["runs"][0]["metrics"]["total_return"]) != Decimal(run["metrics"]["total_return"])


def test_signal_engine_blocks_a_hole_inside_the_lookback() -> None:
    df = make_bars(("AAA",), timeframe="1Day", n=60, seed=3)
    asof = to_dt(df["available_at_ts"].max())
    eng = QuantSignalEngine(CostModel(), horizon_minutes=5 * 390, min_net_edge_bps=0)
    assert "data gap" not in eng.generate(df, asof, ["AAA"], "cz")[0].reason
    holed = df.drop(index=df.index[-10:-7])  # three sessions missing inside the 20-bar lookback
    sig = eng.generate(holed, asof, ["AAA"], "cz")[0]
    assert not sig.is_candidate and "data gap: 3 XNYS sessions missing" in sig.reason
    with pytest.raises(ValueError):
        QuantSignalEngine(CostModel(), max_missing_sessions=-1)
