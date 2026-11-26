"""Bar snapshot contract, point-in-time daily facts and the synthesized quote feed (offline, synthetic data)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from packages.contracts.enums import DataFeed
from services.replay.feed import ReplayFeed
from services.replay.snapshot import BAR, BarEvent, BarSnapshot, SnapshotError, SyntheticSpec, synthetic_snapshot

OPEN = datetime(2024, 3, 5, 14, 30, tzinfo=UTC)  # 2024-03-05 09:30 New York (EST)


def _frame(rows: list[tuple[datetime, float, float, float, float]], symbol: str = "SPY") -> pd.DataFrame:
    return pd.DataFrame([{"t": t, "symbol": symbol, "o": o, "h": h, "l": lo, "c": c, "v": 1000.0}
                         for t, o, h, lo, c in rows])


def test_from_frame_validates_contract() -> None:
    good = _frame([(OPEN, 100, 101, 99, 100.5)])
    snap = BarSnapshot.from_frame(good, source="t", feed=DataFeed.SIP)
    assert snap.symbols == ("SPY",) and snap.sessions == [date(2024, 3, 5)]
    assert snap.provenance.content_sha256 == BarSnapshot.from_frame(good, source="t", feed=DataFeed.SIP).provenance.content_sha256
    with pytest.raises(SnapshotError, match="OHLC"):
        BarSnapshot.from_frame(_frame([(OPEN, 100, 99, 98, 100.5)]), source="t", feed=DataFeed.SIP)
    with pytest.raises(SnapshotError, match="grid"):
        BarSnapshot.from_frame(_frame([(OPEN + timedelta(minutes=2), 100, 101, 99, 100)]), source="t", feed=DataFeed.SIP)
    with pytest.raises(SnapshotError, match="regular session"):
        BarSnapshot.from_frame(_frame([(OPEN - timedelta(minutes=30), 100, 101, 99, 100)]), source="t", feed=DataFeed.SIP)
    with pytest.raises(SnapshotError, match="duplicate"):
        BarSnapshot.from_frame(_frame([(OPEN, 100, 101, 99, 100)] * 2), source="t", feed=DataFeed.SIP)
    bad = good.copy()
    bad.loc[0, "c"] = float("nan")
    with pytest.raises(SnapshotError, match="null"):
        BarSnapshot.from_frame(bad, source="t", feed=DataFeed.SIP)


def test_real_data_inside_holdout_is_refused_synthetic_is_not() -> None:
    inside = datetime(2026, 3, 3, 14, 30, tzinfo=UTC)  # inside HOLDOUT_START..HOLDOUT_END
    real = BarSnapshot.from_frame(_frame([(inside, 100, 101, 99, 100)]), source="t", feed=DataFeed.SIP)
    with pytest.raises(SnapshotError, match="holdout"):
        real.assert_outside_holdout()
    synth = BarSnapshot.from_frame(_frame([(inside, 100, 101, 99, 100)]), source="t", feed=DataFeed.SYNTHETIC)
    synth.assert_outside_holdout()
    BarSnapshot.from_frame(_frame([(OPEN, 100, 101, 99, 100)]), source="t", feed=DataFeed.SIP).assert_outside_holdout()


def test_point_in_time_daily_facts_use_prior_sessions_only(tmp_path: Path) -> None:
    snap = synthetic_snapshot(SyntheticSpec(sessions=3))
    d0, d1, d2 = snap.sessions
    assert snap.adv_shares("SPY", d0) is None and snap.prev_close("SPY", d0) is None
    mask = (snap.bars["day"] == d0).to_numpy()
    vols = snap.bars["v"].to_numpy(dtype=float)[mask]
    closes = snap.bars["c"].to_numpy(dtype=float)[mask]
    assert snap.adv_shares("SPY", d1) == pytest.approx(float(vols.sum()))
    assert snap.prev_close("SPY", d1) == pytest.approx(float(closes[-1]))
    assert snap.window(None, None) == [d1, d2]
    path = tmp_path / "bars.csv"
    snap.save(path)
    again = BarSnapshot.load(path)
    assert again.provenance.feed == DataFeed.SYNTHETIC and again.provenance.content_sha256 == snap.provenance.content_sha256


def test_synthetic_bars_are_contiguous_and_events_apply() -> None:
    snap = synthetic_snapshot(SyntheticSpec(sessions=2, events=(BarEvent(1, 10, "SPY", gap_pct=-0.03),)))
    mask = (snap.bars["day"] == snap.sessions[1]).to_numpy()
    o = snap.bars["o"].to_numpy(dtype=float)[mask]
    c = snap.bars["c"].to_numpy(dtype=float)[mask]
    assert (o[1:9] == c[0:8]).all()  # next open == previous close
    assert o[10] == pytest.approx(c[9] * 0.97, rel=1e-5)


def test_feed_point_in_time_bars_and_quotes() -> None:
    snap = synthetic_snapshot(SyntheticSpec(sessions=2))
    feed = ReplayFeed(snap)
    open1 = datetime(2024, 3, 5, 14, 30, tzinfo=UTC)
    at = open1 + 2 * BAR + timedelta(seconds=30)  # inside the third bar of session 2
    df = feed.load_bars_df(["SPY"], open1 - timedelta(days=5), at, "1Min", asof=at)
    avail = pd.DatetimeIndex(df["available_at_ts"])
    ends = pd.DatetimeIndex(df["end_ts"])
    assert (avail <= pd.Timestamp(at)).all()
    assert ends.max() == pd.Timestamp(open1 + 2 * BAR)  # the running bar is not visible
    bars = feed.get_bars_asof("SPY", at - timedelta(minutes=30), at, "1Min", at)
    assert bars and max(b.end_ts for b in bars) == open1 + 2 * BAR
    assert feed.latest_quote_asof("SPY", at) is None  # nothing emitted yet: never invented
    row = feed.bar("SPY", open1)
    assert row is not None
    q = feed.emit(feed.make_quote("SPY", open1 + timedelta(seconds=1), row.o))
    assert q.feed == DataFeed.SYNTHETIC and q.is_synthetic and q.bid < q.ask
    assert feed.spread_bps("SPY", date(2024, 3, 5)) == Decimal(2)  # MEGA bucket from the prior session's ADV
    assert feed.latest_quote_asof("SPY", open1) is None and feed.latest_quote_asof("SPY", at) == q
    with pytest.raises(ValueError, match="time order"):
        feed.emit(feed.make_quote("SPY", open1, row.o))


def test_intrabar_path_and_stop_crossings() -> None:
    snap = synthetic_snapshot(SyntheticSpec(sessions=1, events=(
        BarEvent(0, 3, "SPY", ohlc=(500.0, 501.0, 497.0, 500.5)),)))
    feed = ReplayFeed(snap)
    row = feed.bar("SPY", datetime(2024, 3, 4, 14, 30, tzinfo=UTC) + 3 * BAR)
    assert row is not None
    path = row.path(5)
    assert path[0] == (1.0, 500.0) and path[-1] == (299.0, 500.5)
    prices = [p for _, p in path]
    assert prices.index(min(prices)) < prices.index(max(prices))  # up bar: low first
    t0, t1 = row.start + timedelta(seconds=96), row.start + timedelta(seconds=101)
    qs = list(feed.crossing_quotes("SPY", t0, 501.0, t1, 498.0, [Decimal("498.25"), Decimal("496.00")]))
    assert [q.bid for q in qs] == [Decimal("498.25")] and t0 < qs[0].exchange_ts < t1
    assert not list(feed.crossing_quotes("SPY", t0, 497.0, t1, 499.0, [Decimal("498.25")]))  # never on the way up
    # A coarse grid still visits the extrema and every minute's manager tick.
    coarse = dict(row.path(7))
    assert coarse[100] == row.l and coarse[200] == row.h
    assert {1, 61, 121, 181, 241} <= coarse.keys()
    # The bid crosses even when both sampled mids are above the stop.
    stop = feed.make_quote("SPY", t1, 499.0).bid
    quotes = list(feed.crossing_quotes("SPY", t0, 501.0, t1, 499.0, [stop]))
    assert stop < Decimal("499.0") and [q.bid for q in quotes] == [stop]


def test_negative_bar_latency_cannot_make_an_unfinished_bar_visible() -> None:
    with pytest.raises(ValueError, match="latency"):
        ReplayFeed(synthetic_snapshot(SyntheticSpec(sessions=1)), bar_latency_s=-1)


def test_snapshot_refuses_fractional_bar_start() -> None:
    snap = synthetic_snapshot(SyntheticSpec(sessions=1))
    frame = snap.research_frame()
    frame["t"] += pd.Timedelta(1, unit="ns")
    with pytest.raises(SnapshotError, match="grid"):
        BarSnapshot.from_frame(frame, source="test", feed=DataFeed.SYNTHETIC)


def test_snapshot_hash_and_adv_distinguish_input_changes() -> None:
    snap = synthetic_snapshot(SyntheticSpec(sessions=3))
    detached = snap.bars
    detached.loc[0, "v"] = -1
    assert snap.bars.loc[0, "v"] != -1
    frame = snap.research_frame()
    frame.loc[0, "v"] += 0.0000001
    revised = BarSnapshot.from_frame(frame, source="test", feed=DataFeed.SYNTHETIC)
    assert revised.provenance.content_sha256 != snap.provenance.content_sha256
    day = snap.sessions[2]
    last = frame.loc[frame["t"].dt.date == snap.sessions[1], "v"].sum()
    two_days = frame.loc[frame["t"].dt.date < day, "v"].sum() / 2
    assert revised.adv_shares("SPY", day, n=1) == pytest.approx(last)
    assert revised.adv_shares("SPY", day, n=2) == pytest.approx(two_days)
    frame.loc[0, "v"] = float("inf")
    with pytest.raises(SnapshotError, match="non-finite"):
        BarSnapshot.from_frame(frame, source="test", feed=DataFeed.SYNTHETIC)
