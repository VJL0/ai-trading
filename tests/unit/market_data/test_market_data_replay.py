"""ReplayMarketDataProvider: strict point-in-time semantics over stored data (no network)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from packages.contracts import MarketDataProvider
from packages.contracts.enums import DataFeed, FeedMode
from packages.core.clock import ManualClock
from services.market_data.calendar import get_calendar
from services.market_data.fixtures import ensure_synthetic_fixture, manifest_sessions
from services.market_data.replay import ReplayMarketDataProvider
from services.market_data.store import MarketDataStore

CAL = get_calendar()


@pytest.fixture(scope="module")
def replay_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[MarketDataStore, dict[str, Any]]:
    d = tmp_path_factory.mktemp("replay")
    manifest = ensure_synthetic_fixture(d, symbols=("SPY", "QQQ"), sessions=2)
    return MarketDataStore(d), manifest


def _bounds(manifest: dict[str, Any]) -> tuple[datetime, datetime]:
    days = manifest_sessions(manifest)
    return CAL.session_open(days[0]), CAL.session_close(days[-1])


def test_implements_protocol_and_labels(replay_data: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = replay_data
    p = ReplayMarketDataProvider(store, ManualClock(_bounds(m)[0]), DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    assert isinstance(p, MarketDataProvider)
    assert p.feed_mode == FeedMode.SIMULATED and p.source_id.startswith("replay:SYNTHETIC")


async def test_nothing_before_first_availability(replay_data: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = replay_data
    open_, _close = _bounds(m)
    clock = ManualClock(open_)
    p = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    assert await p.get_latest_quote("SPY") is None  # first quote at open is only available at open + 50ms
    assert await p.get_bars("SPY", open_ - timedelta(days=1), open_ + timedelta(days=1), "1Min") == []
    clock.advance(0.05)
    q = await p.get_latest_quote("SPY")
    assert q is not None and q.exchange_ts == open_ and q.available_at_ts == clock.now()


async def test_bars_and_quotes_only_up_to_clock(replay_data: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = replay_data
    open_, _close = _bounds(m)
    clock = ManualClock(open_ + timedelta(minutes=10, seconds=0, milliseconds=500))
    p = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    bars = await p.get_bars("SPY", open_, open_ + timedelta(hours=6), "1Min")
    # minute bar i is available at open + (i+1) min + 1 s latency: bars 0..8 are known at +10m00.5s, bar 9 is not
    assert len(bars) == 9 and all(b.available_at_ts <= clock.now() for b in bars)
    q = await p.get_latest_quote("SPY")
    assert q is not None and q.exchange_ts == open_ + timedelta(minutes=10) and q.available_at_ts <= clock.now()
    # an explicit asof can look further back but never past the clock
    past = p.get_bars_asof("SPY", open_, open_ + timedelta(hours=6), "1Min", open_ + timedelta(minutes=3, seconds=2))
    assert len(past) == 3
    future = p.get_bars_asof("SPY", open_, open_ + timedelta(hours=6), "1Min", open_ + timedelta(hours=5))
    assert len(future) == 9  # capped at clock.now()
    assert p.latest_quote_asof("SPY", open_ + timedelta(hours=5)) == q


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(offset_s=st.floats(min_value=-3600, max_value=2 * 86400, allow_nan=False),
       asof_back_s=st.floats(min_value=0, max_value=7200, allow_nan=False))
def test_property_no_record_after_asof(replay_data: tuple[MarketDataStore, dict[str, Any]], offset_s: float,
                                       asof_back_s: float) -> None:
    store, m = replay_data
    open_, close = _bounds(m)
    now = open_ + timedelta(seconds=offset_s)
    asof = now - timedelta(seconds=asof_back_s)
    clock = ManualClock(now)
    p = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    cap = min(now, asof)
    for tf in ("1Min", "1Day"):
        bars = p.get_bars_asof("SPY", open_ - timedelta(days=2), close + timedelta(days=2), tf, asof)
        assert all(b.available_at_ts <= cap for b in bars)
        assert all(b.available_at_ts <= now for b in asyncio.run(
            p.get_bars("QQQ", open_ - timedelta(days=2), close + timedelta(days=2), tf)))
    q = p.latest_quote_asof("SPY", asof)
    if q is not None:
        assert q.available_at_ts <= cap
    # completeness: the provider returns THE latest quote available at the cap (not just any older one)
    stored = store.load_quotes(["SPY"], None, None, feed=DataFeed.SYNTHETIC, asof=cap)
    expected = max(stored, key=lambda x: (x.available_at_ts, x.exchange_ts)) if stored else None
    assert q == expected


async def test_stream_quotes_follows_manual_clock(replay_data: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = replay_data
    open_, _close = _bounds(m)
    clock = ManualClock(open_ + timedelta(seconds=1))
    p = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED, poll_interval_s=0.001)
    got: list[tuple[Any, datetime]] = []

    async def consume() -> None:
        async for q in p.stream_quotes(["SPY", "QQQ"], stop_when_exhausted=False):
            got.append((q, clock.now()))
            if len(got) >= 8:
                return

    task = asyncio.create_task(consume())
    for _ in range(40):  # advance 1 simulated second per poll; quotes every 5 s per symbol
        await asyncio.sleep(0.003)
        clock.advance(1)
        if task.done():
            break
    await asyncio.wait_for(task, timeout=5)
    assert len(got) == 8
    for q, seen_at in got:
        assert q.available_at_ts <= seen_at  # never yielded before it was available
        assert q.available_at_ts > open_ + timedelta(seconds=1)  # starts strictly after the clock at start
    avail = [q.available_at_ts for q, _ in got]
    assert avail == sorted(avail) and {q.symbol for q, _ in got} == {"SPY", "QQQ"}


async def test_iter_history_drives_clock(replay_data: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = replay_data
    open_, _close = _bounds(m)
    clock = ManualClock(open_ - timedelta(hours=1))
    p = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    seen = [(q, clock.now()) async for q in p.iter_history(["SPY"], open_, open_ + timedelta(minutes=1))]
    assert len(seen) == 12  # 5-second quotes during the first minute
    assert all(q.available_at_ts == now for q, now in seen)

    class _Wall:
        def now(self) -> datetime:
            return open_

    wall = ReplayMarketDataProvider(store, _Wall(), DataFeed.SYNTHETIC, FeedMode.SIMULATED)

    async def drain() -> None:
        async for _ in wall.iter_history(["SPY"], open_, open_ + timedelta(minutes=1)):
            pass

    with pytest.raises(TypeError, match="ManualClock"):
        await drain()


async def test_cache_invalidated_by_new_write(tmp_path: Path) -> None:
    ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1, seed=1)
    store = MarketDataStore(tmp_path)
    m2 = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1, seed=2)
    open_, _ = _bounds(m2)
    clock = ManualClock(open_ + timedelta(minutes=5))
    p = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    q2 = await p.get_latest_quote("SPY")
    assert q2 is not None and q2.source_id.endswith("seed=2")  # the latest snapshot fixture is the active series
    ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1, seed=1)  # re-activates seed 1 (generation bump)
    q1 = await p.get_latest_quote("SPY")
    assert q1 is not None and q1.source_id.endswith("seed=1")
