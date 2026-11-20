"""build_market_snapshot: point-in-time only, labeled, fail closed on stale/unknown quotes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from typing import Any

import pytest

from packages.contracts.enums import DataFeed, FeedMode, MarketSession
from packages.contracts.models import Bar, Quote
from packages.core.clock import ManualClock
from services.market_data.calendar import get_calendar
from services.market_data.common import IEX_LIMITATION, SIMULATED_LIMITATION
from services.market_data.fixtures import ensure_synthetic_fixture, manifest_sessions
from services.market_data.replay import ReplayMarketDataProvider
from services.market_data.snapshot import build_market_snapshot
from services.market_data.store import MarketDataStore
from tests.unit.market_data.md_helpers import DAY, make_bar, make_quote, session_open

CAL = get_calendar()
OPEN = session_open(DAY)


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> tuple[MarketDataStore, dict[str, Any]]:
    d = tmp_path_factory.mktemp("snap")
    m = ensure_synthetic_fixture(d, symbols=("SPY",), sessions=1)
    return MarketDataStore(d), m


class StaticProvider:
    """Live-style provider stub returning a fixed quote/bars (IEX), or raising."""

    def __init__(self, quote: Quote | None = None, bars: Sequence[Bar] = (), exc: Exception | None = None,
                 feed_mode: FeedMode = FeedMode.IEX_FREE) -> None:
        self.quote, self.bars, self.exc = quote, list(bars), exc
        self.feed_mode = feed_mode
        self.source_id = "stub:iex"

    async def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> list[Bar]:
        if self.exc:
            raise self.exc
        return list(self.bars)

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        if self.exc:
            raise self.exc
        return self.quote

    async def stream_quotes(self, symbols: Sequence[str]) -> AsyncIterator[Quote]:
        if False:
            yield  # pragma: no cover


async def test_replay_snapshot_in_session_is_simulated_and_fresh(synth: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = synth
    day = manifest_sessions(m)[0]
    asof = CAL.session_open(day) + timedelta(minutes=30, seconds=7)
    p = ReplayMarketDataProvider(store, ManualClock(asof), DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    snap = await build_market_snapshot(p, "SPY", asof, FeedMode.SIMULATED, dataset_version=m["dataset_version"])
    assert snap.is_simulated and snap.feed == DataFeed.SYNTHETIC and snap.session == MarketSession.REGULAR
    assert snap.quote is not None and snap.quote.available_at_ts <= asof
    assert (asof - snap.quote.exchange_ts) <= timedelta(seconds=15)
    assert snap.last_bar is not None and snap.last_bar.available_at_ts <= asof
    assert snap.last_bar.end_ts == CAL.session_open(day) + timedelta(minutes=30)  # the 30th minute bar is complete
    assert snap.last_trade_price == snap.last_bar.close
    assert SIMULATED_LIMITATION in snap.limitations and snap.data_asof == asof
    assert snap.dataset_version == m["dataset_version"] and snap.source_id.startswith("replay:")


async def test_after_close_quote_withheld_fail_closed(synth: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = synth
    day = manifest_sessions(m)[0]
    asof = CAL.session_close(day) + timedelta(minutes=5)
    p = ReplayMarketDataProvider(store, ManualClock(asof), DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    snap = await build_market_snapshot(p, "SPY", asof, FeedMode.SIMULATED, dataset_version=m["dataset_version"])
    assert snap.quote is None
    assert any("No quote within 15" in x and "fail closed" in x for x in snap.limitations)
    assert snap.session == MarketSession.POST and any("regular-session only" in x for x in snap.limitations)


async def test_unknown_symbol_has_no_quote(synth: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = synth
    asof = CAL.session_open(manifest_sessions(m)[0]) + timedelta(hours=1)
    p = ReplayMarketDataProvider(store, ManualClock(asof), DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    snap = await build_market_snapshot(p, "QQQ", asof, FeedMode.SIMULATED, dataset_version=m["dataset_version"])
    assert snap.quote is None and snap.last_bar is None
    assert any("No point-in-time quote" in x for x in snap.limitations)


async def test_live_provider_fresh_quote_not_simulated() -> None:
    asof = OPEN + timedelta(minutes=10)
    q = make_quote("SPY", asof - timedelta(seconds=2))
    bar = make_bar("SPY", asof - timedelta(minutes=2))
    snap = await build_market_snapshot(StaticProvider(q, [bar]), "SPY", asof, FeedMode.IEX_FREE, dataset_version="live-iex")
    assert snap.quote == q and not snap.is_simulated and snap.feed == DataFeed.IEX
    assert IEX_LIMITATION in snap.limitations and snap.last_bar == bar


def _q(ex_back_s: float, received_back_s: float, available_back_s: float) -> Quote:
    asof = OPEN + timedelta(minutes=10)
    return make_quote("SPY", asof - timedelta(seconds=ex_back_s), received_ts=asof - timedelta(seconds=received_back_s),
                      available_at_ts=asof - timedelta(seconds=available_back_s))


@pytest.mark.parametrize(
    ("quote", "expect"),
    [
        (_q(20, 19.9, 19.9), "No quote within 15"),  # stale by exchange time
        (_q(1, 30, 1), "No quote within 15"),  # stale by receive time (local receipt long ago)
        (_q(1, -10, 0.5), "clock skew"),  # receive timestamp in the future relative to the decision time
        (_q(1, 1, -5), "look-ahead"),  # not yet available at asof
    ],
)
async def test_quote_freshness_and_pit_fail_closed(quote: Quote, expect: str) -> None:
    asof = OPEN + timedelta(minutes=10)
    snap = await build_market_snapshot(StaticProvider(quote), "SPY", asof, FeedMode.IEX_FREE, dataset_version="v")
    assert snap.quote is None, expect
    assert any(expect in x for x in snap.limitations), snap.limitations


async def test_provider_failure_is_fail_closed() -> None:
    asof = OPEN + timedelta(minutes=10)
    snap = await build_market_snapshot(StaticProvider(exc=TimeoutError("boom")), "SPY", asof, FeedMode.IEX_FREE,
                                       dataset_version="v")
    assert snap.quote is None and snap.last_bar is None
    assert any("Quote unavailable (TimeoutError)" in x for x in snap.limitations)
    assert any("Bars unavailable" in x for x in snap.limitations)


async def test_wrong_symbol_and_future_bars_withheld() -> None:
    asof = OPEN + timedelta(minutes=10)
    other = make_quote("QQQ", asof - timedelta(seconds=1))
    future_bar = make_bar("SPY", asof)  # ends after asof: not knowable yet
    snap = await build_market_snapshot(StaticProvider(other, [future_bar]), "SPY", asof, FeedMode.IEX_FREE,
                                       dataset_version="v")
    assert snap.quote is None and snap.last_bar is None and snap.last_trade_price is None
    assert any("another symbol" in x for x in snap.limitations)


async def test_naive_asof_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        await build_market_snapshot(StaticProvider(), "SPY", datetime(2026, 11, 3, 15, 0),
                                    FeedMode.IEX_FREE, dataset_version="v")



async def test_live_runtime_receives_quote_before_decision_cutoff() -> None:
    from types import SimpleNamespace

    from services.api.runtime import Runtime

    clock = ManualClock(OPEN + timedelta(minutes=10))
    observed: list[Quote] = []

    class DelayedProvider(StaticProvider):
        async def get_latest_quote(self, symbol: str) -> Quote:
            clock.set(clock.now() + timedelta(seconds=1))
            return make_quote(symbol, clock.now(), received_ts=clock.now(), available_at_ts=clock.now())

    rt = SimpleNamespace(provider=DelayedProvider(), sim_clock=None, now=clock.now,
                         feed_mode=FeedMode.IEX_FREE, dataset_version='test',
                         feed_monitor=SimpleNamespace(observe_quote=observed.append))
    snapshot = await Runtime.market_snapshot(rt, 'SPY')
    assert snapshot.quote is not None
    assert snapshot.quote.available_at_ts <= snapshot.data_asof
    assert observed == [snapshot.quote]


async def test_prefetched_quote_still_obeys_historical_cutoff() -> None:
    cutoff = OPEN + timedelta(minutes=10)
    future = make_quote('SPY', cutoff + timedelta(seconds=1))
    snapshot = await build_market_snapshot(StaticProvider(), 'SPY', cutoff, FeedMode.IEX_FREE,
                                          dataset_version='test', observed_quote=future)
    assert snapshot.quote is None
    assert any('look-ahead' in reason for reason in snapshot.limitations)
