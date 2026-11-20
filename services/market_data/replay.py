"""Point-in-time replay provider over stored data (implements ``packages.contracts.MarketDataProvider``).

STRICT point-in-time semantics: every method only returns records with ``available_at_ts <= clock.now()``
(or ``<= asof`` for the explicit ``*_asof`` variants, which additionally never exceed the clock). Nothing
published after the simulated decision time can leak into a decision.

Quotes are served from an in-memory columnar cache (one per symbol) that is rebuilt whenever the store's
generation token changes, so the per-tick ``get_latest_quote`` is a binary search instead of a DuckDB query.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from packages.contracts.enums import DataFeed, FeedMode
from packages.contracts.models import Bar, Quote
from packages.core.clock import Clock, ManualClock

from .common import to_utc
from .store import MarketDataStore, QuoteColumns
from .trades import Trade


def _right(arr: Any, x: int) -> int:
    return int(np.searchsorted(arr, x, side="right"))


def _left(arr: Any, x: int) -> int:
    return int(np.searchsorted(arr, x, side="left"))


def _us(ts: datetime) -> int:
    delta = to_utc(ts) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class ReplayMarketDataProvider:
    """Serves stored bars/quotes of one feed as if they were arriving live at ``clock.now()``."""

    def __init__(
        self,
        store: MarketDataStore,
        clock: Clock,
        feed: DataFeed,
        feed_mode: FeedMode,
        *,
        dataset_version: str | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.store = store
        self.clock = clock
        self.feed = DataFeed(feed)
        self.feed_mode = FeedMode(feed_mode)
        self.dataset_version = dataset_version
        self.source_id = f"replay:{self.feed.value}:{dataset_version or 'all'}"
        self.poll_interval_s = poll_interval_s
        self._cache: dict[str, QuoteColumns] = {}
        self._cache_gen: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cache
    def _columns(self, symbol: str) -> QuoteColumns:
        gen = self.store.generation()
        with self._lock:
            if gen != self._cache_gen:
                self._cache.clear()
                self._cache_gen = gen
            cols = self._cache.get(symbol)
        if cols is None:
            cols = self.store.quote_columns(symbol, self.feed, dataset_version=self.dataset_version)
            with self._lock:
                if self._cache_gen == gen:
                    self._cache[symbol] = cols
        return cols

    def _cap(self, asof: datetime | None) -> datetime:
        now = to_utc(self.clock.now())
        return now if asof is None else min(now, to_utc(asof))

    # ------------------------------------------------------------------ MarketDataProvider
    async def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> list[Bar]:
        return self.get_bars_asof(symbol, start, end, timeframe, None)

    def get_bars_asof(self, symbol: str, start: datetime, end: datetime, timeframe: str,
                      asof: datetime | None) -> list[Bar]:
        cap = self._cap(asof)
        return self.store.load_bars([symbol], start, end, timeframe, feed=self.feed, asof=cap,
                                    dataset_version=self.dataset_version)

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        return self.latest_quote_asof(symbol, None)

    def latest_quote_asof(self, symbol: str, asof: datetime | None) -> Quote | None:
        cap = self._cap(asof)
        cols = self._columns(symbol)
        if not len(cols):
            return None
        i = _right(cols.available_at_us, _us(cap)) - 1
        if i < 0:
            return None
        quote = cols.quote_at(i)
        if quote.available_at_ts > cap:  # defensive: never return a record from the future
            return None
        return quote

    async def get_latest_trade(self, symbol: str) -> Trade | None:
        return self.latest_trade_asof(symbol, None)

    def latest_trade_asof(self, symbol: str, asof: datetime | None) -> Trade | None:
        cap = self._cap(asof)
        trade = self.store.latest_trade(symbol, cap, feed=self.feed)
        return trade if trade is not None and trade.available_at_ts <= cap else None

    async def stream_quotes(self, symbols: Sequence[str], *, include_latest: bool = False,
                            stop_when_exhausted: bool = True) -> AsyncIterator[Quote]:
        """Yield quotes as they become available while the clock advances (in availability order).

        Starts after ``clock.now()`` (plus the current latest per symbol when ``include_latest``). Ends when every
        symbol's stored data is exhausted (``stop_when_exhausted``), otherwise polls forever.
        """
        start = self.clock.now()
        cursors: dict[str, int] = {}
        for s in symbols:
            cols = self._columns(s)
            k = _right(cols.available_at_us, _us(start))
            cursors[s] = max(k - 1, 0) if include_latest and k > 0 else k
        while True:
            cap = _us(self.clock.now())
            batch: list[tuple[int, str, int]] = []
            exhausted = True
            for s in symbols:
                cols = self._columns(s)
                k = cursors[s]
                hi = _right(cols.available_at_us, cap)
                for i in range(k, hi):
                    batch.append((int(cols.available_at_us[i]), s, i))
                cursors[s] = max(k, hi)
                if cursors[s] < len(cols):
                    exhausted = False
            batch.sort()
            for _avail, s, i in batch:
                yield self._columns(s).quote_at(i)
            if exhausted and stop_when_exhausted:
                return
            await asyncio.sleep(self.poll_interval_s)

    async def iter_history(self, symbols: Sequence[str], start: datetime, end: datetime) -> AsyncIterator[Quote]:
        """Historical replay: advance a ManualClock to each quote's ``available_at_ts`` and yield it."""
        if not isinstance(self.clock, ManualClock):
            raise TypeError("iter_history drives the clock and requires a ManualClock")
        events: list[tuple[int, str, int]] = []
        lo, hi = _us(start), _us(end)
        for s in symbols:
            cols = self._columns(s)
            a = _left(cols.available_at_us, lo)
            b = _right(cols.available_at_us, hi)
            events += [(int(cols.available_at_us[i]), s, i) for i in range(a, b)]
        events.sort()
        for avail, s, i in events:
            when = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=avail)
            if when > self.clock.now():
                self.clock.set(when)
            yield self._columns(s).quote_at(i)
            await asyncio.sleep(0)
