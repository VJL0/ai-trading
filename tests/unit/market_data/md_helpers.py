"""Shared builders for market-data tests (handcrafted records; no network, no credentials)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from packages.contracts.enums import Adjustment, DataFeed, FeedMode, MarketSession
from packages.contracts.models import Bar, Quote
from packages.core.clock import ManualClock
from services.market_data.calendar import get_calendar
from services.market_data.common import PENDING_DATASET_VERSION

# 2026-11-02 (Mon) .. 2026-11-04 (Wed) are regular XNYS sessions (EST: open 14:30Z, close 21:00Z)
DAY = date(2026, 11, 3)


def session_open(d: date = DAY) -> datetime:
    return get_calendar().session_open(d)


def bar_record(symbol: str, start: datetime, *, o: str = "100.00", h: str = "100.50", lo: str = "99.50",
               c: str = "100.20", v: str = "1000", feed: DataFeed = DataFeed.IEX, timeframe: str = "1Min",
               latency_s: float = 1.0, **overrides: Any) -> dict[str, Any]:
    step = timedelta(days=1) if timeframe == "1Day" else timedelta(minutes=1)
    end = overrides.pop("end_ts", start + step)
    rec: dict[str, Any] = {
        "symbol": symbol, "timeframe": timeframe, "start_ts": start, "end_ts": end,
        "open": Decimal(o), "high": Decimal(h), "low": Decimal(lo), "close": Decimal(c), "volume": Decimal(v),
        "vwap": None, "trade_count": None, "feed": feed, "adjustment": Adjustment.RAW,
        "session": MarketSession.REGULAR, "provider_ts": None, "received_ts": end + timedelta(seconds=latency_s),
        "available_at_ts": end + timedelta(seconds=latency_s), "source_id": "test:handcrafted",
        "dataset_version": PENDING_DATASET_VERSION, "is_synthetic": feed == DataFeed.SYNTHETIC,
    }
    rec.update(overrides)
    return rec


def make_bar(symbol: str, start: datetime, **kw: Any) -> Bar:
    return Bar.model_validate(bar_record(symbol, start, **kw))


def minute_bars(symbol: str, start: datetime, n: int, *, base: Decimal = Decimal("100.00"),
                step: Decimal = Decimal("0.01"), feed: DataFeed = DataFeed.IEX, skip: Sequence[int] = (),
                latency_s: float = 1.0) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        if i in skip:
            continue
        px = base + step * i
        out.append(make_bar(symbol, start + timedelta(minutes=i), o=str(px), h=str(px + Decimal("0.05")),
                            lo=str(px - Decimal("0.05")), c=str(px + Decimal("0.01")), feed=feed, latency_s=latency_s))
    return out


def quote_record(symbol: str, ts: datetime, *, bid: str = "100.00", ask: str = "100.02", size: str = "100",
                 feed: DataFeed = DataFeed.IEX, latency_s: float = 0.05, **overrides: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "symbol": symbol, "bid": Decimal(bid), "ask": Decimal(ask), "bid_size": Decimal(size), "ask_size": Decimal(size),
        "bid_exchange": None, "ask_exchange": None, "exchange_ts": ts, "provider_ts": None,
        "received_ts": ts + timedelta(seconds=latency_s), "available_at_ts": ts + timedelta(seconds=latency_s),
        "feed": feed, "conditions": (), "tape": None, "source_id": "test:handcrafted",
        "is_synthetic": feed == DataFeed.SYNTHETIC,
    }
    rec.update(overrides)
    return rec


def make_quote(symbol: str, ts: datetime, **kw: Any) -> Quote:
    return Quote.model_validate(quote_record(symbol, ts, **kw))


class FakeHistoricalProvider:
    """Deterministic in-memory provider: historical bars are served only once knowable at ``clock.now()``.

    ``stream_script`` is a list of quotes (or Exception instances); the stream advances the ManualClock to each
    quote's availability and raises the exception when it reaches one (a disconnect).
    """

    def __init__(self, clock: ManualClock, bars: Sequence[Bar], *, feed: DataFeed = DataFeed.SYNTHETIC,
                 feed_mode: FeedMode = FeedMode.SIMULATED, stream_scripts: Sequence[Sequence[Any]] = ()) -> None:
        self.clock = clock
        self.feed = feed
        self.feed_mode = feed_mode
        self.source_id = "fake:historical"
        self.bars = list(bars)
        self.stream_scripts = [list(s) for s in stream_scripts]
        self.calls: list[tuple[str, datetime, datetime]] = []
        self.fail_next: Exception | None = None
        self.gate: Any = None  # optional asyncio.Event the next get_bars waits on
        self.gated_calls = 0

    async def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> list[Bar]:
        self.calls.append((symbol, start, end))
        if self.gate is not None:
            gate, self.gate = self.gate, None
            self.gated_calls += 1
            await gate.wait()
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        now = self.clock.now()
        return [b for b in self.bars if b.symbol == symbol and start <= b.start_ts < end and b.available_at_ts <= now]

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        return None

    async def stream_quotes(self, symbols: Sequence[str]) -> AsyncIterator[Quote]:
        """Script items: Quote (clock moves to its availability), BaseException (raised = disconnect),
        ``("advance", timedelta)``, ``("gate", asyncio.Event)`` (next get_bars waits on it), ``("wait", asyncio.Event)``
        (block the stream until set)."""
        script = self.stream_scripts.pop(0) if self.stream_scripts else []
        for item in script:
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, tuple):
                op, arg = item
                if op == "advance":
                    self.clock.advance(arg.total_seconds())
                elif op == "gate":
                    self.gate = arg
                elif op == "wait":
                    await arg.wait()
                continue
            if item.available_at_ts > self.clock.now():
                self.clock.set(item.available_at_ts)
            yield item


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)  # type: ignore[misc]
