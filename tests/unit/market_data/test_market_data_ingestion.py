"""IngestionService: explicit states, backfill-before-entries on (re)connect, fail closed (fake provider, no network)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from packages.contracts.enums import DataFeed, FeedMode
from packages.contracts.models import Bar, Quote
from packages.core.clock import ManualClock
from packages.core.db import Database
from services.market_data.ingestion import BackfillError, IngestionService, IngestionState
from services.market_data.quality import FeedQualityMonitor, validate_quotes
from services.market_data.store import MarketDataStore
from services.market_data.trades import Trade
from tests.unit.market_data.md_helpers import DAY, FakeHistoricalProvider, make_quote, minute_bars, quote_record, session_open

OPEN = session_open(DAY)
SYMS = ("SPY", "QQQ")
START = OPEN + timedelta(minutes=60, seconds=30)  # 15:30:30Z, one hour into the session


def session_bars() -> list[Bar]:
    return [b for s in SYMS for b in minute_bars(s, OPEN, 390)]


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    d = Database(tmp_path / "ledger.sqlite")
    yield d
    d.close()


def make_service(tmp_path: Path, provider: Any, clock: ManualClock, db: Database | None = None,
                 **kw: Any) -> IngestionService:
    store = MarketDataStore(tmp_path / "datasets")
    monitor = FeedQualityMonitor(FeedMode.IEX_FREE, provider.source_id, clock, symbols=SYMS)
    kw.setdefault("backfill_lookback", timedelta(hours=2))
    kw.setdefault("reconnect_backoff_s", (0.01, 0.05))
    return IngestionService(provider, store, db, clock, monitor, **kw)


def fresh_quotes(svc: IngestionService, clock: ManualClock) -> None:
    for s in SYMS:
        svc.monitor.observe_quote(make_quote(s, clock.now() - timedelta(seconds=1)))


def fake(clock: ManualClock, bars: Sequence[Bar] | None = None, **kw: Any) -> FakeHistoricalProvider:
    return FakeHistoricalProvider(clock, session_bars() if bars is None else bars, feed=DataFeed.IEX,
                                  feed_mode=FeedMode.IEX_FREE, **kw)


async def test_entries_locked_until_backfill_and_fresh_quotes(tmp_path: Path, db: Database) -> None:
    clock = ManualClock(START)
    svc = make_service(tmp_path, fake(clock), clock, db)
    assert svc.state == IngestionState.DISCONNECTED and not svc.entries_allowed()
    fresh_quotes(svc, clock)
    assert not svc.entries_allowed()  # fresh quotes alone are not enough: never backfilled
    result = await svc.connect()
    assert svc.state == IngestionState.LIVE and result.ok
    assert result.expected_bars == {"SPY": 60, "QQQ": 60} and result.missing_bars == {"SPY": 0, "QQQ": 0}
    bars = svc.store.load_bars(["SPY"], None, None, "1Min", feed=DataFeed.IEX)
    assert [b.start_ts for b in bars] == [OPEN + timedelta(minutes=i) for i in range(60)]
    assert svc.entries_allowed()
    clock.advance(30)
    allowed, reasons = svc.entries_status()
    assert not allowed and any("STALE" in r for r in reasons)  # quotes went stale -> fail closed
    states = [r["to_state"] for r in db.query("SELECT to_state FROM md_feed_state_transitions ORDER BY seq")]
    assert states == ["CONNECTING", "BACKFILLING", "LIVE"]
    run = db.query_one("SELECT status, rows_accepted, gaps FROM md_ingestion_runs")
    assert run == {"status": "OK", "rows_accepted": 120, "gaps": 0}


async def test_reconnect_backfills_gap_before_entries_resume(tmp_path: Path, db: Database) -> None:
    clock = ManualClock(START)
    gate = asyncio.Event()
    hold = asyncio.Event()
    script1: list[Any] = [make_quote("SPY", START + timedelta(seconds=1)), make_quote("QQQ", START + timedelta(seconds=1)),
                          ("advance", timedelta(minutes=20)),  # Mac asleep / network partition for 20 minutes
                          ("gate", gate), ConnectionError("socket closed")]
    t2 = START + timedelta(minutes=20, seconds=5)
    script2: list[Any] = [make_quote("SPY", t2), make_quote("QQQ", t2), ("wait", hold)]
    provider = fake(clock, stream_scripts=[script1, script2])
    svc = make_service(tmp_path, provider, clock, db)
    stop = asyncio.Event()
    task = asyncio.create_task(svc.run(stop))

    async def until(cond: Any, what: str) -> None:
        for _ in range(500):
            if cond():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"timed out waiting for {what}")

    await until(lambda: provider.gated_calls == 1, "the reconnect backfill to start")
    # the reconnect backfill is in flight: fresh quotes are observed, yet entries stay locked
    fresh_quotes(svc, clock)
    allowed, reasons = svc.entries_status()
    assert svc.state == IngestionState.BACKFILLING and not allowed
    assert any("BACKFILLING" in r for r in reasons)
    assert len(svc.store.load_bars(["SPY"], None, None, "1Min", feed=DataFeed.IEX)) == 60  # gap not yet filled
    gate.set()
    await until(lambda: svc.state == IngestionState.LIVE and svc.monitor.status()["status"] == "OK", "LIVE after backfill")
    assert svc.entries_allowed()
    # the gap [15:30, 15:50) was fetched from the last stored bar end and stored contiguously before LIVE
    second_round = [c for c in provider.calls if c[1] == OPEN + timedelta(minutes=60)]
    assert {c[0] for c in second_round} == set(SYMS)
    for s in SYMS:
        bars = svc.store.load_bars([s], None, None, "1Min", feed=DataFeed.IEX)
        assert [b.start_ts for b in bars] == [OPEN + timedelta(minutes=i) for i in range(80)]
    stop.set()
    hold.set()
    await asyncio.wait_for(task, timeout=5)
    states = [r["to_state"] for r in db.query("SELECT to_state FROM md_feed_state_transitions ORDER BY seq")]
    assert states[:7] == ["CONNECTING", "BACKFILLING", "LIVE", "DISCONNECTED", "CONNECTING", "BACKFILLING", "LIVE"]
    assert states[-1] == "DISCONNECTED"  # stopped
    assert len(db.query("SELECT run_id FROM md_ingestion_runs")) == 2
    assert db.query_one("SELECT count(*) AS n FROM failures WHERE component = 'market_data.stream'") == {"n": 1}


async def test_backfill_failure_keeps_entries_locked(tmp_path: Path, db: Database) -> None:
    clock = ManualClock(START)
    provider = fake(clock)
    provider.fail_next = TimeoutError("historical API timeout")
    svc = make_service(tmp_path, provider, clock, db)
    fresh_quotes(svc, clock)
    with pytest.raises(BackfillError, match="SPY"):
        await svc.connect()
    allowed, reasons = svc.entries_status()
    assert svc.state == IngestionState.DISCONNECTED and not allowed
    assert any("last backfill failed" in r for r in reasons)
    assert db.query_one("SELECT status FROM md_ingestion_runs") == {"status": "ERROR"}
    assert db.query_one("SELECT kind FROM failures WHERE component = 'market_data.backfill'") == {"kind": "TimeoutError"}
    await svc.connect()  # the next attempt succeeds
    fresh_quotes(svc, clock)
    assert svc.state == IngestionState.LIVE and svc.entries_allowed()


async def test_thin_backfill_stays_backfilling(tmp_path: Path) -> None:
    clock = ManualClock(START)
    provider = fake(clock, bars=[b for b in session_bars() if b.symbol == "QQQ"])  # SPY returns nothing
    svc = make_service(tmp_path, provider, clock)
    fresh_quotes(svc, clock)
    result = await svc.connect()
    assert result.ok and result.missing_bars["SPY"] == 60
    allowed, reasons = svc.entries_status()
    assert svc.state == IngestionState.BACKFILLING and not allowed
    assert any("too little data" in r for r in reasons)
    provider.bars = session_bars()
    await svc.connect()
    fresh_quotes(svc, clock)
    assert svc.state == IngestionState.LIVE and svc.entries_allowed()


async def test_gaps_are_reported_not_filled(tmp_path: Path, db: Database) -> None:
    clock = ManualClock(START)
    bars = [b for b in session_bars() if not (b.symbol == "SPY" and OPEN + timedelta(minutes=10) <= b.start_ts
                                              < OPEN + timedelta(minutes=13))]
    svc = make_service(tmp_path, fake(clock, bars=bars), clock, db)
    result = await svc.connect()
    assert [(g.symbol, g.start, g.missing) for g in result.gaps] == [("SPY", OPEN + timedelta(minutes=10), 3)]
    stored = svc.store.load_bars(["SPY"], None, None, "1Min", feed=DataFeed.IEX)
    assert len(stored) == 57  # never filled
    gap_rows = db.query("SELECT symbol, kind FROM data_quality_events WHERE kind = 'GAP'")
    assert gap_rows == [{"symbol": "SPY", "kind": "GAP"}]
    assert svc.monitor.status()["symbols"][1]["gaps_today"] == 1  # SPY row (sorted: QQQ, SPY)
    strict = make_service(tmp_path / "strict", fake(clock, bars=bars), clock, block_on_unfilled_gaps=True)
    fresh_quotes(strict, clock)
    await strict.connect()
    assert strict.state == IngestionState.BACKFILLING and not strict.entries_allowed()


async def test_streamed_bar_hole_triggers_backfill(tmp_path: Path) -> None:
    clock = ManualClock(START)
    provider = fake(clock)
    svc = make_service(tmp_path, provider, clock)
    await svc.connect()
    clock.set(OPEN + timedelta(minutes=66, seconds=3))
    streamed = minute_bars("SPY", OPEN + timedelta(minutes=65), 1)[0]  # 15:35 bar; 15:30..15:34 missing
    n_calls = len(provider.calls)
    await svc.on_stream_bar(streamed)
    assert len(provider.calls) > n_calls and provider.calls[n_calls][1] == OPEN + timedelta(minutes=60)
    assert svc.state == IngestionState.LIVE
    await svc.flush()
    starts = [b.start_ts for b in svc.store.load_bars(["SPY"], None, None, "1Min", feed=DataFeed.IEX)]
    assert starts == [OPEN + timedelta(minutes=i) for i in range(66)]


class EventProvider(FakeHistoricalProvider):
    """Provider exposing ``stream_events`` (one multiplexed connection) and the ``on_stream_report`` hook."""

    def __init__(self, clock: ManualClock, bars: Sequence[Bar], events: Sequence[Any], hold: asyncio.Event) -> None:
        super().__init__(clock, bars, feed=DataFeed.IEX, feed_mode=FeedMode.IEX_FREE)
        self.events = list(events)
        self.hold = hold
        self.on_stream_report: Any = None
        self.quote_streams = 0

    async def stream_quotes(self, symbols: Sequence[str]) -> AsyncIterator[Quote]:  # must not be used
        self.quote_streams += 1
        raise AssertionError("separate quote stream opened although stream_events exists")
        yield  # pragma: no cover

    async def stream_events(self, symbols: Sequence[str]) -> AsyncIterator[Quote | Bar | Trade]:
        for item in self.events:
            if isinstance(item, dict):  # a malformed record the provider's validator rejects
                report = validate_quotes([item], validated_at=self.clock.now())
                assert report.rejected
                if self.on_stream_report is not None:
                    self.on_stream_report(report)
                continue
            if item.available_at_ts > self.clock.now():
                self.clock.set(item.available_at_ts)
            yield item
        await self.hold.wait()


async def test_multiplexed_stream_and_stream_rejects_preserved(tmp_path: Path, db: Database) -> None:
    clock = ManualClock(START)
    hold = asyncio.Event()
    bar = minute_bars("SPY", OPEN + timedelta(minutes=60), 1)[0]  # 15:30 bar, available 15:31:01
    t = bar.available_at_ts + timedelta(seconds=1)
    crossed = quote_record("SPY", t, bid="100.10", ask="100.00")
    trade = Trade(symbol="SPY", price=Decimal("100.61"), size=Decimal(100), exchange="V", trade_id="9", exchange_ts=t,
                  received_ts=t + timedelta(milliseconds=40), available_at_ts=t + timedelta(milliseconds=40),
                  feed=DataFeed.IEX, source_id="fake:trades", is_synthetic=False)
    events = [bar, make_quote("SPY", t), crossed, make_quote("QQQ", t), trade]
    provider = EventProvider(clock, session_bars(), events, hold)
    svc = make_service(tmp_path, provider, clock, db, flush_interval_s=3600)
    assert provider.on_stream_report is not None  # hook wired by the service
    stop = asyncio.Event()
    task = asyncio.create_task(svc.run(stop))
    for _ in range(300):
        if svc.monitor.status()["status"] == "OK" and svc._trade_buf:  # pyright: ignore[reportPrivateUsage]
            break
        await asyncio.sleep(0.01)
    assert svc.entries_allowed() and provider.quote_streams == 0
    spy = next(r for r in svc.monitor.status()["symbols"] if r["symbol"] == "SPY")
    assert spy["rejected_today"] == 1
    stop.set()
    hold.set()
    await asyncio.wait_for(task, timeout=5)  # run() flushes on exit
    raw = tmp_path / "datasets" / "raw" / "IEX" / "quotes" / "symbol=SPY" / "date=2026-11-03" / "part.parquet"
    statuses = pq.read_table(raw).column("quality_status").to_pylist()
    assert sorted(statuses) == ["ACCEPTED", "REJECTED"]  # the crossed quote is preserved raw, not dropped
    assert db.query_one("SELECT count(*) AS n FROM data_quality_events WHERE kind = 'CROSSED_QUOTE'") == {"n": 1}
    assert len(svc.store.load_quotes(["SPY", "QQQ"], None, None, feed=DataFeed.IEX)) == 2
    bars = svc.store.load_bars(["SPY"], None, None, "1Min", feed=DataFeed.IEX)
    assert bars[-1].start_ts == OPEN + timedelta(minutes=60)  # streamed bar continued the series and was stored
    assert svc.store.load_trades(["SPY"], None, None, feed=DataFeed.IEX) == [trade]  # trade prints persisted too
    assert svc.monitor.status()["last_trade_ts"] is not None


def test_requires_symbols(tmp_path: Path) -> None:
    clock = ManualClock(START)
    monitor = FeedQualityMonitor(FeedMode.IEX_FREE, "x", clock)
    with pytest.raises(ValueError, match="symbol"):
        IngestionService(fake(clock), MarketDataStore(tmp_path), None, clock, monitor)
