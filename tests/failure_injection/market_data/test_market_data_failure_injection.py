"""Failure injection for market data: corrupt storage, lock contention, stream faults, provider errors.

Everything is local and deterministic (fake streams/clients, temporary directories); nothing touches the network.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from alpaca.data.models.bars import Bar as AlpacaBar
from alpaca.data.models.quotes import Quote as AlpacaQuote
from alpaca.data.models.trades import Trade as AlpacaTrade

from packages.contracts.enums import DataFeed, FeedMode
from packages.core.clock import ManualClock
from packages.core.secrets import SecretStore
from services.market_data.alpaca_provider import AlpacaMarketDataProvider, MarketDataUnavailable, StreamDisconnected
from services.market_data.fixtures import FixtureIntegrityError, ensure_synthetic_fixture
from services.market_data.ingestion import BackfillError, IngestionService, IngestionState
from services.market_data.manifest import DatasetSpec
from services.market_data.quality import FeedQualityMonitor, ValidationReport, validate_quotes
from services.market_data.store import MarketDataStore
from tests.unit.market_data.md_helpers import DAY, make_quote, minute_bars, quote_record, session_open, utc

OPEN = session_open(DAY)
T0 = utc(2026, 11, 3, 15, 0)


def spec(**kw: Any) -> DatasetSpec:
    return DatasetSpec(**{"source_id": "fi:test", "feed": DataFeed.IEX, "generator": "fi", "generator_version": "1",
                          "is_synthetic": False, **kw})


# ------------------------------------------------------------------------------------------ storage
def test_corrupt_partition_quarantined_and_earlier_dataset_fails_verify(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path)
    m1 = store.write_bars(minute_bars("SPY", OPEN, 5), manifest=spec(description="first"))
    part = tmp_path / "raw" / "IEX" / "bars" / "symbol=SPY" / "date=2026-11-03" / "part.parquet"
    part.write_bytes(b"PAR1 this is not parquet anymore")
    m2 = store.write_bars(minute_bars("SPY", OPEN + timedelta(minutes=5), 5, base=Decimal("100.05")),
                          manifest=spec(description="second"))
    entry = next(f for f in m2["files"] if f["path"].endswith("date=2026-11-03/part.parquet"))
    assert entry["quarantined_previous"].startswith("raw/IEX/bars/symbol=SPY/date=2026-11-03/part.parquet.corrupt-")
    assert (tmp_path / entry["quarantined_previous"]).read_bytes() == b"PAR1 this is not parquet anymore"  # evidence kept
    assert store.verify_dataset(m2["dataset_version"])["ok"]
    v1 = store.verify_dataset(m1["dataset_version"])
    # the first dataset's raw rows are gone from the live partition: superseded by a later write, reported as a warning
    assert v1["ok"] and any("superseded" in w for w in v1["warnings"])
    assert len(store.load_bars(["SPY"], None, None, "1Min")) == 10  # normalized tables are unaffected


def test_deleted_fixture_partition_is_regenerated(tmp_path: Path) -> None:
    m = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    victim = tmp_path / m["files"][0]["path"]
    victim.unlink()
    assert not MarketDataStore(tmp_path).verify_dataset(m["dataset_version"])["ok"]
    again = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    assert again["dataset_version"] == m["dataset_version"] and victim.is_file()
    assert MarketDataStore(tmp_path).verify_dataset(m["dataset_version"])["ok"]


def test_removed_raw_rows_are_restored_by_regeneration(tmp_path: Path) -> None:
    m = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    path = tmp_path / next(f["path"] for f in m["files"] if "/quotes/" in f["path"])
    original = path.read_bytes()
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)  # readable, but one raw row silently removed
    assert not MarketDataStore(tmp_path).verify_dataset(m["dataset_version"])["ok"]
    ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    assert path.read_bytes() == original  # deterministic regeneration restores the exact bytes


def test_altered_raw_values_fail_loudly(tmp_path: Path) -> None:
    m = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    path = tmp_path / next(f["path"] for f in m["files"] if "/quotes/" in f["path"])
    rows = pq.read_table(path).to_pylist()
    rows[0]["bid"] = rows[0]["bid"] - Decimal("0.01")  # a readable but falsified price
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    with pytest.raises(FixtureIntegrityError, match="still fails verification"):
        ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)


def test_manifest_json_edit_is_detected_and_restored(tmp_path: Path) -> None:
    m = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    mpath = tmp_path / "processed" / "manifests" / f"{m['dataset_version']}.json"
    data = json.loads(mpath.read_text())
    data["is_synthetic"] = False  # an attempt to relabel synthetic data as real
    mpath.write_text(json.dumps(data))
    check = MarketDataStore(tmp_path).verify_dataset(m["dataset_version"])
    assert not check["ok"] and any("differs" in e for e in check["errors"])
    ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    assert json.loads(mpath.read_text())["is_synthetic"] is True  # registry copy restored


def test_duckdb_lock_contention_retries_then_fails_closed(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path, lock_timeout_s=0.5)
    store.write_bars(minute_bars("SPY", OPEN, 2), manifest=spec())
    holder = subprocess.Popen(  # another process holds the DuckDB write lock for 3 seconds
        [sys.executable, "-c", "import duckdb,sys,time; c=duckdb.connect(sys.argv[1]); print('locked', flush=True); "
         "time.sleep(3)", str(store.duckdb_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        t = time.monotonic()
        with pytest.raises(duckdb.IOException, match=r"(?i)lock"):
            store.load_bars(["SPY"], None, None, "1Min")
        assert time.monotonic() - t >= 0.4  # it retried for the configured window before failing
        patient = MarketDataStore(tmp_path, lock_timeout_s=10)
        assert len(patient.load_bars(["SPY"], None, None, "1Min")) == 2  # succeeds once the lock is released
    finally:
        holder.kill()
        holder.wait()


# ------------------------------------------------------------------------------------------ alpaca stream faults
def secrets(tmp_path: Path) -> SecretStore:
    return SecretStore(tmp_path, use_keychain=False, env={"ALPACA_PAPER_KEY_ID": "PKFAKE", "ALPACA_PAPER_SECRET_KEY": "s"},
                       dotenv_path=tmp_path / "none")


class ReconnectingStream:
    """Mimics alpaca-py DataStream: _run_forever calls _start_ws, dispatches, then silently reconnects."""

    def __init__(self, batches: list[list[Any]]) -> None:
        self.batches = batches
        self.quote_handler: Callable[[Any], Any] | None = None
        self.bar_handler: Callable[[Any], Any] | None = None
        self.trade_handler: Callable[[Any], Any] | None = None
        self.stopped = False
        self.connects = 0

    def subscribe_quotes(self, handler: Callable[[Any], Any], *symbols: str) -> None:
        self.quote_handler = handler

    def subscribe_bars(self, handler: Callable[[Any], Any], *symbols: str) -> None:
        self.bar_handler = handler

    def subscribe_trades(self, handler: Callable[[Any], Any], *symbols: str) -> None:
        self.trade_handler = handler

    async def _start_ws(self) -> None:
        self.connects += 1

    async def _run_forever(self) -> None:
        for batch in self.batches:
            await self._start_ws()  # the SDK reconnects on its own after a websocket error
            for msg in batch:
                handler = (self.bar_handler if isinstance(msg, AlpacaBar) else
                           self.trade_handler if isinstance(msg, AlpacaTrade) else self.quote_handler)
                assert handler is not None
                await handler(msg)
            await asyncio.sleep(0.01)
        await asyncio.sleep(3600)

    async def stop_ws(self) -> None:
        self.stopped = True


def _q(ts: str, bid: float = 680.1, ask: float = 680.12) -> AlpacaQuote:
    return AlpacaQuote("SPY", {"t": ts, "bp": bid, "bs": 1, "ap": ask, "as": 1})


async def test_sdk_silent_reconnect_surfaces_as_disconnect(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    bar = AlpacaBar("SPY", {"t": "2026-11-03T14:59:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "vw": 1})
    trade = AlpacaTrade("SPY", {"t": "2026-11-03T14:59:59.5Z", "x": "V", "p": 680.11, "s": 100, "i": 7, "c": ["@"], "z": "B"})
    fake = ReconnectingStream([[_q("2026-11-03T15:00:00Z"), bar, trade], [_q("2026-11-03T15:00:01Z")]])
    p = AlpacaMarketDataProvider(secrets(tmp_path), FeedMode.IEX_FREE, clock, stream_factory=lambda: fake)
    got: list[Any] = []

    async def drain() -> None:
        stream: AsyncIterator[Any] = p.stream_events(["SPY"])
        async for item in stream:
            got.append(item)

    with pytest.raises(StreamDisconnected, match="reconnected"):
        await asyncio.wait_for(drain(), timeout=5)
    assert fake.connects == 2 and fake.stopped
    assert [type(x).__name__ for x in got] == ["Quote", "Bar", "Trade"]  # one websocket carried all channels
    assert got[2].price == Decimal("680.11") and got[2].trade_id == "7" and got[2].exchange == "V"


async def test_malformed_and_crossed_stream_messages_are_rejected_not_yielded(tmp_path: Path) -> None:
    clock = ManualClock(T0 + timedelta(seconds=3))
    msgs = [_q("2026-11-03T15:00:00Z"), {"S": "SPY", "t": "not-a-time", "bp": 1, "ap": 1, "bs": 1, "as": 1},
            _q("2026-11-03T15:00:02Z", bid=680.2, ask=680.1)]
    fake = ReconnectingStream([msgs])
    p = AlpacaMarketDataProvider(secrets(tmp_path), FeedMode.IEX_FREE, clock, stream_factory=lambda: fake)
    reports: list[ValidationReport] = []
    p.on_stream_report = reports.append
    got: list[Any] = []

    async def take() -> None:
        async for q in p.stream_quotes(["SPY"]):
            got.append(q)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(take(), timeout=1.5)
    assert len(got) == 1
    reasons = [r.reasons for rep in reports for r in rep.rejected]
    assert reasons == [("INVALID_RECORD",), ("CROSSED_QUOTE",)]
    assert "not-a-time" in str(reports[0].rejected[0].record.get("raw_json"))  # verbatim payload kept for raw storage


async def test_stream_queue_overflow_is_a_disconnect(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    fake = ReconnectingStream([[_q(f"2026-11-03T15:00:0{i}Z") for i in range(5)]])
    p = AlpacaMarketDataProvider(secrets(tmp_path), FeedMode.IEX_FREE, clock, stream_factory=lambda: fake,
                                 stream_queue_size=2)

    async def slow() -> None:
        async for _ in p.stream_quotes(["SPY"]):
            await asyncio.sleep(0.05)

    with pytest.raises(StreamDisconnected, match="overflow"):
        await asyncio.wait_for(slow(), timeout=5)


# ------------------------------------------------------------------------------------------ provider errors
class RateLimitedClient:
    def __init__(self) -> None:
        self.calls = 0

    def get_stock_bars(self, req: Any) -> Any:
        self.calls += 1
        err = Exception("too many requests")
        err.status_code = 429  # type: ignore[attr-defined]
        raise err


async def test_rate_limit_during_backfill_keeps_entries_locked(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    client = RateLimitedClient()
    p = AlpacaMarketDataProvider(secrets(tmp_path), FeedMode.IEX_FREE, clock, client=client)
    with pytest.raises(MarketDataUnavailable, match="429"):
        await p.get_bars("SPY", T0 - timedelta(hours=1), T0, "1Min")
    monitor = FeedQualityMonitor(FeedMode.IEX_FREE, p.source_id, clock, symbols=("SPY",))
    svc = IngestionService(p, MarketDataStore(tmp_path / "ds"), None, clock, monitor, backfill_lookback=timedelta(hours=1))
    monitor.observe_quote(make_quote("SPY", T0 - timedelta(seconds=1)))
    with pytest.raises(BackfillError, match="429"):
        await svc.connect()
    assert svc.state == IngestionState.DISCONNECTED and not svc.entries_allowed()
    assert client.calls == 2  # one call each; never a retry storm


async def test_quotes_with_future_timestamps_rejected_as_clock_skew() -> None:
    rec = quote_record("SPY", T0 + timedelta(seconds=30), received_ts=T0, available_at_ts=T0 + timedelta(seconds=30))
    rep = validate_quotes([rec])
    assert [r.reasons for r in rep.rejected] == [("CLOCK_SKEW",)]


async def test_quote_beyond_loaded_calendar_fails_closed() -> None:
    far = utc(2099, 6, 1, 15, 0)
    rep = validate_quotes([quote_record("SPY", far)])
    assert not rep.accepted and rep.rejected[0].reasons == ("INVALID_RECORD",)
