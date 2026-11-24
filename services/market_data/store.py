"""Analytical market-data store: raw partitioned Parquet + normalized DuckDB tables + provenance manifests.

Layout (relative to ``datasets_dir``)::

    raw/<FEED>/bars/symbol=<S>/date=<YYYY-MM-DD>/part.parquet     raw intraday bar events
    raw/<FEED>/bars_daily/symbol=<S>/year=<YYYY>/part.parquet     raw daily bar events (yearly: avoids 1 file/day)
    raw/<FEED>/quotes/symbol=<S>/date=<YYYY-MM-DD>/part.parquet   raw quote events
    raw/<FEED>/trades/symbol=<S>/date=<YYYY-MM-DD>/part.parquet   raw trade prints (streamed time & sales)
    raw/<FEED>/corporate_actions/symbol=<S>/part.parquet          raw corporate-action rows
    analytics.duckdb                                              normalized tables: bars, quotes, trades, corporate_actions,
                                                                  datasets (registry), active_series, universe_snapshots
    processed/manifests/<dataset_version>.json                    one provenance manifest per write batch

Rules
* Prices are ``DECIMAL(18,6)`` (volumes/sizes ``DECIMAL(24,6)``) in DuckDB and ``decimal128`` in Parquet; never
  float. The only float exposure is :meth:`MarketDataStore.load_bars_df` (research convenience, documented there).
* Timestamps are timezone-aware UTC at the API; DuckDB stores them as ``TIMESTAMP`` holding UTC wall time (the
  installed DuckDB needs ``pytz`` for TIMESTAMPTZ parameters, which is not a project dependency).
* ``date=``/``year=`` partitions use the America/New_York calendar date of the record's start/exchange timestamp.
* Raw partitions preserve *every* received record, including ones the validator rejected (``quality_status`` =
  REJECTED with reasons and the verbatim payload in ``raw_json``). Normalized tables hold accepted rows only.
* Partitions are shared by every dataset of the same feed: a later write merges into an existing ``part.parquet``
  (exact-duplicate rows collapse). The manifest records each file's sha256 at write time (``verify_dataset``
  distinguishes "superseded by a later dataset" from "modified/missing") and the ``content_sha256`` of the dataset's
  *own* rows in that partition; only the latter enters ``dataset_version``, so identical content always gets the
  same version whatever else the store already held.
* Point-in-time: every load accepts ``asof`` and returns only rows with ``available_at_ts <= asof``. Among
  revisions of the same bar key the latest one *available at asof* wins (then the latest written).
* Feeds are never spliced: when ``feed`` is not given, each symbol's rows come from exactly one feed (the highest
  priority feed with rows in the requested window: SIP, IEX, LEAN_SAMPLE, SYNTHETIC).
* Series are never spliced either: a *snapshot* dataset (fixture / imported bundle) becomes the active series of its
  (feed, symbol)s and, when one is active, loads read only that dataset; *incremental* datasets (ingestion batches)
  compose by bar key. An explicit ``dataset_version`` bypasses the selection.
* Datasets generated with injected defects (validator tests) are excluded from loads unless requested by version.
* DuckDB connections are short-lived (opened per operation, serialized per process) so a CLI and the API process
  can share the file; lock contention is retried for up to ``lock_timeout_s``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from packages.contracts.enums import Adjustment, DataFeed, MarketSession
from packages.contracts.models import Bar, Quote
from packages.core.canonical import canonical_json

from .calendar import NY
from .common import FEED_PRIORITY, canonical_timeframe, clean_decimal, iso, quantize_price, to_utc
from .corporate_actions import CorporateAction, CorporateActionType
from .manifest import (
    SERIES_SNAPSHOT,
    DatasetSpec,
    build_manifest,
    file_sha256,
    manifest_content_ok,
    read_manifest_file,
    write_manifest_file,
)
from .trades import Trade

log = logging.getLogger("lat.market_data.store")

PRICE_T = pa.decimal128(18, 6)
SIZE_T = pa.decimal128(24, 6)
FACTOR_T = pa.decimal128(24, 12)
TS_UTC = pa.timestamp("us", tz="UTC")
TS_NAIVE = pa.timestamp("us")
_Q12 = Decimal(1).scaleb(-12)

_DAILY_TIMEFRAMES: tuple[str, ...] = ("1Day", "1Week", "1Month")

QUALITY_ACCEPTED = "ACCEPTED"
QUALITY_FLAGGED = "FLAGGED"
QUALITY_REJECTED = "REJECTED"

_BAR_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("symbol", pa.string()),
    ("timeframe", pa.string()),
    ("start_ts", TS_UTC),
    ("end_ts", TS_UTC),
    ("open", PRICE_T),
    ("high", PRICE_T),
    ("low", PRICE_T),
    ("close", PRICE_T),
    ("volume", SIZE_T),
    ("vwap", PRICE_T),
    ("trade_count", pa.int64()),
    ("feed", pa.string()),
    ("adjustment", pa.string()),
    ("session", pa.string()),
    ("provider_ts", TS_UTC),
    ("received_ts", TS_UTC),
    ("available_at_ts", TS_UTC),
    ("source_id", pa.string()),
    ("is_synthetic", pa.bool_()),
)
_QUOTE_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("symbol", pa.string()),
    ("exchange_ts", TS_UTC),
    ("bid", PRICE_T),
    ("ask", PRICE_T),
    ("bid_size", SIZE_T),
    ("ask_size", SIZE_T),
    ("bid_exchange", pa.string()),
    ("ask_exchange", pa.string()),
    ("conditions", pa.string()),
    ("tape", pa.string()),
    ("provider_ts", TS_UTC),
    ("received_ts", TS_UTC),
    ("available_at_ts", TS_UTC),
    ("feed", pa.string()),
    ("source_id", pa.string()),
    ("is_synthetic", pa.bool_()),
)
_TRADE_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("symbol", pa.string()),
    ("exchange_ts", TS_UTC),
    ("price", PRICE_T),
    ("size", SIZE_T),
    ("exchange", pa.string()),
    ("trade_id", pa.string()),
    ("conditions", pa.string()),
    ("tape", pa.string()),
    ("provider_ts", TS_UTC),
    ("received_ts", TS_UTC),
    ("available_at_ts", TS_UTC),
    ("feed", pa.string()),
    ("source_id", pa.string()),
    ("is_synthetic", pa.bool_()),
)
_CA_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("symbol", pa.string()),
    ("action_type", pa.string()),
    ("ex_date", pa.date32()),
    ("available_at_ts", TS_UTC),
    ("split_factor", FACTOR_T),
    ("cash_amount", PRICE_T),
    ("reference_price", PRICE_T),
    ("old_symbol", pa.string()),
    ("new_symbol", pa.string()),
    ("feed", pa.string()),
    ("source_id", pa.string()),
    ("raw", pa.string()),
)
_RAW_EXTRA: tuple[tuple[str, pa.DataType], ...] = (
    ("quality_status", pa.string()),
    ("quality_reasons", pa.string()),
    ("raw_json", pa.string()),
)
BAR_SCHEMA = pa.schema(list(_BAR_FIELDS))
QUOTE_SCHEMA = pa.schema(list(_QUOTE_FIELDS))
CA_SCHEMA = pa.schema(list(_CA_FIELDS))
RAW_BAR_SCHEMA = pa.schema(list(_BAR_FIELDS + _RAW_EXTRA))
RAW_QUOTE_SCHEMA = pa.schema(list(_QUOTE_FIELDS + _RAW_EXTRA))
TRADE_SCHEMA = pa.schema(list(_TRADE_FIELDS))
RAW_TRADE_SCHEMA = pa.schema(list(_TRADE_FIELDS + _RAW_EXTRA))

BAR_DF_COLUMNS: tuple[str, ...] = (
    "symbol", "timeframe", "start_ts", "end_ts", "open", "high", "low", "close", "volume", "vwap",
    "trade_count", "feed", "session", "available_at_ts", "source_id", "dataset_version", "is_synthetic",
)

_DDL = (
    """CREATE TABLE IF NOT EXISTS datasets (
        dataset_version VARCHAR PRIMARY KEY, feed VARCHAR NOT NULL, kind VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL, is_synthetic BOOLEAN NOT NULL, defects_injected BOOLEAN NOT NULL,
        created_at TIMESTAMP NOT NULL, ingest_seq BIGINT NOT NULL, manifest_json VARCHAR NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS bars (
        dataset_version VARCHAR NOT NULL, ingest_seq BIGINT NOT NULL, symbol VARCHAR NOT NULL,
        timeframe VARCHAR NOT NULL, start_ts TIMESTAMP NOT NULL, end_ts TIMESTAMP NOT NULL,
        open DECIMAL(18,6) NOT NULL, high DECIMAL(18,6) NOT NULL, low DECIMAL(18,6) NOT NULL,
        close DECIMAL(18,6) NOT NULL, volume DECIMAL(24,6) NOT NULL, vwap DECIMAL(18,6), trade_count BIGINT,
        feed VARCHAR NOT NULL, adjustment VARCHAR NOT NULL, session VARCHAR NOT NULL, provider_ts TIMESTAMP,
        received_ts TIMESTAMP NOT NULL, available_at_ts TIMESTAMP NOT NULL, source_id VARCHAR NOT NULL,
        is_synthetic BOOLEAN NOT NULL,
        PRIMARY KEY (dataset_version, symbol, timeframe, start_ts))""",
    """CREATE TABLE IF NOT EXISTS quotes (
        dataset_version VARCHAR NOT NULL, ingest_seq BIGINT NOT NULL, symbol VARCHAR NOT NULL,
        exchange_ts TIMESTAMP NOT NULL, bid DECIMAL(18,6) NOT NULL, ask DECIMAL(18,6) NOT NULL,
        bid_size DECIMAL(24,6) NOT NULL, ask_size DECIMAL(24,6) NOT NULL, bid_exchange VARCHAR,
        ask_exchange VARCHAR, conditions VARCHAR, tape VARCHAR, provider_ts TIMESTAMP,
        received_ts TIMESTAMP NOT NULL, available_at_ts TIMESTAMP NOT NULL, feed VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL, is_synthetic BOOLEAN NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS trades (
        dataset_version VARCHAR NOT NULL, ingest_seq BIGINT NOT NULL, symbol VARCHAR NOT NULL,
        exchange_ts TIMESTAMP NOT NULL, price DECIMAL(18,6) NOT NULL, size DECIMAL(24,6) NOT NULL, exchange VARCHAR,
        trade_id VARCHAR, conditions VARCHAR, tape VARCHAR, provider_ts TIMESTAMP, received_ts TIMESTAMP NOT NULL,
        available_at_ts TIMESTAMP NOT NULL, feed VARCHAR NOT NULL, source_id VARCHAR NOT NULL,
        is_synthetic BOOLEAN NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS corporate_actions (
        dataset_version VARCHAR NOT NULL, ingest_seq BIGINT NOT NULL, symbol VARCHAR NOT NULL,
        action_type VARCHAR NOT NULL, ex_date DATE NOT NULL, available_at_ts TIMESTAMP NOT NULL,
        split_factor DECIMAL(24,12), cash_amount DECIMAL(18,6), reference_price DECIMAL(18,6),
        old_symbol VARCHAR, new_symbol VARCHAR, feed VARCHAR NOT NULL, source_id VARCHAR NOT NULL, raw VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS universe_snapshots (
        snapshot_id VARCHAR NOT NULL, asof_ts TIMESTAMP NOT NULL, asof_date DATE NOT NULL, symbol VARCHAR NOT NULL,
        eligible BOOLEAN NOT NULL, reasons VARCHAR NOT NULL, last_close DECIMAL(18,6),
        avg_dollar_volume DECIMAL(24,2), sessions_observed INTEGER NOT NULL, feed VARCHAR,
        dataset_versions VARCHAR NOT NULL, policy_version VARCHAR NOT NULL, created_at TIMESTAMP NOT NULL,
        PRIMARY KEY (snapshot_id, symbol))""",
    """CREATE TABLE IF NOT EXISTS active_series (
        feed VARCHAR NOT NULL, symbol VARCHAR NOT NULL, dataset_version VARCHAR NOT NULL,
        activated_at TIMESTAMP NOT NULL, PRIMARY KEY (feed, symbol))""",
)

# Series selection. A *snapshot* dataset (generated fixture, imported bundle) is a complete, self-contained series and
# becomes the ACTIVE series of each of its (feed, symbol)s. Loads without an explicit dataset_version read exactly the
# active dataset for a (feed, symbol) when one exists, so two fixtures with different parameters can never be spliced
# into one series. Without an active snapshot, the *incremental* datasets of the feed (ingestion batches over time)
# compose and are deduplicated per bar key.

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


class StoreError(RuntimeError):
    """Storage invariant violated (duplicate keys, feed mixing, unknown dataset...). Never silently repaired."""


@dataclass(frozen=True)
class RawExtras:
    """Raw-event annotations for one write batch (produced by ``quality.ValidationReport.raw_extras``).

    ``flags``/``raw_json`` are keyed by the index of the accepted record in the batch passed to the store;
    ``rejected`` holds the original rejected records (dict form) with their reasons.
    """

    rejected: Sequence[tuple[Mapping[str, Any], Sequence[str]]] = ()
    flags: Mapping[int, Sequence[str]] = field(default_factory=dict)
    raw_json: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteColumns:
    """Columnar quotes for one symbol/feed, sorted by (available_at_ts, exchange_ts). Used by the replay cache."""

    symbol: str
    feed: DataFeed
    table: pa.Table
    available_at_us: Any  # numpy int64 array (UTC epoch microseconds)
    exchange_us: Any

    def __len__(self) -> int:
        return self.table.num_rows

    def quote_at(self, i: int) -> Quote:
        row = self.table.slice(i, 1).to_pylist()[0]
        return _quote_from_row(row)


# ------------------------------------------------------------------------------------------ conversions
def _naive(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return to_utc(ts).replace(tzinfo=None)


def _aware(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _req_aware(ts: datetime | None) -> datetime:
    out = _aware(ts)
    if out is None:
        raise StoreError("required timestamp missing in stored row")
    return out


def _q6(value: Decimal | None) -> Decimal | None:
    return None if value is None else quantize_price(value)


def _q12(value: Decimal | None) -> Decimal | None:
    return None if value is None else value.quantize(_Q12)


def _clean(value: Any) -> Any:
    return clean_decimal(value) if isinstance(value, Decimal) else value


def bars_to_table(bars: Sequence[Bar]) -> pa.Table:
    cols: dict[str, list[Any]] = {name: [] for name, _ in _BAR_FIELDS}
    for b in bars:
        cols["symbol"].append(b.symbol)
        cols["timeframe"].append(canonical_timeframe(b.timeframe))
        cols["start_ts"].append(b.start_ts)
        cols["end_ts"].append(b.end_ts)
        cols["open"].append(_q6(b.open))
        cols["high"].append(_q6(b.high))
        cols["low"].append(_q6(b.low))
        cols["close"].append(_q6(b.close))
        cols["volume"].append(_q6(b.volume))
        cols["vwap"].append(_q6(b.vwap))
        cols["trade_count"].append(b.trade_count)
        cols["feed"].append(b.feed.value)
        cols["adjustment"].append(b.adjustment.value)
        cols["session"].append(b.session.value)
        cols["provider_ts"].append(b.provider_ts)
        cols["received_ts"].append(b.received_ts)
        cols["available_at_ts"].append(b.available_at_ts)
        cols["source_id"].append(b.source_id)
        cols["is_synthetic"].append(b.is_synthetic)
    return pa.table({k: pa.array(v, type=t) for (k, t), v in zip(_BAR_FIELDS, cols.values(), strict=True)})


def quotes_to_table(quotes: Sequence[Quote]) -> pa.Table:
    cols: dict[str, list[Any]] = {name: [] for name, _ in _QUOTE_FIELDS}
    for q in quotes:
        cols["symbol"].append(q.symbol)
        cols["exchange_ts"].append(q.exchange_ts)
        cols["bid"].append(_q6(q.bid))
        cols["ask"].append(_q6(q.ask))
        cols["bid_size"].append(_q6(q.bid_size))
        cols["ask_size"].append(_q6(q.ask_size))
        cols["bid_exchange"].append(q.bid_exchange)
        cols["ask_exchange"].append(q.ask_exchange)
        cols["conditions"].append(json.dumps(list(q.conditions)) if q.conditions else None)
        cols["tape"].append(q.tape)
        cols["provider_ts"].append(q.provider_ts)
        cols["received_ts"].append(q.received_ts)
        cols["available_at_ts"].append(q.available_at_ts)
        cols["feed"].append(q.feed.value)
        cols["source_id"].append(q.source_id)
        cols["is_synthetic"].append(q.is_synthetic)
    return pa.table({k: pa.array(v, type=t) for (k, t), v in zip(_QUOTE_FIELDS, cols.values(), strict=True)})


def trades_to_table(trades: Sequence[Trade]) -> pa.Table:
    cols: dict[str, list[Any]] = {name: [] for name, _ in _TRADE_FIELDS}
    for t in trades:
        cols["symbol"].append(t.symbol)
        cols["exchange_ts"].append(t.exchange_ts)
        cols["price"].append(_q6(t.price))
        cols["size"].append(_q6(t.size))
        cols["exchange"].append(t.exchange)
        cols["trade_id"].append(t.trade_id)
        cols["conditions"].append(json.dumps(list(t.conditions)) if t.conditions else None)
        cols["tape"].append(t.tape)
        cols["provider_ts"].append(t.provider_ts)
        cols["received_ts"].append(t.received_ts)
        cols["available_at_ts"].append(t.available_at_ts)
        cols["feed"].append(t.feed.value)
        cols["source_id"].append(t.source_id)
        cols["is_synthetic"].append(t.is_synthetic)
    return pa.table({k: pa.array(v, type=ty) for (k, ty), v in zip(_TRADE_FIELDS, cols.values(), strict=True)})


def corporate_actions_to_table(actions: Sequence[CorporateAction]) -> pa.Table:
    cols: dict[str, list[Any]] = {name: [] for name, _ in _CA_FIELDS}
    for a in actions:
        cols["symbol"].append(a.symbol)
        cols["action_type"].append(a.action_type.value)
        cols["ex_date"].append(a.ex_date)
        cols["available_at_ts"].append(a.available_at_ts)
        cols["split_factor"].append(_q12(a.split_factor))
        cols["cash_amount"].append(_q6(a.cash_amount))
        cols["reference_price"].append(_q6(a.reference_price))
        cols["old_symbol"].append(a.old_symbol)
        cols["new_symbol"].append(a.new_symbol)
        cols["feed"].append(a.feed.value)
        cols["source_id"].append(a.source_id)
        cols["raw"].append(a.raw)
    return pa.table({k: pa.array(v, type=t) for (k, t), v in zip(_CA_FIELDS, cols.values(), strict=True)})


def _bar_from_row(row: Mapping[str, Any]) -> Bar:
    return Bar(
        symbol=row["symbol"],
        timeframe=row["timeframe"],
        start_ts=_req_aware(row["start_ts"]),
        end_ts=_req_aware(row["end_ts"]),
        open=_clean(row["open"]),
        high=_clean(row["high"]),
        low=_clean(row["low"]),
        close=_clean(row["close"]),
        volume=_clean(row["volume"]),
        vwap=_clean(row["vwap"]),
        trade_count=row["trade_count"],
        feed=DataFeed(row["feed"]),
        adjustment=Adjustment(row["adjustment"]),
        session=MarketSession(row["session"]),
        provider_ts=_aware(row["provider_ts"]),
        received_ts=_req_aware(row["received_ts"]),
        available_at_ts=_req_aware(row["available_at_ts"]),
        source_id=row["source_id"],
        dataset_version=row["dataset_version"],
        is_synthetic=bool(row["is_synthetic"]),
    )


def _quote_from_row(row: Mapping[str, Any]) -> Quote:
    conditions = tuple(json.loads(row["conditions"])) if row.get("conditions") else ()
    return Quote(
        symbol=row["symbol"],
        bid=_clean(row["bid"]),
        ask=_clean(row["ask"]),
        bid_size=_clean(row["bid_size"]),
        ask_size=_clean(row["ask_size"]),
        bid_exchange=row.get("bid_exchange"),
        ask_exchange=row.get("ask_exchange"),
        exchange_ts=_req_aware(row["exchange_ts"]),
        provider_ts=_aware(row.get("provider_ts")),
        received_ts=_req_aware(row["received_ts"]),
        available_at_ts=_req_aware(row["available_at_ts"]),
        feed=DataFeed(row["feed"]),
        conditions=conditions,
        tape=row.get("tape"),
        source_id=row["source_id"],
        is_synthetic=bool(row["is_synthetic"]),
    )


def _trade_from_row(row: Mapping[str, Any]) -> Trade:
    conditions = tuple(json.loads(row["conditions"])) if row.get("conditions") else ()
    return Trade(
        symbol=row["symbol"], price=_clean(row["price"]), size=_clean(row["size"]), exchange=row.get("exchange"),
        trade_id=row.get("trade_id"), conditions=conditions, tape=row.get("tape"),
        exchange_ts=_req_aware(row["exchange_ts"]), provider_ts=_aware(row.get("provider_ts")),
        received_ts=_req_aware(row["received_ts"]), available_at_ts=_req_aware(row["available_at_ts"]),
        feed=DataFeed(row["feed"]), source_id=row["source_id"], is_synthetic=bool(row["is_synthetic"]),
    )


def _ca_from_row(row: Mapping[str, Any]) -> CorporateAction:
    return CorporateAction(
        symbol=row["symbol"],
        action_type=CorporateActionType(row["action_type"]),
        ex_date=row["ex_date"],
        available_at_ts=_req_aware(row["available_at_ts"]),
        split_factor=_clean(row["split_factor"]),
        cash_amount=_clean(row["cash_amount"]),
        reference_price=_clean(row["reference_price"]),
        old_symbol=row["old_symbol"],
        new_symbol=row["new_symbol"],
        feed=DataFeed(row["feed"]),
        source_id=row["source_id"],
        raw=row["raw"],
    )


def _to_naive_table(table: pa.Table) -> pa.Table:
    fields = [pa.field(f.name, TS_NAIVE) if pa.types.is_timestamp(f.type) else f for f in table.schema]
    return table.cast(pa.schema(fields))


def _coerce_raw_value(name: str, typ: pa.DataType, value: Any) -> Any:
    """Best-effort typed value for a *rejected* raw record (invalid values become NULL; raw_json keeps them)."""
    if value is None:
        return None
    try:
        if pa.types.is_decimal(typ):
            d = value if isinstance(value, Decimal) else Decimal(str(value))
            if not d.is_finite():
                return None
            scale = typ.scale  # type: ignore[attr-defined]
            q = d.quantize(Decimal(1).scaleb(-scale))
            limit = Decimal(10) ** (typ.precision - scale)  # type: ignore[attr-defined]
            return q if abs(q) < limit else None
        if pa.types.is_timestamp(typ):
            if isinstance(value, datetime):
                return to_utc(value) if value.tzinfo is not None else None
            if isinstance(value, str):
                return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
            return None
        if pa.types.is_integer(typ):
            return int(value) if not isinstance(value, bool) else None
        if pa.types.is_boolean(typ):
            return bool(value)
        if name == "conditions" and isinstance(value, list | tuple):
            return json.dumps([str(v) for v in value])  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if hasattr(value, "value") and not isinstance(value, str):
            return str(value.value)
        return str(value)
    except (ValueError, ArithmeticError, TypeError):
        return None


def _raw_json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return format(obj, "f")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _raw_record_json(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), sort_keys=True, default=_raw_json_default)


def _ny_date(ts: datetime | None) -> str:
    if ts is None:
        return "unknown"
    return to_utc(ts).astimezone(NY).date().isoformat()


def _safe_symbol(value: Any) -> str:
    s = str(value) if value is not None else ""
    if s and all(c.isalnum() or c == "." for c in s) and len(s) <= 10:
        return s
    return "_INVALID_"


def _feed_priority_sql(column: str) -> str:
    whens = " ".join(f"WHEN '{f.value}' THEN {i}" for i, f in enumerate(FEED_PRIORITY))
    return f"CASE {column} {whens} ELSE {len(FEED_PRIORITY)} END"


# ------------------------------------------------------------------------------------------ store
class MarketDataStore:
    """See module docstring. Thread-safe within a process; multi-process safe via DuckDB file locking + retries."""

    def __init__(self, datasets_dir: Path, *, duckdb_path: Path | None = None, lock_timeout_s: float = 20.0) -> None:
        self.datasets_dir = Path(datasets_dir)
        self.raw_dir = self.datasets_dir / "raw"
        self.manifests_dir = self.datasets_dir / "processed" / "manifests"
        self.duckdb_path = duckdb_path or (self.datasets_dir / "analytics.duckdb")
        self.lock_timeout_s = lock_timeout_s
        self._lock = _path_lock(self.duckdb_path)
        self._schema_ready = False
        self._generation_path = self.raw_dir / ".generation"

    # ---------------------------------------------------------------- connections
    @contextlib.contextmanager
    def _con(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            deadline = time.monotonic() + self.lock_timeout_s
            while True:
                try:
                    con = duckdb.connect(str(self.duckdb_path))
                    break
                except duckdb.IOException as exc:  # another process holds the file lock
                    if "lock" not in str(exc).lower() or time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            try:
                con.execute("SET TimeZone='UTC'")
                if not self._schema_ready:
                    for ddl in _DDL:
                        con.execute(ddl)
                    self._schema_ready = True
                yield con
            finally:
                con.close()

    def generation(self) -> str:
        """Opaque token that changes whenever any process writes to this store (replay cache invalidation)."""
        try:
            return self._generation_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "0"

    def _bump_generation(self) -> None:
        self._generation_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._generation_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(uuid.uuid4().hex, encoding="utf-8")
        os.replace(tmp, self._generation_path)

    # ---------------------------------------------------------------- writes
    def write_bars(self, bars: Sequence[Bar], *, manifest: DatasetSpec | Mapping[str, Any],
                   extras: RawExtras | None = None, created_at: datetime | None = None) -> dict[str, Any]:
        """Write accepted bars (+ raw annotations) as one dataset; returns the provenance manifest."""
        return self.write_dataset(manifest, bars=bars, bar_extras=extras, created_at=created_at)

    def write_quotes(self, quotes: Sequence[Quote], *, manifest: DatasetSpec | Mapping[str, Any],
                     extras: RawExtras | None = None, created_at: datetime | None = None) -> dict[str, Any]:
        return self.write_dataset(manifest, quotes=quotes, quote_extras=extras, created_at=created_at)

    def write_trades(self, trades: Sequence[Trade], *, manifest: DatasetSpec | Mapping[str, Any],
                     extras: RawExtras | None = None, created_at: datetime | None = None) -> dict[str, Any]:
        return self.write_dataset(manifest, trades=trades, trade_extras=extras, created_at=created_at)

    def write_dataset(
        self,
        spec_in: DatasetSpec | Mapping[str, Any],
        *,
        bars: Sequence[Bar] = (),
        quotes: Sequence[Quote] = (),
        corporate_actions: Sequence[CorporateAction] = (),
        trades: Sequence[Trade] = (),
        bar_extras: RawExtras | None = None,
        quote_extras: RawExtras | None = None,
        trade_extras: RawExtras | None = None,
        bar_table: pa.Table | None = None,
        quote_table: pa.Table | None = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist one dataset atomically enough to be auditable; idempotent for identical content.

        ``bar_table``/``quote_table`` are a vectorized alternative to model lists (schema ``BAR_SCHEMA`` /
        ``QUOTE_SCHEMA``) used by the synthetic generator; the caller is responsible for having validated them.
        """
        spec = DatasetSpec.coerce(spec_in)
        btab = bar_table if bar_table is not None else (bars_to_table(bars) if bars else None)
        qtab = quote_table if quote_table is not None else (quotes_to_table(quotes) if quotes else None)
        ctab = corporate_actions_to_table(corporate_actions) if corporate_actions else None
        ttab = trades_to_table(trades) if trades else None
        if btab is not None:
            btab = btab.select(list(BAR_SCHEMA.names)).cast(BAR_SCHEMA)
        if qtab is not None:
            qtab = qtab.select(list(QUOTE_SCHEMA.names)).cast(QUOTE_SCHEMA)
        for name, tab in (("bars", btab), ("quotes", qtab), ("corporate_actions", ctab), ("trades", ttab)):
            if tab is not None and tab.num_rows:
                feeds = set(tab.column("feed").to_pylist())
                if feeds != {spec.feed.value}:
                    raise StoreError(f"{name}: feed(s) {sorted(feeds)} do not match dataset feed {spec.feed.value}")
                if name != "corporate_actions":
                    synth = set(tab.column("is_synthetic").to_pylist())
                    if synth != {spec.is_synthetic}:
                        raise StoreError(f"{name}: is_synthetic does not match the dataset spec")
        if btab is not None and btab.num_rows:
            self._check_unique(btab, ("symbol", "timeframe", "start_ts"), "bars")
        bar_rej = list(bar_extras.rejected) if bar_extras else []
        quote_rej = list(quote_extras.rejected) if quote_extras else []
        trade_rej = list(trade_extras.rejected) if trade_extras else []
        if not any(t is not None and t.num_rows for t in (btab, qtab, ctab, ttab)) and not (bar_rej or quote_rej or trade_rej):
            raise StoreError("refusing to write an empty dataset")

        created = to_utc(created_at) if created_at is not None else datetime.now(UTC)
        files: list[dict[str, Any]] = []
        if btab is not None or bar_rej:
            files += self._write_raw("bars", spec.feed, btab, bar_extras, RAW_BAR_SCHEMA, "start_ts")
        if qtab is not None or quote_rej:
            files += self._write_raw("quotes", spec.feed, qtab, quote_extras, RAW_QUOTE_SCHEMA, "exchange_ts")
        if ttab is not None or trade_rej:
            files += self._write_raw("trades", spec.feed, ttab, trade_extras, RAW_TRADE_SCHEMA, "exchange_ts")
        if ctab is not None:
            files += self._write_raw_ca(spec.feed, ctab)

        symbols: set[str] = set()
        timeframes: set[str] = set()
        starts: list[datetime] = []
        ends: list[datetime] = []
        counts: dict[str, int] = {}
        by_symbol: dict[str, dict[str, int]] = defaultdict(dict)
        if btab is not None and btab.num_rows:
            symbols |= set(btab.column("symbol").to_pylist())
            timeframes |= set(btab.column("timeframe").to_pylist())
            starts.append(_req_aware(pc.min(btab.column("start_ts")).as_py()))  # pyright: ignore[reportAttributeAccessIssue]
            ends.append(_req_aware(pc.max(btab.column("end_ts")).as_py()))  # pyright: ignore[reportAttributeAccessIssue]
            counts["bars"] = btab.num_rows
            for tf in sorted(timeframes):
                counts[f"bars_{tf}"] = int(pc.sum(pc.equal(btab.column("timeframe"), tf)).as_py() or 0)  # pyright: ignore[reportAttributeAccessIssue]
            for s, n in _value_counts(btab.column("symbol")).items():
                by_symbol[s]["bars"] = n
        if qtab is not None and qtab.num_rows:
            symbols |= set(qtab.column("symbol").to_pylist())
            starts.append(_req_aware(pc.min(qtab.column("exchange_ts")).as_py()))  # pyright: ignore[reportAttributeAccessIssue]
            ends.append(_req_aware(pc.max(qtab.column("exchange_ts")).as_py()))  # pyright: ignore[reportAttributeAccessIssue]
            counts["quotes"] = qtab.num_rows
            for s, n in _value_counts(qtab.column("symbol")).items():
                by_symbol[s]["quotes"] = n
        if ttab is not None and ttab.num_rows:
            symbols |= set(ttab.column("symbol").to_pylist())
            starts.append(_req_aware(pc.min(ttab.column("exchange_ts")).as_py()))  # pyright: ignore[reportAttributeAccessIssue]
            ends.append(_req_aware(pc.max(ttab.column("exchange_ts")).as_py()))  # pyright: ignore[reportAttributeAccessIssue]
            counts["trades"] = ttab.num_rows
            for s, n in _value_counts(ttab.column("symbol")).items():
                by_symbol[s]["trades"] = n
        if ctab is not None and ctab.num_rows:
            symbols |= set(ctab.column("symbol").to_pylist())
            counts["corporate_actions"] = ctab.num_rows
            for s, n in _value_counts(ctab.column("symbol")).items():
                by_symbol[s]["corporate_actions"] = n
        if bar_rej:
            counts["bars_rejected"] = len(bar_rej)
        if quote_rej:
            counts["quotes_rejected"] = len(quote_rej)
        if trade_rej:
            counts["trades_rejected"] = len(trade_rej)

        manifest = build_manifest(
            spec,
            symbols=sorted(symbols),
            timeframes=sorted(timeframes),
            start=iso(min(starts)) if starts else None,
            end=iso(max(ends)) if ends else None,
            row_counts=counts,
            row_counts_by_symbol=by_symbol,
            files=files,
            created_at=iso(created) or "",
        )
        version = str(manifest["dataset_version"])
        manifest["manifest_path"] = f"processed/manifests/{version}.json"

        with self._con() as con:
            exists = con.execute("SELECT manifest_json FROM datasets WHERE dataset_version = ?", [version]).fetchone()
            if exists is None:
                seq_row = con.execute("SELECT coalesce(max(ingest_seq), 0) + 1 FROM datasets").fetchone()
                seq = int(seq_row[0]) if seq_row else 1
                con.execute("BEGIN TRANSACTION")
                try:
                    con.execute(
                        "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [version, spec.feed.value, spec.kind, spec.source_id, spec.is_synthetic, spec.defects_flag,
                         _naive(created), seq, json.dumps(manifest, sort_keys=True)],
                    )
                    for table_name, tab in (("bars", btab), ("quotes", qtab), ("corporate_actions", ctab),
                                            ("trades", ttab)):
                        if tab is None or not tab.num_rows:
                            continue
                        con.register("incoming_rows", _to_naive_table(tab))
                        try:
                            cols = ", ".join(tab.schema.names)
                            con.execute(
                                f"INSERT INTO {table_name} (dataset_version, ingest_seq, {cols}) "  # noqa: S608
                                f"SELECT ?, ?, {cols} FROM incoming_rows",
                                [version, seq],
                            )
                        finally:
                            con.unregister("incoming_rows")
                    if spec.series_mode == SERIES_SNAPSHOT and not spec.defects_flag:
                        self._activate(con, version, spec.feed, sorted(symbols), created)
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
            else:
                manifest = json.loads(exists[0])
                if spec.series_mode == SERIES_SNAPSHOT and not spec.defects_flag:
                    self._activate(con, version, spec.feed, sorted(symbols), created)
        write_manifest_file(self.manifests_dir, manifest)
        self._bump_generation()
        return manifest

    @staticmethod
    def _activate(con: duckdb.DuckDBPyConnection, version: str, feed: DataFeed, symbols: Sequence[str],
                  when: datetime) -> None:
        for sym in symbols:
            con.execute("INSERT OR REPLACE INTO active_series VALUES (?, ?, ?, ?)", [feed.value, sym, version, _naive(when)])

    def activate_dataset(self, dataset_version: str) -> list[str]:
        """Make a stored snapshot-feed dataset the active series for each of its symbols; returns the symbols."""
        manifest = self.get_manifest(dataset_version)
        if manifest is None:
            raise StoreError(f"unknown dataset_version {dataset_version}")
        feed = DataFeed(manifest["feed"])
        if manifest.get("series_mode") != SERIES_SNAPSHOT:
            raise StoreError(f"{dataset_version} is an incremental dataset; only snapshot datasets can be activated")
        if manifest.get("defects_injected"):
            raise StoreError("a dataset with injected defects can never become an active series")
        symbols = [str(x) for x in manifest.get("symbols", [])]
        with self._con() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                self._activate(con, dataset_version, feed, symbols, datetime.now(UTC))
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        self._bump_generation()
        return symbols

    def active_series(self, feed: DataFeed | None = None) -> dict[str, dict[str, str]]:
        """{feed: {symbol: dataset_version}} of the active snapshot-feed series."""
        sql = "SELECT feed, symbol, dataset_version FROM active_series"
        params: list[Any] = []
        if feed is not None:
            sql += " WHERE feed = ?"
            params.append(DataFeed(feed).value)
        with self._con() as con:
            rows = con.execute(sql + " ORDER BY feed, symbol", params).fetchall()
        out: dict[str, dict[str, str]] = defaultdict(dict)
        for f, sym, dv in rows:
            out[str(f)][str(sym)] = str(dv)
        return dict(out)

    @staticmethod
    def _check_unique(tab: pa.Table, keys: tuple[str, ...], name: str) -> None:
        grouped = tab.group_by(list(keys)).aggregate([([], "count_all")])
        dup = grouped.filter(pc.greater(grouped.column("count_all"), 1))  # pyright: ignore[reportAttributeAccessIssue]
        if dup.num_rows:
            sample = dup.slice(0, 3).to_pylist()
            raise StoreError(f"{name}: {dup.num_rows} duplicate key(s) in one batch, e.g. {sample}; validate first")

    def _partition_path(self, feed: DataFeed, kind: str, symbol: str, partition: str | None) -> Path:
        base = self.raw_dir / feed.value / kind / f"symbol={symbol}"
        return (base / partition / "part.parquet") if partition is not None else (base / "part.parquet")

    def _write_raw(self, kind: str, feed: DataFeed, tab: pa.Table | None, extras: RawExtras | None,
                   schema: pa.Schema, ts_col: str) -> list[dict[str, Any]]:
        parts: list[pa.Table] = []
        if tab is not None and tab.num_rows:
            n = tab.num_rows
            flags = extras.flags if extras else {}
            raw_json = extras.raw_json if extras else {}
            status = [QUALITY_ACCEPTED] * n
            reasons: list[str | None] = [None] * n
            for i, fl in flags.items():
                if fl:
                    status[i] = QUALITY_FLAGGED
                    reasons[i] = ",".join(fl)
            payload: list[str | None] = [None] * n
            for i, js in raw_json.items():
                payload[i] = js
            parts.append(tab.append_column("quality_status", pa.array(status, pa.string()))
                         .append_column("quality_reasons", pa.array(reasons, pa.string()))
                         .append_column("raw_json", pa.array(payload, pa.string())).cast(schema))
        rej_rows: list[dict[str, Any]] = []
        for record, why in (extras.rejected if extras else ()):
            row = {name: _coerce_raw_value(name, schema.field(name).type, record.get(name))
                   for name in schema.names if name not in ("quality_status", "quality_reasons", "raw_json")}
            row["symbol"] = _safe_symbol(record.get("symbol"))
            row["feed"] = feed.value
            row["quality_status"] = QUALITY_REJECTED
            row["quality_reasons"] = ",".join(why)
            raw = record.get("raw_json")
            row["raw_json"] = raw if isinstance(raw, str) else _raw_record_json(record)
            rej_rows.append(row)
        if rej_rows:
            parts.append(pa.Table.from_pylist(rej_rows, schema=schema))
        if not parts:
            return []
        table = pa.concat_tables(parts)
        local = table.column(ts_col).cast(pa.timestamp("us", tz="America/New_York"))
        day = pc.fill_null(pc.strftime(local, format="date=%Y-%m-%d"), "date=unknown")  # pyright: ignore[reportAttributeAccessIssue]
        out_kind = pa.array([kind] * table.num_rows, pa.string())
        if kind == "bars":
            # Daily (and longer) bars are partitioned by year in a sibling ``bars_daily`` kind: one file per
            # (symbol, date) for decades of daily history would mean tens of thousands of tiny files.
            daily_mask = pc.is_in(table.column("timeframe"), value_set=pa.array(_DAILY_TIMEFRAMES))  # pyright: ignore[reportAttributeAccessIssue]
            daily_mask = pc.fill_null(daily_mask, False)  # pyright: ignore[reportAttributeAccessIssue]
            year = pc.fill_null(pc.strftime(local, format="year=%Y"), "year=unknown")  # pyright: ignore[reportAttributeAccessIssue]
            day = pc.if_else(daily_mask, year, day)  # pyright: ignore[reportAttributeAccessIssue]
            out_kind = pc.if_else(daily_mask, pa.scalar("bars_daily"), out_kind)  # pyright: ignore[reportAttributeAccessIssue]
        keyed = table.append_column("_pkind", out_kind).append_column("_pday", day)
        keyed = keyed.sort_by([("_pkind", "ascending"), ("symbol", "ascending"), ("_pday", "ascending")])
        kinds = keyed.column("_pkind").to_pylist()
        syms = keyed.column("symbol").to_pylist()
        days = keyed.column("_pday").to_pylist()
        bounds = [0] + [i for i in range(1, len(syms))
                        if syms[i] != syms[i - 1] or days[i] != days[i - 1] or kinds[i] != kinds[i - 1]] + [len(syms)]
        files: list[dict[str, Any]] = []
        sort_cols = (ts_col, "source_id", "received_ts", "available_at_ts", "quality_status")
        for a, b in itertools.pairwise(bounds):
            part = keyed.slice(a, b - a).drop_columns(["_pkind", "_pday"])
            path = self._partition_path(feed, str(kinds[a]), str(syms[a]), str(days[a]))
            files.append(self._merge_write(path, part, schema, sort_cols))
        return files

    def _write_raw_ca(self, feed: DataFeed, tab: pa.Table) -> list[dict[str, Any]]:
        keyed = tab.sort_by([("symbol", "ascending")])
        syms = keyed.column("symbol").to_pylist()
        bounds = [0] + [i for i in range(1, len(syms)) if syms[i] != syms[i - 1]] + [len(syms)]
        return [
            self._merge_write(self._partition_path(feed, "corporate_actions", str(syms[a]), None),
                              keyed.slice(a, b - a), CA_SCHEMA, ("ex_date", "action_type", "source_id"))
            for a, b in itertools.pairwise(bounds)
        ]

    def _merge_write(self, path: Path, part: pa.Table, schema: pa.Schema, sort_cols: tuple[str, ...]) -> dict[str, Any]:
        """Merge ``part`` into the partition file: exact-duplicate rows collapse, order is fully deterministic."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = part.cast(schema)
        quarantined: str | None = None
        existing: pa.Table | None = None
        if path.exists():
            try:
                existing = pq.read_table(path).cast(schema)
            except (OSError, pa.ArrowException) as exc:
                # Unreadable partition (corruption/truncation): never overwrite evidence silently. Move it aside, write
                # the new rows, and report it in the manifest; verify_dataset of earlier datasets then fails loudly.
                dest = path.with_name(f"part.parquet.corrupt-{uuid.uuid4().hex[:12]}")
                os.replace(path, dest)
                quarantined = dest.relative_to(self.datasets_dir).as_posix()
                log.error("quarantined unreadable raw partition %s -> %s: %s", path, dest, exc)
        if existing is not None:
            combined = pa.concat_tables([existing, table])
            con = duckdb.connect()
            try:
                con.execute("SET TimeZone='UTC'")
                con.register("combined_rows", combined)
                table = con.sql("SELECT DISTINCT * FROM combined_rows").to_arrow_table().cast(schema)
            finally:
                con.close()
        order = [(c, "ascending") for c in sort_cols] + [
            (c, "ascending") for c in schema.names if c not in sort_cols]
        table = table.sort_by(order)
        own = part.cast(schema).sort_by(order)
        tmp = path.with_name(f".part.{uuid.uuid4().hex}.tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
        entry: dict[str, Any] = {"path": path.relative_to(self.datasets_dir).as_posix(), "sha256": file_sha256(path),
                                 "rows": table.num_rows, "content_sha256": _table_sha256(own),
                                 "content_rows": own.num_rows}
        if quarantined is not None:
            entry["quarantined_previous"] = quarantined
        return entry

    # ---------------------------------------------------------------- reads: bars
    def _bars_sql(self, symbols: Sequence[str], start: datetime | None, end: datetime | None, timeframe: str,
                  feed: DataFeed | None, asof: datetime | None, dataset_version: str | None,
                  sessions: Sequence[MarketSession] | None) -> tuple[str, list[Any]]:
        if not symbols:
            raise ValueError("at least one symbol is required")
        where = [f"b.symbol IN ({', '.join('?' for _ in symbols)})", "b.timeframe = ?"]
        params: list[Any] = [*symbols, canonical_timeframe(timeframe)]
        if start is not None:
            where.append("b.start_ts >= ?")
            params.append(_naive(start))
        if end is not None:
            where.append("b.start_ts < ?")
            params.append(_naive(end))
        if feed is not None:
            where.append("b.feed = ?")
            params.append(DataFeed(feed).value)
        if asof is not None:
            where.append("b.available_at_ts <= ?")
            params.append(_naive(asof))
        if dataset_version is not None:
            clause, extra = self._version_scope("b", dataset_version)
            where.extend(clause)
            params.extend(extra)
        else:
            where.append("NOT d.defects_injected")
            where.append("(a.dataset_version IS NULL OR b.dataset_version = a.dataset_version)")
        if sessions:
            where.append(f"b.session IN ({', '.join('?' for _ in sessions)})")
            params.extend(MarketSession(s).value for s in sessions)
        sql = f"""
            WITH cand AS (
                SELECT b.*, {_feed_priority_sql('b.feed')} AS prio
                FROM bars b JOIN datasets d ON d.dataset_version = b.dataset_version
                LEFT JOIN active_series a ON a.feed = b.feed AND a.symbol = b.symbol
                WHERE {' AND '.join(where)}
            ), pick AS (
                SELECT *, min(prio) OVER (PARTITION BY symbol) AS best FROM cand
            ), dedup AS (
                SELECT *, row_number() OVER (PARTITION BY symbol, timeframe, start_ts
                                             ORDER BY available_at_ts DESC, ingest_seq DESC) AS rn
                FROM pick WHERE prio = best
            )
            SELECT symbol, timeframe, start_ts, end_ts, open, high, low, close, volume, vwap, trade_count, feed,
                   adjustment, session, provider_ts, received_ts, available_at_ts, source_id, dataset_version,
                   is_synthetic
            FROM dedup WHERE rn = 1 ORDER BY symbol, start_ts
        """  # noqa: S608 - only placeholders and constant fragments are interpolated
        return sql, params

    def _version_scope(self, alias: str, dataset_version: str) -> tuple[list[str], list[Any]]:
        """SQL filter for one stored dataset, or for a research snapshot: its constituents up to its cutoff."""
        from .research_snapshots import is_snapshot_id, resolve_scope

        if is_snapshot_id(dataset_version):
            versions, cutoff = resolve_scope(self, dataset_version)
            return ([f"{alias}.dataset_version IN ({', '.join('?' for _ in versions)})",
                     f"{alias}.available_at_ts <= ?"], [*versions, _naive(cutoff)])
        return [f"{alias}.dataset_version = ?"], [dataset_version]

    def load_bars(
        self,
        symbols: Sequence[str],
        start: datetime | None,
        end: datetime | None,
        timeframe: str,
        feed: DataFeed | None = None,
        asof: datetime | None = None,
        *,
        dataset_version: str | None = None,
        sessions: Sequence[MarketSession] | None = None,
    ) -> list[Bar]:
        """Bars with ``start <= start_ts < end`` (None = unbounded), PIT-filtered by ``asof``, sorted (symbol, start)."""
        sql, params = self._bars_sql(symbols, start, end, timeframe, feed, asof, dataset_version, sessions)
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [_bar_from_row(dict(zip(names, r, strict=True))) for r in rows]

    def load_bars_df(
        self,
        symbols: Sequence[str],
        start: datetime | None,
        end: datetime | None,
        timeframe: str,
        feed: DataFeed | None = None,
        asof: datetime | None = None,
        *,
        dataset_version: str | None = None,
        sessions: Sequence[MarketSession] | None = None,
        prices: Literal["float", "decimal"] = "float",
    ) -> pd.DataFrame:
        """Research convenience frame with columns ``BAR_DF_COLUMNS`` and the same PIT semantics as ``load_bars``.

        FLOAT CAVEAT: with ``prices="float"`` (default) ``open/high/low/close/volume/vwap`` are float64 converted
        from DECIMAL(18,6)/(24,6). This frame is for features/statistics only; accounting must use ``load_bars`` or
        ``prices="decimal"`` (object columns of exact ``Decimal``). Timestamps are tz-aware UTC.
        """
        sql, params = self._bars_sql(symbols, start, end, timeframe, feed, asof, dataset_version, sessions)
        price_cols = ("open", "high", "low", "close", "volume", "vwap")
        with self._con() as con:
            if prices == "decimal":
                cur = con.execute(sql, params)
                names = [d[0] for d in cur.description]
                df = pd.DataFrame.from_records(cur.fetchall(), columns=names)
            else:
                df = con.execute(sql, params).df()
        df = df.drop(columns=["adjustment", "provider_ts", "received_ts"])
        for col in ("start_ts", "end_ts", "available_at_ts"):
            df[col] = pd.to_datetime(df[col]).dt.tz_localize("UTC")
        for col in price_cols:
            if prices == "decimal":
                df[col] = pd.Series([_clean(v) for v in df[col]], index=df.index, dtype=object)
            else:
                df[col] = df[col].astype("float64")
        df["trade_count"] = df["trade_count"].astype("Int64")
        df["is_synthetic"] = df["is_synthetic"].astype(bool)
        return df.loc[:, list(BAR_DF_COLUMNS)].reset_index(drop=True)

    def latest_bar(self, symbol: str, timeframe: str, *, feed: DataFeed | None = None,
                   asof: datetime | None = None) -> Bar | None:
        sql, params = self._bars_sql([symbol], None, None, timeframe, feed, asof, None, None)
        sql = sql.replace("ORDER BY symbol, start_ts", "ORDER BY start_ts DESC LIMIT 1")
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            row = cur.fetchone()
        return _bar_from_row(dict(zip(names, row, strict=True))) if row else None

    # ---------------------------------------------------------------- reads: quotes
    def _quotes_sql(self, symbols: Sequence[str], start: datetime | None, end: datetime | None,
                    feed: DataFeed | None, asof: datetime | None, dataset_version: str | None) -> tuple[str, list[Any]]:
        if not symbols:
            raise ValueError("at least one symbol is required")
        where = [f"q.symbol IN ({', '.join('?' for _ in symbols)})"]
        params: list[Any] = list(symbols)
        if start is not None:
            where.append("q.exchange_ts >= ?")
            params.append(_naive(start))
        if end is not None:
            where.append("q.exchange_ts < ?")
            params.append(_naive(end))
        if feed is not None:
            where.append("q.feed = ?")
            params.append(DataFeed(feed).value)
        if asof is not None:
            where.append("q.available_at_ts <= ?")
            params.append(_naive(asof))
        if dataset_version is not None:
            clause, extra = self._version_scope("q", dataset_version)
            where.extend(clause)
            params.extend(extra)
        else:
            where.append("NOT d.defects_injected")
            where.append("(a.dataset_version IS NULL OR q.dataset_version = a.dataset_version)")
        sql = f"""
            WITH cand AS (
                SELECT q.*, {_feed_priority_sql('q.feed')} AS prio
                FROM quotes q JOIN datasets d ON d.dataset_version = q.dataset_version
                LEFT JOIN active_series a ON a.feed = q.feed AND a.symbol = q.symbol
                WHERE {' AND '.join(where)}
            ), pick AS (
                SELECT *, min(prio) OVER (PARTITION BY symbol) AS best FROM cand
            ), dedup AS (
                SELECT *, row_number() OVER (PARTITION BY symbol, exchange_ts, bid, ask, bid_size, ask_size, received_ts
                                             ORDER BY ingest_seq DESC) AS rn
                FROM pick WHERE prio = best
            )
            SELECT symbol, exchange_ts, bid, ask, bid_size, ask_size, bid_exchange, ask_exchange, conditions, tape,
                   provider_ts, received_ts, available_at_ts, feed, source_id, is_synthetic
            FROM dedup WHERE rn = 1 ORDER BY symbol, available_at_ts, exchange_ts
        """  # noqa: S608
        return sql, params

    def load_quotes(
        self,
        symbols: Sequence[str],
        start: datetime | None,
        end: datetime | None,
        feed: DataFeed | None = None,
        asof: datetime | None = None,
        *,
        dataset_version: str | None = None,
    ) -> list[Quote]:
        """Quotes with ``start <= exchange_ts < end``, PIT-filtered by ``asof``; sorted (symbol, available_at)."""
        sql, params = self._quotes_sql(symbols, start, end, feed, asof, dataset_version)
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [_quote_from_row(dict(zip(names, r, strict=True))) for r in rows]

    def latest_quote(self, symbol: str, asof: datetime, *, feed: DataFeed | None = None) -> Quote | None:
        """Latest quote with ``available_at_ts <= asof`` (single feed, highest priority with data)."""
        sql, params = self._quotes_sql([symbol], None, None, feed, asof, None)
        sql = sql.replace("ORDER BY symbol, available_at_ts, exchange_ts",
                          "ORDER BY available_at_ts DESC, exchange_ts DESC LIMIT 1")
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            row = cur.fetchone()
        return _quote_from_row(dict(zip(names, row, strict=True))) if row else None

    def quote_columns(self, symbol: str, feed: DataFeed, *, dataset_version: str | None = None) -> QuoteColumns:
        """All stored quotes of one symbol/feed in columnar form, sorted by availability (replay cache)."""
        import numpy as np

        sql, params = self._quotes_sql([symbol], None, None, feed, None, dataset_version)
        with self._con() as con:
            fetched = con.sql(sql, params=params).to_arrow_table()
        if fetched is None:
            raise StoreError("DuckDB returned no result for the quote query")
        table: pa.Table = fetched
        table = table.cast(pa.schema([pa.field(f.name, TS_UTC) if pa.types.is_timestamp(f.type) else f
                                      for f in table.schema]))
        avail = table.column("available_at_ts").cast(pa.int64()).to_numpy() if table.num_rows else np.array([], np.int64)
        exch = table.column("exchange_ts").cast(pa.int64()).to_numpy() if table.num_rows else np.array([], np.int64)
        return QuoteColumns(symbol=symbol, feed=DataFeed(feed), table=table, available_at_us=avail, exchange_us=exch)

    # ---------------------------------------------------------------- reads: trades
    def _trades_sql(self, symbols: Sequence[str], start: datetime | None, end: datetime | None,
                    feed: DataFeed | None, asof: datetime | None) -> tuple[str, list[Any]]:
        if not symbols:
            raise ValueError("at least one symbol is required")
        where = [f"t.symbol IN ({', '.join('?' for _ in symbols)})", "NOT d.defects_injected"]
        params: list[Any] = list(symbols)
        for cond, value in (("t.exchange_ts >= ?", start), ("t.exchange_ts < ?", end), ("t.available_at_ts <= ?", asof)):
            if value is not None:
                where.append(cond)
                params.append(_naive(value))
        if feed is not None:
            where.append("t.feed = ?")
            params.append(DataFeed(feed).value)
        sql = f"""
            WITH cand AS (
                SELECT t.*, {_feed_priority_sql('t.feed')} AS prio
                FROM trades t JOIN datasets d ON d.dataset_version = t.dataset_version WHERE {' AND '.join(where)}
            ), pick AS (SELECT *, min(prio) OVER (PARTITION BY symbol) AS best FROM cand),
            dedup AS (
                SELECT *, row_number() OVER (PARTITION BY symbol, exchange_ts, price, size, coalesce(trade_id, '')
                                             ORDER BY ingest_seq DESC) AS rn
                FROM pick WHERE prio = best
            )
            SELECT symbol, exchange_ts, price, size, exchange, trade_id, conditions, tape, provider_ts, received_ts,
                   available_at_ts, feed, source_id, is_synthetic
            FROM dedup WHERE rn = 1 ORDER BY symbol, available_at_ts, exchange_ts
        """  # noqa: S608
        return sql, params

    def load_trades(self, symbols: Sequence[str], start: datetime | None, end: datetime | None,
                    feed: DataFeed | None = None, asof: datetime | None = None) -> list[Trade]:
        """Trades with ``start <= exchange_ts < end``, PIT-filtered by ``asof``; one feed per symbol."""
        sql, params = self._trades_sql(symbols, start, end, feed, asof)
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [_trade_from_row(dict(zip(names, r, strict=True))) for r in rows]

    def latest_trade(self, symbol: str, asof: datetime, *, feed: DataFeed | None = None) -> Trade | None:
        sql, params = self._trades_sql([symbol], None, None, feed, asof)
        sql = sql.replace("ORDER BY symbol, available_at_ts, exchange_ts",
                          "ORDER BY available_at_ts DESC, exchange_ts DESC LIMIT 1")
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            row = cur.fetchone()
        return _trade_from_row(dict(zip(names, row, strict=True))) if row else None

    # ---------------------------------------------------------------- reads: corporate actions
    def load_corporate_actions(self, symbols: Sequence[str] | None = None, *, asof: datetime | None = None,
                               feed: DataFeed | None = None, dataset_version: str | None = None) -> list[CorporateAction]:
        where: list[str] = ["1=1"]
        params: list[Any] = []
        if symbols:
            where.append(f"c.symbol IN ({', '.join('?' for _ in symbols)})")
            params.extend(symbols)
        if asof is not None:
            where.append("c.available_at_ts <= ?")
            params.append(_naive(asof))
        if feed is not None:
            where.append("c.feed = ?")
            params.append(DataFeed(feed).value)
        if dataset_version is not None:
            clause, extra = self._version_scope("c", dataset_version)
            where.extend(clause)
            params.extend(extra)
        sql = f"""
            WITH dedup AS (
                SELECT c.*, row_number() OVER (PARTITION BY symbol, action_type, ex_date, feed
                                               ORDER BY ingest_seq DESC) AS rn
                FROM corporate_actions c WHERE {' AND '.join(where)}
            )
            SELECT symbol, action_type, ex_date, available_at_ts, split_factor, cash_amount, reference_price,
                   old_symbol, new_symbol, feed, source_id, raw
            FROM dedup WHERE rn = 1 ORDER BY symbol, ex_date, action_type
        """  # noqa: S608
        with self._con() as con:
            cur = con.execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [_ca_from_row(dict(zip(names, r, strict=True))) for r in rows]

    # ---------------------------------------------------------------- manifests / datasets
    def list_manifests(self, feed: DataFeed | None = None, *, include_defective: bool = True) -> list[dict[str, Any]]:
        """All registered datasets, oldest first (registry in DuckDB is authoritative)."""
        sql = "SELECT manifest_json FROM datasets WHERE 1=1"
        params: list[Any] = []
        if feed is not None:
            sql += " AND feed = ?"
            params.append(DataFeed(feed).value)
        if not include_defective:
            sql += " AND NOT defects_injected"
        sql += " ORDER BY ingest_seq"
        with self._con() as con:
            rows = con.execute(sql, params).fetchall()
        return [json.loads(r[0]) for r in rows]

    def list_research_snapshots(self, feed: DataFeed | None = None) -> list[dict[str, Any]]:
        """Immutable cumulative research snapshots (``research_snapshots``), oldest first."""
        from .research_snapshots import list_snapshots

        return list_snapshots(self, feed=feed)

    def latest_manifest(self, feed: DataFeed | None = None) -> dict[str, Any] | None:
        """Most recently written non-defect-injected dataset manifest (optionally for one feed)."""
        manifests = self.list_manifests(feed, include_defective=False)
        return manifests[-1] if manifests else None

    def get_manifest(self, dataset_version: str) -> dict[str, Any] | None:
        from .research_snapshots import get_snapshot, is_snapshot_id

        if is_snapshot_id(dataset_version):
            return get_snapshot(self, dataset_version)
        with self._con() as con:
            row = con.execute("SELECT manifest_json FROM datasets WHERE dataset_version = ?", [dataset_version]).fetchone()
        if row:
            return json.loads(row[0])
        path = self.manifests_dir / f"{dataset_version}.json"
        return read_manifest_file(path) if path.is_file() else None

    def dataset_row_counts(self, dataset_version: str) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._con() as con:
            for table in ("bars", "quotes", "trades", "corporate_actions"):
                row = con.execute(f"SELECT count(*) FROM {table} WHERE dataset_version = ?", [dataset_version]).fetchone()  # noqa: S608
                out[table] = int(row[0]) if row else 0
        return out

    def verify_dataset(self, dataset_version: str) -> dict[str, Any]:
        """Integrity check: manifest hash, DuckDB row counts, Parquet sha256 (or superseded by a later dataset)."""
        manifest = self.get_manifest(dataset_version)
        if manifest is None:
            return {"dataset_version": dataset_version, "ok": False, "errors": ["unknown dataset_version"], "warnings": []}
        errors: list[str] = []
        warnings: list[str] = []
        if not manifest_content_ok(manifest):
            errors.append("manifest content does not hash to its dataset_version (modified?)")
        file_path = self.manifests_dir / f"{dataset_version}.json"
        if not file_path.is_file():
            warnings.append("manifest JSON file missing on disk (registry copy used)")
        else:
            try:
                on_disk = read_manifest_file(file_path)
                if canonical_json(on_disk) != canonical_json(manifest):
                    errors.append("manifest JSON on disk differs from the DuckDB registry copy (modified?)")
            except (ValueError, OSError) as exc:
                errors.append(f"manifest JSON unreadable: {exc}")
        counts = self.dataset_row_counts(dataset_version)
        expected = manifest.get("row_counts", {})
        for table in ("bars", "quotes", "trades", "corporate_actions"):
            if int(expected.get(table, 0)) != counts[table]:
                errors.append(f"{table}: manifest says {expected.get(table, 0)} rows, DuckDB has {counts[table]}")
        later_paths: set[str] = set()
        found = False
        for m in self.list_manifests():
            if found:
                later_paths |= {str(f["path"]) for f in m.get("files", [])}
            if m.get("dataset_version") == dataset_version:
                found = True
        for f in manifest.get("files", []):
            p = self.datasets_dir / str(f["path"])
            if not p.is_file():
                errors.append(f"missing raw file {f['path']}")
                continue
            if file_sha256(p) != f["sha256"]:
                if str(f["path"]) in later_paths:
                    warnings.append(f"{f['path']} superseded (merged) by a later dataset; sha256 changed as expected")
                else:
                    errors.append(f"{f['path']} sha256 mismatch and no later dataset wrote it (tampered?)")
        return {"dataset_version": dataset_version, "ok": not errors, "errors": errors, "warnings": warnings,
                "row_counts": counts}

    def summary(self) -> dict[str, Any]:
        with self._con() as con:
            out: dict[str, Any] = {}
            for table in ("bars", "quotes", "trades", "corporate_actions", "datasets", "universe_snapshots"):
                row = con.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
                out[table] = int(row[0]) if row else 0
            feeds = con.execute(
                "SELECT feed, timeframe, count(*), min(start_ts), max(end_ts) FROM bars GROUP BY feed, timeframe "
                "ORDER BY feed, timeframe").fetchall()
        out["bars_by_feed"] = [
            {"feed": f, "timeframe": tf, "rows": int(n), "start": iso(_aware(a)), "end": iso(_aware(b))}
            for f, tf, n, a, b in feeds
        ]
        return out

    # ---------------------------------------------------------------- universe snapshots
    def write_universe_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        cols = ("snapshot_id", "asof_ts", "asof_date", "symbol", "eligible", "reasons", "last_close",
                "avg_dollar_volume", "sessions_observed", "feed", "dataset_versions", "policy_version", "created_at")
        with self._con() as con:
            con.executemany(
                f"INSERT OR REPLACE INTO universe_snapshots ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",  # noqa: S608
                [[_naive(r[c]) if isinstance(r[c], datetime) else r[c] for c in cols] for r in rows],
            )
        self._bump_generation()

    def load_universe_rows(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._con() as con:
            cur = con.execute("SELECT * FROM universe_snapshots WHERE snapshot_id = ? ORDER BY symbol", [snapshot_id])
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def _table_sha256(table: pa.Table) -> str:
    """Deterministic content hash of one dataset's rows for one partition (Parquet bytes of the sorted rows)."""
    import hashlib

    sink = pa.BufferOutputStream()
    pq.write_table(table.combine_chunks(), sink, compression="zstd")
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _value_counts(col: pa.ChunkedArray) -> dict[str, int]:
    vc = col.value_counts()
    return {str(item["values"]): int(item["counts"]) for item in vc.to_pylist()}
