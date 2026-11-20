"""Durable market-data ingestion with explicit connection states and backfill-before-entries (spec §2).

States: ``DISCONNECTED -> CONNECTING -> BACKFILLING -> LIVE`` (any failure -> ``DISCONNECTED``).

* On every (re)connect the service backfills, per symbol, from the end of the last *stored* bar (or
  ``now - backfill_lookback`` when nothing is stored) to ``now`` through the provider's historical bars, validates
  them (gaps are reported, never filled), stores them as one *incremental* dataset and records data-quality
  events. Only after every symbol's backfill call succeeded does the state become ``LIVE``. A backfill that returns
  too little data (more than ``max_gap_fraction`` of a symbol's expected regular-session bars missing) keeps the
  state ``BACKFILLING`` (continuity unknown) and is retried.
* ``entries_allowed()`` is ``True`` only when the state is ``LIVE``, the last backfill succeeded, the feed-quality
  monitor reports fresh quotes for every tracked symbol (``status == "OK"``) and — when ``block_on_unfilled_gaps``
  — no unfilled gap remains. Anything unknown fails closed.
* While LIVE, streamed quotes (and bars, when the provider can stream bars) are observed by the monitor and flushed
  to the store in batches. Providers exposing ``stream_events`` are consumed over ONE connection (Alpaca permits a
  single market-data websocket per account); streamed records the provider rejected are counted in the monitor and
  preserved verbatim in raw Parquet at the next flush. A streamed bar that does not continue the stored series (a
  hole after a silent reconnect inside the SDK) drops the state to ``BACKFILLING`` until the hole has been backfilled.
* Every state transition is written to ``md_feed_state_transitions`` and every backfill to ``md_ingestion_runs``
  (SQLite, migration 0100); provider failures also go to the core ``failures`` table.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from packages.contracts.enums import DataFeed, FeedMode
from packages.contracts.models import Bar, Quote
from packages.core.audit import record_failure
from packages.core.clock import Clock
from packages.core.db import Database, utc_iso
from packages.core.ids import new_id

from .calendar import CalendarOutOfRange, MarketCalendar, get_calendar
from .common import FEED_MODE_TO_FEED, PENDING_DATASET_VERSION, canonical_timeframe, iso, timeframe_delta, to_utc
from .manifest import DatasetSpec
from .quality import (
    FeedConnectionState,
    FeedQualityMonitor,
    Gap,
    QualityIssue,
    QualityPolicy,
    ValidationReport,
    compute_bar_gaps,
    only_available,
    record_quality_events,
    rejected_raw_json,
    validate_bars,
    validate_quotes,
    validate_trades,
    with_gaps,
)
from .store import MarketDataStore
from .trades import Trade

log = logging.getLogger("lat.market_data.ingestion")

GENERATOR = "services.market_data.ingestion"
GENERATOR_VERSION = "1.0.0"


class IngestionState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    BACKFILLING = "BACKFILLING"
    LIVE = "LIVE"


_TO_MONITOR: dict[IngestionState, FeedConnectionState] = {
    IngestionState.DISCONNECTED: FeedConnectionState.DISCONNECTED,
    IngestionState.CONNECTING: FeedConnectionState.CONNECTING,
    IngestionState.BACKFILLING: FeedConnectionState.BACKFILLING,
    IngestionState.LIVE: FeedConnectionState.LIVE,
}


class BackfillError(RuntimeError):
    """At least one symbol's historical backfill failed; entries stay locked."""


@dataclass
class BackfillResult:
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    windows: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    rows_accepted: int = 0
    rows_rejected: int = 0
    gaps: list[Gap] = field(default_factory=list)
    dataset_version: str | None = None
    errors: dict[str, str] = field(default_factory=dict)
    expected_bars: dict[str, int] = field(default_factory=dict)
    missing_bars: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "started_at": iso(self.started_at), "finished_at": iso(self.finished_at),
            "windows": {s: list(w) for s, w in self.windows.items()}, "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected, "gaps": [g.as_dict() for g in self.gaps],
            "dataset_version": self.dataset_version, "errors": dict(self.errors), "ok": self.ok,
            "expected_bars": dict(self.expected_bars), "missing_bars": dict(self.missing_bars),
        }


def _provider_feed(provider: Any) -> DataFeed:
    feed = getattr(provider, "feed", None)
    if isinstance(feed, DataFeed):
        return feed
    mode = FeedMode(provider.feed_mode)
    return FEED_MODE_TO_FEED[mode]


class IngestionService:
    def __init__(
        self,
        provider: Any,
        store: MarketDataStore,
        db: Database | None,
        clock: Clock,
        quality_monitor: FeedQualityMonitor,
        *,
        symbols: Sequence[str] | None = None,
        timeframe: str = "1Min",
        feed: DataFeed | None = None,
        backfill_lookback: timedelta = timedelta(days=3),
        policy: QualityPolicy | None = None,
        block_on_unfilled_gaps: bool = False,
        max_gap_fraction: float | None = 0.5,
        min_expected_for_fraction: int = 10,
        gap_grace: timedelta = timedelta(seconds=5),
        quote_flush_size: int = 2_000,
        flush_interval_s: float = 60.0,
        reconnect_backoff_s: tuple[float, float] = (1.0, 30.0),
        calendar: MarketCalendar | None = None,
    ) -> None:
        syms = list(symbols) if symbols is not None else list(quality_monitor.symbols)
        if not syms:
            raise ValueError("IngestionService needs at least one symbol")
        self.provider = provider
        self.store = store
        self.db = db
        self.clock = clock
        self.monitor = quality_monitor
        self.symbols = [s.upper() for s in syms]
        for s in self.symbols:  # tracked symbols without data keep the monitor non-OK (fail closed)
            self.monitor.track(s)
        self.timeframe = canonical_timeframe(timeframe)
        self.step = timeframe_delta(self.timeframe)
        self.feed = feed or _provider_feed(provider)
        self.backfill_lookback = backfill_lookback
        self.policy = policy or QualityPolicy()
        self.block_on_unfilled_gaps = block_on_unfilled_gaps
        self.max_gap_fraction = max_gap_fraction
        self.min_expected_for_fraction = min_expected_for_fraction
        self._thin_backfill: dict[str, str] = {}
        self.gap_grace = gap_grace
        self.quote_flush_size = quote_flush_size
        self.flush_interval_s = flush_interval_s
        self.reconnect_backoff_s = reconnect_backoff_s
        self.calendar = calendar or get_calendar()
        self._state = IngestionState.DISCONNECTED
        self._state_reason = "not connected"
        self.last_backfill: BackfillResult | None = None
        self._unfilled_gaps: list[Gap] = []
        self._quote_buf: list[Quote] = []
        self._bar_buf: list[Bar] = []
        self._last_stored_end: dict[str, datetime] = {}
        self._stream_rejected: dict[str, list[tuple[Mapping[str, Any], Sequence[str]]]] = {
            "quotes": [], "bars": [], "trades": []}
        self._trade_buf: list[Trade] = []
        self._stream_issues: list[QualityIssue] = []
        self._stream_rejected_count = 0
        self._lock = asyncio.Lock()
        self.monitor.set_connection_state(FeedConnectionState.DISCONNECTED, self._state_reason)
        if hasattr(provider, "on_stream_report"):
            provider.on_stream_report = self._on_stream_report

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> IngestionState:
        return self._state

    def _transition(self, to: IngestionState, reason: str) -> None:
        frm = self._state
        self._state = to
        self._state_reason = reason
        self.monitor.set_connection_state(_TO_MONITOR[to], reason)
        log.info("feed state %s -> %s: %s", frm.value, to.value, reason)
        if self.db is not None and frm != to:
            provider = str(getattr(self.provider, "source_id", type(self.provider).__name__))
            now = utc_iso(self.clock.now())
            self.db.write(lambda c: c.execute(
                "INSERT INTO md_feed_state_transitions(ts, provider, from_state, to_state, reason) VALUES (?,?,?,?,?)",
                (now, provider, frm.value, to.value, reason[:2000])))

    def entries_status(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self._state != IngestionState.LIVE:
            reasons.append(f"market data feed {self._state.value}: {self._state_reason}")
        if self.last_backfill is None:
            reasons.append("no successful backfill since connect")
        elif not self.last_backfill.ok:
            reasons.append(f"last backfill failed: {sorted(self.last_backfill.errors)}")
        st = self.monitor.status()
        if st.get("status") != "OK":
            reasons.append(f"feed quality {st.get('status')}: {st.get('detail')}")
        if self.block_on_unfilled_gaps and self._unfilled_gaps:
            reasons.append(f"{len(self._unfilled_gaps)} unfilled bar gap(s) after backfill")
        if self._thin_backfill:
            reasons.append(f"backfill returned too little data: {self._thin_backfill}")
        return (not reasons), reasons

    def entries_allowed(self) -> bool:
        return self.entries_status()[0]

    def status(self) -> dict[str, Any]:
        allowed, reasons = self.entries_status()
        return {"state": self._state.value, "reason": self._state_reason, "entries_allowed": allowed,
                "entry_block_reasons": reasons, "symbols": self.symbols, "timeframe": self.timeframe,
                "feed": self.feed.value, "unfilled_gaps": [g.as_dict() for g in self._unfilled_gaps],
                "last_backfill": self.last_backfill.as_dict() if self.last_backfill else None}

    # ------------------------------------------------------------------ connect / backfill
    async def connect(self) -> BackfillResult:
        """CONNECTING -> BACKFILLING -> LIVE. Raises :class:`BackfillError` (state DISCONNECTED) on failure."""
        self._transition(IngestionState.CONNECTING, "connecting to provider")
        return await self._backfill_then_live("initial/reconnect backfill")

    async def _backfill_then_live(self, why: str) -> BackfillResult:
        result = await self.backfill(reason=why)
        if not result.ok:
            self._transition(IngestionState.DISCONNECTED, f"backfill failed for {sorted(result.errors)}; entries locked")
            raise BackfillError(f"backfill failed: {result.errors}")
        if self._thin_backfill:
            # e.g. the provider answered with (almost) no bars for a window that should have them: continuity unknown
            self._transition(IngestionState.BACKFILLING, f"backfill too thin, entries stay locked: {self._thin_backfill}")
            return result
        if self.block_on_unfilled_gaps and self._unfilled_gaps:
            self._transition(IngestionState.BACKFILLING,
                             f"{len(self._unfilled_gaps)} unfilled gap(s) remain after backfill (block_on_unfilled_gaps)")
            return result
        self._transition(IngestionState.LIVE, f"backfill complete ({result.rows_accepted} bars, {len(result.gaps)} gaps)")
        return result

    def _last_end(self, symbol: str) -> datetime | None:
        cached = self._last_stored_end.get(symbol)
        last = self.store.latest_bar(symbol, self.timeframe, feed=self.feed)
        end = last.end_ts if last is not None else None
        if cached is not None and (end is None or cached > end):
            return cached
        return end

    async def _fetch(self, symbol: str, start: datetime, end: datetime) -> tuple[list[Any], dict[int, str]]:
        fetch_raw: Any = getattr(self.provider, "fetch_bar_records", None)
        if fetch_raw is not None:
            records, raw = await fetch_raw(symbol, start, end, self.timeframe)
            return list(records), dict(raw)
        bars = await self.provider.get_bars(symbol, start, end, self.timeframe)
        return list(bars), {}

    async def backfill(self, symbols: Sequence[str] | None = None, *, reason: str = "backfill") -> BackfillResult:
        """Fetch + validate + store missing bars up to now for ``symbols`` (default: all). Never fills gaps."""
        async with self._lock:
            self._transition(IngestionState.BACKFILLING, reason)
            now = to_utc(self.clock.now())
            result = BackfillResult(run_id=new_id("mdrun"), started_at=now)
            records: list[Any] = []
            raw_json: dict[int, str] = {}
            ranges: dict[str, tuple[datetime, datetime]] = {}
            for sym in symbols or self.symbols:
                last_end = await asyncio.to_thread(self._last_end, sym)
                start = last_end if last_end is not None else now - self.backfill_lookback
                if start >= now:
                    result.windows[sym] = (iso(start), iso(now))
                    continue
                result.windows[sym] = (iso(start), iso(now))
                try:
                    recs, raw = await self._fetch(sym, start, now)
                except Exception as exc:  # unknown continuity => fail closed
                    result.errors[sym] = f"{type(exc).__name__}: {exc}"
                    self._failure("backfill", type(exc).__name__, f"{sym}: {exc}")
                    continue
                base = len(records)
                records += recs
                raw_json.update({base + i: js for i, js in raw.items()})
                ranges[sym] = (start, now - self.gap_grace)
            report = self._validate_bars(records, ranges, now) if records or ranges else None
            if report is not None:
                result.rows_accepted = len(report.accepted)
                result.rows_rejected = len(report.rejected)
                result.gaps = list(report.gaps)
                if report.accepted or report.rejected:
                    manifest = await asyncio.to_thread(self._store_bars, report, raw_json, result.run_id, ranges)
                    result.dataset_version = str(manifest["dataset_version"])
                for b in report.accepted:
                    if b.timeframe == self.timeframe:
                        prev = self._last_stored_end.get(b.symbol)
                        if prev is None or b.end_ts > prev:
                            self._last_stored_end[b.symbol] = b.end_ts
                self.monitor.observe_report(report)
                if self.db is not None:
                    await asyncio.to_thread(record_quality_events, self.db, report, self.feed,
                                            context={"run_id": result.run_id, "kind": "backfill"})
            backfilled = set(symbols or self.symbols)
            self._unfilled_gaps = [g for g in self._unfilled_gaps if g.symbol not in backfilled] + list(result.gaps)
            for sym, (lo, hi) in ranges.items():
                result.expected_bars[sym] = self._expected_count(lo, hi)
                result.missing_bars[sym] = sum(g.missing for g in result.gaps if g.symbol == sym)
            thin = {s: v for s, v in self._thin_backfill.items() if s not in backfilled}
            if self.max_gap_fraction is not None:
                for sym, expected in result.expected_bars.items():
                    missing = result.missing_bars.get(sym, 0)
                    if expected >= self.min_expected_for_fraction and missing > self.max_gap_fraction * expected:
                        thin[sym] = f"{missing}/{expected} regular-session bars missing"
            self._thin_backfill = thin
            result.finished_at = to_utc(self.clock.now())
            self.last_backfill = result
            self._record_run(result, "backfill")
            return result

    def _expected_count(self, lo: datetime, hi: datetime) -> int:
        if hi <= lo:
            return 0
        try:
            days = self.calendar.sessions_in_range(self.calendar.ny_date(lo), self.calendar.ny_date(hi))
        except CalendarOutOfRange:
            return 0
        return sum(1 for d in days for t in self.calendar.regular_bar_starts(d, self.step) if lo <= t and t + self.step <= hi)

    def _validate_bars(self, records: Sequence[Any], ranges: dict[str, tuple[datetime, datetime]],
                       now: datetime) -> ValidationReport:
        # gaps are evaluated per symbol over its own backfill window (bars must be complete by now - grace)
        report = validate_bars(records, self.calendar, _no_gaps(self.policy), validated_at=now)
        # a still-forming bar (available_at_ts > now) is not knowable yet: drop it, the next backfill fetches it
        report = only_available(report, now)

        gaps: list[Gap] = []
        for sym, (lo, hi) in sorted(ranges.items()):
            if hi <= lo:
                continue
            sym_bars = [b for b in report.accepted if b.symbol == sym and b.timeframe == self.timeframe]
            gaps += compute_bar_gaps(sym_bars, self.calendar, expected_range=(lo, hi), expected_symbols=[sym])
        return with_gaps(report, gaps) if self.policy.report_gaps else report

    def _spec(self, kind: str, run_id: str, params: dict[str, Any]) -> DatasetSpec:
        return DatasetSpec(
            source_id=str(getattr(self.provider, "source_id", "unknown")),
            feed=self.feed,
            generator=GENERATOR,
            generator_version=GENERATOR_VERSION,
            is_synthetic=self.feed == DataFeed.SYNTHETIC,
            kind=kind,
            description=f"{kind} from {getattr(self.provider, 'source_id', 'provider')} ({run_id})",
            limitations=tuple(self.monitor.status().get("limitations", ())),
            params={"run_id": run_id, "timeframe": self.timeframe, **params},
        )

    def _store_bars(self, report: ValidationReport, raw_json: dict[int, str], run_id: str,
                    ranges: dict[str, tuple[datetime, datetime]]) -> dict[str, Any]:
        bars = [b.model_copy(update={"dataset_version": PENDING_DATASET_VERSION}) for b in report.accepted]
        raw = {**rejected_raw_json(report), **raw_json}
        spec = self._spec("ingestion_backfill_bars", run_id,
                          {"windows": {s: [iso(a), iso(b)] for s, (a, b) in sorted(ranges.items())}})
        return self.store.write_bars(bars, manifest=spec, extras=report.raw_extras(raw))

    # ------------------------------------------------------------------ live streaming
    async def run(self, stop: asyncio.Event) -> None:
        """Connect, backfill, stream; on any disconnect go DISCONNECTED, back off, reconnect (and backfill again)."""
        attempt = 0
        lo, hi = self.reconnect_backoff_s
        while not stop.is_set():
            try:
                await self.connect()
                if self._state == IngestionState.LIVE:
                    attempt = 0
                    await self._consume(stop)
                else:  # blocked on unfilled gaps: retry backfill after a pause
                    await _wait(stop, hi)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._transition(IngestionState.DISCONNECTED, f"{type(exc).__name__}: {exc}")
                self._failure("stream", type(exc).__name__, str(exc))
            finally:
                await self.flush()
            if stop.is_set():
                break
            attempt += 1
            await _wait(stop, min(hi, lo * (2 ** min(attempt - 1, 10))))
        if self._state != IngestionState.DISCONNECTED:
            self._transition(IngestionState.DISCONNECTED, "ingestion stopped")

    async def _consume(self, stop: asyncio.Event) -> None:
        if callable(getattr(self.provider, "stream_events", None)):
            # one multiplexed connection for quotes + bars (Alpaca allows a single market-data websocket)
            tasks = [asyncio.create_task(self._consume_events(), name="md-events")]
        else:
            tasks = [asyncio.create_task(self._consume_quotes(), name="md-quotes")]
            if callable(getattr(self.provider, "stream_bars", None)):
                tasks.append(asyncio.create_task(self._consume_bars(), name="md-bars"))
        stopper = asyncio.create_task(stop.wait(), name="md-stop")
        flusher = asyncio.create_task(self._periodic_flush(stop), name="md-flush")
        try:
            done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
            if stopper in done:
                return
            for t in done:
                exc = t.exception()
                if exc is not None:
                    raise exc
            from .alpaca_provider import StreamDisconnected

            raise StreamDisconnected("provider stream ended")
        finally:
            for t in [*tasks, stopper, flusher]:
                t.cancel()
            for t in [*tasks, stopper, flusher]:
                with contextlib.suppress(BaseException):
                    await t

    async def _consume_events(self) -> None:
        stream: AsyncIterator[Quote | Bar | Trade] = self.provider.stream_events(self.symbols)
        async for item in stream:
            if isinstance(item, Quote):
                await self.on_stream_quote(item)
            elif isinstance(item, Bar):
                await self.on_stream_bar(item)
            elif isinstance(item, Trade):
                await self.on_stream_trade(item)

    async def _consume_quotes(self) -> None:
        stream: AsyncIterator[Quote] = self.provider.stream_quotes(self.symbols)
        async for q in stream:
            await self.on_stream_quote(q)

    async def on_stream_quote(self, quote: Quote) -> None:
        self.monitor.observe_quote(quote)
        self._quote_buf.append(quote)
        if len(self._quote_buf) >= self.quote_flush_size:
            await self.flush_quotes()

    async def on_stream_trade(self, trade: Trade) -> None:
        self.monitor.observe_trade(trade)
        self._trade_buf.append(trade)
        if len(self._trade_buf) >= self.quote_flush_size:
            await self.flush_trades()

    def _on_stream_report(self, report: ValidationReport) -> None:
        """Provider hook: a streamed record had issues. Rejects are counted now and persisted at the next flush
        (raw record preserved in Parquet, issue rows in data_quality_events)."""
        for sym, n in report.rejected_by_symbol().items():
            self.monitor.record_rejects(sym, n)
        bucket = report.kind
        for r in report.rejected:
            self._stream_rejected_count += 1
            if len(self._stream_rejected[bucket]) < 10_000:  # bounded; the count stays exact
                self._stream_rejected[bucket].append((dict(r.record), r.reasons))
        if len(self._stream_issues) < 10_000:
            self._stream_issues.extend(report.issues)

    async def _consume_bars(self) -> None:
        stream: AsyncIterator[Bar] = self.provider.stream_bars(self.symbols)
        async for b in stream:
            await self.on_stream_bar(b)

    async def on_stream_bar(self, bar: Bar) -> None:
        """Continuity check for a streamed bar: a hole before it triggers a symbol backfill before LIVE resumes."""
        if bar.timeframe != self.timeframe:
            return
        last_end = self._last_stored_end.get(bar.symbol)
        if last_end is None:
            last_end = await asyncio.to_thread(self._last_end, bar.symbol)
        if last_end is not None and bar.start_ts > last_end:
            expected = [t for d in self.calendar.sessions_in_range(self.calendar.ny_date(last_end),
                                                                    self.calendar.ny_date(bar.start_ts))
                        for t in self.calendar.regular_bar_starts(d, self.step) if last_end <= t < bar.start_ts]
            if expected:
                await self.flush_bars()
                await self._backfill_then_live(f"hole of {len(expected)} bar(s) before streamed {bar.symbol} bar "
                                               f"{iso(bar.start_ts)}")
        self.monitor.observe_bar(bar)
        self._bar_buf.append(bar)
        prev = self._last_stored_end.get(bar.symbol)
        if prev is None or bar.end_ts > prev:
            self._last_stored_end[bar.symbol] = bar.end_ts

    async def _periodic_flush(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await _wait(stop, self.flush_interval_s)
            await self.flush()

    async def flush(self) -> None:
        await self.flush_quotes()
        await self.flush_trades()
        await self.flush_bars()
        await self._flush_stream_issues()

    def _take_stream_rejected(self, bucket: str) -> list[tuple[Mapping[str, Any], Sequence[str]]]:
        taken, self._stream_rejected[bucket] = self._stream_rejected[bucket], []
        return taken

    async def _flush_stream_issues(self) -> None:
        if not self._stream_issues or self.db is None:
            self._stream_issues = []
            return
        issues, self._stream_issues = self._stream_issues, []
        now = to_utc(self.clock.now())
        rejected = sum(1 for i in issues if i.action.value == "REJECT")
        report = ValidationReport(kind="quotes", accepted=[], accepted_index=[], rejected=[], issues=issues, gaps=[],
                                  validated_at=now, input_count=rejected)
        await asyncio.to_thread(record_quality_events, self.db, report, self.feed,
                                context={"kind": "stream_validation", "stream_rejected_total": self._stream_rejected_count})

    async def flush_quotes(self) -> dict[str, Any] | None:
        stream_rej = self._take_stream_rejected("quotes")
        if not self._quote_buf and not stream_rej:
            return None
        batch, self._quote_buf = self._quote_buf, []
        now = to_utc(self.clock.now())
        report = validate_quotes(batch, self.calendar, self.policy, validated_at=now)
        if not report.accepted and not report.rejected and not stream_rej:
            return None
        run_id = new_id("mdrun")
        spec = self._spec("ingestion_stream_quotes", run_id,
                          {"symbols": sorted({q.symbol for q in batch} | {str(r.get("symbol")) for r, _ in stream_rej})})
        extras = report.raw_extras(rejected_raw_json(report))
        extras = replace(extras, rejected=[*extras.rejected, *stream_rej])
        manifest = await asyncio.to_thread(self.store.write_quotes, list(report.accepted), manifest=spec, extras=extras)
        if self.db is not None and report.issues:
            await asyncio.to_thread(record_quality_events, self.db, report, self.feed,
                                    context={"run_id": run_id, "kind": "stream_quotes"})
        return manifest

    async def flush_trades(self) -> dict[str, Any] | None:
        stream_rej = self._take_stream_rejected("trades")
        if not self._trade_buf and not stream_rej:
            return None
        batch, self._trade_buf = self._trade_buf, []
        now = to_utc(self.clock.now())
        report = validate_trades(batch, self.calendar, self.policy, validated_at=now)
        if not report.accepted and not report.rejected and not stream_rej:
            return None
        run_id = new_id("mdrun")
        spec = self._spec("ingestion_stream_trades", run_id,
                          {"symbols": sorted({t.symbol for t in batch} | {str(r.get("symbol")) for r, _ in stream_rej})})
        extras = report.raw_extras(rejected_raw_json(report))
        extras = replace(extras, rejected=[*extras.rejected, *stream_rej])
        manifest = await asyncio.to_thread(self.store.write_trades, list(report.accepted), manifest=spec, extras=extras)
        if self.db is not None and report.issues:
            await asyncio.to_thread(record_quality_events, self.db, report, self.feed,
                                    context={"run_id": run_id, "kind": "stream_trades"})
        return manifest

    async def flush_bars(self) -> dict[str, Any] | None:
        stream_rej = self._take_stream_rejected("bars")
        if not self._bar_buf and not stream_rej:
            return None
        batch, self._bar_buf = self._bar_buf, []
        now = to_utc(self.clock.now())
        report = validate_bars(batch, self.calendar, _no_gaps(self.policy), validated_at=now)
        if not report.accepted and not report.rejected and not stream_rej:
            return None
        run_id = new_id("mdrun")
        bars = [b.model_copy(update={"dataset_version": PENDING_DATASET_VERSION}) for b in report.accepted]
        spec = self._spec("ingestion_stream_bars", run_id,
                          {"symbols": sorted({b.symbol for b in batch} | {str(r.get("symbol")) for r, _ in stream_rej})})
        extras = report.raw_extras(rejected_raw_json(report))
        extras = replace(extras, rejected=[*extras.rejected, *stream_rej])
        manifest = await asyncio.to_thread(self.store.write_bars, bars, manifest=spec, extras=extras)
        if self.db is not None and report.issues:
            await asyncio.to_thread(record_quality_events, self.db, report, self.feed,
                                    context={"run_id": run_id, "kind": "stream_bars"})
        return manifest

    # ------------------------------------------------------------------ persistence helpers
    def _record_run(self, result: BackfillResult, kind: str) -> None:
        if self.db is None:
            return
        provider = str(getattr(self.provider, "source_id", type(self.provider).__name__))
        mode = str(getattr(getattr(self.provider, "feed_mode", None), "value", "unknown"))
        starts = [w[0] for w in result.windows.values() if w[0]]
        ends = [w[1] for w in result.windows.values() if w[1]]
        row = (result.run_id, utc_iso(result.started_at), utc_iso(result.finished_at) if result.finished_at else None,
               provider, mode, kind, json.dumps(sorted(result.windows)), min(starts, default=None),
               max(ends, default=None), "OK" if result.ok else "ERROR",
               result.rows_accepted, result.rows_rejected, len(result.gaps), result.dataset_version,
               json.dumps({"errors": result.errors}) if result.errors else None)
        self.db.write(lambda c: c.execute(
            "INSERT INTO md_ingestion_runs(run_id, started_at, finished_at, provider, feed_mode, kind, symbols, "
            "window_start, window_end, status, rows_accepted, rows_rejected, gaps, dataset_version, detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row))

    def _failure(self, kind: str, exc_name: str, detail: str) -> None:
        log.warning("market data %s failure: %s %s", kind, exc_name, detail)
        if self.db is not None:
            with contextlib.suppress(Exception):
                self.db.write(lambda c: record_failure(c, component=f"market_data.{kind}", kind=exc_name, detail=detail))


def _no_gaps(policy: QualityPolicy) -> QualityPolicy:
    return replace(policy, report_gaps=False)


async def _wait(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=max(0.0, seconds))
