"""Scheduled historical backfill + incremental ingestion that keeps research snapshots current (roadmap Phase 2).

``HistoryPipeline.run()`` is idempotent and safe to schedule (``python -m services.market_data.cli pipeline``):

1. **plan** per symbol from what is stored for this feed/timeframe: *history* windows that extend the series back to
   ``history_days`` and *increment* windows from the last stored bar to the close of the most recent XNYS session
   that ended at least ``settle`` ago. Only whole sessions are requested (never a still-forming one), which also
   keeps historical SIP requests outside Alpaca's most-recent-15-minutes restriction. Windows are chunked
   (``chunk_sessions``) and bounded per run (``max_requests``); newest data first.
2. **fetch + validate** every window through the existing provider and quality checks (duplicates, out-of-order,
   OHLC, outliers, look-ahead; gaps are reported, never filled) and **store** it as its own incremental, content-
   addressed dataset (manifest with source id, window, generator and integrity hashes). Each window is one
   ``md_ingestion_runs`` row; findings go to ``data_quality_events``.
3. **snapshot**: when anything new was stored (or no snapshot exists) build the cumulative research snapshot for
   the feed (``research_snapshots.build_snapshot``: compatible partitions only, knowledge cutoff, lineage,
   coverage, content hash).
4. **research trigger**: compare the new snapshot with the one research last used (``research_state.json``). Only
   when it adds at least ``min_new_sessions`` complete sessions (or new symbols) is it marked ``ready`` with the
   reason; ``services.research.autonomous`` switches to it then and not on every minute of new data.

No paid feed is ever enabled or purchased here: the feed is whatever the caller's provider is configured for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.contracts.enums import DataFeed, FeedMode
from packages.core.audit import append_audit, record_failure
from packages.core.clock import Clock
from packages.core.db import Database, utc_iso
from packages.core.ids import new_id

from .calendar import CalendarOutOfRange, MarketCalendar, get_calendar
from .common import FEED_MODE_TO_FEED, PENDING_DATASET_VERSION, canonical_timeframe, iso, timeframe_delta, to_utc
from .manifest import DatasetSpec
from .quality import QualityPolicy, compute_bar_gaps, only_available, record_quality_events, rejected_raw_json, validate_bars
from .research_snapshots import SnapshotError, build_snapshot, get_snapshot, new_complete_sessions, snapshot_dir
from .store import MarketDataStore

log = logging.getLogger("lat.market_data.pipeline")
GENERATOR = "services.market_data.pipeline"
GENERATOR_VERSION = "1.0.0"
STATE_FILE = "research_state.json"


@dataclass(frozen=True)
class PipelineConfig:
    symbols: tuple[str, ...]
    timeframe: str = "1Min"
    history_days: int = 730
    chunk_sessions: int = 10
    settle: timedelta = timedelta(minutes=20)
    max_requests: int = 200
    min_new_sessions: int = 5
    request_pause_s: float = 0.35
    """Spacing between provider requests (Alpaca Basic allows ~200 requests/minute)."""


@dataclass(frozen=True)
class Window:
    symbol: str
    start: datetime
    end: datetime
    kind: str  # history | increment | initial

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "start": iso(self.start), "end": iso(self.end), "kind": self.kind}


@dataclass
class PipelineRun:
    run_id: str
    started_at: datetime
    planned: list[Window] = field(default_factory=list)
    done: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    skipped_budget: int = 0
    snapshot_id: str | None = None
    snapshot_reused: bool = False
    research: dict[str, Any] = field(default_factory=dict)
    finished_at: datetime | None = None

    @property
    def rows_accepted(self) -> int:
        return sum(int(d.get("rows_accepted", 0)) for d in self.done)

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "started_at": iso(self.started_at), "finished_at": iso(self.finished_at),
                "planned": len(self.planned), "windows": self.done, "errors": self.errors,
                "skipped_for_budget": self.skipped_budget, "rows_accepted": self.rows_accepted,
                "snapshot_id": self.snapshot_id, "snapshot_reused": self.snapshot_reused, "research": self.research,
                "ok": not self.errors}


def _feed_of(provider: Any) -> DataFeed:
    feed = getattr(provider, "feed", None)
    if isinstance(feed, DataFeed):
        return feed
    return FEED_MODE_TO_FEED[FeedMode(provider.feed_mode)]


class HistoryPipeline:
    def __init__(self, provider: Any, store: MarketDataStore, db: Database | None, clock: Clock, config: PipelineConfig,
                 *, calendar: MarketCalendar | None = None, policy: QualityPolicy | None = None,
                 limitations: Sequence[str] = ()) -> None:
        if not config.symbols:
            raise ValueError("the pipeline needs at least one symbol")
        self.provider = provider
        self.store = store
        self.db = db
        self.clock = clock
        self.config = replace(config, symbols=tuple(sorted({s.upper() for s in config.symbols})),
                              timeframe=canonical_timeframe(config.timeframe))
        self.calendar = calendar or get_calendar()
        self.policy = policy or QualityPolicy()
        self.feed = _feed_of(provider)
        self.step = timeframe_delta(self.config.timeframe)
        self.limitations = tuple(limitations)

    # ------------------------------------------------------------------ planning
    def last_complete_close(self, now: datetime) -> datetime | None:
        """Close of the most recent session that ended at least ``settle`` before ``now``."""
        horizon = to_utc(now) - self.config.settle
        d = self.calendar.ny_date(horizon)
        try:
            for _ in range(10):
                if self.calendar.is_session(d) and self.calendar.session_close(d) <= horizon:
                    return self.calendar.session_close(d)
                d = self.calendar.previous_session(d)
        except CalendarOutOfRange:
            return None
        return None

    def coverage(self) -> dict[str, tuple[datetime, datetime]]:
        """(first bar start, last bar end) stored per symbol for this feed and timeframe (non-defective)."""
        syms = list(self.config.symbols)
        marks = ", ".join("?" for _ in syms)
        where = f"b.feed = ? AND b.timeframe = ? AND NOT d.defects_injected AND b.symbol IN ({marks})"
        sql = "SELECT b.symbol, min(b.start_ts), max(b.end_ts) FROM bars b JOIN datasets d ON " \
              "d.dataset_version = b.dataset_version WHERE " + where + " GROUP BY b.symbol"  # noqa: S608
        with self.store._con() as con:
            rows = con.execute(sql, [self.feed.value, self.config.timeframe, *syms]).fetchall()
        out: dict[str, tuple[datetime, datetime]] = {}
        for sym, lo, hi in rows:
            out[str(sym)] = (lo.replace(tzinfo=UTC), hi.replace(tzinfo=UTC))
        return out

    def _chunks(self, symbol: str, start: datetime, end: datetime, kind: str) -> list[Window]:
        if end <= start:
            return []
        try:
            days = self.calendar.sessions_in_range(self.calendar.ny_date(start), self.calendar.ny_date(end))
        except CalendarOutOfRange:
            return []
        days = [d for d in days if self.calendar.session_close(d) > start and self.calendar.session_open(d) < end]
        out: list[Window] = []
        n = max(1, self.config.chunk_sessions)
        for i in range(0, len(days), n):
            group: list[date] = days[i:i + n]
            lo = max(start, self.calendar.session_open(group[0]))
            hi = min(end, self.calendar.session_close(group[-1]))
            if hi > lo:
                out.append(Window(symbol, lo, hi, kind))
        return out

    def plan(self, now: datetime | None = None) -> list[Window]:
        now = to_utc(now or self.clock.now())
        cut = self.last_complete_close(now)
        if cut is None:
            return []
        try:
            first_day = self.calendar.ny_date(cut - timedelta(days=self.config.history_days))
            target_start = self.calendar.session_open(
                first_day if self.calendar.is_session(first_day) else self.calendar.next_session(first_day))
        except CalendarOutOfRange:
            return []
        cov = self.coverage()
        windows: list[Window] = []
        for sym in self.config.symbols:
            if sym not in cov:
                windows += reversed(self._chunks(sym, target_start, cut, "initial"))
                continue
            first, last = cov[sym]
            if last < cut:
                windows += reversed(self._chunks(sym, last, cut, "increment"))
            if first - target_start > timedelta(days=1):
                windows += reversed(self._chunks(sym, target_start, first, "history"))
        # newest first across symbols: increments, then initial/history going backwards in time
        rank = {"increment": 0, "initial": 1, "history": 2}
        return sorted(windows, key=lambda w: (rank[w.kind], -w.end.timestamp(), w.symbol))

    # ------------------------------------------------------------------ run
    async def run(self, now: datetime | None = None, *, build: bool = True) -> PipelineRun:
        started = to_utc(now or self.clock.now())
        run = PipelineRun(run_id=new_id("mdpipe"), started_at=started)
        run.planned = self.plan(started)
        todo = run.planned[: self.config.max_requests]
        run.skipped_budget = max(0, len(run.planned) - len(todo))
        for i, w in enumerate(todo):
            if i:
                await asyncio.sleep(self.config.request_pause_s)
            try:
                run.done.append(await self._window(w, run.run_id))
            except Exception as exc:  # one bad window never stops the run; it is retried next time
                detail = f"{type(exc).__name__}: {exc}"[:500]
                run.errors.append({**w.as_dict(), "error": detail})
                self._failure("pipeline_window", type(exc).__name__, f"{w.symbol} {iso(w.start)}..{iso(w.end)}: {detail}")
                if "rate limit" in detail.lower() or "429" in detail:
                    break  # back off: the next scheduled run continues
        if build:
            self._snapshot(run, started)
        run.finished_at = to_utc(self.clock.now())
        self._audit(run)
        return run

    async def _fetch(self, w: Window) -> tuple[list[Any], dict[int, str]]:
        fetch_raw: Any = getattr(self.provider, "fetch_bar_records", None)
        if fetch_raw is not None:
            records, raw = await fetch_raw(w.symbol, w.start, w.end, self.config.timeframe)
            return list(records), dict(raw)
        return list(await self.provider.get_bars(w.symbol, w.start, w.end, self.config.timeframe)), {}

    async def _window(self, w: Window, run_id: str) -> dict[str, Any]:
        wid = new_id("mdrun")
        at = to_utc(self.clock.now())
        records, raw = await self._fetch(w)
        report = validate_bars(records, self.calendar, replace(self.policy, report_gaps=False), validated_at=at)
        report = only_available(report, at)
        bars = [b for b in report.accepted if b.symbol == w.symbol and b.timeframe == self.config.timeframe]
        gaps = compute_bar_gaps(bars, self.calendar, expected_range=(w.start, w.end), expected_symbols=[w.symbol])
        version: str | None = None
        if report.accepted or report.rejected:
            spec = DatasetSpec(
                source_id=str(getattr(self.provider, "source_id", "unknown")), feed=self.feed, generator=GENERATOR,
                generator_version=GENERATOR_VERSION, is_synthetic=self.feed == DataFeed.SYNTHETIC,
                kind=f"pipeline_{w.kind}_bars", description=f"{w.kind} {w.symbol} {iso(w.start)}..{iso(w.end)} ({run_id})",
                limitations=self.limitations,
                params={"run_id": run_id, "window_run_id": wid, "timeframe": self.config.timeframe,
                        "windows": {w.symbol: [iso(w.start), iso(w.end)]}})
            pending = [b.model_copy(update={"dataset_version": PENDING_DATASET_VERSION}) for b in report.accepted]
            manifest = await asyncio.to_thread(self.store.write_bars, pending, manifest=spec,
                                               extras=report.raw_extras({**rejected_raw_json(report), **raw}),
                                               created_at=at)
            version = str(manifest["dataset_version"])
        if self.db is not None:
            await asyncio.to_thread(record_quality_events, self.db, report, self.feed,
                                    context={"run_id": wid, "kind": f"pipeline_{w.kind}"})
            missing = sum(g.missing for g in gaps)
            row = (wid, utc_iso(at), utc_iso(to_utc(self.clock.now())),
                   str(getattr(self.provider, "source_id", "unknown")),
                   str(getattr(getattr(self.provider, "feed_mode", None), "value", self.feed.value)), f"pipeline_{w.kind}",
                   json.dumps([w.symbol]), iso(w.start), iso(w.end), "OK", len(report.accepted), len(report.rejected),
                   len(gaps), version, json.dumps({"pipeline_run": run_id, "missing_bars": missing}))
            self.db.write(lambda c: c.execute(
                "INSERT INTO md_ingestion_runs(run_id, started_at, finished_at, provider, feed_mode, kind, symbols, "
                "window_start, window_end, status, rows_accepted, rows_rejected, gaps, dataset_version, detail) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row))
        return {**w.as_dict(), "rows_accepted": len(report.accepted), "rows_rejected": len(report.rejected),
                "gaps": len(gaps), "missing_bars": sum(g.missing for g in gaps), "dataset_version": version}

    # ------------------------------------------------------------------ snapshot + research trigger
    def state_path(self) -> Path:
        return snapshot_dir(self.store) / STATE_FILE

    def research_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _snapshot(self, run: PipelineRun, now: datetime) -> None:
        state = self.research_state()
        key = f"{self.feed.value}:{self.config.timeframe}"
        feed_state = dict(state.get(key) or {})
        if not run.rows_accepted and feed_state.get("latest"):
            run.snapshot_id, run.snapshot_reused = str(feed_state["latest"]), True
            run.research = {"ready": feed_state.get("ready"), "reason": "no new bars stored this run"}
            return
        cutoff = self.last_complete_close(now) or now
        try:
            at = to_utc(self.clock.now())
            snap = build_snapshot(self.store, feed=self.feed, timeframe=self.config.timeframe,
                                  cutoff=max(cutoff, at), calendar=self.calendar,
                                  note=f"pipeline run {run.run_id}", created_at=at)
        except SnapshotError as exc:
            run.research = {"ready": feed_state.get("ready"), "reason": f"no snapshot: {exc}"}
            return
        run.snapshot_id = str(snap["snapshot_id"])
        baseline_id = feed_state.get("ready")
        baseline = get_snapshot(self.store, str(baseline_id)) if baseline_id else None
        growth = new_complete_sessions(baseline, snap)
        new_symbols = sorted(set(snap.get("symbols") or []) - set((baseline or {}).get("symbols") or []))
        is_ready = baseline is None or growth >= self.config.min_new_sessions or bool(new_symbols)
        feed_state["latest"] = run.snapshot_id
        if is_ready and baseline_id != run.snapshot_id:
            feed_state["ready"] = run.snapshot_id
            feed_state["ready_at"] = iso(now)
            feed_state["ready_reason"] = ("first snapshot" if baseline is None else
                                          f"{growth} new complete session(s)" + (f", new symbols {new_symbols}"
                                                                                 if new_symbols else ""))
        state[key] = feed_state
        path = self.state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        run.research = {"ready": feed_state.get("ready"), "latest": run.snapshot_id, "new_complete_sessions": growth,
                        "new_symbols": new_symbols, "threshold": self.config.min_new_sessions,
                        "reason": feed_state.get("ready_reason") if feed_state.get("ready") == run.snapshot_id
                        else f"only {growth} new complete session(s) (< {self.config.min_new_sessions}); research keeps "
                             f"{feed_state.get('ready')}"}

    def _audit(self, run: PipelineRun) -> None:
        if self.db is None:
            return
        payload = {k: v for k, v in run.as_dict().items() if k != "windows"} | {"windows_done": len(run.done)}
        self.db.write(lambda c: append_audit(c, event_type="MARKET_DATA_PIPELINE", source="market_data.pipeline",
                                             mode="RESEARCH", payload=payload))

    def _failure(self, kind: str, exc_name: str, detail: str) -> None:
        log.warning("market data pipeline %s: %s %s", kind, exc_name, detail)
        if self.db is not None:
            with contextlib.suppress(Exception):  # recording a failure must never mask it
                self.db.write(lambda c: record_failure(c, component=f"market_data.{kind}", kind=exc_name, detail=detail))


def ready_snapshot(datasets_dir: Path, feed: DataFeed | str, timeframe: str = "1Min") -> str | None:
    """The snapshot research should use for a feed (``None`` when the pipeline never marked one ready)."""
    path = Path(datasets_dir) / "processed" / "snapshots" / STATE_FILE
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = state.get(f"{DataFeed(feed).value}:{canonical_timeframe(timeframe)}") or {}
    ready = entry.get("ready")
    return str(ready) if ready else None
