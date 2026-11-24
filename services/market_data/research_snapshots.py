"""Immutable, versioned *cumulative* research snapshots over compatible stored partitions (roadmap Phase 2).

Ingestion writes one *incremental* dataset per backfill/flush run. Research needs one complete history instead,
and it must be reproducible later even though new partitions keep arriving and old bars can be revised. A research
snapshot is that history, frozen as a *definition*, never a copy:

* **constituents**: the exact dataset versions it reads. Only compatible partitions: one feed (never IEX + SIP, never
  synthetic + real), one bar adjustment, incremental series (a self-contained snapshot-mode series such as a fixture
  or import is used alone, as its active series), no injected defects, and only datasets *registered* by the cutoff;
* **cutoff**: a knowledge time. Only rows with ``available_at_ts <= cutoff`` exist for the snapshot, and among
  revisions of a bar the latest one available at the cutoff wins (the store's point-in-time rule), so a revision
  published after the cutoff can never leak into a reproduced experiment;
* **content hash**: SHA-256 over every resolved bar (values + availability + source dataset), computed by DuckDB when
  the snapshot is built and recomputed by :func:`verify_snapshot`;
* **lineage + coverage**: constituent manifests (their own content-addressed versions and source ids), per-symbol
  sessions, missing sessions, minute coverage, revised bars and stale tails.

``snapshot_id = "rs_" + sha256(canonical definition)[:16]`` -- the same constituents + cutoff + scope always give the
same id. The store resolves a snapshot id wherever it accepts a ``dataset_version`` (bars, quotes, corporate
actions), so research code reads a snapshot exactly like a dataset.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from packages.contracts.enums import DataFeed
from packages.core.canonical import canonical_json, sha256_hex

from .calendar import CalendarOutOfRange, MarketCalendar, get_calendar
from .common import iso, to_utc

SNAPSHOT_SCHEMA = "lat-research-snapshot/1"
SNAPSHOT_PREFIX = "rs_"
SNAPSHOT_KIND = "research_snapshot"
EXCLUDED_KINDS = frozenset({"lean_sample_import"})
INCOMPLETE_SESSION_COVERAGE = 0.5
"""A regular session with fewer than this fraction of its minute bars is reported as incomplete (IEX bars exist
only for minutes with an IEX trade, so this is a report, never a fill)."""


class SnapshotError(ValueError):
    pass


def is_snapshot_id(value: str | None) -> bool:
    return bool(value) and str(value).startswith(SNAPSHOT_PREFIX)


def _naive(ts: datetime) -> datetime:
    return to_utc(ts).replace(tzinfo=None)


def _aware(ts: Any) -> datetime | None:
    if ts is None:
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _created(m: Mapping[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(m["created_at"]).replace("Z", "+00:00")).astimezone(UTC)
    except (KeyError, ValueError):
        return None


def select_constituents(store: Any, *, feed: DataFeed, timeframe: str, cutoff: datetime,
                        adjustment: str = "raw") -> list[dict[str, Any]]:
    """Compatible datasets for one (feed, timeframe, adjustment) registered by ``cutoff`` (registration order)."""
    feed = DataFeed(feed)
    synthetic = feed == DataFeed.SYNTHETIC
    active = store.active_series(feed).get(feed.value, {})
    out: list[dict[str, Any]] = []
    for m in store.list_manifests(feed, include_defective=False):
        if m.get("defects_injected") or str(m.get("kind")) in EXCLUDED_KINDS:
            continue
        if bool(m.get("is_synthetic")) != synthetic:
            continue  # never splice synthetic and real rows
        created = _created(m)
        if created is None or created > cutoff:
            continue  # not known yet at the cutoff
        tfs = list(m.get("timeframes") or [])
        counts = dict(m.get("row_counts") or {})
        has_bars = timeframe in tfs
        side_data = not tfs and (counts.get("quotes") or counts.get("corporate_actions") or counts.get("trades"))
        if not (has_bars or side_data):
            continue
        if has_bars and str(m.get("adjustment") or "raw") != adjustment:
            continue
        out.append(m)
    if active:
        # A self-contained snapshot-mode series (fixture/import) is used alone: never composed with increments.
        versions = set(active.values())
        snap = [m for m in out if m.get("dataset_version") in versions]
        if snap:
            return snap
    return [m for m in out if m.get("series_mode", "incremental") != "snapshot"]


def _bars_cte(versions: Sequence[str], timeframe: str, cutoff: datetime,
              symbols: Sequence[str] | None) -> tuple[str, list[Any]]:
    where = [f"b.dataset_version IN ({', '.join('?' for _ in versions)})", "b.timeframe = ?", "b.available_at_ts <= ?"]
    params: list[Any] = [*versions, timeframe, _naive(cutoff)]
    if symbols:
        where.append(f"b.symbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(symbols)
    cte = f"""
        WITH cand AS (SELECT b.* FROM bars b WHERE {' AND '.join(where)}),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY symbol, timeframe, start_ts
                                         ORDER BY available_at_ts DESC, ingest_seq DESC) AS rn,
                   count(*) OVER (PARTITION BY symbol, timeframe, start_ts) AS nrev
            FROM cand),
        resolved AS (SELECT * FROM ranked WHERE rn = 1)
    """  # noqa: S608 - placeholders only
    return cte, params


def _content(store: Any, versions: Sequence[str], timeframe: str, cutoff: datetime,
             symbols: Sequence[str] | None) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[date, tuple[int, int]]]]:
    """(rows sha256, per-symbol stats, per-symbol {ny_date: (regular bars, all bars)})."""
    cte, params = _bars_cte(versions, timeframe, cutoff, symbols)
    hash_sql = cte + """
        SELECT sha256(coalesce(string_agg(concat_ws('|', symbol, timeframe, CAST(start_ts AS VARCHAR),
                   CAST(open AS VARCHAR), CAST(high AS VARCHAR), CAST(low AS VARCHAR), CAST(close AS VARCHAR),
                   CAST(volume AS VARCHAR), coalesce(CAST(vwap AS VARCHAR), ''), CAST(available_at_ts AS VARCHAR),
                   dataset_version), chr(10) ORDER BY symbol, start_ts), '')), count(*)
        FROM resolved"""  # noqa: S608 - constant SQL around placeholder-only CTE
    stats_sql = cte + """
        SELECT symbol, count(*), min(start_ts), max(end_ts), max(available_at_ts), sum(CASE WHEN nrev > 1 THEN 1 ELSE 0 END)
        FROM resolved GROUP BY symbol ORDER BY symbol"""  # noqa: S608
    day_sql = cte + """
        SELECT symbol, CAST(timezone('America/New_York', start_ts::TIMESTAMPTZ) AS DATE) AS d,
               sum(CASE WHEN session = 'REGULAR' THEN 1 ELSE 0 END), count(*)
        FROM resolved GROUP BY symbol, d ORDER BY symbol, d"""  # noqa: S608
    with store._con() as con:
        digest, total = con.execute(hash_sql, params).fetchone()
        stat_rows = con.execute(stats_sql, params).fetchall()
        day_rows = con.execute(day_sql, params).fetchall()
    stats: dict[str, dict[str, Any]] = {}
    for sym, n, first, last, last_avail, revised in stat_rows:
        stats[str(sym)] = {"bars": int(n), "first_bar_start": iso(_aware(first)), "last_bar_end": iso(_aware(last)),
                           "last_available_at": iso(_aware(last_avail)), "revised_bars": int(revised or 0)}
    days: dict[str, dict[date, tuple[int, int]]] = defaultdict(dict)
    for sym, d, regular, allbars in day_rows:
        days[str(sym)][d] = (int(regular or 0), int(allbars))
    stats["__total__"] = {"bars": int(total or 0)}
    return str(digest), stats, days


def coverage_report(days: Mapping[str, Mapping[date, tuple[int, int]]], stats: Mapping[str, Mapping[str, Any]], *,
                    cutoff: datetime, calendar: MarketCalendar | None = None, timeframe: str = "1Min") -> dict[str, Any]:
    """Per symbol: sessions, missing sessions, incomplete sessions and the stale tail at the cutoff."""
    cal = calendar or get_calendar()
    out: dict[str, Any] = {}
    cutoff_day = cal.ny_date(cutoff)
    for sym, by_day in sorted(days.items()):
        if not by_day:
            continue
        first, last = min(by_day), max(by_day)
        try:
            expected = cal.sessions_in_range(first, last)
            last_complete = cal.previous_session(cutoff_day) if not (
                cal.is_session(cutoff_day) and cutoff >= cal.session_close(cutoff_day)) else cutoff_day
            tail = cal.sessions_in_range(last, last_complete)[1:]
        except CalendarOutOfRange:
            expected, tail = sorted(by_day), []
        missing = [d for d in expected if d not in by_day]
        fractions: list[float] = []
        incomplete: list[str] = []
        for d in expected:
            if d not in by_day:
                continue
            minutes = cal.regular_minutes_count(d) if timeframe == "1Min" else 1
            frac = by_day[d][0] / minutes if minutes else 0.0
            fractions.append(frac)
            if timeframe == "1Min" and frac < INCOMPLETE_SESSION_COVERAGE:
                incomplete.append(d.isoformat())
        fractions.sort()
        out[sym] = {
            **dict(stats.get(sym, {})), "sessions": len(by_day), "expected_sessions": len(expected),
            "missing_sessions": len(missing), "missing_session_dates": [d.isoformat() for d in missing[:50]],
            "incomplete_sessions": len(incomplete), "incomplete_session_dates": incomplete[:50],
            "median_regular_minute_coverage": round(fractions[len(fractions) // 2], 4) if fractions else None,
            "stale_sessions_at_cutoff": len(tail),
        }
    return out


def _definition(feed: DataFeed, timeframe: str, adjustment: str, cutoff: datetime, symbols: Sequence[str] | None,
                constituents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "snapshot_schema": SNAPSHOT_SCHEMA, "feed": feed.value, "timeframe": timeframe, "adjustment": adjustment,
        "cutoff": iso(cutoff), "symbols_requested": sorted(symbols) if symbols else None,
        "constituents": [str(m["dataset_version"]) for m in constituents],
    }


def build_snapshot(store: Any, *, feed: DataFeed, timeframe: str = "1Min", cutoff: datetime | None = None,
                   symbols: Sequence[str] | None = None, adjustment: str = "raw", note: str = "",
                   calendar: MarketCalendar | None = None, created_at: datetime | None = None) -> dict[str, Any]:
    """Build (or return the identical existing) research snapshot. Raises :class:`SnapshotError` when empty."""
    feed = DataFeed(feed)
    cut = to_utc(cutoff or datetime.now(UTC))
    constituents = select_constituents(store, feed=feed, timeframe=timeframe, cutoff=cut, adjustment=adjustment)
    bar_versions = [str(m["dataset_version"]) for m in constituents if timeframe in (m.get("timeframes") or [])]
    if not bar_versions:
        raise SnapshotError(f"no compatible {feed.value} {timeframe} datasets registered by {iso(cut)}")
    definition = _definition(feed, timeframe, adjustment, cut, symbols, constituents)
    snapshot_id = SNAPSHOT_PREFIX + sha256_hex(canonical_json(definition))[:16]
    existing = get_snapshot(store, snapshot_id)
    if existing is not None:
        return existing
    digest, stats, days = _content(store, bar_versions, timeframe, cut, symbols)
    total = int(stats.pop("__total__")["bars"])
    if total == 0:
        raise SnapshotError(f"constituents resolve to no {timeframe} bars at {iso(cut)}")
    coverage = coverage_report(days, stats, cutoff=cut, calendar=calendar, timeframe=timeframe)
    firsts = [v["first_bar_start"] for v in stats.values() if v.get("first_bar_start")]
    lasts = [v["last_bar_end"] for v in stats.values() if v.get("last_bar_end")]
    limitations = sorted({str(x) for m in constituents for x in (m.get("limitations") or [])})
    manifest = {
        **definition, "snapshot_id": snapshot_id, "dataset_version": snapshot_id, "kind": SNAPSHOT_KIND,
        "series_mode": "research_snapshot", "is_synthetic": feed == DataFeed.SYNTHETIC, "defects_injected": [],
        "symbols": sorted(stats), "timeframes": [timeframe], "start": min(firsts), "end": max(lasts),
        "rows_sha256": digest, "row_counts": {"bars": total, f"bars_{timeframe}": total},
        "row_counts_by_symbol": {s: {"bars": v["bars"]} for s, v in sorted(stats.items())},
        "coverage": coverage,
        "lineage": [{"dataset_version": m["dataset_version"], "kind": m.get("kind"), "source_id": m.get("source_id"),
                     "created_at": m.get("created_at"), "start": m.get("start"), "end": m.get("end"),
                     "row_counts": m.get("row_counts"), "generator": m.get("generator")} for m in constituents],
        "source_ids": sorted({str(m.get("source_id")) for m in constituents}),
        "limitations": limitations, "note": note[:500],
        "created_at": iso(to_utc(created_at or datetime.now(UTC))),
    }
    _persist(store, manifest)
    return manifest


def _ensure_table(con: Any) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS research_snapshots (
        snapshot_id VARCHAR PRIMARY KEY, feed VARCHAR NOT NULL, timeframe VARCHAR NOT NULL, cutoff TIMESTAMP NOT NULL,
        created_at TIMESTAMP NOT NULL, seq BIGINT NOT NULL, manifest_json VARCHAR NOT NULL)""")


def _persist(store: Any, manifest: Mapping[str, Any]) -> None:
    sid = str(manifest["snapshot_id"])
    with store._con() as con:
        _ensure_table(con)
        if con.execute("SELECT 1 FROM research_snapshots WHERE snapshot_id = ?", [sid]).fetchone() is None:
            seq_row = con.execute("SELECT coalesce(max(seq), 0) + 1 FROM research_snapshots").fetchone()
            con.execute("INSERT INTO research_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", [
                sid, manifest["feed"], manifest["timeframe"],
                _naive(datetime.fromisoformat(str(manifest["cutoff"]).replace("Z", "+00:00"))),
                _naive(datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))),
                int(seq_row[0]) if seq_row else 1, json.dumps(manifest, sort_keys=True)])
    path = snapshot_dir(store) / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    store._bump_generation()


def snapshot_dir(store: Any) -> Path:
    return Path(store.datasets_dir) / "processed" / "snapshots"


def get_snapshot(store: Any, snapshot_id: str) -> dict[str, Any] | None:
    if not is_snapshot_id(snapshot_id):
        return None
    with store._con() as con:
        _ensure_table(con)
        row = con.execute("SELECT manifest_json FROM research_snapshots WHERE snapshot_id = ?", [snapshot_id]).fetchone()
    return json.loads(row[0]) if row else None


def list_snapshots(store: Any, *, feed: DataFeed | None = None, timeframe: str | None = None) -> list[dict[str, Any]]:
    """Snapshots in build order (oldest first)."""
    sql, params = "SELECT manifest_json FROM research_snapshots WHERE 1=1", []
    if feed is not None:
        sql += " AND feed = ?"
        params.append(DataFeed(feed).value)
    if timeframe is not None:
        sql += " AND timeframe = ?"
        params.append(timeframe)
    with store._con() as con:
        _ensure_table(con)
        rows = con.execute(sql + " ORDER BY seq", params).fetchall()
    return [json.loads(r[0]) for r in rows]


def resolve_scope(store: Any, snapshot_id: str) -> tuple[list[str], datetime]:
    """(constituent dataset versions, cutoff) the store filters by for a snapshot id."""
    snap = get_snapshot(store, snapshot_id)
    if snap is None:
        raise SnapshotError(f"unknown research snapshot {snapshot_id}")
    versions = [str(v) for v in snap["constituents"]]
    cutoff = datetime.fromisoformat(str(snap["cutoff"]).replace("Z", "+00:00")).astimezone(UTC)
    return versions, cutoff


def verify_snapshot(store: Any, snapshot_id: str) -> dict[str, Any]:
    """Recompute the definition hash, every constituent manifest hash and the resolved-content hash."""
    from .manifest import manifest_content_ok

    snap = get_snapshot(store, snapshot_id)
    if snap is None:
        return {"snapshot_id": snapshot_id, "ok": False, "errors": ["unknown snapshot"]}
    errors: list[str] = []
    definition = {k: snap[k] for k in ("snapshot_schema", "feed", "timeframe", "adjustment", "cutoff",
                                       "symbols_requested", "constituents")}
    if SNAPSHOT_PREFIX + sha256_hex(canonical_json(definition))[:16] != snapshot_id:
        errors.append("snapshot definition does not hash to its id (modified?)")
    for v in snap["constituents"]:
        m = store.get_manifest(str(v))
        if m is None:
            errors.append(f"constituent {v} missing")
        elif not manifest_content_ok(m):
            errors.append(f"constituent {v} manifest does not hash to its version")
    versions, cutoff = resolve_scope(store, snapshot_id)
    bar_versions = [v for v in versions if snap["timeframe"] in ((store.get_manifest(v) or {}).get("timeframes") or [])]
    digest, stats, _ = _content(store, bar_versions, str(snap["timeframe"]), cutoff, snap.get("symbols_requested"))
    total = int(stats.pop("__total__")["bars"])
    if digest != snap["rows_sha256"]:
        errors.append("resolved bars differ from the snapshot content hash (constituent rows changed?)")
    if total != int(snap["row_counts"]["bars"]):
        errors.append(f"resolved {total} bars, snapshot recorded {snap['row_counts']['bars']}")
    return {"snapshot_id": snapshot_id, "ok": not errors, "errors": errors, "rows": total, "rows_sha256": digest}


def new_complete_sessions(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> int:
    """Complete-session growth of ``new`` over ``old`` (minimum over the shared symbols; new symbols count fully)."""
    def sessions(m: Mapping[str, Any]) -> dict[str, int]:
        cov = dict(m.get("coverage") or {})
        return {s: int(v.get("sessions", 0)) - int(v.get("incomplete_sessions", 0)) for s, v in cov.items()}

    after = sessions(new)
    if old is None:
        return min(after.values(), default=0)
    before = sessions(old)
    shared = [after[s] - before[s] for s in after if s in before]
    if shared:
        return min(shared)
    return max(after.values(), default=0)
