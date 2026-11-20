"""Point-in-time symbol eligibility / research-universe snapshots.

A snapshot answers "which symbols may the system consider at ``asof``?" using only data available at ``asof``:

* candidate must be on the risk policy allowlist (``RiskPolicy.symbol_allowlist``; liquid stocks + standard ETFs)
* never on the denylist (``RiskPolicy.symbol_denylist``: leveraged / inverse / volatility ETPs) — checked first
* last point-in-time close ``>= RiskPolicy.min_price`` ($5: no penny stocks)
* average daily dollar volume over the last ``lookback_sessions`` *completed* sessions ``>= min_avg_dollar_volume``
* at least ``min_sessions`` completed sessions observed (fail closed: no/insufficient data => ineligible)
* data present for the latest completed session (otherwise ``STALE_DATA``: the last close may be weeks old)

Daily bars are used when available; otherwise complete sessions are aggregated from regular-session minute bars
(dollar volume = vwap*volume when vwap exists, else close*volume). All arithmetic is ``Decimal``.

Snapshots are persisted with their as-of date in DuckDB (``universe_snapshots``, one row per symbol) and, when a
``Database`` is given, in SQLite (``md_universe_snapshots``, migration 0100). ``snapshot_id`` is content-derived,
so re-running with identical inputs is idempotent.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any

from packages.contracts.enums import DataFeed, MarketSession
from packages.contracts.models import Bar
from packages.contracts.policy import RiskPolicy
from packages.core.canonical import canonical_json, sha256_hex
from packages.core.db import Database, utc_iso

from .calendar import CalendarOutOfRange, MarketCalendar, get_calendar
from .common import iso, to_utc
from .store import MarketDataStore

DEFAULT_MIN_AVG_DOLLAR_VOLUME = Decimal("20000000")  # $20M/day
DEFAULT_LOOKBACK_SESSIONS = 20
DEFAULT_MIN_SESSIONS = 5
_SYMBOL_OK = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")


class EligibilityReason(StrEnum):
    INVALID_SYMBOL = "INVALID_SYMBOL"
    DENYLISTED_LEVERAGED_INVERSE = "DENYLISTED_LEVERAGED_INVERSE"
    NOT_ALLOWLISTED = "NOT_ALLOWLISTED"
    NO_DATA = "NO_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    STALE_DATA = "STALE_DATA"
    BELOW_MIN_PRICE = "BELOW_MIN_PRICE"
    BELOW_MIN_DOLLAR_VOLUME = "BELOW_MIN_DOLLAR_VOLUME"


@dataclass(frozen=True)
class EligibilityRow:
    symbol: str
    eligible: bool
    reasons: tuple[str, ...]
    last_close: Decimal | None
    avg_dollar_volume: Decimal | None
    sessions_observed: int
    feed: str | None
    source: str | None  # "1Day" | "1Min-aggregated" | None
    dataset_versions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "eligible": self.eligible, "reasons": list(self.reasons),
            "last_close": None if self.last_close is None else format(self.last_close, "f"),
            "avg_dollar_volume": None if self.avg_dollar_volume is None else format(self.avg_dollar_volume, "f"),
            "sessions_observed": self.sessions_observed, "feed": self.feed, "source": self.source,
            "dataset_versions": list(self.dataset_versions),
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    snapshot_id: str
    asof_ts: datetime
    asof_date: date
    policy_version: str
    params: dict[str, Any]
    rows: tuple[EligibilityRow, ...]
    created_at: datetime = field(compare=False)

    @property
    def eligible_symbols(self) -> list[str]:
        return [r.symbol for r in self.rows if r.eligible]

    @property
    def dataset_versions(self) -> list[str]:
        return sorted({v for r in self.rows for v in r.dataset_versions})

    @property
    def feeds(self) -> list[str]:
        return sorted({r.feed for r in self.rows if r.feed})

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id, "asof_ts": iso(self.asof_ts), "asof_date": self.asof_date.isoformat(),
            "policy_version": self.policy_version, "params": self.params,
            "eligible_symbols": self.eligible_symbols, "dataset_versions": self.dataset_versions,
            "rows": [r.as_dict() for r in self.rows], "created_at": iso(self.created_at),
        }


def _fixed_iso(ts: datetime) -> str:
    """Fixed-width UTC timestamp so SQLite string comparison orders correctly (microseconds always present)."""
    return to_utc(ts).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _completed_sessions(cal: MarketCalendar, asof: datetime, count: int) -> list[date]:
    """The last ``count`` sessions whose close is <= asof (today's session only once it has closed)."""
    d = cal.ny_date(asof)
    if not cal.is_session(d) or cal.session_close(d) > asof:
        d = cal.previous_session(d)
    return cal.sessions_ending(d, count)


def _dollar_volume(b: Bar) -> Decimal:
    px = b.vwap if b.vwap is not None and b.vwap > 0 else b.close
    return px * b.volume


def _session_stats(store: MarketDataStore, symbol: str, sessions: Sequence[date], cal: MarketCalendar,
                   asof: datetime, feed: DataFeed | None) -> tuple[dict[date, Decimal], Decimal | None, str | None,
                                                                    str | None, set[str]]:
    """(dollar volume per completed session, last PIT close, feed, source, dataset versions)."""
    start = cal.session_open(sessions[0])
    daily = store.load_bars([symbol], start, asof, "1Day", feed=feed, asof=asof)
    per: dict[date, Decimal] = {}
    if daily:
        wanted = set(sessions)
        for b in daily:
            d = cal.ny_date(b.start_ts)
            if d in wanted:
                per[d] = _dollar_volume(b)
        last = max(daily, key=lambda b: b.end_ts)
        return per, last.close, last.feed.value, "1Day", {b.dataset_version for b in daily}
    minute = store.load_bars([symbol], start, asof, "1Min", feed=feed, asof=asof, sessions=[MarketSession.REGULAR])
    if not minute:
        return {}, None, None, None, set()
    grouped: dict[date, list[Bar]] = defaultdict(list)
    for b in minute:
        grouped[cal.ny_date(b.start_ts)].append(b)
    for d in sessions:
        bars = grouped.get(d)
        if bars and cal.session_close(d) <= asof:
            per[d] = sum((_dollar_volume(b) for b in bars), Decimal(0))
    last = max(minute, key=lambda b: b.end_ts)
    return per, last.close, last.feed.value, "1Min-aggregated", {b.dataset_version for b in minute}


def build_universe_snapshot(
    store: MarketDataStore,
    policy: RiskPolicy,
    asof: datetime,
    *,
    candidates: Iterable[str] | None = None,
    feed: DataFeed | None = None,
    min_avg_dollar_volume: Decimal = DEFAULT_MIN_AVG_DOLLAR_VOLUME,
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    calendar: MarketCalendar | None = None,
    created_at: datetime | None = None,
) -> UniverseSnapshot:
    """Eligibility of ``candidates`` (default: the policy allowlist) at ``asof`` using only PIT data."""
    if isinstance(min_avg_dollar_volume, float):
        raise TypeError("min_avg_dollar_volume must be a Decimal")
    if lookback_sessions <= 0 or min_sessions <= 0 or min_sessions > lookback_sessions:
        raise ValueError("need 0 < min_sessions <= lookback_sessions")
    cal = calendar or get_calendar()
    asof = to_utc(asof)
    allow = {s.upper() for s in policy.symbol_allowlist}
    deny = {s.upper() for s in policy.symbol_denylist}
    syms = list(dict.fromkeys(s.strip().upper() for s in (candidates if candidates is not None else policy.symbol_allowlist)))
    try:
        sessions = _completed_sessions(cal, asof, lookback_sessions)
    except CalendarOutOfRange:
        sessions = []
    rows: list[EligibilityRow] = []
    for sym in sorted(syms):
        reasons: list[str] = []
        if not sym or sym[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(sym) > 10 or not set(sym) <= _SYMBOL_OK:
            rows.append(EligibilityRow(sym, False, (EligibilityReason.INVALID_SYMBOL.value,), None, None, 0, None, None, ()))
            continue
        if sym in deny:
            reasons.append(EligibilityReason.DENYLISTED_LEVERAGED_INVERSE.value)
        if sym not in allow:
            reasons.append(EligibilityReason.NOT_ALLOWLISTED.value)
        per: dict[date, Decimal] = {}
        last_close: Decimal | None = None
        row_feed: str | None = None
        source: str | None = None
        versions: set[str] = set()
        if sessions:
            per, last_close, row_feed, source, versions = _session_stats(store, sym, sessions, cal, asof, feed)
        avg: Decimal | None = None
        if last_close is None:
            reasons.append(EligibilityReason.NO_DATA.value)
        else:
            if len(per) < min_sessions:
                reasons.append(EligibilityReason.INSUFFICIENT_HISTORY.value)
            if sessions and sessions[-1] not in per:  # no data for the latest completed session: stale, fail closed
                reasons.append(EligibilityReason.STALE_DATA.value)
            if last_close < policy.min_price:
                reasons.append(EligibilityReason.BELOW_MIN_PRICE.value)
            if per:
                avg = (sum(per.values(), Decimal(0)) / Decimal(len(per))).quantize(Decimal("0.01"))
                if avg < min_avg_dollar_volume:
                    reasons.append(EligibilityReason.BELOW_MIN_DOLLAR_VOLUME.value)
        rows.append(EligibilityRow(sym, not reasons, tuple(reasons), last_close, avg, len(per), row_feed, source,
                                   tuple(sorted(versions))))
    params = {
        "min_price": format(policy.min_price, "f"), "min_avg_dollar_volume": format(min_avg_dollar_volume, "f"),
        "lookback_sessions": lookback_sessions, "min_sessions": min_sessions,
        "feed": feed.value if feed is not None else None,
        "sessions": [d.isoformat() for d in sessions],
        "candidates": "policy_allowlist" if candidates is None else sorted(syms),
    }
    content = {"asof": iso(asof), "policy_version": policy.version, "params": params, "rows": [r.as_dict() for r in rows]}
    snapshot_id = "univ_" + sha256_hex(canonical_json(content))[:16]
    return UniverseSnapshot(snapshot_id=snapshot_id, asof_ts=asof, asof_date=cal.ny_date(asof),
                            policy_version=policy.version, params=params, rows=tuple(rows),
                            created_at=created_at or datetime.now(UTC))


def persist_universe_snapshot(snapshot: UniverseSnapshot, *, store: MarketDataStore, db: Database | None = None) -> None:
    """Persist to DuckDB (per-symbol rows) and, when given, the SQLite ``md_universe_snapshots`` table (idempotent)."""
    store.write_universe_rows([
        {"snapshot_id": snapshot.snapshot_id, "asof_ts": snapshot.asof_ts, "asof_date": snapshot.asof_date,
         "symbol": r.symbol, "eligible": r.eligible, "reasons": json.dumps(list(r.reasons)), "last_close": r.last_close,
         "avg_dollar_volume": r.avg_dollar_volume, "sessions_observed": r.sessions_observed, "feed": r.feed,
         "dataset_versions": json.dumps(list(r.dataset_versions)), "policy_version": snapshot.policy_version,
         "created_at": snapshot.created_at}
        for r in snapshot.rows
    ])
    if db is None:
        return
    payload = canonical_json(snapshot.as_dict())

    def _w(conn: Any) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO md_universe_snapshots(snapshot_id, asof_date, asof_ts, policy_version, feed, "
            "dataset_versions, eligible_symbols, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (snapshot.snapshot_id, snapshot.asof_date.isoformat(), _fixed_iso(snapshot.asof_ts), snapshot.policy_version,
             ",".join(snapshot.feeds) or None, json.dumps(snapshot.dataset_versions),
             json.dumps(snapshot.eligible_symbols), payload, utc_iso(snapshot.created_at)),
        )

    db.write(_w)


def latest_universe_snapshot(db: Database, *, asof: datetime | None = None) -> dict[str, Any] | None:
    """Most recent persisted snapshot with ``asof_ts <= asof`` (PIT lookup); payload as a dict."""
    if asof is None:
        row = db.query_one("SELECT payload_json FROM md_universe_snapshots ORDER BY asof_ts DESC, created_at DESC LIMIT 1")
    else:
        row = db.query_one(
            "SELECT payload_json FROM md_universe_snapshots WHERE asof_ts <= ? ORDER BY asof_ts DESC, created_at DESC "
            "LIMIT 1", (_fixed_iso(asof),))
    return json.loads(row["payload_json"]) if row else None


def eligible_at(db: Database, asof: datetime) -> list[str]:
    """Eligible symbols of the latest snapshot at/before ``asof``; empty (fail closed) when none exists."""
    snap = latest_universe_snapshot(db, asof=asof)
    return list(snap["eligible_symbols"]) if snap else []


def stale_snapshot(snapshot: dict[str, Any] | None, asof: datetime, max_age: timedelta = timedelta(days=3)) -> bool:
    if snapshot is None:
        return True
    ts = datetime.fromisoformat(str(snapshot["asof_ts"]).replace("Z", "+00:00"))
    return to_utc(asof) - ts > max_age
