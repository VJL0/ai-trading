"""Entries lock: the single answer to "may a new order be sent right now?" (spec invariant 4, §7, §11).

Entries are locked whenever *any* of these holds (fail closed):

* an operator/system halt is active (``trading_halts``; cleared only by the operator, after a CLEAN
  broker reconciliation);
* a system-state flag is active: ``startup_unreconciled``, ``broker_disconnected``,
  ``unknown_orders_present``, ``reconciliation_mismatch``, ``engine_heartbeat_stale``,
  ``ledger_anomaly``, ``process_suspended``. Every flag is reconciliation-gated: it is cleared only by a
  persisted CLEAN reconciliation run that *started after* the flag was (last) raised (epoch rule);
* an order is in ``UNKNOWN`` state (defence in depth, independent of the flag);
* no CLEAN broker reconciliation has ever been recorded;
* no CLEAN broker reconciliation has completed *in this process* yet (process-start guard: a restarted
  process is locked from the moment its ledger ``Database`` is opened until its own startup
  reconciliation is CLEAN, even before ``startup_recovery`` gets to raise ``startup_unreconciled``);
* the mode cannot place orders (RESEARCH_ONLY, or a live mode in a build where live is compiled off).

Flags live in ``system_state`` under ``entries_flag:<name>``; a monotonically increasing epoch
(``entries_lock_epoch``) is bumped on every raise so a reconciliation that began before a flag was
raised can never clear it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from packages.contracts.enums import ReconciliationStatus, TradingMode
from packages.contracts.models import ReconciliationReport
from packages.core import modes
from packages.core.audit import add_halt, append_audit, clear_halt, set_state
from packages.core.clock import Clock, SystemClock
from packages.core.db import Database

from .errors import UnlockDenied
from .timeutil import iso, parse_iso

FLAG_PREFIX = "entries_flag:"
EPOCH_KEY = "entries_lock_epoch"


class Flag(StrEnum):
    STARTUP_UNRECONCILED = "startup_unreconciled"
    BROKER_DISCONNECTED = "broker_disconnected"
    UNKNOWN_ORDERS_PRESENT = "unknown_orders_present"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    ENGINE_HEARTBEAT_STALE = "engine_heartbeat_stale"
    LEDGER_ANOMALY = "ledger_anomaly"
    PROCESS_SUSPENDED = "process_suspended"


RECONCILIATION_GATED: frozenset[Flag] = frozenset(Flag)
"""Every flag needs a CLEAN broker-authoritative reconciliation to clear."""

PROCESS_GUARD_REASON = "startup reconciliation pending: no CLEAN broker reconciliation in this process yet"

# Process-start guard, keyed by the open ``Database`` object (one per ledger file per process). Every
# ``EntriesLock`` on the same Database shares it; a new Database (a restarted process) starts locked.
_process_reconciled: weakref.WeakKeyDictionary[Database, str] = weakref.WeakKeyDictionary()
_guard_lock = threading.Lock()


def process_reconciled_run(db: Database) -> str | None:
    """Run id of the CLEAN reconciliation that released the process-start guard, or None."""
    with _guard_lock:
        return _process_reconciled.get(db)


@dataclass(frozen=True)
class FlagState:
    name: str
    reason: str
    source: str
    set_at: datetime
    epoch: int


# ---------------------------------------------------------------------------- in-transaction helpers


def _state_json(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value_json FROM system_state WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def current_epoch_conn(conn: sqlite3.Connection) -> int:
    value = _state_json(conn, EPOCH_KEY)
    return int(value) if value is not None else 0


def set_flag_conn(
    conn: sqlite3.Connection,
    flag: Flag | str,
    *,
    reason: str,
    source: str,
    mode: str,
    at: datetime,
) -> int:
    """Raise (or re-raise) a flag inside an open writer transaction. Returns the new epoch."""
    name = Flag(flag).value
    epoch = current_epoch_conn(conn) + 1
    set_state(conn, EPOCH_KEY, epoch)
    existing = _state_json(conn, FLAG_PREFIX + name)
    newly = not (isinstance(existing, dict) and existing.get("active"))
    value = {
        "active": True,
        "reason": reason[:500],
        "source": source,
        "set_at": iso(at) if newly else existing.get("set_at", iso(at)),  # type: ignore[union-attr]
        "last_raised_at": iso(at),
        "epoch": epoch,
    }
    set_state(conn, FLAG_PREFIX + name, value)
    if newly:
        append_audit(
            conn, event_type="ENTRIES_FLAG_SET", source=source, mode=mode,
            payload={"flag": name, "reason": reason[:500], "epoch": epoch}, ts=at,
        )
    return epoch


def active_flags_conn(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in conn.execute("SELECT key, value_json FROM system_state WHERE key LIKE ?", (FLAG_PREFIX + "%",)):
        value = json.loads(row[1])
        if isinstance(value, dict) and value.get("active"):
            out[str(row[0])[len(FLAG_PREFIX):]] = value
    return out


def _clear_flag_conn(conn: sqlite3.Connection, name: str, *, by: str, run_id: str, mode: str, at: datetime) -> None:
    value = _state_json(conn, FLAG_PREFIX + name) or {}
    value.update({"active": False, "cleared_at": iso(at), "cleared_by": by, "cleared_by_run": run_id})
    set_state(conn, FLAG_PREFIX + name, value)
    append_audit(
        conn, event_type="ENTRIES_FLAG_CLEARED", source=by, mode=mode,
        payload={"flag": name, "reconciliation_run_id": run_id}, ts=at,
    )


def clear_flags_after_clean_conn(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    observed_epoch: int,
    mode: str,
    at: datetime,
    keep: frozenset[str] = frozenset(),
    by: str = "reconciler",
) -> list[str]:
    """Clear reconciliation-gated flags raised at or before ``observed_epoch`` (CLEAN runs only)."""
    run = conn.execute("SELECT status FROM reconciliation_runs WHERE run_id=?", (run_id,)).fetchone()
    if run is None or run[0] != ReconciliationStatus.CLEAN.value:
        raise UnlockDenied(f"reconciliation run {run_id} is not a persisted CLEAN run")
    cleared: list[str] = []
    for name, value in active_flags_conn(conn).items():
        if name in keep or name not in {f.value for f in RECONCILIATION_GATED}:
            continue
        if int(value.get("epoch", 0)) > observed_epoch:
            continue  # raised while the reconciliation was running: must survive
        _clear_flag_conn(conn, name, by=by, run_id=run_id, mode=mode, at=at)
        cleared.append(name)
    return cleared


def has_clean_reconciliation_conn(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM reconciliation_runs WHERE status=? LIMIT 1", (ReconciliationStatus.CLEAN.value,)
    ).fetchone()
    return row is not None


def last_reconciliation_conn(conn: sqlite3.Connection) -> tuple[str, str] | None:
    # Insertion order (rowid) is the truth for "most recent": simulated/manual clocks can tie or run backwards.
    row = conn.execute("SELECT run_id, status FROM reconciliation_runs ORDER BY rowid DESC LIMIT 1").fetchone()
    return (str(row[0]), str(row[1])) if row else None


# ---------------------------------------------------------------------------- service object


class EntriesLock:
    def __init__(self, db: Database, mode: TradingMode, *, clock: Clock | None = None) -> None:
        self.db = db
        self.mode = mode
        self.clock: Clock = clock or SystemClock()

    # ------------------------------------------------------------------ reads
    def is_locked(self) -> tuple[bool, list[str]]:
        """``(locked, human-readable reasons)``. Any error while reading state => locked."""
        try:
            reasons = self.db.read(self.reasons_conn)
        except Exception as exc:  # fail closed: unknown state blocks entries
            return True, [f"entries-lock state unreadable: {type(exc).__name__}"]
        return bool(reasons), reasons

    def reasons_conn(self, conn: sqlite3.Connection, *, assume_process_reconciled: bool = False) -> list[str]:
        """Lock reasons computed on an open connection (also usable inside a writer transaction).

        ``assume_process_reconciled`` is used only by the reconciler while it persists a CLEAN run (which is
        about to release the process-start guard).
        """
        reasons: list[str] = []
        if not assume_process_reconciled and not self.process_reconciled():
            reasons.append(PROCESS_GUARD_REASON)
        if self.mode == TradingMode.RESEARCH_ONLY:
            reasons.append("mode RESEARCH_ONLY never places orders")
        if self.mode in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE) and not modes.LIVE_EXECUTION_COMPILED:
            reasons.append(f"mode {self.mode.value}: live execution is compiled off in this build")
        for row in conn.execute(
            "SELECT halt_id, reason, source FROM trading_halts WHERE cleared_at IS NULL ORDER BY created_at"
        ):
            reasons.append(f"halt {row[0]}: {row[1]} (source={row[2]})")
        for name, value in sorted(active_flags_conn(conn).items()):
            reasons.append(f"{name}: {value.get('reason', '')}")
        # Defence in depth: an order with unknown outcome locks entries even if its flag were cleared.
        unknown = conn.execute("SELECT client_order_id FROM orders WHERE state='UNKNOWN' ORDER BY created_at LIMIT 5").fetchall()
        if unknown:
            reasons.append("orders with unknown outcome: " + ", ".join(str(r[0]) for r in unknown))
        if not has_clean_reconciliation_conn(conn):
            reasons.append("no CLEAN broker reconciliation has completed yet")
        return reasons

    def flags(self) -> dict[str, FlagState]:
        raw = self.db.read(active_flags_conn)
        return {
            name: FlagState(
                name=name, reason=str(v.get("reason", "")), source=str(v.get("source", "")),
                set_at=parse_iso(str(v.get("set_at"))) if v.get("set_at") else datetime.min.replace(tzinfo=UTC),
                epoch=int(v.get("epoch", 0)),
            )
            for name, v in raw.items()
        }

    def is_flag_active(self, flag: Flag | str) -> bool:
        return Flag(flag).value in self.db.read(active_flags_conn)

    def epoch(self) -> int:
        return self.db.read(current_epoch_conn)

    def process_reconciled(self) -> bool:
        """True once a CLEAN broker reconciliation has completed in this process (for this Database)."""
        return process_reconciled_run(self.db) is not None

    def mark_process_reconciled(self, report: ReconciliationReport) -> bool:
        """Release the process-start guard. Only a *persisted* CLEAN reconciliation run qualifies."""
        if report.status != ReconciliationStatus.CLEAN:
            return False
        row = self.db.query_one("SELECT status FROM reconciliation_runs WHERE run_id=?", (report.run_id,))
        if row is None or row["status"] != ReconciliationStatus.CLEAN.value:
            raise UnlockDenied(f"reconciliation {report.run_id} is not a persisted CLEAN run")
        with _guard_lock:
            _process_reconciled.setdefault(self.db, report.run_id)
        return True

    def halts(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM trading_halts WHERE cleared_at IS NULL ORDER BY created_at")

    # ------------------------------------------------------------------ writes
    def lock(self, reason: str, source: str) -> str:
        """Add a stop-new-orders halt (always allowed: fail-safe direction). Returns the halt id."""
        return self.db.write(lambda c: add_halt(c, reason=reason, source=source, mode=self.mode.value))

    def set_flag(self, flag: Flag | str, reason: str, source: str) -> int:
        at = self.clock.now()
        return self.db.write(
            lambda c: set_flag_conn(c, flag, reason=reason, source=source, mode=self.mode.value, at=at)
        )

    def clear_flag(self, flag: Flag | str, *, report: ReconciliationReport, by: str = "reconciler") -> None:
        """Clear one flag. Requires a persisted CLEAN reconciliation that started after the flag was raised."""
        name = Flag(flag).value
        if report.status != ReconciliationStatus.CLEAN:
            raise UnlockDenied(f"cannot clear {name}: reconciliation {report.run_id} is {report.status.value}, not CLEAN")
        at = self.clock.now()

        def _w(conn: sqlite3.Connection) -> None:
            run = conn.execute("SELECT status FROM reconciliation_runs WHERE run_id=?", (report.run_id,)).fetchone()
            if run is None or run[0] != ReconciliationStatus.CLEAN.value:
                raise UnlockDenied(f"cannot clear {name}: reconciliation {report.run_id} is not a persisted CLEAN run")
            meta = conn.execute(
                "SELECT observed_epoch FROM reconciliation_meta WHERE run_id=?", (report.run_id,)
            ).fetchone()
            observed = int(meta[0]) if meta else -1
            active = active_flags_conn(conn).get(name)
            if active is None:
                return
            if int(active.get("epoch", 0)) > observed:
                raise UnlockDenied(f"cannot clear {name}: it was raised after reconciliation {report.run_id} started")
            _clear_flag_conn(conn, name, by=by, run_id=report.run_id, mode=self.mode.value, at=at)

        self.db.write(_w)

    def clear_halt(self, halt_id: str, *, by: str, reason: str) -> bool:
        """Operator clearance of a halt. The caller (API) must already have verified the operator token.

        Refused unless the most recent reconciliation run is CLEAN.
        """

        def _w(conn: sqlite3.Connection) -> bool:
            last = last_reconciliation_conn(conn)
            if last is None or last[1] != ReconciliationStatus.CLEAN.value:
                raise UnlockDenied("clearing a halt requires the most recent broker reconciliation to be CLEAN")
            return clear_halt(conn, halt_id, by=by, reason=reason, mode=self.mode.value)

        return self.db.write(_w)

    # ------------------------------------------------------------------ watchdogs
    def observe_heartbeat(self, source: str, last_seen: datetime | None, *, max_age_s: float) -> bool:
        """Raise ``engine_heartbeat_stale`` when ``last_seen`` is missing/older than ``max_age_s``. Returns fresh?"""
        now = self.clock.now()
        if last_seen is not None and 0 <= (now - last_seen).total_seconds() <= max_age_s:
            return True
        age = "never" if last_seen is None else f"{(now - last_seen).total_seconds():.0f}s ago"
        self.set_flag(Flag.ENGINE_HEARTBEAT_STALE, f"{source} heartbeat stale (last {age}, max {max_age_s:.0f}s)", source)
        return False


class SuspensionWatchdog:
    """Detects Mac sleep / process suspension as a jump in the clock between ticks.

    The local service does not run while the Mac sleeps; broker-held orders may have changed meanwhile,
    so a jump larger than ``max_gap_s`` raises ``process_suspended`` (entries locked until a CLEAN
    reconciliation).
    """

    def __init__(self, entries: EntriesLock, clock: Clock, *, max_gap_s: float = 30.0) -> None:
        self.entries = entries
        self.clock = clock
        self.max_gap_s = max_gap_s
        self._last: datetime | None = None

    def tick(self) -> bool:
        """Returns True when a suspension was detected on this tick."""
        now = self.clock.now()
        last, self._last = self._last, now
        if last is None:
            return False
        gap = (now - last).total_seconds()
        if gap > self.max_gap_s or gap < 0:
            self.entries.set_flag(
                Flag.PROCESS_SUSPENDED,
                f"clock jumped {gap:.0f}s between watchdog ticks (sleep/suspend?); broker state must be reconciled",
                "watchdog",
            )
            return True
        return False
