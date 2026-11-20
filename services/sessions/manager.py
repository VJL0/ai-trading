"""Autonomous PAPER trading sessions, recurring schedules and restart recovery.

The manager lives in the local API process (``services.api.runtime.Runtime`` builds it and runs ``run()`` as a
background task), so a session keeps running when the AI chat that started it disconnects, ends or hits a usage
limit. It never calls a model itself: every cycle is ``runtime.run_cycle`` (quant engine -> optional in-app model
review -> independent risk engine -> signed intent -> paper gateway -> reconciliation). An AI failure inside a cycle
only degrades that cycle's GPT_REVIEW; the slot is re-run quant-only and AI review backs off.

Safety properties:

* sessions start only in PAPER/REPLAY with a non-LIVE execution label; symbols must be in the active risk policy
  allowlist and the configured universe; caps: duration 1..480 min, cadence 1..60 min, one active session (also a
  partial UNIQUE index in SQLite);
* stopping is always allowed (fail-safe direction); an in-flight cycle is never cancelled mid-order, the session
  goes STOPPING and becomes STOPPED as soon as that cycle returns;
* cycles run only inside XNYS regular hours (exchange_calendars; unknown -> closed), at most one per cadence slot
  (no catch-up bursts after a sleep), and the risk engine stays authoritative for every order;
* a scheduled start happens at most once per schedule and trading day (primary key of
  ``session_schedule_runs``), also across restarts; a window that passed while the service was down is MISSED and
  never run late;
* on restart ``recover(clean=...)`` resumes a RUNNING session only after a CLEAN broker-authoritative startup
  reconciliation and only inside its window; otherwise it becomes INTERRUPTED with the reason;
* every state change is persisted through the single SQLite writer, audited (hash chain), written to the activity
  log (kind ``session``) and published on the bus (``session`` / ``schedule`` events).
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Final, Protocol

from packages.contracts import RiskPolicy
from packages.contracts.enums import ExecutionLabel, TradingMode
from packages.core.audit import append_audit, record_failure
from packages.core.canonical import canonical_json
from packages.core.ids import new_id
from packages.core.logging import log_event

from . import calendar as xcal
from .calendar import et_label, iso, parse_utc
from .errors import SessionError
from .explain import explain_decisions
from .report import build_report
from .schedules import (
    Occurrence,
    next_occurrence,
    normalize_weekdays,
    pending_occurrences,
    validate_start_time,
    validate_window,
)

log = logging.getLogger("lat.sessions")

ACTIVE_STATES: Final[tuple[str, ...]] = ("RUNNING", "STOPPING")
TERMINAL_STATES: Final[tuple[str, ...]] = ("COMPLETED", "STOPPED", "INTERRUPTED", "FAILED")
SESSION_MODES: Final[frozenset[TradingMode]] = frozenset({TradingMode.PAPER, TradingMode.REPLAY})
DURATION_RANGE: Final = (1, 480)
CADENCE_RANGE: Final = (1, 60)
MAX_SYMBOLS: Final = 10
MAX_SCHEDULES: Final = 20
MAX_CONSECUTIVE_FAILURES: Final = 3
AI_BACKOFF: Final = timedelta(minutes=30)
AI_UNAVAILABLE_STATUSES: Final[frozenset[str]] = frozenset({"ERROR", "TIMEOUT"})
RECOVER_TIMEOUT_S: Final = 120.0
EMIT_TIMEOUT_S: Final = 5.0
_SYMBOL_RE: Final = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_ACTOR_RE: Final = re.compile(r"[^A-Za-z0-9_.:@-]+")


class SessionRuntime(Protocol):
    """What the manager needs from ``services.api.runtime.Runtime`` (tests pass a fake)."""

    @property
    def settings(self) -> Any: ...

    @property
    def db(self) -> Any: ...

    @property
    def bus(self) -> Any: ...

    @property
    def activity(self) -> Any: ...

    @property
    def execution_label(self) -> ExecutionLabel: ...

    def now(self) -> datetime: ...

    def entries_status(self) -> tuple[bool, list[str]]: ...

    async def reconcile(self) -> Any: ...

    async def run_cycle(self, *, symbols: list[str] | None, use_ai: bool, actor: str,
                        session_id: str | None = None) -> dict[str, Any]: ...


def _actor(text: Any) -> str:
    s = _ACTOR_RE.sub("-", str(text or "system")).strip("-")
    return (s or "system")[:80]


def _text(text: Any, n: int = 300) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _int_in(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionError(422, f"{name} must be a whole number between {lo} and {hi}")
    if not lo <= value <= hi:
        raise SessionError(422, f"{name} must be between {lo} and {hi} (got {value})")
    return value


def _row_out(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": r["session_id"], "state": r["state"], "started_at": r["started_at"], "ends_at": r["ends_at"],
        "ended_at": r["ended_at"], "symbols": json.loads(r["symbols_json"]), "cadence_minutes": r["cadence_minutes"],
        "use_ai": bool(r["use_ai"]), "requested_by": r["requested_by"], "cycles_run": r["cycles_run"],
        "trades": r["trades"], "stop_reason": r["stop_reason"], "schedule_id": r["schedule_id"],
        "duration_minutes": r["duration_minutes"], "created_at": r["created_at"], "next_cycle_at": r["next_cycle_at"],
        "last_cycle_at": r["last_cycle_at"], "execution_label": r["execution_label"],
        "ai_suspended_until": r["ai_suspended_until"],
    }


class SessionManager:
    """See module docstring. Sync methods are thread-safe (the API calls them from worker threads)."""

    def __init__(self, runtime: SessionRuntime, *, tick_s: float = 1.0,
                 max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
                 recover_timeout_s: float = RECOVER_TIMEOUT_S, ai_backoff: timedelta = AI_BACKOFF) -> None:
        self.rt = runtime
        self.tick_s = tick_s
        self.max_consecutive_failures = max_consecutive_failures
        self.recover_timeout_s = recover_timeout_s
        self.ai_backoff = ai_backoff
        self.process_id = new_id("proc")
        self._lock = threading.Lock()
        self._cycle_session: str | None = None
        self._recovered = False
        self._created_mono = time.monotonic()
        self._closed_notified: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._running = False
        self._last_tick_at: datetime | None = None
        self._force_schedule_scan = True

    # ================================================================== reads
    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> builtins.list[dict[str, Any]]:
        return self.rt.db.query(sql, params)

    def _get(self, session_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM trading_sessions WHERE session_id=?", (session_id,))
        return rows[0] if rows else None

    def _active_row(self) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM trading_sessions WHERE state IN ('RUNNING','STOPPING')")
        return rows[0] if rows else None

    def active(self) -> dict[str, Any] | None:
        row = self._active_row()
        return None if row is None else _row_out(row)

    def get(self, session_id: str) -> dict[str, Any] | None:
        row = self._get(session_id)
        return None if row is None else _row_out(row)

    def list(self, limit: int = 20) -> builtins.list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        rows = self._query("SELECT * FROM trading_sessions ORDER BY created_at DESC, session_id DESC LIMIT ?", (limit,))
        return [_row_out(r) for r in rows]

    def cycles(self, session_id: str) -> builtins.list[dict[str, Any]]:
        rows = self._query("SELECT * FROM session_cycles WHERE session_id=? ORDER BY cycle_no", (session_id,))
        for r in rows:
            r["lock_reasons"] = json.loads(r.pop("lock_reasons_json") or "[]")
            r["use_ai"] = bool(r["use_ai"])
            r["entries_locked"] = None if r["entries_locked"] is None else bool(r["entries_locked"])
        return rows

    def latest_session_id(self) -> str | None:
        act = self._active_row()
        if act is not None:
            return str(act["session_id"])
        rows = self._query("SELECT session_id FROM trading_sessions ORDER BY created_at DESC, session_id DESC LIMIT 1")
        return str(rows[0]["session_id"]) if rows else None

    def report(self, session_id: str | None = None) -> dict[str, Any] | None:
        sid = session_id or self.latest_session_id()
        if sid is None:
            return None
        row = self._get(sid)
        if row is None:
            return None
        label = self.rt.execution_label.value
        return build_report(self.rt.db, _row_out(row), self.cycles(sid), now=self.rt.now(), label=label,
                            is_simulated=label == ExecutionLabel.SIMULATED.value)

    def explain(self, limit: int = 10) -> dict[str, Any]:
        try:
            entries: tuple[bool, builtins.list[str]] | None = self.rt.entries_status()
        except Exception:
            entries = None
        return explain_decisions(self.rt.db, limit, entries=entries, now=self.rt.now())

    def scheduler_status(self) -> dict[str, Any]:
        now = self.rt.now()
        nxt = None if xcal.is_market_open(now) else xcal.next_open(now)
        act = self._active_row()
        return {"loop_running": self._running, "recovered": self._recovered, "last_tick_at": iso(self._last_tick_at),
                "market_open": xcal.is_market_open(now), "next_open": iso(nxt),
                "active_session_id": act["session_id"] if act else None, "process_id": self.process_id}

    # ================================================================== validation
    def _require_mode(self) -> None:
        mode = self.rt.settings.mode
        if mode not in SESSION_MODES or self.rt.execution_label == ExecutionLabel.LIVE:
            raise SessionError(403, f"sessions are PAPER/REPLAY only (mode {getattr(mode, 'value', mode)}, label "
                                    f"{self.rt.execution_label.value}); live trading is impossible in this build")

    def _policy(self) -> RiskPolicy:
        pol = getattr(self.rt, "policy", None)
        if isinstance(pol, RiskPolicy):
            return pol
        rows = self._query("SELECT policy_json FROM risk_policies WHERE is_active=1")
        if not rows:
            raise SessionError(503, "no active risk policy (fail closed)")
        return RiskPolicy.model_validate_json(rows[0]["policy_json"])

    def _resolve_symbols(self, symbols: Sequence[str] | None) -> builtins.list[str]:
        universe = [str(s) for s in self.rt.settings.symbols]
        if symbols is None:
            syms = universe
        else:
            if isinstance(symbols, str) or not symbols:
                raise SessionError(422, "symbols must be a non-empty list such as ['SPY'] (or null for all)")
            syms = builtins.list(dict.fromkeys(str(s).strip().upper() for s in symbols))
        if not syms or len(syms) > MAX_SYMBOLS:
            raise SessionError(422, f"a session trades 1..{MAX_SYMBOLS} symbols")
        pol = self._policy()
        bad = [s for s in syms if not _SYMBOL_RE.fullmatch(s)]
        if bad:
            raise SessionError(422, f"invalid symbols: {bad}")
        outside = [s for s in syms if s not in pol.symbol_allowlist or s in pol.symbol_denylist]
        if outside:
            raise SessionError(422, f"symbols not in the active risk policy allowlist: {outside}")
        unknown = [s for s in syms if s not in universe]
        if unknown:
            raise SessionError(422, f"symbols outside the configured universe {universe}: {unknown}")
        return syms

    def _sim_finished(self) -> bool:
        sim = getattr(self.rt, "sim_clock", None)
        return bool(getattr(sim, "finished", False)) if sim is not None else False

    # ================================================================== emission
    def _on_loop(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the event-loop thread (the bus is an asyncio structure); wait briefly, never deadlock."""
        loop = self._loop
        if loop is None or not loop.is_running() or threading.get_ident() == self._loop_thread:
            fn()
            return
        done = threading.Event()

        def wrapper() -> None:
            try:
                fn()
            finally:
                done.set()

        try:
            loop.call_soon_threadsafe(wrapper)
        except RuntimeError:  # loop closed meanwhile
            fn()
            return
        if not done.wait(EMIT_TIMEOUT_S):
            log_event(log, "session event emission delayed (event loop busy)", logging.WARNING)

    def _emit(self, message: str, *, actor: str, session: dict[str, Any] | None = None, kind: str = "session",
              event: str = "session", payload: Any = None, detail: dict[str, Any] | None = None) -> None:
        sid = session["session_id"] if session else None

        def go() -> None:
            try:
                self.rt.activity.activity(kind, message, actor=actor, session_id=sid, detail=detail or {})
            except Exception as exc:
                log_event(log, "activity write failed", logging.WARNING, error=type(exc).__name__)
            try:
                self.rt.bus.publish(event, payload if payload is not None else session)
            except Exception as exc:
                log_event(log, "bus publish failed", logging.WARNING, error=type(exc).__name__)

        self._on_loop(go)

    def _audit(self, conn: sqlite3.Connection, event_type: str, source: str, payload: dict[str, Any]) -> None:
        append_audit(conn, event_type=event_type, source=source[:80], mode=self.rt.settings.mode.value,
                     payload=payload, policy_version=getattr(getattr(self.rt, "policy", None), "version", None))

    def _failure(self, component: str, exc: BaseException) -> None:
        kind, detail = type(exc).__name__, str(exc)[:1000]
        log_event(log, "session manager error", logging.ERROR, component=component, error=kind)
        with contextlib.suppress(Exception):
            self.rt.db.write(lambda c: record_failure(c, component=f"sessions:{component}", kind=kind, detail=detail))

    # ================================================================== lifecycle: start
    def start(self, duration_minutes: int, *, symbols: Sequence[str] | None = None, cadence_minutes: int = 5,
              use_ai: bool = True, requested_by: str) -> dict[str, Any]:
        """Start a PAPER session now. Raises ``SessionError`` (403/409/422) when refused."""
        return self._start(duration_minutes, symbols=symbols, cadence_minutes=cadence_minutes, use_ai=use_ai,
                           requested_by=requested_by)

    def _start(self, duration_minutes: int, *, symbols: Sequence[str] | None, cadence_minutes: int, use_ai: bool,
               requested_by: str, schedule_id: str | None = None, occurrence: Occurrence | None = None) -> dict[str, Any]:
        self._require_mode()
        duration = _int_in("duration_minutes", duration_minutes, *DURATION_RANGE)
        cadence = _int_in("cadence_minutes", cadence_minutes, *CADENCE_RANGE)
        if not isinstance(use_ai, bool):
            raise SessionError(422, "use_ai must be true or false")
        syms = self._resolve_symbols(symbols)
        who = _actor(requested_by)
        now = self.rt.now()
        if self._sim_finished():
            raise SessionError(409, "the simulated replay has reached the end of the synthetic data; restart the API "
                                    "to replay again")
        ends = occurrence.end if occurrence is not None else now + timedelta(minutes=duration)
        if ends <= now:
            raise SessionError(409, "the session window has already passed")
        if not xcal.regular_minutes_between(now, ends):
            nxt = xcal.next_open(now)
            raise SessionError(409, "XNYS is closed for the whole requested window (sessions trade regular hours "
                                    f"only); next regular open {et_label(nxt) if nxt else 'unknown'}. Use "
                                    "schedule_session to run it then.")
        sid = new_id("sess")
        label = self.rt.execution_label.value
        row = {"session_id": sid, "state": "RUNNING", "created_at": iso(now), "started_at": iso(now),
               "ends_at": iso(ends), "symbols_json": json.dumps(syms), "cadence_minutes": cadence,
               "duration_minutes": duration, "use_ai": int(use_ai), "requested_by": who, "schedule_id": schedule_id,
               "mode": self.rt.settings.mode.value, "execution_label": label, "owner_process": self.process_id,
               "next_cycle_at": iso(now), "updated_at": iso(now)}

        def _w(conn: sqlite3.Connection) -> None:
            if schedule_id is not None and occurrence is not None:
                en = conn.execute("SELECT enabled FROM session_schedules WHERE schedule_id=?", (schedule_id,)).fetchone()
                if en is None or not en[0]:
                    raise SessionError(409, f"schedule {schedule_id} is cancelled")
                try:
                    conn.execute("INSERT INTO session_schedule_runs(schedule_id, run_date, due_at, window_end, status,"
                                 " session_id, detail, recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                                 (schedule_id, occurrence.run_date.isoformat(), iso(occurrence.start),
                                  iso(occurrence.end), "STARTED", sid, f"started at {iso(now)}", iso(now)))
                except sqlite3.IntegrityError:
                    raise SessionError(409, f"schedule {schedule_id} already handled {occurrence.run_date}") from None
            try:
                conn.execute(f"INSERT INTO trading_sessions({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",  # noqa: S608
                             tuple(row.values()))
            except sqlite3.IntegrityError:
                act = conn.execute("SELECT session_id FROM trading_sessions WHERE state IN ('RUNNING','STOPPING')"
                                   ).fetchone()
                raise SessionError(409, f"a paper session is already active ({act[0] if act else 'unknown'}); stop it "
                                        "first or wait for it to end") from None
            self._audit(conn, "SESSION_STARTED", who, {"session_id": sid, "symbols": syms, "duration_minutes": duration,
                                                       "cadence_minutes": cadence, "use_ai": use_ai,
                                                       "ends_at": iso(ends), "schedule_id": schedule_id,
                                                       "execution_label": label})

        self.rt.db.write(_w)
        session = self.get(sid)
        assert session is not None
        late = ""
        if occurrence is not None and now > occurrence.start + timedelta(minutes=1):
            late = f" (started {int((now - occurrence.start).total_seconds() // 60)} min after its scheduled time; it " \
                   "still ends at the scheduled time)"
        self._emit(f"Started a {label} paper session on {', '.join(syms)}: decision cycle every {cadence} min, AI review "
                   f"{'on' if use_ai else 'off'}, ends {et_label(ends)}{late}. Requested by {who}.", actor=who,
                   session=session, detail={"event": "started", "schedule_id": schedule_id})
        return session

    # ================================================================== lifecycle: stop / finish
    def stop(self, session_id: str | None = None, *, reason: str, requested_by: str) -> dict[str, Any]:
        """Always allowed. Returns {"outcome": STOPPED|STOPPING|ALREADY_ENDED|NO_ACTIVE_SESSION, "session"}."""
        who = _actor(requested_by)
        why = _text(reason) or "stop requested"
        now = self.rt.now()
        with self._lock:
            row = self._get(session_id) if session_id else self._active_row()
            if row is None:
                if session_id:
                    raise SessionError(404, f"unknown session {session_id}")
                return {"outcome": "NO_ACTIVE_SESSION", "session": None}
            sid = str(row["session_id"])
            if row["state"] in TERMINAL_STATES:
                return {"outcome": "ALREADY_ENDED", "session": _row_out(row)}
            in_flight = self._cycle_session == sid
            if in_flight:
                target, frm = "STOPPING", ("RUNNING",)
            else:
                target, frm = "STOPPED", ACTIVE_STATES
            changed = self._transition(sid, target, from_states=frm, reason=f"stopped by {who}: {why}", now=now,
                                       actor=who, event_type="SESSION_STOP_REQUESTED" if in_flight else "SESSION_STOPPED")
        session = self.get(sid)
        if not changed:  # it was already STOPPING (in-flight) or finished concurrently
            return {"outcome": "STOPPING" if session and session["state"] == "STOPPING" else "ALREADY_ENDED",
                    "session": session}
        if target == "STOPPING":
            self._emit(f"Stop requested by {who}: {why}. The in-flight decision cycle finishes first (an order is never "
                       "cancelled mid-submission); the session then stops.", actor=who, session=session)
        else:
            self._emit(f"Session stopped by {who}: {why}. It ran {session['cycles_run'] if session else 0} decision "
                       "cycle(s).", actor=who, session=session, detail={"event": "stopped"})
        return {"outcome": target, "session": session}

    def _transition(self, sid: str, to_state: str, *, from_states: Sequence[str], reason: str | None, now: datetime,
                    actor: str, event_type: str, extra: dict[str, Any] | None = None) -> bool:
        ended = to_state in TERMINAL_STATES

        def _w(conn: sqlite3.Connection) -> bool:
            sets = ["state=?", "updated_at=?"]
            params: builtins.list[Any] = [to_state, iso(now)]
            if reason is not None:
                sets.append("stop_reason=?")
                params.append(reason[:500])
            if ended:
                sets += ["ended_at=?", "next_cycle_at=NULL"]
                params.append(iso(now))
            for k, v in (extra or {}).items():
                sets.append(f"{k}=?")
                params.append(v)
            ph = ",".join("?" for _ in from_states)
            cur = conn.execute(f"UPDATE trading_sessions SET {', '.join(sets)} WHERE session_id=? AND state IN ({ph})",  # noqa: S608
                               (*params, sid, *from_states))
            if cur.rowcount == 0:
                return False
            self._audit(conn, event_type, actor, {"session_id": sid, "state": to_state, "reason": reason})
            return True

        return bool(self.rt.db.write(_w))

    def _finish(self, sid: str, state: str, reason: str, *, actor: str = "system") -> bool:
        now = self.rt.now()
        changed = self._transition(sid, state, from_states=ACTIVE_STATES, reason=reason, now=now, actor=actor,
                                   event_type=f"SESSION_{state}")
        if changed:
            self._closed_notified.discard(sid)
            s = self.get(sid)
            n = s["cycles_run"] if s else 0
            t = s["trades"] if s else 0
            verb = {"COMPLETED": "completed", "STOPPED": "stopped", "FAILED": "failed", "INTERRUPTED": "was interrupted"}
            self._emit(f"Session {verb.get(state, state.lower())}: {reason}. It ran {n} decision cycle(s) and sent {t} "
                       "order(s) to the paper gateway.", actor=actor, session=s,
                       kind="error" if state == "FAILED" else "session", detail={"event": state.lower()})
        return changed

    async def _final_reconcile(self) -> None:
        try:
            await self.rt.reconcile()
        except Exception as exc:
            self._failure("final_reconcile", exc)

    # ================================================================== schedules
    def add_schedule(self, weekdays: Sequence[str], start_time_et: str, duration_minutes: int,
                     symbols: Sequence[str] | None = None, cadence_minutes: int = 5, use_ai: bool = True,
                     created_by: str = "system") -> dict[str, Any]:
        self._require_mode()
        days = normalize_weekdays(weekdays)
        start = validate_start_time(start_time_et)
        duration = _int_in("duration_minutes", duration_minutes, *DURATION_RANGE)
        cadence = _int_in("cadence_minutes", cadence_minutes, *CADENCE_RANGE)
        if not isinstance(use_ai, bool):
            raise SessionError(422, "use_ai must be true or false")
        syms = self._resolve_symbols(symbols)
        who = _actor(created_by)
        now = self.rt.now()
        validate_window(days, start, duration, today=xcal.et_date(now))
        existing = [s for s in self.list_schedules() if s["enabled"]]
        if len(existing) >= MAX_SCHEDULES:
            raise SessionError(409, f"at most {MAX_SCHEDULES} active schedules; cancel one first")
        a0 = _minutes(start)
        for s in existing:
            if set(s["weekdays"]) & set(days):
                b0 = _minutes(s["start_time_et"])
                if a0 < b0 + int(s["duration_minutes"]) and b0 < a0 + duration:
                    raise SessionError(409, f"overlaps schedule {s['schedule_id']} ({','.join(s['weekdays'])} "
                                            f"{s['start_time_et']} ET for {s['duration_minutes']} min); only one "
                                            "session can run at a time")
        nxt = next_occurrence(days, start, duration, created_at=now, now=now, recorded=())
        schedule_id = new_id("sched")

        def _w(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO session_schedules(schedule_id, weekdays_json, start_time_et, duration_minutes,"
                         " symbols_json, cadence_minutes, use_ai, enabled, next_run_at, created_by, created_at,"
                         " updated_at) VALUES (?,?,?,?,?,?,?,1,?,?,?,?)",
                         (schedule_id, json.dumps(days), start, duration, json.dumps(syms), cadence, int(use_ai),
                          iso(nxt.start) if nxt else None, who, iso(now), iso(now)))
            self._audit(conn, "SESSION_SCHEDULE_ADDED", who, {"schedule_id": schedule_id, "weekdays": days,
                                                              "start_time_et": start, "duration_minutes": duration,
                                                              "symbols": syms, "cadence_minutes": cadence,
                                                              "use_ai": use_ai})

        self.rt.db.write(_w)
        sch = self.get_schedule(schedule_id)
        assert sch is not None
        self._emit(f"Scheduled a {duration}-minute paper session on {', '.join(syms)} every {','.join(days)} at {start} ET "
                   f"(next: {et_label(nxt.start) if nxt else 'none within 3 weeks'}). Created by {who}.", actor=who,
                   event="schedule", payload=sch, detail={"event": "schedule_added", "schedule_id": schedule_id})
        return sch

    def _schedule_out(self, r: dict[str, Any], last: dict[str, Any] | None) -> dict[str, Any]:
        return {"schedule_id": r["schedule_id"], "weekdays": json.loads(r["weekdays_json"]),
                "start_time_et": r["start_time_et"], "duration_minutes": r["duration_minutes"],
                "symbols": json.loads(r["symbols_json"]), "cadence_minutes": r["cadence_minutes"],
                "use_ai": bool(r["use_ai"]), "enabled": bool(r["enabled"]), "next_run_at": r["next_run_at"],
                "created_by": r["created_by"], "created_at": r["created_at"], "cancelled_at": r["cancelled_at"],
                "last_run": None if last is None else {"run_date": last["run_date"], "status": last["status"],
                                                       "session_id": last["session_id"], "due_at": last["due_at"],
                                                       "detail": last["detail"]}}

    def _last_runs(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self._query("SELECT * FROM session_schedule_runs ORDER BY run_date"):
            out[str(r["schedule_id"])] = r
        return out

    def get_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM session_schedules WHERE schedule_id=?", (schedule_id,))
        if not rows:
            return None
        return self._schedule_out(rows[0], self._last_runs().get(schedule_id))

    def list_schedules(self) -> builtins.list[dict[str, Any]]:
        last = self._last_runs()
        rows = self._query("SELECT * FROM session_schedules ORDER BY enabled DESC, created_at DESC LIMIT 50")
        return [self._schedule_out(r, last.get(str(r["schedule_id"]))) for r in rows]

    def schedule_runs(self, schedule_id: str) -> builtins.list[dict[str, Any]]:
        return self._query("SELECT * FROM session_schedule_runs WHERE schedule_id=? ORDER BY run_date", (schedule_id,))

    def cancel_schedule(self, schedule_id: str, *, requested_by: str = "system") -> dict[str, Any]:
        """Disable a schedule (a session it already started keeps running; stop that with ``stop``)."""
        who = _actor(requested_by)
        now = self.rt.now()

        def _w(conn: sqlite3.Connection) -> str:
            row = conn.execute("SELECT enabled FROM session_schedules WHERE schedule_id=?", (schedule_id,)).fetchone()
            if row is None:
                raise SessionError(404, f"unknown schedule {schedule_id}")
            if not row[0]:
                return "ALREADY_CANCELLED"
            conn.execute("UPDATE session_schedules SET enabled=0, next_run_at=NULL, cancelled_at=?, cancelled_by=?,"
                         " updated_at=? WHERE schedule_id=?", (iso(now), who, iso(now), schedule_id))
            self._audit(conn, "SESSION_SCHEDULE_CANCELLED", who, {"schedule_id": schedule_id})
            return "CANCELLED"

        outcome = self.rt.db.write(_w)
        sch = self.get_schedule(schedule_id)
        if outcome == "CANCELLED":
            self._emit(f"Cancelled schedule {schedule_id} ({who}). A session it already started keeps running until "
                       "stopped or finished.", actor=who, event="schedule", payload=sch,
                       detail={"event": "schedule_cancelled", "schedule_id": schedule_id})
        return {"outcome": outcome, "schedule": sch}

    def _record_run(self, sch: dict[str, Any], occ: Occurrence, status: str, detail: str, now: datetime) -> bool:
        def _w(conn: sqlite3.Connection) -> bool:
            cur = conn.execute("INSERT OR IGNORE INTO session_schedule_runs(schedule_id, run_date, due_at, window_end,"
                               " status, session_id, detail, recorded_at) VALUES (?,?,?,?,?,NULL,?,?)",
                               (sch["schedule_id"], occ.run_date.isoformat(), iso(occ.start), iso(occ.end), status,
                                detail[:500], iso(now)))
            if cur.rowcount:
                self._audit(conn, "SESSION_SCHEDULE_RUN", "session-scheduler",
                            {"schedule_id": sch["schedule_id"], "run_date": occ.run_date.isoformat(), "status": status,
                             "detail": detail[:500]})
            return bool(cur.rowcount)

        recorded = bool(self.rt.db.write(_w))
        if recorded:
            verb = "MISSED (its window passed while the trading service was not running; never run late)" \
                if status == "MISSED" else f"SKIPPED: {detail}"
            self._emit(f"Scheduled session {sch['schedule_id']} for {et_label(occ.start)} {verb}.", actor="system",
                       event="schedule", payload=self.get_schedule(str(sch["schedule_id"])),
                       detail={"event": f"schedule_{status.lower()}", "schedule_id": sch["schedule_id"],
                               "run_date": occ.run_date.isoformat()})
        return recorded

    def _process_schedules(self, now: datetime, *, allow_start: bool, force: bool = False) -> builtins.list[dict[str, Any]]:
        """Start due occurrences (once), record MISSED/SKIPPED ones, refresh next_run_at. Returns recorded runs."""
        out: builtins.list[dict[str, Any]] = []
        rows = self._query("SELECT * FROM session_schedules WHERE enabled=1")
        for r in rows:
            sch = self._schedule_out(r, None)
            nra = parse_utc(r["next_run_at"])
            if not force and nra is not None and nra > now:
                continue
            created = parse_utc(r["created_at"]) or now
            recorded = {datetime.fromisoformat(x["run_date"]).date() for x in self.schedule_runs(sch["schedule_id"])}
            pending = pending_occurrences(sch["weekdays"], sch["start_time_et"], sch["duration_minutes"],
                                          created_at=created, now=now, recorded=recorded)
            for occ in pending:
                if now >= occ.end:
                    if self._record_run(sch, occ, "MISSED", "window passed while the trading service was not running",
                                        now):
                        out.append({"schedule_id": sch["schedule_id"], "run_date": occ.run_date.isoformat(),
                                    "status": "MISSED"})
                    recorded.add(occ.run_date)
                elif allow_start:
                    try:
                        s = self._start(sch["duration_minutes"], symbols=sch["symbols"],
                                        cadence_minutes=sch["cadence_minutes"], use_ai=sch["use_ai"],
                                        requested_by=f"schedule:{sch['schedule_id']}", schedule_id=sch["schedule_id"],
                                        occurrence=occ)
                        out.append({"schedule_id": sch["schedule_id"], "run_date": occ.run_date.isoformat(),
                                    "status": "STARTED", "session_id": s["session_id"]})
                    except SessionError as exc:
                        if self._record_run(sch, occ, "SKIPPED", exc.message, now):
                            out.append({"schedule_id": sch["schedule_id"], "run_date": occ.run_date.isoformat(),
                                        "status": "SKIPPED"})
                    recorded.add(occ.run_date)
            nxt = next_occurrence(sch["weekdays"], sch["start_time_et"], sch["duration_minutes"], created_at=created,
                                  now=now, recorded=recorded)
            new = iso(nxt.start) if nxt else None
            if new != r["next_run_at"]:
                sid = sch["schedule_id"]
                self.rt.db.write(lambda c, n=new, s=sid: c.execute(
                    "UPDATE session_schedules SET next_run_at=?, updated_at=? WHERE schedule_id=? AND enabled=1",
                    (n, iso(now), s)))
        return out

    # ================================================================== recovery
    async def recover(self, *, clean: bool, reason: str | None = None) -> dict[str, Any]:
        """Restart recovery (called by the API right after the startup reconciliation)."""
        now = self.rt.now()
        resumed: builtins.list[str] = []
        interrupted: builtins.list[str] = []
        stopped: builtins.list[str] = []
        rows = [r for r in self._query("SELECT * FROM trading_sessions WHERE state IN ('RUNNING','STOPPING')")
                if r["owner_process"] != self.process_id]
        for r in rows:
            sid = str(r["session_id"])
            ends = parse_utc(r["ends_at"])
            if r["state"] == "STOPPING":
                if self._finish(sid, "STOPPED", r["stop_reason"] or "stop was requested before the API restarted"):
                    stopped.append(sid)
            elif clean and ends is not None and now < ends:
                nxt = parse_utc(r["next_cycle_at"])
                nxt = now if nxt is None or nxt < now else nxt

                def _w(conn: sqlite3.Connection, sid: str = sid, nxt: datetime = nxt) -> bool:
                    cur = conn.execute("UPDATE trading_sessions SET owner_process=?, next_cycle_at=?, resumed_count="
                                       "resumed_count+1, updated_at=? WHERE session_id=? AND state='RUNNING'",
                                       (self.process_id, iso(nxt), iso(now), sid))
                    if cur.rowcount:
                        self._audit(conn, "SESSION_RESUMED", "session-manager",
                                    {"session_id": sid, "reason": "API restart; startup reconciliation CLEAN"})
                    return bool(cur.rowcount)

                if self.rt.db.write(_w):
                    resumed.append(sid)
                    left = int((ends - now).total_seconds() // 60)
                    self._emit(f"Session resumed after an API restart (startup broker reconciliation CLEAN); {left} min "
                               f"left until {et_label(ends)}.", actor="system", session=self.get(sid),
                               detail={"event": "resumed"})
            else:
                why = (reason or "the API restarted and the startup broker reconciliation was not CLEAN") if not clean \
                    else f"the API restarted after the session window ended ({et_label(ends) if ends else 'unknown'})"
                if self._finish(sid, "INTERRUPTED", why):
                    interrupted.append(sid)
        missed = self._process_schedules(now, allow_start=False, force=True)
        self._recovered = True
        self._force_schedule_scan = True
        return {"clean": clean, "resumed": resumed, "interrupted": interrupted, "stopped": stopped, "schedule_runs": missed}

    # ================================================================== loop
    async def run(self, stop_event: asyncio.Event) -> None:
        """Background loop (1 s ticks against ``runtime.now()``). Survives every exception in a tick or cycle."""
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._running = True
        try:
            while not stop_event.is_set():
                try:
                    await self.tick()
                except Exception as exc:
                    self._failure("tick", exc)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=self.tick_s)
        finally:
            self._running = False

    async def tick(self) -> None:
        now = self.rt.now()
        self._last_tick_at = now
        if not self._recovered:
            if time.monotonic() - self._created_mono < self.recover_timeout_s:
                return  # wait for the startup reconciliation to call recover(); never act on unknown state
            await self.recover(clean=False, reason=f"no startup reconciliation result within {self.recover_timeout_s:.0f}s")
        force, self._force_schedule_scan = self._force_schedule_scan, False
        self._process_schedules(now, allow_start=True, force=force)
        row = self._active_row()
        if row is None or row["owner_process"] != self.process_id:
            return
        await self._drive(row, now)

    async def _drive(self, row: dict[str, Any], now: datetime) -> None:
        sid = str(row["session_id"])
        if row["state"] == "STOPPING":
            with self._lock:
                busy = self._cycle_session == sid
            if not busy:
                self._finish(sid, "STOPPED", row["stop_reason"] or "stop requested")
            return
        if self._sim_finished():
            self._finish(sid, "COMPLETED", "the simulated replay reached the end of the synthetic data")
            await self._final_reconcile()
            return
        ends = parse_utc(row["ends_at"])
        if ends is None or now >= ends:
            self._finish(sid, "COMPLETED", "planned duration elapsed")
            await self._final_reconcile()
            return
        if not xcal.is_market_open(now):
            if sid not in self._closed_notified:
                self._closed_notified.add(sid)
                nxt = xcal.next_open(now)
                self._emit(f"Market closed: the session waits for XNYS regular hours (next open "
                           f"{et_label(nxt) if nxt else 'unknown'}); no decision cycles until then.", actor="system",
                           session=_row_out(row), detail={"event": "market_closed"})
            return
        self._closed_notified.discard(sid)
        nxt = parse_utc(row["next_cycle_at"])
        if nxt is not None and now < nxt:
            return
        await self._run_slot(row, scheduled_for=nxt or now)

    async def _run_slot(self, row: dict[str, Any], *, scheduled_for: datetime) -> None:
        sid = str(row["session_id"])
        with self._lock:
            cur = self._get(sid)
            if cur is None or cur["state"] != "RUNNING":
                return
            self._cycle_session = sid
        failures = int(cur["consecutive_failures"])
        try:
            now = self.rt.now()
            suspended = parse_utc(cur["ai_suspended_until"])
            use_ai = bool(cur["use_ai"]) and (suspended is None or now >= suspended)
            summary, error = await self._one_cycle(sid, json.loads(cur["symbols_json"]), use_ai)
            ai_failed = use_ai and self._ai_failed(summary)
            note = None
            if bool(cur["use_ai"]) and not use_ai:
                note = f"AI review suspended until {suspended.isoformat() if suspended else '?'} after an AI failure"
            failed = (summary is None or summary.get("outcome") == "ERROR") and not ai_failed
            failures = failures + 1 if failed else 0
            extra: dict[str, Any] = {}
            if ai_failed:
                until = now + self.ai_backoff
                extra["ai_suspended_until"] = iso(until)
                note = "AI review unavailable (model error/timeout/usage limit); this slot is re-run quant-only"
            self._record_cycle(sid, scheduled_for, now, summary, error, use_ai, note, failures, extra)
            if ai_failed:
                self._emit(f"AI review failed in this cycle ({_ai_reason(summary)}); the quant engine continues alone "
                           f"and AI review is retried after {et_label(now + self.ai_backoff)}.", actor="system",
                           session=self.get(sid), kind="gpt_call", detail={"event": "ai_fallback"})
                if self._state(sid) == "RUNNING":
                    t2 = self.rt.now()
                    summary2, error2 = await self._one_cycle(sid, json.loads(cur["symbols_json"]), False)
                    failed2 = summary2 is None or summary2.get("outcome") == "ERROR"
                    failures = failures + 1 if failed2 else 0
                    self._record_cycle(sid, scheduled_for, t2, summary2, error2, False,
                                       "quant-only fallback after an AI review failure", failures, {})
        finally:
            with self._lock:
                self._cycle_session = None
        state = self._state(sid)
        if state == "STOPPING":
            s = self._get(sid)
            self._finish(sid, "STOPPED", (s or {}).get("stop_reason") or "stop requested")
        elif state == "RUNNING" and failures >= self.max_consecutive_failures:
            last = self._query("SELECT reason FROM session_cycles WHERE session_id=? ORDER BY cycle_no DESC LIMIT 1",
                               (sid,))
            self._finish(sid, "FAILED", f"{failures} consecutive decision-cycle failures (last: "
                                        f"{_text(last[0]['reason'] if last else 'unknown', 200)})")

    def _state(self, sid: str) -> str | None:
        r = self._get(sid)
        return None if r is None else str(r["state"])

    async def _one_cycle(self, sid: str, symbols: builtins.list[str], use_ai: bool
                         ) -> tuple[dict[str, Any] | None, BaseException | None]:
        try:
            summary = await self.rt.run_cycle(symbols=symbols, use_ai=use_ai, actor=f"session:{sid}", session_id=sid)
            return (summary if isinstance(summary, dict) else {"outcome": "ERROR", "reason": "malformed cycle result"}), None
        except Exception as exc:  # a cycle failure never kills the loop
            self._failure("cycle", exc)
            return None, exc

    def _ai_failed(self, summary: dict[str, Any] | None) -> bool:
        """True when the cycle's in-app model review was unavailable (not when the model decided NO_TRADE)."""
        if summary is None:
            return False
        cid = summary.get("causal_id")
        if not cid:
            return False
        outcome = summary.get("outcome")
        if outcome == "ERROR":
            rows = self._query("SELECT 1 AS x FROM activity_events WHERE causal_id=? AND kind='stage' AND "
                               "stage='GPT_REVIEW' AND status='error' LIMIT 1", (str(cid),))
            return bool(rows)
        if outcome == "NO_TRADE" and summary.get("model_runs"):
            rows = self._query("SELECT status FROM model_runs WHERE causal_id=?", (str(cid),))
            return bool(rows) and all(str(r["status"]) in AI_UNAVAILABLE_STATUSES for r in rows)
        return False

    def _record_cycle(self, sid: str, scheduled_for: datetime, started: datetime, summary: dict[str, Any] | None,
                      error: BaseException | None, use_ai: bool, note: str | None, failures: int,
                      extra: dict[str, Any]) -> None:
        finished = self.rt.now()
        if summary is None:
            outcome, reason = "ERROR", f"{type(error).__name__}: {error}" if error else "cycle failed"
            causal = cand = coid = None
            locked: bool | None = None
            reasons: builtins.list[str] = []
        else:
            outcome = str(summary.get("outcome") or "UNKNOWN")
            reason = summary.get("reason")
            causal, cand, coid = summary.get("causal_id"), summary.get("candidate_symbol"), summary.get("client_order_id")
            lk = summary.get("entries_locked")
            locked = lk if isinstance(lk, bool) else None
            reasons = [str(x) for x in (summary.get("lock_reasons") or [])][:10]

        def _w(conn: sqlite3.Connection) -> None:
            r = conn.execute("SELECT cycles_run, cadence_minutes, next_cycle_at FROM trading_sessions WHERE session_id=?",
                             (sid,)).fetchone()
            if r is None:
                return
            cycle_no = int(r[0]) + 1
            cadence = timedelta(minutes=int(r[1]))
            nxt = scheduled_for + cadence
            while nxt <= finished:  # skip missed slots: never a burst of catch-up cycles
                nxt += cadence
            conn.execute("INSERT INTO session_cycles(session_id, cycle_no, scheduled_for, started_at, finished_at,"
                         " causal_id, outcome, candidate_symbol, client_order_id, reason, use_ai, entries_locked,"
                         " lock_reasons_json, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (sid, cycle_no, iso(scheduled_for), iso(started), iso(finished), causal, outcome, cand, coid,
                          None if reason is None else _text(reason, 500), int(use_ai),
                          None if locked is None else int(locked), canonical_json(reasons), note))
            sets = ["cycles_run=cycles_run+1", "trades=trades+?", "consecutive_failures=?", "last_cycle_at=?",
                    "next_cycle_at=?", "updated_at=?"]
            params: builtins.list[Any] = [1 if outcome == "SUBMITTED" else 0, failures, iso(finished), iso(nxt),
                                          iso(finished)]
            for k, v in extra.items():
                sets.append(f"{k}=?")
                params.append(v)
            conn.execute(f"UPDATE trading_sessions SET {', '.join(sets)} WHERE session_id=?", (*params, sid))  # noqa: S608

        self.rt.db.write(_w)
        session = self.get(sid)
        if outcome == "ERROR":
            self._emit(f"Session decision cycle failed: {_text(reason, 200)} (consecutive failures: {failures}/"
                       f"{self.max_consecutive_failures}); the session continues.", actor="system", session=session,
                       kind="error", detail={"event": "cycle_error", "causal_id": causal})
        else:
            with contextlib.suppress(Exception):
                self._on_loop(lambda: self.rt.bus.publish("session", session))


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _ai_reason(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "unknown"
    return _text(summary.get("reason") or "model unavailable", 160)
