"""The crew's own queue and memory: ``experiments/agents/crew.sqlite3`` (SQLite WAL, stdlib only).

This is NOT the trading ledger and never touches ``var/<namespace>/*.sqlite3``. Tables:

* ``tasks``: research work items. ``id`` is a hash of ``(kind, key)`` and ``key`` is UNIQUE, so enqueueing is
  idempotent. A task is claimed atomically (``UPDATE ... WHERE state='QUEUED' RETURNING``), so two crew processes
  on one file never run the same task.
* ``runs``: one row per provider invocation (outcome, usage, sanitized detail).
* ``provider_state``: cooldowns after usage limits, consecutive-limit counter, disabled reason.
* ``notes``: append-only shared memory of both providers (SQLite triggers refuse UPDATE and DELETE). Agent-written
  notes carry ``evidence_status='AGENT_NOTE'``; they never become facts, signals or validation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any, Final

from .contracts import Provider, TaskKind, TaskState

AGENT_NOTE: Final = "AGENT_NOTE"      # agent-authored text: proposal / opinion only
LEDGER_DEV: Final = "LEDGER_DEV"      # deterministic trial-ledger output on development data (copied, not agent-reported)
CREW_EVENT: Final = "CREW_EVENT"      # crew bookkeeping (recovery, expiry, refusals by deterministic validation)
INTERRUPTED: Final = "INTERRUPTED"    # run outcome for a run that never finished (crash / signal)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, key TEXT NOT NULL UNIQUE, payload TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL, author_provider TEXT, exclude_provider TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 2, not_before TEXT, expires_at TEXT,
    created TEXT NOT NULL, updated TEXT NOT NULL, result TEXT, parent_id TEXT);
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state, priority);
CREATE INDEX IF NOT EXISTS tasks_parent ON tasks(parent_id);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, provider TEXT NOT NULL, started TEXT NOT NULL, ended TEXT,
    outcome TEXT, usage TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS runs_provider ON runs(provider, started);
CREATE INDEX IF NOT EXISTS runs_task ON runs(task_id);
CREATE TABLE IF NOT EXISTS provider_state (
    provider TEXT PRIMARY KEY, cooldown_until TEXT, consecutive_limits INTEGER NOT NULL DEFAULT 0,
    disabled_reason TEXT, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, task_id TEXT, provider TEXT, kind TEXT NOT NULL,
    mechanism TEXT, body TEXT NOT NULL, evidence_status TEXT NOT NULL DEFAULT 'AGENT_NOTE');
CREATE INDEX IF NOT EXISTS notes_kind ON notes(kind, seq);
CREATE INDEX IF NOT EXISTS notes_task ON notes(task_id);
CREATE TRIGGER IF NOT EXISTS notes_no_update BEFORE UPDATE ON notes
    BEGIN SELECT RAISE(ABORT, 'crew notes are append-only'); END;
CREATE TRIGGER IF NOT EXISTS notes_no_delete BEFORE DELETE ON notes
    BEGIN SELECT RAISE(ABORT, 'crew notes are append-only'); END;
"""


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.astimezone(UTC).isoformat()


def parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    dt = datetime.fromisoformat(raw)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def day_start(now: datetime) -> datetime:
    n = now.astimezone(UTC)
    return n.replace(hour=0, minute=0, second=0, microsecond=0)


def task_id_for(kind: TaskKind | str, key: str) -> str:
    return hashlib.sha256(f"{TaskKind(kind).value}\x1f{key}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TaskRow:
    id: str
    kind: TaskKind
    key: str
    payload: dict[str, Any]
    priority: int
    state: TaskState
    author_provider: Provider | None
    exclude_provider: Provider | None
    attempts: int
    max_attempts: int
    not_before: datetime | None
    expires_at: datetime | None
    created: datetime
    updated: datetime
    result: dict[str, Any] | None
    parent_id: str | None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> TaskRow:
        return cls(
            id=r["id"], kind=TaskKind(r["kind"]), key=r["key"], payload=json.loads(r["payload"]),
            priority=int(r["priority"]), state=TaskState(r["state"]),
            author_provider=Provider(r["author_provider"]) if r["author_provider"] else None,
            exclude_provider=Provider(r["exclude_provider"]) if r["exclude_provider"] else None,
            attempts=int(r["attempts"]), max_attempts=int(r["max_attempts"]),
            not_before=parse_iso(r["not_before"]), expires_at=parse_iso(r["expires_at"]),
            created=parse_iso(r["created"]) or datetime.min.replace(tzinfo=UTC),
            updated=parse_iso(r["updated"]) or datetime.min.replace(tzinfo=UTC),
            result=json.loads(r["result"]) if r["result"] else None, parent_id=r["parent_id"])

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "key": self.key, "state": self.state.value,
                "priority": self.priority, "attempts": self.attempts, "max_attempts": self.max_attempts,
                "author_provider": self.author_provider, "exclude_provider": self.exclude_provider,
                "not_before": self.not_before.isoformat() if self.not_before else None,
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
                "created": self.created.isoformat(), "updated": self.updated.isoformat(),
                "parent_id": self.parent_id, "payload": self.payload, "result": self.result}


@dataclass(frozen=True)
class ProviderState:
    provider: Provider
    cooldown_until: datetime | None = None
    consecutive_limits: int = 0
    disabled_reason: str | None = None

    def cooling(self, now: datetime) -> bool:
        return self.cooldown_until is not None and self.cooldown_until > now

    def public(self, now: datetime) -> dict[str, Any]:
        return {"provider": self.provider.value,
                "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
                "cooling_down": self.cooling(now), "consecutive_limits": self.consecutive_limits,
                "disabled_reason": self.disabled_reason}


@dataclass(frozen=True)
class Note:
    seq: int
    at: datetime
    task_id: str | None
    provider: str | None
    kind: str
    mechanism: str | None
    body: dict[str, Any]
    evidence_status: str

    def public(self) -> dict[str, Any]:
        return {"seq": self.seq, "at": self.at.isoformat(), "task_id": self.task_id, "provider": self.provider,
                "kind": self.kind, "mechanism": self.mechanism, "body": self.body,
                "evidence_status": self.evidence_status}


class CrewLocked(RuntimeError):
    pass


class CrewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=15000")
            yield db
        finally:
            db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    # ----------------------------------------------------------------------------------------------- tasks
    def enqueue(self, kind: TaskKind, key: str, payload: dict[str, Any], *, now: datetime, priority: int = 0,
                author_provider: Provider | None = None, exclude_provider: Provider | None = None,
                max_attempts: int = 2, not_before: datetime | None = None, expires_at: datetime | None = None,
                parent_id: str | None = None) -> tuple[str, bool]:
        """Insert a task unless one with this key exists. Returns (task id, created)."""
        tid = task_id_for(kind, key)
        with self._tx() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO tasks(id,kind,key,payload,priority,state,author_provider,exclude_provider,"
                "attempts,max_attempts,not_before,expires_at,created,updated,result,parent_id) "
                "VALUES (?,?,?,?,?,?,?,?,0,?,?,?,?,?,NULL,?)",
                (tid, TaskKind(kind).value, key, json.dumps(payload, sort_keys=True, default=str), int(priority),
                 TaskState.QUEUED.value, author_provider.value if author_provider else None,
                 exclude_provider.value if exclude_provider else None, int(max_attempts),
                 iso(not_before) if not_before else None, iso(expires_at) if expires_at else None,
                 iso(now), iso(now), parent_id))
            return tid, cur.rowcount == 1

    def get_task(self, task_id: str) -> TaskRow | None:
        with self._db() as db:
            r = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return TaskRow.from_row(r) if r else None

    def tasks(self, kind: TaskKind | None = None, states: Sequence[TaskState] | None = None,
              limit: int = 10_000) -> list[TaskRow]:
        q, args = "SELECT * FROM tasks WHERE 1=1", []
        if kind is not None:
            q += " AND kind=?"
            args.append(TaskKind(kind).value)
        if states:
            q += f" AND state IN ({','.join('?' * len(states))})"
            args.extend(TaskState(s).value for s in states)
        q += " ORDER BY created, id LIMIT ?"
        args.append(int(limit))
        with self._db() as db:
            return [TaskRow.from_row(r) for r in db.execute(q, args).fetchall()]

    def keys(self) -> set[str]:
        with self._db() as db:
            return {r[0] for r in db.execute("SELECT key FROM tasks")}

    def runnable(self, now: datetime) -> list[TaskRow]:
        """QUEUED tasks whose not_before has passed and that have not expired, highest priority first."""
        n = iso(now)
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM tasks WHERE state='QUEUED' AND (not_before IS NULL OR not_before<=?) "
                "AND (expires_at IS NULL OR expires_at>?) ORDER BY priority DESC, created, id", (n, n)).fetchall()
        return [TaskRow.from_row(r) for r in rows]

    def claim(self, task_id: str, now: datetime) -> TaskRow | None:
        """Atomically move one QUEUED task to RUNNING; None if another process got it first."""
        stamp = iso(now)
        with self._db() as db:
            r = db.execute("UPDATE tasks SET state='RUNNING', updated=? WHERE id=? AND state='QUEUED' "
                           "AND (not_before IS NULL OR not_before<=?) AND (expires_at IS NULL OR expires_at>?) RETURNING *",
                           (stamp, task_id, stamp, stamp)).fetchone()
        return TaskRow.from_row(r) if r else None

    def finish(self, task_id: str, state: TaskState, result: dict[str, Any] | None, *, now: datetime,
               consume_attempt: bool = True, author_provider: Provider | None = None) -> None:
        """Terminal state. ``author_provider`` stamps who produced a hypothesis (its review excludes that provider)."""
        with self._db() as db:
            db.execute("UPDATE tasks SET state=?, result=?, updated=?, attempts=attempts+?, "
                       "author_provider=COALESCE(?, author_provider) WHERE id=?",
                       (TaskState(state).value, json.dumps(result, default=str) if result is not None else None,
                        iso(now), 1 if consume_attempt else 0, author_provider.value if author_provider else None,
                        task_id))

    def requeue(self, task_id: str, *, now: datetime, consume_attempt: bool,
                not_before: datetime | None = None) -> TaskRow | None:
        """RUNNING -> QUEUED; with consume_attempt the attempt counts and the task FAILs at max_attempts."""
        with self._db() as db:
            r = db.execute(
                "UPDATE tasks SET attempts=attempts+?, updated=?, not_before=?, "
                "state=CASE WHEN ? AND attempts+1>=max_attempts THEN 'FAILED' ELSE 'QUEUED' END "
                "WHERE id=? RETURNING *",
                (1 if consume_attempt else 0, iso(now), iso(not_before) if not_before else None,
                 1 if consume_attempt else 0, task_id)).fetchone()
        return TaskRow.from_row(r) if r else None

    def set_result(self, task_id: str, result: dict[str, Any], *, now: datetime) -> None:
        with self._db() as db:
            db.execute("UPDATE tasks SET result=?, updated=? WHERE id=?",
                       (json.dumps(result, default=str), iso(now), task_id))

    def expire(self, now: datetime) -> list[str]:
        """QUEUED tasks past ``expires_at`` -> CANCELLED (reviews: no independent reviewer was available)."""
        out: list[str] = []
        with self._tx() as db:
            rows = db.execute("SELECT id, kind FROM tasks WHERE state='QUEUED' AND expires_at IS NOT NULL "
                              "AND expires_at<=?", (iso(now),)).fetchall()
            for r in rows:
                reason = "no independent reviewer" if r["kind"] == TaskKind.REVIEW.value else "expired"
                db.execute("UPDATE tasks SET state='CANCELLED', result=?, updated=? WHERE id=? AND state='QUEUED'",
                           (json.dumps({"reason": reason}), iso(now), r["id"]))
                self._note(db, now=now, task_id=r["id"], provider=None, kind="event", mechanism=None,
                           body={"event": "CANCELLED", "reason": reason}, evidence_status=CREW_EVENT)
                out.append(r["id"])
        return out

    def recover(self, now: datetime) -> list[str]:
        """Startup recovery: RUNNING -> QUEUED without consuming an attempt; unfinished runs -> INTERRUPTED."""
        out: list[str] = []
        with self._tx() as db:
            for r in db.execute("SELECT id FROM tasks WHERE state='RUNNING'").fetchall():
                db.execute("UPDATE tasks SET state='QUEUED', updated=? WHERE id=?", (iso(now), r["id"]))
                self._note(db, now=now, task_id=r["id"], provider=None, kind="event", mechanism=None,
                           body={"event": "RECOVERED", "detail": "RUNNING at startup -> QUEUED (attempt not consumed)"},
                           evidence_status=CREW_EVENT)
                out.append(r["id"])
            db.execute("UPDATE runs SET ended=?, outcome=?, detail=COALESCE(detail,'') || 'interrupted' "
                       "WHERE ended IS NULL", (iso(now), INTERRUPTED))
        return out

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._db() as db:
            for kind, state, n in db.execute("SELECT kind, state, COUNT(*) FROM tasks GROUP BY kind, state"):
                out.setdefault(str(kind), {})[str(state)] = int(n)
        return out

    # ------------------------------------------------------------------------------------------------ runs
    def start_run(self, task_id: str, provider: Provider, now: datetime) -> str:
        rid = uuid.uuid4().hex[:16]
        with self._db() as db:
            db.execute("INSERT INTO runs(id,task_id,provider,started) VALUES (?,?,?,?)",
                       (rid, task_id, Provider(provider).value, iso(now)))
        return rid

    def end_run(self, run_id: str, outcome: str, *, now: datetime, usage: dict[str, Any] | None = None,
                detail: str = "") -> None:
        with self._db() as db:
            db.execute("UPDATE runs SET ended=?, outcome=?, usage=?, detail=? WHERE id=? AND ended IS NULL",
                       (iso(now), str(outcome), json.dumps(usage or {}, default=str), detail[:2000], run_id))

    def runs_since(self, since: datetime, provider: Provider | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM runs WHERE started>=?", [iso(since)]
        if provider is not None:
            q += " AND provider=?"
            args.append(Provider(provider).value)
        with self._db() as db:
            rows = [dict(r) for r in db.execute(q + " ORDER BY started", args).fetchall()]
        for r in rows:
            r["usage"] = json.loads(r["usage"]) if r.get("usage") else {}
        return rows

    def runs_today(self, provider: Provider, now: datetime) -> int:
        """Runs started this UTC day that spent (or may have spent) quota: all except a CLI that was not runnable."""
        with self._db() as db:
            return int(db.execute(
                "SELECT COUNT(*) FROM runs WHERE provider=? AND started>=? AND COALESCE(outcome,'')<>'UNAVAILABLE'",
                (Provider(provider).value, iso(day_start(now)))).fetchone()[0])

    def runs_in_window(self, provider: Provider, now: datetime, window: timedelta) -> int:
        """Runs started in the rolling subscription window ``(now - window, now]`` that may have spent quota."""
        with self._db() as db:
            return int(db.execute(
                "SELECT COUNT(*) FROM runs WHERE provider=? AND started>? AND COALESCE(outcome,'')<>'UNAVAILABLE'",
                (Provider(provider).value, iso(now - window))).fetchone()[0])

    def task_runs(self, task_id: str) -> list[dict[str, Any]]:
        with self._db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs WHERE task_id=? ORDER BY started",
                                                 (task_id,)).fetchall()]

    # ---------------------------------------------------------------------------------------- providers
    def provider_state(self, provider: Provider) -> ProviderState:
        with self._db() as db:
            r = db.execute("SELECT * FROM provider_state WHERE provider=?", (Provider(provider).value,)).fetchone()
        if r is None:
            return ProviderState(Provider(provider))
        return ProviderState(Provider(provider), parse_iso(r["cooldown_until"]), int(r["consecutive_limits"]),
                             r["disabled_reason"])

    def put_provider_state(self, st: ProviderState, now: datetime) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO provider_state(provider,cooldown_until,consecutive_limits,disabled_reason,updated) "
                "VALUES (?,?,?,?,?) ON CONFLICT(provider) DO UPDATE SET cooldown_until=excluded.cooldown_until, "
                "consecutive_limits=excluded.consecutive_limits, disabled_reason=excluded.disabled_reason, "
                "updated=excluded.updated",
                (st.provider.value, iso(st.cooldown_until) if st.cooldown_until else None, int(st.consecutive_limits),
                 st.disabled_reason, iso(now)))

    # -------------------------------------------------------------------------------------------- notes
    @staticmethod
    def _note(db: sqlite3.Connection, *, now: datetime, task_id: str | None, provider: Provider | str | None,
              kind: str, mechanism: str | None, body: dict[str, Any], evidence_status: str) -> int:
        cur = db.execute("INSERT INTO notes(at,task_id,provider,kind,mechanism,body,evidence_status) "
                         "VALUES (?,?,?,?,?,?,?)",
                         (iso(now), task_id, str(provider) if provider else None, kind, mechanism,
                          json.dumps(body, sort_keys=True, default=str), evidence_status))
        return int(cur.lastrowid or 0)

    def add_note(self, kind: str, body: dict[str, Any], *, now: datetime, task_id: str | None = None,
                 provider: Provider | str | None = None, mechanism: str | None = None,
                 evidence_status: str = AGENT_NOTE) -> int:
        with self._db() as db:
            return self._note(db, now=now, task_id=task_id, provider=provider, kind=kind, mechanism=mechanism,
                              body=body, evidence_status=evidence_status)

    def notes(self, *, kind: str | Sequence[str] | None = None, limit: int = 50, task_id: str | None = None,
              since: datetime | None = None, mechanism: str | None = None) -> list[Note]:
        """Newest first."""
        q, args = "SELECT * FROM notes WHERE 1=1", []
        if kind is not None:
            kinds = [kind] if isinstance(kind, str) else list(kind)
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        if task_id is not None:
            q += " AND task_id=?"
            args.append(task_id)
        if mechanism is not None:
            q += " AND mechanism=?"
            args.append(mechanism)
        if since is not None:
            q += " AND at>=?"
            args.append(iso(since))
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(int(limit))
        with self._db() as db:
            rows = db.execute(q, args).fetchall()
        return [Note(int(r["seq"]), parse_iso(r["at"]) or datetime.min.replace(tzinfo=UTC), r["task_id"],
                     r["provider"], r["kind"], r["mechanism"], json.loads(r["body"]), r["evidence_status"])
                for r in rows]

    def raw_connection(self) -> sqlite3.Connection:
        """For tests and maintenance: a plain connection (the append-only triggers still apply)."""
        return sqlite3.connect(self.path)


class SingleInstanceLock:
    """``fcntl`` lock file so only one ``run`` loop uses a store; released automatically if the process dies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise CrewLocked(f"another crew process holds {self.path}") from None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
