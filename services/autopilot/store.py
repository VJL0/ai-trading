"""Durable autopilot state: its own SQLite file under ``experiments/autopilot/`` (never the trading ledger).

* ``events``: append-only log (UPDATE/DELETE abort in triggers). Every decision, action, refusal and alert.
* ``jobs``: one row per (trading day, job). A job is idempotent per day: ``DONE`` / ``SKIPPED`` / ``MISSED`` /
  ``BLOCKED`` / ``SUPERSEDED`` / ``FAILED`` are terminal and never re-run; ``RUNNING`` left behind by a crash is
  resumed by the job's own resume rule (pre-open re-inspects the API before any request).
* ``kv``: small current-state values (stand-down flag, last intraday check, crew ticks done, holdout attempts).

A single-instance lock (``fcntl.flock`` on ``autopilot.lock``) is held for the life of ``run``; ``status`` and
``plan`` read without it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from stat import S_ISREG
from typing import Any, Final, TextIO

log = logging.getLogger("lat.autopilot.events")

TERMINAL: Final = frozenset({"DONE", "SKIPPED", "MISSED", "BLOCKED", "SUPERSEDED", "FAILED"})
JOB_STATES: Final = frozenset({"PENDING", "RUNNING", "UNKNOWN", *TERMINAL})
EVENT_KINDS: Final = frozenset({"START", "STOP", "TICK", "DECISION", "ACTION", "RESULT", "REFUSED", "ALERT",
                                "STANDDOWN", "STANDDOWN_CLEARED", "SLEEP_GAP", "RESUME", "REPORT"})

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT, kind TEXT NOT NULL, job TEXT,
    message TEXT NOT NULL, data_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_day ON events(day, seq);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'autopilot events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'autopilot events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_replace BEFORE INSERT ON events
    WHEN EXISTS (SELECT 1 FROM events WHERE seq = NEW.seq)
    BEGIN SELECT RAISE(ABORT, 'autopilot events are append-only'); END;
CREATE TABLE IF NOT EXISTS jobs (
    day TEXT NOT NULL, job TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(day, job));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


class AutopilotLocked(RuntimeError):
    pass


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _standalone(info: os.stat_result, path: Path) -> None:
    if not S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"autopilot state path must be a standalone regular file: {path}")


class Store:
    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        self.path = path
        self.readonly = readonly
        if readonly and not path.exists():
            raise FileNotFoundError(path)
        if not readonly:
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._db()) as db, db:
                db.executescript(_SCHEMA)

    def _db(self) -> sqlite3.Connection:
        try:
            _standalone(self.path.lstat(), self.path)
        except FileNotFoundError:
            if self.readonly:
                raise
        if self.readonly:
            db = sqlite3.connect(f"{self.path.absolute().as_uri()}?mode=ro", uri=True, timeout=10)
        else:
            db = sqlite3.connect(self.path, timeout=10)
            try:
                db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                db.close()
                raise
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------------ events (append-only)
    def event(self, now: datetime, kind: str, message: str, *, day: str | None = None, job: str | None = None,
              data: dict[str, Any] | None = None) -> int:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        with closing(self._db()) as db, db:
            cur = db.execute("INSERT INTO events(ts, day, kind, job, message, data_json) VALUES (?,?,?,?,?,?)",
                             (_iso(now), day, kind, job, message[:2000],
                              json.dumps(data or {}, default=str, sort_keys=True)))
            seq = int(cur.lastrowid or 0)
        log.info("%s %s%s: %s", kind, f"{day} " if day else "", job or "-", message[:300])
        return seq

    def events(self, *, day: str | None = None, kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        q = "SELECT * FROM events"
        cond: list[str] = []
        args: list[Any] = []
        if day is not None:
            cond.append("day=?")
            args.append(day)
        if kind is not None:
            cond.append("kind=?")
            args.append(kind)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(int(limit))
        with closing(self._db()) as db:
            rows = db.execute(q, args).fetchall()
        return [{**dict(r), "data": json.loads(r["data_json"])} for r in reversed(rows)]

    def last_seq(self, kinds: tuple[str, ...], *, exclude_jobs: tuple[str, ...] = ()) -> int:
        """Highest event sequence number of the given kinds (0 if none), ignoring events of ``exclude_jobs``."""
        q = "SELECT MAX(seq) FROM events WHERE 1=1"
        q += f" AND kind IN ({','.join('?' * len(kinds))})"
        args: list[Any] = list(kinds)
        if exclude_jobs:
            q += f" AND COALESCE(job, '') NOT IN ({','.join('?' * len(exclude_jobs))})"
            args.extend(exclude_jobs)
        with closing(self._db()) as db:
            r = db.execute(q, args).fetchone()
        return int(r[0] or 0)

    # ------------------------------------------------------------------ jobs
    def job(self, day: str, job: str) -> dict[str, Any] | None:
        with closing(self._db()) as db:
            r = db.execute("SELECT * FROM jobs WHERE day=? AND job=?", (day, job)).fetchone()
        return None if r is None else {**dict(r), "detail": json.loads(r["detail_json"])}

    def jobs(self, day: str | None = None) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            if day is None:
                rows = db.execute("SELECT * FROM jobs ORDER BY day, job").fetchall()
            else:
                rows = db.execute("SELECT * FROM jobs WHERE day=? ORDER BY job", (day,)).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail_json"])} for r in rows]

    def set_job(self, now: datetime, day: str, job: str, status: str, *, detail: dict[str, Any] | None = None,
                attempt: bool = False) -> dict[str, Any]:
        """Upsert a job. A terminal job is never reopened (returns the stored row unchanged)."""
        if status not in JOB_STATES:
            raise ValueError(f"unknown job status {status!r}")
        with closing(self._db()) as db, db:
            r = db.execute("SELECT status, detail_json FROM jobs WHERE day=? AND job=?", (day, job)).fetchone()
            if r is not None and r["status"] in TERMINAL:
                pass
            else:
                merged = json.loads(r["detail_json"]) if r is not None else {}
                merged.update(detail or {})
                text = json.dumps(merged, default=str, sort_keys=True)
                if r is None:
                    db.execute("INSERT INTO jobs(day, job, status, attempts, created_at, updated_at, detail_json)"
                               " VALUES (?,?,?,?,?,?,?)", (day, job, status, int(attempt), _iso(now), _iso(now), text))
                else:
                    db.execute("UPDATE jobs SET status=?, attempts=attempts+?, updated_at=?, detail_json=?"
                               " WHERE day=? AND job=?", (status, int(attempt), _iso(now), text, day, job))
        row = self.job(day, job)
        assert row is not None
        return row

    def open_jobs(self) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            rows = db.execute("SELECT * FROM jobs WHERE status IN ('PENDING','RUNNING','UNKNOWN') ORDER BY day, job"
                              ).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail_json"])} for r in rows]

    # ------------------------------------------------------------------ kv
    def get(self, key: str, default: Any = None) -> Any:
        with closing(self._db()) as db:
            r = db.execute("SELECT value_json FROM kv WHERE key=?", (key,)).fetchone()
        return default if r is None else json.loads(r["value_json"])

    def put(self, now: datetime, key: str, value: Any) -> None:
        with closing(self._db()) as db, db:
            db.execute("INSERT INTO kv(key, value_json, updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET"
                       " value_json=excluded.value_json, updated_at=excluded.updated_at",
                       (key, json.dumps(value, default=str, sort_keys=True), _iso(now)))

    def delete(self, key: str) -> None:
        with closing(self._db()) as db, db:
            db.execute("DELETE FROM kv WHERE key=?", (key,))


@contextlib.contextmanager
def instance_lock(path: Path) -> Iterator[TextIO]:
    """Hold the single-instance lock or raise ``AutopilotLocked``. The PID is written for ``status``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = os.fdopen(
        os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600), "r+",
    )
    try:
        _standalone(os.fstat(fh.fileno()), path)
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AutopilotLocked(f"another autopilot process holds {path}") from None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        yield fh
    finally:
        fh.close()


def lock_holder(path: Path) -> int | None:
    """PID of the running autopilot, or None when the lock is free."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as fh:
        _standalone(os.fstat(fh.fileno()), path)
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            text = fh.read().strip()
            return int(text) if text.isdigit() else -1
        fcntl.flock(fh, fcntl.LOCK_UN)
    return None
