"""SQLite operational store: WAL mode, explicit migrations, single durable writer (spec §11).

* All writes go through ``Database.write(fn)``, executed on one dedicated writer thread inside
  ``BEGIN IMMEDIATE ... COMMIT`` (rolled back on any exception). This is the single-writer discipline.
* Reads use short-lived connections (``Database.read(fn)``), safe concurrently with the writer in WAL.
* ``synchronous=FULL`` because the order ledger must survive a crash right after a broker accepts.
* Migrations are ``NNNN_name.sql`` files discovered in ``packages/core/migrations`` and every
  ``services/*/migrations`` directory, applied once in filename order and checksummed. Number ranges:
  0001-0099 core, 0100 market_data, 0200 ai, 0300 risk, 0400 order_ledger, 0500 api, 0600 research.
"""

from __future__ import annotations

import contextlib
import queue
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from .canonical import sha256_hex
from .config import PROJECT_ROOT

T = TypeVar("T")

MIGRATION_DIRS: tuple[Path, ...] = (PROJECT_ROOT / "packages" / "core" / "migrations",)


def discover_migrations(extra_dirs: tuple[Path, ...] = ()) -> list[Path]:
    dirs = list(MIGRATION_DIRS) + sorted((PROJECT_ROOT / "services").glob("*/migrations")) + list(extra_dirs)
    files: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.glob("*.sql"):
            if f.name in files:
                raise RuntimeError(f"duplicate migration name {f.name}: {files[f.name]} and {f}")
            files[f.name] = f
    return [files[k] for k in sorted(files)]


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10, check_same_thread=False)
    else:
        conn = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
    return conn


def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(UTC)
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


_STOP = object()


class Database:
    def __init__(self, path: Path, *, extra_migration_dirs: tuple[Path, ...] = ()) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._extra = extra_migration_dirs
        self._queue: queue.Queue[Any] = queue.Queue()
        self._writer_conn = _connect(path)
        self._thread = threading.Thread(target=self._run_writer, name=f"sqlite-writer:{path.name}", daemon=True)
        self._closed = False
        self._thread.start()
        self.applied_migrations = self.write(self._migrate)

    # ---------------------------------------------------------------- writer thread
    def _run_writer(self) -> None:
        conn = self._writer_conn
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            fn, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                conn.execute("BEGIN IMMEDIATE")
                result = fn(conn)
                conn.execute("COMMIT")
            except BaseException as exc:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        conn.close()

    def submit(self, fn: Callable[[sqlite3.Connection], T]) -> Future[T]:
        if self._closed:
            raise RuntimeError("database closed")
        fut: Future[T] = Future()
        self._queue.put((fn, fut))
        return fut

    def write(self, fn: Callable[[sqlite3.Connection], T], timeout: float = 30.0) -> T:
        """Run ``fn(conn)`` in one IMMEDIATE transaction on the single writer thread."""
        if threading.current_thread() is self._thread:
            raise RuntimeError("nested write() from inside a write transaction")
        return self.submit(fn).result(timeout=timeout)

    async def awrite(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        import asyncio

        return await asyncio.wrap_future(self.submit(fn))

    # ---------------------------------------------------------------- reads
    def read(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        conn = _connect(self.path, readonly=True)
        try:
            return fn(conn)
        finally:
            conn.close()

    def query(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        return self.read(lambda c: [dict(r) for r in c.execute(sql, params).fetchall()])

    def query_one(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- migrations
    def _migrate(self, conn: sqlite3.Connection) -> list[str]:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        applied = {r["name"]: r["checksum"] for r in conn.execute("SELECT name, checksum FROM schema_migrations")}
        newly: list[str] = []
        for path in discover_migrations(self._extra):
            sql = path.read_text()
            checksum = sha256_hex(sql)
            if path.name in applied:
                if applied[path.name] != checksum:
                    raise RuntimeError(
                        f"migration {path.name} was modified after being applied; add a new migration instead"
                    )
                continue
            # executescript would COMMIT our transaction; run statements individually instead.
            for stmt in _split_sql(sql):
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO schema_migrations(name, checksum, applied_at) VALUES (?,?,?)",
                (path.name, checksum, utc_iso()),
            )
            newly.append(path.name)
        return newly

    # ---------------------------------------------------------------- lifecycle
    def backup(self, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = _connect(self.path, readonly=True)
        try:
            out = sqlite3.connect(dest)
            with out:
                src.backup(out)
            out.close()
        finally:
            src.close()
        return dest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=10)

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _split_sql(sql: str) -> list[str]:
    """Split a migration into statements, keeping CREATE TRIGGER ... END blocks intact."""
    statements: list[str] = []
    buf: list[str] = []
    in_trigger = False
    for line in sql.splitlines():
        stripped = line.strip()
        if not buf and (not stripped or stripped.startswith("--")):
            continue
        buf.append(line)
        upper = stripped.upper()
        if upper.startswith("CREATE TRIGGER"):
            in_trigger = True
        if in_trigger:
            if upper.endswith("END;"):
                statements.append("\n".join(buf))
                buf, in_trigger = [], False
            continue
        if stripped.endswith(";"):
            stmt = "\n".join(buf).strip()
            if sqlite3.complete_statement(stmt):
                statements.append(stmt)
                buf = []
    tail = "\n".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements
