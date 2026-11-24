"""Research provenance: append-only trial ledger and the one-use protected holdout vault.

* ``TrialLedger`` records EVERY evaluated variant (failures included) with its family, parameters, data
  fingerprint, code hash and development statistics. Deflated-Sharpe trial counts and Sharpe variances are taken
  from it, so searching more variants automatically raises the bar.
* ``HoldoutVault`` fixes the protected holdout interval once (immutable after the first registration) and allows
  each candidate (family + spec hash) at most ONE look; each family has a hard look budget. The access is
  committed BEFORE the evaluation runs, so a crash still consumes the look. Development code receives panels
  sliced to end strictly before the holdout start (``dev_end``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

HOLDOUT_LOOKS_PER_FAMILY = 3
"""Hard budget of holdout evaluations per strategy family (protects against holdout shopping)."""


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:20]


def code_hash() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db


class TrialLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(_connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS trials (
                    id INTEGER PRIMARY KEY, at TEXT NOT NULL, family TEXT NOT NULL, variant TEXT NOT NULL,
                    spec_hash TEXT NOT NULL, spec TEXT NOT NULL, data_fingerprint TEXT NOT NULL,
                    code_hash TEXT NOT NULL, stage TEXT NOT NULL, sharpe_per_period REAL, ann_sharpe REAL,
                    days INTEGER, metrics TEXT NOT NULL, status TEXT NOT NULL, note TEXT);
                CREATE INDEX IF NOT EXISTS trials_family ON trials(family);
                CREATE TRIGGER IF NOT EXISTS trials_no_update BEFORE UPDATE ON trials
                    BEGIN SELECT RAISE(ABORT, 'trial ledger is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS trials_no_delete BEFORE DELETE ON trials
                    BEGIN SELECT RAISE(ABORT, 'trial ledger is append-only'); END;
            """)

    def record(self, *, family: str, variant: str, spec: dict[str, Any], data_fingerprint: str, stage: str,
               metrics: dict[str, Any], status: str, note: str = "") -> int:
        sr_pp = metrics.get("sharpe", 0.0) / (252 ** 0.5) if metrics.get("sharpe") is not None else None
        with closing(_connect(self.path)) as db, db:
            cur = db.execute(
                "INSERT INTO trials(at,family,variant,spec_hash,spec,data_fingerprint,code_hash,stage,"
                "sharpe_per_period,ann_sharpe,days,metrics,status,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(), family, variant, spec_hash(spec), json.dumps(spec, default=str),
                 data_fingerprint, code_hash(), stage, sr_pp, metrics.get("sharpe"), metrics.get("days"),
                 json.dumps(metrics, default=str), status, note))
            return int(cur.lastrowid or 0)

    def count(self, family: str | None = None, stage: str = "DEV") -> int:
        q = "SELECT COUNT(*) FROM trials WHERE stage=?" + (" AND family=?" if family else "")  # noqa: S608 constant SQL
        args: tuple[Any, ...] = (stage, family) if family else (stage,)
        with closing(_connect(self.path)) as db:
            return int(db.execute(q, args).fetchone()[0])

    def sharpes(self, family: str | None = None, stage: str = "DEV") -> list[float]:
        q = "SELECT sharpe_per_period FROM trials WHERE stage=? AND sharpe_per_period IS NOT NULL" + \
            (" AND family=?" if family else "")  # noqa: S608 constant SQL fragments, values bound
        args: tuple[Any, ...] = (stage, family) if family else (stage,)
        with closing(_connect(self.path)) as db:
            return [float(r[0]) for r in db.execute(q, args).fetchall()]

    def rows(self, family: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM trials" + (" WHERE family=?" if family else "") + " ORDER BY id DESC LIMIT ?"  # noqa: S608
        args: tuple[Any, ...] = (family, limit) if family else (limit,)
        with closing(_connect(self.path)) as db:
            return [dict(r) for r in db.execute(q, args).fetchall()]


class HoldoutLookRefused(PermissionError):
    pass


class HoldoutVault:
    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(_connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS holdout (
                    name TEXT PRIMARY KEY, start TEXT NOT NULL, end TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS looks (
                    id INTEGER PRIMARY KEY, holdout TEXT NOT NULL, family TEXT NOT NULL, spec_hash TEXT NOT NULL,
                    candidate TEXT NOT NULL, at TEXT NOT NULL, pid INTEGER NOT NULL, result TEXT,
                    UNIQUE(holdout, family, spec_hash));
                CREATE TRIGGER IF NOT EXISTS holdout_immutable BEFORE UPDATE ON holdout
                    BEGIN SELECT RAISE(ABORT, 'holdout interval is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS holdout_no_delete BEFORE DELETE ON holdout
                    BEGIN SELECT RAISE(ABORT, 'holdout interval is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS looks_no_delete BEFORE DELETE ON looks
                    BEGIN SELECT RAISE(ABORT, 'holdout looks are permanent'); END;
            """)

    def register(self, name: str, start: date, end: date) -> tuple[date, date]:
        """Define the holdout once; later calls return the stored interval (arguments are ignored)."""
        with closing(_connect(self.path)) as db, db:
            row = db.execute("SELECT start,end FROM holdout WHERE name=?", (name,)).fetchone()
            if row is None:
                db.execute("INSERT INTO holdout(name,start,end,created) VALUES (?,?,?,?)",
                           (name, start.isoformat(), end.isoformat(), datetime.now(UTC).isoformat()))
                return start, end
            return date.fromisoformat(row["start"]), date.fromisoformat(row["end"])

    def interval(self, name: str) -> tuple[date, date] | None:
        with closing(_connect(self.path)) as db:
            row = db.execute("SELECT start,end FROM holdout WHERE name=?", (name,)).fetchone()
        return None if row is None else (date.fromisoformat(row["start"]), date.fromisoformat(row["end"]))

    def claim(self, name: str, family: str, spec: dict[str, Any], candidate: str) -> int:
        """Commit a look BEFORE evaluating. Raises if this spec already looked or the family budget is spent."""
        sh = spec_hash(spec)
        with closing(_connect(self.path)) as db, db:
            if db.execute("SELECT 1 FROM holdout WHERE name=?", (name,)).fetchone() is None:
                raise HoldoutLookRefused(f"holdout {name!r} is not registered")
            if db.execute("SELECT 1 FROM looks WHERE holdout=? AND family=? AND spec_hash=?",
                          (name, family, sh)).fetchone():
                raise HoldoutLookRefused(f"{family}/{sh} already evaluated on holdout {name!r} (one look only)")
            used = db.execute("SELECT COUNT(*) FROM looks WHERE holdout=? AND family=?", (name, family)).fetchone()[0]
            if used >= HOLDOUT_LOOKS_PER_FAMILY:
                raise HoldoutLookRefused(f"family {family!r} has used its {HOLDOUT_LOOKS_PER_FAMILY} holdout looks")
            cur = db.execute("INSERT INTO looks(holdout,family,spec_hash,candidate,at,pid) VALUES (?,?,?,?,?,?)",
                             (name, family, sh, candidate, datetime.now(UTC).isoformat(), os.getpid()))
            return int(cur.lastrowid or 0)

    def record_result(self, look_id: int, result: dict[str, Any]) -> None:
        with closing(_connect(self.path)) as db, db:
            row = db.execute("SELECT result FROM looks WHERE id=?", (look_id,)).fetchone()
            if row is None or row["result"] is not None:
                raise HoldoutLookRefused("look missing or result already recorded")
            db.execute("UPDATE looks SET result=? WHERE id=?", (json.dumps(result, default=str), look_id))

    def looks(self, name: str | None = None) -> list[dict[str, Any]]:
        with closing(_connect(self.path)) as db:
            q = "SELECT * FROM looks" + (" WHERE holdout=?" if name else "") + " ORDER BY id"  # noqa: S608
            return [dict(r) for r in db.execute(q, (name,) if name else ()).fetchall()]
