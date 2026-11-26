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
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

HOLDOUT_LOOKS_PER_FAMILY = 3
"""Hard budget of holdout evaluations per strategy family (protects against holdout shopping)."""


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:20]


def variant_name(spec: dict[str, Any]) -> str:
    """Canonical variant label (sorted key=value pairs); walk-forward rows are recorded as WALK_FORWARD:<label>."""
    return ",".join(f"{k}={v}" for k, v in sorted(spec.items()))


_REPO = Path(__file__).resolve().parents[2]
RESULT_DEPENDENCIES: tuple[str, ...] = (
    "services/research/costs.py", "services/research/stats.py", "services/market_data/calendar.py",
    "services/market_data/edgar/parsers.py", "services/market_data/edgar/store.py",
    "services/positions/policy.py", "packages/contracts/policy.py")
"""Modules outside quantlab whose code changes research results (cost model, statistics, calendar, policy
defaults used by the execution tier and the simulator)."""

METHODOLOGY_FILES: tuple[str, ...] = (
    "backtest.py", "costs.py", "factors.py", "events.py", "gates.py", "research.py", "validation.py", "panel.py",
    "strategies/momentum.py", "strategies/features.py")
"""``factors.agent_family`` builds the agent family (targets, backtest config, benchmark), so its code is covered."""
"""The code that decides an agent factor's verdict: a change here is new evidence for a hypothesis."""


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()[:16]


def code_hash() -> str:
    root = Path(__file__).resolve().parent
    return _hash_files(sorted(root.rglob("*.py")) + [_REPO / d for d in RESULT_DEPENDENCIES])


def methodology_hash() -> str:
    root = Path(__file__).resolve().parent
    return _hash_files([root / f for f in METHODOLOGY_FILES] + [_REPO / d for d in RESULT_DEPENDENCIES])


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

    def prior_outcomes(self, family: str, spec: dict[str, Any],
                       canonicalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Earlier development-gate outcomes (walk-forward rows) of exactly this spec in this family, newest first.
        With ``canonicalize``, specs recorded under an older spelling that canonicalize to ``spec`` match too.
        Used to refuse re-testing a hypothesis on the same evidence (see :func:`duplicate_on_same_evidence`)."""
        labels = {"WALK_FORWARD:" + variant_name(spec)}
        with closing(_connect(self.path)) as db:
            if canonicalize is not None:
                for (raw,) in db.execute("SELECT DISTINCT spec FROM trials WHERE family=? AND stage='DEV'", (family,)):
                    try:
                        old = json.loads(raw)
                        if canonicalize(old) == spec:
                            labels.add("WALK_FORWARD:" + variant_name(old))
                    except Exception:  # noqa: S112 - unparsable legacy rows simply do not match
                        continue
            marks = ",".join("?" * len(labels))
            rows = db.execute("SELECT at, status, note, data_fingerprint, code_hash FROM trials WHERE family=? AND "  # noqa: S608
                              f"stage='WF' AND variant IN ({marks}) ORDER BY id DESC",
                              (family, *sorted(labels))).fetchall()
        return [dict(r) for r in rows]

    def evaluated_with(self, family: str, specs: list[dict[str, Any]], data_fingerprint: str, code: str) -> bool:
        """True when EVERY spec already has a development trial on this exact data and code AND an earlier run of
        the family completed (walk-forward, gate, summary) on the same data and code. An interrupted run, an
        errored variant or a changed input therefore never causes a skip."""
        if not specs:
            return False
        want = {spec_hash(s) for s in specs}
        with closing(_connect(self.path)) as db:
            got = {r[0] for r in db.execute(
                "SELECT DISTINCT spec_hash FROM trials WHERE family=? AND stage='DEV' AND data_fingerprint=? AND "
                "code_hash=? AND status='OK'", (family, data_fingerprint, code)).fetchall()}
            done = db.execute("SELECT 1 FROM trials WHERE family=? AND stage='RUN' AND status='DONE' AND "
                              "data_fingerprint=? AND code_hash=? LIMIT 1", (family, data_fingerprint, code)).fetchone()
        return want <= got and done is not None

    def rows(self, family: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM trials" + (" WHERE family=?" if family else "") + " ORDER BY id DESC LIMIT ?"  # noqa: S608
        args: tuple[Any, ...] = (family, limit) if family else (limit,)
        with closing(_connect(self.path)) as db:
            return [dict(r) for r in db.execute(q, args).fetchall()]


def duplicate_on_same_evidence(prior: list[dict[str, Any]], *, fingerprint: str, dev_fingerprint: str,
                               methodology: str = "") -> dict[str, Any] | None:
    """The earlier outcome that makes a re-test redundant: the same development-period data (``dev:<hash>``) under
    the same methodology (``m:<hash>``), or — for rows recorded before those keys existed — the same full-panel
    fingerprint. Corrected or extended development data, or a methodology fix (backtester, costs, gates), is new
    evidence; a delta that only adds rows after the holdout start, or a re-based adjusted history, is not."""
    for row in prior:
        fp = str(row.get("data_fingerprint") or "")
        parts = fp.split("|")
        keyed = any(x.startswith("dev:") for x in parts)
        if not keyed and fp == fingerprint:
            return row
        same_dev = bool(dev_fingerprint) and f"dev:{dev_fingerprint}" in parts
        recorded_m = next((x[2:] for x in parts if x.startswith("m:")), None)
        if same_dev and (recorded_m is None or not methodology or recorded_m == methodology):
            return row
    return None


class HoldoutLookRefused(PermissionError):
    pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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

    def open_look(self, name: str, look_id: int) -> dict[str, Any] | None:
        """The committed look ``look_id`` on holdout ``name`` if it has no result yet, else None."""
        with closing(_connect(self.path)) as db:
            row = db.execute("SELECT * FROM looks WHERE id=? AND holdout=? AND result IS NULL",
                             (int(look_id), name)).fetchone()
        return dict(row) if row else None

    def unresolved_look(self, name: str, family: str, spec: dict[str, Any]) -> int | None:
        """This spec's own committed look that never received a result, taken over by this process ONLY if the
        process that claimed it is gone (a lease): a concurrent evaluation of the same look is refused."""
        with closing(_connect(self.path)) as db, db:
            row = db.execute("SELECT id, pid FROM looks WHERE holdout=? AND family=? AND spec_hash=? AND result IS NULL",
                             (name, family, spec_hash(spec))).fetchone()
            if row is None:
                return None
            owner = int(row["pid"])
            if owner != os.getpid() and _pid_alive(owner):
                raise HoldoutLookRefused(f"look {row['id']} is being evaluated by live process {owner}")
            cur = db.execute("UPDATE looks SET pid=? WHERE id=? AND result IS NULL AND pid=?",
                             (os.getpid(), int(row["id"]), owner))
            if cur.rowcount != 1:
                raise HoldoutLookRefused(f"look {row['id']} changed while resuming")
            return int(row["id"])

    def record_result(self, look_id: int, result: dict[str, Any]) -> None:
        with closing(_connect(self.path)) as db, db:
            cur = db.execute("UPDATE looks SET result=? WHERE id=? AND result IS NULL",  # atomic: first writer wins
                             (json.dumps(result, default=str), look_id))
            if cur.rowcount != 1:
                raise HoldoutLookRefused("look missing or result already recorded")

    def looks(self, name: str | None = None) -> list[dict[str, Any]]:
        with closing(_connect(self.path)) as db:
            q = "SELECT * FROM looks" + (" WHERE holdout=?" if name else "") + " ORDER BY id"  # noqa: S608
            return [dict(r) for r in db.execute(q, (name,) if name else ()).fetchall()]
