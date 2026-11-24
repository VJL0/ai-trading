"""Reversible strategy lifecycle registry (append-only transitions, evidence-gated promotions).

    RESEARCH -> DEV_PASSED -> VALIDATED -> SHADOW -> PAPER
         \\-> REJECTED      \\-> REJECTED            PAPER -> SUSPENDED -> SHADOW | RETIRED

* DEV_PASSED needs a passing development report (``gates.dev_verdict``) whose spec hash matches.
* VALIDATED needs a recorded holdout look for that spec hash that passed ``gates.holdout_verdict``.
* SHADOW is automatic after validation: prospective signals are journaled, no orders.
* PAPER needs: SHADOW for ``SHADOW_GATE['min_sessions']`` sessions with tracking within tolerance, the asset
  class execution-eligible in the latest capability snapshot, and an execution lifecycle that supports the
  strategy's holding period. PAPER strategies still only *propose*; the risk engine authorizes every order.
* Any degradation (drawdown/drift breach, capability loss, operator request) suspends immediately.

The current state is the last transition; nothing is ever updated in place or deleted.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeGuard

from .gates import SHADOW_GATE


class State(StrEnum):
    RESEARCH = "RESEARCH"
    DEV_PASSED = "DEV_PASSED"
    VALIDATED = "VALIDATED"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    SUSPENDED = "SUSPENDED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


ALLOWED: dict[State, frozenset[State]] = {
    State.RESEARCH: frozenset({State.DEV_PASSED, State.REJECTED, State.RETIRED}),
    State.DEV_PASSED: frozenset({State.VALIDATED, State.REJECTED, State.RETIRED}),
    State.VALIDATED: frozenset({State.SHADOW, State.RETIRED}),
    State.SHADOW: frozenset({State.PAPER, State.SUSPENDED, State.RETIRED}),
    State.PAPER: frozenset({State.SUSPENDED, State.RETIRED}),
    State.SUSPENDED: frozenset({State.SHADOW, State.RETIRED}),
    State.REJECTED: frozenset({State.RETIRED}),
    State.RETIRED: frozenset(),
}


class TransitionRefused(PermissionError):
    pass


class StrategyRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._db()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY, family TEXT NOT NULL, spec_hash TEXT NOT NULL, spec TEXT NOT NULL,
                    asset_class TEXT NOT NULL, holding TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY, strategy_id TEXT NOT NULL, at TEXT NOT NULL, from_state TEXT,
                    to_state TEXT NOT NULL, reason TEXT NOT NULL, evidence TEXT NOT NULL, actor TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS shadow (
                    id INTEGER PRIMARY KEY, strategy_id TEXT NOT NULL, session TEXT NOT NULL, decided_at TEXT NOT NULL,
                    targets TEXT NOT NULL, realized REAL, replayed REAL, UNIQUE(strategy_id, session));
                CREATE TABLE IF NOT EXISTS shadow_marks (
                    strategy_id TEXT NOT NULL, session TEXT NOT NULL, net REAL NOT NULL, gross REAL NOT NULL,
                    cost REAL NOT NULL, replayed REAL NOT NULL, provenance TEXT NOT NULL,
                    PRIMARY KEY(strategy_id, session));
                CREATE TRIGGER IF NOT EXISTS marks_no_update BEFORE UPDATE ON shadow_marks
                    BEGIN SELECT RAISE(ABORT, 'prospective marks are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS marks_no_delete BEFORE DELETE ON shadow_marks
                    BEGIN SELECT RAISE(ABORT, 'prospective marks are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS tr_no_update BEFORE UPDATE ON transitions
                    BEGIN SELECT RAISE(ABORT, 'transitions are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS tr_no_delete BEFORE DELETE ON transitions
                    BEGIN SELECT RAISE(ABORT, 'transitions are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS st_no_delete BEFORE DELETE ON strategies
                    BEGIN SELECT RAISE(ABORT, 'strategies are permanent'); END;
            """)
            if "metadata" not in {r[1] for r in db.execute("PRAGMA table_info(shadow)")}:
                db.execute("ALTER TABLE shadow ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            db.executescript("""
                CREATE TRIGGER IF NOT EXISTS shadow_decision_immutable
                    BEFORE UPDATE OF strategy_id,session,decided_at,targets,metadata ON shadow
                    BEGIN SELECT RAISE(ABORT, 'prospective decisions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS shadow_no_delete BEFORE DELETE ON shadow
                    BEGIN SELECT RAISE(ABORT, 'prospective decisions are permanent'); END;
                CREATE TRIGGER IF NOT EXISTS shadow_settle_once BEFORE UPDATE OF realized,replayed ON shadow
                    WHEN OLD.realized IS NOT NULL
                    BEGIN SELECT RAISE(ABORT, 'prospective settlements are immutable'); END;
            """)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # ------------------------------------------------------------------ reads
    def state(self, strategy_id: str) -> State | None:
        with closing(self._db()) as db:
            row = db.execute("SELECT to_state FROM transitions WHERE strategy_id=? ORDER BY id DESC LIMIT 1",
                             (strategy_id,)).fetchone()
        return None if row is None else State(row[0])

    def get(self, strategy_id: str) -> dict[str, Any] | None:
        with closing(self._db()) as db:
            row = db.execute("SELECT * FROM strategies WHERE strategy_id=?", (strategy_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["spec"] = json.loads(out["spec"])
        out["state"] = str(self.state(strategy_id))
        return out

    def list(self) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            ids = [r[0] for r in db.execute("SELECT strategy_id FROM strategies ORDER BY created").fetchall()]
        return [s for i in ids if (s := self.get(i)) is not None]

    def transitions(self, strategy_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        q = ("SELECT * FROM transitions" + (" WHERE strategy_id=?" if strategy_id else "")  # noqa: S608 constant SQL
             + " ORDER BY id DESC LIMIT ?")
        args: tuple[Any, ...] = (strategy_id, limit) if strategy_id else (limit,)
        with closing(self._db()) as db:
            rows = [dict(r) for r in db.execute(q, args).fetchall()]
        for r in rows:
            r["evidence"] = json.loads(r["evidence"])
        return rows

    def in_state(self, *states: State) -> list[dict[str, Any]]:
        want = {str(s) for s in states}
        return [s for s in self.list() if s["state"] in want]

    # ------------------------------------------------------------------ writes
    def register(self, strategy_id: str, *, family: str, spec_hash: str, spec: dict[str, Any], asset_class: str,
                 holding: str, actor: str = "quantlab") -> None:
        with closing(self._db()) as db, db:
            if db.execute("SELECT 1 FROM strategies WHERE strategy_id=?", (strategy_id,)).fetchone():
                return
            db.execute("INSERT INTO strategies VALUES (?,?,?,?,?,?,?)",
                       (strategy_id, family, spec_hash, json.dumps(spec, sort_keys=True, default=str), asset_class,
                        holding, datetime.now(UTC).isoformat()))
            db.execute("INSERT INTO transitions(strategy_id,at,from_state,to_state,reason,evidence,actor) "
                       "VALUES (?,?,?,?,?,?,?)", (strategy_id, datetime.now(UTC).isoformat(), None,
                                                  str(State.RESEARCH), "registered", "{}", actor))

    def transition(self, strategy_id: str, to: State, *, reason: str, evidence: dict[str, Any],
                   actor: str = "quantlab") -> None:
        cur = self.state(strategy_id)
        if cur is None:
            raise TransitionRefused(f"unknown strategy {strategy_id}")
        if to not in ALLOWED[cur]:
            raise TransitionRefused(f"{strategy_id}: {cur} -> {to} is not an allowed transition")
        info = self.get(strategy_id) or {}
        self._check_prerequisites(info, cur, to, evidence)
        with closing(self._db()) as db, db:
            db.execute("INSERT INTO transitions(strategy_id,at,from_state,to_state,reason,evidence,actor) "
                       "VALUES (?,?,?,?,?,?,?)", (strategy_id, datetime.now(UTC).isoformat(), str(cur), str(to),
                                                  reason[:500], json.dumps(evidence, default=str), actor))

    def _check_prerequisites(self, info: dict[str, Any], cur: State, to: State, ev: dict[str, Any]) -> None:
        sh = info.get("spec_hash")
        if to == State.DEV_PASSED:
            if not ev.get("dev_passed") or ev.get("spec_hash") != sh:
                raise TransitionRefused("DEV_PASSED requires a passing development report for this exact spec")
        elif to == State.VALIDATED:
            if not ev.get("holdout_passed") or ev.get("spec_hash") != sh or not ev.get("look_id"):
                raise TransitionRefused("VALIDATED requires a recorded, passing one-use holdout look for this spec")
        elif to == State.PAPER:
            sessions = self.shadow_sessions(info["strategy_id"])
            if sessions < SHADOW_GATE["min_sessions"]:
                raise TransitionRefused(f"PAPER requires >= {int(SHADOW_GATE['min_sessions'])} shadow sessions "
                                        f"(have {sessions})")
            if not ev.get("execution_eligible"):
                raise TransitionRefused("PAPER requires the asset class to be execution-eligible "
                                        "(capability snapshot + risk policy)")
            if not ev.get("lifecycle_supported"):
                raise TransitionRefused("PAPER requires an execution lifecycle that supports this holding period")
            gap = ev.get("tracking_gap_bps")
            if not _finite(gap) or gap < 0 or gap > SHADOW_GATE["max_tracking_gap_bps"]:
                raise TransitionRefused("PAPER requires shadow tracking within tolerance")

    # ------------------------------------------------------------------ shadow journal
    def record_shadow(self, strategy_id: str, session: str, targets: dict[str, float], *,
                      metadata: dict[str, Any] | None = None, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("shadow decision time must be timezone-aware")
        if any(not _finite(w) or w < 0 for w in targets.values()) or sum(targets.values()) > 1 + 1e-9:
            raise ValueError("shadow target weights must be finite, long-only and unleveraged")
        if metadata:
            if metadata.get("model_version") != "shadow-after-cost-v2":
                raise ValueError("unknown prospective model")
            complete = datetime.fromisoformat(metadata["data_complete_at"])
            execution = datetime.fromisoformat(metadata["execution_open"])
            if not complete <= now < execution:
                raise ValueError("prospective decision requires complete data before a future execution open")
            if metadata.get("decision_kind") not in ("REBALANCE", "HOLD"):
                raise ValueError("invalid prospective decision kind")
        with closing(self._db()) as db, db:
            try:
                db.execute("INSERT INTO shadow(strategy_id,session,decided_at,targets,metadata) VALUES (?,?,?,?,?)",
                           (strategy_id, session, now.isoformat(), json.dumps(targets, sort_keys=True, allow_nan=False),
                            json.dumps(metadata or {}, sort_keys=True, allow_nan=False)))
            except sqlite3.IntegrityError:
                return False
        return True

    def settle_shadow(self, strategy_id: str, session: str, realized: float, replayed: float | None) -> None:
        if not _finite(realized) or not _finite(replayed):
            raise ValueError("non-finite shadow settlement")
        with closing(self._db()) as db, db:
            db.execute("UPDATE shadow SET realized=?, replayed=? WHERE strategy_id=? AND session=? AND realized IS NULL",
                       (realized, replayed, strategy_id, session))

    def shadow_sessions(self, strategy_id: str) -> int:
        """Count after-cost, provenance-checked daily marks; legacy gross settlements earn no promotion credit."""
        with closing(self._db()) as db:
            return int(db.execute("SELECT COUNT(*) FROM shadow_marks WHERE strategy_id=?",
                                  (strategy_id,)).fetchone()[0])

    def record_shadow_mark(self, strategy_id: str, session: str, *, net: float, gross: float, cost: float,
                           replayed: float, provenance: dict[str, Any]) -> bool:
        if not all(_finite(x) for x in (net, gross, cost, replayed)) or cost < 0 or net < -1 or gross < -1:
            raise ValueError("invalid after-cost prospective mark")
        with closing(self._db()) as db, db:
            decision = db.execute("SELECT decided_at,metadata FROM shadow WHERE id=? AND strategy_id=?",
                                  (provenance.get("decision_id"), strategy_id)).fetchone()
            if decision is None:
                raise ValueError("prospective mark requires its recorded decision")
            metadata = json.loads(decision["metadata"])
            if metadata.get("model_version") != "shadow-after-cost-v2":
                raise ValueError("legacy shadow decisions cannot produce promotion evidence")
            op = datetime.fromisoformat(provenance["execution_open"])
            close = datetime.fromisoformat(provenance["data_complete_at"])
            observed = datetime.fromisoformat(provenance["observed_at"])
            if not datetime.fromisoformat(decision["decided_at"]) < op < close <= observed:
                raise ValueError("prospective mark must follow a committed decision and a completed session")
            if metadata["execution_session"] > session:
                raise ValueError("decision was not executable during the marked session")
            for name in ("model_version", "code_hash", "config_hash"):
                if provenance.get(name) != metadata.get(name):
                    raise ValueError("prospective mark provenance differs from the decision")
            try:
                db.execute("INSERT INTO shadow_marks VALUES (?,?,?,?,?,?,?)",
                           (strategy_id, session, net, gross, cost, replayed,
                            json.dumps(provenance, sort_keys=True, allow_nan=False)))
            except sqlite3.IntegrityError:
                return False
        return True

    def shadow_marks(self, strategy_id: str) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM shadow_marks WHERE strategy_id=? ORDER BY session",
                                              (strategy_id,))]
        for r in rows:
            r["provenance"] = json.loads(r["provenance"])
        return rows

    def shadow_rows(self, strategy_id: str, limit: int = 60) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM shadow WHERE strategy_id=? ORDER BY session DESC "
                                                "LIMIT ?", (strategy_id, limit)).fetchall()]
        for r in rows:
            r["targets"] = json.loads(r["targets"])
            r["metadata"] = json.loads(r["metadata"])
        return rows


def _finite(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
