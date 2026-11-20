"""Versioned, immutable risk-policy store and the authenticated operator change flow (spec §6).

* A policy's version id is the SHA-256 of its canonical JSON (``RiskPolicy.version``); a stored row's
  content can never change (DB triggers in 0001_core.sql + 0300_risk_policy_guards.sql). An edit is a
  *new* row; activation is the only mutable flag and flips atomically inside one writer transaction.
* ``active()`` re-hashes the stored JSON and refuses (raises) on any mismatch: fail closed.
* ``apply_change`` is the ONLY way to change limits. It requires the human operator token
  (``OperatorAuth``), validates the proposal against the hard ceilings in ``limits``, writes the new
  version, flips ``is_active``, appends a hash-chained ``POLICY_CHANGED`` audit event with a
  field-level diff and a ``risk_policy_changes`` row. Rejected attempts (wrong token, unsafe policy)
  are audited as ``POLICY_CHANGE_REJECTED`` and raise.
* Models and the dashboard get ``read_only_view()`` (a plain JSON dict) or a ``PolicyReader``; neither
  exposes a write capability. The operator token is never logged, stored in the DB or audited.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypeVar

from pydantic import ValidationError

from packages.contracts.enums import TradingMode
from packages.contracts.policy import RiskPolicy
from packages.core.audit import append_audit
from packages.core.canonical import canonical_json
from packages.core.clock import Clock, SystemClock
from packages.core.db import Database, utc_iso
from packages.core.ids import new_id

from . import limits
from .operator_auth import OperatorAuth, OperatorAuthError

BOOTSTRAP_ACTOR: Final = "system-bootstrap"
AUDIT_SOURCE: Final = "risk.policy_store"
ACTIVATION_EVENTS: Final[tuple[str, str]] = ("POLICY_BOOTSTRAPPED", "POLICY_CHANGED")
RESERVED_ACTORS: Final[frozenset[str]] = frozenset({BOOTSTRAP_ACTOR, "system", "ai", "model", "agent"})
MAX_ACTOR_CHARS: Final = 100
MAX_REASON_CHARS: Final = 500
_T = TypeVar("_T")
CHANGE_PROCEDURE: Final = (
    "Risk limits are read-only for models, the GUI agent and the dashboard. Only the human operator can "
    "change them, through the authenticated local settings flow (operator token, var/<namespace>/keys/"
    "operator.token). Every change creates a new immutable, audited policy version."
)


class PolicyStoreError(RuntimeError):
    """Base class: the policy store cannot provide a trustworthy policy. Callers must fail closed."""


class PolicyIntegrityError(PolicyStoreError):
    """Stored policy content does not re-hash to its version id (tampering or corruption)."""


class PolicyUnavailableError(PolicyStoreError):
    """No active policy exists."""


class PolicyChangeRejected(ValueError):
    """A proposed policy change was refused (unsafe, invalid or no-op). Carries every violation."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        super().__init__("policy change rejected: " + "; ".join(self.violations))


# ---------------------------------------------------------------------------- helpers


def verify_policy_row(row: sqlite3.Row | dict[str, Any]) -> RiskPolicy:
    """Parse a ``risk_policies`` row and prove its content hashes to its version (else raise)."""
    version = str(row["policy_version"])
    raw = str(row["policy_json"])
    try:
        policy = RiskPolicy.model_validate_json(raw)
    except ValidationError as exc:
        raise PolicyIntegrityError(f"stored policy {version} does not parse: {exc.error_count()} error(s)") from exc
    if policy.canonical_json() != raw:
        raise PolicyIntegrityError(f"stored policy {version} is not in canonical form (content altered)")
    if policy.version != version:
        raise PolicyIntegrityError(f"stored policy content hashes to {policy.version}, not its version id {version}")
    if policy.name != str(row["name"]):
        raise PolicyIntegrityError(f"stored policy {version} name column does not match its content")
    return policy


def check_activation_audit_conn(conn: sqlite3.Connection, version: str) -> None:
    """Prove the active version was activated by ``PolicyStore`` (else raise ``PolicyIntegrityError``).

    ``is_active`` is the one mutable column, so a flip made by any other code path would be unaudited. The
    newest hash-chained activation event (POLICY_BOOTSTRAPPED / POLICY_CHANGED, always written in the same
    transaction as the flip) must name exactly the active version.
    """
    row = conn.execute(
        "SELECT event_id, policy_version FROM audit_events WHERE event_type IN (?, ?) ORDER BY seq DESC LIMIT 1",
        ACTIVATION_EVENTS,
    ).fetchone()
    if row is None:
        raise PolicyIntegrityError(f"active policy {version} has no activation audit event (activated outside the audited flow)")
    if str(row["policy_version"]) != version:
        raise PolicyIntegrityError(
            f"active policy {version} does not match the last audited activation ({row['policy_version']}, "
            f"event {row['event_id']}); activated outside the audited flow"
        )


def active_policy_conn(conn: sqlite3.Connection) -> RiskPolicy:
    """Read, integrity-check and activation-check the active policy on one connection/snapshot."""
    rows = list(conn.execute("SELECT * FROM risk_policies WHERE is_active = 1").fetchall())
    if not rows:
        raise PolicyUnavailableError("no active risk policy; trading is blocked until one is installed")
    if len(rows) != 1:  # pragma: no cover - prevented by the unique partial index
        raise PolicyIntegrityError(f"{len(rows)} active policies")
    policy = verify_policy_row(rows[0])
    check_activation_audit_conn(conn, policy.version)
    return policy


def policy_diff(old: RiskPolicy | None, new: RiskPolicy) -> dict[str, Any]:
    """Field-level diff (JSON values). List-valued fields also report added/removed members."""
    before: dict[str, Any] = old.model_dump(mode="json") if old is not None else {}
    after: dict[str, Any] = new.model_dump(mode="json")
    diff: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        a = before.get(key)
        b = after.get(key)
        if a == b:
            continue
        entry: dict[str, Any] = {"from": a, "to": b}
        if isinstance(a, list) and isinstance(b, list):
            a_items = [str(i) for i in a]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            b_items = [str(i) for i in b]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            entry["added"] = sorted(set(b_items) - set(a_items))
            entry["removed"] = sorted(set(a_items) - set(b_items))
        diff[key] = entry
    return diff


def _history_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        verify_policy_row(row)
        integrity_ok = True
    except PolicyIntegrityError:
        integrity_ok = False
    return {
        "policy_version": row["policy_version"],
        "name": row["name"],
        "created_at": row["created_at"],
        "created_by": row["created_by"],
        "approved_by": row["approved_by"],
        "supersedes": row["supersedes"],
        "is_active": bool(row["is_active"]),
        "integrity_ok": integrity_ok,
    }


# ---------------------------------------------------------------------------- read-only access


class PolicyReader:
    """Read-only access to the policy store. This is what AI/dashboard code may hold.

    It keeps only the database *path* and opens SQLite in ``mode=ro`` for every query, so it holds no
    reference to the single writer and SQLite itself refuses any write through it.
    """

    __slots__ = ("_path",)

    def __init__(self, db: Database | Path) -> None:
        self._path = Path(db.path if isinstance(db, Database) else db)

    @property
    def path(self) -> Path:
        return self._path

    def _read(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run ``fn`` inside one read-only snapshot transaction (all its SELECTs see the same state)."""
        conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, timeout=10, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            try:
                return fn(conn)
            finally:
                conn.execute("ROLLBACK")
        finally:
            conn.close()

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self._read(lambda conn: list(conn.execute(sql, params).fetchall()))

    def active(self) -> RiskPolicy:
        """The active policy, proven intact *and* activated through the audited flow (else raise)."""
        return self._read(active_policy_conn)

    def active_version(self) -> str:
        return self.active().version

    def get(self, version: str) -> RiskPolicy:
        rows = self._query("SELECT * FROM risk_policies WHERE policy_version = ?", (version,))
        if not rows:
            raise PolicyUnavailableError(f"unknown policy version {version}")
        return verify_policy_row(rows[0])

    def history(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM risk_policies ORDER BY created_at, rowid")
        return [_history_row(r) for r in rows]

    def changes(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM risk_policy_changes ORDER BY at DESC, rowid DESC LIMIT ?", (max(1, min(limit, 1000)),)
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = dict(r)
            item["diff"] = json.loads(item.pop("diff_json"))
            item["violations"] = json.loads(item.pop("violations_json"))
            out.append(item)
        return out

    def read_only_view(self) -> dict[str, Any]:
        """Plain JSON-safe dict for models and the dashboard. A fresh copy; it has no write path."""
        policy = self.active()
        view: dict[str, Any] = {
            "read_only": True,
            "policy_version": policy.version,
            "policy": policy.model_dump(mode="json"),
            "hard_ceilings": limits.hard_ceilings(),
            "structural_invariants": list(limits.STRUCTURAL_INVARIANTS),
            "engine_constants": {
                "max_context_age_seconds": format(limits.MAX_CONTEXT_AGE_SECONDS, "f"),
                "max_account_snapshot_age_seconds": format(limits.MAX_ACCOUNT_SNAPSHOT_AGE_SECONDS, "f"),
                "max_proposal_data_age_seconds": format(limits.MAX_PROPOSAL_DATA_AGE_SECONDS, "f"),
                "max_through_touch_bps": format(limits.MAX_THROUGH_TOUCH_BPS, "f"),
                "min_stop_distance_bps": format(limits.MIN_STOP_DISTANCE_BPS, "f"),
            },
            "change_procedure": CHANGE_PROCEDURE,
            "note": "protective stops are not guaranteed execution prices; planned loss is an estimate",
        }
        return json.loads(canonical_json(view))


# ---------------------------------------------------------------------------- store (writes)


class PolicyStore:
    """Policy persistence. Writes happen only in ``ensure_default`` and ``apply_change``."""

    def __init__(
        self,
        db: Database,
        mode: TradingMode,
        *,
        operator_auth: OperatorAuth | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._db = db
        self._mode = mode
        # Default: the token lives next to the ledger, var/<namespace>/keys/operator.token.
        self._auth = operator_auth or OperatorAuth(Path(db.path).parent / "keys")
        self._clock: Clock = clock or SystemClock()
        self._reader = PolicyReader(db)

    @property
    def mode(self) -> TradingMode:
        return self._mode

    def reader(self) -> PolicyReader:
        return self._reader

    # ------------------------------------------------------------------ reads
    def active(self) -> RiskPolicy:
        return self._reader.active()

    def active_version(self) -> str:
        return self._reader.active_version()

    def get(self, version: str) -> RiskPolicy:
        return self._reader.get(version)

    def history(self) -> list[dict[str, Any]]:
        return self._reader.history()

    def changes(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._reader.changes(limit)

    def read_only_view(self) -> dict[str, Any]:
        return self._reader.read_only_view()

    # ------------------------------------------------------------------ bootstrap
    def ensure_default(self) -> RiskPolicy:
        """Install ``RiskPolicy()`` as the active policy if none is active. Returns the active policy."""
        default = RiskPolicy()
        violations = limits.validate_policy(default)
        if violations:  # a code change broke the defaults: refuse rather than install an unsafe policy
            raise PolicyChangeRejected(["default policy violates hard limits", *violations])
        now = utc_iso(self._now())

        def _tx(conn: sqlite3.Connection) -> None:
            row = conn.execute("SELECT * FROM risk_policies WHERE is_active = 1").fetchone()
            if row is not None:
                active_policy_conn(conn)  # intact and audited, else raise: the runtime fails closed
                return
            existing = conn.execute(
                "SELECT * FROM risk_policies WHERE policy_version = ?", (default.version,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO risk_policies(policy_version, name, policy_json, created_at, created_by,"
                    " approved_by, supersedes, is_active) VALUES (?,?,?,?,?,?,NULL,0)",
                    (default.version, default.name, default.canonical_json(), now, BOOTSTRAP_ACTOR, BOOTSTRAP_ACTOR),
                )
            else:
                verify_policy_row(existing)
            conn.execute("UPDATE risk_policies SET is_active = 1 WHERE policy_version = ?", (default.version,))
            append_audit(
                conn, event_type="POLICY_BOOTSTRAPPED", source=AUDIT_SOURCE, mode=self._mode.value,
                payload={"policy_version": default.version, "actor": BOOTSTRAP_ACTOR,
                         "reactivated": existing is not None},
                policy_version=default.version,
            )

        self._db.write(_tx)
        return self.active()

    # ------------------------------------------------------------------ the only change path
    def apply_change(self, new_policy: RiskPolicy, *, operator_token: str, actor: str, reason: str) -> str:
        """Authenticate the human operator, validate, store a new immutable version and activate it.

        Raises ``OperatorAuthError`` (wrong/missing token) or ``PolicyChangeRejected`` (unsafe/invalid/
        no-op). Every attempt — accepted or rejected — is audited. Returns the new active version.
        """
        actor_s = actor.strip() if isinstance(actor, str) else ""  # pyright: ignore[reportUnnecessaryIsInstance]
        reason_s = reason.strip() if isinstance(reason, str) else ""  # pyright: ignore[reportUnnecessaryIsInstance]
        presented = operator_token if isinstance(operator_token, str) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        # Free text goes into the immutable audit log: never let a pasted token be persisted.
        actor_s = self._auth.redact(actor_s, presented)
        reason_s = self._auth.redact(reason_s, presented)
        if not self._auth.verify(operator_token):
            self._audit_rejection(
                actor_s or "unknown", reason_s, None, ["operator authentication failed"],
                token_presented=bool(operator_token),
            )
            raise OperatorAuthError("operator authentication failed; policy unchanged")

        problems: list[str] = []
        if not actor_s or len(actor_s) > MAX_ACTOR_CHARS:
            problems.append(f"actor is required (1-{MAX_ACTOR_CHARS} chars)")
        elif actor_s.lower() in RESERVED_ACTORS:
            problems.append(f"actor name {actor_s!r} is reserved")
        if not reason_s or len(reason_s) > MAX_REASON_CHARS:
            problems.append(f"reason is required (1-{MAX_REASON_CHARS} chars)")
        candidate: RiskPolicy | None = None
        if not isinstance(new_policy, RiskPolicy):  # pyright: ignore[reportUnnecessaryIsInstance]
            problems.append("new_policy must be a RiskPolicy")
        else:
            try:
                # 1) python-mode re-validation of the raw attribute values: defeats model_construct() and
                #    float smuggling (JSON-mode dumping would launder a float into a decimal string first);
                # 2) a JSON round trip so the stored form is exactly the canonical, re-parseable one.
                RiskPolicy.model_validate(new_policy.model_dump(warnings=False))
                candidate = RiskPolicy.model_validate_json(json.dumps(new_policy.model_dump(mode="json", warnings=False)))
            except (ValidationError, TypeError, ValueError) as exc:
                problems.append(f"policy does not validate: {exc}")
        if candidate is not None:
            problems.extend(limits.validate_policy(candidate))
        if problems or candidate is None:
            self._audit_rejection(actor_s or "unknown", reason_s, getattr(candidate, "version", None), problems)
            raise PolicyChangeRejected(problems or ["invalid policy"])

        policy = candidate
        now = utc_iso(self._now())

        def _tx(conn: sqlite3.Connection) -> str:
            active_row = conn.execute("SELECT * FROM risk_policies WHERE is_active = 1").fetchone()
            old: RiskPolicy | None = verify_policy_row(active_row) if active_row is not None else None
            if old is not None and old.version == policy.version:
                raise PolicyChangeRejected(["proposed policy is identical to the active policy"])
            existing = conn.execute(
                "SELECT * FROM risk_policies WHERE policy_version = ?", (policy.version,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO risk_policies(policy_version, name, policy_json, created_at, created_by,"
                    " approved_by, supersedes, is_active) VALUES (?,?,?,?,?,?,?,0)",
                    (policy.version, policy.name, policy.canonical_json(), now, actor_s, actor_s,
                     old.version if old is not None else None),
                )
            else:
                verify_policy_row(existing)  # re-activating an earlier, intact version
            if old is not None:
                conn.execute("UPDATE risk_policies SET is_active = 0 WHERE policy_version = ?", (old.version,))
            conn.execute("UPDATE risk_policies SET is_active = 1 WHERE policy_version = ?", (policy.version,))
            diff = policy_diff(old, policy)
            from_version = old.version if old is not None else None
            payload: dict[str, Any] = {
                "actor": actor_s, "reason": reason_s,
                "from_version": from_version, "to_version": policy.version,
                "reactivated_existing_version": existing is not None, "diff": diff,
            }
            audit_id = append_audit(
                conn, event_type="POLICY_CHANGED", source=AUDIT_SOURCE, mode=self._mode.value,
                payload=payload, policy_version=policy.version,
            )
            self._insert_change(
                conn, at=now, actor=actor_s, result="ACCEPTED", from_version=from_version,
                to_version=policy.version, reason=reason_s, diff=diff, violations=[], audit_id=audit_id,
            )
            return policy.version

        try:
            return self._db.write(_tx)
        except PolicyChangeRejected as exc:
            self._audit_rejection(actor_s, reason_s, policy.version, exc.violations)
            raise

    # ------------------------------------------------------------------ internals
    def _now(self) -> datetime:
        return self._clock.now()

    @staticmethod
    def _insert_change(
        conn: sqlite3.Connection,
        *,
        at: str,
        actor: str,
        result: str,
        from_version: str | None,
        to_version: str | None,
        reason: str,
        diff: dict[str, Any],
        violations: list[str],
        audit_id: str,
    ) -> None:
        conn.execute(
            "INSERT INTO risk_policy_changes(change_id, at, actor, result, from_version, to_version, reason,"
            " diff_json, violations_json, audit_event_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (new_id("polchg"), at, actor[:MAX_ACTOR_CHARS], result, from_version, to_version,
             reason[:MAX_REASON_CHARS], canonical_json(diff), canonical_json(violations), audit_id),
        )

    def _audit_rejection(
        self,
        actor: str,
        reason: str,
        proposed_version: str | None,
        violations: list[str],
        *,
        token_presented: bool = True,
    ) -> None:
        now = utc_iso(self._now())

        def _tx(conn: sqlite3.Connection) -> None:
            row = conn.execute("SELECT policy_version FROM risk_policies WHERE is_active = 1").fetchone()
            active_version = str(row["policy_version"]) if row is not None else None
            payload = {
                "actor": actor[:MAX_ACTOR_CHARS], "reason": reason[:MAX_REASON_CHARS],
                "active_version": active_version, "proposed_version": proposed_version,
                "violations": violations, "operator_token_presented": token_presented,
            }
            audit_id = append_audit(
                conn, event_type="POLICY_CHANGE_REJECTED", source=AUDIT_SOURCE, mode=self._mode.value,
                payload=payload, policy_version=active_version,
            )
            self._insert_change(
                conn, at=now, actor=actor or "unknown", result="REJECTED", from_version=active_version,
                to_version=proposed_version, reason=reason or "(none)", diff={}, violations=violations,
                audit_id=audit_id,
            )

        self._db.write(_tx)
