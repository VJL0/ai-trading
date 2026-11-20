"""Hash-chained, append-only audit log plus shared system-state helpers (halts, entries lock).

Each audit row stores ``hash = sha256(prev_hash || canonical(row))`` so tampering with any earlier
row breaks ``verify_chain``. Rows are written only inside a writer transaction.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from .canonical import canonical_json, sha256_hex
from .db import Database, utc_iso
from .ids import new_id

GENESIS = "0" * 64


def append_audit(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    source: str,
    mode: str,
    payload: Any,
    causal_id: str | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    policy_version: str | None = None,
    ts: datetime | None = None,
) -> str:
    """Append one audit event using an open writer-transaction connection. Returns event_id."""
    row = conn.execute("SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
    prev_hash = row[0] if row else GENESIS
    event_id = new_id("aud")
    body = {
        "event_id": event_id,
        "ts": utc_iso(ts),
        "event_type": event_type,
        "causal_id": causal_id,
        "source": source,
        "mode": mode,
        "dataset_version": dataset_version,
        "model_version": model_version,
        "policy_version": policy_version,
        "payload": payload,
    }
    payload_json = canonical_json(payload)
    digest = sha256_hex(prev_hash + canonical_json(body))
    conn.execute(
        "INSERT INTO audit_events(event_id, ts, event_type, causal_id, source, mode, dataset_version,"
        " model_version, policy_version, payload_json, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id, body["ts"], event_type, causal_id, source, mode, dataset_version, model_version,
            policy_version, payload_json, prev_hash, digest,
        ),
    )
    return event_id


def verify_chain(db: Database) -> tuple[bool, int, str]:
    """Recompute the chain. Returns (ok, rows_checked, detail)."""
    rows = db.query("SELECT * FROM audit_events ORDER BY seq")
    prev = GENESIS
    for r in rows:
        body = {
            "event_id": r["event_id"],
            "ts": r["ts"],
            "event_type": r["event_type"],
            "causal_id": r["causal_id"],
            "source": r["source"],
            "mode": r["mode"],
            "dataset_version": r["dataset_version"],
            "model_version": r["model_version"],
            "policy_version": r["policy_version"],
            "payload": json.loads(r["payload_json"]),
        }
        if r["prev_hash"] != prev:
            return False, r["seq"], f"prev_hash break at seq {r['seq']}"
        expected = sha256_hex(prev + canonical_json(body))
        if expected != r["hash"]:
            return False, r["seq"], f"hash mismatch at seq {r['seq']}"
        prev = r["hash"]
    return True, len(rows), "ok"


# ------------------------------------------------------------------ system state / halts


def set_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO system_state(key, value_json, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
        (key, canonical_json(value), utc_iso()),
    )


def get_state(db: Database, key: str, default: Any = None) -> Any:
    row = db.query_one("SELECT value_json FROM system_state WHERE key=?", (key,))
    return json.loads(row["value_json"]) if row else default


def add_halt(conn: sqlite3.Connection, *, reason: str, source: str, mode: str) -> str:
    halt_id = new_id("halt")
    conn.execute(
        "INSERT INTO trading_halts(halt_id, reason, source, created_at) VALUES (?,?,?,?)",
        (halt_id, reason, source, utc_iso()),
    )
    append_audit(conn, event_type="HALT_ADDED", source=source, mode=mode, payload={"halt_id": halt_id, "reason": reason})
    return halt_id


def clear_halt(conn: sqlite3.Connection, halt_id: str, *, by: str, reason: str, mode: str) -> bool:
    cur = conn.execute(
        "UPDATE trading_halts SET cleared_at=?, cleared_by=?, clear_reason=? WHERE halt_id=? AND cleared_at IS NULL",
        (utc_iso(), by, reason, halt_id),
    )
    if cur.rowcount:
        append_audit(conn, event_type="HALT_CLEARED", source=by, mode=mode, payload={"halt_id": halt_id, "reason": reason})
    return cur.rowcount > 0


def active_halts(db: Database) -> list[dict[str, Any]]:
    return db.query("SELECT * FROM trading_halts WHERE cleared_at IS NULL ORDER BY created_at")


def record_failure(conn: sqlite3.Connection, *, component: str, kind: str, detail: str, causal_id: str | None = None) -> None:
    conn.execute(
        "INSERT INTO failures(ts, component, kind, detail, causal_id) VALUES (?,?,?,?,?)",
        (utc_iso(), component, kind, detail[:4000], causal_id),
    )
