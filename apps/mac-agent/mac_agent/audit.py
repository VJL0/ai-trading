"""Append-only JSON-lines audit log for GUI-agent policy decisions and actions.

* One line per record, written with a single ``O_APPEND`` write so concurrent writers (the Python
  controller and the Node/Playwright guard in ``browser/src/audit.mjs``) never interleave lines.
* Each writer session keeps a hash chain (``prev`` -> ``hash``) so deletions/edits inside a session
  are detectable; :func:`verify_chain` checks every writer's chain, including Node writers. The
  canonical form is ``JSON(sort_keys, separators=(",", ":"), ensure_ascii=False)`` — the Node writer
  produces byte-identical canonical JSON.
* Records never contain typed text (only its length), secrets, or pixels. Every string value (and
  key) is scrubbed with :func:`packages.core.logging.redact` plus the ``audit_redact_patterns`` from
  ``rules.json`` *before* hashing, so a stored line always verifies.

The authoritative operational audit chain lives in SQLite via ``packages.core.audit``; this file is
the GUI agent's local evidence trail (``apps/mac-agent/artifacts`` is git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packages.core.logging import redact

if TYPE_CHECKING:
    from .policy import Decision

DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "audit.jsonl"
_RULES_PATH = Path(__file__).with_name("rules.json")
GENESIS = "GENESIS"


@lru_cache(maxsize=1)
def _extra_patterns() -> tuple[re.Pattern[str], ...]:
    raw = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    return tuple(re.compile(p) for p in raw["audit_redact_patterns"])


def scrub(text: str) -> str:
    """Redact anything secret-looking from one string."""
    text = redact(text)
    for pat in _extra_patterns():
        text = pat.sub("[REDACTED]", text)
    return text


def scrub_value(value: Any) -> Any:
    """Recursively scrub every string (keys included) of a JSON-like value."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {scrub(str(k)): scrub_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_value(v) for v in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    # floats/Decimals/paths/enums: stringify so both writers hash identical text
    return scrub(str(value))


def canonical_json(body: dict[str, Any]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_hash(prev: str, body: dict[str, Any]) -> str:
    return hashlib.sha256((prev + canonical_json(body)).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class NullAuditLog:
    """Discards records (used when a caller explicitly opts out, e.g. pure classification helpers)."""

    def record_decision(self, decision: Decision, *, actor: str, policy_version: str) -> None:
        return None

    def record_event(self, event: str, **fields: Any) -> None:
        return None


class AuditLog:
    def __init__(self, path: Path = DEFAULT_AUDIT_PATH, *, writer: str = "py") -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer_id = f"{writer}-{secrets.token_hex(6)}"
        self._prev = GENESIS
        self._seq = 0
        self._lock = threading.Lock()

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            body = scrub_value({"ts": _now(), "writer": self.writer_id, "seq": self._seq, **record})
            prev = self._prev
            digest = chain_hash(prev, body)
            stored = {**body, "prev": prev, "hash": digest}
            line = canonical_json(stored) + "\n"
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
            self._prev = digest
            return stored

    def record_decision(self, decision: Decision, *, actor: str, policy_version: str) -> None:
        self._append({"kind": "policy_decision", "actor": actor, "policy_version": policy_version,
                      **decision.as_record()})

    def record_event(self, event: str, **fields: Any) -> None:
        self._append({"kind": "event", "event": event, **fields})


def read_records(path: Path = DEFAULT_AUDIT_PATH) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify_chain(path: Path = DEFAULT_AUDIT_PATH) -> dict[str, int]:
    """Verify each writer session's hash chain. Returns {writer_id: records}; raises ValueError on breaks."""
    last: dict[str, str] = {}
    counts: dict[str, int] = {}
    for rec in read_records(path):
        writer = str(rec.get("writer", "?"))
        prev = rec.get("prev")
        digest = rec.get("hash")
        body = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
        expected_prev = last.get(writer, GENESIS)
        if prev != expected_prev:
            raise ValueError(f"audit chain break for writer {writer} at seq {rec.get('seq')}")
        if chain_hash(str(prev), body) != digest:
            raise ValueError(f"audit hash mismatch for writer {writer} at seq {rec.get('seq')}")
        last[writer] = str(digest)
        counts[writer] = counts.get(writer, 0) + 1
    return counts
