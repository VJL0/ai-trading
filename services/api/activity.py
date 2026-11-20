"""Live operations log: pipeline stage transitions, human-readable activity and decision cards.

Every record is persisted (append-only ``activity_events``) *and* published on the SSE bus, so the read-only dashboard
always shows real backend state and the timeline survives restarts. Writes go through the single DB writer.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any

from packages.core.canonical import canonical_json
from packages.core.clock import Clock
from packages.core.db import Database, utc_iso

from .events import EventBus

STAGES: tuple[str, ...] = ("MARKET_DATA", "FEATURES", "MODELS", "GPT_REVIEW", "RISK_ENGINE", "ORDER_DECISION",
                           "BROKER", "RECONCILIATION")
STATUSES = frozenset({"idle", "running", "passed", "rejected", "error", "skipped"})
KINDS = frozenset({"scan", "predict", "evaluate", "gpt_call", "risk", "reject", "submit", "fill", "reconcile", "session",
                   "agent_tool", "error", "info"})


class ActivityLog:
    def __init__(self, db: Database, bus: EventBus, clock: Clock) -> None:
        self._db, self._bus, self._clock = db, bus, clock
        self._lock = threading.Lock()
        self._cycle: dict[str, Any] | None = None
        self._task: dict[str, Any] | None = None
        self._decision: dict[str, Any] | None = None
        self._t0: dict[str, float] = {}

    # ------------------------------------------------------------------ persistence
    def _persist(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        ts = utc_iso(self._clock.now())
        body = {"ts": ts, **body}

        def _w(conn: Any) -> int:
            cur = conn.execute(
                "INSERT INTO activity_events(ts, kind, cycle_id, causal_id, session_id, stage, status, item, actor,"
                " message, duration_ms, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, kind, body.get("cycle_id"), body.get("causal_id"), body.get("session_id"), body.get("stage"),
                 body.get("status"), body.get("item"), body.get("actor"), body.get("message"), body.get("duration_ms"),
                 canonical_json(body)),
            )
            return int(cur.lastrowid or 0)

        body["seq"] = self._db.write(_w)
        return body

    # ------------------------------------------------------------------ cycles and stages
    def begin_cycle(self, cycle_id: str, causal_id: str, session_id: str | None) -> None:
        with self._lock:
            self._cycle = {"cycle_id": cycle_id, "causal_id": causal_id, "session_id": session_id,
                           "started_at": utc_iso(self._clock.now()),
                           "stages": [{"stage": s, "status": "idle", "item": None, "started_at": None,
                                       "duration_ms": None, "detail": ""} for s in STAGES]}
            self._t0.clear()
            self._decision = None

    def stage(self, stage: str, status: str, *, item: str | None = None, detail: str = "") -> None:
        if stage not in STAGES or status not in STATUSES:
            raise ValueError(f"bad stage/status {stage}/{status}")
        with self._lock:
            cyc = self._cycle or {}
            now_perf = time.perf_counter()
            duration: int | None = None
            if status == "running":
                self._t0[stage] = now_perf
            elif stage in self._t0:
                duration = int((now_perf - self._t0.pop(stage)) * 1000)
            for st in cyc.get("stages", []):
                if st["stage"] == stage:
                    st.update(status=status, item=item or st.get("item"), detail=detail[:500],
                              started_at=st["started_at"] if status != "running" else utc_iso(self._clock.now()),
                              duration_ms=duration if duration is not None else st.get("duration_ms"))
            body = {"cycle_id": cyc.get("cycle_id"), "causal_id": cyc.get("causal_id"),
                    "session_id": cyc.get("session_id"), "stage": stage, "status": status, "item": item,
                    "duration_ms": duration, "detail": detail[:500]}
        self._bus.publish("stage", self._persist("stage", body))

    def activity(self, kind: str, message: str, *, actor: str = "system", causal_id: str | None = None,
                 session_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        if kind not in KINDS:
            kind = "info"
        sess = session_id or (self._cycle or {}).get("session_id")
        body = {"kind": kind, "actor": actor, "message": message[:400], "causal_id": causal_id,
                "session_id": sess, "detail": detail or {}}
        with self._lock:
            self._task = {"message": message[:400], "kind": kind, "since": utc_iso(self._clock.now())}
        self._bus.publish("activity", self._persist("activity", body))

    def decision(self, card: dict[str, Any]) -> None:
        cyc = self._cycle or {}
        card = {"cycle_id": cyc.get("cycle_id"), "causal_id": cyc.get("causal_id"), **card}
        with self._lock:
            self._decision = card
        self._bus.publish("decision", self._persist("decision", card))

    def end_cycle(self) -> None:
        with self._lock:
            self._task = None

    # ------------------------------------------------------------------ reads
    def current(self) -> dict[str, Any]:
        with self._lock:
            cycle = json.loads(json.dumps(self._cycle)) if self._cycle else None
            task = dict(self._task) if self._task else None
            dec = dict(self._decision) if self._decision else None
        if cycle is None:  # after a restart, rebuild the last decision card from the log
            row = self._db.query_one("SELECT payload_json FROM activity_events WHERE kind='decision' ORDER BY seq DESC LIMIT 1")
            dec = json.loads(row["payload_json"]) if row else None
        return {"cycle": cycle, "current_task": task, "decision": dec}

    def recent(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self._db.query("SELECT seq, kind, payload_json FROM activity_events WHERE kind=? ORDER BY seq DESC LIMIT ?",
                                  (kind, limit))
        else:
            rows = self._db.query("SELECT seq, kind, payload_json FROM activity_events ORDER BY seq DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            body = json.loads(r["payload_json"])
            body["seq"], body["kind"] = r["seq"], body.get("kind", r["kind"])
            body.setdefault("event", r["kind"])
            out.append(body)
        return out


def iso(dt: datetime) -> str:
    return utc_iso(dt)
