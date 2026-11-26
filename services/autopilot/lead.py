"""Scheduling and supervision of the lead agent, the autopilot's only AI participant.

The autopilot stays the deterministic floor: it requests and guards the day's PAPER session and runs the data
pipeline, the shadow journal and the holdout gate whether or not any model answers. On top of that it launches one
*lead cycle* at a time as a background child (``python -m services.agents lead``) and never waits for it, so a
slow, failing or rate-limited lead can delay nothing on the trading path.

* **When** (:func:`lead_due`, pure): from the trading day's pre-open until the evening research window ends; the
  first turn of a trading day, the lead's own requested wake time, new material evidence in the autopilot's event
  log (results, alerts, decisions, stand-down changes) or the maximum interval. Never more often than
  ``LEAD_MIN_INTERVAL_MIN``, never more than ``LEAD_MAX_CYCLES`` per day, never in stand-down.
* **What it sees** (:func:`write_briefing`): a compact snapshot of read-only tool outputs and the autopilot's own
  jobs, alerts and events, with execution labels. The file is data for the lead; nothing reads it back.
* **What comes back** (:func:`parse_cycle`): the child's JSON summary. Only two fail-safe session requests are
  recognized (:data:`SESSION_ACTIONS`); everything else is recorded and ignored. The supervisor validates each
  request against the current trading day and executes it through the guarded backend.

This module never imports the crew (``services.agents``): the boundary is the child process and its JSON output.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

from .research import StepOutcome

LEAD: Final = "lead"
LEAD_ACTION_JOB: Final = "lead_action"
K_LEAD: Final = "lead_state"
K_LEAD_ACTIONS: Final = "lead_actions"
SESSION_ACTIONS: Final = frozenset({"stand_down_today", "request_halt"})
# Event kinds that count as new evidence for the lead (its own events are excluded by job name).
EVIDENCE_KINDS: Final = ("RESULT", "ALERT", "DECISION", "STANDDOWN", "STANDDOWN_CLEARED")
_CONTROL: Final = re.compile(r"[\x00-\x1f\x7f]")
_MAX_REASON: Final = 200


def _d(v: Any) -> dict[str, Any]:
    return cast(dict[str, Any], v) if isinstance(v, dict) else {}


def _l(v: Any) -> list[Any]:
    return cast(list[Any], v) if isinstance(v, list) else []


def _text(v: Any, limit: int = _MAX_REASON) -> str:
    return _CONTROL.sub(" ", str(v)).strip()[:limit]


@dataclass(frozen=True)
class LeadState:
    """Per trading day: cycles launched, last launch, the lead's requested wake time, the evidence watermark."""

    day: str | None = None
    cycles: int = 0
    last_launch: datetime | None = None
    next_wake: datetime | None = None
    evidence_seq: int = 0

    @classmethod
    def load(cls, raw: Any, parse: Any) -> LeadState:
        r = _d(raw)
        return cls(r.get("day"), int(r.get("cycles") or 0), parse(r.get("last_launch")), parse(r.get("next_wake")),
                   int(r.get("evidence_seq") or 0))

    def dump(self) -> dict[str, Any]:
        return {"day": self.day, "cycles": self.cycles,
                "last_launch": self.last_launch.isoformat() if self.last_launch else None,
                "next_wake": self.next_wake.isoformat() if self.next_wake else None,
                "evidence_seq": self.evidence_seq}


def lead_due(state: LeadState, *, day: str, now: datetime, evidence_seq: int, max_cycles: int,
             min_interval: timedelta, max_interval: timedelta) -> tuple[bool, str]:
    """(due, reason). Pure: the same state and clock always give the same answer."""
    cycles = state.cycles if state.day == day else 0
    if cycles >= max_cycles:
        return False, f"lead turns for {day} used up ({cycles}/{max_cycles})"
    last = state.last_launch
    if last is not None and now - last < min_interval:
        return False, f"next lead turn not before {(last + min_interval).isoformat()}"
    if state.day != day or last is None:
        return True, f"first lead turn for {day}"
    if state.next_wake is not None and now >= state.next_wake:
        return True, "the lead's requested wake time"
    if evidence_seq > state.evidence_seq:
        return True, "new evidence since the last lead turn"
    if now - last >= max_interval:
        return True, f"no lead turn for {int(max_interval.total_seconds() // 60)} min"
    wake = state.next_wake.isoformat() if state.next_wake else (last + max_interval).isoformat()
    return False, f"lead idle until {wake} or new evidence"


def write_briefing(path: Path, briefing: Mapping[str, Any]) -> Path:
    """Atomic, owner-only write. The briefing carries no credential: only tool outputs and autopilot state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(briefing, fh, default=str, sort_keys=True, indent=1)
    os.replace(tmp, path)
    return path


def compact_session(status: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("session_id", "state", "requested_by", "started_at", "ends_at", "ended_at", "cycles_run", "trades",
            "stop_reason", "next_cycle_at", "last_cycle_at", "use_ai")
    return {"active": {k: _d(status.get("active")).get(k) for k in keys} if status.get("active") else None,
            "recent": [{k: _d(s).get(k) for k in keys} for s in _l(status.get("recent"))[:3]],
            "market_open": status.get("market_open"), "next_open": status.get("next_open")}


def compact_risk(risk: Mapping[str, Any]) -> dict[str, Any]:
    return {"mode": risk.get("mode"), "live_execution_compiled": risk.get("live_execution_compiled"),
            "entries_locked": risk.get("entries_locked"), "lock_reasons": _l(risk.get("lock_reasons"))[:5],
            "exits_locked": risk.get("exits_locked"), "halts": _l(risk.get("halts"))[:5],
            "position_safety": _l(risk.get("position_safety"))[:5], "utilization": _l(risk.get("utilization"))[:8],
            "recent_decisions": _l(risk.get("recent_decisions"))[:5]}


def compact_report(rep: Mapping[str, Any]) -> dict[str, Any]:
    return {"cycles": rep.get("cycles"), "decisions": rep.get("decisions"),
            "top_no_trade_reasons": _l(rep.get("top_no_trade_reasons"))[:5],
            "top_risk_rejections": _l(rep.get("top_risk_rejections"))[:5], "trades": len(_l(rep.get("trades"))),
            "narrative": _text(rep.get("narrative") or "", 600)}


def compact_health(health: Mapping[str, Any], feed: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    degraded = [{k: _d(c).get(k) for k in ("name", "status", "detail")} for c in _l(health.get("components"))
                if _d(c).get("status") != "OK"]
    return ({"overall": health.get("overall"), "degraded": degraded[:8]},
            {k: feed.get(k) for k in ("status", "feed_mode", "feed", "detail", "staleness_s", "last_bar_ts")})


PERFORMANCE_KEYS: Final = ("mode_label", "portfolio_value", "cash", "daily_pnl", "total_pnl", "realized_pnl",
                           "unrealized_pnl", "max_drawdown", "current_drawdown", "active_positions", "open_orders")


@dataclass
class CycleSummary:
    ok: bool
    returncode: int | None
    seconds: float
    lead_outcome: str | None = None
    lead_provider: str | None = None
    situation: str = ""
    priorities: list[str] = field(default_factory=list)
    delegated: list[dict[str, Any]] = field(default_factory=list)
    refused: int = 0
    ran: list[dict[str, Any]] = field(default_factory=list)
    session_actions: list[dict[str, str]] = field(default_factory=list)
    next_wake_minutes: int | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {"ok": self.ok, "returncode": self.returncode, "seconds": round(self.seconds, 1),
                "lead_outcome": self.lead_outcome, "lead_provider": self.lead_provider,
                "situation": self.situation, "priorities": self.priorities, "delegated": self.delegated,
                "refused": self.refused, "ran": self.ran, "session_actions": self.session_actions,
                "next_wake_minutes": self.next_wake_minutes, "error": self.error}


def parse_cycle(out: StepOutcome, *, locked_exit: int) -> CycleSummary:
    """The child's JSON summary, reduced to scalars and short strings. Agent text is data, never instructions."""
    s = CycleSummary(ok=out.ok, returncode=out.returncode, seconds=out.seconds)
    if out.returncode == locked_exit:
        s.error = "another crew process holds the crew store; lead cycle skipped"
        return s
    if out.returncode is None:
        s.error = _text(out.tail or "timeout", 200)
        return s
    data = _d(out.json())
    cycle = _d(data.get("lead_cycle"))
    if not out.ok or not cycle:
        s.ok = False
        s.error = _text(data.get("error") or f"lead child exited {out.returncode}", 300)
        return s
    lead = _d(cycle.get("lead"))
    s.lead_outcome = _text(lead.get("outcome") or "", 40) or None
    s.lead_provider = _text(lead.get("provider") or "", 20) or None
    decision = _d(cycle.get("decision"))
    s.situation = _text(decision.get("situation") or "", 600)
    s.priorities = [_text(p, 200) for p in _l(decision.get("priorities"))[:8]]
    deleg = _d(cycle.get("delegations"))
    s.delegated = [{k: _text(_d(a).get(k) or "", 120) for k in ("task_id", "kind", "role")}
                   for a in _l(deleg.get("accepted"))[:12]]
    s.refused = len(_l(deleg.get("refused")))
    s.ran = [{k: _text(_d(r).get(k) or "", 40) for k in ("task_id", "kind", "provider", "outcome")}
             for r in _l(cycle.get("ran"))[:24]]
    for sa in _l(cycle.get("session_actions"))[:2]:
        action = str(_d(sa).get("action") or "")
        reason = _text(_d(sa).get("reason") or "")
        if action in SESSION_ACTIONS and reason:
            s.session_actions.append({"action": action, "reason": reason})
    wake = cycle.get("next_wake_minutes")
    s.next_wake_minutes = wake if isinstance(wake, int) and not isinstance(wake, bool) else None
    return s
