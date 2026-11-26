"""Pre-open and intraday health verdicts, computed deterministically from tool outputs.

Tool outputs are DATA: halt reasons and lock reasons are only truncated and recorded, never interpreted as
instructions. A verdict is one of:

* ``ok``: every hard check passed (warnings may still be listed);
* ``stand_down``: an active halt (any scope) or the operator kill switch: the autopilot takes no action until the
  operator clears it;
* otherwise failed: do not start; record the reasons and raise an alert.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, cast

HARD_COMPONENTS: Final = ("broker", "reconciliation", "database", "engine", "scheduler")
FEED_OK: Final = "OK"
FEED_PREOPEN_TOLERATED: Final = frozenset({"STALE"})  # no regular-hours prints yet; the risk gate re-checks per cycle
_KILL_SWITCH_REASON: Final = re.compile(r"^halt \S+ \(scope=ALL, exits blocked\):")
_MAX_REASON: Final = 200


def _short(text: Any) -> str:
    return str(text)[:_MAX_REASON]


def _list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


@dataclass
class Verdict:
    ok: bool = True
    stand_down: bool = False
    kill_switch: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(_short(reason))

    def public(self) -> dict[str, Any]:
        return {"ok": self.ok, "stand_down": self.stand_down, "kill_switch": self.kill_switch,
                "reasons": self.reasons, "warnings": self.warnings, "provenance": self.provenance}


def halt_state(risk: Mapping[str, Any]) -> tuple[bool, bool, list[str]]:
    """(halted, kill_switch, reasons) from ``get_risk_status``. Any active halt halts; an ALL-scope halt (the
    operator kill switch, which also blocks protective exits) is reported separately."""
    halts = [_dict(h) for h in _list(risk.get("halts"))]
    exit_reasons = [str(r) for r in _list(risk.get("exit_lock_reasons"))]
    kill_reasons = [r for r in exit_reasons if _KILL_SWITCH_REASON.match(r)]
    reasons = [_short(f"halt {h.get('halt_id')} (source={h.get('source')}): {h.get('reason')}") for h in halts]
    if not halts:
        reasons.extend(_short(r) for r in kill_reasons)
    return bool(halts or kill_reasons), bool(kill_reasons), reasons


def _provenance(*outputs: Mapping[str, Any]) -> dict[str, Any]:
    labels = sorted({str(o.get("execution_label")) for o in outputs if o.get("execution_label")})
    return {"execution_labels": labels, "is_simulated": any(bool(o.get("is_simulated")) for o in outputs),
            "asof": max((str(o.get("asof")) for o in outputs if o.get("asof")), default=None)}


def preopen_verdict(*, risk: Mapping[str, Any], health: Mapping[str, Any], feed: Mapping[str, Any],
                    market_open: bool) -> Verdict:
    v = Verdict(provenance=_provenance(risk, health, feed))
    v.provenance["feed_mode"] = feed.get("feed_mode")
    v.provenance["feed"] = feed.get("feed")
    halted, kill, halt_reasons = halt_state(risk)
    if halted:
        v.ok = False
        v.stand_down = True
        v.kill_switch = kill
        v.reasons.extend(halt_reasons or ["an active halt"])
        return v
    if risk.get("live_execution_compiled") is not False:
        v.fail("live execution is not reported compiled off")
    if str(risk.get("mode")) != "PAPER":
        v.fail(f"API mode is {risk.get('mode')!r}, not PAPER")
    if "LIVE" in v.provenance["execution_labels"]:
        v.fail("an output carried the LIVE execution label")
    if risk.get("entries_locked") is not False:
        locks = [str(r) for r in _list(risk.get("lock_reasons"))] or ["entries locked (no reason given)"]
        for r in locks[:5]:
            v.fail(f"entries locked: {r}")
    for ps in _list(risk.get("position_safety"))[:5]:
        p = _dict(ps)
        v.fail(f"position {p.get('symbol')} {p.get('state')}: {p.get('reason')}")
    if health.get("overall") == "DOWN":
        v.fail("system health DOWN")
    comps = {str(_dict(c).get("name")): _dict(c) for c in _list(health.get("components"))}
    for name in HARD_COMPONENTS:
        c = comps.get(name)
        if c is None:
            v.fail(f"health component {name} missing")
        elif c.get("status") != "OK":
            v.fail(f"health {name} {c.get('status')}: {c.get('detail')}")
    status = str(feed.get("status"))
    if status != FEED_OK:
        if status in FEED_PREOPEN_TOLERATED and not market_open:
            v.warnings.append(_short(f"feed {status} before the open: {feed.get('detail')}"))
        else:
            v.fail(f"feed {status}: {feed.get('detail')}")
    return v


def intraday_verdict(*, risk: Mapping[str, Any], status: Mapping[str, Any]) -> Verdict:
    """Observation only: halts mean stand down; a failed/interrupted session raises an alert. No actions."""
    v = Verdict(provenance=_provenance(risk, status))
    halted, kill, halt_reasons = halt_state(risk)
    if halted:
        v.ok = False
        v.stand_down = True
        v.kill_switch = kill
        v.reasons.extend(halt_reasons or ["an active halt"])
    for s in [_dict(status.get("active")), *(_dict(x) for x in _list(status.get("recent"))[:3])]:
        if s.get("state") in ("FAILED", "INTERRUPTED") and s.get("requested_by") == "agent:autopilot":
            v.warnings.append(_short(f"session {s.get('session_id')} {s.get('state')}: {s.get('stop_reason')}"))
    if risk.get("entries_locked") and not halted:
        v.warnings.extend(_short(f"entries locked: {r}") for r in _list(risk.get("lock_reasons"))[:3])
    return v
