"""Deterministic answer to "why didn't you trade?" (no LLM): the last N decision cycles from the audit trail with
their outcome, candidate, exact NO_TRADE reason or failed risk checks, AI-review status and entries-lock reasons,
plus an aggregate and a plain-English summary built from those facts only."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Final

from packages.core.db import Database

MAX_LIMIT: Final = 50
TEXT_CAP: Final = 300
ENTRIES_CHECK: Final = "health_entries_unlocked"


def _cap(text: Any, n: int = TEXT_CAP) -> str | None:
    if text is None:
        return None
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _placeholders(items: Iterable[Any]) -> str:
    return ",".join("?" for _ in items)


def failed_checks(decision: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for c in decision.get("checks") or []:
        if isinstance(c, dict) and not c.get("passed", True):
            out.append((str(c.get("name")), _cap(c.get("detail"), 200) or ""))
    return out


def _by_causal(db: Database, sql: str, ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not ids:
        return out
    for r in db.query(sql.format(ph=_placeholders(ids)), tuple(ids)):
        out.setdefault(str(r["causal_id"]), []).append(r)
    return out


def cycle_facts(db: Database, causal_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Per causal id: proposal NO_TRADE reason, failed risk checks, session lock state, GPT_REVIEW stage status."""
    props = _by_causal(db, "SELECT causal_id, payload_json FROM proposals WHERE causal_id IN ({ph}) ORDER BY created_at",
                       causal_ids)
    decs = _by_causal(db, "SELECT causal_id, payload_json FROM risk_decisions WHERE causal_id IN ({ph})"
                          " ORDER BY evaluated_at", causal_ids)
    sess = _by_causal(db, "SELECT causal_id, session_id, entries_locked, lock_reasons_json, note FROM session_cycles"
                          " WHERE causal_id IN ({ph}) ORDER BY seq", causal_ids)
    stages = _by_causal(db, "SELECT causal_id, status, payload_json FROM activity_events WHERE causal_id IN ({ph})"
                            " AND kind='stage' AND stage='GPT_REVIEW' ORDER BY seq", causal_ids)
    facts: dict[str, dict[str, Any]] = {}
    for cid in causal_ids:
        f: dict[str, Any] = {"no_trade_reason": None, "proposer": None, "failed_checks": [], "entries_locked": None,
                             "lock_reasons": [], "ai_review": None, "note": None, "session_id": None}
        if cid in props:
            p = json.loads(props[cid][-1]["payload_json"])
            f["no_trade_reason"] = _cap(p.get("no_trade_reason"))
            f["proposer"] = p.get("proposer")
        if cid in decs:
            f["failed_checks"] = failed_checks(json.loads(decs[cid][-1]["payload_json"]))
        if cid in sess:
            row = sess[cid][-1]
            f["session_id"] = row["session_id"]
            f["entries_locked"] = None if row["entries_locked"] is None else bool(row["entries_locked"])
            f["lock_reasons"] = [_cap(x, 200) for x in json.loads(row["lock_reasons_json"] or "[]")][:8]
            f["note"] = _cap(row["note"], 200)
        else:
            locks = [d for n, d in f["failed_checks"] if n == ENTRIES_CHECK]
            if locks:
                f["entries_locked"], f["lock_reasons"] = True, locks[:8]
        if cid in stages:
            st = stages[cid][-1]
            detail = json.loads(st["payload_json"]).get("detail") or ""
            f["ai_review"] = _cap(f"{st['status']}: {detail}" if detail else st["status"], 200)
        facts[cid] = f
    return facts


def _top(counter: Counter[str], key: str, n: int = 5) -> list[dict[str, Any]]:
    return [{key: k, "count": v} for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def explain_decisions(db: Database, limit: int = 10, *, entries: tuple[bool, list[str]] | None = None,
                      now: datetime | None = None) -> dict[str, Any]:
    limit = max(1, min(int(limit), MAX_LIMIT))
    rows = db.query("SELECT seq, causal_id, payload_json FROM audit_events WHERE event_type='DECISION_CYCLE'"
                    " ORDER BY seq DESC LIMIT ?", (limit,))
    briefs = [(r, json.loads(r["payload_json"])) for r in rows]
    ids = [str(b.get("causal_id") or r["causal_id"]) for r, b in briefs]
    facts = cycle_facts(db, ids)
    cycles: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    checks: Counter[str] = Counter()
    for (_, b), cid in zip(briefs, ids, strict=True):
        f = facts.get(cid, {})
        outcome = str(b.get("outcome") or "UNKNOWN")
        outcomes[outcome] += 1
        no_trade = f.get("no_trade_reason") or (_cap(b.get("reason")) if outcome == "NO_TRADE" else None)
        if outcome == "NO_TRADE" and no_trade:
            reasons[no_trade] += 1
        fc: list[tuple[str, str]] = f.get("failed_checks") or []
        if outcome in ("REJECTED", "BLOCKED"):
            for name, _ in fc:
                checks[name] += 1
        cycles.append({
            "causal_id": cid, "ran_at": b.get("ran_at"), "session_id": b.get("session_id") or f.get("session_id"),
            "outcome": outcome, "candidate_symbol": b.get("candidate_symbol"), "reason": _cap(b.get("reason")),
            "no_trade_reason": no_trade, "failed_checks": [f"{n}: {d}" if d else n for n, d in fc][:12],
            "entries_locked": f.get("entries_locked"), "lock_reasons": f.get("lock_reasons") or [],
            "ai_review": f.get("ai_review"), "note": f.get("note"), "client_order_id": b.get("client_order_id"),
        })
    locked_now = entries[0] if entries is not None else None
    lock_now = [str(_cap(x, 200)) for x in (entries[1] if entries is not None else [])][:8]
    aggregate = {
        "cycles": len(cycles), "by_outcome": dict(sorted(outcomes.items())),
        "top_no_trade_reasons": _top(reasons, "reason"), "top_failed_checks": _top(checks, "check"),
        "entries_locked_now": locked_now, "lock_reasons_now": lock_now,
    }
    return {"asof": None if now is None else now.isoformat().replace("+00:00", "Z"), "answer": _answer(cycles, aggregate),
            "cycles": cycles, "aggregate": aggregate}


def _answer(cycles: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    parts: list[str] = []
    if not cycles:
        parts.append("No decision cycle has run yet, so nothing was evaluated (start a paper session or run a quant "
                     "cycle).")
    else:
        counts = ", ".join(f"{n} {o}" for o, n in sorted(agg["by_outcome"].items(), key=lambda kv: (-kv[1], kv[0])))
        parts.append(f"Last {agg['cycles']} decision cycle(s): {counts}.")
        if agg["by_outcome"].get("SUBMITTED", 0) == 0:
            parts.append("No order was submitted.")
        if agg["top_no_trade_reasons"]:
            r = agg["top_no_trade_reasons"][0]
            parts.append(f"Most common NO_TRADE reason: \"{r['reason']}\" ({r['count']}x).")
        if agg["top_failed_checks"]:
            c = ", ".join(f"{x['check']} ({x['count']}x)" for x in agg["top_failed_checks"][:3])
            parts.append(f"Risk engine rejections (final, not retried): {c}.")
        last = cycles[0]
        why = last.get("no_trade_reason") or ("; ".join(last["failed_checks"][:2]) if last["failed_checks"] else None) \
            or last.get("reason") or "no reason recorded"
        cand = f" on {last['candidate_symbol']}" if last.get("candidate_symbol") else ""
        parts.append(f"Most recent ({last.get('ran_at')}): {last['outcome']}{cand} - {why}.")
    if agg["entries_locked_now"] is True:
        parts.append("New entries are locked right now: " + ("; ".join(agg["lock_reasons_now"][:3]) or "unknown reason")
                     + ".")
    elif agg["entries_locked_now"] is False:
        parts.append("New entries are currently unlocked.")
    return " ".join(parts)[:1200]
