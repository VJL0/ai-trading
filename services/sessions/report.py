"""Session report (``GET /api/v1/sessions/{id}/report``): decision breakdown, top NO_TRADE reasons, top risk
rejections (aggregated from persisted risk_decisions checks), trades with P&L from fills, API cost from model_runs,
and a plain-English narrative built deterministically from those numbers (no LLM)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from packages.core.db import Database

from .calendar import et_label, parse_utc
from .explain import failed_checks
from .performance import ZERO, api_cost, compute_performance, fifo, latest_positions, load_fills, position_marks

CORE_OUTCOMES: Final[tuple[str, ...]] = ("NO_TRADE", "REJECTED", "SUBMITTED")


def _ph(n: int) -> str:
    return ",".join("?" for _ in range(n))


def _money(x: Decimal) -> str:
    q = x.quantize(Decimal("0.01"))
    return f"+{q}" if q > 0 else f"{q}"


def session_trades(db: Database, causal_ids: list[str], *, cut: datetime) -> list[dict[str, Any]]:
    """Orders created by the session's cycles, with P&L from fills: realized (FIFO) for SELL orders, unrealized of
    the still-open lots for BUY orders (marked at the latest broker position price; null without a mark)."""
    if not causal_ids:
        return []
    orders = db.query(f"SELECT client_order_id, symbol, side, quantity, filled_quantity, avg_fill_price, state,"  # noqa: S608
                      f" created_at FROM orders WHERE causal_id IN ({_ph(len(causal_ids))}) ORDER BY created_at",
                      tuple(causal_ids))
    if not orders:
        return []
    fills = load_fills(db, until=cut)
    result = fifo(fills)
    by_order: dict[str, list[str]] = {}
    for f in fills:
        by_order.setdefault(f.client_order_id, []).append(f.fill_id)
    marks = position_marks(latest_positions(db, cut) or [])
    out: list[dict[str, Any]] = []
    for o in orders:
        coid, side = str(o["client_order_id"]), str(o["side"]).upper()
        pnl: Decimal | None = None
        basis: str | None = None
        if by_order.get(coid):
            if side == "SELL":
                pnl = sum((result.realized_by_fill.get(fid, ZERO) for fid in by_order[coid]), ZERO)
                basis = "realized"
            else:
                lots = [lot for lot in result.open_lots.get(str(o["symbol"]), []) if lot.client_order_id == coid]
                mark = marks.get(str(o["symbol"]))
                if not lots:
                    pnl, basis = ZERO, "closed"  # realized P&L of its lots is booked on the closing SELL order(s)
                elif mark is not None:
                    pnl = sum((lot.quantity * (mark - lot.price) for lot in lots), ZERO)
                    basis = "unrealized"
        out.append({"client_order_id": coid, "symbol": o["symbol"], "side": side, "qty": o["quantity"],
                    "filled_qty": o["filled_quantity"], "avg_price": o["avg_fill_price"], "state": o["state"],
                    "pnl": None if pnl is None else format(pnl, "f"), "pnl_basis": basis})
    return out


def risk_rejections(db: Database, causal_ids: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not causal_ids:
        return counts
    rows = db.query(f"SELECT payload_json FROM risk_decisions WHERE verdict='REJECTED' AND causal_id IN"  # noqa: S608
                    f" ({_ph(len(causal_ids))})", tuple(causal_ids))
    for r in rows:
        for name, _ in failed_checks(json.loads(r["payload_json"])):
            counts[name] += 1
    return counts


def build_report(db: Database, session: dict[str, Any], cycles: list[dict[str, Any]], *, now: datetime, label: str,
                 is_simulated: bool) -> dict[str, Any]:
    started = parse_utc(session.get("started_at")) or parse_utc(session.get("created_at")) or now
    ended = parse_utc(session.get("ended_at"))
    # Completed reports are historical records. A restarted synthetic replay can have
    # an earlier clock; it must not move the report cutoff before its own session.
    cut = ended if ended is not None else now
    causal_ids = [str(c["causal_id"]) for c in cycles if c.get("causal_id")]
    decisions: Counter[str] = Counter({k: 0 for k in CORE_OUTCOMES})
    reasons: Counter[str] = Counter()
    for c in cycles:
        decisions[str(c["outcome"])] += 1
        if c["outcome"] == "NO_TRADE" and c.get("reason"):
            reasons[" ".join(str(c["reason"]).split())[:300]] += 1
    rejections = risk_rejections(db, causal_ids)
    trades = session_trades(db, causal_ids, cut=cut)
    cost = api_cost(db, causal_ids=causal_ids)
    perf = compute_performance(db, now=now, label=label, is_simulated=is_simulated, since=started, until=cut)
    top_reasons = [{"reason": k, "count": v} for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    top_rej = [{"check": k, "count": v} for k, v in sorted(rejections.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    report = {
        "session": session, "performance": perf, "cycles": len(cycles),
        "decisions": dict(decisions), "top_no_trade_reasons": top_reasons, "top_risk_rejections": top_rej,
        "trades": trades, "api_cost_usd": format(cost, "f"),
    }
    report["narrative"] = narrative(session, report, started=started, cut=cut)
    return report


def narrative(session: dict[str, Any], rep: dict[str, Any], *, started: datetime, cut: datetime) -> str:
    perf = rep["performance"]
    label = perf["mode_label"]
    sid, state = session["session_id"], session["state"]
    syms = ", ".join(session.get("symbols") or [])
    span = f"from {et_label(started)} to {et_label(cut)}"
    status = {"RUNNING": "is still running", "STOPPING": "is stopping", "COMPLETED": "completed its planned window",
              "STOPPED": "was stopped", "INTERRUPTED": "was interrupted", "FAILED": "failed"}.get(state, state.lower())
    parts = [f"{label} paper session {sid} on {syms} {status} ({span})."]
    if session.get("stop_reason") and state != "RUNNING":
        parts.append(f"Reason: {session['stop_reason']}.")
    n = rep["cycles"]
    if n == 0:
        parts.append("No decision cycle ran (cycles run only during XNYS regular hours).")
    else:
        d = rep["decisions"]
        extra = ", ".join(f"{v} {k}" for k, v in sorted(d.items()) if k not in CORE_OUTCOMES and v)
        parts.append(f"It ran {n} decision cycle(s): {d.get('NO_TRADE', 0)} NO_TRADE, {d.get('REJECTED', 0)} rejected "
                     f"by the risk engine, {d.get('SUBMITTED', 0)} submitted" + (f", {extra}" if extra else "") + ".")
    if rep["top_no_trade_reasons"]:
        r = rep["top_no_trade_reasons"][0]
        parts.append(f"Most common reason for not trading: \"{r['reason']}\" ({r['count']}x).")
    if rep["top_risk_rejections"]:
        r = rep["top_risk_rejections"][0]
        parts.append(f"Most frequent failed risk check: {r['check']} ({r['count']}x).")
    trades = rep["trades"]
    if trades:
        filled = [t for t in trades if t["filled_qty"] not in (None, "0", "0.0")]
        parts.append(f"{len(trades)} order(s) were sent to the paper gateway, {len(filled)} with fills.")
    else:
        parts.append("No orders were placed.")
    parts.append(f"Realized P&L {_money(Decimal(perf['realized_pnl']))} USD, unrealized "
                 f"{_money(Decimal(perf['unrealized_pnl']))} USD; model API cost {rep['api_cost_usd']} USD.")
    parts.append("Fills are simulated paper fills; live trading is impossible in this build."
                 if label == "SIMULATED" else "Fills are broker paper-trading fills; live trading is impossible in this build.")
    return " ".join(parts)
