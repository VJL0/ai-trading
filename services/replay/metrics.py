"""Executable-portfolio metrics from a replay journal (everything derives from journaled records only).

Equity is broker cash plus positions at the latest observed synthesized mid, minus the
runtime cost model's per-leg charge (fees + fixed slippage + impact; the half spread is already in the fill
price). Returns are per session close; annualisation uses 252 sessions. All figures are SIMULATED replay output,
not evidence of profitability.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from services.order_ledger.timeutil import parse_iso

ZERO = Decimal(0)
SESSIONS_PER_YEAR = 252


def _d(x: Any) -> Decimal:
    return Decimal(str(x)) if x is not None else ZERO


def _f(x: Decimal | float | None, nd: int = 6) -> float | None:
    if x is None:
        return None
    v = float(x)
    return round(v, nd) if math.isfinite(v) else None


def round_trips(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Closed position plans with the modelled cost of the legs that filled inside them."""
    costs = [(parse_iso(r["at"]), r["symbol"], _d(r["charged_usd"])) for r in records if r["kind"] == "LEG_COST"]
    fills = [r for r in records if r["kind"] == "FILL"]
    out: list[dict[str, Any]] = []
    for r in records:
        if r["kind"] != "ROUND_TRIP":
            continue
        opened = parse_iso(r["opened_at"])
        closed = parse_iso(r["closed_at"]) if r.get("closed_at") else opened
        sym = r["symbol"]
        cost = sum((c for at, s, c in costs if s == sym and opened <= at <= closed), ZERO)
        buys = [f for f in fills if f["symbol"] == sym and f["side"] == "BUY" and opened <= parse_iso(f["at"]) <= closed]
        sells = [f for f in fills if f["symbol"] == sym and f["side"] == "SELL" and opened <= parse_iso(f["at"]) <= closed]
        realized = _d(r.get("realized_pl"))
        entry_px, exit_px = r.get("entry_avg_price"), r.get("exit_avg_price")
        gross_bps = (float(_d(exit_px) / _d(entry_px) - 1) * 1e4) if entry_px and exit_px and _d(entry_px) else None
        qty = _d(r.get("entry_quantity"))
        notional = qty * _d(entry_px) if entry_px else ZERO
        out.append({
            "plan_id": r["plan_id"], "symbol": sym, "strategy_id": r.get("strategy_id"), "opened_at": r["opened_at"],
            "closed_at": r.get("closed_at"), "entry_at": buys[0]["at"] if buys else r["opened_at"],
            "exit_at": sells[-1]["at"] if sells else r.get("closed_at"), "entry_quantity": r.get("entry_quantity"),
            "exit_quantity": r.get("exit_quantity"), "entry_avg_price": entry_px, "exit_avg_price": exit_px,
            "exit_reason": r.get("exit_reason"),
            "close_reason": r.get("close_reason"), "realized_pl": str(realized), "modelled_cost_usd": str(cost),
            "net_pl": str(realized - cost), "gross_bps": _f(gross_bps, 3),
            "net_bps": _f((float((realized - cost) / notional) * 1e4) if notional else None, 3),
            "stop_price": r.get("stop_price"), "entry_fills": len(buys), "exit_fills": len(sells),
            "stop_fill": any(f.get("protective_stop_fill") for f in sells),
        })
    return out


def summarize(records: Sequence[dict[str, Any]], *, equity: Decimal) -> dict[str, Any]:
    e0 = Decimal(equity)
    costs = sorted((parse_iso(r["at"]), _d(r["charged_usd"]), _d(r["notional"])) for r in records
                   if r["kind"] == "LEG_COST")

    def charged_until(t: datetime) -> Decimal:
        return sum((c for at, c, _ in costs if at <= t), ZERO)

    # A session-close mark supersedes the regular bar mark at the same instant for exposure and
    # time-weighted averages. Keep both for drawdown so a close-time update cannot hide a loss.
    marks = [r for r in records if r["kind"] == "MARK"]
    bar_marks: dict[datetime, tuple[Decimal, Decimal]] = {}
    curve: list[tuple[datetime, Decimal, Decimal]] = []  # (at, net equity, exposure fraction)
    for m in marks:
        at = parse_iso(m["at"])
        eq = _d(m["equity"])
        net = eq - charged_until(at)
        expo = _d(m["position_value"]) / eq if eq > 0 else ZERO
        curve.append((at, net, expo))
        bar_marks[at] = (net, expo)
    closes = [(parse_iso(m["at"]), _d(m["equity"]) - charged_until(parse_iso(m["at"]))) for m in marks
              if m.get("session_end")]
    series = [e0] + [v for _, v in closes]
    rets = [float(b / a - 1) for a, b in itertools.pairwise(series) if a > 0]
    n = len(closes)
    final = curve[-1][1] if curve else e0
    cagr = (float(series[-1] / e0) ** (SESSIONS_PER_YEAR / n) - 1) if n and series[-1] >= 0 else None
    sharpe = None
    if len(rets) >= 2:
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
        sharpe = mu / sd * math.sqrt(SESSIONS_PER_YEAR) if sd > 0 else None
    peak, mdd = e0, 0.0
    for _, v, _ in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, float(v / peak - 1))
    fills = [r for r in records if r["kind"] == "FILL"]
    traded = sum((_d(f["quantity"]) * _d(f["price"]) for f in fills), ZERO)
    total_cost = sum((c for _, c, _ in costs), ZERO)
    avg_eq = (sum((v for v, _ in bar_marks.values()), ZERO) / len(bar_marks)) if bar_marks else e0
    expos = [float(x) for _, x in bar_marks.values()]
    rejections: Counter[str] = Counter()
    rejections_entries: Counter[str] = Counter()
    rejections_exits: Counter[str] = Counter()
    for r in records:
        if r["kind"] != "REJECTION":
            continue
        is_exit = r.get("side") == "SELL"
        for name in r.get("failed_checks") or ["(no failing check recorded)"]:
            rejections[name] += 1
            (rejections_exits if is_exit else rejections_entries)[name] += 1
    gate: Counter[str] = Counter()
    for r in records:
        if r["kind"] == "ORDER" and r.get("to_state") == "REJECTED":
            where = "final_gate" if r.get("from_state") in ("RISK_APPROVED", "HUMAN_APPROVED") else "broker"
            gate[f"{where}: {(r.get('reason') or '')[:120]}"] += 1
    decisions = Counter(str(r.get("outcome")) for r in records if r["kind"] == "DECISION")
    matches = Counter(str(r.get("outcome")) for r in records if r["kind"] == "ENTRY_MATCH")
    trips = round_trips(records)
    gross_pl = (_d(marks[-1]["equity"]) - e0) if marks else ZERO
    return {
        "sessions": n,
        "initial_equity": str(e0),
        "final_equity_net": str(final),
        "gross_pl": str(gross_pl),
        "modelled_costs_usd": str(total_cost),
        "net_pl": str(gross_pl - total_cost),
        "cagr": _f(cagr),
        "sharpe": _f(sharpe),
        "max_drawdown": _f(mdd),
        "turnover": {"traded_notional": str(traded), "per_session": _f(float(traded / avg_eq) / n if n and avg_eq
                                                                          else None),
                     "total_x_equity": _f(float(traded / avg_eq) if avg_eq else None)},
        "cost_drag": {"usd": str(total_cost), "pct_of_initial_equity": _f(float(total_cost / e0)),
                      "annualized_pct": _f(float(total_cost / e0) * SESSIONS_PER_YEAR / n if n else None),
                      "bps_of_traded_notional": _f(float(total_cost / traded) * 1e4 if traded else None)},
        "capital_exposure": {"mean": _f(sum(expos) / len(expos) if expos else None), "max": _f(max(expos, default=0.0)),
                             "fraction_of_bars_invested": _f(sum(1 for x in expos if x > 0) / len(expos)
                                                             if expos else None)},
        "rejections_by_check": dict(sorted(rejections.items())),
        "rejections_by_check_entries": dict(sorted(rejections_entries.items())),
        "rejections_by_check_exits": dict(sorted(rejections_exits.items())),
        "order_rejections": dict(sorted(gate.items())),
        "decisions": dict(sorted(decisions.items())),
        "entry_matching": dict(sorted(matches.items())),
        "round_trips": len(trips),
        "trades": trips,
        "note": "SIMULATED replay over the given snapshot; quotes synthesized from bars; not evidence of profitability",
    }
