"""Paper comparison hook: exported paper-ledger orders and fills for a period vs the replay of the same period.

``export_paper_ledger`` opens a ledger database strictly read-only (SQLite URI ``mode=ro`` + ``PRAGMA query_only``)
and reads only the ``orders`` and ``fills`` tables in ``[start, end)`` (no account snapshots, no keys, no
credentials). ``compare_paper`` matches order legs by symbol and side to the nearest replay leg within
``match_window_s`` and reports entries/exits present on one side only, fill price and time deltas, quantity
differences and order-state counts. Paper fills come from the broker's paper simulator: they are not live fills, and
the replay's are SIMULATED; neither is evidence of real execution quality.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from services.order_ledger.timeutil import iso, parse_iso

ZERO = Decimal(0)


@dataclass
class PaperExport:
    db_path: str
    start: str
    end: str
    mode: str | None
    orders: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)


def export_paper_ledger(db_path: Path, start: datetime, end: datetime, *, mode: str | None = "PAPER") -> PaperExport:
    if not db_path.is_file():
        raise FileNotFoundError(f"paper ledger {db_path} not found")
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        q = ("SELECT client_order_id, mode, gateway, symbol, side, quantity, order_type, limit_price, state, "
             "filled_quantity, avg_fill_price, created_at, submitted_at FROM orders WHERE created_at >= ? AND "
             "created_at < ?")
        params: list[Any] = [iso(start), iso(end)]
        if mode is not None:
            q += " AND mode = ?"
            params.append(mode)
        orders = [dict(r) for r in conn.execute(q + " ORDER BY created_at, client_order_id", params)]
        fq = ("SELECT f.fill_id, f.client_order_id, f.symbol, f.side, f.quantity, f.price, f.ts, o.order_type "
              "FROM fills f JOIN orders o ON o.client_order_id = f.client_order_id WHERE f.ts >= ? AND f.ts < ?")
        fparams: list[Any] = [iso(start), iso(end)]
        if mode is not None:
            fq += " AND o.mode = ?"
            fparams.append(mode)
        fills = [dict(r) for r in conn.execute(fq + " ORDER BY f.ts, f.fill_id", fparams)]
    finally:
        conn.close()
    return PaperExport(str(db_path), iso(start), iso(end), mode, orders, fills)


def _legs(fills: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One leg per order: first fill time, total quantity, average price."""
    by: dict[str, dict[str, Any]] = {}
    for f in fills:
        coid = str(f["client_order_id"])
        qty, px = Decimal(str(f["quantity"])), Decimal(str(f["price"]))
        leg = by.setdefault(coid, {"client_order_id": coid, "symbol": f["symbol"], "side": f["side"],
                                   "order_type": f.get("order_type"), "at": parse_iso(str(f.get("ts") or f["at"])),
                                   "quantity": ZERO, "notional": ZERO})
        leg["quantity"] += qty
        leg["notional"] += qty * px
    out = []
    for leg in by.values():
        leg["avg_price"] = leg["notional"] / leg["quantity"] if leg["quantity"] else ZERO
        out.append(leg)
    return sorted(out, key=lambda x: (x["at"], x["client_order_id"]))


def compare_paper(paper: PaperExport, replay_records: Sequence[dict[str, Any]], *,
                  match_window_s: float = 300.0) -> dict[str, Any]:
    paper_legs = _legs(paper.fills)
    lo, hi = parse_iso(paper.start), parse_iso(paper.end)
    replay_fills = [r for r in replay_records if r["kind"] == "FILL" and lo <= parse_iso(r["at"]) < hi]
    replay_legs = _legs(replay_fills)
    used: set[str] = set()
    matched: list[dict[str, Any]] = []
    paper_only: list[dict[str, Any]] = []
    for p in paper_legs:
        best: dict[str, Any] | None = None
        for r in replay_legs:
            if r["client_order_id"] in used or r["symbol"] != p["symbol"] or r["side"] != p["side"]:
                continue
            dt = abs((r["at"] - p["at"]).total_seconds())
            if dt <= match_window_s and (best is None or dt < abs((best["at"] - p["at"]).total_seconds())):
                best = r
        if best is None:
            paper_only.append(_leg_json(p))
            continue
        used.add(best["client_order_id"])
        matched.append({
            "symbol": p["symbol"], "side": p["side"], "paper_client_order_id": p["client_order_id"],
            "replay_client_order_id": best["client_order_id"], "paper_at": iso(p["at"]), "replay_at": iso(best["at"]),
            "time_delta_s": round((best["at"] - p["at"]).total_seconds(), 3),
            "paper_avg_price": str(p["avg_price"]), "replay_avg_price": str(best["avg_price"]),
            "price_delta_bps": round(float(best["avg_price"] / p["avg_price"] - 1) * 1e4, 3) if p["avg_price"] else None,
            "paper_quantity": str(p["quantity"]), "replay_quantity": str(best["quantity"]),
            "quantity_delta": str(best["quantity"] - p["quantity"]),
        })
    replay_only = [_leg_json(r) for r in replay_legs if r["client_order_id"] not in used]
    deltas = [abs(m["price_delta_bps"]) for m in matched if m["price_delta_bps"] is not None]

    def entries(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [x for x in legs if x["side"] == "BUY"]

    def exits(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [x for x in legs if x["side"] == "SELL"]

    return {
        "period": {"start": paper.start, "end": paper.end}, "paper_db": paper.db_path, "paper_mode_filter": paper.mode,
        "paper_orders": len(paper.orders), "paper_order_states": dict(Counter(str(o["state"]) for o in paper.orders)),
        "replay_orders": len({r["client_order_id"] for r in replay_records if r["kind"] == "ORDER"
                              and lo <= parse_iso(r["at"]) < hi}),
        "paper_legs": len(paper_legs), "replay_legs": len(replay_legs), "matched_legs": len(matched),
        "entries_paper_only": entries(paper_only), "entries_replay_only": entries(replay_only),
        "exits_paper_only": exits(paper_only), "exits_replay_only": exits(replay_only),
        "price_delta_bps": {"mean_abs": round(sum(deltas) / len(deltas), 3) if deltas else None,
                            "max_abs": round(max(deltas), 3) if deltas else None},
        "matched": matched, "match_window_s": match_window_s,
        "labels": {"paper": "PAPER (broker paper simulator; not live fills)", "replay": "REPLAY (SIMULATED)"},
    }


def _leg_json(leg: dict[str, Any]) -> dict[str, Any]:
    return {"client_order_id": leg["client_order_id"], "symbol": leg["symbol"], "side": leg["side"],
            "order_type": leg.get("order_type"), "at": iso(leg["at"]), "quantity": str(leg["quantity"]),
            "avg_price": str(leg["avg_price"])}
