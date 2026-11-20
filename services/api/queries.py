"""Read-side queries over the operational ledger, shaped to docs/architecture/api-contract.md.

Everything returned here is actual local state (SQLite rows, experiment result files). Nothing is
fabricated: missing data is returned as empty lists / nulls and the dashboard shows empty states.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import GatewayKind, TradingMode
from packages.core.audit import verify_chain
from packages.core.db import Database
from packages.core.modes import execution_label


def _payload(rows: list[dict[str, Any]], key: str = "payload_json") -> list[dict[str, Any]]:
    return [json.loads(r[key]) for r in rows]


def _order_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "client_order_id": r["client_order_id"],
        "broker_order_id": r["broker_order_id"],
        "intent_id": r["intent_id"],
        "symbol": r["symbol"],
        "side": r["side"],
        "quantity": r["quantity"],
        "filled_quantity": r["filled_quantity"],
        "avg_fill_price": r["avg_fill_price"],
        "limit_price": r["limit_price"],
        "state": r["state"],
        "gateway": r["gateway"],
        "execution_label": execution_label(TradingMode(r["mode"]), GatewayKind(r["gateway"])).value,
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def signals(db: Database, limit: int = 50) -> dict[str, Any]:
    rows = db.query("SELECT payload_json FROM quant_signals ORDER BY generated_at DESC, signal_id DESC LIMIT ?", (limit,))
    latest = db.query_one("SELECT value_json FROM system_state WHERE key='last_cycle'")
    return {"signals": _payload(rows), "latest_cycle": json.loads(latest["value_json"]) if latest else None}


def research_runs(db: Database, limit: int = 50) -> dict[str, Any]:
    runs = _payload(db.query("SELECT payload_json FROM model_runs ORDER BY started_at DESC LIMIT ?", (limit,)))
    findings = _payload(
        db.query("SELECT payload_json FROM research_findings ORDER BY created_at DESC LIMIT ?", (limit,))
    )
    agg = db.query_one(
        "SELECT COUNT(*) n, COALESCE(SUM(input_tokens),0) inp, COALESCE(SUM(output_tokens),0) outp,"
        " SUM(CASE WHEN transport='FAKE' THEN 1 ELSE 0 END) fake,"
        " SUM(CASE WHEN transport='REAL' THEN 1 ELSE 0 END) real FROM model_runs"
    ) or {}
    cost = sum(
        (Decimal(r["estimated_cost_usd"]) for r in db.query("SELECT estimated_cost_usd FROM model_runs")), Decimal(0)
    )
    return {
        "runs": runs,
        "findings": findings,
        "totals": {
            "runs": agg.get("n", 0) or 0,
            "input_tokens": agg.get("inp", 0) or 0,
            "output_tokens": agg.get("outp", 0) or 0,
            "estimated_cost_usd": format(cost, "f"),
            "fake_runs": agg.get("fake", 0) or 0,
            "real_runs": agg.get("real", 0) or 0,
        },
    }


def proposals(db: Database, limit: int = 50) -> dict[str, Any]:
    rows = db.query("SELECT proposal_id, payload_json FROM proposals ORDER BY created_at DESC LIMIT ?", (limit,))
    out = []
    for r in rows:
        dec = db.query_one(
            "SELECT payload_json FROM risk_decisions WHERE proposal_id=? ORDER BY evaluated_at DESC LIMIT 1",
            (r["proposal_id"],),
        )
        order = db.query_one("SELECT client_order_id FROM orders WHERE proposal_id=?", (r["proposal_id"],))
        out.append(
            {
                "proposal": json.loads(r["payload_json"]),
                "decision": json.loads(dec["payload_json"]) if dec else None,
                "intent_client_order_id": order["client_order_id"] if order else None,
            }
        )
    return {"proposals": out}


def orders(db: Database, limit: int = 100) -> dict[str, Any]:
    rows = db.query("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))
    return {"orders": [_order_row(r) for r in rows]}


def order_detail(db: Database, client_order_id: str) -> dict[str, Any] | None:
    r = db.query_one("SELECT * FROM orders WHERE client_order_id=?", (client_order_id,))
    if r is None:
        return None
    events = []
    for e in db.query("SELECT * FROM order_events WHERE client_order_id=? ORDER BY seq", (client_order_id,)):
        body = json.loads(e["payload_json"])
        body["applied"] = bool(e["applied"])
        body["apply_note"] = e["apply_note"]
        events.append(body)
    transitions = db.query(
        "SELECT from_state, to_state, at, reason FROM order_state_transitions WHERE client_order_id=? ORDER BY seq",
        (client_order_id,),
    )
    fills = db.query(
        "SELECT fill_id, quantity, price, ts, broker_exec_id FROM fills WHERE client_order_id=? ORDER BY ts",
        (client_order_id,),
    )
    return {
        "order": _order_row(r),
        "intent": json.loads(r["intent_json"]),
        "events": events,
        "transitions": transitions,
        "fills": fills,
    }


def portfolio(db: Database, label: str) -> dict[str, Any]:
    acct = db.query_one("SELECT payload_json FROM account_snapshots ORDER BY asof DESC LIMIT 1")
    group = db.query_one("SELECT snapshot_group FROM position_snapshots ORDER BY seq DESC LIMIT 1")
    broker_positions = (
        _payload(db.query("SELECT payload_json FROM position_snapshots WHERE snapshot_group=?", (group["snapshot_group"],)))
        if group
        else []
    )
    ledger_positions = db.query("SELECT symbol, quantity, avg_cost, realized_pl FROM ledger_positions ORDER BY symbol")
    curve = [
        {"time": _unix(r["asof"]), "equity": r["equity"]}
        for r in db.query("SELECT asof, equity FROM account_snapshots ORDER BY asof LIMIT 5000")
    ]
    pnl_row = db.query_one("SELECT value_json FROM system_state WHERE key='daily_pnl'")
    pnl = json.loads(pnl_row["value_json"]) if pnl_row else {
        "daily_realized": "0", "daily_unrealized": "0", "start_of_day_equity": None,
    }
    return {
        "execution_label": label,
        "account": json.loads(acct["payload_json"]) if acct else None,
        "broker_positions": broker_positions,
        "ledger_positions": ledger_positions,
        "pnl": pnl,
        "equity_curve": curve,
    }


def risk_decisions(db: Database, limit: int = 25) -> list[dict[str, Any]]:
    return _payload(db.query("SELECT payload_json FROM risk_decisions ORDER BY evaluated_at DESC LIMIT ?", (limit,)))


def policy_history(db: Database) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT policy_version, created_at, created_by, is_active FROM risk_policies ORDER BY created_at DESC"
    )
    for r in rows:
        r["is_active"] = bool(r["is_active"])
    return rows


def reconciliation(db: Database, limit: int = 20) -> dict[str, Any]:
    return {
        "runs": _payload(
            db.query("SELECT payload_json FROM reconciliation_runs ORDER BY started_at DESC LIMIT ?", (limit,))
        )
    }


def audit(db: Database, limit: int = 200, event_type: str | None = None) -> dict[str, Any]:
    if event_type:
        rows = db.query(
            "SELECT * FROM audit_events WHERE event_type=? ORDER BY seq DESC LIMIT ?", (event_type, limit)
        )
    else:
        rows = db.query("SELECT * FROM audit_events ORDER BY seq DESC LIMIT ?", (limit,))
    events = []
    for r in rows:
        events.append(
            {
                "seq": r["seq"], "event_id": r["event_id"], "ts": r["ts"], "event_type": r["event_type"],
                "causal_id": r["causal_id"], "source": r["source"], "mode": r["mode"],
                "policy_version": r["policy_version"], "payload": json.loads(r["payload_json"]),
            }
        )
    ok, n, detail = verify_chain(db)
    return {"events": events, "chain": {"ok": ok, "rows": n, "detail": detail}}


def failures(db: Database, limit: int = 100) -> dict[str, Any]:
    return {
        "failures": db.query(
            "SELECT ts, component, kind, detail, causal_id FROM failures ORDER BY seq DESC LIMIT ?", (limit,)
        )
    }


def data_quality_events(db: Database, limit: int = 100) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT ts, symbol, feed, kind, severity, detail_json FROM data_quality_events ORDER BY seq DESC LIMIT ?", (limit,)
    )
    for r in rows:
        r["detail"] = json.loads(r.pop("detail_json"))
    return rows


def backtests(experiments_dir: Path) -> dict[str, Any]:
    """Collect real result files written by LEAN runs and Python baseline runs."""
    runs: list[dict[str, Any]] = []
    if experiments_dir.is_dir():
        for path in sorted(experiments_dir.glob("**/summary.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or "run_id" not in data:
                continue
            data.setdefault("source_path", str(path.relative_to(experiments_dir.parent)))
            runs.append(data)
    runs.sort(key=lambda d: str(d.get("started_at", "")), reverse=True)
    return {"runs": runs}


def _unix(iso: str) -> int:
    from datetime import datetime

    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
