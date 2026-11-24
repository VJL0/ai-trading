"""Durable position plans and the append-only position-manager action log (SQLite, single writer)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from packages.core.audit import append_audit
from packages.core.db import Database
from packages.core.ids import new_id
from services.order_ledger.timeutil import iso, parse_iso, trading_day

ZERO = Decimal(0)


def _d(v: object | None) -> Decimal | None:
    return None if v is None else Decimal(str(v))


def _fmt(v: Decimal | None) -> str | None:
    return None if v is None else format(v, "f")


@dataclass(frozen=True)
class PositionPlan:
    plan_id: str
    symbol: str
    status: str
    opened_at: datetime
    trading_day: str
    strategy_id: str | None
    entry_client_order_id: str | None
    stop_price: Decimal
    stop_source: str
    horizon_minutes: int
    exit_deadline: datetime
    exit_reason: str | None
    exit_started_at: datetime | None
    max_quantity: Decimal
    closed_at: datetime | None
    close_reason: str | None
    entry_quantity: Decimal | None
    entry_avg_price: Decimal | None
    exit_quantity: Decimal | None
    exit_avg_price: Decimal | None
    realized_pl: Decimal | None
    policy_version: str
    holding_sessions: int = 1
    entry_risk_policy_version: str | None = None

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> PositionPlan:
        def ts(k: str) -> datetime | None:
            return parse_iso(str(r[k])) if r.get(k) else None

        opened, deadline = ts("opened_at"), ts("exit_deadline")
        assert opened is not None and deadline is not None
        stop = _d(r["stop_price"])
        assert stop is not None
        return cls(
            plan_id=r["plan_id"], symbol=r["symbol"], status=r["status"], opened_at=opened, trading_day=r["trading_day"],
            strategy_id=r.get("strategy_id"), entry_client_order_id=r.get("entry_client_order_id"), stop_price=stop,
            stop_source=r["stop_source"], horizon_minutes=int(r["horizon_minutes"]), exit_deadline=deadline,
            exit_reason=r.get("exit_reason"), exit_started_at=ts("exit_started_at"),
            max_quantity=_d(r.get("max_quantity")) or ZERO, closed_at=ts("closed_at"), close_reason=r.get("close_reason"),
            entry_quantity=_d(r.get("entry_quantity")), entry_avg_price=_d(r.get("entry_avg_price")),
            exit_quantity=_d(r.get("exit_quantity")), exit_avg_price=_d(r.get("exit_avg_price")),
            realized_pl=_d(r.get("realized_pl")), policy_version=r["policy_version"],
            holding_sessions=int(r.get("holding_sessions", 1)),
            entry_risk_policy_version=r.get("entry_risk_policy_version"),
        )

    def to_json(self) -> dict[str, Any]:
        def t(v: datetime | None) -> str | None:
            return None if v is None else iso(v)

        return {
            "plan_id": self.plan_id, "symbol": self.symbol, "status": self.status, "opened_at": t(self.opened_at),
            "trading_day": self.trading_day, "strategy_id": self.strategy_id,
            "entry_client_order_id": self.entry_client_order_id, "stop_price": _fmt(self.stop_price),
            "stop_source": self.stop_source, "horizon_minutes": self.horizon_minutes,
            "exit_deadline": t(self.exit_deadline), "exit_reason": self.exit_reason,
            "exit_started_at": t(self.exit_started_at), "max_quantity": _fmt(self.max_quantity),
            "closed_at": t(self.closed_at), "close_reason": self.close_reason,
            "entry_quantity": _fmt(self.entry_quantity), "entry_avg_price": _fmt(self.entry_avg_price),
            "exit_quantity": _fmt(self.exit_quantity), "exit_avg_price": _fmt(self.exit_avg_price),
            "realized_pl": _fmt(self.realized_pl), "policy_version": self.policy_version,
            "holding_sessions": self.holding_sessions, "entry_risk_policy_version": self.entry_risk_policy_version,
        }


class PositionStore:
    def __init__(self, db: Database, mode: str) -> None:
        self.db = db
        self.mode = mode

    # ------------------------------------------------------------------ plans
    def open_plan(self, symbol: str) -> PositionPlan | None:
        row = self.db.query_one("SELECT * FROM position_plans WHERE symbol=? AND status='OPEN'", (symbol,))
        return None if row is None else PositionPlan.from_row(row)

    def open_plans(self) -> list[PositionPlan]:
        return [PositionPlan.from_row(r) for r in
                self.db.query("SELECT * FROM position_plans WHERE status='OPEN' ORDER BY symbol")]

    def closed_plans(self, limit: int = 50, since: datetime | None = None) -> list[PositionPlan]:
        if since is None:
            rows = self.db.query("SELECT * FROM position_plans WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT ?",
                                 (limit,))
        else:
            rows = self.db.query("SELECT * FROM position_plans WHERE status='CLOSED' AND closed_at >= ? "
                                 "ORDER BY closed_at DESC LIMIT ?", (iso(since), limit))
        return [PositionPlan.from_row(r) for r in rows]

    def last_closed_at(self, symbol: str) -> datetime | None:
        row = self.db.query_one("SELECT MAX(closed_at) AS c FROM position_plans WHERE symbol=? AND status='CLOSED'",
                                (symbol,))
        return parse_iso(str(row["c"])) if row and row["c"] else None

    def create_plan(self, *, symbol: str, opened_at: datetime, strategy_id: str | None, entry_coid: str | None,
                    stop_price: Decimal, stop_source: str, horizon_minutes: int, exit_deadline: datetime,
                    quantity: Decimal, policy_version: str, at: datetime, holding_sessions: int = 1,
                    entry_risk_policy_version: str | None = None) -> PositionPlan:
        plan_id = new_id("pos")

        def _w(conn: sqlite3.Connection) -> None:
            if conn.execute("SELECT 1 FROM position_plans WHERE symbol=? AND status='OPEN'", (symbol,)).fetchone():
                return  # another writer opened it first (unique index also enforces this)
            conn.execute(
                "INSERT INTO position_plans(plan_id, symbol, status, opened_at, trading_day, strategy_id,"
                " entry_client_order_id, stop_price, stop_source, horizon_minutes, exit_deadline, max_quantity,"
                " policy_version, created_at, updated_at, holding_sessions, entry_risk_policy_version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (plan_id, symbol, "OPEN", iso(opened_at), trading_day(opened_at).isoformat(), strategy_id, entry_coid,
                 format(stop_price, "f"), stop_source, horizon_minutes, iso(exit_deadline), format(quantity, "f"),
                 policy_version, iso(at), iso(at), holding_sessions, entry_risk_policy_version),
            )
            self._action_conn(conn, plan_id, symbol, at, "PLAN_OPENED", None, entry_coid, "OK",
                              f"stop {stop_price} ({stop_source}), horizon {horizon_minutes}m, "
                              f"holding {holding_sessions} session(s), deadline {iso(exit_deadline)}")
            append_audit(conn, event_type="POSITION_OPENED", source="position_manager", mode=self.mode, ts=at,
                         payload={"plan_id": plan_id, "symbol": symbol, "stop_price": format(stop_price, "f"),
                                  "stop_source": stop_source, "exit_deadline": iso(exit_deadline),
                                  "holding_sessions": holding_sessions, "entry_risk_policy_version": entry_risk_policy_version,
                                  "note": "planned stop is a trigger, not guaranteed protection"})

        self.db.write(_w)
        plan = self.open_plan(symbol)
        assert plan is not None
        return plan

    def update_plan(self, plan_id: str, at: datetime, **fields: Any) -> None:
        allowed = {"exit_reason", "exit_started_at", "max_quantity"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"cannot update plan fields {sorted(set(fields) - allowed)}")
        vals = {k: (iso(v) if isinstance(v, datetime) else _fmt(v) if isinstance(v, Decimal) else v)
                for k, v in fields.items()}
        sets = ", ".join(f"{k}=?" for k in vals)
        self.db.write(lambda c: c.execute(
            f"UPDATE position_plans SET {sets}, updated_at=? WHERE plan_id=? AND status='OPEN'",  # noqa: S608
            (*vals.values(), iso(at), plan_id)))

    def close_plan(self, plan: PositionPlan, *, at: datetime, reason: str, entry_qty: Decimal,
                   entry_avg: Decimal | None, exit_qty: Decimal, exit_avg: Decimal | None, realized: Decimal) -> None:
        def _w(conn: sqlite3.Connection) -> None:
            cur = conn.execute(
                "UPDATE position_plans SET status='CLOSED', closed_at=?, close_reason=?, entry_quantity=?,"
                " entry_avg_price=?, exit_quantity=?, exit_avg_price=?, realized_pl=?, updated_at=?"
                " WHERE plan_id=? AND status='OPEN'",
                (iso(at), reason, _fmt(entry_qty), _fmt(entry_avg), _fmt(exit_qty), _fmt(exit_avg), _fmt(realized),
                 iso(at), plan.plan_id),
            )
            if not cur.rowcount:
                return
            self._action_conn(conn, plan.plan_id, plan.symbol, at, "PLAN_CLOSED", reason, None, "OK",
                              f"realized P&L {_fmt(realized)} on {_fmt(exit_qty)} shares")
            append_audit(conn, event_type="POSITION_CLOSED", source="position_manager", mode=self.mode, ts=at,
                         payload={"plan_id": plan.plan_id, "symbol": plan.symbol, "reason": reason,
                                  "entry_quantity": _fmt(entry_qty), "entry_avg_price": _fmt(entry_avg),
                                  "exit_quantity": _fmt(exit_qty), "exit_avg_price": _fmt(exit_avg),
                                  "realized_pl": _fmt(realized)})

        self.db.write(_w)

    # ------------------------------------------------------------------ actions
    @staticmethod
    def _action_conn(conn: sqlite3.Connection, plan_id: str | None, symbol: str, at: datetime, action: str,
                     purpose: str | None, coid: str | None, outcome: str, detail: str | None) -> None:
        conn.execute(
            "INSERT INTO position_actions(plan_id, symbol, at, action, purpose, client_order_id, outcome, detail)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (plan_id, symbol, iso(at), action, purpose, coid, outcome[:200], None if detail is None else detail[:1000]),
        )

    def record_action(self, *, plan_id: str | None, symbol: str, at: datetime, action: str, purpose: str | None,
                      coid: str | None, outcome: str, detail: str | None = None) -> None:
        self.db.write(lambda c: self._action_conn(c, plan_id, symbol, at, action, purpose, coid, outcome, detail))

    def actions(self, symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if symbol is None:
            return self.db.query("SELECT * FROM position_actions ORDER BY seq DESC LIMIT ?", (limit,))
        return self.db.query("SELECT * FROM position_actions WHERE symbol=? ORDER BY seq DESC LIMIT ?", (symbol, limit))

    def last_action_at(self, symbol: str, action: str) -> datetime | None:
        row = self.db.query_one("SELECT at FROM position_actions WHERE symbol=? AND action=? ORDER BY seq DESC LIMIT 1",
                                (symbol, action))
        return parse_iso(str(row["at"])) if row else None
