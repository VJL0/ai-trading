"""LLM invocation gate: expensive models are invoked only for a selected candidate, at most once per symbol per
interval, within a daily run and cost budget -- never per market tick.

Two clocks:
* the *decision* clock (replay/sim time) enforces ``min_interval_per_symbol_s`` between reviews of a symbol;
* the *wall* clock (real time, where money is spent) enforces ``min_wall_interval_per_symbol_s`` and the daily
  budget day (``budget_timezone``), so a fast replay cannot burn through the API budget.

Budget accounting is pessimistic: a run reserves ``est_max_cost_per_run_usd`` (REAL transport; FAKE bills
nothing) before it starts and settles to the recorded cost afterwards. With a database the ledger
(``ai_invocations``) is reloaded on start; unsettled reservations count at their reserved amount (fail closed).
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from packages.core.clock import Clock, SystemClock
from packages.core.db import Database, utc_iso
from packages.core.ids import new_id

from .config import AiConfig


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    symbol: str
    invocation_id: str | None = None
    reserved_usd: Decimal = Decimal(0)


@dataclass
class _Reservation:
    invocation_id: str
    symbol: str
    day: date
    reserved: Decimal


class LlmInvocationGate:
    def __init__(
        self,
        config: AiConfig,
        *,
        decision_clock: Clock,
        wall_clock: Clock | None = None,
        db: Database | None = None,
        transport: str = "fake",
    ) -> None:
        self._cfg = config
        self._decision_clock = decision_clock
        self._wall_clock = wall_clock or SystemClock()
        self._db = db
        self._transport = transport.upper()
        self._tz = ZoneInfo(config.budget_timezone)
        self._lock = threading.Lock()
        self._last_decision: dict[str, datetime] = {}
        self._last_wall: dict[str, datetime] = {}
        self._in_flight: dict[str, _Reservation] = {}
        self._day: date = self._today()
        self._runs_today = 0
        self._spent_today = Decimal(0)
        self._reserved_today = Decimal(0)
        if db is not None:
            self._load(db)

    # ------------------------------------------------------------------ state
    def _today(self) -> date:
        return self._wall_clock.now().astimezone(self._tz).date()

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._runs_today = 0
            self._spent_today = Decimal(0)
            self._reserved_today = sum((r.reserved for r in self._in_flight.values() if r.day == today), Decimal(0))

    def _load(self, db: Database) -> None:
        day = self._day.isoformat()
        rows = db.query(
            "SELECT status, reserved_usd, actual_usd FROM ai_invocations WHERE budget_day=? AND transport=?"
            " AND status IN ('RESERVED','SETTLED')",
            (day, self._transport),
        )
        for r in rows:
            self._runs_today += 1
            if r["status"] == "SETTLED" and r["actual_usd"] is not None:
                self._spent_today += Decimal(r["actual_usd"])
            else:  # never settled: the process died mid-run; count the full reservation
                self._spent_today += Decimal(r["reserved_usd"])
        for r in db.query(
            "SELECT symbol, MAX(wall_ts) ts FROM ai_invocations WHERE status IN ('RESERVED','SETTLED') AND transport=?"
            " GROUP BY symbol",
            (self._transport,),
        ):
            self._last_wall[r["symbol"]] = datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))

    # ------------------------------------------------------------------ decisions
    def _evaluate(self, symbol: str, *, candidate_selected: bool) -> tuple[bool, str, Decimal]:
        cfg = self._cfg
        if not candidate_selected:
            return False, "no selected candidate: models are never invoked per tick", Decimal(0)
        if symbol in self._in_flight:
            return False, f"a model run for {symbol} is already in flight", Decimal(0)
        now = self._decision_clock.now()
        last = self._last_decision.get(symbol)
        if last is not None:
            delta = (now - last).total_seconds()
            if 0 <= delta < cfg.min_interval_per_symbol_s:
                return False, (
                    f"{symbol} was reviewed {delta:.0f}s ago (decision clock); minimum interval "
                    f"{cfg.min_interval_per_symbol_s:.0f}s"
                ), Decimal(0)
        wall = self._wall_clock.now()
        last_wall = self._last_wall.get(symbol)
        if last_wall is not None:
            wdelta = (wall - last_wall).total_seconds()
            if wdelta < cfg.min_wall_interval_per_symbol_s:
                return False, (
                    f"{symbol} was reviewed {max(wdelta, 0):.0f}s ago (wall clock); minimum interval "
                    f"{cfg.min_wall_interval_per_symbol_s:.0f}s"
                ), Decimal(0)
        if self._runs_today >= cfg.max_runs_per_day:
            return False, f"daily model-run limit reached ({self._runs_today}/{cfg.max_runs_per_day})", Decimal(0)
        reserve = cfg.est_max_cost_per_run_usd if self._transport == "REAL" else Decimal(0)
        committed = self._spent_today + self._reserved_today
        if self._transport == "REAL" and committed + reserve > cfg.daily_budget_usd:
            return False, (
                f"daily model budget exhausted: committed ${committed} + reservation ${reserve} > "
                f"${cfg.daily_budget_usd}"
            ), Decimal(0)
        return True, "allowed", reserve

    def check(self, symbol: str, *, candidate_selected: bool = True) -> GateDecision:
        """Side-effect-free preview of :meth:`acquire`."""
        with self._lock:
            self._roll_day()
            allowed, reason, reserve = self._evaluate(symbol, candidate_selected=candidate_selected)
        return GateDecision(allowed, reason, symbol, None, reserve)

    def acquire(self, symbol: str, causal_id: str, *, candidate_selected: bool = True) -> GateDecision:
        with self._lock:
            self._roll_day()
            allowed, reason, reserve = self._evaluate(symbol, candidate_selected=candidate_selected)
            invocation_id = new_id("aiinv")
            decision_ts = self._decision_clock.now()
            wall_ts = self._wall_clock.now()
            if allowed:
                self._in_flight[symbol] = _Reservation(invocation_id, symbol, self._day, reserve)
                self._runs_today += 1
                self._reserved_today += reserve
                self._last_decision[symbol] = decision_ts
                self._last_wall[symbol] = wall_ts
            day = self._day
        self._write_row(invocation_id, causal_id, symbol, decision_ts, wall_ts, day, allowed, reason, reserve)
        return GateDecision(allowed, reason, symbol, invocation_id if allowed else None, reserve)

    def settle(self, decision: GateDecision, actual_cost_usd: Decimal | None) -> None:
        """Release the reservation and charge the recorded cost (the reservation itself when unknown)."""
        if not decision.allowed or decision.invocation_id is None:
            return
        with self._lock:
            res = self._in_flight.pop(decision.symbol, None)
            if res is None or res.invocation_id != decision.invocation_id:
                if res is not None:
                    self._in_flight[decision.symbol] = res
                return
            cost = res.reserved if actual_cost_usd is None else max(actual_cost_usd, Decimal(0))
            if res.day == self._day:
                self._reserved_today -= res.reserved
                self._spent_today += cost
        if self._db is not None:
            self._db.write(lambda c: c.execute(
                "UPDATE ai_invocations SET status='SETTLED', actual_usd=?, settled_at=? WHERE invocation_id=?",
                (format(cost, "f"), utc_iso(self._wall_clock.now()), res.invocation_id),
            ))

    def _write_row(
        self, invocation_id: str, causal_id: str, symbol: str, decision_ts: datetime, wall_ts: datetime, day: date,
        allowed: bool, reason: str, reserve: Decimal,
    ) -> None:
        if self._db is None:
            return

        def _w(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO ai_invocations(invocation_id, causal_id, symbol, decision_ts, wall_ts, budget_day, transport,"
                " status, reason, reserved_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (invocation_id, causal_id, symbol, utc_iso(decision_ts), utc_iso(wall_ts), day.isoformat(),
                 self._transport, "RESERVED" if allowed else "DENIED", reason[:500], format(reserve, "f")),
            )

        self._db.write(_w)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._roll_day()
            return {
                "transport": self._transport,
                "budget_day": self._day.isoformat(),
                "runs_today": self._runs_today,
                "max_runs_per_day": self._cfg.max_runs_per_day,
                "spent_today_usd": format(self._spent_today, "f"),
                "reserved_usd": format(self._reserved_today, "f"),
                "daily_budget_usd": format(self._cfg.daily_budget_usd, "f"),
                "in_flight": sorted(self._in_flight),
                "min_interval_per_symbol_s": self._cfg.min_interval_per_symbol_s,
                "min_wall_interval_per_symbol_s": self._cfg.min_wall_interval_per_symbol_s,
            }
