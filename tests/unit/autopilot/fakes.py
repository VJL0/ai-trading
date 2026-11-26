"""Fake clock, tool backend and subprocess runner for autopilot unit tests (no network, no API, no children)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from services.autopilot.backend import GuardedBackend
from services.autopilot.config import AutopilotConfig
from services.autopilot.research import StepOutcome
from services.autopilot.store import Store
from services.autopilot.supervisor import Autopilot
from services.tools.handlers import ToolError

ET = ZoneInfo("America/New_York")


def et(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET).astimezone(UTC)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.t = now

    def now(self) -> datetime:
        return self.t

    def advance(self, **kw: float) -> None:
        self.t += timedelta(**kw)

    def set(self, now: datetime) -> None:
        self.t = now


def _base(now: datetime) -> dict[str, Any]:
    return {"execution_label": "SIMULATED", "is_simulated": True, "asof": now.isoformat()}


def _session(sid: str, state: str, *, requested_by: str = "agent:autopilot", started_at: str | None = None,
             ends_at: str = "2026-11-25T20:30:00Z") -> dict[str, Any]:
    return {"session_id": sid, "state": state, "started_at": started_at, "ends_at": ends_at, "ended_at": None,
            "duration_minutes": 360, "symbols": ["SPY"], "cadence_minutes": 5, "use_ai": False,
            "requested_by": requested_by, "cycles_run": 0, "trades": 0, "stop_reason": None, "schedule_id": None,
            "next_cycle_at": None, "last_cycle_at": None, "ai_suspended_until": None}


class FakeApi:
    """A deterministic stand-in for the agent tool interface. ``calls`` records every (name, args)."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.halts: list[dict[str, Any]] = []
        self.kill_switch = False
        self.entries_locked = False
        self.lock_reasons: list[str] = []
        self.health_overrides: dict[str, str] = {}
        self.feed_mode = "IEX_FREE"
        self.feed_status = "OK"
        self.active: dict[str, Any] | None = None
        self.recent: list[dict[str, Any]] = []
        self.schedules: list[dict[str, Any]] = []
        self.start_error: ToolError | None = None
        self.start_side_effect_before_error = False
        self.families: list[dict[str, Any]] = []
        self.unreachable = False
        self.n = 0

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(args)))
        now = self.clock.now()
        if self.unreachable:
            raise ToolError(503, "local trading API is not reachable")
        b = _base(now)
        if name == "get_session_status":
            return {**b, "active": copy.deepcopy(self.active), "recent": copy.deepcopy(self.recent),
                    "schedules": copy.deepcopy(self.schedules), "entries_locked": self.entries_locked,
                    "lock_reasons": [], "market_open": None, "next_open": None, "scheduler_running": True,
                    "detail": ""}
        if name == "get_risk_status":
            halts = [*self.halts]
            exit_reasons = []
            if self.kill_switch:
                halts.append({"halt_id": "halt_k", "reason": "operator kill", "source": "operator", "created_at": None})
                exit_reasons.append("halt halt_k (scope=ALL, exits blocked): operator kill (source=operator)")
            locked = self.entries_locked or bool(halts)
            return {**b, "mode": "PAPER", "live_execution_compiled": False, "policy_version": "p1",
                    "entries_locked": locked, "lock_reasons": [*self.lock_reasons],
                    "exits_locked": bool(exit_reasons), "exit_lock_reasons": exit_reasons, "position_safety": [],
                    "halts": halts, "limits": {}, "utilization": [], "recent_decisions": [],
                    "agent_max_proposals_per_hour": 6}
        if name == "get_system_health":
            comps = [{"name": c, "status": self.health_overrides.get(c, "OK"), "detail": "", "last_ok": None}
                     for c in ("market_data", "broker", "reconciliation", "database", "engine", "scheduler")]
            overall = "DOWN" if any(c["status"] == "DOWN" for c in comps) else "OK"
            return {**b, "overall": overall, "components": comps}
        if name == "get_feed_quality":
            return {**b, "feed_mode": self.feed_mode, "provider": "fixture", "feed": "SYNTHETIC",
                    "status": self.feed_status, "detail": "", "staleness_s": None, "max_staleness_s": None,
                    "last_bar_ts": None, "last_quote_ts": None, "limitations": [], "symbols": [], "recent_events": []}
        if name == "start_paper_session":
            if self.start_error is not None:
                if self.start_side_effect_before_error:
                    self.active = _session("sess_unknown", "SCHEDULED")
                raise self.start_error
            if self.active is not None:
                raise ToolError(409, "a paper session is already active")
            self.n += 1
            self.active = _session(f"sess_{self.n}", "SCHEDULED")
            return {**b, "outcome": "SCHEDULED", "session": copy.deepcopy(self.active), "detail": "queued"}
        if name == "get_session_report":
            sess = self.active or _session(str(args.get("session_id")), "COMPLETED")
            return {**b, "session": sess, "performance": {}, "cycles": 3, "decisions": {"NO_TRADE": 3},
                    "top_no_trade_reasons": [], "top_risk_rejections": [{"check": "spread", "count": 2}],
                    "trades": [], "api_cost_usd": "0", "narrative": "quiet day"}
        if name == "get_performance":
            return {**b, "mode_label": "SIMULATED", "portfolio_value": "100000", "cash": "100000", "daily_pnl": "0",
                    "total_pnl": "0", "realized_pnl": "0", "unrealized_pnl": "0", "win_rate": None, "round_trips": 0,
                    "max_drawdown": "0", "current_drawdown": "0", "active_positions": 0, "open_orders": 0,
                    "api_cost_usd": "0", "equity_points": 0, "equity_curve": []}
        if name == "get_orders":
            return {**b, "orders": []}
        if name == "get_research_results":
            return {**b, "status": "ok", "updated_at": None, "holdout": [], "families": copy.deepcopy(self.families),
                    "holdout_results": [], "detail": None, "no_edge_note": None}
        raise AssertionError(f"unexpected tool {name}")


class FakeRunner:
    """Records subprocess argv; returns canned outcomes per command prefix."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], float]] = []
        self.results: dict[str, StepOutcome] = {}

    async def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> StepOutcome:
        args = tuple(argv)
        self.calls.append((args, timeout_s))
        key = args[2] if args[1] == "services.quantlab.cli" else "crew"
        return self.results.get(key, StepOutcome(args, 0, 1.0, '{"ticks": [{"ran": [], "idle_reason": "no task"}]}'))

    def commands(self) -> list[str]:
        return [a[2] if a[1] == "services.quantlab.cli" else "crew" for a, _ in self.calls]


class FakeDailyJob:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.runs = 0
        self.done: Any = None

    def factory(self, root: Path, *, clock: Any, runner: Any) -> FakeDailyJob:
        self.clock = clock
        return self

    async def run_once(self) -> dict[str, Any] | None:
        self.runs += 1
        return {"session": "x", "ok": self.ok, "steps": []}

    def last_done(self) -> Any:
        return self.done


def make(tmp: Path, now: datetime, **cfg_kw: Any) -> tuple[Autopilot, FakeClock, FakeApi, FakeRunner, FakeDailyJob]:
    clock = FakeClock(now)
    api = FakeApi(clock)
    runner = FakeRunner()
    daily = FakeDailyJob()
    cfg = AutopilotConfig(project_root=tmp, state_dir=tmp / "ap", report_dir=tmp / "ap" / "reports", enabled=True,
                          **cfg_kw)
    ap = Autopilot(cfg, Store(cfg.store_path), GuardedBackend(api), clock=clock, runner=runner,
                   daily_job_factory=daily.factory)
    return ap, clock, api, runner, daily
