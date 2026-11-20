"""The exact runtime surface the tool handlers may use (``services.api.runtime.Runtime`` satisfies it).

Read members return the same shapes as the public local API. The only state-changing members are
``submit_agent_proposal`` (risk engine -> signed intent -> final gate -> paper gateway; rate limited; refuses
non-PAPER/REPLAY), ``halt`` (fail-safe direction), ``cancel``, ``run_cycle`` (deterministic quant cycle
through the same risk path) and ``sessions`` (the PAPER session manager: its only state-changing members are
``SESSION_MUTATING_METHODS`` -- start/stop a PAPER session, add/cancel a schedule -- and every order a session
makes still goes through the risk engine). ``activity`` is used read-only (``recent``). There is intentionally
no member to resume trading, change policy or mode, approve orders, reconcile, read secrets or touch files --
so no tool can do any of that.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from packages.contracts import RiskPolicy, TradeProposal
from packages.contracts.enums import ExecutionLabel
from packages.core.config import Settings

READ_METHODS: tuple[str, ...] = (
    "search_memory", "status", "market_bars", "market_quality", "market_snapshot", "predictions", "model_report", "signals",
    "portfolio", "orders", "order_detail", "risk_view", "now", "performance", "health_report", "activity",
)
MUTATING_METHODS: tuple[str, ...] = ("submit_agent_proposal", "halt", "cancel", "run_cycle", "sessions")
# Members of ``ToolRuntime.sessions`` (``SessionControl``). Read tools may call only the read members.
SESSION_READ_METHODS: tuple[str, ...] = ("active", "list", "report", "list_schedules", "explain", "scheduler_status")
SESSION_MUTATING_METHODS: tuple[str, ...] = ("start", "stop", "add_schedule", "cancel_schedule")


class SessionControl(Protocol):
    """``services.sessions.manager.SessionManager`` as seen by the tools (sync, thread-safe; raises SessionError)."""

    def start(self, duration_minutes: int, *, symbols: Sequence[str] | None = None, cadence_minutes: int = 5,
              use_ai: bool = True, requested_by: str) -> dict[str, Any]: ...

    def stop(self, session_id: str | None = None, *, reason: str, requested_by: str) -> dict[str, Any]: ...

    def add_schedule(self, weekdays: Sequence[str], start_time_et: str, duration_minutes: int,
                     symbols: Sequence[str] | None = None, cadence_minutes: int = 5, use_ai: bool = True,
                     created_by: str = "system") -> dict[str, Any]: ...

    def cancel_schedule(self, schedule_id: str, *, requested_by: str = "system") -> dict[str, Any]: ...

    def active(self) -> dict[str, Any] | None: ...

    def list(self, limit: int = 20) -> builtins.list[dict[str, Any]]: ...

    def report(self, session_id: str | None = None) -> dict[str, Any] | None: ...

    def list_schedules(self) -> builtins.list[dict[str, Any]]: ...

    def explain(self, limit: int = 10) -> dict[str, Any]: ...

    def scheduler_status(self) -> dict[str, Any]: ...


class ActivityReader(Protocol):
    def recent(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class ToolRuntime(Protocol):
    @property
    def settings(self) -> Settings: ...

    @property
    def policy(self) -> RiskPolicy: ...

    @property
    def dataset_version(self) -> str: ...

    @property
    def execution_label(self) -> ExecutionLabel: ...

    def search_memory(self, query: str, *, symbol: str | None = None,
                      kind: str | None = None, limit: int = 10) -> dict[str, Any]: ...

    def now(self) -> datetime: ...

    def status(self) -> dict[str, Any]: ...

    def market_bars(self, symbol: str, timeframe: str, limit: int) -> dict[str, Any]: ...

    def market_quality(self) -> dict[str, Any]: ...

    async def market_snapshot(self, symbol: str) -> Any: ...

    def predictions(self, symbol: str | None = None) -> dict[str, Any]: ...

    def model_report(self) -> dict[str, Any]: ...

    def signals(self, limit: int = 50) -> dict[str, Any]: ...

    def portfolio(self) -> dict[str, Any]: ...

    def orders(self, limit: int = 100) -> dict[str, Any]: ...

    def order_detail(self, client_order_id: str) -> dict[str, Any] | None: ...

    def risk_view(self) -> dict[str, Any]: ...

    async def submit_agent_proposal(self, proposal: TradeProposal, actor: str) -> dict[str, Any]: ...

    def halt(self, reason: str, source: str) -> str: ...

    async def cancel(self, client_order_id: str) -> dict[str, Any]: ...

    async def run_cycle(self, *, symbols: list[str] | None, use_ai: bool, actor: str) -> dict[str, Any]: ...

    @property
    def sessions(self) -> SessionControl | None: ...

    @property
    def activity(self) -> ActivityReader: ...

    def performance(self) -> dict[str, Any]: ...

    def health_report(self) -> dict[str, Any]: ...
