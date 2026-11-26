"""The autopilot's only door into the trading system: the agent tool interface, through an allowlist.

* Transport: the same ``HttpToolBackend`` the MCP server uses (loopback HTTP, ``X-LAT-Agent-Token`` read from
  ``var/<ns>/keys/agent.token``, actor ``autopilot`` so sessions record ``requested_by=agent:autopilot``). The
  operator token is never read: the backend is handed the agent token path and nothing else.
* Authority: ``GuardedBackend`` refuses every tool outside ``ALLOWED_TOOLS``. The one mutating tool is
  ``start_paper_session`` (a bounded PAPER session whose every order still goes through the risk engine). There is
  no way through this module to stop a session, cancel or propose orders, halt or resume, schedule, run research
  experiments, change policy, or reach a broker.
* Mode: ``require_paper_runtime`` refuses to build a backend unless the configured runtime is Paper on the FAKE or
  ALPACA_PAPER gateway. The autopilot never selects Live; it never even sets a mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from stat import S_ISREG
from typing import Any, Final, Protocol

from packages.contracts.enums import GatewayKind, TradingMode
from packages.core.config import Settings
from packages.core.modes import OperatorTradingMode

ACTOR: Final = "autopilot"
REQUESTED_BY: Final = f"agent:{ACTOR}"
AGENT_TOKEN_FILENAME: Final = "agent.token"  # noqa: S105 - a file name, not a secret

READ_TOOLS: Final = frozenset({
    "get_session_status", "list_schedules", "get_session_report", "get_performance", "get_system_health",
    "get_feed_quality", "get_risk_status", "get_orders", "get_research_results", "explain_decisions",
})
MUTATING_TOOLS: Final = frozenset({"start_paper_session"})
ALLOWED_TOOLS: Final = READ_TOOLS | MUTATING_TOOLS
_START_FIELDS: Final = frozenset({"duration_minutes", "symbols", "cadence_minutes", "use_ai"})


class AuthorityError(PermissionError):
    """The autopilot tried to do something outside its authority (a bug, never a runtime condition)."""


class ToolPort(Protocol):
    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]: ...


class GuardedBackend:
    def __init__(self, inner: ToolPort, *, token_path: Path | None = None) -> None:
        self._inner = inner
        self._token_path = token_path

    async def call(self, name: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if name not in ALLOWED_TOOLS:
            raise AuthorityError(f"tool {name!r} is outside the autopilot's authority")
        payload = dict(args or {})
        if name == "start_paper_session" and set(payload) - _START_FIELDS:
            raise AuthorityError(f"start_paper_session fields outside the contract: {sorted(set(payload) - _START_FIELDS)}")
        if self._token_path is not None:
            try:
                info = self._token_path.lstat()
            except FileNotFoundError:
                pass  # let the HTTP transport report a token which has yet to be created
            else:
                if not S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise AuthorityError("agent token must be a standalone regular file")
        return await self._inner.call(name, payload)

    async def aclose(self) -> None:
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()


def require_paper_runtime(settings: Settings) -> None:
    """Refuse anything but the Paper runtime on a paper or simulated gateway."""
    if settings.trading_mode == OperatorTradingMode.LIVE:
        raise AuthorityError("the autopilot never runs with Live selected")
    if settings.mode != TradingMode.PAPER:
        raise AuthorityError(f"the autopilot runs only in Paper mode (configured: {settings.mode.value})")
    if settings.gateway not in (GatewayKind.FAKE, GatewayKind.ALPACA_PAPER):
        raise AuthorityError(f"the autopilot runs only on a paper or simulated gateway (configured: {settings.gateway.value})")


def agent_token_path(settings: Settings) -> Path:
    return settings.keys_dir / AGENT_TOKEN_FILENAME


def make_http_backend(settings: Settings) -> GuardedBackend:
    """Use the MCP transport with the autopilot actor. API startup is handled explicitly before pre-open checks."""
    from services.tools.backends import HttpToolBackend

    require_paper_runtime(settings)
    token_path = agent_token_path(settings)
    return GuardedBackend(HttpToolBackend(settings.api_url, token_path, ACTOR, lazy_token=True), token_path=token_path)


def ensure_api_running(settings: Settings) -> dict[str, Any]:
    """Start the API if nothing serves this workspace (same launcher and checks as the MCP path)."""
    from services.api.launcher import ensure_api

    require_paper_runtime(settings)
    return ensure_api(settings, require_gateway_match=settings.trading_mode is not None
                      or "gateway" in settings.model_fields_set)
