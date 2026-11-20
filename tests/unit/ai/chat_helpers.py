"""Fakes for the natural-language chat tests: a local trading API (``httpx.MockTransport`` behind the real
``HttpToolBackend``) whose live catalog = the shared registry's core tools PLUS catalog-only session/activity tools
that the local registry does not know (exercising the catalog-only call path), and a scripted console.

The served registry subset is fixed (``BASE_TOOLS``) so these tests do not depend on tools other agents add to
``services.tools`` concurrently; the integration test against the real API uses whatever the registry contains."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from services.tools.handlers import ToolError, dispatch
from services.tools.registry import catalog
from tests.unit.tools.fake_runtime import FakeToolRuntime

AGENT_TOKEN = "agent-token-test-0123456789abcdef"  # a test value, asserted never to reach a model
BASE_TOOLS: tuple[str, ...] = (
    "get_market_snapshot", "get_bars", "get_feed_quality", "get_predictions", "get_model_report", "get_signals",
    "get_portfolio", "get_orders", "get_order", "get_risk_status", "propose_paper_trade", "request_halt",
    "cancel_paper_order", "run_quant_cycle",
)

START_SESSION: dict[str, Any] = {
    "name": "start_test_session",
    "description": "Start a PAPER trading session run by the trading service (independent of this chat).",
    "category": "sessions",
    "mutating": True,
    "read_only": False,
    "input_schema": {
        "type": "object",
        "properties": {
            "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 480},
            "symbols": {"type": ["array", "null"], "items": {"type": "string"}},
            "cadence_minutes": {"type": ["integer", "null"]},
            "use_ai": {"type": ["boolean", "null"]},
        },
        "required": ["duration_minutes", "symbols", "cadence_minutes", "use_ai"],
        "additionalProperties": False,
    },
    "output_schema": {"type": "object"},
    "registry_version": "lat-tools-v1",
}
GET_ACTIVITY: dict[str, Any] = {
    "name": "get_test_activity",
    "description": "Recent activity: stage transitions, decisions (with exact NO_TRADE reasons) and session events.",
    "category": "sessions",
    "mutating": False,
    "read_only": True,
    "input_schema": {
        "type": "object",
        "properties": {"limit": {"type": ["integer", "null"]}, "kind": {"type": ["string", "null"]}},
        "required": ["limit", "kind"],
        "additionalProperties": False,
    },
    "output_schema": {"type": "object"},
    "registry_version": "lat-tools-v1",
}
NO_TRADE_REASON = "net edge 2.1 bps is below the estimated round-trip cost of 3.4 bps"


class FakeAgentApi:
    """``POST/GET /api/v1/agent/tools`` of the local API: registry tools run through the REAL ``dispatch`` on a
    ``FakeToolRuntime`` (real risk engine); catalog-only tools are answered here."""

    def __init__(self, *, extra_tools: list[dict[str, Any]] | None = None, start_status: int = 200) -> None:
        self.runtime = FakeToolRuntime()
        self.extra = extra_tools if extra_tools is not None else [START_SESSION, GET_ACTIVITY]
        self.requests: list[tuple[str, str, Any]] = []
        self.auth_failures = 0
        self.start_status = start_status
        self.sessions: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-lat-agent-token") != AGENT_TOKEN:
            self.auth_failures += 1
            return httpx.Response(401, json={"detail": "bad agent token"})
        path = request.url.path
        body: Any = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, path, body))
        if request.method == "GET" and path == "/api/v1/agent/tools":
            base = [t for t in catalog() if t["name"] in BASE_TOOLS]
            return httpx.Response(200, json={"tools": [*base, *self.extra]})
        if request.method == "POST" and path.startswith("/api/v1/agent/tools/"):
            name = path.rsplit("/", 1)[-1]
            if name == START_SESSION["name"] and any(t["name"] == name for t in self.extra):
                if self.start_status != 200:
                    return httpx.Response(self.start_status, json={"detail": "internal error after accepting"})
                sess = {"session_id": f"sess_test_{len(self.sessions) + 1}", "state": "RUNNING",
                        "started_at": "2026-11-05T15:00:00Z", "ends_at": "2026-11-05T16:00:00Z",
                        "symbols": body.get("symbols") or ["SPY"], "duration_minutes": body["duration_minutes"]}
                self.sessions.append(sess)
                return httpx.Response(200, json={"session": sess, "execution_label": "SIMULATED", "is_simulated": True})
            if name == GET_ACTIVITY["name"] and any(t["name"] == name for t in self.extra):
                return httpx.Response(200, json={"events": [
                    {"kind": "decision", "ts": "2026-11-05T15:05:00Z", "symbol": "SPY", "action": "NO_TRADE",
                     "status": "NO_TRADE", "reason": NO_TRADE_REASON, "risk_failures": []},
                    {"kind": "scan", "ts": "2026-11-05T15:04:59Z", "message": "Scanned SPY,QQQ"},
                ], "execution_label": "SIMULATED", "is_simulated": True, "asof": "2026-11-05T15:05:01Z"})
            if name not in BASE_TOOLS:
                return httpx.Response(404, json={"detail": f"unknown tool {name}"})
            try:
                return httpx.Response(200, json=await dispatch(self.runtime, name, body or {}, actor="chat"))
            except ToolError as exc:
                return httpx.Response(exc.status, json={"detail": exc.detail})
        return httpx.Response(404, json={"detail": "not found"})

    def calls(self, name: str) -> list[Any]:
        return [b for m, p, b in self.requests if m == "POST" and p.endswith("/" + name)]


def write_agent_token(root: Path) -> Path:
    path = root / "agent.token"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(AGENT_TOKEN + "\n")
    return path


class ScriptedConsole:
    """Console I/O for tests: ``answers`` are returned to prompts in order (confirmations); output is captured."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or [])
        self.prompts: list[str] = []
        self.out: list[str] = []

    async def read(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else None

    def write(self, text: str) -> None:
        self.out.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.out)


def never_called(*_a: Any, **_k: Any) -> Callable[..., Any]:
    raise AssertionError("no network/tool access expected")
