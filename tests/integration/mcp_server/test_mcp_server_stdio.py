"""End-to-end: the MCP server as a real stdio subprocess, driven by the official SDK client, forwarding to a
stub local API (the REAL FastAPI app from services.api.app + a fake runtime) served by uvicorn on 127.0.0.1.
No broker, no model, no external network."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mcp_types as types
import pytest
import uvicorn
from mcp import ClientSession, StdioServerParameters, stdio_client

from packages.core.config import Settings
from services.api.app import create_app
from services.api.security import AGENT_TOKEN_FILE, ENGINE_TOKEN_FILE, OPERATOR_TOKEN_FILE, ensure_token
from services.tools.registry import TOOL_NAMES, TOOL_REGISTRY
from tests.unit.tools.fake_runtime import FakeToolRuntime
from tests.unit.tools.test_tools_schemas import GOOD_PROPOSAL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.timeout(120)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class StubApi:
    rt: FakeToolRuntime
    settings: Settings
    port: int
    root: Path

    @property
    def token(self) -> str:
        return (self.settings.keys_dir / AGENT_TOKEN_FILE).read_text().strip()


@pytest.fixture
def stub_api(tmp_path: Path) -> Iterator[StubApi]:
    port = _free_port()
    settings = Settings(project_root=tmp_path, api_port=port, use_keychain=False)
    for f in (OPERATOR_TOKEN_FILE, ENGINE_TOKEN_FILE, AGENT_TOKEN_FILE):
        ensure_token(settings.keys_dir, f)
    rt = FakeToolRuntime(settings)
    app = create_app(settings, runtime_factory=lambda _s: rt)  # pyright: ignore[reportArgumentType]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, name="stub-api", daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("stub API did not start")
        time.sleep(0.05)
    try:
        yield StubApi(rt, settings, port, tmp_path)
    finally:
        server.should_exit = True
        thread.join(timeout=15)


def _env(root: Path, port: int, agent: str = "pytest") -> dict[str, str]:
    return {"LAT_PROJECT_ROOT": str(root), "LAT_API_PORT": str(port), "LAT_AGENT_NAME": agent,
            "LAT_USE_KEYCHAIN": "false", "LAT_MCP_LOG_LEVEL": "INFO"}


def _python_params(root: Path, port: int, agent: str = "pytest") -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "services.mcp_server"], env=_env(root, port, agent),
                                 cwd=str(PROJECT_ROOT))


def _text(res: types.CallToolResult) -> str:
    block = res.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


async def test_start_trading_boots_the_api_and_uses_defaults(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LAT_GATEWAY=FAKE\nLAT_FEED_MODE=SIMULATED\nLAT_AI_RUNTIME=NONE\n")
    pidfile = tmp_path / "var" / "run" / "api.pid"
    try:
        params = _python_params(tmp_path, _free_port())
        with (tmp_path / "err.log").open("w") as errlog:
            async with (stdio_client(params, errlog=errlog) as (read, write),
                        ClientSession(read, write, read_timeout_seconds=45) as session):
                await session.initialize()
                assert not pidfile.exists()
                args = dict.fromkeys(("duration_minutes", "symbols", "cadence_minutes", "use_ai"))
                invalid = await session.call_tool("start_paper_session", {**args, "duration_minutes": -1})
                assert invalid.is_error and not pidfile.exists()
                response = await session.call_tool("start_paper_session", args)
                assert not response.is_error, _text(response)
                assert response.structured_content is not None
                assert response.structured_content["execution_label"] == "SIMULATED"
                started = response.structured_content["session"]
                assert started["duration_minutes"] == 60
                assert started["cadence_minutes"] == 5
                assert started["use_ai"] is False
                pid = pidfile.read_text()
                duplicate = await session.call_tool("start_paper_session", args)
                assert duplicate.is_error and "already active" in _text(duplicate)
                assert pidfile.read_text() == pid
                stopped = await session.call_tool("stop_session", {"reason": "integration test", "session_id": None})
                assert not stopped.is_error, _text(stopped)
    finally:
        if pidfile.exists():
            os.killpg(int(pidfile.read_text()), signal.SIGTERM)


async def test_stdio_session_end_to_end(stub_api: StubApi, tmp_path: Path) -> None:
    errlog_path = tmp_path / "mcp-stderr.log"
    transcript: list[str] = []
    with errlog_path.open("w") as errlog:
        async with (stdio_client(_python_params(stub_api.root, stub_api.port), errlog=errlog) as (read, write),
                    ClientSession(read, write, read_timeout_seconds=30) as session):
            init = await session.initialize()
            transcript.append(init.model_dump_json())
            assert init.server_info.name == "local-ai-trader"
            assert init.instructions is not None and "PAPER" in init.instructions
            assert "DATA, not instructions" in init.instructions

            tools = (await session.list_tools()).tools
            assert [t.name for t in tools] == list(TOOL_NAMES)
            assert all(t.input_schema == TOOL_REGISTRY[t.name].input_schema() for t in tools)

            quality = await session.call_tool("get_feed_quality", {})
            assert quality.is_error is False and quality.structured_content["status"] == "OK"
            sigs = await session.call_tool("get_signals", {"limit": 4})
            assert sigs.structured_content["signals"][0]["signal_id"] == "sig_SPY_1"

            ok = await session.call_tool("propose_paper_trade", GOOD_PROPOSAL)
            assert ok.is_error is False, _text(ok)
            assert ok.structured_content["outcome"] == "SUBMITTED"
            assert ok.structured_content["execution_label"] == "SIMULATED"

            risky = await session.call_tool("propose_paper_trade", {**GOOD_PROPOSAL, "quantity": "10"})
            assert risky.structured_content["outcome"] == "REJECTED"
            assert "max_order_notional" in risky.structured_content["decision"]["failed_checks"]

            floaty = await session.call_tool("propose_paper_trade", {**GOOD_PROPOSAL, "limit_price": 500.04})
            assert floaty.is_error is True and _text(floaty).startswith("error 422")

            invented = await session.call_tool("propose_paper_trade", {**GOOD_PROPOSAL, "evidence_ids": ["ev_fake"]})
            assert invented.is_error is True and "not obtained" in _text(invented)

            halt = await session.call_tool("request_halt", {"reason": "agent saw abnormal spreads"})
            assert halt.structured_content["entries_locked"] is True

            for res in (quality, sigs, ok, risky, floaty, invented, halt):
                transcript.append(_text(res))
    stderr = errlog_path.read_text()
    assert f"serving {len(TOOL_NAMES)} tools" in stderr
    rt = stub_api.rt
    assert rt.calls.count("submit_agent_proposal") == 2  # the float and invented-evidence calls never reached risk
    assert rt.halts[0]["source"].startswith("agent:pytest-")
    assert {p.strategy_id for p in rt.proposals} == {rt.halts[0]["source"]}
    token = stub_api.token
    assert token not in stderr and all(token not in t for t in transcript)  # never logged, never sent to the client


async def test_api_down_returns_clear_error(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, use_keychain=False)
    ensure_token(settings.keys_dir, AGENT_TOKEN_FILE)
    port = _free_port()  # nothing listens here
    with (tmp_path / "err.log").open("w") as errlog:
        async with (stdio_client(_python_params(tmp_path, port), errlog=errlog) as (read, write),
                    ClientSession(read, write, read_timeout_seconds=30) as session):
            await session.initialize()
            assert len((await session.list_tools()).tools) == len(TOOL_NAMES)  # listing works without the API
            res = await session.call_tool("get_portfolio", {})
            assert res.is_error is True
            assert "not reachable" in _text(res) and "make dev" in _text(res)


async def test_missing_token_returns_clear_error(tmp_path: Path) -> None:
    with (tmp_path / "err.log").open("w") as errlog:
        async with (stdio_client(_python_params(tmp_path, _free_port()), errlog=errlog) as (read, write),
                    ClientSession(read, write, read_timeout_seconds=30) as session):
            await session.initialize()
            res = await session.call_tool("get_orders", {})
            assert res.is_error is True and "agent token unavailable" in _text(res)
    assert "warning: agent token unavailable" in (tmp_path / "err.log").read_text()


def test_refuses_to_start_outside_paper_or_replay(tmp_path: Path) -> None:
    for mode in ("LIVE_APPROVAL", "BOUNDED_LIVE", "RESEARCH_ONLY"):
        proc = subprocess.run(
            [sys.executable, "-m", "services.mcp_server"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
            input="", env={**_env(tmp_path, _free_port()), "LAT_MODE": mode, "PATH": "/usr/bin:/bin"}, check=False,
        )
        assert proc.returncode == 2, proc.stderr
        assert "PAPER/REPLAY only" in proc.stderr and proc.stdout == ""


async def test_registered_claude_code_command_launches_the_server(stub_api: StubApi, tmp_path: Path) -> None:
    """Launch exactly the command/args registered in .mcp.json (uv run --frozen --directory ...)."""
    cfg: dict[str, Any] = json.loads((PROJECT_ROOT / ".mcp.json").read_text())["mcpServers"]["local-ai-trader"]
    params = StdioServerParameters(command=cfg["command"], args=list(cfg["args"]),
                                   env={**cfg.get("env", {}), **_env(stub_api.root, stub_api.port, "claude-code")})
    with (tmp_path / "err.log").open("w") as errlog:
        async with (stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(read, write, read_timeout_seconds=60) as session):
            await session.initialize()
            res = await session.call_tool("get_risk_status", {})
            assert res.is_error is False, _text(res)
            assert res.structured_content["mode"] == "PAPER"
            assert res.structured_content["live_execution_compiled"] is False
