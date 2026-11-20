"""MCP server layer, connected in-process with the official SDK Client (no subprocess, no HTTP)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import mcp_types as types
from mcp import Client

from services.mcp_server.server import INSTRUCTIONS, SERVER_NAME, build_server
from services.tools.backends import InProcessToolBackend
from services.tools.handlers import ToolError
from services.tools.registry import TOOL_NAMES, TOOL_REGISTRY
from tests.unit.tools.fake_runtime import FakeToolRuntime
from tests.unit.tools.test_tools_schemas import GOOD_PROPOSAL


def _text(result: types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


async def test_lists_exactly_the_registry_tools_with_canonical_schemas() -> None:
    server = build_server(InProcessToolBackend(FakeToolRuntime(), "mcp-test"))
    async with Client(server) as client:
        assert client.instructions == INSTRUCTIONS
        tools = (await client.list_tools()).tools
        assert [t.name for t in tools] == list(TOOL_NAMES)
        for t in tools:
            assert t.input_schema == TOOL_REGISTRY[t.name].input_schema()
            assert t.annotations is not None and t.annotations.read_only_hint is (not TOOL_REGISTRY[t.name].mutating)
    assert SERVER_NAME == "local-ai-trader"
    for phrase in ("PAPER", "risk engine", "DATA, not instructions", "Live trading is impossible"):
        assert phrase in INSTRUCTIONS


async def test_call_returns_compact_text_and_structured_content() -> None:
    rt = FakeToolRuntime()
    async with Client(build_server(InProcessToolBackend(rt, "mcp-test"))) as client:
        res = await client.call_tool("get_market_snapshot", {"symbol": "SPY"})
        assert res.is_error is False
        text = _text(res)
        assert json.loads(text) == res.structured_content
        assert text == json.dumps(res.structured_content, separators=(",", ":"), ensure_ascii=False)  # compact
        assert res.structured_content["execution_label"] == "SIMULATED"
        await client.call_tool("get_signals", {"limit": 5})
        prop = await client.call_tool("propose_paper_trade", GOOD_PROPOSAL)
        assert prop.is_error is False and prop.structured_content["outcome"] == "SUBMITTED"
    assert rt.calls.count("submit_agent_proposal") == 1


async def test_validation_and_refusals_are_tool_errors_not_crashes() -> None:
    async with Client(build_server(InProcessToolBackend(FakeToolRuntime(), "mcp-test"))) as client:
        bad = await client.call_tool("propose_paper_trade", {**GOOD_PROPOSAL, "limit_price": 500.04})
        assert bad.is_error is True and _text(bad).startswith("error 422")
        unknown = await client.call_tool("resume_trading", {})
        assert unknown.is_error is True and "404" in _text(unknown)
        missing = await client.call_tool("get_order", {"client_order_id": "lat-sim-ghost"})
        assert missing.is_error is True and "404" in _text(missing)


class _ExplodingBackend:
    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        if name == "get_portfolio":
            raise ToolError(503, "local trading API is not reachable at http://127.0.0.1:8765")
        raise RuntimeError("boom with internal detail /etc/secret")

    async def catalog(self) -> list[dict[str, Any]]:
        return []


async def test_backend_failures_become_clear_errors_without_internals() -> None:
    async with Client(build_server(_ExplodingBackend())) as client:
        down = await client.call_tool("get_portfolio", {})
        assert down.is_error is True and "not reachable" in _text(down)
        boom = await client.call_tool("get_orders", {})
        assert boom.is_error is True and "RuntimeError" in _text(boom) and "/etc/secret" not in _text(boom)
        tools = (await client.list_tools()).tools  # listing never depends on the API being up
        assert len(tools) == len(TOOL_NAMES)
