"""Every tool offered to every agent, in every scenario, is read-only. No order/limit/broker tool ever exists."""

from __future__ import annotations

import json
from datetime import timedelta

from services.ai.fake_transport import FakeScenario, medium_flagged_evidence
from services.ai.tools import (
    CLAUDE_TOOL_NAMES,
    FORBIDDEN_TOOL_NAME_RE,
    SOL_OUTPUT_TOOL,
    SOL_TOOL_NAMES,
    claude_tools,
    sol_tools,
)

from .conftest import T0

SOL_EXPECTED = {*SOL_TOOL_NAMES, SOL_OUTPUT_TOOL}
CLAUDE_EXPECTED = set(CLAUDE_TOOL_NAMES)


def test_declared_tool_lists_are_read_only() -> None:
    assert {t.name for t in sol_tools()} == set(SOL_TOOL_NAMES)
    assert {t.name for t in claude_tools()} == CLAUDE_EXPECTED
    assert "request_deep_review" not in CLAUDE_EXPECTED  # no recursion, Claude cannot call Sol
    for name in SOL_EXPECTED | CLAUDE_EXPECTED:
        assert not FORBIDDEN_TOOL_NAME_RE.search(name), name
    # sanity: the guard regex does catch execution-style names
    for bad in ("submit_order", "place_order", "cancel_order", "submit_live_order", "set_risk_limits", "update_policy",
                "get_api_key", "unlock_entries", "change_mode", "replace_order", "propose_paper_trade", "request_halt",
                "cancel_paper_order", "run_quant_cycle", "resume_trading", "read_agent_token", "chatgpt_login"):
        assert FORBIDDEN_TOOL_NAME_RE.search(bad), bad


async def test_every_offered_tool_in_every_request_is_allowlisted(harness_factory) -> None:
    seen_sol: set[str] = set()
    seen_claude: set[str] = set()
    for sol, claude in [
        (FakeScenario.TRADE, FakeScenario.TRADE),
        (FakeScenario.TRADE, FakeScenario.NO_TRADE),
        (FakeScenario.NO_TRADE, FakeScenario.TRADE),
        (FakeScenario.INJECTION, FakeScenario.INJECTION),
        (FakeScenario.SCHEMA_INVALID, FakeScenario.SCHEMA_INVALID),
        (FakeScenario.TRADE, FakeScenario.TIMEOUT),
    ]:
        h = harness_factory(sol, claude)
        await h.run(extra_evidence=[medium_flagged_evidence(T0 - timedelta(minutes=1))])
        assert h.sol.exchanges, (sol, claude)
        for names in h.sol.offered_tool_names():
            assert set(names) == SOL_EXPECTED, names
            seen_sol.update(names)
        for names in h.claude.offered_tool_names():
            assert set(names) == CLAUDE_EXPECTED, names
            seen_claude.update(names)
        for body in [*h.sol.requests, *h.claude.requests]:
            for tool in body.get("tools", []):
                assert not FORBIDDEN_TOOL_NAME_RE.search(tool["name"]), tool["name"]
            # no server-side/native tools (web, code execution, computer use, MCP) are ever offered
            assert all(t.get("type", "function") in ("function", "custom") for t in body.get("tools", []))
            assert "mcp_servers" not in body
    assert seen_sol == SOL_EXPECTED and seen_claude == CLAUDE_EXPECTED


async def test_attempted_order_tool_is_not_executable(harness_factory) -> None:
    h = harness_factory(FakeScenario.INJECTION, FakeScenario.TRADE)
    r = await h.run()
    # the model asked for submit_order; PydanticAI answered with an unknown-tool error and nothing ran
    outputs = [i for b in h.sol.requests for i in b["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    calls = {i["call_id"]: i["name"] for b in h.sol.requests for i in b["input"]
             if isinstance(i, dict) and i.get("type") == "function_call"}
    order_outputs = [o for o in outputs if calls.get(o["call_id"]) == "submit_order"]
    assert order_outputs and all("Unknown tool" in o["output"] or "unknown" in o["output"].lower() for o in order_outputs)
    assert r.proposal.action.value == "NO_TRADE"


async def test_no_secret_or_key_material_in_any_request_body(harness_factory) -> None:
    h = harness_factory()
    await h.run()
    blob = json.dumps(h.sol.requests) + json.dumps(h.claude.requests)
    for needle in ("api_key", "sk-ant-", "lat-fake-transport", "ALPACA", "APCA", "Bearer", "secret_key"):
        assert needle not in blob, needle
