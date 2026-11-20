"""The PydanticAI export runs inside a real Agent (scripted FunctionModel, no network): tool calls execute through
the backend, a 422 becomes a model retry, and the tool definitions sent to the model are the canonical ones."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from services.tools.backends import InProcessToolBackend
from services.tools.registry import TOOL_NAMES, TOOL_REGISTRY, pydantic_ai_tools
from tests.unit.tools.fake_runtime import FakeToolRuntime
from tests.unit.tools.test_tools_schemas import GOOD_PROPOSAL


def _parts(messages: list[ModelMessage], kind: type) -> list[Any]:
    return [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, kind)]


async def test_agent_run_executes_registry_tools_through_backend() -> None:
    rt = FakeToolRuntime()
    seen_defs: dict[str, Any] = {}
    script = [
        [ToolCallPart("get_signals", {"limit": 3, "symbol": None, "candidates_only": None})],
        [ToolCallPart("propose_paper_trade", {**GOOD_PROPOSAL, "limit_price": 500.04})],  # float -> 422 -> retry
        [ToolCallPart("propose_paper_trade", GOOD_PROPOSAL)],
        [TextPart("done")],
    ]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_defs.update({t.name: t for t in info.function_tools})
        return ModelResponse(parts=script.pop(0))

    agent = Agent(FunctionModel(model), tools=pydantic_ai_tools(InProcessToolBackend(rt, "chatgpt")))
    result = await agent.run("research SPY")
    assert result.output == "done"
    assert list(seen_defs) == list(TOOL_NAMES)
    for name, td in seen_defs.items():
        assert td.parameters_json_schema == TOOL_REGISTRY[name].input_schema()
        assert td.strict is True
    msgs = result.all_messages()
    returns = [p for p in _parts(msgs, ToolReturnPart) if p.tool_name == "propose_paper_trade"]
    assert len(returns) == 2
    assert returns[0].outcome == "retried"
    assert "binary floats" in str(returns[0].content)  # the 422 reached the model as a correctable retry
    assert returns[1].outcome == "success" and returns[1].content["outcome"] == "SUBMITTED"
    assert rt.calls.count("submit_agent_proposal") == 1  # the invalid call never reached the runtime
    assert rt.proposals[0].strategy_id == "agent:chatgpt"


async def test_refusal_becomes_terminal_tool_failure() -> None:
    rt = FakeToolRuntime()
    script = [[ToolCallPart("get_order", {"client_order_id": "lat-sim-ghost"})], [TextPart("ok")]]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=script.pop(0))

    agent = Agent(FunctionModel(model), tools=pydantic_ai_tools(InProcessToolBackend(rt, "x")))
    result = await agent.run("look up an order")
    failed = [p for p in _parts(result.all_messages(), ToolReturnPart) if p.tool_name == "get_order"]
    assert len(failed) == 1 and failed[0].outcome == "failed"  # ToolFailed: terminal, not a retry
    assert "unknown client_order_id" in str(failed[0].content)
