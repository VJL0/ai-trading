"""The API-key pipeline on the ONE shared tool interface: with a ``services.tools`` backend, Sol and Claude are
offered the pipeline-context tools plus the shared registry's READ-ONLY tools (PydanticAI export bound to an
InProcessToolBackend) -- never propose_paper_trade / request_halt / cancel_paper_order / run_quant_cycle."""

from __future__ import annotations

import json
from typing import Any

from packages.contracts.enums import DecisionAction, ModelRunStatus
from services.ai.fake_transport import FakeScenario
from services.ai.tools import (
    CONTEXT_TOOL_NAMES,
    FORBIDDEN_TOOL_NAME_RE,
    SOL_OUTPUT_TOOL,
    market_tools,
    pipeline_tool_names,
    shared_read_only_tool_names,
)
from services.tools.backends import InProcessToolBackend
from services.tools.registry import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, TOOL_REGISTRY
from tests.unit.tools.fake_runtime import FakeToolRuntime

from .chatgpt_helpers import MUTATING_RUNTIME_CALLS

SOL_SHARED = {*CONTEXT_TOOL_NAMES, "request_deep_review", *READ_ONLY_TOOL_NAMES, SOL_OUTPUT_TOOL}
CLAUDE_SHARED = {*CONTEXT_TOOL_NAMES, *READ_ONLY_TOOL_NAMES}


def test_shared_market_tools_are_read_only() -> None:
    rt = FakeToolRuntime()
    tools = market_tools(InProcessToolBackend(rt, "ai-pipeline"))
    assert {t.name for t in tools} == set(READ_ONLY_TOOL_NAMES) == set(shared_read_only_tool_names())
    assert all(not TOOL_REGISTRY[t.name].mutating for t in tools)
    assert not {t.name for t in tools} & set(MUTATING_TOOL_NAMES)
    assert set(pipeline_tool_names(shared=True, deep_review=True)) | {SOL_OUTPUT_TOOL} == SOL_SHARED
    for name in SOL_SHARED:
        assert not FORBIDDEN_TOOL_NAME_RE.search(name), name
    for name in MUTATING_TOOL_NAMES:  # the guard regex catches every mutating shared tool name
        assert FORBIDDEN_TOOL_NAME_RE.search(name), name


async def test_pipeline_with_shared_backend_trade_path(harness_factory: Any) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    rt = FakeToolRuntime()
    backend = InProcessToolBackend(rt, "ai-pipeline")
    r = await h.pipeline.run(causal_id="cz_shared", candidate=_sig(), snapshot=_snap(), evidence=_ev(),
                             policy=_pol(), tool_backend=backend)
    assert r.proposal.action == DecisionAction.TRADE, r.proposal.no_trade_reason
    assert {x.status for x in r.model_runs} == {ModelRunStatus.OK}
    for names in h.sol.offered_tool_names():
        assert set(names) == SOL_SHARED, names
    for names in h.claude.offered_tool_names():
        assert set(names) == CLAUDE_SHARED, names
    # the shared tools really executed through the backend, and only read methods were touched
    assert "market_snapshot" in rt.calls and "predictions" in rt.calls
    assert not [c for c in rt.calls if c in MUTATING_RUNTIME_CALLS]
    # the strict shared schema reached the provider unchanged
    sol_tools = {t["name"]: t for t in h.sol.requests[0]["tools"]}
    assert sol_tools["get_market_snapshot"]["parameters"]["required"] == ["symbol"]
    assert h.sol.violations == [] and h.claude.violations == []
    blob = json.dumps(h.sol.requests) + json.dumps(h.claude.requests)
    for needle in ("agent.token", "X-LAT-Agent-Token", "operator.token", "engine.token"):
        assert needle not in blob


async def test_attribute_backend_is_used_and_injection_scenario_still_refused(harness_factory: Any) -> None:
    h = harness_factory(FakeScenario.INJECTION, FakeScenario.TRADE)
    rt = FakeToolRuntime()
    h.pipeline.tool_backend = InProcessToolBackend(rt, "ai-pipeline")
    r = await h.pipeline.run(causal_id="cz_inj", candidate=_sig(), snapshot=_snap(), evidence=_ev(), policy=_pol())
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert any("submit_order" in n for n in r.notes) or "submit_order" in (r.proposal.no_trade_reason or "")
    assert not [c for c in rt.calls if c in MUTATING_RUNTIME_CALLS]


def _sig() -> Any:
    from .conftest import make_signal

    return make_signal()


def _snap() -> Any:
    from .conftest import make_snapshot

    return make_snapshot()


def _pol() -> Any:
    from packages.contracts import RiskPolicy

    return RiskPolicy()


def _ev() -> list[Any]:
    from services.ai.evidence import build_internal_evidence

    return build_internal_evidence(_sig(), _snap(), _pol())


def test_anthropic_schema_adaptation_is_equivalent() -> None:
    from services.ai.tools import nullable_types_as_any_of

    src = {"type": "object", "properties": {"limit": {"type": ["integer", "null"], "minimum": 1, "description": "n"},
                                            "symbol": {"type": "string"}},
           "required": ["limit", "symbol"], "additionalProperties": False}
    out = nullable_types_as_any_of(src)
    assert out["properties"]["limit"] == {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
                                          "description": "n"}
    assert out["properties"]["symbol"] == {"type": "string"} and out["required"] == ["limit", "symbol"]
    rt = FakeToolRuntime()
    for tool in market_tools(InProcessToolBackend(rt, "x"), provider="anthropic"):
        blob = json.dumps(tool.function_schema.json_schema)
        assert '"type": [' not in blob, tool.name
        assert tool.strict is True
