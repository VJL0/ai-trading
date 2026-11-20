"""Adapter construction + exact request shapes on the provider wire (asserted from the fake transports)."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import httpx2
import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from packages.core.config import Settings
from packages.core.secrets import SecretStore
from services.ai.adapters import (
    AiTransportError,
    build_claude_model,
    build_sol_model,
)
from services.ai.fake_transport import (
    FAKE_API_KEY,
    FakeAnthropicHandler,
    FakeOpenAIHandler,
    FakeScenario,
    fake_http_client,
)
from services.ai.versions import ADAPTER_VERSION, prompt_bundle_version, prompt_version


async def _trade_run(harness_factory: Any) -> Any:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    await h.run()
    return h


async def test_claude_request_shape(harness_factory) -> None:
    h = await _trade_run(harness_factory)
    assert len(h.claude.exchanges) == 3  # list -> get_evidence (parallel) -> final structured output
    for ex in h.claude.exchanges:
        b = ex.body
        assert ex.method == "POST" and ex.path == "/v1/messages"
        assert b["model"] == "claude-opus-5-5"
        assert b["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert "budget_tokens" not in json.dumps(b["thinking"])
        assert b["output_config"]["effort"] == "high"
        fmt = b["output_config"]["format"]
        assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
        assert set(fmt["schema"]["required"]) >= {"recommendation", "citations", "counterarguments", "no_trade_reason"}
        assert "confidence_label" in fmt["schema"]["properties"]
        assert not {"temperature", "top_p", "top_k"} & set(b)
        assert b.get("tool_choice", {"type": "auto"})["type"] == "auto"  # forced tool_choice 400s on Opus 5.5
        assert b["stream"] is False and b["max_tokens"] == 16000
        assert b["messages"][-1]["role"] == "user"  # no prefill
        assert ex.headers.get("authorization") == "[REDACTED]" or ex.headers.get("x-api-key") == "[REDACTED]"
        assert ex.headers.get("anthropic-version")
    # thinking blocks returned by the provider are replayed unchanged in the tool loop
    second = h.claude.exchanges[1].body["messages"]
    assert any(blk.get("type") == "thinking" for m in second if m["role"] == "assistant" for blk in m["content"])
    assert h.claude.violations == []


async def test_sol_request_shape_is_responses_api(harness_factory) -> None:
    h = await _trade_run(harness_factory)
    assert len(h.sol.exchanges) == 3
    for ex in h.sol.exchanges:
        b = ex.body
        assert ex.method == "POST" and ex.path == "/v1/responses"
        assert b["model"] == "gpt-6.1-sol"
        assert b["reasoning"]["effort"] == "medium"
        assert "messages" not in b and "functions" not in b  # never Chat Completions
        assert isinstance(b["input"], list)
        assert b["store"] is False
        for tool in b["tools"]:
            assert tool["type"] == "function" and "function" not in tool and "parameters" in tool
        out = next(t for t in b["tools"] if t["name"] == "propose_trade")
        assert out["strict"] is True
        props = out["parameters"]["properties"]
        assert {"quantity", "limit_price", "citations", "confidence_label"} <= set(props)
        assert not {"max_order_notional", "policy", "probability", "limits"} & set(props)
    # tool results flow back as Responses function_call_output items
    last = h.sol.exchanges[-1].body["input"]
    assert any(i.get("type") == "function_call_output" for i in last if isinstance(i, dict))
    assert h.sol.violations == []


async def test_escalated_claude_effort_reaches_the_wire(settings, secrets) -> None:
    from pydantic_ai import Agent

    handler = FakeAnthropicHandler(FakeScenario.TRADE)
    model = build_claude_model(settings, secrets, transport="fake", http_client=fake_http_client(handler), effort="max")
    agent = Agent(model, output_type=str)
    with contextlib.suppress(Exception):  # the scripted handler expects research tools; only the request matters
        await agent.run("hello")
    assert handler.requests[0]["output_config"]["effort"] == "max"
    assert handler.requests[0]["thinking"]["type"] == "adaptive"


def test_fake_builders_return_pydantic_ai_models_and_never_read_keys(settings, tmp_path) -> None:
    store = SecretStore(tmp_path, use_keychain=False, env={"ANTHROPIC_API_KEY": "sk-ant-should-never-be-read-123456",
                                                          "OPENAI_API_KEY": "sk-should-never-be-read-1234567890"},
                        dotenv_path=tmp_path / "x.env")
    claude = build_claude_model(settings, store, transport="fake")
    sol = build_sol_model(settings, store, transport="fake")
    assert isinstance(claude, AnthropicModel) and isinstance(sol, OpenAIResponsesModel)
    assert claude.client.api_key == FAKE_API_KEY  # pyright: ignore[reportAttributeAccessIssue]
    assert sol.client.api_key == FAKE_API_KEY
    assert claude.model_name == "claude-opus-5-5" and sol.model_name == "gpt-6.1-sol"


def test_fake_transport_refuses_a_networked_client(settings, secrets) -> None:
    with pytest.raises(AiTransportError, match="MockTransport"):
        build_claude_model(settings, secrets, transport="fake", http_client=httpx2.AsyncClient())
    with pytest.raises(AiTransportError, match="MockTransport"):
        build_sol_model(settings, secrets, transport="fake", http_client=httpx2.AsyncClient())


def test_real_transport_requires_explicit_setting_and_key(tmp_path) -> None:
    keys = {"ANTHROPIC_API_KEY": "sk-ant-test-dummy-000000", "OPENAI_API_KEY": "sk-test-dummy-000000000000000"}
    with_keys = SecretStore(tmp_path, use_keychain=False, env=keys, dotenv_path=tmp_path / "x.env")
    no_keys = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "x.env")
    fake_setting = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False)
    real_setting = Settings(project_root=tmp_path, ai_transport="real", use_keychain=False)
    # keys present but LAT_AI_TRANSPORT != real -> refused (never inferred from key presence)
    with pytest.raises(AiTransportError, match="LAT_AI_TRANSPORT"):
        build_claude_model(fake_setting, with_keys, transport="real")
    with pytest.raises(AiTransportError, match="LAT_AI_TRANSPORT"):
        build_sol_model(fake_setting, with_keys, transport="real")
    # explicit real but no key -> refused
    with pytest.raises(AiTransportError, match="ANTHROPIC_API_KEY"):
        build_claude_model(real_setting, no_keys, transport="real")
    with pytest.raises(AiTransportError, match="OPENAI_API_KEY"):
        build_sol_model(real_setting, no_keys, transport="real")
    # real transport never accepts a mock client (it would mislabel FAKE runs as REAL)
    with pytest.raises(AiTransportError, match="mislabel"):
        build_claude_model(real_setting, with_keys, transport="real",
                           http_client=fake_http_client(FakeAnthropicHandler()))
    # explicit real + key: built (no request is sent here), key comes from the SecretStore
    claude = build_claude_model(real_setting, with_keys, transport="real")
    sol = build_sol_model(real_setting, with_keys, transport="real", effort="high")
    assert claude.client.api_key == keys["ANTHROPIC_API_KEY"]  # pyright: ignore[reportAttributeAccessIssue]
    assert sol.client.api_key == keys["OPENAI_API_KEY"]
    assert str(claude.client.base_url).startswith("https://api.anthropic.com")
    assert str(sol.client.base_url).startswith("https://api.openai.com/v1")
    assert claude.client.max_retries == 0 and sol.client.max_retries == 0  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AiTransportError):
        build_claude_model(real_setting, with_keys, transport="bogus")  # type: ignore[arg-type]


def test_versions_recorded_on_runs_are_prompt_hashes() -> None:
    assert ADAPTER_VERSION.startswith("lat-ai-adapters/") and "pydantic-ai/2." in ADAPTER_VERSION
    v = prompt_version("sol_orchestrator")
    assert v.startswith("sol_orchestrator@") and len(v.split("@")[1]) == 16
    assert prompt_bundle_version().startswith("prompts@")


async def test_runs_record_adapter_and_prompt_versions(harness_factory) -> None:
    h = harness_factory()
    r = await h.run()
    for rec in r.model_runs:
        assert rec.adapter_version == ADAPTER_VERSION
        assert rec.prompt_version in (prompt_version("sol_orchestrator"), prompt_version("claude_research"))
        assert rec.output_schema.split("@")[0] in ("TradeProposalDraft", "ResearchFindingDraft")
    assert {r.effort for r in r.model_runs} == {"medium", "high"}


def test_fake_handlers_reject_documented_constraint_violations() -> None:
    a = FakeAnthropicHandler()
    bad = {"model": "claude-opus-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "x"}],
           "thinking": {"type": "enabled", "budget_tokens": 2000}, "temperature": 0.2,
           "tool_choice": {"type": "any"}}
    resp = a(httpx2.Request("POST", "https://api.anthropic.com/v1/messages", json=bad, headers={"x-api-key": "k",
                                                                                                 "anthropic-version": "v"}))
    assert resp.status_code == 400
    joined = " ".join(a.violations)
    for needle in ("thinking.type.enabled", "budget_tokens", "temperature", 'tool_choice: type "tool" and "any"'):
        assert needle in joined
    o = FakeOpenAIHandler()
    chat_shape = {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "none"},
                  "tools": [{"type": "function", "function": {"name": "f"}}]}
    resp2 = o(httpx2.Request("POST", "https://api.openai.com/v1/chat/completions", json=chat_shape,
                             headers={"authorization": "Bearer k"}))
    assert resp2.status_code == 400
    joined2 = " ".join(o.violations)
    for needle in ("/v1/responses", "Chat Completions", "effort 'none'", "Responses-API function tool"):
        assert needle in joined2
