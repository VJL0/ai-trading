"""ChatGPT-plan research turns over the fake streaming Responses API: exact request shape, read-only shared tools,
function-call loop, deterministic conversion, documented error codes, pauses (never a tight retry loop)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from packages.contracts.enums import AiRuntimeMode, DecisionAction, ModelProvider, ModelRunStatus, ModelTransport, Proposer
from services.ai.chatgpt_fake import ChatGptScenario
from services.ai.chatgpt_oauth import PLAN_PRICING_SOURCE, USAGE_URL, ChatGptError
from services.ai.chatgpt_responses import (
    UNSUPPORTED_FIELDS,
    ChatGptConfig,
    assert_supported_body,
    build_request_body,
    error_from_payload,
    parse_draft,
    read_only_function_tools,
    replayable_item,
)
from services.ai.fake_transport import injection_evidence
from services.tools.registry import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, TOOL_REGISTRY

from .conftest import T0, make_signal, make_snapshot

pytestmark = pytest.mark.timeout(60)


def _flat_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in body.get("tools", []):
        out += t["tools"] if t.get("type") == "namespace" else [t]
    return out


async def test_trade_turn_request_shape_and_conversion(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    r = await h.research()
    p = r.proposal
    assert p.action == DecisionAction.TRADE and p.proposer == Proposer.AI_PIPELINE
    assert (p.symbol, p.quantity, p.limit_price) == ("SPY", 1, p.limit_price)
    assert p.stop_price is not None and p.stop_price < p.limit_price  # stop/loss arithmetic is ours
    assert set(p.evidence_ids) <= {e.evidence_id for e in r.evidence} and p.finding_ids == ()
    (run,) = r.model_runs
    assert run.provider == ModelProvider.OPENAI and run.transport == ModelTransport.FAKE
    assert run.status == ModelRunStatus.OK and run.model_id == "gpt-6.1-sol"
    assert run.estimated_cost_usd == 0 and run.pricing_source == PLAN_PRICING_SOURCE
    assert run.input_tokens > 0 and run.output_tokens > 0 and (run.reasoning_tokens or 0) > 0
    assert run.prompt_version.startswith("chatgpt_research@") and "chatgpt-sse" in run.adapter_version
    assert p.model_run_ids == (run.run_id,)
    bodies = h.server.responses_bodies
    assert len(bodies) == 2  # tools turn + final answer
    for b in bodies:
        assert b["store"] is False and b["stream"] is True
        assert not (UNSUPPORTED_FIELDS & set(b)), set(b)
        assert b["instructions"] and b["model"] == "gpt-6.1-sol"
        assert all(i.get("role") != "system" and "id" not in i for i in b["input"])
        assert b["text"]["format"]["type"] == "json_schema" and b["text"]["format"]["strict"] is True
        assert b["reasoning"] == {"effort": "medium"}
        assert b["tools"][0]["type"] == "namespace"
    # the second request carries the full history (no previous_response_id): calls + outputs
    types = [i["type"] for i in bodies[1]["input"]]
    assert types.count("function_call") == types.count("function_call_output") == 4
    assert "reasoning" in types
    assert h.server.violations == []
    assert h.mutating_calls() == []


async def test_only_read_only_shared_tools_are_offered(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(scenario=ChatGptScenario.MUTATING_TOOL)
    await h.sign_in()
    r = await h.research()
    offered = {t["name"] for b in h.server.responses_bodies for t in _flat_tools(b)}
    assert offered == set(READ_ONLY_TOOL_NAMES)
    assert not offered & set(MUTATING_TOOL_NAMES)
    assert all(not TOOL_REGISTRY[n].mutating for n in offered)
    for t in _flat_tools(h.server.responses_bodies[0]):  # unchanged from the shared registry's OpenAI export
        spec = TOOL_REGISTRY[t["name"]]
        assert t["parameters"] == spec.input_schema() and t["strict"] is True and t["type"] == "function"
    # the model tried propose_paper_trade + request_halt: refused, never executed, TRADE downgraded
    assert h.mutating_calls() == []
    outputs = [i for i in h.server.responses_bodies[1]["input"] if i["type"] == "function_call_output"]
    refused = [json.loads(o["output"]) for o in outputs if "not available" in o["output"]]
    assert len(refused) == 2
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert "propose_paper_trade" in (r.proposal.no_trade_reason or "")


def test_read_only_function_tools_export() -> None:
    tools = read_only_function_tools()
    assert [t["name"] for t in tools] == list(READ_ONLY_TOOL_NAMES)
    assert all(t["type"] == "function" for t in tools)


async def test_top_level_layout_and_no_backend(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(config=ChatGptConfig(tool_layout="top_level", structured_output=False))
    await h.sign_in()
    r = await h.research()
    b = h.server.responses_bodies[0]
    assert all(t["type"] == "function" for t in b["tools"]) and "text" not in b
    assert r.proposal.action == DecisionAction.TRADE
    h2 = chatgpt_factory()
    await h2.sign_in()
    r2 = await h2.research(backend=None)  # no tool backend: evidence pack only, no tools offered
    assert "tools" not in h2.server.responses_bodies[0] and len(h2.server.responses_bodies) == 1
    assert r2.proposal.action == DecisionAction.TRADE


async def test_model_selection_from_models_list(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(models=[{"slug": "gpt-hidden", "visibility": "hide"},
                                {"slug": "gpt-6.1-mini", "visibility": "list"}])
    await h.sign_in()
    r = await h.research()
    assert h.server.responses_bodies[0]["model"] == "gpt-6.1-mini" and r.model_runs[0].model_id == "gpt-6.1-mini"
    assert h.rt.status()["chatgpt"]["model"] == "gpt-6.1-mini"
    h2 = chatgpt_factory(models=[{"slug": "gpt-hidden", "visibility": "hide"}])
    await h2.sign_in()
    r2 = await h2.research()
    assert r2.proposal.action == DecisionAction.NO_TRADE and h2.server.responses_bodies == []
    assert r2.model_runs[0].status == ModelRunStatus.ERROR


@pytest.mark.parametrize(("scenario", "status", "needle"), [
    (ChatGptScenario.NO_TRADE, ModelRunStatus.OK, "too thin"),
    (ChatGptScenario.SCHEMA_INVALID, ModelRunStatus.SCHEMA_INVALID, "schema"),
    (ChatGptScenario.INJECTION_ECHO, ModelRunStatus.OK, "injection markers"),
    (ChatGptScenario.INCOMPLETE, ModelRunStatus.ERROR, "INCOMPLETE"),
    (ChatGptScenario.TRUNCATED, ModelRunStatus.ERROR, "without response.completed"),
    (ChatGptScenario.TIMEOUT, ModelRunStatus.TIMEOUT, "timed out"),
    (ChatGptScenario.LOOP, ModelRunStatus.ERROR, "tool-call budget exceeded"),
])
async def test_failure_paths_return_no_trade(chatgpt_factory: Any, scenario: ChatGptScenario, status: ModelRunStatus,
                                             needle: str) -> None:
    h = chatgpt_factory(scenario=scenario,
                        config=ChatGptConfig(max_total_tool_calls=3) if scenario == ChatGptScenario.LOOP else None)
    await h.sign_in()
    r = await h.research()
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert needle in (r.proposal.no_trade_reason or ""), r.proposal.no_trade_reason
    (run,) = r.model_runs
    assert run.status == status and run.transport == ModelTransport.FAKE
    assert run.provider == ModelProvider.OPENAI and run.pricing_source == PLAN_PRICING_SOURCE
    assert h.mutating_calls() == []


async def test_turn_limit(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(scenario=ChatGptScenario.LOOP, config=ChatGptConfig(max_turns=3))
    await h.sign_in()
    r = await h.research()
    assert len(h.server.responses_bodies) == 3 and "within 3 requests" in (r.proposal.no_trade_reason or "")


@pytest.mark.parametrize(("scenario", "code", "kind"), [
    (ChatGptScenario.USAGE_LIMIT_HTTP, "subscription_sharing_usage_limit_exceeded", "USAGE_LIMIT"),
    (ChatGptScenario.USAGE_LIMIT_STREAM, "subscription_sharing_usage_limit_exceeded", "USAGE_LIMIT"),
    (ChatGptScenario.NOT_ELIGIBLE, "subscription_sharing_user_not_eligible", "NOT_ELIGIBLE"),
    (ChatGptScenario.USAGE_UNAVAILABLE, "subscription_sharing_usage_unavailable", "USAGE_UNAVAILABLE"),
    (ChatGptScenario.UNSUPPORTED, "subscription_sharing_unsupported_capability", "UNSUPPORTED_CAPABILITY"),
    (ChatGptScenario.ROUTE_NOT_SUPPORTED, "subscription_sharing_route_not_supported", "ROUTE_NOT_SUPPORTED"),
    (ChatGptScenario.INVALID_USER, "subscription_sharing_invalid_user", "INVALID_USER"),
])
async def test_documented_error_codes_map_to_clear_statuses(chatgpt_factory: Any, scenario: ChatGptScenario,
                                                            code: str, kind: str) -> None:
    h = chatgpt_factory(scenario=scenario)
    await h.sign_in()
    r = await h.research()
    assert r.proposal.action == DecisionAction.NO_TRADE
    (run,) = r.model_runs
    assert run.status == ModelRunStatus.ERROR and run.pricing_source == PLAN_PRICING_SOURCE
    assert code in (run.error or "") and kind in (run.error or "")
    st = h.rt.status()["chatgpt"]
    assert kind in (st["last_error"] or "")
    assert st["usage_limit_hit"] is (kind == "USAGE_LIMIT")
    if kind == "USAGE_LIMIT":
        assert USAGE_URL in (run.error or "") and "not inferred" in (run.error or "")


async def test_usage_limit_pauses_without_tight_retry(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory(scenario=ChatGptScenario.USAGE_LIMIT_HTTP)
    await h.sign_in()
    await h.research()
    sent = len(h.server.responses_bodies)
    assert sent == 1  # one attempt, no automatic retry
    again = await h.research(causal_id="cz_2")
    assert len(h.server.responses_bodies) == sent  # paused: nothing sent
    assert again.model_runs[0].status == ModelRunStatus.SKIPPED
    assert "paused" in (again.proposal.no_trade_reason or "") and USAGE_URL in (again.proposal.no_trade_reason or "")
    st = h.rt.status()["chatgpt"]
    assert st["usage_limit_hit"] and st["paused_until"] and "not the plan's reset time" in st["pause_reason"]
    clock.advance(seconds=901)
    h.server.scenario = ChatGptScenario.TRADE
    ok = await h.research(causal_id="cz_3")
    assert ok.proposal.action == DecisionAction.TRADE and h.rt.status()["chatgpt"]["usage_limit_hit"] is False


async def test_eligibility_block_until_new_sign_in(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(scenario=ChatGptScenario.NOT_ELIGIBLE)
    await h.sign_in()
    await h.research()
    await h.research(causal_id="cz_2")
    assert len(h.server.responses_bodies) == 1  # blocked: the same request is never resent
    h.server.scenario = ChatGptScenario.TRADE
    await h.sign_in()  # a new sign-in may change the account
    assert (await h.research(causal_id="cz_3")).proposal.action == DecisionAction.TRADE


async def test_unavailable_backs_off_and_401_refreshes_once(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory(scenario=ChatGptScenario.USAGE_UNAVAILABLE)
    await h.sign_in()
    await h.research()
    await h.research(causal_id="cz_2")
    assert len(h.server.responses_bodies) == 1
    clock.advance(seconds=61)
    h.server.scenario = ChatGptScenario.INVALID_USER
    await h.research(causal_id="cz_3")
    token_calls = len([r for r in h.server.recorded if r.path.endswith("/oauth/token")])
    h.server.scenario = ChatGptScenario.TRADE
    ok = await h.research(causal_id="cz_4")
    assert ok.proposal.action == DecisionAction.TRADE
    # the 401 dropped the in-memory access token, so exactly one refresh happened before the next request
    assert len([r for r in h.server.recorded if r.path.endswith("/oauth/token")]) == token_calls + 1


async def test_preconditions_and_injection_never_call_the_model(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    stale = make_snapshot(at=T0 - timedelta(minutes=30))
    r1 = await h.research(snapshot=stale)
    r2 = await h.research(signal=make_signal(is_candidate=False), causal_id="cz_2")
    r3 = await h.research(extra_evidence=injection_evidence(T0 - timedelta(minutes=1)), causal_id="cz_3")
    r4 = await h.research(symbol="QQQ", causal_id="cz_4")
    assert all(r.proposal.action == DecisionAction.NO_TRADE for r in (r1, r2, r3, r4))
    assert "stale" in (r1.proposal.no_trade_reason or "")
    assert "prompt injection" in (r3.proposal.no_trade_reason or "")
    assert r3.model_runs[0].status == ModelRunStatus.INJECTION_BLOCKED
    assert "not the selected candidate" in (r4.proposal.no_trade_reason or "")
    assert h.server.responses_bodies == []


async def test_wrong_mode_never_calls_the_model(chatgpt_factory: Any) -> None:
    for mode in (AiRuntimeMode.NONE, AiRuntimeMode.API_KEYS, AiRuntimeMode.CLAUDE_CODE, AiRuntimeMode.CODEX):
        h = chatgpt_factory(mode=mode)
        await h.sign_in()
        r = await h.research()
        assert r.proposal.action == DecisionAction.NO_TRADE and "CHATGPT_OAUTH" in (r.proposal.no_trade_reason or "")
        assert h.server.responses_bodies == [] and r.model_runs == []


async def test_not_signed_in_is_a_recorded_error(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    r = await h.research()
    assert r.proposal.action == DecisionAction.NO_TRADE and "NOT_SIGNED_IN" in (r.proposal.no_trade_reason or "")
    assert r.model_runs[0].status == ModelRunStatus.ERROR and h.server.responses_bodies == []


async def test_no_tokens_or_secrets_reach_the_model(chatgpt_factory: Any, tmp_path: Any) -> None:
    keys = tmp_path / "var" / "paper" / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    planted = {"operator.token": "op-SECRET-1", "engine.token": "eng-SECRET-2", "agent.token": "agent-SECRET-3"}
    for name, value in planted.items():
        (keys / name).write_text(value)
    h = chatgpt_factory()
    await h.sign_in()
    await h.research()
    grant = next(iter(h.server.grants.values()))
    blob = json.dumps(h.server.responses_bodies)
    for needle in [*planted.values(), grant.refresh_token, grant.access_token, "Bearer", "oaiapp_"]:
        assert needle not in blob, needle


async def test_gate_limits_repeat_turns_per_symbol(chatgpt_factory: Any) -> None:
    from services.ai.config import AiConfig

    h = chatgpt_factory(ai_config=AiConfig(min_interval_per_symbol_s=900, min_wall_interval_per_symbol_s=30))
    await h.sign_in()
    assert (await h.research()).proposal.action == DecisionAction.TRADE
    r = await h.research(causal_id="cz_2")
    assert "gated" in (r.proposal.no_trade_reason or "") and len(h.server.responses_bodies) == 2


async def test_repeatable_content(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    seen = []
    for i in range(5):
        p = (await h.research(causal_id=f"cz_{i}")).proposal
        seen.append((p.action, p.symbol, p.side, p.quantity, p.limit_price, p.stop_price, p.max_planned_loss,
                     len(p.evidence_ids), p.rationale))
    assert len(set(seen)) == 1


def test_body_guards() -> None:
    cfg = ChatGptConfig()
    body = build_request_body(model="m", instructions="i", input_items=[], tools=[], effort="medium", config=cfg)
    for field in sorted(UNSUPPORTED_FIELDS):
        with pytest.raises(ChatGptError):
            assert_supported_body({**body, field: 1})
    with pytest.raises(ChatGptError):
        assert_supported_body({**body, "store": True})
    with pytest.raises(ChatGptError):
        assert_supported_body({**body, "input": [{"type": "message", "role": "system", "content": "x"}]})
    with pytest.raises(ChatGptError):
        assert_supported_body({**body, "input": [{"type": "function_call", "id": "fc_1"}]})
    with pytest.raises(ChatGptError):
        assert_supported_body({**body, "tools": [{"type": "web_search"}]})
    assert replayable_item({"type": "function_call", "id": "fc", "call_id": "c", "name": "n", "arguments": "{}",
                            "status": "completed"}) == {"type": "function_call", "call_id": "c", "name": "n",
                                                        "arguments": "{}"}
    assert replayable_item({"type": "reasoning", "id": "rs", "summary": []}) is None  # no encrypted content


def test_error_payload_mapping() -> None:
    e = error_from_payload(400, {"error": {"code": "subscription_sharing_unsupported_capability", "param": "tools[0]"}})
    assert e.kind == "UNSUPPORTED_CAPABILITY" and "tools[0]" in str(e)
    assert error_from_payload(403, {"detail": "region"}).kind == "ADMISSION_DENIED"
    assert error_from_payload(401, {"detail": "x"}).kind == "INVALID_USER"
    assert error_from_payload(503, {}).kind == "TRANSIENT"
    assert error_from_payload(429, {}).kind == "RATE_LIMITED"


def test_parse_draft() -> None:
    good = {"recommendation": "NO_TRADE", "symbol": None, "side": None, "quantity": None, "limit_price": None,
            "stop_price": None, "citations": [], "evidence_ids": [], "thesis": "t", "counterarguments": [],
            "risks": [], "confidence_label": "LOW", "no_trade_reason": "r"}
    assert parse_draft(json.dumps(good)).recommendation == "NO_TRADE"
    assert parse_draft("```json\n" + json.dumps(good) + "\n```").no_trade_reason == "r"
    with pytest.raises(ValueError):
        parse_draft("no json here")
    with pytest.raises(ValueError):
        parse_draft(json.dumps({**good, "max_order_notional": "1000000"}))  # extra fields are forbidden
