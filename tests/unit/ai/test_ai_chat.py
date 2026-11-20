"""Natural-language chat client (services.ai.chat): full fake-transport conversations, no network.

The tool side is the REAL ``HttpToolBackend`` talking to a fake local API (MockTransport) whose live catalog adds
catalog-only session/activity tools to the shared registry; the model side is the scripted ChatScript behind
either the Sign-in-with-ChatGPT SSE fake or the OpenAI Responses fake used by PydanticAI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from packages.contracts.enums import AiRuntimeMode, ModelRunStatus, ModelTransport
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.secrets import SecretStore
from services.ai.chat import (
    ApiKeyChatModel,
    ChatConfig,
    ChatSession,
    Console,
    build_session,
    clean_terminal,
    mode_explanation,
    parse_catalog,
    run_chat,
)
from services.ai.chat_fake import ChatScript, FakeChatOpenAIHandler, build_args, duration_minutes
from services.ai.chatgpt_fake import ChatGptScenario, FakeChatGptServer
from services.ai.chatgpt_oauth import USAGE_URL, ChatGptOAuthConfig
from services.ai.chatgpt_responses import UNSUPPORTED_FIELDS
from services.ai.runtime_modes import AiRuntime
from services.ai.token_store import MemoryTokenStore
from services.tools.backends import HttpToolBackend
from services.tools.registry import MUTATING_TOOL_NAMES

from .chat_helpers import (
    AGENT_TOKEN,
    BASE_TOOLS,
    GET_ACTIVITY,
    NO_TRADE_REASON,
    START_SESSION,
    FakeAgentApi,
    ScriptedConsole,
    write_agent_token,
)
from .conftest import T0

pytestmark = pytest.mark.timeout(60)
START = START_SESSION["name"]  # catalog-only tools (unknown to the local registry)
ACTIVITY = GET_ACTIVITY["name"]


def _settings(root: Path, mode: AiRuntimeMode, transport: Literal["fake", "real"] = "fake") -> Settings:
    return Settings(project_root=root, ai_transport=transport, use_keychain=False, ai_runtime=mode, chatgpt_oauth_port=0)


def _secrets(root: Path) -> SecretStore:
    return SecretStore(root, use_keychain=False, env={}, dotenv_path=root / "no.env")


def _backend(root: Path, api: FakeAgentApi) -> HttpToolBackend:
    return HttpToolBackend("http://127.0.0.1:8765", write_agent_token(root), actor="chat", transport=api.transport())


class PlanRig:
    """Chat on the ChatGPT plan (FAKE): AiRuntime + FakeChatGptServer + fake local API + scripted console."""

    def __init__(self, root: Path, *, answers: list[str] | None = None, api: FakeAgentApi | None = None,
                 scenario: ChatGptScenario = ChatGptScenario.TRADE, config: ChatConfig | None = None) -> None:
        self.clock = ManualClock(T0)
        self.settings = _settings(root, AiRuntimeMode.CHATGPT_OAUTH)
        oauth = ChatGptOAuthConfig.fake()
        self.server = FakeChatGptServer(oauth, scenario=scenario, now=lambda: self.clock.now().timestamp())
        self.rt = AiRuntime(self.settings, _secrets(root), db=None, clock=self.clock, wall_clock=self.clock,
                            token_store=MemoryTokenStore(), fake_server=self.server, oauth_config=oauth)
        self.api = api or FakeAgentApi()
        self.io = ScriptedConsole(answers)
        self.console = Console(read=self.io.read, write=self.io.write)
        self.session: ChatSession = build_session(self.settings, _secrets(root), self.console,
                                                  backend=_backend(root, self.api), ai_runtime=self.rt,
                                                  config=config, run_log=root / "chat_runs.jsonl")

    async def sign_in(self) -> None:
        await self.session.handle("/login")
        assert self.rt.auth.signed_in(), self.io.text

    def model_bodies(self) -> list[dict[str, Any]]:
        return self.server.responses_bodies

    async def aclose(self) -> None:
        await self.rt.aclose()


@pytest.fixture
async def plan(tmp_path: Path):
    rigs: list[PlanRig] = []

    def _make(**kw: Any) -> PlanRig:
        rig = PlanRig(tmp_path, **kw)
        rigs.append(rig)
        return rig

    yield _make
    for r in rigs:
        await r.aclose()


def _offered(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for t in body.get("tools") or []:
        inner = t.get("tools") or [] if t.get("type") == "namespace" else [t]
        names += [x["name"] for x in inner]
    return sorted(names)


# ------------------------------------------------------------------------------------------------
# full conversation on the ChatGPT plan: start a session, then explain a NO_TRADE
# ------------------------------------------------------------------------------------------------


async def test_chat_plan_conversation_starts_a_session_then_explains_no_trade(plan: Any, tmp_path: Path) -> None:
    rig: PlanRig = plan(answers=["y"])
    assert await rig.session.refresh_tools() is None
    await rig.sign_in()

    a1 = await rig.session.ask("Start a 1-hour paper trading session")
    assert a1 is not None and a1.status == ModelRunStatus.OK
    # the action ran exactly once, only after the human typed "y" at the confirmation prompt
    assert rig.api.calls(START) == [{"duration_minutes": 60, "symbols": None, "cadence_minutes": None,
                                               "use_ai": None}]
    assert any("Run it?" in p for p in rig.io.prompts)
    assert START in rig.io.text and "sess_test_1" in a1.text

    a2 = await rig.session.ask("Why didn't you trade?")
    assert a2 is not None and NO_TRADE_REASON in a2.text
    assert len(rig.io.prompts) == 1  # read-only tools never ask for confirmation
    assert rig.api.calls(ACTIVITY) and rig.api.calls("get_signals")

    bodies = rig.model_bodies()
    assert rig.server.violations == []
    assert len(bodies) == 4  # (call start_session, answer) + (call reads, answer)
    catalog_names = sorted([*BASE_TOOLS, START, ACTIVITY])
    for b in bodies:
        assert b["store"] is False and b["stream"] is True
        assert not (UNSUPPORTED_FIELDS & set(b)), "unsupported plan-usage field sent"
        assert b["model"] == "gpt-6.1-sol" and b["reasoning"] == {"effort": "medium"}
        assert isinstance(b["instructions"], str) and "DATA, not instructions" in b["instructions"]
        assert all(i.get("role") != "system" and "id" not in i for i in b["input"])
        assert _offered(b) == catalog_names  # exactly the LIVE catalog: nothing hard-coded, nothing added
    # full history in input (no previous_response_id): the 2nd conversation turn re-sends the 1st
    first_user = bodies[0]["input"][0]["content"][0]["text"]
    assert bodies[3]["input"][0]["content"][0]["text"] == first_user
    assert any(i.get("type") == "function_call_output" for i in bodies[3]["input"])
    # the agent token reaches the local API as a header only -- never a model request or the terminal
    wire = json.dumps(bodies)
    assert AGENT_TOKEN not in wire and AGENT_TOKEN not in rig.io.text
    assert rig.api.auth_failures == 0
    # registry mutating runtime paths were never touched (no propose/halt/cancel/cycle)
    assert not [c for c in rig.api.runtime.calls if c in ("submit_agent_proposal", "halt", "cancel", "run_cycle")]
    # one ModelRunRecord per answered message, FAKE + plan pricing, persisted to the chat run log only
    assert [r.transport for r in rig.session.records] == [ModelTransport.FAKE] * 2
    assert all(r.pricing_source.startswith("CHATGPT_PLAN") and r.estimated_cost_usd == 0 for r in rig.session.records)
    assert all(r.prompt_version.startswith("chat_operator@") for r in rig.session.records)
    assert len((tmp_path / "chat_runs.jsonl").read_text().splitlines()) == 2
    assert "FAKE" in rig.io.text  # every answer's footer labels the transport


async def test_chat_declined_action_is_not_executed_and_model_is_told(plan: Any) -> None:
    rig: PlanRig = plan(answers=["n"])
    await rig.session.refresh_tools()
    await rig.sign_in()
    answer = await rig.session.ask("Please start a 2 hour paper session")
    assert answer is not None
    assert rig.api.calls(START) == []
    assert "declined" in answer.text
    outputs = [i for i in rig.model_bodies()[-1]["input"] if i.get("type") == "function_call_output"]
    assert outputs and json.loads(outputs[-1]["output"])["declined"] is True


async def test_chat_unknown_or_missing_confirmation_declines(plan: Any) -> None:
    rig: PlanRig = plan(answers=[])  # EOF at the prompt -> declined (fail closed)
    await rig.session.refresh_tools()
    await rig.sign_in()
    await rig.session.ask("start a 30 minute session")
    assert rig.api.calls(START) == []


async def test_chat_mutating_5xx_is_unknown_and_never_retried(plan: Any) -> None:
    rig: PlanRig = plan(answers=["y", "y"], api=FakeAgentApi(start_status=500))
    await rig.session.refresh_tools()
    await rig.sign_in()
    answer = await rig.session.ask("Start a 1-hour paper trading session")
    assert answer is not None
    assert len(rig.api.calls(START)) == 1  # not retried by the client
    out = [json.loads(i["output"]) for i in rig.model_bodies()[-1]["input"] if i.get("type") == "function_call_output"]
    assert "UNKNOWN" in out[-1]["error"] and out[-1]["do_not_retry"] is True


async def test_chat_usage_limit_is_friendly_and_pauses_without_retry_loop(plan: Any) -> None:
    rig: PlanRig = plan(scenario=ChatGptScenario.USAGE_LIMIT_HTTP)
    await rig.session.refresh_tools()
    await rig.sign_in()
    assert await rig.session.ask("Show my performance") is None
    assert USAGE_URL in rig.io.text
    assert "keeps running" in rig.io.text and "Claude Code / Codex" in rig.io.text
    assert len(rig.model_bodies()) == 1
    n_requests = len(rig.server.recorded)
    assert await rig.session.ask("Show my performance") is None  # paused locally: nothing is sent
    assert len(rig.server.recorded) == n_requests
    assert "paused" in rig.io.out[-1]
    assert rig.rt.status()["chatgpt"]["usage_limit_hit"] is True


async def test_chat_refused_request_shape_blocks_only_the_chat_not_research(plan: Any) -> None:
    rig: PlanRig = plan(scenario=ChatGptScenario.UNSUPPORTED)
    await rig.session.refresh_tools()
    await rig.sign_in()
    assert await rig.session.ask("Show my performance") is None
    sent = len(rig.model_bodies())
    assert sent == 1
    assert await rig.session.ask("Show my performance") is None  # not resent
    assert len(rig.model_bodies()) == sent and "/new" in rig.io.out[-1]
    assert rig.rt.researcher.unavailable_reason() is None  # research turns are not blocked by a chat request
    await rig.session.handle("/new")
    rig.server.scenario = ChatGptScenario.TRADE
    assert await rig.session.ask("Show my performance") is not None


async def test_chat_not_signed_in_points_to_login_and_sends_nothing(plan: Any) -> None:
    rig: PlanRig = plan()
    await rig.session.refresh_tools()
    assert await rig.session.ask("Why didn't you trade?") is None
    assert "/login" in rig.io.text and "keeps running" in rig.io.text
    assert rig.model_bodies() == []


async def test_chat_identity_only_sign_in_is_never_used_for_inference(tmp_path: Path) -> None:
    rig = PlanRig(tmp_path)
    rig.server.grant_plan_usage = False
    try:
        await rig.session.refresh_tools()
        await rig.session.handle("/login")
        assert await rig.session.ask("Show my performance") is None
        assert "plan usage" in rig.io.text and rig.model_bodies() == []
    finally:
        await rig.aclose()


async def test_chat_action_limit_per_message(plan: Any) -> None:
    rig: PlanRig = plan(answers=["y"], config=ChatConfig(max_actions_per_message=0))
    await rig.session.refresh_tools()
    await rig.sign_in()
    await rig.session.ask("Start a 1-hour paper trading session")
    assert rig.api.calls(START) == [] and rig.io.prompts == []


async def test_chat_without_api_says_how_to_start_it(tmp_path: Path) -> None:
    rig = PlanRig(tmp_path)
    rig.session.caller.backend.token_path = tmp_path / "missing.token"
    rig.session.caller.backend._token = None
    try:
        assert await rig.session.refresh_tools() is not None
        assert await rig.session.ask("Show my performance") is None
        assert "make dev" in rig.io.text and rig.model_bodies() == []
    finally:
        await rig.aclose()


async def test_chat_output_is_terminal_safe(plan: Any) -> None:
    rig: PlanRig = plan()
    await rig.session.refresh_tools()
    await rig.sign_in()

    def evil(body: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"type": "message", "id": "msg_evil", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "ok\x1b[2J\x1b]0;pwned\x07‮ done", "annotations": []}]}]

    rig.server.responder = evil
    answer = await rig.session.ask("hello")
    assert answer is not None
    assert all("\x1b" not in line and "‮" not in line and "\x07" not in line for line in rig.io.out)
    assert clean_terminal("a\x1b[31mb​c") == "a[31mbc"


# ------------------------------------------------------------------------------------------------
# API_KEYS: PydanticAI OpenAIResponsesModel over the OpenAI Responses fake
# ------------------------------------------------------------------------------------------------


async def test_chat_api_keys_conversation_uses_responses_api_with_catalog_tools(tmp_path: Path) -> None:
    settings = _settings(tmp_path, AiRuntimeMode.API_KEYS)
    handler = FakeChatOpenAIHandler(expected_model=settings.sol_model, script=ChatScript())
    io = ScriptedConsole(["y"])
    console = Console(read=io.read, write=io.write)
    api = FakeAgentApi()
    model = ApiKeyChatModel(settings, _secrets(tmp_path), config=ChatConfig(), fake_handler=handler)
    session = build_session(settings, _secrets(tmp_path), console, backend=_backend(tmp_path, api), api_model=model,
                            run_log=False)
    await session.refresh_tools()
    a1 = await session.ask("Start a 1-hour paper trading session")
    a2 = await session.ask("Why didn't you trade?")
    assert a1 is not None and "sess_test_1" in a1.text
    assert a2 is not None and NO_TRADE_REASON in a2.text
    assert len(api.calls(START)) == 1 and len(io.prompts) == 1
    assert handler.violations == []
    catalog_names = sorted([*BASE_TOOLS, START, ACTIVITY])
    for ex in handler.exchanges:
        assert ex.method == "POST" and ex.path.endswith("/v1/responses")  # Responses API, never Chat Completions
        assert "messages" not in ex.body and ex.body["model"] == "gpt-6.1-sol"
        assert ex.body.get("reasoning", {}).get("effort") == "medium"
        assert ex.body.get("store") is False
        assert sorted(t["name"] for t in ex.body["tools"]) == catalog_names
        assert all(t["type"] == "function" for t in ex.body["tools"])
        assert AGENT_TOKEN not in json.dumps(ex.body)
    assert [r.transport for r in session.records] == [ModelTransport.FAKE] * 2
    assert all(r.pricing_source.startswith("FAKE_TRANSPORT_NOT_BILLED") for r in session.records)
    assert all(r.input_tokens > 0 and r.reasoning_tokens for r in session.records)


async def test_chat_api_keys_real_transport_without_key_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path, AiRuntimeMode.API_KEYS, transport="real")
    io = ScriptedConsole()
    console = Console(read=io.read, write=io.write)
    model = ApiKeyChatModel(settings, _secrets(tmp_path), config=ChatConfig())
    assert model.error is not None and "OPENAI_API_KEY" in model.error
    session = build_session(settings, _secrets(tmp_path), console, backend=_backend(tmp_path, FakeAgentApi()),
                            api_model=model, run_log=False)
    await session.refresh_tools()
    assert await session.ask("Show my performance") is None
    assert "unavailable" in io.text and "keeps running" in io.text


# ------------------------------------------------------------------------------------------------
# modes where the app calls no model
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "needle"), [
    (AiRuntimeMode.CLAUDE_CODE, "talk to Claude Code directly"),
    (AiRuntimeMode.CODEX, "talk to Codex directly"),
    (AiRuntimeMode.NONE, "quant only"),
])
async def test_chat_explains_modes_without_an_in_app_model(tmp_path: Path, mode: AiRuntimeMode, needle: str) -> None:
    io = ScriptedConsole()
    console = Console(read=io.read, write=io.write)
    code = await run_chat(_settings(tmp_path, mode), _secrets(tmp_path), console, commands=["Start a session"])
    assert code == 0
    assert needle in io.text and "local-ai-trader" in io.text
    assert io.prompts == []
    assert mode_explanation(AiRuntimeMode.CHATGPT_OAUTH) is None and mode_explanation(AiRuntimeMode.API_KEYS) is None


async def test_chat_run_chat_fake_plan_end_to_end_with_commands(tmp_path: Path) -> None:
    settings = _settings(tmp_path, AiRuntimeMode.CHATGPT_OAUTH)
    io = ScriptedConsole(["y"])
    console = Console(read=io.read, write=io.write)
    api = FakeAgentApi()
    clock = ManualClock(T0)
    oauth = ChatGptOAuthConfig.fake()
    server = FakeChatGptServer(oauth, now=lambda: clock.now().timestamp())
    rt = AiRuntime(settings, _secrets(tmp_path), clock=clock, wall_clock=clock, token_store=MemoryTokenStore(),
                   fake_server=server, oauth_config=oauth)
    session = build_session(settings, _secrets(tmp_path), console, backend=_backend(tmp_path, api), ai_runtime=rt,
                            run_log=False)
    try:
        code = await run_chat(settings, _secrets(tmp_path), console, session=session,
                              commands=["/tools", "Start a 1-hour paper trading session", "Why didn't you trade?", "/quit"])
    finally:
        await rt.aclose()
    assert code == 0
    assert "FAKE transport" in io.text and "user must confirm eligibility" in io.text  # banner honesty
    assert "simulated ChatGPT account" in io.text  # FAKE sign-in is labelled
    assert "[action]" in io.text and START in io.text
    assert NO_TRADE_REASON in io.text and len(api.calls(START)) == 1
    assert server.violations == []


# ------------------------------------------------------------------------------------------------
# catalog parsing + scripted model helpers
# ------------------------------------------------------------------------------------------------


def test_chat_catalog_parsing_fails_closed() -> None:
    raw: list[Any] = [
        {"name": "get_portfolio", "description": "d", "mutating": False, "input_schema": {"type": "object"}},
        {"name": "propose_paper_trade", "description": "d", "mutating": False,  # registry says mutating: wins
         "input_schema": {"type": "object"}},
        {"name": "start_session", "input_schema": {"type": "object"}},  # flag missing -> treated as an action
        {"name": "bad name!", "mutating": False, "input_schema": {"type": "object"}},
        {"name": "no_schema", "mutating": False},
        {"name": "get_portfolio", "mutating": True, "input_schema": {"type": "object"}},  # duplicate dropped
        "junk",
    ]
    tools = {t.name: t for t in parse_catalog(raw)}
    assert sorted(tools) == ["get_portfolio", "propose_paper_trade", "start_session"]
    assert tools["get_portfolio"].mutating is False
    assert tools["propose_paper_trade"].mutating is True and tools["start_session"].mutating is True
    stricter = parse_catalog([{"name": "get_orders", "mutating": True, "input_schema": {"type": "object"}}])
    assert stricter[0].mutating is True  # a catalog that is stricter than the local registry wins too
    assert "propose_paper_trade" in MUTATING_TOOL_NAMES


def test_chat_fake_script_builds_args_from_the_offered_schema() -> None:
    assert duration_minutes("Start a 1-hour paper trading session") == 60
    assert duration_minutes("run for 90 minutes") == 90
    assert duration_minutes("start an hour long session") == 60
    args = build_args(START_SESSION["input_schema"], "start a 2 hour session on SPY and QQQ")
    assert args == {"duration_minutes": 120, "symbols": ["SPY", "QQQ"], "cadence_minutes": None, "use_ai": None}
