"""Natural-language chat client for the modes in which the app itself talks to a model.

Run: ``uv run --frozen python -m services.ai.chat`` (``make chat``). The owner types plain requests ("Start a
1-hour paper trading session", "Why didn't you trade?", "Show my performance"); GPT-6.1 Sol answers by calling
the app's tools through the ONE validated tool interface of the running local API.

* Tools: the LIVE catalog from ``GET /api/v1/agent/tools`` (``services.tools.backends.HttpToolBackend``, agent token
  read from ``var/<ns>/keys/agent.token`` and sent only as a header -- it never appears in a prompt). Nothing is
  hard-coded: tools added to the API later (e.g. session tools) are offered automatically.
* Actions: every tool whose catalog entry (or local registry entry) is ``mutating`` -- or whose flag is missing --
  runs only after the human types ``y`` at the confirmation prompt. The model cannot bypass this; a declined action
  is reported to the model as declined. Mutating calls are never retried; an UNKNOWN outcome is surfaced.
* Models:
  - ``CHATGPT_OAUTH``: Sign in with ChatGPT plan usage (``services.ai.chatgpt_oauth``) and the streaming
    Responses client of ``services.ai.chatgpt_responses`` (``store:false``, ``stream:true``, ``instructions``, full
    history in ``input``, function tools in a namespace, no unsupported fields). The usage-limit/backoff/block
    state is shared with the research path (one pause per ChatGPT session; never a tight retry loop).
  - ``API_KEYS``: PydanticAI ``OpenAIResponsesModel`` for ``gpt-6.1-sol`` (Responses API, ``reasoning.effort``;
    never Chat Completions) built by ``services.ai.adapters.build_sol_model``; real calls need
    ``LAT_AI_TRANSPORT=real`` and the OpenAI key; a per-chat cost cap applies.
  - ``CLAUDE_CODE`` / ``CODEX``: the app calls no model; the owner talks to Claude Code / Codex directly (MCP
    server ``local-ai-trader``). ``NONE``: quant only. The client explains this and exits.
* With ``LAT_AI_TRANSPORT=fake`` every answer comes from the scripted :mod:`services.ai.chat_fake` stand-in and
  is labelled FAKE.

The trading service never depends on this client: an AI disconnect, a usage limit or a closed chat leaves
running sessions, risk checks and reconciliation untouched. This process never writes the SQLite ledger (the API
is the single writer); it appends one ``ModelRunRecord`` per answered message to ``var/<ns>/logs/
chat_model_runs.jsonl`` (no prompt or tool content, no secrets).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Protocol, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from packages.contracts import ModelRunRecord
from packages.contracts.enums import AiRuntimeMode, ModelProvider, ModelRunStatus, ModelTransport
from packages.core.canonical import canonical_json, sha256_hex
from packages.core.clock import Clock, SystemClock
from packages.core.config import Settings, load_settings
from packages.core.ids import new_causal_id, new_id
from packages.core.logging import JsonFormatter, log_event, redact
from packages.core.secrets import SecretStore

from .accounting import RequestTokens, end_capture, estimate_cost, start_capture, total_tokens, usage_from_messages
from .chatgpt_oauth import ELIGIBILITY_NOTE, PLAN_PRICING_SOURCE, USAGE_URL, ChatGptError
from .chatgpt_responses import (
    CHATGPT_ADAPTER_VERSION,
    ChatGptConfig,
    assert_supported_body,
    replayable_item,
    resolve_tool_name,
    tools_payload,
)
from .versions import ADAPTER_VERSION, load_prompt, prompt_version

log = logging.getLogger("lat.ai.chat")

CHAT_PROMPT: Final = "chat_operator"
MCP_SERVER_NAME: Final = "local-ai-trader"
SERVICE_KEEPS_RUNNING: Final = (
    "The trading service keeps running on its own: sessions, risk checks and reconciliation are unaffected."
)
ALTERNATIVES: Final = f"Retry later, or talk to Claude Code / Codex through the MCP server '{MCP_SERVER_NAME}'."
CHAT_NAMESPACE_DESCRIPTION: Final = (
    "Tools of the local PAPER-trading app: reads, plus paper-trading actions that run only after the owner confirms."
)
_TOOL_NAME_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
# C0/C1 controls (incl. ESC: no terminal escape sequences from model or tool text), bidi and zero-width chars.
_TERMINAL_UNSAFE: Final = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁠-⁤⁦-⁩﻿]")


class ChatConfig(BaseSettings):
    """Chat tunables (env prefix ``LAT_CHAT_``)."""

    model_config = SettingsConfigDict(env_prefix="LAT_CHAT_", env_file=None, extra="ignore", frozen=True)

    max_requests_per_message: int = Field(default=8, ge=1, le=30, description="model requests per owner message")
    max_tool_calls_per_message: int = Field(default=12, ge=0, le=60)
    max_actions_per_message: int = Field(default=3, ge=0, le=10, description="confirmed mutating calls per message")
    max_tool_output_chars: int = Field(default=40_000, ge=1000)
    max_history_turns: int = Field(default=12, ge=1, le=100)
    api_max_tokens: int = Field(default=4000, gt=0, le=128000, description="API_KEYS path max output tokens")
    max_session_cost_usd: Decimal = Field(default=Decimal("2.00"), ge=0, description="API_KEYS cost cap per chat")
    confirm_timeout_s: float = Field(default=300.0, gt=0, description="unanswered confirmation = declined")
    login_timeout_s: float = Field(default=300.0, gt=0)


def clean_terminal(text: str) -> str:
    """Model and tool text is untrusted: strip control/escape/bidi/zero-width characters before printing."""
    return _TERMINAL_UNSAFE.sub("", text)


def compact(obj: Any, limit: int = 600) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ------------------------------------------------------------------------------------------------
# console
# ------------------------------------------------------------------------------------------------

ReadFn = Callable[[str], Awaitable[str | None]]
WriteFn = Callable[[str], None]


async def _stdin_read(prompt: str) -> str | None:
    try:
        return await asyncio.to_thread(input, prompt)
    except EOFError:
        return None


def _stdout_write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


class Console:
    def __init__(self, *, read: ReadFn | None = None, write: WriteFn | None = None) -> None:
        self._read = read or _stdin_read
        self._write = write or _stdout_write

    def say(self, text: str) -> None:
        self._write(clean_terminal(text))

    async def ask(self, prompt: str) -> str | None:
        return await self._read(clean_terminal(prompt))


# ------------------------------------------------------------------------------------------------
# the live tool catalog
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogTool:
    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool
    category: str | None = None

    def function_tool(self) -> dict[str, Any]:
        """Responses API function tool (the catalog's canonical strict schema, unchanged)."""
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": self.parameters, "strict": True}


def parse_catalog(raw: Sequence[Any]) -> list[CatalogTool]:
    """Validate the API's catalog. A tool counts as read-only only if the catalog says ``mutating: false`` AND the
    local registry (if it knows the tool) agrees; anything malformed is dropped (fail closed)."""
    from services.tools.registry import get_spec

    out: list[CatalogTool] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        e = cast(dict[str, Any], entry)
        name = e.get("name")
        params = e.get("input_schema")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name) or name in seen:
            continue
        if not isinstance(params, dict) or cast(dict[str, Any], params).get("type") != "object":
            continue
        spec = get_spec(name)
        mutating = e.get("mutating") is not False or e.get("read_only") is False or bool(spec and spec.mutating)
        desc = e.get("description")
        category = e.get("category")
        out.append(CatalogTool(name, (desc if isinstance(desc, str) else name)[:2000], cast(dict[str, Any], params),
                               mutating, category if isinstance(category, str) else None))
        seen.add(name)
    return out


class ToolCaller:
    """Calls tools of the running API through a ``services.tools`` backend.

    Tools the local registry knows go through ``backend.call`` (output re-validated against the shared contract).
    A tool that only the live catalog knows (added to the API after this client's code was loaded) is POSTed to the
    same endpoint with the same auth by the HTTP backend, without local output re-validation (the API validated
    it); ``HttpToolBackend.call`` would execute it and then discard the result, which is unsafe for an action.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    async def catalog(self) -> list[CatalogTool]:
        return parse_catalog(await self.backend.catalog())

    async def call(self, name: str, args: Mapping[str, Any], *, mutating: bool) -> dict[str, Any]:
        from services.tools.backends import HttpToolBackend
        from services.tools.registry import get_spec

        if get_spec(name) is not None or not isinstance(self.backend, HttpToolBackend):
            return await self.backend.call(name, args)
        return await _catalog_only_call(self.backend, name, args, mutating=mutating)


async def _catalog_only_call(backend: Any, name: str, args: Mapping[str, Any], *, mutating: bool) -> dict[str, Any]:
    from services.tools import backends as b
    from services.tools.handlers import ToolError

    try:
        body = json.dumps(dict(args), default=b._json_default, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ToolError(422, str(exc)) from None
    resp = await backend._request("POST", f"/api/v1/agent/tools/{name}", body, mutating=mutating)
    payload = b._decode(resp)
    if resp.status_code >= 400:
        err = b._error_from(resp.status_code, payload, backend.base_url)
        if mutating and resp.status_code >= 500:
            raise ToolError(err.status, f"{err.message()}; the outcome of {name} is UNKNOWN. Check the state before "
                                        "trying again; do not retry blindly.")
        raise err
    if not isinstance(payload, dict):
        raise ToolError(502, f"unexpected response from the local API for {name}")
    return cast(dict[str, Any], payload)


ConfirmFn = Callable[[CatalogTool, Mapping[str, Any]], Awaitable[bool]]


def console_confirm(console: Console, *, timeout_s: float = 300.0) -> ConfirmFn:
    async def _confirm(tool: CatalogTool, args: Mapping[str, Any]) -> bool:
        console.say(f"  Action requested: {tool.name} -- this changes the paper-trading state.")
        console.say(f"  arguments: {compact(dict(args))}")
        try:
            answer = await asyncio.wait_for(console.ask("  Run it? [y/N] "), timeout=timeout_s)
        except TimeoutError:
            console.say("  (no answer: declined)")
            return False
        return (answer or "").strip().lower() in ("y", "yes")

    return _confirm


@dataclass
class ActionRecord:
    tool: str
    args: dict[str, Any]
    outcome: str  # "ran" | "declined" | "error:<status>" | "refused"
    summary: str = ""


class ToolExecutor:
    """Executes the model's tool calls for one owner message: catalog-only, bounded, actions confirmed."""

    def __init__(self, caller: ToolCaller, tools: Sequence[CatalogTool], confirm: ConfirmFn, config: ChatConfig,
                 console: Console | None = None) -> None:
        self.caller = caller
        self.tools = {t.name: t for t in tools}
        self.confirm = confirm
        self.config = config
        self.console = console
        self.calls: list[str] = []
        self.actions: list[ActionRecord] = []
        self.unknown: list[str] = []
        self._lock = asyncio.Lock()  # one confirmation prompt at a time

    async def execute(self, name: str, args: Any) -> dict[str, Any]:
        from services.tools.handlers import ToolError

        tool = self.tools.get(name)
        if tool is None:
            self.unknown.append(name[:80])
            return {"error": f"tool {name[:64]!r} is not available; use only the tools you were given"}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        a = cast(dict[str, Any], args)
        if len(self.calls) >= self.config.max_tool_calls_per_message:
            return {"error": "tool-call limit for this message reached; answer with what you have"}
        self.calls.append(name)
        if tool.mutating:
            async with self._lock:
                if len([x for x in self.actions if x.outcome != "declined"]) >= self.config.max_actions_per_message:
                    self.actions.append(ActionRecord(name, a, "refused", "action limit per message"))
                    return {"error": "action limit for this message reached; ask the owner to send a new request"}
                if not await self.confirm(tool, a):
                    self.actions.append(ActionRecord(name, a, "declined"))
                    return {"declined": True, "error": "The owner declined this action. Do not call it again unless "
                                                       "the owner asks for it."}
                return await self._run(tool, a)
        try:
            return self._bounded(await self.caller.call(name, a, mutating=False))
        except ToolError as exc:
            return {"error": exc.message(), "status": exc.status}

    async def _run(self, tool: CatalogTool, args: dict[str, Any]) -> dict[str, Any]:
        from services.tools.handlers import ToolError

        try:
            result = await self.caller.call(tool.name, args, mutating=True)
        except ToolError as exc:
            self.actions.append(ActionRecord(tool.name, args, f"error:{exc.status}", exc.message()[:300]))
            if self.console is not None:
                self.console.say(f"  {tool.name} failed ({exc.status}): {exc.message()[:300]}")
            return {"error": exc.message(), "status": exc.status, "do_not_retry": True}
        self.actions.append(ActionRecord(tool.name, args, "ran", compact(result, 300)))
        return self._bounded(result)

    def _bounded(self, result: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(result, default=str)) > self.config.max_tool_output_chars:
            return {"error": f"tool output exceeded {self.config.max_tool_output_chars} characters; request less data"}
        return result


# ------------------------------------------------------------------------------------------------
# model backends
# ------------------------------------------------------------------------------------------------


class ChatModelError(Exception):
    """A model-side failure with an owner-facing message; the trading service is unaffected."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class ChatAnswer:
    text: str
    status: ModelRunStatus
    model_id: str
    transport: ModelTransport
    provider: ModelProvider
    requests: list[RequestTokens] = field(default_factory=list[RequestTokens])
    request_hashes: list[str] = field(default_factory=list[str])
    response_hashes: list[str] = field(default_factory=list[str])
    cost_usd: Decimal = Decimal(0)
    pricing_source: str = ""
    error: str | None = None


class ChatModel(Protocol):
    label: str
    transport: ModelTransport

    def status_lines(self) -> list[str]: ...

    def reset(self) -> None: ...

    async def answer(self, text: str, executor: ToolExecutor) -> ChatAnswer: ...


_REQUEST_SHAPE_KINDS: Final = frozenset({"UNSUPPORTED_CAPABILITY", "BAD_RESPONSE", "INCOMPLETE"})


def friendly_chatgpt_error(exc: ChatGptError) -> str:
    kind = exc.kind
    if kind in ("NOT_SIGNED_IN", "SIGN_IN_REQUIRED"):
        head = "You are not signed in with ChatGPT. Type /login to sign in in your browser."
    elif kind == "PLAN_USAGE_NOT_GRANTED":
        head = ("The ChatGPT sign-in did not grant plan usage (chatgpt.tokens.use.direct), so it is never used for "
                "inference. Type /login and approve plan usage.")
    elif kind == "USAGE_LIMIT":
        head = f"ChatGPT plan usage limit reached; see {USAGE_URL} (the reset time is not inferred)."
    elif kind in ("TRANSIENT", "USAGE_UNAVAILABLE", "STREAM_INTERRUPTED", "RATE_LIMITED", "TIMEOUT"):
        head = f"ChatGPT is temporarily unavailable ({exc.describe()})."
    else:
        head = f"ChatGPT request failed: {exc.describe()}."
    return f"{head} {SERVICE_KEEPS_RUNNING} {ALTERNATIVES}"


class PlanChatModel:
    """GPT-6.1 Sol on the owner's ChatGPT plan (Sign in with ChatGPT), streaming Responses client."""

    provider = ModelProvider.OPENAI

    def __init__(self, ai_runtime: Any, *, config: ChatConfig, chatgpt_config: ChatGptConfig | None = None,
                 effort: str | None = None) -> None:
        self.ai = ai_runtime
        self.config = config
        self.chatgpt_config = chatgpt_config or ai_runtime.chatgpt_config
        self.effort = effort if effort is not None else ai_runtime.settings.sol_effort
        self.history: list[dict[str, Any]] = []
        self._turn_starts: list[int] = []
        self.transport: ModelTransport = ai_runtime.transport
        self.label = "GPT-6.1 Sol on your ChatGPT plan (Sign in with ChatGPT)"
        self.blocked_reason: str | None = None  # chat-local: a refused request shape is not retried

    def reset(self) -> None:
        self.history.clear()
        self._turn_starts.clear()
        self.blocked_reason = None

    def status_lines(self) -> list[str]:
        st = self.ai.status().get("chatgpt", {})
        lines = [f"signed in: {st.get('signed_in')}  plan usage granted: {st.get('plan_usage_granted')}  "
                 f"account: {st.get('email_masked')}  model: {st.get('model')}"]
        if st.get("pause_reason") or st.get("blocked_reason"):
            lines.append(f"paused/blocked: {st.get('pause_reason') or st.get('blocked_reason')}")
        if st.get("last_error"):
            lines.append(f"last error: {st.get('last_error')}")
        return lines

    def _trim(self) -> None:
        while len(self._turn_starts) > self.config.max_history_turns:
            cut = self._turn_starts[1]
            del self.history[:cut]
            self._turn_starts = [i - cut for i in self._turn_starts[1:]]

    async def answer(self, text: str, executor: ToolExecutor) -> ChatAnswer:
        researcher = self.ai.researcher
        auth = self.ai.auth
        unavailable = researcher.unavailable_reason()
        if unavailable:  # paused after a usage limit / backoff, or blocked: no request is sent
            raise ChatModelError("PAUSED", f"{unavailable}. {SERVICE_KEEPS_RUNNING} {ALTERNATIVES}")
        if self.blocked_reason:
            raise ChatModelError("BLOCKED", f"This conversation's requests were refused ({self.blocked_reason}); the "
                                            "same request is not resent. Type /new to start over.")
        if not auth.signed_in():
            raise ChatModelError("NOT_SIGNED_IN", friendly_chatgpt_error(ChatGptError("NOT_SIGNED_IN", "not signed in")))
        out = ChatAnswer("", ModelRunStatus.ERROR, self.ai.settings.chatgpt_model, self.transport, self.provider,
                         pricing_source=PLAN_PRICING_SOURCE)
        self._turn_starts.append(len(self.history))
        self.history.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
        self._trim()
        functions = [t.function_tool() for t in executor.tools.values()]
        tools = tools_payload(self.chatgpt_config, functions, description=CHAT_NAMESPACE_DESCRIPTION)
        instructions = load_prompt(CHAT_PROMPT)
        try:
            for _ in range(self.config.max_requests_per_message):
                token = await auth.access_token(require_plan_usage=True)
                out.model_id = await researcher.select_model(token)
                body = build_chat_body(model=out.model_id, instructions=instructions, history=self.history, tools=tools,
                                       effort=self.effort, config=self.chatgpt_config)
                result = await researcher.client.create(body, token)
                researcher.note_success()
                out.request_hashes.append(result.request_sha256)
                out.response_hashes.append(result.response_sha256)
                if result.usage is not None:
                    out.requests.append(result.usage)
                calls = [i for i in result.output if i.get("type") == "function_call"]
                for item in result.output:
                    replay = replayable_item(item)
                    if replay is not None:
                        self.history.append(replay)
                if not calls:
                    out.text, out.status = result.text.strip(), ModelRunStatus.OK
                    return out
                for call in calls:
                    name = resolve_tool_name(call, self.chatgpt_config.tool_namespace)
                    try:
                        args: Any = json.loads(str(call.get("arguments") or "{}"))
                    except json.JSONDecodeError:
                        args = None
                    payload = await executor.execute(name, args) if args is not None else {"error": "arguments are not JSON"}
                    self.history.append({"type": "function_call_output", "call_id": call.get("call_id"),
                                         "output": json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)})
            out.error = f"no final answer within {self.config.max_requests_per_message} model requests"
            out.text = "I stopped before finishing (step limit for one message). Ask again or narrow the request."
            return out
        except ChatGptError as exc:
            if exc.kind in _REQUEST_SHAPE_KINDS:
                # caused by THIS conversation's request (tools/input), not the account: block the chat only, so
                # the gated research turns (different request) keep working
                self.blocked_reason = exc.describe()
            else:  # account/service level: shared pause/backoff/block state with the research path
                researcher.note_failure(exc)
            log_event(log, "chat plan request failed", logging.WARNING, kind=exc.kind, code=exc.code)
            raise ChatModelError(exc.kind, friendly_chatgpt_error(exc)) from None


def build_chat_body(*, model: str, instructions: str, history: Sequence[Mapping[str, Any]],
                    tools: Sequence[Mapping[str, Any]], effort: str | None, config: ChatGptConfig) -> dict[str, Any]:
    """A plan-usage Responses request for the chat (free-text answer; same wire rules as the research turn)."""
    body: dict[str, Any] = {"model": model, "instructions": instructions, "input": [dict(i) for i in history],
                            "store": False, "stream": True}
    if tools:
        body["tools"] = [dict(t) for t in tools]
        body["tool_choice"] = "auto"
        body["parallel_tool_calls"] = True
    if effort:
        body["reasoning"] = {"effort": effort}
    if config.include_encrypted_reasoning:
        body["include"] = ["reasoning.encrypted_content"]
    assert_supported_body(body)
    return body


class ApiKeyChatModel:
    """GPT-6.1 Sol via API keys: PydanticAI ``OpenAIResponsesModel`` (Responses API) with catalog tools."""

    provider = ModelProvider.OPENAI

    def __init__(self, settings: Settings, secret_store: SecretStore, *, config: ChatConfig,
                 http_client: Any | None = None, fake_handler: Any | None = None) -> None:
        from pydantic_ai import Agent

        from .adapters import AiTransportError, build_sol_model
        from .chat_fake import FakeChatOpenAIHandler
        from .config import AiConfig
        from .fake_transport import fake_http_client

        self.settings = settings
        self.config = config
        self.transport = ModelTransport.REAL if settings.ai_transport == "real" else ModelTransport.FAKE
        self.fake_handler: Any | None = None
        self._owned_http: Any | None = None
        if self.transport == ModelTransport.FAKE and http_client is None:
            handler = fake_handler or FakeChatOpenAIHandler(expected_model=settings.sol_model)
            self.fake_handler = handler
            http_client = fake_http_client(handler)
        elif http_client is None and settings.ai_transport == "real":
            import httpx2

            http_client = self._owned_http = httpx2.AsyncClient(
                timeout=httpx2.Timeout(settings.model_timeout_s, connect=10.0))
        self.error: str | None = None
        self._model: Any | None = None
        try:
            self._model = build_sol_model(settings, secret_store, transport=settings.ai_transport,
                                          http_client=http_client, effort=settings.sol_effort,
                                          config=AiConfig(sol_max_tokens=config.api_max_tokens))
        except AiTransportError as exc:  # e.g. LAT_AI_TRANSPORT=real without OPENAI_API_KEY: no model, no call
            self.error = str(exc)
        self.agent: Any = Agent(None, instructions=load_prompt(CHAT_PROMPT), output_type=str, name="lat-chat",
                                defer_model_check=True)
        self._turns: list[list[Any]] = []
        self.spent = Decimal(0)
        self.label = f"GPT-6.1 Sol via the OpenAI Responses API (API keys, {settings.sol_model})"

    def reset(self) -> None:
        self._turns.clear()

    async def aclose(self) -> None:
        if self._owned_http is not None:
            await self._owned_http.aclose()

    def status_lines(self) -> list[str]:
        lines = [f"model: {self.settings.sol_model}  effort: {self.settings.sol_effort}  "
                 f"estimated spend this chat: ${self.spent} (cap ${self.config.max_session_cost_usd})"]
        if self.error:
            lines.append(f"unavailable: {self.error}")
        return lines

    async def answer(self, text: str, executor: ToolExecutor) -> ChatAnswer:
        from pydantic_ai import Tool
        from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
        from pydantic_ai.toolsets import FunctionToolset
        from pydantic_ai.usage import UsageLimits

        from .pipeline import classify_exception

        if self._model is None:
            raise ChatModelError("TRANSPORT_UNAVAILABLE", f"The API-key model is unavailable: {self.error}. "
                                                          f"{SERVICE_KEEPS_RUNNING} {ALTERNATIVES}")
        if self.spent >= self.config.max_session_cost_usd and self.transport == ModelTransport.REAL:
            raise ChatModelError("BUDGET", f"Estimated spend ${self.spent} reached the chat cap "
                                           f"(LAT_CHAT_MAX_SESSION_COST_USD). {SERVICE_KEEPS_RUNNING}")
        tools: list[Tool[Any]] = []
        for t in executor.tools.values():
            tool = Tool.from_schema(_bound(executor, t.name), name=t.name, description=t.description,
                                    json_schema=t.parameters, takes_ctx=False, sequential=t.mutating)
            tool.strict = True
            tools.append(tool)
        history = [m for turn in self._turns[-self.config.max_history_turns:] for m in turn]
        out = ChatAnswer("", ModelRunStatus.ERROR, self.settings.sol_model, self.transport, self.provider)
        cap, token = start_capture(ModelProvider.OPENAI)
        try:
            result = await self.agent.run(
                text, model=self._model, message_history=history, toolsets=[FunctionToolset(tools)],
                usage_limits=UsageLimits(request_limit=self.config.max_requests_per_message,
                                         tool_calls_limit=self.config.max_tool_calls_per_message),
            )
        except asyncio.CancelledError:
            raise
        except UsageLimitExceeded:
            out.requests = list(cap.wire_usage)
            out.text = "I stopped before finishing (step limit for one message). Ask again or narrow the request."
            out.error = "usage limit (requests/tool calls per message) exceeded"
            self._account(out, cap)
            return out
        except ModelHTTPError as exc:
            out.requests = list(cap.wire_usage)
            self._account(out, cap)
            hint = ("OpenAI rate limit reached" if exc.status_code == 429 else
                    "the OpenAI API key was rejected" if exc.status_code in (401, 403) else
                    f"OpenAI API error {exc.status_code}")
            raise ChatModelError("API_ERROR", f"{hint}. {SERVICE_KEEPS_RUNNING} {ALTERNATIVES}") from None
        except Exception as exc:
            out.requests = list(cap.wire_usage)
            self._account(out, cap)
            status = classify_exception(exc)
            log_event(log, "chat api-key request failed", logging.WARNING, status=status.value,
                      error=redact(type(exc).__name__))
            raise ChatModelError(status.value, f"The model request failed ({status.value}: {type(exc).__name__}). "
                                               f"{SERVICE_KEEPS_RUNNING} {ALTERNATIVES}") from None
        finally:
            end_capture(token)
        new = list(result.new_messages())
        self._turns.append(new)
        out.requests = usage_from_messages(new)
        out.text, out.status = str(result.output).strip(), ModelRunStatus.OK
        self._account(out, cap)
        return out

    def _account(self, out: ChatAnswer, cap: Any) -> None:
        out.request_hashes = list(cap.request_hashes)
        out.response_hashes = list(cap.response_hashes)
        out.cost_usd, out.pricing_source = estimate_cost(self.settings.sol_model, out.requests, self.transport)
        self.spent += out.cost_usd


def _bound(executor: ToolExecutor, name: str) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _call(**kwargs: Any) -> dict[str, Any]:
        return await executor.execute(name, kwargs)

    _call.__name__ = name
    return _call


# ------------------------------------------------------------------------------------------------
# the chat session (REPL logic, transport-independent)
# ------------------------------------------------------------------------------------------------

HELP: Final = """Type a request in plain language, for example:
  Start a 1-hour paper trading session
  Why didn't you trade?
  Show my performance
Commands: /tools  /status  /new (forget the conversation)  /login  /logout  /help  /quit"""


class ChatSession:
    def __init__(self, settings: Settings, model: ChatModel | None, caller: ToolCaller, console: Console, *,
                 config: ChatConfig | None = None, confirm: ConfirmFn | None = None, ai_runtime: Any | None = None,
                 clock: Clock | None = None, run_log: Path | None = None) -> None:
        self.settings = settings
        self.model = model
        self.caller = caller
        self.console = console
        self.config = config or ChatConfig()
        self.confirm = confirm or console_confirm(console, timeout_s=self.config.confirm_timeout_s)
        self.ai_runtime = ai_runtime
        self.clock: Clock = clock or SystemClock()
        self.run_log = run_log
        self.tools: list[CatalogTool] = []
        self.records: list[ModelRunRecord] = []
        self.answers: list[ChatAnswer] = []
        self._carry_note: str | None = None  # actions that ran in a turn whose answer failed

    # ------------------------------------------------------------------ catalog
    async def refresh_tools(self) -> str | None:
        from services.tools.handlers import ToolError

        try:
            tools = await self.caller.catalog()
        except ToolError as exc:
            return f"The local trading API is not reachable or refused the agent token ({exc.status}): {exc.message()}"
        except Exception as exc:  # never crash the REPL
            return f"Could not load the tool catalog ({type(exc).__name__})."
        if not tools:
            return "The local trading API returned an empty tool catalog."
        self.tools = tools
        return None

    # ------------------------------------------------------------------ banner
    def banner(self) -> list[str]:
        mode = self.settings.ai_runtime
        transport = "REAL" if self.settings.ai_transport == "real" else "FAKE"
        lines = [f"local-ai-trader chat - mode {mode.value} - transport {transport}"]
        if self.model is not None:
            lines.append(f"Model: {self.model.label}.")
        if transport == "FAKE":
            lines.append("FAKE transport: answers come from a scripted local stand-in, not a real model "
                         "(set LAT_AI_TRANSPORT=real for real model calls).")
        if mode == AiRuntimeMode.CHATGPT_OAUTH:
            lines.append(f"Plan usage: {ELIGIBILITY_NOTE}.")
            rt = self.ai_runtime
            if rt is not None and rt.transport == ModelTransport.REAL and not rt.auth.signed_in():
                lines.append("Not signed in with ChatGPT yet: type /login.")
        lines.append("PAPER trading only; every order passes the independent risk engine. Actions ask for your "
                     "confirmation first.")
        lines.append(f"Tools from the running app: {len(self.tools)} "
                     f"({sum(t.mutating for t in self.tools)} actions). Type /help for help, /quit to leave.")
        return lines

    # ------------------------------------------------------------------ input handling
    async def handle(self, line: str) -> bool:
        """Process one input line; returns False when the owner wants to quit."""
        text = line.strip()
        if not text:
            return True
        if text.startswith("/"):
            return await self._command(text)
        await self.ask(text)
        return True

    async def _command(self, text: str) -> bool:
        cmd = text.split()[0].lower()
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            self.console.say(HELP)
        elif cmd == "/tools":
            problem = await self.refresh_tools()
            if problem:
                self.console.say(problem)
            for t in self.tools:
                self.console.say(f"  {t.name:<26} {'[action]' if t.mutating else '[read]  '} {t.description[:90]}")
        elif cmd == "/status":
            self.console.say(f"mode {self.settings.ai_runtime.value}, transport {self.settings.ai_transport.upper()}, "
                             f"{len(self.tools)} tools")
            for line in self.model.status_lines() if self.model is not None else []:
                self.console.say("  " + line)
        elif cmd == "/new":
            if self.model is not None:
                self.model.reset()
            self.console.say("Conversation cleared.")
        elif cmd == "/login":
            await self._login()
        elif cmd == "/logout":
            await self._logout()
        else:
            self.console.say(f"Unknown command {cmd}. {HELP}")
        return True

    async def _login(self) -> None:
        rt = self.ai_runtime
        if rt is None or self.settings.ai_runtime != AiRuntimeMode.CHATGPT_OAUTH:
            self.console.say("/login is for Sign in with ChatGPT (LAT_AI_RUNTIME=CHATGPT_OAUTH).")
            return
        try:
            info = await rt.chatgpt_login()
        except ChatGptError as exc:
            self.console.say(f"Sign-in could not start: {exc.describe()}")
            return
        if info.get("simulated"):
            self.console.say("FAKE transport: signing in with a simulated ChatGPT account (no OpenAI account involved).")
        else:
            self.console.say("Open this URL in your browser and sign in with ChatGPT:")
            self.console.say(str(info["authorize_url"]))
            self.console.say(f"Waiting for the callback on {info['callback']} (up to {int(self.config.login_timeout_s)} s; "
                             "Ctrl-C cancels).")
        try:
            err = await rt.auth.wait_login(self.config.login_timeout_s)
        except TimeoutError:
            await rt.auth.cancel_login()
            err = "sign-in timed out"
        if err:
            self.console.say(f"Sign-in not completed: {err}")
            return
        st = rt.status().get("chatgpt", {})
        self.console.say(f"Signed in as {st.get('email_masked')}; plan usage granted: {st.get('plan_usage_granted')}.")
        if not st.get("plan_usage_granted"):
            self.console.say("Without plan usage (chatgpt.tokens.use.direct) the sign-in is never used for inference.")

    async def _logout(self) -> None:
        rt = self.ai_runtime
        if rt is None or self.settings.ai_runtime != AiRuntimeMode.CHATGPT_OAUTH:
            self.console.say("/logout is for Sign in with ChatGPT (LAT_AI_RUNTIME=CHATGPT_OAUTH).")
            return
        res = await rt.chatgpt_logout()
        self.console.say(f"Signed out (remote revocation: {'confirmed' if res.get('revoked') else 'NOT confirmed'}; "
                         f"{res.get('detail')}).")

    # ------------------------------------------------------------------ one owner message
    async def ask(self, text: str) -> ChatAnswer | None:
        if self.model is None:
            self.console.say("No model is configured for this chat.")
            return None
        problem = await self.refresh_tools()  # the LIVE catalog, every message
        if problem and not self.tools:
            self.console.say(f"{problem} Start it with `make dev` (or `make paper`) and try again.")
            return None
        executor = ToolExecutor(self.caller, self.tools, self.confirm, self.config, self.console)
        prompt = text
        if self._carry_note:
            prompt = f"{self._carry_note}\n\n{text}"
        started = self.clock.now()
        t0 = time.perf_counter()
        try:
            answer = await self.model.answer(prompt, executor)
        except ChatModelError as exc:
            self._remember_actions(executor)
            self.console.say(str(exc))
            return None
        latency = max(0, int((time.perf_counter() - t0) * 1000))
        self._carry_note = None
        if executor.unknown:
            log_event(log, "model asked for unavailable tools", logging.WARNING, tools=sorted(set(executor.unknown)))
        self.console.say(answer.text or "(no answer)")
        self.console.say(self._footer(answer, executor))
        self.answers.append(answer)
        self._record(answer, started, latency)
        return answer

    def _remember_actions(self, executor: ToolExecutor) -> None:
        ran = [a for a in executor.actions if a.outcome == "ran" or a.outcome.startswith("error:5")]
        if ran:
            done = "; ".join(f"{a.tool} {compact(a.args, 200)} -> {a.outcome}" for a in ran)
            self._carry_note = (f"[Client note, not from the owner: in the previous turn, whose answer failed, these "
                                f"actions already ran or have an UNKNOWN outcome: {done}. Do not repeat them; check the "
                                "state with a read tool first.]")

    def _footer(self, answer: ChatAnswer, executor: ToolExecutor) -> str:
        bits = [answer.transport.value, answer.model_id, f"{len(answer.requests)} request(s)"]
        if executor.calls:
            bits.append("tools: " + ", ".join(dict.fromkeys(executor.calls)))
        if answer.pricing_source == PLAN_PRICING_SOURCE:
            bits.append("ChatGPT plan (no per-token charge to the app)")
        elif answer.transport == ModelTransport.REAL:
            bits.append(f"est. ${answer.cost_usd}")
        return "  [" + " | ".join(bits) + "]"

    def _record(self, answer: ChatAnswer, started: datetime, latency_ms: int) -> None:
        totals = total_tokens(answer.requests)
        material = answer.request_hashes or [sha256_hex(canonical_json({"started": started.isoformat()}))]
        adapter = CHATGPT_ADAPTER_VERSION if answer.pricing_source == PLAN_PRICING_SOURCE else ADAPTER_VERSION
        try:
            record = ModelRunRecord(
                run_id=new_id("mrun"), causal_id=new_causal_id(), provider=answer.provider, model_id=answer.model_id,
                adapter_version=f"{adapter} chat", prompt_version=prompt_version(CHAT_PROMPT),
                effort=self.settings.sol_effort, transport=answer.transport, status=answer.status, started_at=started,
                finished_at=max(self.clock.now(), started), latency_ms=latency_ms, input_tokens=totals.input_tokens,
                output_tokens=totals.output_tokens, cache_read_tokens=totals.cache_read_tokens,
                cache_write_tokens=totals.cache_write_tokens, reasoning_tokens=totals.reasoning_tokens,
                estimated_cost_usd=answer.cost_usd, pricing_source=answer.pricing_source or "UNVERIFIED",
                request_sha256=sha256_hex("\n".join(material)),
                response_sha256=sha256_hex("\n".join(answer.response_hashes)) if answer.response_hashes else None,
                output_schema="text", error=answer.error,
            )
        except Exception as exc:  # accounting never breaks the conversation
            log_event(log, "chat model-run record invalid", logging.WARNING, error=type(exc).__name__)
            return
        self.records.append(record)
        if self.run_log is not None:
            append_private_jsonl(self.run_log, record.model_dump(mode="json"))


def append_private_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(obj), sort_keys=True, default=str) + "\n")
    except OSError as exc:
        log_event(log, "chat run log not written", logging.WARNING, error=type(exc).__name__)


# ------------------------------------------------------------------------------------------------
# wiring + entry point
# ------------------------------------------------------------------------------------------------


def mode_explanation(mode: AiRuntimeMode) -> str | None:
    """Why this client does not run a model in ``mode`` (``None`` when it does)."""
    if mode in (AiRuntimeMode.CLAUDE_CODE, AiRuntimeMode.CODEX):
        client = "Claude Code" if mode == AiRuntimeMode.CLAUDE_CODE else "Codex"
        return (f"AI runtime mode {mode.value}: talk to {client} directly. Your {client} session operates the app through "
                f"the local MCP server '{MCP_SERVER_NAME}' (`uv run --frozen python -m services.mcp_server`; register it "
                "with `make register-mcp`), using the same validated tool interface. This app calls no model itself "
                "in this mode, and the trading service keeps running on its own.")
    if mode == AiRuntimeMode.NONE:
        return ("AI runtime mode NONE: quant only, so the app calls no model and there is nobody to chat with. To chat, "
                "set LAT_AI_RUNTIME=CHATGPT_OAUTH (Sign in with ChatGPT) or API_KEYS, or use Claude Code / Codex "
                f"through the MCP server '{MCP_SERVER_NAME}'.")
    return None


def build_session(settings: Settings, secrets: SecretStore, console: Console, *, backend: Any | None = None,
                  config: ChatConfig | None = None, ai_runtime: Any | None = None, api_model: ChatModel | None = None,
                  confirm: ConfirmFn | None = None, run_log: Path | bool | None = True) -> ChatSession:
    """Wire a chat session for the configured mode (no network I/O happens here)."""
    from services.tools.backends import HttpToolBackend

    cfg = config or ChatConfig()
    if backend is None:
        backend = HttpToolBackend(f"http://{settings.api_host}:{settings.api_port}",
                                  settings.keys_dir / "agent.token", actor="chat", lazy_token=True)
    caller = ToolCaller(backend)
    mode = settings.ai_runtime
    model: ChatModel | None = None
    if mode == AiRuntimeMode.CHATGPT_OAUTH:
        if ai_runtime is None:
            from .runtime_modes import AiRuntime

            rt_settings = settings if settings.ai_transport == "real" else settings.model_copy(
                update={"chatgpt_oauth_port": 0})  # simulated sign-in: any free loopback port
            ai_runtime = AiRuntime(rt_settings, secrets, db=None)
        if ai_runtime.transport == ModelTransport.FAKE and ai_runtime.fake_server is not None \
                and ai_runtime.fake_server.responder is None:
            from .chat_fake import ChatScript

            ai_runtime.fake_server.responder = ChatScript()
        model = PlanChatModel(ai_runtime, config=cfg)
    elif mode == AiRuntimeMode.API_KEYS:
        model = api_model or ApiKeyChatModel(settings, secrets, config=cfg)
    log_path = (settings.log_dir / "chat_model_runs.jsonl") if run_log is True else (run_log or None)
    return ChatSession(settings, model, caller, console, config=cfg, confirm=confirm, ai_runtime=ai_runtime,
                       run_log=log_path)


async def run_chat(settings: Settings, secrets: SecretStore, console: Console, *, session: ChatSession | None = None,
                   commands: Sequence[str] | None = None) -> int:
    """The REPL. Returns the process exit code (0 also when the mode needs no chat)."""
    why = mode_explanation(settings.ai_runtime)
    if why is not None:
        console.say(why)
        return 0
    s = session or build_session(settings, secrets, console)
    problem = await s.refresh_tools()
    for line in s.banner():
        console.say(line)
    if problem:
        console.say(f"{problem} Start it with `make dev` (or `make paper`); the chat retries on every message.")
    if s.ai_runtime is not None and s.ai_runtime.transport == ModelTransport.FAKE and not s.ai_runtime.auth.signed_in():
        await s.handle("/login")  # FAKE: simulated sign-in so the scripted demo works end to end
    try:
        if commands is not None:
            for c in commands:
                console.say(f"> {c}")
                if not await s.handle(c):
                    break
            return 0
        while True:
            line = await console.ask("\nyou> ")
            if line is None:
                return 0
            try:
                if not await s.handle(line):
                    return 0
            except Exception as exc:  # a client bug must not end the conversation (or affect the service)
                log_event(log, "chat turn failed", logging.ERROR, error=redact(repr(exc))[:500])
                console.say(f"Internal chat error ({type(exc).__name__}). {SERVICE_KEEPS_RUNNING}")
    finally:
        model_close = getattr(s.model, "aclose", None)
        if model_close is not None:
            with contextlib.suppress(Exception):
                await model_close()
        if s.ai_runtime is not None:
            with contextlib.suppress(Exception):
                await s.ai_runtime.auth.cancel_login()
        aclose = getattr(s.caller.backend, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


def _configure_file_logging(settings: Settings) -> None:
    """JSON logs to ``var/<ns>/logs/chat.log`` only (the terminal shows the conversation, not log lines)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(settings.log_dir / "chat.log")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    except OSError:
        root.addHandler(logging.NullHandler())
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "anthropic"):  # request URLs stay out of the log
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m services.ai.chat",
                                     description="Talk to the local paper-trading app in plain language.")
    parser.add_argument("-c", "--command", action="append", metavar="TEXT",
                        help="send this message (repeatable) and exit instead of starting the REPL")
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = load_settings()
    _configure_file_logging(settings)
    secrets = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
    console = Console()
    try:
        return asyncio.run(run_chat(settings, secrets, console, commands=args.command))
    except KeyboardInterrupt:
        console.say("\nBye. The trading service keeps running.")
        return 0


__all__ = [
    "ApiKeyChatModel",
    "CatalogTool",
    "ChatAnswer",
    "ChatConfig",
    "ChatModelError",
    "ChatSession",
    "Console",
    "PlanChatModel",
    "ToolCaller",
    "ToolExecutor",
    "build_chat_body",
    "build_session",
    "clean_terminal",
    "console_confirm",
    "main",
    "mode_explanation",
    "parse_catalog",
    "run_chat",
]


if __name__ == "__main__":
    raise SystemExit(main())
