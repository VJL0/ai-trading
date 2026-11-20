"""AI runtime modes: how (and whether) an AI participates. The quant engine never depends on any of them.

=============== ==========================================================================================
``NONE``        quant only; the app never calls a model.
``API_KEYS``    in-app Claude Opus 5.5 + GPT-6.1 Sol adapters (``services.ai.pipeline.ResearchPipeline``),
                separate API billing, ``LAT_AI_TRANSPORT=real`` + keys required for real calls.
``CLAUDE_CODE`` the user's own Claude Code session operates the app through the local MCP server
                (``services.mcp_server``) and the validated tool interface; the app calls no model itself.
``CODEX``       the same for the user's own Codex session.
``CHATGPT_OAUTH`` research turns on the user's ChatGPT plan through the official Sign in with ChatGPT flow
                (``services.ai.chatgpt_oauth`` + ``services.ai.chatgpt_responses``), function tools = the shared
                registry's READ-ONLY tools, invoked only on an explicit operator action or the gated research
                cycle.
=============== ==========================================================================================

Deterministic risk limits stay authoritative in every mode: every proposal (AI or not) goes through the
independent risk engine, the signed-intent gateway and the paper/simulated gateway. Models never receive the
operator/engine/agent tokens: in-app runtimes reach tools through ``InProcessToolBackend`` (no token at all) and
the MCP server keeps its token out of every tool output.

``AiRuntime.status()`` returns the ``GET /api/v1/ai/runtime`` body except the ``mcp`` and ``limits`` blocks, which
the API fills from ``services.tools`` and the proposals table.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from packages.contracts import Evidence, MarketSnapshot, QuantSignal, RiskPolicy
from packages.contracts.enums import AiRuntimeMode, ModelTransport
from packages.core.clock import Clock, SystemClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.logging import log_event
from packages.core.secrets import SecretName, SecretStore

from .chatgpt_fake import ChatGptScenario, FakeChatGptServer, fake_browser_sign_in
from .chatgpt_oauth import (
    ELIGIBILITY_NOTE,
    INVOCATION_POLICY,
    PLAN_PRICING_SOURCE,
    USAGE_URL,
    ChatGptAuth,
    ChatGptOAuthConfig,
)
from .chatgpt_responses import ChatGptConfig, ChatGptResearcher
from .config import AiConfig
from .token_store import KeychainTokenStore, MemoryTokenStore, TokenStore

log = logging.getLogger("lat.ai.runtime")

MODE_DETAIL: Final[dict[AiRuntimeMode, str]] = {
    AiRuntimeMode.NONE: "Quant only: no AI participates; signals, model predictions, risk checks and the paper path "
                        "run deterministically.",
    AiRuntimeMode.API_KEYS: "In-app Claude Opus 5.5 (research) + GPT-6.1 Sol (orchestration) via API keys; separate "
                            "API billing; real calls only with LAT_AI_TRANSPORT=real.",
    AiRuntimeMode.CLAUDE_CODE: "Your own Claude Code session operates the app through the local MCP server and the "
                               "validated tool interface; the app calls no model itself and the quant engine runs alone.",
    AiRuntimeMode.CODEX: "Your own Codex session operates the app through the local MCP server and the validated "
                         "tool interface; the app calls no model itself and the quant engine runs alone.",
    AiRuntimeMode.CHATGPT_OAUTH: "Research turns on your ChatGPT plan via the official Sign in with ChatGPT flow "
                                 "(Responses API, read-only function tools from the validated tool interface).",
}


@dataclass
class _ChatGptParts:
    auth: ChatGptAuth
    researcher: ChatGptResearcher
    http: httpx.AsyncClient
    fake_server: FakeChatGptServer | None


# One ChatGPT session per (namespace dir, transport) in this process: two sessions would race the rotating
# refresh token. Only used when nothing was injected (tests inject their own store/transport).
_SHARED: dict[tuple[str, str], _ChatGptParts] = {}
_SHARED_LOCK = threading.Lock()


def _is_mock(client: httpx.AsyncClient) -> bool:
    return isinstance(getattr(client, "_transport", None), httpx.MockTransport)


class AiRuntime:
    def __init__(
        self,
        settings: Settings,
        secret_store: SecretStore,
        *,
        db: Database | None = None,
        clock: Clock | None = None,
        wall_clock: Clock | None = None,
        ai_config: AiConfig | None = None,
        chatgpt_config: ChatGptConfig | None = None,
        token_store: TokenStore | None = None,
        http_client: httpx.AsyncClient | None = None,
        oauth_config: ChatGptOAuthConfig | None = None,
        fake_server: FakeChatGptServer | None = None,
        auto_complete_fake_login: bool = True,
    ) -> None:
        self.settings = settings
        self._secrets = secret_store
        self.db = db
        self.clock: Clock = clock or SystemClock()
        self.wall_clock: Clock = wall_clock or SystemClock()
        self.ai_config = ai_config or AiConfig()
        self.chatgpt_config = chatgpt_config or ChatGptConfig()
        self.transport = ModelTransport.REAL if settings.ai_transport == "real" else ModelTransport.FAKE
        self.auto_complete_fake_login = auto_complete_fake_login and self.transport == ModelTransport.FAKE
        self._tasks: set[asyncio.Task[Any]] = set()
        injected = any(x is not None for x in (token_store, http_client, oauth_config, fake_server))
        if injected:
            self._parts = self._build(token_store, http_client, oauth_config, fake_server)
        else:
            key = (str(settings.var_dir), self.transport.value)
            with _SHARED_LOCK:
                parts = _SHARED.get(key)
                if parts is None:
                    parts = self._build(None, None, None, None)
                    _SHARED[key] = parts
            self._parts = parts

    def _build(self, token_store: TokenStore | None, http_client: httpx.AsyncClient | None,
               oauth_config: ChatGptOAuthConfig | None, fake_server: FakeChatGptServer | None) -> _ChatGptParts:
        if self.transport == ModelTransport.FAKE:
            config = oauth_config or (fake_server.config if fake_server is not None else ChatGptOAuthConfig.fake())
            if fake_server is None and http_client is None:
                fake_server = FakeChatGptServer(config, scenario=ChatGptScenario(self.chatgpt_config.fake_scenario),
                                                now=lambda: self.wall_clock.now().timestamp())
            http = http_client or (fake_server.client() if fake_server is not None else None)
            assert http is not None
            if not _is_mock(http):
                raise ValueError("FAKE ChatGPT transport requires an httpx.MockTransport client (no network)")
            store = token_store or MemoryTokenStore()  # simulated sign-ins never touch the user's Keychain
        else:
            config = oauth_config or ChatGptOAuthConfig()
            if http_client is not None and _is_mock(http_client):
                raise ValueError("REAL ChatGPT transport refuses a MockTransport client (it would mislabel FAKE as REAL)")
            http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=False)
            store = token_store or KeychainTokenStore()
        auth = ChatGptAuth(self.settings, transport=self.transport, token_store=store, http_client=http, config=config,
                           wall_clock=self.wall_clock)
        researcher = ChatGptResearcher(self.settings, auth, http, config=self.chatgpt_config, ai_config=self.ai_config,
                                       db=self.db, clock=self.clock, wall_clock=self.wall_clock)
        auth.on_signed_in.append(researcher.clear_block)
        return _ChatGptParts(auth, researcher, http, fake_server)

    # ------------------------------------------------------------------ accessors
    @property
    def mode(self) -> AiRuntimeMode:
        return self.settings.ai_runtime

    @property
    def auth(self) -> ChatGptAuth:
        return self._parts.auth

    @property
    def researcher(self) -> ChatGptResearcher:
        return self._parts.researcher

    @property
    def fake_server(self) -> FakeChatGptServer | None:
        return self._parts.fake_server

    @property
    def app_calls_models(self) -> bool:
        return self.mode in (AiRuntimeMode.API_KEYS, AiRuntimeMode.CHATGPT_OAUTH)

    # ------------------------------------------------------------------ status
    def _has(self, name: SecretName) -> bool:
        try:
            return self._secrets.has(name)
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        mode = self.mode
        chatgpt: dict[str, Any] = {}
        try:
            chatgpt.update(self.auth.status())
            chatgpt.update(self.researcher.status())
        except Exception as exc:  # status never raises; the quant engine is unaffected
            chatgpt["last_error"] = f"{type(exc).__name__}"
        chatgpt.setdefault("signed_in", False)
        chatgpt.setdefault("plan_usage_granted", False)
        chatgpt.setdefault("email_masked", None)
        chatgpt.setdefault("expires_at", None)
        chatgpt.setdefault("usage_limit_hit", False)
        if chatgpt.get("last_error") is None:
            chatgpt["last_error"] = chatgpt.get("last_inference_error")
        chatgpt.update({
            "model": self.researcher.selected_model or self.settings.chatgpt_model,
            "active": mode == AiRuntimeMode.CHATGPT_OAUTH,
            "callback": self.auth.pending_callback
            or f"http://127.0.0.1:{self.settings.chatgpt_oauth_port}/auth/callback",
            "usage_url": USAGE_URL,
            "pricing_source": PLAN_PRICING_SOURCE,
            "eligibility": ELIGIBILITY_NOTE,
            "invocation_policy": INVOCATION_POLICY,
        })
        return {
            "mode": mode.value,
            "quant_engine_independent": True,
            "app_calls_models": self.app_calls_models,
            "detail": MODE_DETAIL[mode],
            "risk_limits_authoritative": True,
            "api_keys": {
                "transport": "REAL" if self.settings.ai_transport == "real" else "FAKE",
                "anthropic_key": self._has(SecretName.ANTHROPIC_API_KEY),
                "openai_key": self._has(SecretName.OPENAI_API_KEY),
                "claude_model": self.settings.claude_model,
                "sol_model": self.settings.sol_model,
                "active": mode == AiRuntimeMode.API_KEYS,
            },
            "external_agent": {
                "active": mode in (AiRuntimeMode.CLAUDE_CODE, AiRuntimeMode.CODEX),
                "client": {AiRuntimeMode.CLAUDE_CODE: "Claude Code", AiRuntimeMode.CODEX: "Codex"}.get(mode),
                "detail": "operates the app via the local MCP server (services.mcp_server); the app itself calls no "
                          "model, and proposals still pass the deterministic risk engine",
            },
            "chatgpt": chatgpt,
        }

    # ------------------------------------------------------------------ Sign in with ChatGPT
    async def chatgpt_login(self) -> dict[str, Any]:
        """Start the official PKCE loopback sign-in; the human completes it in their own browser."""
        info = await self.auth.start_login()
        out: dict[str, Any] = {"authorize_url": info["authorize_url"], "callback": info["callback"],
                               "transport": info["transport"], "simulated": info["simulated"],
                               "expires_in_s": info["expires_in_s"]}
        server = self.fake_server
        if self.auto_complete_fake_login and server is not None:
            # FAKE transport: simulate the browser so the whole flow runs locally without any OpenAI account.
            task = asyncio.create_task(self._fake_browser(server, str(info["authorize_url"])), name="fake-chatgpt-login")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            out["note"] = "FAKE transport: sign-in is simulated locally (no OpenAI account is involved)"
        return out

    async def _fake_browser(self, server: FakeChatGptServer, url: str) -> None:
        try:
            await fake_browser_sign_in(server, url)
        except Exception as exc:
            log_event(log, "simulated chatgpt sign-in failed", logging.WARNING, error=type(exc).__name__)

    async def chatgpt_logout(self) -> dict[str, Any]:
        return await self.auth.logout()

    async def chatgpt_research(
        self,
        symbol: str | None,
        *,
        tool_backend: Any | None,
        candidate: QuantSignal | None,
        snapshot: MarketSnapshot,
        evidence: Sequence[Evidence],
        policy: RiskPolicy,
        causal_id: str,
    ) -> Any:
        """One bounded research turn on the ChatGPT plan -> ``PipelineResult`` (proposal or NO_TRADE + run records).

        ``tool_backend`` is a ``services.tools`` ``ToolBackend`` (normally ``InProcessToolBackend(runtime)``); only
        its READ-ONLY tools are offered and executed. The returned proposal still has to pass the risk engine."""
        return await self.researcher.research(symbol=symbol, tool_backend=tool_backend, candidate=candidate,
                                              snapshot=snapshot, evidence=evidence, policy=policy, causal_id=causal_id)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.auth.aclose()


def reset_shared_runtimes() -> None:
    """Forget process-shared ChatGPT sessions (tests)."""
    with _SHARED_LOCK:
        _SHARED.clear()


__all__ = ["MODE_DETAIL", "AiRuntime", "reset_shared_runtimes"]
