"""Provider adapters (PydanticAI) for the two fixed models.

* Claude Opus 5.5 -- ``pydantic_ai.models.anthropic.AnthropicModel`` over the Anthropic **Messages API**
  (``POST /v1/messages``), model ``claude-opus-5-5``, ``thinking={"type": "adaptive", "display": "summarized"}``
  (thinking cannot be disabled on this model; ``budget_tokens`` is rejected) and an explicit
  ``output_config.effort`` (``high`` by default; ``xhigh``/``max`` for scoped escalation -- the model's own
  default would be ``medium``). Structured output uses ``output_config.format`` (PydanticAI ``NativeOutput``)
  because forced ``tool_choice`` returns a 400 on this model.
* GPT-6.1 Sol -- ``pydantic_ai.models.openai.OpenAIResponsesModel`` over the OpenAI **Responses API**
  (``POST /v1/responses``; never Chat Completions), model ``gpt-6.1-sol``, ``reasoning={"effort": "medium"}``
  (adjustable), function tools, ``store=False``.

Transport selection is explicit and fail-closed:
* ``transport='fake'`` never reads a key and only accepts an ``httpx2.MockTransport`` client (default: the
  scripted fake handlers), so no network request is possible. Runs are recorded ``transport=FAKE``.
* ``transport='real'`` requires ``Settings.ai_transport == 'real'`` (``LAT_AI_TRANSPORT=real``) AND the key from
  ``packages.core.secrets.SecretStore``; it is never inferred from key presence and refuses mock clients so a
  run can never be mislabelled ``REAL``. SDK retries are disabled (``max_retries=0``): a timeout is reported, not
  silently replayed.

Note: the installed ``anthropic`` 2.x / ``openai`` 3.x SDKs are built on ``httpx2`` (``httpx`` API), so custom
clients must be ``httpx2.AsyncClient``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx2
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from packages.contracts.enums import ModelProvider, ModelTransport
from packages.core.config import Settings
from packages.core.secrets import SecretName, SecretStore

from .accounting import install_capture_hooks
from .config import AiConfig
from .fake_transport import FAKE_API_KEY, FakeAnthropicHandler, FakeOpenAIHandler, FakeScenario, fake_http_client, is_mock_client
from .versions import ADAPTER_VERSION

TransportName = Literal["fake", "real"]
ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]
SolEffort = Literal["low", "medium", "high", "xhigh", "max"]

CLAUDE_MODEL_ID = "claude-opus-5-5"
SOL_MODEL_ID = "gpt-6.1-sol"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENAI_BASE_URL = "https://api.openai.com/v1"

__all__ = [
    "ADAPTER_VERSION",
    "CLAUDE_MODEL_ID",
    "SOL_MODEL_ID",
    "AiTransportError",
    "ModelHandle",
    "build_claude_handle",
    "build_claude_model",
    "build_sol_handle",
    "build_sol_model",
    "claude_model_settings",
    "sol_model_settings",
]


class AiTransportError(RuntimeError):
    """The requested model transport is not available (fail closed: the pipeline returns NO_TRADE)."""


def _transport_enum(transport: str) -> ModelTransport:
    if transport == "fake":
        return ModelTransport.FAKE
    if transport == "real":
        return ModelTransport.REAL
    raise AiTransportError(f"unknown AI transport {transport!r} (expected 'fake' or 'real')")


def _real_key(settings: Settings, secret_store: SecretStore, name: SecretName) -> str:
    if settings.ai_transport != "real":
        raise AiTransportError(
            "real model transport requested but LAT_AI_TRANSPORT is not 'real' (paid API calls must be enabled "
            "explicitly; they are never inferred from key presence)"
        )
    secret = secret_store.get(name)
    if secret is None:
        raise AiTransportError(f"real model transport requires {name.value} (Keychain/env/.env); it is not configured")
    return secret.get_secret_value()


def _client(
    transport: TransportName,
    http_client: httpx2.AsyncClient | None,
    fake_factory: Callable[[], httpx2.AsyncClient],
    settings: Settings,
) -> httpx2.AsyncClient:
    if transport == "fake":
        client = http_client or fake_factory()
        if not is_mock_client(client):
            raise AiTransportError("fake transport requires an httpx2.MockTransport client (no network allowed)")
    else:
        if http_client is not None and is_mock_client(http_client):
            raise AiTransportError("real transport refuses a MockTransport client (it would mislabel FAKE runs as REAL)")
        client = http_client or httpx2.AsyncClient(timeout=httpx2.Timeout(settings.model_timeout_s, connect=10.0))
    install_capture_hooks(client)
    return client


def claude_model_settings(config: AiConfig, effort: ClaudeEffort, settings: Settings) -> AnthropicModelSettings:
    return AnthropicModelSettings(
        max_tokens=config.claude_max_tokens,
        anthropic_thinking={"type": "adaptive", "display": "summarized"},
        anthropic_effort=effort,
        anthropic_cache=True,  # top-level automatic prompt caching across the tool loop (5-minute TTL)
        timeout=settings.model_timeout_s,
    )


def sol_model_settings(config: AiConfig, effort: SolEffort, settings: Settings) -> OpenAIResponsesModelSettings:
    return OpenAIResponsesModelSettings(
        max_tokens=config.sol_max_tokens,
        openai_reasoning_effort=effort,
        openai_store=False,  # local-first: do not persist responses server-side
        timeout=settings.model_timeout_s,
    )


def build_claude_model(
    settings: Settings,
    secret_store: SecretStore,
    *,
    transport: TransportName,
    http_client: httpx2.AsyncClient | None = None,
    effort: ClaudeEffort = "high",
    config: AiConfig | None = None,
    fake_scenario: FakeScenario | str | None = None,
) -> AnthropicModel:
    """PydanticAI ``AnthropicModel`` for ``claude-opus-5-5`` (adaptive thinking + explicit effort)."""
    cfg = config or AiConfig()
    _transport_enum(transport)
    scenario = FakeScenario(fake_scenario or cfg.fake_scenario)
    api_key = FAKE_API_KEY if transport == "fake" else _real_key(settings, secret_store, SecretName.ANTHROPIC_API_KEY)
    client = _client(transport, http_client, lambda: fake_http_client(FakeAnthropicHandler(scenario)), settings)
    sdk = AsyncAnthropic(api_key=api_key, base_url=ANTHROPIC_BASE_URL, http_client=client, max_retries=0,
                         timeout=settings.model_timeout_s)
    return AnthropicModel(
        settings.claude_model,
        provider=AnthropicProvider(anthropic_client=sdk),
        settings=claude_model_settings(cfg, effort, settings),
    )


def build_sol_model(
    settings: Settings,
    secret_store: SecretStore,
    *,
    transport: TransportName,
    http_client: httpx2.AsyncClient | None = None,
    effort: SolEffort = "medium",
    config: AiConfig | None = None,
    fake_scenario: FakeScenario | str | None = None,
) -> OpenAIResponsesModel:
    """PydanticAI ``OpenAIResponsesModel`` for ``gpt-6.1-sol`` (Responses API, ``reasoning.effort``)."""
    cfg = config or AiConfig()
    _transport_enum(transport)
    scenario = FakeScenario(fake_scenario or cfg.fake_scenario)
    api_key = FAKE_API_KEY if transport == "fake" else _real_key(settings, secret_store, SecretName.OPENAI_API_KEY)
    client = _client(transport, http_client, lambda: fake_http_client(FakeOpenAIHandler(scenario)), settings)
    sdk = AsyncOpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, http_client=client, max_retries=0,
                      timeout=settings.model_timeout_s)
    return OpenAIResponsesModel(
        settings.sol_model,
        provider=OpenAIProvider(openai_client=sdk),
        settings=sol_model_settings(cfg, effort, settings),
    )


@dataclass(frozen=True)
class ModelHandle:
    """A built model plus the metadata every ModelRunRecord needs."""

    model: AnthropicModel | OpenAIResponsesModel
    provider: ModelProvider
    model_id: str
    effort: str
    transport: ModelTransport
    adapter_version: str = ADAPTER_VERSION


def build_claude_handle(
    settings: Settings, secret_store: SecretStore, *, transport: TransportName, http_client: httpx2.AsyncClient | None = None,
    effort: ClaudeEffort = "high", config: AiConfig | None = None,
) -> ModelHandle:
    model = build_claude_model(settings, secret_store, transport=transport, http_client=http_client, effort=effort,
                               config=config)
    return ModelHandle(model, ModelProvider.ANTHROPIC, settings.claude_model, effort, _transport_enum(transport))


def build_sol_handle(
    settings: Settings, secret_store: SecretStore, *, transport: TransportName, http_client: httpx2.AsyncClient | None = None,
    effort: SolEffort = "medium", config: AiConfig | None = None,
) -> ModelHandle:
    model = build_sol_model(settings, secret_store, transport=transport, http_client=http_client, effort=effort,
                            config=config)
    return ModelHandle(model, ModelProvider.OPENAI, settings.sol_model, effort, _transport_enum(transport))
