"""Live Sign-in-with-ChatGPT smoke tests (the user's real ChatGPT plan; no API billing to the app).

They run only when BOTH hold: ``LAT_AI_TRANSPORT=real`` is set explicitly AND the user has already signed in (a
refresh token and issued client id exist in the macOS Keychain, service ``local-ai-trader``). The environment is
checked first, so a default run never even reads the Keychain. Nothing here places an order: the research turn
only returns a proposal (never sent to the risk engine or a gateway) and the tool backend is the deterministic
``FakeToolRuntime`` (SIMULATED data); the chat test asks a read-only question and auto-declines every action.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from packages.contracts import RiskPolicy
from packages.contracts.enums import AiRuntimeMode, ModelProvider, ModelTransport
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.secrets import SecretStore
from services.ai.evidence import build_internal_evidence
from tests.unit.ai.conftest import T0, make_signal, make_snapshot

pytestmark = pytest.mark.live_models


def _require_signed_in() -> tuple[Settings, SecretStore]:
    if os.environ.get("LAT_AI_TRANSPORT") != "real":
        pytest.skip("live ChatGPT tests need LAT_AI_TRANSPORT=real (real plan requests are never implicit)")
    from services.ai.token_store import ACCOUNT_CLIENT_ID, ACCOUNT_REFRESH_TOKEN, KeychainTokenStore, TokenStoreError

    try:
        store = KeychainTokenStore()
        signed_in = store.get(ACCOUNT_REFRESH_TOKEN) is not None and store.get(ACCOUNT_CLIENT_ID) is not None
    except TokenStoreError as exc:
        pytest.skip(f"macOS Keychain unavailable: {exc}")
    if not signed_in:
        pytest.skip("not signed in with ChatGPT (run the sign-in first: make chat, then /login)")
    settings = Settings(ai_runtime=AiRuntimeMode.CHATGPT_OAUTH, ai_transport="real")
    return settings, SecretStore(settings.project_root, use_keychain=settings.use_keychain)


async def test_live_chatgpt_research_turn_returns_a_validated_proposal() -> None:
    settings, secrets = _require_signed_in()
    from services.ai.runtime_modes import AiRuntime
    from services.tools.backends import InProcessToolBackend
    from tests.unit.tools.fake_runtime import FakeToolRuntime

    clock = ManualClock(T0)  # decision clock for the point-in-time inputs; wall clock stays real
    rt = AiRuntime(settings, secrets, db=None, clock=clock)
    try:
        sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
        result = await rt.chatgpt_research(sig.symbol, tool_backend=InProcessToolBackend(FakeToolRuntime(), "live-test"),
                                           candidate=sig, snapshot=snap, evidence=build_internal_evidence(sig, snap, pol),
                                           policy=pol, causal_id="cz_live_chatgpt")
    finally:
        await rt.aclose()
    assert result.proposal.action.value in ("TRADE", "NO_TRADE")
    runs = result.model_runs
    assert runs and all(r.transport == ModelTransport.REAL and r.provider == ModelProvider.OPENAI for r in runs)
    assert all(r.pricing_source.startswith("CHATGPT_PLAN") for r in runs)


async def test_live_chat_answers_a_read_only_question() -> None:
    settings, secrets = _require_signed_in()
    from services.ai.chat import ChatSession, Console, build_session
    from services.ai.runtime_modes import AiRuntime
    from services.tools.backends import InProcessToolBackend
    from tests.unit.tools.fake_runtime import FakeToolRuntime

    out: list[str] = []

    async def read(_prompt: str) -> str | None:
        return None  # any action confirmation is declined

    async def never(*_a: Any) -> bool:
        return False

    rt = AiRuntime(settings, secrets, db=None)
    session: ChatSession = build_session(settings, secrets, Console(read=read, write=out.append),
                                         backend=InProcessToolBackend(FakeToolRuntime(), "live-chat"), ai_runtime=rt,
                                         confirm=never, run_log=False)
    try:
        assert await session.refresh_tools() is None
        answer = await session.ask("What is my paper account equity? Use the portfolio tool.")
    finally:
        await rt.aclose()
    assert answer is not None, "\n".join(out)
    assert answer.transport == ModelTransport.REAL and answer.text
