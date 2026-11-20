"""AI runtime modes: status shape per mode, models are only called by the app in API_KEYS / CHATGPT_OAUTH, the
quant engine stays independent, and transports can never be mislabelled."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from packages.contracts.enums import AiRuntimeMode, DecisionAction
from packages.core.config import Settings
from packages.core.secrets import SecretStore
from services.ai.chatgpt_fake import FakeChatGptServer
from services.ai.fake_transport import FakeScenario
from services.ai.runtime_modes import AiRuntime, reset_shared_runtimes

CONTRACT_CHATGPT_KEYS = {"signed_in", "plan_usage_granted", "email_masked", "model", "expires_at", "last_error",
                         "usage_limit_hit"}
CONTRACT_API_KEYS = {"transport", "anthropic_key", "openai_key", "claude_model", "sol_model"}


def _rt(tmp_path: Path, mode: AiRuntimeMode, **over: Any) -> AiRuntime:
    settings = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False, ai_runtime=mode,
                        chatgpt_oauth_port=0, **over)
    secrets = SecretStore(tmp_path, use_keychain=False, env={"OPENAI_API_KEY": "sk-test-not-real"},
                          dotenv_path=tmp_path / "no.env")
    return AiRuntime(settings, secrets, fake_server=FakeChatGptServer(), auto_complete_fake_login=False)


@pytest.mark.parametrize("mode", list(AiRuntimeMode))
async def test_status_shape_for_every_mode(tmp_path: Path, mode: AiRuntimeMode) -> None:
    rt = _rt(tmp_path, mode)
    try:
        st = rt.status()
        assert st["mode"] == mode.value and st["quant_engine_independent"] is True
        assert st["risk_limits_authoritative"] is True
        assert set(st["api_keys"]) >= CONTRACT_API_KEYS and set(st["chatgpt"]) >= CONTRACT_CHATGPT_KEYS
        assert st["api_keys"]["openai_key"] is True and st["api_keys"]["anthropic_key"] is False
        assert st["api_keys"]["transport"] == "FAKE"
        assert st["app_calls_models"] is (mode in (AiRuntimeMode.API_KEYS, AiRuntimeMode.CHATGPT_OAUTH))
        assert st["external_agent"]["active"] is (mode in (AiRuntimeMode.CLAUDE_CODE, AiRuntimeMode.CODEX))
        assert "eligibility" in st["chatgpt"] and "OpenAI terms" in st["chatgpt"]["eligibility"]
        assert "never per market tick" in st["chatgpt"]["invocation_policy"]
        assert st["chatgpt"]["signed_in"] is False and st["chatgpt"]["model"] == "gpt-6.1-sol"
        assert "mcp" not in st  # filled by the API from services.tools
        assert "sk-test-not-real" not in repr(st)
    finally:
        await rt.aclose()


@pytest.mark.parametrize("mode", [AiRuntimeMode.NONE, AiRuntimeMode.CLAUDE_CODE, AiRuntimeMode.CODEX])
async def test_pipeline_calls_no_model_outside_api_keys(tmp_path: Path, harness_factory: Any, mode: AiRuntimeMode,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    monkeypatch.setattr(h.pipeline, "settings", h.pipeline.settings.model_copy(update={"ai_runtime": mode}))
    r = await h.run()
    assert r.proposal.action == DecisionAction.NO_TRADE and mode.value in (r.proposal.no_trade_reason or "")
    assert h.sol.exchanges == [] and h.claude.exchanges == [] and r.model_runs == []


async def test_pipeline_delegates_chatgpt_mode_to_the_runtime(tmp_path: Path, harness_factory: Any,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    monkeypatch.setattr(h.pipeline, "settings", h.pipeline.settings.model_copy(
        update={"ai_runtime": AiRuntimeMode.CHATGPT_OAUTH, "chatgpt_oauth_port": 0}))
    seen: dict[str, Any] = {}

    class Stub:
        async def chatgpt_research(self, symbol: str | None, **kw: Any) -> Any:
            seen.update(kw, symbol=symbol)
            from services.ai.pipeline import PipelineResult
            from services.ai.schemas import no_trade_proposal

            p = no_trade_proposal("stub", causal_id=kw["causal_id"], now=kw["snapshot"].data_asof,
                                  snapshot=kw["snapshot"], policy=kw["policy"], candidate=kw["candidate"])
            return PipelineResult(p, [], [])

    h.pipeline.attach_runtime(Stub())
    r = await h.run()
    assert seen["symbol"] == "SPY" and seen["tool_backend"] is None
    assert r.proposal.no_trade_reason == "stub" and h.sol.exchanges == [] and h.claude.exchanges == []


async def test_transports_cannot_be_mislabelled(tmp_path: Path) -> None:
    secrets = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "no.env")
    fake = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False)
    real = Settings(project_root=tmp_path, ai_transport="real", use_keychain=False)
    with pytest.raises(ValueError, match="MockTransport"):
        AiRuntime(fake, secrets, http_client=httpx.AsyncClient())  # FAKE must never reach the network
    with pytest.raises(ValueError, match="mislabel"):
        AiRuntime(real, secrets, http_client=FakeChatGptServer().client())  # REAL must never be a mock


async def test_fake_runtime_never_uses_the_keychain_and_shares_one_session(tmp_path: Path) -> None:
    reset_shared_runtimes()
    settings = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False, chatgpt_oauth_port=0)
    secrets = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "no.env")
    a, b = AiRuntime(settings, secrets), AiRuntime(settings, secrets)
    try:
        assert a.auth is b.auth  # one session per namespace: never two refreshers racing a rotating token
        assert a.auth.store.kind == "memory" and a.auth.config.simulated
        assert a.auth.config.issuer.endswith(".invalid")  # a simulated authorize URL can never reach OpenAI
    finally:
        await a.aclose()
        reset_shared_runtimes()


async def test_fake_login_auto_completes_locally(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False, chatgpt_oauth_port=0,
                        ai_runtime=AiRuntimeMode.CHATGPT_OAUTH)
    secrets = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "no.env")
    rt = AiRuntime(settings, secrets, fake_server=FakeChatGptServer())
    try:
        info = await rt.chatgpt_login()
        assert info["simulated"] is True and info["transport"] == "FAKE" and "FAKE" in info["note"]
        assert info["authorize_url"].startswith("https://auth.openai-fake.invalid/")
        assert await rt.auth.wait_login(10) is None
        assert rt.status()["chatgpt"]["signed_in"] is True
        out = await rt.chatgpt_logout()
        assert out["signed_in"] is False and out["revoked"] is True
    finally:
        await rt.aclose()
