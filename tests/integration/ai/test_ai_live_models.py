"""Live smoke tests against the REAL Anthropic and OpenAI APIs (paid).

They run only when BOTH conditions hold: ``LAT_AI_TRANSPORT=real`` is set explicitly AND the provider key is
available from the SecretStore (Keychain / env / .env). Otherwise they skip. They never place orders: the
pipeline only returns a proposal, and these tests do not call the risk engine or any gateway.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from packages.contracts import RiskPolicy
from packages.contracts.enums import ModelRunStatus, ModelTransport
from packages.core.config import PROJECT_ROOT, Settings
from packages.core.secrets import SecretName, SecretStore
from services.ai.evidence import build_internal_evidence
from tests.unit.ai.conftest import make_signal, make_snapshot

pytestmark = pytest.mark.live_models


def _require(*names: SecretName) -> tuple[Settings, SecretStore]:
    if os.environ.get("LAT_AI_TRANSPORT") != "real":
        pytest.skip("live model tests need LAT_AI_TRANSPORT=real (paid API calls are never implicit)")
    store = SecretStore(PROJECT_ROOT)
    missing = [n.value for n in names if not store.has(n)]
    if missing:
        pytest.skip(f"live model tests need {missing} in Keychain/env/.env")
    return Settings(), store


async def test_live_claude_adapter_smoke() -> None:
    settings, store = _require(SecretName.ANTHROPIC_API_KEY)
    from pydantic_ai import Agent

    from services.ai.adapters import build_claude_model

    model = build_claude_model(settings, store, transport="real", effort="low")
    result = await Agent(model, output_type=str).run("Reply with the single word: ready")
    assert "ready" in result.output.lower()
    assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0


async def test_live_sol_adapter_smoke() -> None:
    settings, store = _require(SecretName.OPENAI_API_KEY)
    from pydantic_ai import Agent

    from services.ai.adapters import build_sol_model

    model = build_sol_model(settings, store, transport="real", effort="low")
    result = await Agent(model, output_type=str).run("Reply with the single word: ready")
    assert "ready" in result.output.lower()


async def test_live_pipeline_returns_validated_proposal(tmp_path: Path) -> None:
    settings, store = _require(SecretName.ANTHROPIC_API_KEY, SecretName.OPENAI_API_KEY)
    from packages.core.clock import ManualClock
    from services.ai.pipeline import ResearchPipeline
    from tests.unit.ai.conftest import T0

    clock = ManualClock(T0)
    sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
    pipeline = ResearchPipeline(settings, store, transport="real", clock=clock, wall_clock=clock)
    try:
        r = await pipeline.run(causal_id="cz_live", candidate=sig, snapshot=snap,
                               evidence=build_internal_evidence(sig, snap, pol), policy=pol)
    finally:
        await pipeline.aclose()
    assert r.proposal.action.value in ("TRADE", "NO_TRADE")
    assert all(x.transport == ModelTransport.REAL for x in r.model_runs)
    assert r.model_runs and r.model_runs[0].status in tuple(ModelRunStatus)
