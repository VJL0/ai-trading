"""Operating-mode tests (spec §8, acceptance 12): RESEARCH_ONLY never trades, REPLAY only with the FAKE
gateway, and LIVE_APPROVAL / BOUNDED_LIVE are always denied in this build - including when the mode is
switched through the ``LAT_MODE`` / ``LAT_GATEWAY`` environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.contracts.enums import DataFeed, GatewayKind, TradingMode
from packages.contracts.policy import RiskPolicy
from packages.core import modes
from packages.core.config import load_settings
from packages.core.signing import IntentSigner
from services.risk.engine import RiskAuthorizationError
from tests.unit.risk.risk_factories import SIGNER, failed, make_ctx, make_engine, make_market, make_proposal

ALLOWED_PAIRS = {
    (TradingMode.REPLAY, GatewayKind.FAKE),
    (TradingMode.PAPER, GatewayKind.FAKE),
    (TradingMode.PAPER, GatewayKind.ALPACA_PAPER),
    (TradingMode.PAPER, GatewayKind.LEAN_PAPER_SIM),
    (TradingMode.PAPER, GatewayKind.LEAN_ALPACA_PAPER),
}


@pytest.mark.parametrize("mode", list(TradingMode))
@pytest.mark.parametrize("gateway", list(GatewayKind))
def test_mode_gateway_matrix(mode: TradingMode, gateway: GatewayKind) -> None:
    feed = DataFeed.IEX if gateway in (GatewayKind.LEAN_ALPACA_PAPER, GatewayKind.ALPACA_PAPER) else DataFeed.SYNTHETIC
    ctx = make_ctx(mode=mode, gateway=gateway, market=make_market(feed=feed))
    engine = make_engine()
    proposal = make_proposal()
    decision = engine.evaluate(proposal, ctx)
    assert decision.approved == ((mode, gateway) in ALLOWED_PAIRS), decision.reasons
    if not decision.approved:
        assert "mode_permits_trading" in failed(decision)
        with pytest.raises(RiskAuthorizationError):
            engine.authorize(proposal, decision, ctx)


@pytest.mark.parametrize(
    ("env_mode", "env_gateway"),
    [
        ("LIVE_APPROVAL", None),
        ("LIVE_APPROVAL", "LEAN_ALPACA_LIVE"),
        ("BOUNDED_LIVE", None),
        ("BOUNDED_LIVE", "LEAN_ALPACA_LIVE"),
        ("PAPER", "LEAN_ALPACA_LIVE"),
        ("RESEARCH_ONLY", None),
        ("REPLAY", "LEAN_ALPACA_PAPER"),
    ],
)
def test_env_var_mode_switch_cannot_enable_trading(
    monkeypatch: pytest.MonkeyPatch, env_mode: str, env_gateway: str | None
) -> None:
    monkeypatch.setenv("LAT_MODE", env_mode)
    if env_gateway is None:
        monkeypatch.delenv("LAT_GATEWAY", raising=False)
    else:
        monkeypatch.setenv("LAT_GATEWAY", env_gateway)
    settings = load_settings()
    assert settings.mode.value == env_mode
    ctx = make_ctx(mode=settings.mode, gateway=settings.gateway, market=make_market(feed=DataFeed.IEX))
    engine = make_engine()
    proposal = make_proposal()
    decision = engine.evaluate(proposal, ctx)
    assert not decision.approved
    assert "mode_permits_trading" in failed(decision)
    with pytest.raises(RiskAuthorizationError):
        engine.authorize(proposal, decision, ctx)


def test_env_switch_to_live_uses_a_separate_namespace_and_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LAT_MODE", "PAPER")
    paper = load_settings(project_root=tmp_path)
    monkeypatch.setenv("LAT_MODE", "LIVE_APPROVAL")
    live = load_settings(project_root=tmp_path)
    assert paper.keys_dir != live.keys_dir and paper.db_path != live.db_path
    paper_signer = IntentSigner.load_or_create(paper.keys_dir)
    live_signer = IntentSigner.load_or_create(live.keys_dir)
    assert paper_signer.key_id != live_signer.key_id
    # an intent signed with the paper key never verifies under the live namespace key
    engine = make_engine(signer=paper_signer)
    ctx = make_ctx()
    p = make_proposal()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    assert not live_signer.verifier.verify(intent.signing_string(), intent.signature)


def test_live_flags_are_compiled_off() -> None:
    assert modes.LIVE_EXECUTION_COMPILED is False
    assert modes.BOUNDED_LIVE_COMPILED is False
    failures = modes.live_activation_failures(
        TradingMode.LIVE_APPROVAL, live_credentials_present=True, release_lock_enabled=True,
        broker_identity_verified=True,
    )
    assert failures  # even with every runtime condition claimed, the compiled flag denies


def test_human_approval_forced_in_live_approval_is_moot_because_live_is_denied() -> None:
    engine = make_engine(RiskPolicy(require_human_approval=False))
    ctx = make_ctx(mode=TradingMode.LIVE_APPROVAL, gateway=GatewayKind.LEAN_ALPACA_LIVE)
    decision = engine.evaluate(make_proposal(), ctx)
    assert not decision.approved
    assert decision.mode == TradingMode.LIVE_APPROVAL


def test_engine_signer_matches_test_signer() -> None:
    assert make_engine().key_id == SIGNER.key_id
