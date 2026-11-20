"""Gateway factory + live release gate: every live path is denied (env switch, CLI flag, model output, GUI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, Side, TradingMode
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.modes import LIVE_EXECUTION_COMPILED, LiveActivationDenied, ModeViolation
from packages.core.secrets import LiveSecretAccessDenied, SecretName, SecretStore
from packages.core.signing import IntentSigner
from services.order_ledger.alpaca_gateway import AlpacaPaperGateway
from services.order_ledger.alpaca_reader import AlpacaPaperStateReader
from services.order_ledger.errors import MissingPaperCredentials
from services.order_ledger.fake_broker import FakeBroker
from services.order_ledger.gateway_factory import ResearchOnlyGateway, build_gateway
from services.order_ledger.ledger import OrderLedger
from services.order_ledger.live_gate import LIVE_REQUIREMENTS, LiveReleaseGate, main
from tests.unit.order_ledger.ol_support import Harness

FAKE_PAPER = {"ALPACA_PAPER_KEY_ID": "PKTESTONLY0000000000", "ALPACA_PAPER_SECRET_KEY": "test-secret-not-real"}
FAKE_LIVE = {"ALPACA_LIVE_KEY_ID": "AKTESTONLY0000000000", "ALPACA_LIVE_SECRET_KEY": "test-live-secret-not-real"}


def _secrets(env: dict[str, str] | None = None) -> SecretStore:
    return SecretStore(Path("/nonexistent"), use_keychain=False, env=env or {}, dotenv_path=Path("/nonexistent/.env"))


def _settings(tmp: Path, **kw: Any) -> Settings:
    return Settings(project_root=tmp, use_keychain=False, **kw)


def _decisions(db: Database) -> list[dict[str, Any]]:
    return db.query("SELECT to_mode, actor, result, detail FROM mode_transitions ORDER BY seq")


def test_fake_gateway(h: Harness, tmp_path: Path) -> None:
    settings = _settings(tmp_path, gateway=GatewayKind.FAKE)
    signer = IntentSigner.load_or_create(settings.keys_dir)
    gw = build_gateway(settings, _secrets(), ledger=h.ledger, clock=h.clock, quote_source=h.quotes)
    assert isinstance(gw, FakeBroker)
    assert gw.state_path == tmp_path / "var" / "paper" / "fake_broker_state.json"
    assert gw.verifier is not None and gw.verifier.key_id == signer.key_id
    [row] = _decisions(h.db)
    assert row["result"] == "ACCEPTED" and "SIMULATED" in row["detail"]
    assert h.db.query_one("SELECT COUNT(*) n FROM audit_events WHERE event_type='MODE_ACCEPTED'")["n"] == 1  # type: ignore[index]


def test_fake_gateway_without_verification_key_fails_closed(h: Harness, tmp_path: Path) -> None:
    with pytest.raises(ModeViolation, match="no intent verification key"):
        build_gateway(_settings(tmp_path, gateway=GatewayKind.FAKE), _secrets(), ledger=h.ledger, clock=h.clock,
                      quote_source=h.quotes)
    [row] = _decisions(h.db)
    assert row["result"] == "DENIED"


@pytest.mark.parametrize("kind", [GatewayKind.LEAN_PAPER_SIM, GatewayKind.LEAN_ALPACA_PAPER])
def test_retired_lean_execution_is_refused(h: Harness, tmp_path: Path, kind: GatewayKind) -> None:
    with pytest.raises(ModeViolation, match="LEAN execution retired"):
        build_gateway(_settings(tmp_path, gateway=kind), _secrets(), ledger=h.ledger,
                      clock=h.clock, quote_source=None)


def test_sdk_paper_requires_keys_and_signature_verifier(h: Harness, tmp_path: Path) -> None:
    settings = _settings(tmp_path, gateway=GatewayKind.ALPACA_PAPER)
    with pytest.raises(MissingPaperCredentials, match="ALPACA_PAPER_KEY_ID"):
        build_gateway(settings, _secrets(), ledger=h.ledger, clock=h.clock, quote_source=None)
    with pytest.raises(ModeViolation, match="verification key"):
        build_gateway(settings, _secrets(FAKE_PAPER), ledger=h.ledger, clock=h.clock, quote_source=None)
    gw = build_gateway(settings, _secrets(FAKE_PAPER), ledger=h.ledger, clock=h.clock, quote_source=None,
                       verifier=h.signer.verifier)
    assert isinstance(gw, AlpacaPaperGateway)


def test_research_only_has_no_execution_path(tmp_path: Path) -> None:
    db = Database(tmp_path / "r.sqlite")
    try:
        from packages.core.clock import ManualClock
        from tests.unit.order_ledger.ol_support import T0

        ledger = OrderLedger(db, ManualClock(T0), TradingMode.RESEARCH_ONLY)
        gw = build_gateway(_settings(tmp_path, mode=TradingMode.RESEARCH_ONLY), _secrets(), ledger=ledger,
                           clock=ledger.clock, quote_source=None)
        assert isinstance(gw, ResearchOnlyGateway)
    finally:
        db.close()


def test_ledger_mode_must_match_settings(h: Harness, tmp_path: Path) -> None:
    with pytest.raises(ModeViolation):
        build_gateway(_settings(tmp_path, mode=TradingMode.REPLAY), _secrets(), ledger=h.ledger, clock=h.clock,
                      quote_source=None)


@pytest.mark.parametrize(
    ("mode", "gateway"),
    [
        (TradingMode.PAPER, GatewayKind.LEAN_ALPACA_LIVE),
        (TradingMode.LIVE_APPROVAL, GatewayKind.LEAN_ALPACA_LIVE),
        (TradingMode.LIVE_APPROVAL, GatewayKind.FAKE),
        (TradingMode.BOUNDED_LIVE, GatewayKind.LEAN_ALPACA_LIVE),
        (TradingMode.BOUNDED_LIVE, GatewayKind.LEAN_ALPACA_PAPER),
    ],
)
def test_every_live_request_is_denied_with_all_requirements(h: Harness, tmp_path: Path, mode: TradingMode,
                                                            gateway: GatewayKind) -> None:
    settings = _settings(tmp_path, mode=mode, gateway=gateway)
    # even with (fake) live credentials present, activation is denied
    with pytest.raises(LiveActivationDenied) as info:
        build_gateway(settings, _secrets({**FAKE_PAPER, **FAKE_LIVE}), ledger=h.ledger, clock=h.clock, quote_source=None)
    text = "; ".join(info.value.failed_requirements)
    assert "LIVE_EXECUTION_COMPILED is False" in text
    assert "release lock" in text
    assert "identity" in text
    assert "per order" in text
    if mode == TradingMode.BOUNDED_LIVE:
        assert "BOUNDED_LIVE" in text
    row = _decisions(h.db)[-1]
    assert row["result"] == "DENIED" and row["to_mode"] == mode.value
    assert h.db.query_one("SELECT COUNT(*) n FROM audit_events WHERE event_type='MODE_DENIED'")["n"] >= 1  # type: ignore[index]


def test_live_credentials_absent_is_listed(h: Harness, tmp_path: Path) -> None:
    with pytest.raises(LiveActivationDenied) as info:
        build_gateway(_settings(tmp_path, mode=TradingMode.LIVE_APPROVAL, gateway=GatewayKind.LEAN_ALPACA_LIVE),
                      _secrets(), ledger=h.ledger, clock=h.clock, quote_source=None)
    assert any("live credentials are absent" in f for f in info.value.failed_requirements)


def test_lat_mode_environment_switch_cannot_enable_live(h: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAT_MODE", "LIVE_APPROVAL")
    monkeypatch.setenv("LAT_GATEWAY", "LEAN_ALPACA_LIVE")
    settings = Settings(project_root=tmp_path, use_keychain=False)
    assert settings.mode == TradingMode.LIVE_APPROVAL
    assert settings.namespace == "live"  # isolated from paper state
    with pytest.raises(LiveActivationDenied):
        build_gateway(settings, _secrets(FAKE_PAPER), ledger=h.ledger, clock=h.clock, quote_source=None)
    # flipping only the gateway variable in PAPER mode is denied too
    monkeypatch.setenv("LAT_MODE", "PAPER")
    with pytest.raises(LiveActivationDenied):
        build_gateway(Settings(project_root=tmp_path, use_keychain=False), _secrets(FAKE_PAPER), ledger=h.ledger,
                      clock=h.clock, quote_source=None)
    assert LIVE_EXECUTION_COMPILED is False


def test_cli_flag_is_denied_and_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path))
    assert main(["--activate"]) == 3
    assert "DENIED" in capsys.readouterr().err
    assert main(["--activate", "--mode", "BOUNDED_LIVE"]) == 3
    assert main([]) == 3
    assert "UNMET" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["model_output", "gui_request", "api_body", "prompt"])
def test_model_gui_and_api_requests_are_denied_and_audited(h: Harness, tmp_path: Path, source: str) -> None:
    gate = LiveReleaseGate(_settings(tmp_path), _secrets({**FAKE_PAPER, **FAKE_LIVE}), db=h.db, clock=h.clock)
    request: dict[str, object] = {"mode": "BOUNDED_LIVE", "reason": "the model says it is confident", "approved": True,
                                  "release_lock": "signed-by-operator", "broker_identity_verified": True}
    with pytest.raises(LiveActivationDenied) as info:
        gate.request_from(source, request)
    assert "LiveReleaseGate.activate() is non-executable in this build" in info.value.failed_requirements
    row = _decisions(h.db)[-1]
    assert row["result"] == "DENIED" and row["actor"] == f"live_gate:{source}" and row["to_mode"] == "BOUNDED_LIVE"


def test_release_lock_file_alone_never_counts(h: Harness, tmp_path: Path) -> None:
    lock = tmp_path / "var" / "live" / "keys" / "release.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("enabled by someone")
    gate = LiveReleaseGate(_settings(tmp_path), _secrets(FAKE_LIVE))
    assert any("release lock" in f for f in gate.evaluate(TradingMode.LIVE_APPROVAL))
    assert {r.key for r in LIVE_REQUIREMENTS} == {"compiled", "live_credentials", "release_lock", "broker_identity",
                                                  "per_order_confirmation", "independent_risk"}


def test_model_produced_live_intent_is_refused_everywhere(h: Harness) -> None:
    """A model (or anything else) producing a live intent can't get it recorded, queued or executed."""
    live = h.factory.make(record=False, mode=TradingMode.PAPER, gateway=GatewayKind.LEAN_ALPACA_LIVE, side=Side.BUY)
    with pytest.raises((LiveActivationDenied, ModeViolation)):
        h.ledger.record_intent(live)
    live_mode = OrderLedger(h.db, h.clock, TradingMode.LIVE_APPROVAL)
    with pytest.raises((LiveActivationDenied, ModeViolation)):
        live_mode.record_intent(h.factory.make(record=False, mode=TradingMode.LIVE_APPROVAL,
                                               gateway=GatewayKind.LEAN_ALPACA_LIVE))
    assert h.db.query_one("SELECT COUNT(*) n FROM orders")["n"] == 0  # type: ignore[index]


def test_live_secrets_are_unreadable_and_paper_keys_never_go_live(tmp_path: Path) -> None:
    store = _secrets({**FAKE_PAPER, **FAKE_LIVE, "APCA_API_KEY_ID": "AKAMBIGUOUS"})
    with pytest.raises(LiveSecretAccessDenied):
        store.get(SecretName.ALPACA_LIVE_KEY_ID)
    reader = AlpacaPaperStateReader(store)
    client = reader._api._client
    assert getattr(client._base_url, "value", client._base_url) == "https://paper-api.alpaca.markets"


async def test_factory_fake_broker_only_executes_intents_claimed_through_the_ledger(h: Harness, tmp_path: Path) -> None:
    """Gateway ownership: a validly signed intent handed to the broker object directly (bypassing the ledger claim,
    i.e. the ExecutionService and its final gate) is refused; once claimed through the ledger it executes."""
    from packages.contracts.interfaces import GatewayRejected
    from tests.unit.order_ledger.ol_support import claim

    gw = build_gateway(_settings(tmp_path, gateway=GatewayKind.FAKE), _secrets(), ledger=h.ledger, clock=h.clock,
                       quote_source=h.quotes, verifier=h.signer.verifier)
    assert isinstance(gw, FakeBroker)
    intent = h.factory.make(limit="99.00")
    with pytest.raises(GatewayRejected, match="not claimed"):
        await gw.submit(intent)
    unrecorded = h.factory.make(limit="99.00", record=False)
    with pytest.raises(GatewayRejected, match="not claimed"):
        await gw.submit(unrecorded)
    assert gw.accepted_submissions == []
    order = await gw.submit(claim(h, intent))
    assert order.client_order_id == intent.client_order_id and gw.accepted_submissions == [intent.client_order_id]
