"""Foundation tests: contracts reject floats/naive times, signing, modes, audit chain, DB discipline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from packages.contracts import (
    ApprovedOrderIntent,
    DecisionAction,
    GatewayKind,
    OrderType,
    Proposer,
    RiskPolicy,
    Side,
    TimeInForce,
    TradeProposal,
    TradingMode,
)
from packages.core import modes
from packages.core.audit import add_halt, append_audit, verify_chain
from packages.core.config import Settings
from packages.core.db import Database, _split_sql
from packages.core.ids import client_order_id, new_id, ulid
from packages.core.money import D, floor_shares, round_price
from packages.core.secrets import LiveSecretAccessDenied, SecretName, SecretStore
from packages.core.signing import IntentSigner

NOW = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)


def _intent_fields(signer: IntentSigner, **over: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        intent_id="int_1", client_order_id="lat-sim-abc", proposal_id="prop_1", decision_id="dec_1",
        causal_id="cz_1", mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, symbol="SPY", side=Side.BUY,
        quantity=Decimal("3"), order_type=OrderType.LIMIT, limit_price=Decimal("500.12"),
        time_in_force=TimeInForce.DAY, policy_version="pol_x", created_at=NOW,
        expires_at=NOW + timedelta(seconds=60), requires_human_approval=False, key_id=signer.key_id,
    )
    base.update(over)
    return base


def _signed(signer: IntentSigner, **over: object) -> ApprovedOrderIntent:
    fields = _intent_fields(signer, **over)
    msg = ApprovedOrderIntent.model_construct(**fields, signature="0" * 128).signing_string()  # type: ignore[arg-type]
    return ApprovedOrderIntent(**fields, signature=signer.sign(msg))  # type: ignore[arg-type]


class TestContracts:
    def test_float_money_rejected(self) -> None:
        signer = IntentSigner.generate()
        with pytest.raises(ValidationError, match="float"):
            _signed(signer, quantity=3.0)

    def test_naive_datetime_rejected(self) -> None:
        signer = IntentSigner.generate()
        with pytest.raises(ValidationError, match="naive"):
            _signed(signer, created_at=datetime(2026, 11, 5, 15, 0))

    def test_extended_hours_intent_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extended"):
            _signed(IntentSigner.generate(), extended_hours=True)

    def test_trade_proposal_requires_fields_and_evidence(self) -> None:
        with pytest.raises(ValidationError, match="missing required"):
            TradeProposal(
                proposal_id="p", causal_id="c", created_at=NOW, data_asof=NOW, action=DecisionAction.TRADE,
                proposer=Proposer.TEST, strategy_id="s", dataset_version="ds",
            )
        ok = TradeProposal(
            proposal_id="p", causal_id="c", created_at=NOW, data_asof=NOW, action=DecisionAction.NO_TRADE,
            no_trade_reason="nothing", proposer=Proposer.TEST, strategy_id="s", dataset_version="ds",
        )
        assert ok.action == DecisionAction.NO_TRADE

    def test_market_orders_not_supported(self) -> None:
        with pytest.raises(ValidationError, match="LIMIT"):
            TradeProposal(
                proposal_id="p", causal_id="c", created_at=NOW, data_asof=NOW, action=DecisionAction.TRADE,
                proposer=Proposer.TEST, strategy_id="s", dataset_version="ds", symbol="SPY", side=Side.BUY,
                quantity=Decimal(1), limit_price=Decimal(1), max_planned_loss=Decimal(1),
                order_type=OrderType.MARKET, evidence_ids=("e1",),
            )

    def test_decimal_json_round_trip_is_string(self) -> None:
        intent = _signed(IntentSigner.generate())
        dumped = intent.model_dump(mode="json")
        assert dumped["quantity"] == "3" and dumped["limit_price"] == "500.12"
        assert dumped["created_at"].endswith("Z")
        assert ApprovedOrderIntent.model_validate(dumped) == intent

    def test_policy_version_is_content_hash(self) -> None:
        a, b = RiskPolicy(), RiskPolicy(max_order_notional=Decimal("1000"))
        assert a.version == RiskPolicy().version
        assert a.version != b.version


class TestSigning:
    def test_sign_verify_and_tamper(self) -> None:
        signer = IntentSigner.generate()
        intent = _signed(signer)
        assert signer.verifier.verify(intent.signing_string(), intent.signature)
        tampered = intent.model_copy(update={"quantity": Decimal("30")})
        assert not signer.verifier.verify(tampered.signing_string(), intent.signature)

    def test_foreign_key_does_not_verify(self) -> None:
        intent = _signed(IntentSigner.generate())
        assert not IntentSigner.generate().verifier.verify(intent.signing_string(), intent.signature)

    def test_signing_string_rejects_delimiter_injection(self) -> None:
        signer = IntentSigner.generate()
        fields = _intent_fields(signer, policy_version="pol|quantity=999")
        with pytest.raises(ValueError, match="illegal"):
            ApprovedOrderIntent.model_construct(**fields, signature="0" * 128).signing_string()  # type: ignore[arg-type]

    def test_key_persistence_and_permissions(self, tmp_path: Path) -> None:
        s1 = IntentSigner.load_or_create(tmp_path / "keys")
        s2 = IntentSigner.load_or_create(tmp_path / "keys")
        assert s1.key_id == s2.key_id
        assert (tmp_path / "keys" / "intent_signing.pem").stat().st_mode & 0o077 == 0


class TestModes:
    @pytest.mark.parametrize("mode", list(TradingMode))
    def test_live_gateway_always_denied(self, mode: TradingMode) -> None:
        with pytest.raises(modes.LiveActivationDenied):
            modes.assert_gateway_allowed(mode, GatewayKind.LEAN_ALPACA_LIVE)

    def test_compiled_flags_off(self) -> None:
        assert modes.LIVE_EXECUTION_COMPILED is False
        assert modes.BOUNDED_LIVE_COMPILED is False

    def test_live_activation_failures_even_with_all_inputs(self) -> None:
        failures = modes.live_activation_failures(
            TradingMode.LIVE_APPROVAL, live_credentials_present=True, release_lock_enabled=True,
            broker_identity_verified=True,
        )
        assert any("LIVE_EXECUTION_COMPILED" in f for f in failures)

    def test_research_only_has_no_gateways(self) -> None:
        assert modes.allowed_gateways(TradingMode.RESEARCH_ONLY) == frozenset()
        with pytest.raises(modes.ModeViolation):
            modes.assert_gateway_allowed(TradingMode.RESEARCH_ONLY, GatewayKind.FAKE)

    def test_env_switch_to_live_mode_isolates_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAT_MODE", "LIVE_APPROVAL")
        s = Settings()
        assert s.mode == TradingMode.LIVE_APPROVAL and s.namespace == "live"
        assert "paper" not in str(s.db_path)
        assert modes.allowed_gateways(s.mode) == frozenset()

    def test_api_host_must_be_loopback(self) -> None:
        with pytest.raises(ValidationError):
            Settings(api_host="0.0.0.0")  # noqa: S104


class TestSecrets:
    def test_ambiguous_names_ignored_and_live_unreadable(self, tmp_path: Path) -> None:
        env = {"APCA_API_KEY_ID": "PKAMBIGUOUS", "ALPACA_LIVE_KEY_ID": "AKLIVE"}
        store = SecretStore(tmp_path, use_keychain=False, env=env, dotenv_path=tmp_path / "none")
        assert store.get(SecretName.ALPACA_PAPER_KEY_ID) is None
        assert store.ignored_ambiguous_present() == ["APCA_API_KEY_ID"]
        assert store.has(SecretName.ALPACA_LIVE_KEY_ID)
        with pytest.raises(LiveSecretAccessDenied):
            store.get(SecretName.ALPACA_LIVE_KEY_ID)

    def test_dotenv_parsing(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text('export ANTHROPIC_API_KEY="sk-ant-test-123"\n# c\n')
        store = SecretStore(tmp_path, use_keychain=False, env={})
        secret = store.get(SecretName.ANTHROPIC_API_KEY)
        assert secret is not None and secret.get_secret_value() == "sk-ant-test-123"
        assert "sk-ant" not in repr(secret)


class TestDatabase:
    def test_migrations_audit_chain_and_immutability(self, tmp_path: Path) -> None:
        with Database(tmp_path / "l.sqlite") as db:
            assert "0001_core.sql" in db.applied_migrations
            for i in range(5):
                db.write(lambda c, i=i: append_audit(c, event_type="T", source="t", mode="PAPER", payload={"i": i}))
            db.write(lambda c: add_halt(c, reason="r", source="t", mode="PAPER"))
            ok, rows, _ = verify_chain(db)
            assert ok and rows == 6
            with pytest.raises(Exception, match="append-only"):
                db.write(lambda c: c.execute("DELETE FROM audit_events"))
            pol = RiskPolicy()
            db.write(lambda c: c.execute(
                "INSERT INTO risk_policies(policy_version,name,policy_json,created_at,created_by,approved_by,is_active)"
                " VALUES (?,?,?,?,?,?,1)", (pol.version, pol.name, pol.canonical_json(), "t", "t", "t")))
            with pytest.raises(Exception, match="immutable"):
                db.write(lambda c: c.execute("UPDATE risk_policies SET policy_json='{}'"))

    def test_failed_write_rolls_back(self, tmp_path: Path) -> None:
        with Database(tmp_path / "l.sqlite") as db:
            def bad(c: object) -> None:
                append_audit(c, event_type="T", source="t", mode="PAPER", payload={})  # type: ignore[arg-type]
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError):
                db.write(bad)
            assert db.query_one("SELECT COUNT(*) n FROM audit_events") == {"n": 0}

    def test_reopen_does_not_reapply(self, tmp_path: Path) -> None:
        Database(tmp_path / "l.sqlite").close()
        with Database(tmp_path / "l.sqlite") as db:
            assert db.applied_migrations == []

    def test_split_sql_keeps_triggers(self) -> None:
        sql = "CREATE TABLE a(x);\nCREATE TRIGGER t BEFORE UPDATE ON a\nBEGIN SELECT RAISE(ABORT,'x'); END;\n"
        assert len(_split_sql(sql)) == 2


class TestIdsAndMoney:
    def test_ids_unique_sortable(self) -> None:
        ids = [ulid() for _ in range(2000)]
        assert len(set(ids)) == 2000 and ids == sorted(ids)
        assert new_id("prop").startswith("prop_")
        assert client_order_id("paper").startswith("lat-paper-") and len(client_order_id("sim")) <= 128

    def test_money_rejects_float(self) -> None:
        with pytest.raises(TypeError):
            D(1.5)
        assert D("1.50") == Decimal("1.50")

    @given(st.decimals(min_value=Decimal("0.0001"), max_value=Decimal("100000"), places=6))
    def test_round_price_tick_and_side(self, p: Decimal) -> None:
        buy, sell = round_price(p, side_buy=True), round_price(p, side_buy=False)
        assert buy <= p <= sell
        tick = Decimal("0.01") if p >= 1 else Decimal("0.0001")
        assert (sell - buy) <= tick

    def test_floor_shares(self) -> None:
        assert floor_shares(Decimal("3.99")) == Decimal(3)
