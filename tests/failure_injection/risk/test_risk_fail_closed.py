"""Failure injection: every internal error, missing input or corrupted state ends in REJECTED / refusal,
never in an approval or a signed intent (spec invariant 4: fail closed)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import TradingMode
from packages.contracts.models import RiskContext, TradeProposal
from packages.contracts.policy import RiskPolicy
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.signing import IntentSigner
from services.risk import market_hours
from services.risk import policy_store as ps
from services.risk.engine import RiskAuthorizationError, RiskEngine
from services.risk.operator_auth import OperatorAuth, OperatorAuthError
from services.risk.policy_store import PolicyStore
from tests.unit.risk.risk_factories import NOW, SIGNER, failed, make_ctx, make_engine, make_proposal


class BrokenClock:
    def now(self) -> datetime:
        raise RuntimeError("clock unavailable")


class NaiveClock:
    def now(self) -> datetime:
        return datetime(2026, 11, 5, 15, 0)


@pytest.mark.parametrize("clock", [BrokenClock(), NaiveClock()])
def test_bad_clock_rejects_without_raising(clock: Any) -> None:
    engine = RiskEngine(RiskPolicy(), SIGNER, clock)
    decision = engine.evaluate(make_proposal(), make_ctx())
    assert not decision.approved
    assert decision.checks[0].name == "evaluation_error"


def test_broken_calendar_fails_session_check_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine()

    def boom() -> Any:
        raise OSError("calendar data unavailable")

    monkeypatch.setattr(market_hours, "_calendar", boom)
    decision = engine.evaluate(make_proposal(), make_ctx())
    assert not decision.approved and "regular_session" in failed(decision)


def test_garbage_context_is_rejected() -> None:
    garbage = RiskContext.model_construct(**{**make_ctx().model_dump(), "health": None, "account": None})
    decision = make_engine().evaluate(make_proposal(), garbage)
    assert not decision.approved


def test_float_smuggled_into_proposal_is_rejected() -> None:
    smuggled = TradeProposal.model_construct(**{**make_proposal().model_dump(), "quantity": 3.0})
    decision = make_engine().evaluate(smuggled, make_ctx())
    assert not decision.approved


def test_float_smuggled_into_policy_blocks_everything() -> None:
    smuggled = RiskPolicy.model_construct(**{**RiskPolicy().model_dump(), "max_order_notional": 2000.0})
    engine = RiskEngine(smuggled, SIGNER, ManualClock(NOW))
    decision = engine.evaluate(make_proposal(), make_ctx())
    assert not decision.approved and "policy_valid" in failed(decision)


class ExplodingSigner(IntentSigner):
    def sign(self, message: str) -> str:
        raise RuntimeError("HSM unavailable")


class LyingSigner(IntentSigner):
    def sign(self, message: str) -> str:
        return "0" * 128


@pytest.mark.parametrize("signer_cls", [ExplodingSigner, LyingSigner])
def test_signing_failures_never_produce_an_intent(signer_cls: type[IntentSigner]) -> None:
    from cryptography.hazmat.primitives.asymmetric import ec

    signer = signer_cls(ec.generate_private_key(ec.SECP256R1()))
    engine = make_engine(signer=signer)
    ctx = make_ctx()
    p = make_proposal()
    decision = engine.evaluate(p, ctx)
    assert decision.approved
    with pytest.raises((RuntimeError, RiskAuthorizationError)):
        engine.authorize(p, decision, ctx)


def test_final_gate_with_raising_verifier_fails_closed() -> None:
    engine = make_engine()
    ctx = make_ctx()
    p = make_proposal()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)

    class BadVerifier:
        key_id = SIGNER.key_id

        def verify(self, message: str, signature_hex: str) -> bool:
            raise RuntimeError("verifier crashed")

    fails = engine.final_gate(intent, ctx, BadVerifier(), engine.policy_version)  # type: ignore[arg-type]
    assert fails and "fail closed" in fails[0]


def test_human_approval_verifier_crash_fails_closed() -> None:
    engine = make_engine(RiskPolicy(require_human_approval=True))
    ctx = make_ctx()
    p = make_proposal()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    from packages.contracts.models import HumanApproval

    approval = HumanApproval(approval_id="hap_x", intent_id=intent.intent_id, approver="v", confirmed_at=NOW,
                             expires_at=NOW.replace(hour=16), statement="ok", signature="a" * 64)

    def crash(*_a: Any) -> bool:
        raise RuntimeError("boom")

    fails = engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy_version, self_recorded=False,
                              human_approval=approval, approval_verifier=crash)
    assert any("verification error" in f for f in fails)


# ------------------------------------------------------------------------------------------ policy store


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "ledger.sqlite")
    yield database
    database.close()


def test_audit_failure_rolls_back_the_whole_policy_change(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    auth = OperatorAuth(db.path.parent / "keys")
    auth.ensure_token()
    store = PolicyStore(db, TradingMode.PAPER)
    store.ensure_default()
    before = store.active_version()
    real_append = ps.append_audit

    def failing_append(conn: sqlite3.Connection, **kw: Any) -> str:
        if kw.get("event_type") == "POLICY_CHANGED":
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return real_append(conn, **kw)

    monkeypatch.setattr(ps, "append_audit", failing_append)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")),
                           operator_token=auth.token_path.read_text().strip(), actor="victor", reason="x")
    assert store.active_version() == before
    assert len(db.query("SELECT * FROM risk_policies")) == 1  # the new version insert was rolled back too
    assert db.query("SELECT * FROM risk_policy_changes WHERE result='ACCEPTED'") == []


def test_missing_operator_token_file_blocks_changes(db: Database) -> None:
    store = PolicyStore(db, TradingMode.PAPER)
    store.ensure_default()
    with pytest.raises(OperatorAuthError):
        store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")),
                           operator_token="x" * 43, actor="victor", reason="no token file exists")


def test_engine_refuses_to_run_on_unknown_policy_state(db: Database) -> None:
    store = PolicyStore(db, TradingMode.PAPER)
    with pytest.raises(ps.PolicyUnavailableError):
        store.active()  # nothing installed => the runtime cannot construct a RiskEngine at all


@pytest.mark.parametrize("args", [("proposal", None), (None, "ctx"), (None, None), ("garbage", 42)])
def test_evaluate_never_raises_on_non_contract_inputs(args: tuple[Any, Any]) -> None:
    proposal = make_proposal() if args[0] == "proposal" else args[0]
    ctx = make_ctx() if args[1] == "ctx" else args[1]
    engine = make_engine()
    decision = engine.evaluate(proposal, ctx)
    assert not decision.approved and decision.checks[0].name == "evaluation_error"
    with pytest.raises(Exception):  # noqa: B017 - any refusal is fine, an intent is not
        engine.authorize(proposal, decision, ctx)
    assert engine.final_gate(proposal, ctx, SIGNER.verifier, engine.policy_version)  # pyright: ignore[reportArgumentType]
