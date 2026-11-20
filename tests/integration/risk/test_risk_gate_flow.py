"""Risk gate flow on real local state: PolicyStore (SQLite) -> RiskEngine -> signed intent -> final gate with
the public key loaded from the namespace keys dir; an operator policy change invalidates outstanding intents;
and the duplicate-fingerprint contract holds against the real OrderLedger."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import TradingMode
from packages.contracts.policy import RiskPolicy
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.signing import IntentSigner, load_verifier
from services.risk.engine import RiskEngine
from services.risk.operator_auth import OperatorAuth
from services.risk.policy_store import PolicyStore
from tests.unit.risk.risk_factories import NOW, failed, make_ctx, make_proposal


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[Database, PolicyStore, IntentSigner, OperatorAuth]]:
    var = tmp_path / "var" / "paper"
    db = Database(var / "ledger.sqlite")
    auth = OperatorAuth(var / "keys")
    auth.ensure_token()
    signer = IntentSigner.load_or_create(var / "keys")
    store = PolicyStore(db, TradingMode.PAPER)
    store.ensure_default()
    yield db, store, signer, auth
    db.close()


def test_end_to_end_gate_with_keys_dir_verifier(env: tuple[Database, PolicyStore, IntentSigner, OperatorAuth]) -> None:
    db, store, signer, _ = env
    verifier = load_verifier(db.path.parent / "keys")  # what the execution boundary / LEAN holds
    assert verifier.key_id == signer.key_id
    engine = RiskEngine(store.active(), signer, ManualClock(NOW))
    ctx = make_ctx()
    proposal = make_proposal()
    decision = engine.evaluate(proposal, ctx)
    assert decision.approved, decision.reasons
    intent = engine.authorize(proposal, decision, ctx)
    assert intent.policy_version == store.active_version()
    assert engine.final_gate(intent, ctx, verifier, store.active_version(), self_recorded=False) == []


def test_operator_policy_change_invalidates_outstanding_intents(
    env: tuple[Database, PolicyStore, IntentSigner, OperatorAuth],
) -> None:
    _, store, signer, auth = env
    clock = ManualClock(NOW)
    engine = RiskEngine(store.active(), signer, clock)
    ctx = make_ctx()
    proposal = make_proposal()
    intent = engine.authorize(proposal, engine.evaluate(proposal, ctx), ctx)

    store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")),
                       operator_token=auth.token_path.read_text().strip(), actor="victor", reason="tighten")
    fails = engine.final_gate(intent, ctx, signer.verifier, store.active_version(), self_recorded=False)
    assert any("no longer active" in f for f in fails)

    rebuilt = RiskEngine(store.active(), signer, clock)  # what the runtime does after a policy change
    again = rebuilt.evaluate(proposal, ctx)
    assert not again.approved and "max_order_notional" in failed(again)


def test_duplicate_contract_with_real_order_ledger(env: tuple[Database, PolicyStore, IntentSigner, OperatorAuth]) -> None:
    ledger_mod = pytest.importorskip("services.order_ledger.ledger")
    db, store, signer, _ = env
    clock = ManualClock(NOW)
    ledger = ledger_mod.OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier)
    policy = store.active()
    engine = RiskEngine(policy, signer, clock)

    proposal = make_proposal()
    ctx = make_ctx()
    decision = engine.evaluate(proposal, ctx)
    intent = engine.authorize(proposal, decision, ctx)
    ledger.record_proposal(proposal)
    ledger.record_decision(decision)
    ledger.record_intent(intent)

    window = NOW - timedelta(seconds=policy.duplicate_window_seconds)
    gate_ctx = make_ctx(recent_intent_fingerprints=tuple(ledger.recent_fingerprints(window)),
                        open_orders=tuple(ledger.open_orders()))
    # the intent's own ledger row/fingerprint does not make it a duplicate of itself at the final gate
    assert engine.final_gate(intent, gate_ctx, signer.verifier, policy.version) == []

    # an identical second proposal is rejected as a duplicate (and as exceeding pending orders)
    second = make_proposal(proposal_id="prop_second", causal_id="cz_second")
    dup = engine.evaluate(second, gate_ctx)
    assert not dup.approved
    assert {"duplicate_order", "max_pending_orders"} <= failed(dup)
