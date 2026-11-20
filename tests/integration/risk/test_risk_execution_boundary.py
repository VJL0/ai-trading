"""The risk final gate as the real order boundary (SIMULATED: deterministic in-process FakeBroker, no network).

Wiring mirrors the runtime: PolicyStore (SQLite) -> RiskEngine.evaluate/authorize -> OrderLedger.record_intent
-> fresh RiskContext from broker + ledger + EntriesLock -> ExecutionService.submit, whose final-gate callback is
``RiskEngine.final_gate``. Each test proves that an intent the gate refuses never reaches the broker
(``FakeBroker.submission_count == 0``), and that a valid one is submitted exactly once.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, OrderState, ReconciliationStatus, TradingMode
from packages.contracts.models import ApprovedOrderIntent, RiskContext
from packages.contracts.policy import RiskPolicy
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.signing import IntentSigner
from services.risk.context import build_risk_context, build_system_health
from services.risk.engine import RiskEngine
from services.risk.operator_auth import OperatorAuth
from services.risk.policy_store import PolicyStore
from tests.unit.risk.risk_factories import NOW, failed, make_market, make_proposal, make_quote

execution_mod = pytest.importorskip("services.order_ledger.execution")
entries_mod = pytest.importorskip("services.order_ledger.entries")
fake_mod = pytest.importorskip("services.order_ledger.fake_broker")
ledger_mod = pytest.importorskip("services.order_ledger.ledger")
reconcile_mod = pytest.importorskip("services.order_ledger.reconcile")
errors_mod = pytest.importorskip("services.order_ledger.errors")


@dataclass
class Rig:
    db: Database
    clock: ManualClock
    store: PolicyStore
    auth: OperatorAuth
    signer: IntentSigner
    engine: RiskEngine
    ledger: Any
    entries: Any
    broker: Any
    execution: Any
    gate_ctx: list[RiskContext]
    gate_results: list[list[str]]

    async def fresh_ctx(self) -> RiskContext:
        """What the runtime captures right before submission (``prepare_gate``)."""
        now = self.clock.now()
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        locked, reasons = self.entries.is_locked()
        health = build_system_health(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True,
                                     entries_locked=locked, lock_reasons=reasons)
        policy = self.store.active()
        return build_risk_context(
            now=now, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE, account=account, positions=positions,
            open_orders=self.ledger.open_orders(), market=make_market(now=now), health=health,
            recent_submission_times=self.ledger.recent_submission_times(now - timedelta(days=1)),
            recent_intent_fingerprints=self.ledger.recent_fingerprints(
                now - timedelta(seconds=policy.duplicate_window_seconds)),
            start_of_day_equity=account.equity,
        )

    def final_gate(self, intent: ApprovedOrderIntent) -> list[str]:
        ctx = self.gate_ctx.pop() if self.gate_ctx else None
        if ctx is None:
            return ["no fresh risk context captured for the final gate"]
        result = self.engine.final_gate(intent, ctx, self.signer.verifier, self.store.active_version())
        self.gate_results.append(result)
        return result

    async def approve_and_record(self, **proposal_kw: Any) -> ApprovedOrderIntent:
        proposal = make_proposal(**proposal_kw)
        ctx = await self.fresh_ctx()
        decision = self.engine.evaluate(proposal, ctx)
        assert decision.approved, decision.reasons
        intent = self.engine.authorize(proposal, decision, ctx)
        self.ledger.record_proposal(proposal)
        self.ledger.record_decision(decision)
        self.ledger.record_intent(intent)
        return intent

    async def submit(self, intent: ApprovedOrderIntent) -> Any:
        self.gate_ctx.append(await self.fresh_ctx())
        return await self.execution.submit(intent)


@pytest.fixture
async def rig(tmp_path: Path) -> AsyncIterator[Rig]:
    var = tmp_path / "var" / "paper"
    db = Database(var / "ledger.sqlite")
    clock = ManualClock(NOW)
    auth = OperatorAuth(var / "keys")
    auth.ensure_token()
    signer = IntentSigner.load_or_create(var / "keys")
    store = PolicyStore(db, TradingMode.PAPER, clock=clock)
    store.ensure_default()
    ledger = ledger_mod.OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier)
    entries = entries_mod.EntriesLock(db, TradingMode.PAPER, clock=clock)
    quotes = fake_mod.QuoteBook()
    quotes.set(make_quote("SPY", now=NOW, exchange_age_s=0, received_age_s=0))
    broker = fake_mod.FakeBroker(clock, quotes, starting_cash=Decimal("10000"), verifier=signer.verifier)
    reconciler = reconcile_mod.Reconciler(db, ledger, broker, entries, clock, confirm_delay_s=0)
    report = await reconciler.startup_recovery()
    assert report.status == ReconciliationStatus.CLEAN, report.detail
    assert entries.is_locked() == (False, [])
    r = Rig(db, clock, store, auth, signer, RiskEngine(store.active(), signer, clock), ledger, entries, broker,
            None, [], [])
    r.execution = execution_mod.ExecutionService(ledger, broker, r.final_gate, clock, entries_lock=entries)
    yield r
    db.close()


async def test_approved_intent_passes_the_gate_and_reaches_the_broker_once(rig: Rig) -> None:
    intent = await rig.approve_and_record()
    order = await rig.submit(intent)
    assert rig.gate_results == [[]]
    assert order is not None
    assert rig.broker.submission_count(intent.client_order_id) == 1
    state = rig.ledger.get_order(intent.client_order_id).state
    assert state in (OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED)
    # a retried submit of the same intent is refused by the ledger state machine: never a second order
    assert await rig.submit(intent) is None
    assert rig.broker.submission_count(intent.client_order_id) == 1


async def test_operator_policy_change_after_authorization_blocks_at_the_boundary(rig: Rig) -> None:
    intent = await rig.approve_and_record()
    rig.store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")),
                           operator_token=rig.auth.token_path.read_text().strip(), actor="victor", reason="tighten")
    assert await rig.submit(intent) is None
    assert any("no longer active" in f for f in rig.gate_results[-1])
    assert rig.broker.submission_count(intent.client_order_id) == 0
    assert rig.ledger.get_order(intent.client_order_id).state == OrderState.REJECTED


async def test_halt_after_authorization_is_refused_by_the_risk_gate_itself(rig: Rig) -> None:
    intent = await rig.approve_and_record()
    rig.entries.lock("operator pressed stop-new-orders", "test")
    ctx = await rig.fresh_ctx()
    gate = rig.engine.final_gate(intent, ctx, rig.signer.verifier, rig.store.active_version())
    assert any(f.startswith("health: entries locked") and "operator pressed stop-new-orders" in f for f in gate)
    assert await rig.submit(intent) is None
    assert rig.broker.submission_count(intent.client_order_id) == 0


async def test_expired_intent_never_reaches_the_broker(rig: Rig) -> None:
    intent = await rig.approve_and_record()
    rig.clock.advance(RiskPolicy().intent_ttl_seconds + 1)
    assert await rig.submit(intent) is None
    assert rig.broker.submission_count(intent.client_order_id) == 0


async def test_forged_intent_in_a_misconfigured_ledger_is_stopped_by_the_gate(rig: Rig) -> None:
    good = await rig.approve_and_record()
    foreign = IntentSigner.generate()
    unsigned = good.model_copy(update={"intent_id": good.intent_id + "F", "client_order_id": good.client_order_id + "f",
                                       "quantity": Decimal("4"), "key_id": foreign.key_id})
    forged = ApprovedOrderIntent.model_validate(
        {**unsigned.model_dump(), "signature": foreign.sign(unsigned.signing_string())})
    # the ledger configured with the risk public key refuses it outright ...
    with pytest.raises(errors_mod.IntentRefused):
        rig.ledger.record_intent(forged)
    # ... and a ledger *without* a verifier (misconfiguration) records it, but the risk gate stops it
    rig.ledger.verifier = None
    rig.ledger.record_intent(forged)
    assert await rig.submit(forged) is None
    assert any("signature" in f for f in rig.gate_results[-1])
    assert rig.broker.submission_count(forged.client_order_id) == 0


async def test_identical_follow_up_proposal_is_a_duplicate(rig: Rig) -> None:
    first = await rig.approve_and_record()
    await rig.submit(first)
    ctx = await rig.fresh_ctx()
    again = rig.engine.evaluate(make_proposal(proposal_id="prop_again", causal_id="cz_again"), ctx)
    assert not again.approved
    assert "duplicate_order" in failed(again)
