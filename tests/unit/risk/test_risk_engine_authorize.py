"""authorize() signs only APPROVED decisions this engine issued; final_gate() refuses tampered, expired,
foreign-key, live-mode, stale-policy, unhealthy, duplicate and unapproved intents at the order boundary."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import DataFeed, GatewayKind, RiskVerdict, Side, TradingMode
from packages.contracts.models import ApprovedOrderIntent, RiskContext
from packages.contracts.policy import RiskPolicy
from packages.core import modes
from packages.core.clock import ManualClock
from packages.core.signing import IntentSigner
from services.risk.context import fingerprint
from services.risk.engine import RiskAuthorizationError, RiskEngine
from services.risk.operator_auth import OperatorAuth
from tests.unit.risk.risk_factories import (
    AFTER_HOURS,
    NOW,
    SIGNER,
    d,
    healthy,
    make_ctx,
    make_engine,
    make_market,
    make_no_trade,
    make_order,
    make_position,
    make_proposal,
)


def approved(engine: RiskEngine | None = None, **ctx_kw: Any) -> tuple[RiskEngine, Any, Any, RiskContext]:
    engine = engine or make_engine()
    proposal = make_proposal()
    ctx = make_ctx(**ctx_kw)
    decision = engine.evaluate(proposal, ctx)
    assert decision.approved, decision.reasons
    return engine, proposal, decision, ctx


def signed(engine: RiskEngine | None = None, **ctx_kw: Any) -> tuple[RiskEngine, ApprovedOrderIntent, RiskContext]:
    engine, proposal, decision, ctx = approved(engine, **ctx_kw)
    return engine, engine.authorize(proposal, decision, ctx), ctx


def resign(intent: ApprovedOrderIntent, signer: IntentSigner, **update: Any) -> ApprovedOrderIntent:
    """Re-sign a modified intent with ``signer`` (simulates a stolen or foreign key)."""
    unsigned = intent.model_copy(update=update)
    return ApprovedOrderIntent.model_validate(
        {**unsigned.model_dump(), "signature": signer.sign(unsigned.signing_string())}
    )


def gate(engine: RiskEngine, intent: ApprovedOrderIntent, ctx: RiskContext | None = None, **kw: Any) -> list[str]:
    kw.setdefault("self_recorded", False)
    return engine.final_gate(intent, ctx or make_ctx(), SIGNER.verifier, engine.policy_version, **kw)


# ------------------------------------------------------------------------------------------ authorize


def test_authorize_builds_a_signed_intent_matching_the_proposal() -> None:
    engine, intent, _ = signed()
    assert intent.symbol == "SPY" and intent.side == Side.BUY
    assert intent.quantity == d("3") and intent.limit_price == d("500.04")
    assert intent.extended_hours is False
    assert intent.key_id == SIGNER.key_id and intent.signature_alg == "ECDSA_P256_SHA256"
    assert SIGNER.verifier.verify(intent.signing_string(), intent.signature)
    assert intent.expires_at - intent.created_at == timedelta(seconds=RiskPolicy().intent_ttl_seconds)
    assert intent.policy_version == engine.policy_version
    assert intent.client_order_id.startswith("lat-sim-")
    assert intent.requires_human_approval is False
    assert intent.mode == TradingMode.PAPER and intent.gateway == GatewayKind.FAKE


@pytest.mark.parametrize(
    ("mode", "gateway", "feed", "prefix"),
    [
        (TradingMode.REPLAY, GatewayKind.FAKE, DataFeed.SYNTHETIC, "lat-replay-"),
        (TradingMode.PAPER, GatewayKind.LEAN_PAPER_SIM, DataFeed.SYNTHETIC, "lat-sim-"),
        (TradingMode.PAPER, GatewayKind.LEAN_ALPACA_PAPER, DataFeed.IEX, "lat-paper-"),
    ],
)
def test_client_order_id_carries_mode_tag(mode: TradingMode, gateway: GatewayKind, feed: DataFeed, prefix: str) -> None:
    _, intent, _ = signed(mode=mode, gateway=gateway, market=make_market(feed=feed))
    assert intent.client_order_id.startswith(prefix)


def test_policy_requiring_human_approval_marks_intent() -> None:
    engine = make_engine(RiskPolicy(require_human_approval=True))
    _, intent, _ = signed(engine)
    assert intent.requires_human_approval is True


def test_authorize_refuses_rejected_decision() -> None:
    engine = make_engine()
    p = make_proposal(quantity=d("10"), max_planned_loss=d("50.40"))
    ctx = make_ctx()
    dec = engine.evaluate(p, ctx)
    assert dec.verdict == RiskVerdict.REJECTED
    with pytest.raises(RiskAuthorizationError):
        engine.authorize(p, dec, ctx)


def test_authorize_refuses_forged_approval_of_a_rejected_decision() -> None:
    engine = make_engine()
    p = make_proposal(quantity=d("10"), max_planned_loss=d("50.40"))
    ctx = make_ctx()
    forged = engine.evaluate(p, ctx).model_copy(update={"verdict": RiskVerdict.APPROVED, "reasons": ()})
    with pytest.raises(RiskAuthorizationError, match="not issued by this risk engine"):
        engine.authorize(p, forged, ctx)


def test_no_trade_never_yields_an_intent() -> None:
    engine = make_engine()
    nt = make_no_trade()
    ctx = make_ctx()
    dec = engine.evaluate(nt, ctx)
    with pytest.raises(RiskAuthorizationError):
        engine.authorize(nt, dec, ctx)
    # even a forged APPROVED decision cannot authorize a NO_TRADE proposal
    forged = dec.model_copy(update={"verdict": RiskVerdict.APPROVED})
    with pytest.raises(RiskAuthorizationError):
        engine.authorize(nt, forged, ctx)


def test_authorize_refuses_decision_from_another_engine() -> None:
    _, proposal, decision, ctx = approved()
    other = make_engine()
    with pytest.raises(RiskAuthorizationError, match="not issued by this risk engine"):
        other.authorize(proposal, decision, ctx)


def test_authorize_refuses_mismatched_proposal() -> None:
    engine, _, decision, ctx = approved()
    other = make_proposal(proposal_id="prop_other")
    with pytest.raises(RiskAuthorizationError, match="does not belong"):
        engine.authorize(other, decision, ctx)


def test_authorize_reevaluates_and_refuses_when_context_turned_bad() -> None:
    engine, proposal, decision, _ = approved()
    locked = make_ctx(health=healthy(entries_locked=True, lock_reasons=("halt",)))
    with pytest.raises(RiskAuthorizationError, match="re-evaluation rejected"):
        engine.authorize(proposal, decision, locked)


def test_authorize_refuses_mode_switch_between_evaluate_and_authorize() -> None:
    engine, proposal, decision, _ = approved()
    with pytest.raises(RiskAuthorizationError):
        engine.authorize(proposal, decision, make_ctx(mode=TradingMode.REPLAY))


# ------------------------------------------------------------------------------------------ final gate


def test_final_gate_passes_a_fresh_valid_intent() -> None:
    engine, intent, _ = signed()
    assert gate(engine, intent) == []


def test_final_gate_refuses_tampered_quantity() -> None:
    engine, intent, _ = signed()
    tampered = intent.model_copy(update={"quantity": d("30")})
    fails = gate(engine, tampered)
    assert any("signature: invalid" in f for f in fails)


@pytest.mark.parametrize(
    "update",
    [{"symbol": "QQQ"}, {"side": Side.SELL}, {"limit_price": d("510.00")}, {"client_order_id": "lat-sim-x"},
     {"policy_version": "pol_x"}, {"requires_human_approval": False, "expires_at": NOW + timedelta(hours=1)}],
)
def test_final_gate_refuses_any_tampered_signed_field(update: dict[str, Any]) -> None:
    engine, intent, _ = signed()
    assert any("signature: invalid" in f for f in gate(engine, intent.model_copy(update=update)))


def test_final_gate_refuses_foreign_key_intent() -> None:
    engine, intent, _ = signed()
    foreign = IntentSigner.generate()
    forged = resign(intent, foreign, key_id=foreign.key_id)
    fails = gate(engine, forged)
    assert any("key id" in f for f in fails)
    assert any("signature: invalid" in f for f in fails)


def test_final_gate_refuses_foreign_key_claiming_our_key_id() -> None:
    engine, intent, _ = signed()
    forged = resign(intent, IntentSigner.generate(), quantity=d("40"))
    assert any("signature: invalid" in f for f in gate(engine, forged))


def test_final_gate_refuses_a_foreign_verifier() -> None:
    engine, intent, _ = signed()
    foreign = IntentSigner.generate()
    fails = engine.final_gate(intent, make_ctx(), foreign.verifier, engine.policy_version, self_recorded=False)
    assert any("verifier is not this risk engine" in f for f in fails)


def test_final_gate_refuses_expired_intent() -> None:
    clock = ManualClock(NOW)
    engine = RiskEngine(RiskPolicy(), SIGNER, clock)
    _, intent, _ = signed(engine)
    clock.advance(RiskPolicy().intent_ttl_seconds + 1)
    later = make_ctx(now=clock.now(), market=make_market(now=clock.now()))
    fails = gate(engine, intent, later)
    assert any("expiry: intent expired" in f for f in fails)


def test_final_gate_refuses_live_mode_intent_even_if_signed_with_our_key() -> None:
    engine, intent, _ = signed()
    live = resign(intent, SIGNER, mode=TradingMode.LIVE_APPROVAL, gateway=GatewayKind.LEAN_ALPACA_LIVE,
                  client_order_id="lat-live-" + intent.client_order_id.split("-", 2)[2])
    live_ctx = make_ctx(mode=TradingMode.LIVE_APPROVAL, gateway=GatewayKind.LEAN_ALPACA_LIVE)
    fails = gate(engine, live, live_ctx)
    assert any("live execution is denied" in f for f in fails)
    assert any("LIVE_EXECUTION_COMPILED is False" in f for f in fails)


def test_final_gate_refuses_mode_mismatch_with_runtime() -> None:
    engine, intent, _ = signed()
    fails = gate(engine, intent, make_ctx(mode=TradingMode.REPLAY))
    assert any(f.startswith("mode: intent PAPER/FAKE") for f in fails)


def test_final_gate_refuses_wrong_client_order_tag() -> None:
    engine, intent, _ = signed()
    wrong = resign(intent, SIGNER, client_order_id="lat-paper-" + intent.client_order_id.split("-", 2)[2])
    assert any("not tagged 'sim'" in f for f in gate(engine, wrong))


def test_final_gate_refuses_superseded_policy_version() -> None:
    engine, intent, _ = signed()
    new_version = RiskPolicy(max_order_notional=d("1000.00")).version
    fails = engine.final_gate(intent, make_ctx(), SIGNER.verifier, new_version, self_recorded=False)
    assert any("no longer active" in f for f in fails)
    assert any("stale policy" in f for f in fails)


@pytest.mark.parametrize(
    "health_kw",
    [{"entries_locked": True, "lock_reasons": ("reconciliation_mismatch",)}, {"broker_connected": False},
     {"reconciled": False}, {"feed_ok": False}, {"engine_ok": False}, {"strategy_state_ok": False},
     {"permissions_ok": False}],
)
def test_final_gate_health_gates(health_kw: dict[str, Any]) -> None:
    engine, intent, _ = signed()
    fails = gate(engine, intent, make_ctx(health=healthy(**health_kw)))
    assert any(f.startswith("health:") for f in fails)


def test_final_gate_refuses_active_halts() -> None:
    engine, intent, _ = signed()
    fails = gate(engine, intent, active_halts=[{"halt_id": "halt_1", "reason": "operator stop"}])
    assert any(f.startswith("halt:") for f in fails)


def test_final_gate_refuses_when_market_closed() -> None:
    clock = ManualClock(NOW)
    engine = RiskEngine(RiskPolicy(), SIGNER, clock)
    _, intent, _ = signed(engine)
    clock.set(AFTER_HOURS)
    # a context captured "now" (after hours) with an otherwise-valid intent window
    fails = gate(engine, intent, make_ctx(now=AFTER_HOURS))
    assert any(f.startswith("session:") for f in fails)


def test_final_gate_refuses_missing_or_blocked_account() -> None:
    engine, intent, _ = signed()
    assert any("account: snapshot missing" in f for f in gate(engine, intent, make_ctx(account=None)))
    from tests.unit.risk.risk_factories import make_account

    blocked = make_ctx(account=make_account(trading_blocked=True))
    assert any("account: blocked" in f for f in gate(engine, intent, blocked))


def test_final_gate_refuses_stale_context() -> None:
    engine, intent, _ = signed()
    old = make_ctx(now=NOW - timedelta(seconds=30))
    assert any(f.startswith("context:") for f in gate(engine, intent, old))


def test_final_gate_duplicates() -> None:
    engine, intent, _ = signed()
    fp = fingerprint(intent)
    # the ledger already recorded this intent: its own fingerprint appears once => fine
    assert gate(engine, intent, make_ctx(recent_intent_fingerprints=(fp,)), self_recorded=True) == []
    # a second identical authorization in the window => duplicate
    dup = gate(engine, intent, make_ctx(recent_intent_fingerprints=(fp, fp)), self_recorded=True)
    assert any(f.startswith("duplicate:") for f in dup)
    # not yet recorded: any identical prior fingerprint is a duplicate
    assert any(f.startswith("duplicate:") for f in gate(engine, intent, make_ctx(recent_intent_fingerprints=(fp,))))


def test_final_gate_identical_open_order_under_other_coid_is_duplicate() -> None:
    engine, intent, _ = signed()
    twin = make_order("SPY", Side.BUY, "3", "500.04", coid="lat-sim-other")
    fails = gate(engine, intent, make_ctx(open_orders=(twin,)))
    assert any("identical open order" in f for f in fails)
    # its own row in open_orders is ignored
    own = make_order("SPY", Side.BUY, "3", "500.04", coid=intent.client_order_id)
    assert gate(engine, intent, make_ctx(open_orders=(own,))) == []


def test_final_gate_refuses_sell_beyond_position() -> None:
    engine = make_engine()
    p = make_proposal(side=Side.SELL, quantity=d("2"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    ctx = make_ctx(positions=(make_position("SPY", "2"),))
    dec = engine.evaluate(p, ctx)
    intent = engine.authorize(p, dec, ctx)
    assert gate(engine, intent, ctx) == []
    fails = gate(engine, intent, make_ctx(positions=(make_position("SPY", "1"),)))
    assert any(f.startswith("long-only:") for f in fails)


def test_final_gate_never_raises_on_garbage_context() -> None:
    engine, intent, _ = signed()
    garbage = RiskContext.model_construct(now=None, mode=None, health=None)  # type: ignore[arg-type]
    fails = engine.final_gate(intent, garbage, SIGNER.verifier, engine.policy_version)
    assert fails and "fail closed" in fails[0]


# ------------------------------------------------------------------------------------------ human approval


def test_human_approval_required_and_verified(tmp_path: Path) -> None:
    engine = make_engine(RiskPolicy(require_human_approval=True))
    _, intent, _ = signed(engine)
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    token = auth.token_path.read_text().strip()

    missing = gate(engine, intent)
    assert any("human approval: required but absent" in f for f in missing)

    approval = auth.create_approval(token, intent, approver="victor", now=NOW)
    ok = gate(engine, intent, human_approval=approval, approval_verifier=auth.verify_approval)
    assert ok == []

    no_verifier = gate(engine, intent, human_approval=approval)
    assert any("no verifier supplied" in f for f in no_verifier)

    tampered = approval.model_copy(update={"statement": "I approve 300 shares"})
    assert any("signature invalid" in f for f in gate(engine, intent, human_approval=tampered,
                                                      approval_verifier=auth.verify_approval))

    auth.rotate()  # rotating the operator token invalidates outstanding approvals
    assert any("signature invalid" in f for f in gate(engine, intent, human_approval=approval,
                                                      approval_verifier=auth.verify_approval))


def test_human_approval_bound_to_one_intent_and_expires(tmp_path: Path) -> None:
    clock = ManualClock(NOW)
    engine = RiskEngine(RiskPolicy(require_human_approval=True), SIGNER, clock)
    _, intent, _ = signed(engine)
    _, other, _ = signed(engine)
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    token = auth.token_path.read_text().strip()
    approval = auth.create_approval(token, other, approver="victor", now=NOW, ttl_seconds=30)
    fails = gate(engine, intent, human_approval=approval, approval_verifier=auth.verify_approval)
    assert any("different intent" in f for f in fails)

    own = auth.create_approval(token, intent, approver="victor", now=NOW, ttl_seconds=30)
    clock.advance(31)
    fails = gate(engine, intent, make_ctx(now=clock.now()), human_approval=own, approval_verifier=auth.verify_approval)
    assert any("human approval: expired" in f for f in fails)


def test_wrong_operator_token_cannot_create_approval(tmp_path: Path) -> None:
    engine = make_engine(RiskPolicy(require_human_approval=True))
    _, intent, _ = signed(engine)
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    from services.risk.operator_auth import OperatorAuthError

    with pytest.raises(OperatorAuthError):
        auth.create_approval("not-the-token", intent, approver="model", now=NOW)


def test_live_gateways_are_never_allowed_by_core_modes() -> None:
    assert modes.LIVE_EXECUTION_COMPILED is False
    for mode in TradingMode:
        assert GatewayKind.LEAN_ALPACA_LIVE not in modes.allowed_gateways(mode)


def test_authorize_is_idempotent_per_decision() -> None:
    engine, proposal, decision, ctx = approved()
    first = engine.authorize(proposal, decision, ctx)
    again = engine.authorize(proposal, decision, ctx)
    assert again == first  # same intent id and client order id: a retry never mints a second order


def test_authorize_after_intent_expiry_requires_new_evaluation() -> None:
    clock = ManualClock(NOW)
    engine = RiskEngine(RiskPolicy(), SIGNER, clock)
    proposal = make_proposal()
    ctx = make_ctx()
    decision = engine.evaluate(proposal, ctx)
    engine.authorize(proposal, decision, ctx)
    clock.advance(RiskPolicy().intent_ttl_seconds + 1)
    later = make_ctx(now=clock.now(), market=make_market(now=clock.now()))
    with pytest.raises(RiskAuthorizationError, match="expired"):
        engine.authorize(proposal, decision, later)


def test_authorize_refuses_proposal_edited_after_evaluation() -> None:
    engine, proposal, decision, ctx = approved()
    edited = proposal.model_copy(update={"quantity": d("2"), "max_planned_loss": d("10.08")})  # still within limits
    with pytest.raises(RiskAuthorizationError, match="differs"):
        engine.authorize(edited, decision, ctx)


@pytest.mark.parametrize(
    ("target", "field", "raw"),
    [("health", "feed_ok", "no"), ("health", "reconciled", "0"), ("health", "permissions_ok", "false"),
     ("account", "is_paper", "false")],
)
def test_final_gate_reads_revalidated_context(target: str, field: str, raw: str) -> None:
    engine, intent, ctx = signed()
    part = getattr(ctx, target)
    smuggled = ctx.model_copy(update={target: part.model_copy(update={field: raw})})
    fails = gate(engine, intent, smuggled)
    assert any(f.startswith("inputs:") for f in fails)
    assert any(f.startswith(("health:", "account:")) for f in fails), fails


def test_final_gate_refuses_intent_with_values_set_without_validation() -> None:
    engine, intent, _ = signed()
    smuggled = intent.model_copy(update={"extended_hours": "false"})  # truthy raw string; signature unchanged
    fails = gate(engine, smuggled)
    assert any(f.startswith("inputs: intent") for f in fails)


def test_final_gate_rechecks_no_margin_cash_cover() -> None:
    from tests.unit.risk.risk_factories import make_account

    engine, intent, _ = signed()
    spent = make_ctx(account=make_account(cash="1000.00", buying_power="40000.00"))  # cash spent elsewhere
    assert any(f.startswith("no-margin:") for f in gate(engine, intent, spent))
    committed = make_ctx(open_orders=(make_order("QQQ", Side.BUY, "20", "450.00", coid="lat-sim-other"),))
    assert any(f.startswith("no-margin:") for f in gate(engine, intent, committed))
    unknown = make_ctx(open_orders=(make_order("QQQ", Side.BUY, "1", None, coid="lat-sim-nolimit"),))
    assert any("cash commitment unknown" in f for f in gate(engine, intent, unknown))


def test_final_gate_rechecks_daily_loss_stop() -> None:
    engine, intent, _ = signed()
    assert any(f.startswith("daily-loss:") for f in gate(engine, intent, make_ctx(daily_realized_pl=d("-250.00"))))
    assert any("unknown" in f for f in gate(engine, intent, make_ctx(start_of_day_equity=None)))


def test_final_gate_money_rechecks_do_not_apply_to_exits() -> None:
    engine = make_engine()
    p = make_proposal(side=Side.SELL, quantity=d("2"), limit_price=d("500.00"), stop_price=None, max_planned_loss=d("0"))
    ctx = make_ctx(positions=(make_position("SPY", "2"),))
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    later = make_ctx(positions=(make_position("SPY", "2"),), daily_realized_pl=d("-400.00"))
    assert gate(engine, intent, later) == []  # an exit is still allowed after the daily loss stop
