"""Unit tests for the Phase 1 contract and risk-engine rules for exits and protective stops.

* STOP_LIMIT exists only as a protective long-only SELL; its stop price is signed (``lat-intent-v2``) while LIMIT
  intents keep the original ``lat-intent-v1`` string byte for byte (LEAN verifier compatibility).
* Exits use the narrower exit lock (``SystemHealth.exits_*``; unset => same as entries, fail closed).
* No new entries in the last 30 minutes of the session; protective SELLs never consume entry pending slots.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from packages.contracts.enums import OrderState, OrderType, Side
from packages.contracts.models import ApprovedOrderIntent, BrokerOrder, SystemHealth, TradeProposal
from services.risk import limits
from tests.unit.risk.risk_factories import (
    NOW,
    SIGNER,
    check,
    failed,
    healthy,
    make_ctx,
    make_engine,
    make_order,
    make_position,
    make_proposal,
)

D = Decimal


def stop_exit(**kw: Any) -> TradeProposal:
    """Protective STOP_LIMIT SELL 2 SPY: stop 495.00, limit 492.52 (quote 500.00/500.04)."""
    base: dict[str, Any] = {"side": Side.SELL, "quantity": D(2), "order_type": OrderType.STOP_LIMIT,
                            "limit_price": D("492.52"), "stop_price": D("495.00"), "max_planned_loss": D(0)}
    base.update(kw)
    return make_proposal(**base)


def limit_exit(**kw: Any) -> TradeProposal:
    base: dict[str, Any] = {"side": Side.SELL, "quantity": D(2), "limit_price": D("500.00"), "stop_price": None,
                            "max_planned_loss": D(0)}
    base.update(kw)
    return make_proposal(**base)


def held(**kw: Any) -> Any:
    return make_ctx(positions=(make_position(qty="2", now=kw.get("now", NOW)),), **kw)


# ------------------------------------------------------------------------------------------------ contracts
def test_stop_limit_proposal_must_be_a_protective_sell() -> None:
    stop_exit()  # valid
    with pytest.raises(ValidationError, match="only for protective SELL"):
        make_proposal(order_type=OrderType.STOP_LIMIT, stop_price=D("499"), limit_price=D("498"))
    with pytest.raises(ValidationError, match="positive stop_price"):
        stop_exit(stop_price=None)
    with pytest.raises(ValidationError, match="at or below stop_price"):
        stop_exit(limit_price=D("496.00"))
    with pytest.raises(ValidationError, match="only LIMIT"):
        make_proposal(order_type=OrderType.MARKET)


def _intent(**kw: Any) -> ApprovedOrderIntent:
    base: dict[str, Any] = {
        "intent_id": "intent_1", "client_order_id": "lat-sim-x1", "proposal_id": "p", "decision_id": "d",
        "causal_id": "c", "mode": "PAPER", "gateway": "FAKE", "symbol": "SPY", "side": "SELL", "quantity": "2",
        "order_type": "LIMIT", "limit_price": "500.00", "time_in_force": "DAY", "policy_version": "pol_x",
        "created_at": NOW, "expires_at": NOW + timedelta(seconds=60), "requires_human_approval": False,
        "key_id": SIGNER.key_id, "signature": "0" * 128,
    }
    base.update(kw)
    return ApprovedOrderIntent.model_validate(base)


def test_limit_intents_keep_the_v1_signing_string_and_stop_limits_sign_their_stop() -> None:
    limit = _intent()
    text = limit.signing_string()
    assert text.startswith("lat-intent-v1|") and "stop_price" not in text
    stop = _intent(order_type="STOP_LIMIT", limit_price="492.52", stop_price="495.00")
    stext = stop.signing_string()
    assert stext.startswith("lat-intent-v2|") and stext.endswith("|stop_price=495.00")
    # the stop is covered by the signature: changing it invalidates a signed intent
    sig = SIGNER.sign(stext)
    assert SIGNER.verifier.verify(stext, sig)
    tampered = stop.model_copy(update={"stop_price": D("480.00")})
    assert not SIGNER.verifier.verify(tampered.signing_string(), sig)
    with pytest.raises(ValidationError):
        _intent(stop_price="495.00")  # a stop on a LIMIT intent is refused
    with pytest.raises(ValidationError):
        _intent(order_type="STOP_LIMIT", side="BUY", limit_price="492.52", stop_price="495.00")


def test_system_health_exit_lock_falls_back_to_entries_when_unset() -> None:
    legacy = SystemHealth(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True, entries_locked=True,
                          lock_reasons=("halt h1",))
    assert legacy.exits_blocked and legacy.exit_block_reasons == ("halt h1",)  # fail closed for old callers
    split = SystemHealth(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True, entries_locked=True,
                         lock_reasons=("halt h1",), exits_locked=False)
    assert not split.exits_blocked
    assert SystemHealth(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True, entries_locked=False,
                        exits_locked=False, exit_lock_reasons=("unknown order",)).exits_blocked


# ------------------------------------------------------------------------------------------------ evaluate
def test_protective_stop_limit_exit_is_approved_and_signed_with_its_stop() -> None:
    engine = make_engine()
    ctx = held()
    decision = engine.evaluate(stop_exit(), ctx)
    assert decision.approved, decision.reasons
    assert "protective STOP_LIMIT" in check(decision, "order_type_limit_only").detail
    intent = engine.authorize(stop_exit(), decision, ctx)
    assert intent.order_type == OrderType.STOP_LIMIT and intent.stop_price == D("495.00")
    assert SIGNER.verifier.verify(intent.signing_string(), intent.signature)


@pytest.mark.parametrize(("kw", "failing"), [
    ({"stop_price": D("499.99"), "limit_price": D("499.00")}, "limit_vs_touch"),  # would trigger at once
    ({"stop_price": D("495.00"), "limit_price": D("479.00")}, "limit_vs_mid"),  # stop-limit gap > 300 bps
    ({"stop_price": D("390.00"), "limit_price": D("389.00")}, "limit_vs_mid"),  # > 2000 bps: not protection
    ({"stop_price": D("495.005"), "limit_price": D("492.52")}, "limit_price_tick"),
    ({"quantity": D(3)}, "long_only"),  # more than the long position
])
def test_bad_protective_geometry_is_rejected(kw: dict[str, Any], failing: str) -> None:
    decision = make_engine().evaluate(stop_exit(**kw), held())
    assert not decision.approved and failing in failed(decision), decision.reasons


def test_exits_pass_an_entries_only_lock_but_not_an_exit_lock() -> None:
    engine = make_engine()
    entries_only = healthy(entries_locked=True, lock_reasons=("halt h1: stop new entries (scope=ENTRIES)",),
                           exits_locked=False)
    assert engine.evaluate(limit_exit(), held(health=entries_only)).approved
    assert engine.evaluate(stop_exit(), held(health=entries_only)).approved
    assert "health_entries_unlocked" in failed(engine.evaluate(make_proposal(), make_ctx(health=entries_only)))
    blind = healthy(entries_locked=True, lock_reasons=("broker_disconnected",), exits_locked=True,
                    exit_lock_reasons=("broker_disconnected",))
    assert "health_entries_unlocked" in failed(engine.evaluate(limit_exit(), held(health=blind)))
    legacy = healthy(entries_locked=True, lock_reasons=("halt",))  # exits_locked unset: same as entries
    assert "health_entries_unlocked" in failed(engine.evaluate(limit_exit(), held(health=legacy)))


def test_no_new_entries_in_the_last_30_minutes_but_exits_continue() -> None:
    close = NOW.replace(hour=21, minute=0)  # 16:00 ET
    late = close - timedelta(seconds=limits.ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS - 60)
    engine = make_engine(now=late)
    decision = engine.evaluate(make_proposal(now=late), make_ctx(now=late))
    assert failed(decision) == {"entry_window"}, decision.reasons
    assert engine.evaluate(limit_exit(now=late), held(now=late)).approved
    early = close - timedelta(seconds=limits.ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS + 60)
    assert make_engine(now=early).evaluate(make_proposal(now=early), make_ctx(now=early)).approved


def _protective_order(qty: str = "2") -> BrokerOrder:
    o = make_order(side=Side.SELL, qty=qty, limit="492.52", state=OrderState.ACKNOWLEDGED, coid="lat-sim-prot1")
    return o.model_copy(update={"order_type": OrderType.STOP_LIMIT, "stop_price": D("495.00")})


def test_working_protective_stops_do_not_consume_entry_pending_slots() -> None:
    engine = make_engine()
    # default policy: max_pending_orders=1, max_positions=1; adding to the held symbol is still one position
    ctx = make_ctx(positions=(make_position(qty="2"),), open_orders=(_protective_order(),))
    assert "max_pending_orders" not in failed(engine.evaluate(make_proposal(quantity=D(1)), ctx))
    working_limit_sell = make_order(side=Side.SELL, qty="1", limit="500.10", coid="lat-sim-sell1")
    ctx2 = make_ctx(positions=(make_position(qty="2"),), open_orders=(working_limit_sell,))
    assert "max_pending_orders" in failed(engine.evaluate(make_proposal(quantity=D(1)), ctx2))


def test_sells_have_a_per_symbol_cap_instead_of_the_entry_pending_cap() -> None:
    engine = make_engine()
    one = held(open_orders=(_protective_order("1"),))
    # a top-up next to an existing stop (one tick lower: an identical open order would be a duplicate)
    assert engine.evaluate(stop_exit(quantity=D(1), limit_price=D("492.51")), one).approved
    two = held(open_orders=(_protective_order("1"), make_order(side=Side.SELL, qty="0", limit="501",
                                                               coid="lat-sim-s2").model_copy(update={"quantity": D(1)})))
    decision = engine.evaluate(stop_exit(quantity=D(1), limit_price=D("492.51")), two)
    assert "max_pending_orders" in failed(decision)


# ------------------------------------------------------------------------------------------------ final gate
def _authorized(proposal: TradeProposal, ctx: Any) -> tuple[Any, ApprovedOrderIntent]:
    engine = make_engine()
    decision = engine.evaluate(proposal, ctx)
    assert decision.approved, decision.reasons
    return engine, engine.authorize(proposal, decision, ctx)


def test_final_gate_accepts_a_protective_exit_under_an_entries_halt_but_not_a_kill_switch() -> None:
    entries_only = healthy(entries_locked=True, lock_reasons=("halt",), exits_locked=False)
    engine, intent = _authorized(stop_exit(), held())
    ctx = held(health=entries_only)
    assert engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy_version,
                             active_halts=({"halt_id": "h1", "scope": "ENTRIES"},)) == []
    fails = engine.final_gate(intent, ctx, SIGNER.verifier, engine.policy_version,
                              active_halts=({"halt_id": "h2", "scope": "ALL"},))
    assert any("scope ALL" in f for f in fails)
    blind = held(health=healthy(entries_locked=True, lock_reasons=("x",), exits_locked=True, exit_lock_reasons=("x",)))
    assert any(f.startswith("health: exits locked") for f in
               engine.final_gate(intent, blind, SIGNER.verifier, engine.policy_version))


def test_final_gate_refuses_entries_inside_the_pre_close_cutoff() -> None:
    late = NOW.replace(hour=20, minute=45)  # 15:45 ET
    engine, intent = _authorized(make_proposal(), make_ctx())
    engine._clock.set(late)  # type: ignore[attr-defined]
    fails = engine.final_gate(intent.model_copy(update={"created_at": late, "expires_at": late + timedelta(seconds=30)}),
                              make_ctx(now=late), SIGNER.verifier, engine.policy_version)
    assert "session: inside the pre-close entry cutoff (no new entries)" in fails
