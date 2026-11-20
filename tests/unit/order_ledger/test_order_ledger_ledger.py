"""OrderLedger: durable state machine over real SQLite (temp files). No broker, no network."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from decimal import Decimal

import pytest

from packages.contracts.enums import (
    DataFeed,
    EventSource,
    FeedMode,
    GatewayKind,
    MarketSession,
    OrderEventType,
    OrderState,
    RiskVerdict,
    Side,
    TradingMode,
)
from packages.contracts.models import BrokerOrder, HumanApproval, MarketSnapshot, OrderType, RiskDecision, TimeInForce
from packages.core.audit import verify_chain
from packages.core.ids import new_id
from packages.core.modes import LiveActivationDenied, ModeViolation
from packages.core.signing import IntentSigner
from services.order_ledger.entries import Flag
from services.order_ledger.errors import IntentRefused, LedgerConflict
from services.order_ledger.ledger import OrderLedger
from tests.unit.order_ledger.ol_support import (
    POLICY_VERSION,
    T0,
    Harness,
    claim,
    flags,
    make_quote,
    state_of,
    stream_event,
)

S = OrderState
E = OrderEventType


def _fill(coid: str, qty: str, px: str, cum: str, exec_id: str, *, state: OrderState = S.PARTIALLY_FILLED,
          etype: OrderEventType = E.PARTIAL_FILL, at=T0) -> object:
    return stream_event(coid, etype, state, at, fill=(qty, px), cum=cum, exec_id=exec_id)


# ------------------------------------------------------------------------------------------ artefacts


def test_record_artifacts_are_insert_once_and_conflicts_raise(h: Harness) -> None:
    intent = h.factory.make()
    prop_row = h.db.query_one("SELECT payload_json FROM proposals WHERE proposal_id=?", (intent.proposal_id,))
    assert prop_row is not None
    dec_row = h.db.query_one("SELECT payload_json FROM risk_decisions WHERE decision_id=?", (intent.decision_id,))
    decision = RiskDecision.model_validate_json(dec_row["payload_json"])  # type: ignore[index]
    assert h.ledger.record_decision(decision) is False  # identical content: no-op
    with pytest.raises(LedgerConflict):
        h.ledger.record_decision(decision.model_copy(update={"reasons": ("changed",)}))
    snap = MarketSnapshot(
        snapshot_id=new_id("snap"), symbol="SPY", feed=DataFeed.SYNTHETIC, feed_mode=FeedMode.SIMULATED,
        session=MarketSession.REGULAR, quote=make_quote("SPY", "99.99", "100.01", T0), data_asof=T0, source_id="t",
        dataset_version="ds", is_simulated=True,
    )
    assert h.ledger.record_market_snapshot(snap) is True
    assert h.ledger.record_market_snapshot(snap) is False
    types = {r["event_type"] for r in h.db.query("SELECT event_type FROM audit_events")}
    assert {"PROPOSAL_RECORDED", "RISK_DECISION_RECORDED", "MARKET_SNAPSHOT_RECORDED", "ORDER_INTENT_RECORDED"} <= types


def test_record_intent_creates_risk_approved_order_with_fingerprint(h: Harness) -> None:
    intent = h.factory.make()
    rec = h.ledger.get_record(intent.client_order_id)
    assert rec is not None
    assert rec.state == S.RISK_APPROVED
    assert rec.intent == intent
    assert rec.intent_fingerprint.startswith("fp_")
    assert [t["to_state"] for t in h.ledger.transitions(intent.client_order_id)] == ["PROPOSED", "RISK_APPROVED"]
    assert h.ledger.record_intent(intent).client_order_id == intent.client_order_id  # idempotent


def test_record_intent_refusals(h: Harness, tmp_path) -> None:
    # 1. no recorded decision
    unrecorded = h.factory.make(record=False)
    with pytest.raises(IntentRefused, match="no recorded risk decision"):
        h.ledger.record_intent(unrecorded)
    # 2. mode mismatch
    other_mode = OrderLedger(h.db, h.clock, TradingMode.REPLAY)
    with pytest.raises(IntentRefused, match="mode"):
        other_mode.record_intent(h.factory.make(record=False))
    # 3. already expired
    stale = h.factory.make(record=False, ttl_s=5)
    h.clock.advance(10)
    with pytest.raises(IntentRefused, match="expired"):
        h.ledger.record_intent(stale)
    # 4. live gateway always denied
    with pytest.raises((LiveActivationDenied, ModeViolation)):
        h.ledger.record_intent(h.factory.make(record=False, gateway=GatewayKind.LEAN_ALPACA_LIVE))
    # 5. signature checked when a verifier is configured
    strict = OrderLedger(h.db, h.clock, TradingMode.PAPER, verifier=IntentSigner.generate().verifier)
    with pytest.raises(IntentRefused, match="signature"):
        strict.record_intent(h.factory.make(record=False))
    # 6. conflicting content under the same client order id
    intent = h.factory.make()
    with pytest.raises(LedgerConflict):
        h.ledger.record_intent(intent.model_copy(update={"quantity": Decimal("11")}))


def test_record_intent_refuses_rejected_decision(h: Harness) -> None:
    intent = h.factory.make(record=False)
    proposal_row = None  # proposal is recorded implicitly below
    rejected = RiskDecision(
        decision_id=intent.decision_id, proposal_id=intent.proposal_id, causal_id=intent.causal_id,
        verdict=RiskVerdict.REJECTED, checks=(), reasons=("too big",), policy_version=POLICY_VERSION,
        mode=TradingMode.PAPER, evaluated_at=T0,
    )
    from packages.contracts.enums import DecisionAction, Proposer
    from packages.contracts.models import TradeProposal

    h.ledger.record_proposal(TradeProposal(
        proposal_id=intent.proposal_id, causal_id=intent.causal_id, created_at=T0, data_asof=T0,
        action=DecisionAction.NO_TRADE, no_trade_reason="x", proposer=Proposer.TEST, strategy_id="s",
        dataset_version="d",
    ))
    h.ledger.record_decision(rejected)
    assert proposal_row is None
    with pytest.raises(IntentRefused, match="not APPROVED"):
        h.ledger.record_intent(intent)
    assert h.db.query_one("SELECT COUNT(*) n FROM orders")["n"] == 0  # type: ignore[index]


# ------------------------------------------------------------------------------------------ claim / human approval


def test_claim_is_atomic_and_never_returns_twice(h: Harness) -> None:
    intent = h.factory.make()
    first = h.ledger.claim_for_submission(intent.client_order_id, "a")
    second = h.ledger.claim_for_submission(intent.client_order_id, "b")
    assert first == intent and second is None
    rec = h.ledger.get_record(intent.client_order_id)
    assert rec is not None and rec.state == S.SUBMITTING and rec.claimed_by == "a"
    assert h.ledger.recent_submission_times(T0 - timedelta(seconds=1)) == [T0]


def test_claim_refuses_expired_intent(h: Harness) -> None:
    intent = h.factory.make(ttl_s=30)
    h.clock.advance(31)
    assert h.ledger.claim_for_submission(intent.client_order_id, "a") is None
    assert state_of(h, intent.client_order_id) == S.EXPIRED


def _approval(intent_id: str, at, *, ttl: int = 120) -> HumanApproval:
    return HumanApproval(approval_id=new_id("appr"), intent_id=intent_id, approver="victor", confirmed_at=at,
                         expires_at=at + timedelta(seconds=ttl), statement="I confirm this paper order",
                         signature="ab" * 32)


def test_human_approval_required_before_claim(h: Harness) -> None:
    intent = h.factory.make(requires_human=True)
    assert h.ledger.claim_for_submission(intent.client_order_id, "x") is None
    assert state_of(h, intent.client_order_id) == S.RISK_APPROVED
    h.clock.advance(1)
    h.ledger.record_human_approval(_approval(intent.intent_id, h.clock.now()))
    assert state_of(h, intent.client_order_id) == S.HUMAN_APPROVED
    assert h.ledger.claim_for_submission(intent.client_order_id, "x") == intent


def test_human_approval_must_be_fresh_and_valid(h: Harness) -> None:
    intent = h.factory.make(requires_human=True)
    with pytest.raises(IntentRefused, match="predates"):
        h.ledger.record_human_approval(_approval(intent.intent_id, T0 - timedelta(minutes=5)))
    with pytest.raises(IntentRefused, match="future"):
        h.ledger.record_human_approval(_approval(intent.intent_id, T0 + timedelta(minutes=5)))
    plain = h.factory.make()
    with pytest.raises(IntentRefused, match="does not require"):
        h.ledger.record_human_approval(_approval(plain.intent_id, T0))
    strict = OrderLedger(h.db, h.clock, TradingMode.PAPER, approval_verifier=lambda _a: False)
    with pytest.raises(IntentRefused, match="signature"):
        strict.record_human_approval(_approval(intent.intent_id, T0))


def test_expired_human_approval_blocks_claim(h: Harness) -> None:
    intent = h.factory.make(requires_human=True, ttl_s=600)
    h.ledger.record_human_approval(_approval(intent.intent_id, T0, ttl=10))
    h.clock.advance(11)
    assert h.ledger.claim_for_submission(intent.client_order_id, "x") is None
    assert state_of(h, intent.client_order_id) == S.HUMAN_APPROVED


# ------------------------------------------------------------------------------------------ submission outcomes


def _broker_order(intent, state: OrderState, status: str, *, filled: str = "0", avg: str | None = None,
                  boid: str = "bo-1") -> BrokerOrder:
    return BrokerOrder(
        client_order_id=intent.client_order_id, broker_order_id=boid, intent_id=intent.intent_id, gateway=intent.gateway,
        symbol=intent.symbol, side=intent.side, quantity=intent.quantity, filled_quantity=Decimal(filled),
        avg_fill_price=Decimal(avg) if avg else None, order_type=OrderType.LIMIT, limit_price=intent.limit_price,
        time_in_force=TimeInForce.DAY, broker_status=status, state=state, submitted_at=T0, updated_at=T0,
    )


def test_mark_submitted_and_unknown(h: Harness) -> None:
    a = h.factory.make()
    claim(h, a)
    res = h.ledger.mark_submitted(a.client_order_id, _broker_order(a, S.SUBMITTED, "accepted"))
    assert res.applied and res.state == S.SUBMITTED
    rec = h.ledger.get_record(a.client_order_id)
    assert rec is not None and rec.broker_order_id == "bo-1" and rec.submitted_at == T0

    b = h.factory.make(symbol="QQQ")
    claim(h, b)
    assert h.ledger.mark_unknown(b.client_order_id, "timeout") == S.UNKNOWN
    assert Flag.UNKNOWN_ORDERS_PRESENT.value in flags(h)
    assert h.ledger.unknown_orders() == [b.client_order_id]
    locked, reasons = h.entries.is_locked()
    assert locked and any("unknown outcome" in r for r in reasons)


def test_reject_locally_only_before_submission(h: Harness) -> None:
    a = h.factory.make()
    assert h.ledger.reject_locally(a.client_order_id, ["final gate: stale quote"])
    assert state_of(h, a.client_order_id) == S.REJECTED
    b = h.factory.make(symbol="QQQ")
    claim(h, b)
    assert not h.ledger.reject_locally(b.client_order_id, ["too late"])
    assert state_of(h, b.client_order_id) == S.SUBMITTING


# ------------------------------------------------------------------------------------------ events


def _submitted(h: Harness, **kw: str):
    intent = h.factory.make(**kw)  # type: ignore[arg-type]
    claim(h, intent)
    return intent


def test_happy_path_events_fill_and_book_position(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.apply_event(stream_event(c, E.ACCEPTED, S.SUBMITTED, T0))
    h.ledger.apply_event(stream_event(c, E.NEW, S.ACKNOWLEDGED, T0))
    r1 = h.ledger.apply_event(_fill(c, "4", "100.00", "4", "x1"))
    assert r1.booked_quantity == Decimal(4) and r1.state == S.PARTIALLY_FILLED
    r2 = h.ledger.apply_event(_fill(c, "6", "99.50", "10", "x2", state=S.FILLED, etype=E.FILL))
    assert r2.state == S.FILLED
    order = h.ledger.get_order(c)
    assert order is not None and order.filled_quantity == Decimal(10)
    assert order.avg_fill_price == Decimal("99.70000000")
    [pos] = h.ledger.positions()
    assert pos.symbol == "SPY" and pos.quantity == Decimal(10) and pos.cost_basis == Decimal("997.00")
    assert [f["broker_exec_id"] for f in h.ledger.fills(c)] == ["x1", "x2"]
    assert [t["to_state"] for t in h.ledger.transitions(c)][-4:] == ["SUBMITTED", "ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED"]
    assert verify_chain(h.db)[0]


def test_duplicate_events_are_recorded_once_and_never_reapplied(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    fill = _fill(c, "10", "100", "10", "x1", state=S.FILLED, etype=E.FILL)
    assert h.ledger.apply_event(fill).applied
    again = h.ledger.apply_event(fill)
    assert again.duplicate and not again.applied
    # same execution id under a different event id (re-delivery) is still a duplicate
    redelivered = fill.model_copy(update={"event_id": "other-event-id"})  # type: ignore[union-attr]
    assert h.ledger.apply_event(redelivered).duplicate  # type: ignore[arg-type]
    assert h.ledger.positions()[0].quantity == Decimal(10)
    assert len(h.ledger.fills(c)) == 1
    assert h.ledger.duplicate_count() == 2
    assert len(h.ledger.events(c)) == 1  # SUBMITTING has no event row; one stored fill event


def test_fill_before_ack_and_ack_after_fill(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.apply_event(_fill(c, "10", "100", "10", "x1", state=S.FILLED, etype=E.FILL))
    late_ack = h.ledger.apply_event(stream_event(c, E.NEW, S.ACKNOWLEDGED, T0))
    late_accept = h.ledger.apply_event(stream_event(c, E.ACCEPTED, S.SUBMITTED, T0))
    assert not late_ack.applied and not late_accept.applied
    assert state_of(h, c) == S.FILLED
    stored = h.ledger.events(c)
    assert [e["applied"] for e in stored] == [True, False, False]
    assert "regress" in stored[1]["apply_note"]


def test_cumulative_filled_is_monotonic_and_capped(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.apply_event(_fill(c, "6", "100", "6", "x2"))
    h.ledger.apply_event(_fill(c, "4", "100", "10", "x3", state=S.FILLED, etype=E.FILL))
    # an over-fill report books nothing beyond the order quantity and raises the anomaly flag
    over = h.ledger.apply_event(_fill(c, "5", "100", "15", "x4", state=S.FILLED, etype=E.FILL))
    assert over.booked_quantity == 0 and over.anomalies
    assert h.ledger.get_order(c).filled_quantity == Decimal(10)  # type: ignore[union-attr]
    assert Flag.LEDGER_ANOMALY.value in flags(h)


def test_fill_after_cancel_pending_and_after_canceled_is_booked(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.apply_event(stream_event(c, E.NEW, S.ACKNOWLEDGED, T0))
    assert h.ledger.request_cancel(c) == S.CANCEL_PENDING
    h.ledger.apply_event(_fill(c, "3", "100", "3", "x1"))
    assert state_of(h, c) == S.CANCEL_PENDING  # stale partial does not undo the cancel request
    h.ledger.apply_event(stream_event(c, E.CANCELED, S.CANCELED, T0, cum="3"))
    assert state_of(h, c) == S.CANCELED
    # a racing stream delivers the remaining execution after CANCELED: booked, never dropped
    late = h.ledger.apply_event(_fill(c, "7", "100", "10", "x2", state=S.FILLED, etype=E.FILL))
    assert late.booked_quantity == Decimal(7)
    assert state_of(h, c) == S.FILLED
    assert h.ledger.positions()[0].quantity == Decimal(10)


def test_partial_fill_then_cancel_keeps_partial_position(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.apply_event(_fill(c, "4", "100", "4", "x1"))
    h.ledger.apply_event(stream_event(c, E.CANCELED, S.CANCELED, T0, cum="4"))
    h.ledger.apply_event(stream_event(c, E.NEW, S.ACKNOWLEDGED, T0))  # late ack
    assert state_of(h, c) == S.CANCELED
    assert h.ledger.positions()[0].quantity == Decimal(4)


def test_rejection_and_expiry(h: Harness) -> None:
    a = _submitted(h)
    h.ledger.apply_event(stream_event(a.client_order_id, E.REJECTED, S.REJECTED, T0))
    assert state_of(h, a.client_order_id) == S.REJECTED
    b = _submitted(h, symbol="QQQ")
    h.ledger.apply_event(stream_event(b.client_order_id, E.NEW, S.ACKNOWLEDGED, T0, broker_order_id="bo-2"))
    h.ledger.apply_event(stream_event(b.client_order_id, E.EXPIRED, S.EXPIRED, T0, broker_order_id="bo-2"))
    assert state_of(h, b.client_order_id) == S.EXPIRED
    assert h.ledger.open_orders() == []


def test_unsent_intents_expire(h: Harness) -> None:
    a = h.factory.make(ttl_s=10)
    h.clock.advance(11)
    assert h.ledger.open_orders() == []  # expired, unsent: not pending any more
    assert h.ledger.expire_stale_intents() == [a.client_order_id]
    assert state_of(h, a.client_order_id) == S.EXPIRED


def test_orphan_and_inconsistent_events_lock_entries(h: Harness) -> None:
    res = h.ledger.apply_event(stream_event("lat-sim-not-ours", E.FILL, S.FILLED, T0, fill=("1", "1"), exec_id="z"))
    assert not res.applied and res.state is None
    assert Flag.LEDGER_ANOMALY.value in flags(h)
    assert h.ledger.positions() == []  # never booked
    i = _submitted(h)
    c = i.client_order_id
    wrong_gw = stream_event(c, E.NEW, S.ACKNOWLEDGED, T0, gateway=GatewayKind.LEAN_PAPER_SIM)
    assert not h.ledger.apply_event(wrong_gw).applied
    h.ledger.apply_event(stream_event(c, E.NEW, S.ACKNOWLEDGED, T0, broker_order_id="bo-A"))
    conflict = h.ledger.apply_event(stream_event(c, E.PARTIAL_FILL, S.PARTIALLY_FILLED, T0, broker_order_id="bo-B",
                                                 fill=("1", "100"), cum="1", exec_id="q"))
    assert not conflict.applied and "conflicts" in conflict.note
    assert h.ledger.positions() == []
    replaced = h.ledger.apply_event(stream_event(c, E.REPLACED, S.ACKNOWLEDGED, T0, broker_order_id="bo-A"))
    assert not replaced.applied and "unsupported" in replaced.note


def test_broker_evidence_for_never_submitted_order_is_anomaly(h: Harness) -> None:
    i = h.factory.make()
    res = h.ledger.apply_event(stream_event(i.client_order_id, E.NEW, S.ACKNOWLEDGED, T0))
    assert not res.applied
    assert state_of(h, i.client_order_id) == S.RISK_APPROVED
    assert Flag.LEDGER_ANOMALY.value in flags(h)


def test_malformed_executions_are_not_booked(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    bad = [
        stream_event(c, E.PARTIAL_FILL, S.PARTIALLY_FILLED, T0, fill=("0", "100"), cum="0", exec_id="m1"),
        stream_event(c, E.PARTIAL_FILL, S.PARTIALLY_FILLED, T0, fill=("-1", "100"), cum="1", exec_id="m2"),
        stream_event(c, E.PARTIAL_FILL, S.PARTIALLY_FILLED, T0, fill=("2", "0"), cum="2", exec_id="m3"),
        stream_event(c, E.PARTIAL_FILL, S.PARTIALLY_FILLED, T0, fill=("5", "100"), cum="3", exec_id="m4"),
        stream_event(c, E.PARTIAL_FILL, S.PARTIALLY_FILLED, T0, fill=("2", "100"), cum=None, exec_id=None),
    ]
    for ev in bad:
        assert h.ledger.apply_event(ev).booked_quantity == 0
    assert h.ledger.fills(c) == []
    assert h.ledger.positions() == []


def test_unknown_resolution_and_reconciled_absent(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.mark_unknown(c, "timeout")
    res = h.ledger.mark_reconciled_absent(c)
    assert res.state == S.RECONCILED
    assert h.ledger.fills(c) == [] and h.ledger.positions() == []  # never invents fills
    # the broker later shows the order after all: corrected to broker truth + anomaly flag
    later = h.ledger.apply_broker_order(_broker_order(i, S.ACKNOWLEDGED, "new"), EventSource.RECONCILE)
    assert later.state == S.ACKNOWLEDGED and later.anomalies
    assert Flag.LEDGER_ANOMALY.value in flags(h)


def test_reconcile_snapshot_catch_up_then_late_reports_are_absorbed(h: Harness) -> None:
    i = _submitted(h)
    c = i.client_order_id
    h.ledger.apply_event(_fill(c, "4", "100", "4", "x1"))  # x2 (6 @ 101) is lost by the stream
    snap = _broker_order(i, S.FILLED, "filled", filled="10", avg="100.6")
    res = h.ledger.apply_broker_order(snap, EventSource.RECONCILE)
    assert res.booked_quantity == Decimal(6) and res.state == S.FILLED
    fills = h.ledger.fills(c)
    assert [f["synthetic"] for f in fills] == [0, 1]
    assert Decimal(fills[1]["price"]) == Decimal("101.00000000")
    # the lost report shows up late: absorbed by the catch-up range, never double-booked
    late = h.ledger.apply_event(_fill(c, "6", "101", "10", "x2", state=S.FILLED, etype=E.FILL))
    assert late.booked_quantity == 0 and late.absorbed_quantity == Decimal(6)
    assert h.ledger.positions()[0].quantity == Decimal(10)
    assert h.ledger.positions()[0].cost_basis == Decimal("1006.00")
    # applying the same broker snapshot again is a no-op
    assert h.ledger.apply_broker_order(snap, EventSource.RECONCILE).duplicate


def test_snapshot_that_contradicts_the_intent_is_not_applied(h: Harness) -> None:
    i = _submitted(h)
    wrong = _broker_order(i, S.ACKNOWLEDGED, "new").model_copy(update={"quantity": Decimal(50)})
    res = h.ledger.apply_broker_order(wrong, EventSource.RECONCILE)
    assert not res.applied and "differs" in res.note
    assert state_of(h, i.client_order_id) == S.SUBMITTING


def test_buy_sell_realized_pl_and_daily_pl(h: Harness) -> None:
    b = _submitted(h, qty="10", limit="100.00")
    h.ledger.apply_event(_fill(b.client_order_id, "10", "100", "10", "b1", state=S.FILLED, etype=E.FILL))
    h.clock.advance(60)
    s1 = _submitted(h, side=Side.SELL, qty="4", limit="105.00")  # type: ignore[arg-type]
    h.ledger.apply_event(stream_event(s1.client_order_id, E.FILL, S.FILLED, h.clock.now(), fill=("4", "105"), cum="4",
                                      exec_id="s1", broker_order_id="bo-s1"))
    [pos] = h.ledger.positions()
    assert pos.quantity == Decimal(6) and pos.avg_entry_price == Decimal("100.00000000")
    assert h.ledger.realized_pl_total() == Decimal("20.00000000")
    assert h.ledger.daily_realized_pl(T0.date()) == Decimal("20.00000000")
    assert h.ledger.daily_realized_pl(T0) == Decimal("20.00000000")
    assert h.ledger.daily_realized_pl((T0 + timedelta(days=1)).date()) == 0


def test_sell_beyond_position_is_anomaly(h: Harness) -> None:
    s = _submitted(h, side=Side.SELL, qty="5")  # type: ignore[arg-type]
    res = h.ledger.apply_event(stream_event(s.client_order_id, E.FILL, S.FILLED, T0, fill=("5", "100"), cum="5", exec_id="s"))
    assert res.anomalies and Flag.LEDGER_ANOMALY.value in flags(h)


def test_query_helpers(h: Harness) -> None:
    a = h.factory.make()
    b = h.factory.make(symbol="QQQ")
    claim(h, b)
    open_ids = {o.client_order_id for o in h.ledger.open_orders()}
    assert open_ids == {a.client_order_id, b.client_order_id}
    fps = h.ledger.recent_fingerprints(T0 - timedelta(seconds=1))
    assert len(fps) == 2 and all(f.startswith("fp_") for f in fps)
    assert h.ledger.recent_fingerprints(T0 + timedelta(seconds=1)) == []
    assert h.ledger.get_order("nope") is None
    assert h.ledger.events(b.client_order_id) == []
    assert h.ledger.transitions(b.client_order_id)[-1]["to_state"] == "SUBMITTING"


def test_event_log_tables_are_append_only(h: Harness) -> None:
    i = _submitted(h)
    h.ledger.apply_event(_fill(i.client_order_id, "1", "100", "1", "x1"))

    def tamper(sql: str):
        return lambda conn: conn.execute(sql)

    for sql in ("UPDATE order_events SET applied=0", "DELETE FROM order_events", "UPDATE fills SET price='1'",
                "DELETE FROM fills", "DELETE FROM order_state_transitions", "UPDATE audit_events SET source='x'"):
        with pytest.raises(sqlite3.IntegrityError):
            h.db.write(tamper(sql))
