"""LEAN engine bridge (server side) and LeanBridgeGateway. A scripted 'engine' stands in for the C# process."""

from __future__ import annotations

import json
import stat
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, OrderState, ReconciliationStatus, TradingMode
from packages.contracts.interfaces import ExecutionGateway, GatewayRejected
from packages.contracts.models import intent_signing_string
from packages.core.clock import ManualClock
from packages.core.modes import LiveActivationDenied, ModeViolation
from services.order_ledger.bridge import (
    LEAN_STATUS_MAP,
    EngineTokenAuth,
    LeanBridgeGateway,
    claim_next_intents,
    engine_heartbeat_status,
    ingest_engine_event,
    parse_engine_payload,
    record_engine_rejection,
    record_heartbeat,
)
from services.order_ledger.entries import Flag
from services.order_ledger.errors import BrokerPayloadError, CancelNotSupported, MissingPaperCredentials
from services.order_ledger.execution import ExecutionService
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import T0, Harness, IntentFactory, flags, pass_gate, state_of

S = OrderState
SIM = GatewayKind.LEAN_PAPER_SIM


def _lean(h: Harness, wall: ManualClock | None = None) -> tuple[LeanBridgeGateway, IntentFactory]:
    gw = LeanBridgeGateway(h.ledger, SIM, h.clock, wall_clock=wall or h.clock, poll_interval_s=0.005)
    factory = IntentFactory(h.ledger, h.signer, h.clock, gateway=SIM)
    return gw, factory


def _beat(h: Harness, *, status: str = "RUNNING", gateway: str = "LEAN_PAPER_SIM", mode: str = "PAPER") -> dict[str, Any]:
    return record_heartbeat(h.db, {"engine_id": "lean-1", "mode": mode, "gateway": gateway, "algorithm": "LatGated",
                                   "lean_commit": "abc123", "ts": "2026-11-04T15:00:00Z", "status": status},
                            clock=h.clock)


async def _ready(h: Harness, gw: LeanBridgeGateway) -> None:
    _beat(h)
    report = await Reconciler(h.db, h.ledger, gw, h.entries, h.clock, confirm_delay_s=0).reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches


async def _queued(h: Harness) -> tuple[LeanBridgeGateway, Any]:
    gw, factory = _lean(h)
    await _ready(h, gw)
    intent = factory.make()
    svc = ExecutionService(h.ledger, gw, pass_gate, h.clock, entries_lock=h.entries)
    order = await svc.submit(intent)
    assert order is not None and order.state == S.SUBMITTING
    return gw, intent


def _event(intent: Any, status: str, event_id: int, *, fill: tuple[str, str] | None = None, cum: str = "0",
           engine: str = "lean-1", lean_order_id: int = 7) -> dict[str, Any]:
    return {
        "engine_id": engine, "client_order_id": intent.client_order_id, "intent_id": intent.intent_id,
        "lean_order_id": lean_order_id, "broker_order_id": "alp-123", "status": status,
        "fill_quantity": fill[0] if fill else "0", "fill_price": fill[1] if fill else "0",
        "cum_filled_quantity": cum, "event_ts": "2026-11-04T15:00:01Z", "lean_event_id": event_id, "message": "",
    }


# ------------------------------------------------------------------------------------------ auth


def test_engine_token_file_is_private_and_compared_in_constant_time(tmp_path: Path) -> None:
    auth = EngineTokenAuth(tmp_path / "keys")
    path = auth.ensure()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    token = path.read_text().strip()
    assert len(token) >= 32
    assert auth.verify(token) and auth.verify(token + "\n")
    assert not auth.verify(token[:-1] + ("A" if token[-1] != "A" else "B"))
    assert not auth.verify("") and not auth.verify(None)
    auth.ensure()
    assert path.read_text().strip() == token  # never rotated silently
    path.chmod(0o644)
    auth.ensure()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not EngineTokenAuth(tmp_path / "nokeys").verify(token)


# ------------------------------------------------------------------------------------------ claim


async def test_claim_hands_out_each_intent_once_with_signing_material(h: Harness) -> None:
    _, intent = await _queued(h)
    [item] = claim_next_intents(h.ledger, "lean-1", 5)
    assert item["client_order_id"] == intent.client_order_id
    assert item["signing_string"] == intent.signing_string()
    assert item["signing_string"] == intent_signing_string(intent.model_dump(mode="json", exclude={"signature"}))
    assert h.signer.verifier.verify(item["signing_string"], item["signature"])
    assert item["key_id"] == h.signer.key_id and item["signature_alg"] == "ECDSA_P256_SHA256"
    assert claim_next_intents(h.ledger, "lean-1", 5) == []
    assert claim_next_intents(h.ledger, "lean-2", 5) == []
    assert h.ledger.engine_queue_status(intent.client_order_id) == "CLAIMED"
    assert state_of(h, intent.client_order_id) == S.SUBMITTING


async def test_expired_intents_are_never_handed_out(h: Harness) -> None:
    _, intent = await _queued(h)
    h.clock.advance(61)
    assert claim_next_intents(h.ledger, "lean-1", 1) == []
    assert state_of(h, intent.client_order_id) == S.EXPIRED
    assert h.ledger.engine_queue_status(intent.client_order_id) == "EXPIRED"


async def test_unapproved_intents_are_never_queued(h: Harness) -> None:
    gw, factory = _lean(h)
    await _ready(h, gw)
    factory.make()  # RISK_APPROVED but never passed the ExecutionService final gate
    assert claim_next_intents(h.ledger, "lean-1", 10) == []


def test_bad_engine_id_refused(h: Harness) -> None:
    with pytest.raises(BrokerPayloadError):
        claim_next_intents(h.ledger, "lean 1; DROP TABLE", 1)


# ------------------------------------------------------------------------------------------ events


async def test_engine_lifecycle_maps_lean_statuses(h: Harness) -> None:
    _, intent = await _queued(h)
    claim_next_intents(h.ledger, "lean-1", 1)
    k = GatewayKind.LEAN_PAPER_SIM
    assert ingest_engine_event(h.ledger, _event(intent, "Submitted", 1), gateway_kind=k)["state"] == "ACKNOWLEDGED"
    res = ingest_engine_event(h.ledger, _event(intent, "PartiallyFilled", 2, fill=("4", "100.00"), cum="4"), gateway_kind=k)
    assert res == {**res, "accepted": True, "state": "PARTIALLY_FILLED", "duplicate": False}
    dup = ingest_engine_event(h.ledger, _event(intent, "PartiallyFilled", 2, fill=("4", "100.00"), cum="4"), gateway_kind=k)
    assert dup["duplicate"] is True
    done = ingest_engine_event(h.ledger, _event(intent, "Filled", 3, fill=("6", "99.99"), cum="10"), gateway_kind=k)
    assert done["state"] == "FILLED"
    late = ingest_engine_event(h.ledger, _event(intent, "Submitted", 4), gateway_kind=k)
    assert late["state"] == "FILLED"  # out-of-order status never regresses
    [pos] = h.ledger.positions()
    assert pos.quantity == Decimal(10) and pos.cost_basis == Decimal("999.94")
    rec = h.ledger.get_record(intent.client_order_id)
    assert rec is not None and rec.broker_order_id == "alp-123"


@pytest.mark.parametrize(("status", "state"), [("Canceled", "CANCELED"), ("Invalid", "REJECTED"),
                                               ("CancelPending", "CANCEL_PENDING"), ("New", "SUBMITTED")])
async def test_terminal_and_pending_statuses(h: Harness, status: str, state: str) -> None:
    _, intent = await _queued(h)
    claim_next_intents(h.ledger, "lean-1", 1)
    if status == "CancelPending":
        ingest_engine_event(h.ledger, _event(intent, "Submitted", 1), gateway_kind=SIM)
    res = ingest_engine_event(h.ledger, _event(intent, status, 2), gateway_kind=SIM)
    assert res["state"] == state


async def test_none_and_update_submitted_statuses(h: Harness) -> None:
    _, intent = await _queued(h)
    claim_next_intents(h.ledger, "lean-1", 1)
    res = ingest_engine_event(h.ledger, _event(intent, "None", 1), gateway_kind=SIM)
    assert res["accepted"] and res["state"] == "SUBMITTING"
    upd = ingest_engine_event(h.ledger, _event(intent, "UpdateSubmitted", 2), gateway_kind=SIM)
    assert upd["accepted"] is False  # order updates were never requested: refused + anomaly
    assert Flag.LEDGER_ANOMALY.value in flags(h)
    assert set(LEAN_STATUS_MAP) == {"new", "submitted", "partiallyfilled", "filled", "canceled", "invalid",
                                    "cancelpending", "updatesubmitted", "none"}


async def test_malformed_engine_events_are_refused(h: Harness) -> None:
    _, intent = await _queued(h)
    claim_next_intents(h.ledger, "lean-1", 1)
    raw = (json.dumps(_event(intent, "PartiallyFilled", 1, cum="4"))
           .replace('"fill_quantity": "0"', '"fill_quantity": 4').replace('"fill_price": "0"', '"fill_price": 100.1'))
    parsed = ingest_engine_event(h.ledger, raw, gateway_kind=SIM)  # JSON numbers become Decimal, never float
    assert parsed["accepted"] is True and parsed["state"] == "PARTIALLY_FILLED"
    [fill] = h.ledger.fills(intent.client_order_id)
    assert Decimal(fill["price"]) == Decimal("100.1") and Decimal(fill["quantity"]) == Decimal(4)
    no_price = {**_event(intent, "PartiallyFilled", 9, fill=("2", "0"), cum="6")}
    assert ingest_engine_event(h.ledger, no_price, gateway_kind=SIM)["accepted"] is False  # execution without price
    bad = [
        {**_event(intent, "Filled", 2), "fill_quantity": 1.5},  # a binary float smuggled in a dict
        {**_event(intent, "Filled", 3), "lean_order_id": "x"},
        {**_event(intent, "Filled", 4), "event_ts": "not-a-time"},
        {**_event(intent, "Filled", 5), "status": "Exploded"},
        {**_event(intent, "Filled", 6), "client_order_id": ""},
        {**_event(intent, "Filled", 7), "cum_filled_quantity": "NaN"},
    ]
    for payload in bad:
        assert ingest_engine_event(h.ledger, payload, gateway_kind=SIM)["accepted"] is False, payload
    assert ingest_engine_event(h.ledger, b"{not json", gateway_kind=SIM)["accepted"] is False
    assert state_of(h, intent.client_order_id) == S.PARTIALLY_FILLED
    assert len(h.ledger.fills(intent.client_order_id)) == 1
    assert h.db.query_one("SELECT COUNT(*) n FROM failures WHERE component='lean_bridge'")["n"] >= 7  # type: ignore[index]
    with pytest.raises(BrokerPayloadError):
        parse_engine_payload({"x": [1, {"y": 2.0}]})


async def test_engine_event_for_wrong_intent_or_gateway_is_refused(h: Harness) -> None:
    _, intent = await _queued(h)
    claim_next_intents(h.ledger, "lean-1", 1)
    wrong_intent = {**_event(intent, "Filled", 1, fill=("10", "100"), cum="10"), "intent_id": "intent_forged"}
    assert ingest_engine_event(h.ledger, wrong_intent, gateway_kind=SIM)["accepted"] is False
    assert ingest_engine_event(h.ledger, _event(intent, "Filled", 2, fill=("10", "100"), cum="10"),
                               gateway_kind=GatewayKind.FAKE)["accepted"] is False
    assert h.ledger.positions() == []
    assert Flag.LEDGER_ANOMALY.value in flags(h)


async def test_engine_event_for_unknown_order_is_anomaly(h: Harness) -> None:
    gw, factory = _lean(h)
    ghost = factory.make(record=False)
    res = ingest_engine_event(h.ledger, _event(ghost, "Filled", 1, fill=("1", "100"), cum="1"), gateway_kind=SIM)
    assert res["accepted"] is False
    assert h.ledger.positions() == []
    assert Flag.LEDGER_ANOMALY.value in flags(h)
    assert gw.kind == SIM


# ------------------------------------------------------------------------------------------ heartbeat / rejection


def test_heartbeat_recorded_and_staleness_detected(h: Harness) -> None:
    wall = ManualClock(T0)
    assert record_heartbeat(h.db, {"engine_id": "lean-1", "mode": "PAPER", "gateway": "LEAN_PAPER_SIM",
                                   "ts": "2026-11-04T15:00:00Z", "status": "RUNNING"}, clock=wall)["ok"]
    ok, detail, received = engine_heartbeat_status(h.db, wall, 15)
    assert ok and received == T0, detail
    wall.advance(16)
    ok, detail, _ = engine_heartbeat_status(h.db, wall, 15)
    assert not ok and "stale" in detail
    gw = LeanBridgeGateway(h.ledger, SIM, h.clock, wall_clock=wall)
    assert gw.engine_health()[0] is False


def test_engine_running_another_gateway_or_mode_is_unhealthy(h: Harness) -> None:
    """A fresh heartbeat from an engine configured for a different (paper) gateway or mode fails closed."""
    gw = LeanBridgeGateway(h.ledger, SIM, h.clock, wall_clock=h.clock)
    _beat(h)
    assert gw.engine_health() == (True, "LEAN heartbeat 0s ago (RUNNING)")
    _beat(h, gateway="LEAN_ALPACA_PAPER")
    ok, detail = gw.engine_health()
    assert not ok and "expected LEAN_PAPER_SIM" in detail
    _beat(h, mode="REPLAY")
    ok, detail = gw.engine_health()
    assert not ok and "expected PAPER" in detail


def test_heartbeat_refuses_live_or_unknown_gateways(h: Harness) -> None:
    for mode, gateway in (("LIVE_APPROVAL", "LEAN_PAPER_SIM"), ("PAPER", "LEAN_ALPACA_LIVE"), ("PAPER", "FAKE")):
        res = _beat(h, mode=mode, gateway=gateway)
        assert res["ok"] is False
    assert h.entries.is_flag_active(Flag.ENGINE_HEARTBEAT_STALE)
    assert not record_heartbeat(h.db, {"engine_id": "lean-1"})["ok"]
    assert not engine_heartbeat_status(h.db, h.clock, 15)[0]
    _beat(h, status="STOPPED")
    ok, detail, _ = engine_heartbeat_status(h.db, h.clock, 15)
    assert not ok and "STOPPED" in detail


async def test_engine_rejection_rejects_the_order_and_is_audited(h: Harness) -> None:
    _, intent = await _queued(h)
    claim_next_intents(h.ledger, "lean-1", 1)
    res = record_engine_rejection(h.db, {"engine_id": "lean-1", "client_order_id": intent.client_order_id,
                                         "intent_id": intent.intent_id, "reason": "signature mismatch"}, clock=h.clock)
    assert res == {"ok": True, "order_rejected": True}
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.db.query_one("SELECT COUNT(*) n FROM failures WHERE kind='engine_rejection'")["n"] == 1  # type: ignore[index]
    assert h.db.query_one("SELECT COUNT(*) n FROM audit_events WHERE event_type='ENGINE_REJECTION'")["n"] == 1  # type: ignore[index]
    assert record_engine_rejection(h.db, {"engine_id": "lean-1", "client_order_id": None, "reason": "bad"})["ok"]
    assert not record_engine_rejection(h.db, {"engine_id": "lean-1"})["ok"]


# ------------------------------------------------------------------------------------------ gateway


def test_gateway_construction_rules(h: Harness) -> None:
    gw = LeanBridgeGateway(h.ledger, SIM, h.clock)
    assert isinstance(gw, ExecutionGateway) and gw.is_paper
    with pytest.raises(LiveActivationDenied):
        LeanBridgeGateway(h.ledger, GatewayKind.LEAN_ALPACA_LIVE, h.clock)
    with pytest.raises(ModeViolation):
        LeanBridgeGateway(h.ledger, GatewayKind.FAKE, h.clock)
    with pytest.raises(MissingPaperCredentials):
        LeanBridgeGateway(h.ledger, GatewayKind.LEAN_ALPACA_PAPER, h.clock)
    from services.order_ledger.ledger import OrderLedger

    with pytest.raises(ModeViolation):
        LeanBridgeGateway(OrderLedger(h.db, h.clock, TradingMode.REPLAY), SIM, h.clock)


async def test_gateway_submit_refuses_foreign_intents(h: Harness) -> None:
    gw, _ = _lean(h)
    with pytest.raises(GatewayRejected):
        await gw.submit(h.factory.make(record=False))  # FAKE-gateway intent
    _, factory = _lean(h)
    unclaimed = factory.make()
    with pytest.raises(GatewayRejected, match="not SUBMITTING"):
        await gw.submit(unclaimed)


async def test_cancel_before_pickup_withdraws_after_pickup_not_supported(h: Harness) -> None:
    gw, intent = await _queued(h)
    await gw.cancel(intent.client_order_id)
    assert state_of(h, intent.client_order_id) == S.CANCELED
    assert claim_next_intents(h.ledger, "lean-1", 1) == []
    gw2, intent2 = gw, None
    _, factory = _lean(h)
    intent2 = factory.make(symbol="QQQ")
    await ExecutionService(h.ledger, gw2, pass_gate, h.clock, entries_lock=h.entries).submit(intent2)
    claim_next_intents(h.ledger, "lean-1", 1)
    with pytest.raises(CancelNotSupported):
        await gw2.cancel(intent2.client_order_id)


async def test_order_events_stream_yields_ingested_engine_events(h: Harness) -> None:
    import asyncio

    gw, intent = await _queued(h)
    stream = gw.order_events()
    claim_next_intents(h.ledger, "lean-1", 1)
    ingest_engine_event(h.ledger, _event(intent, "Submitted", 1), gateway_kind=SIM)
    ev = await asyncio.wait_for(stream.__anext__(), 1)
    assert ev.client_order_id == intent.client_order_id and ev.state == S.ACKNOWLEDGED
    assert h.ledger.apply_event(ev).duplicate  # the consumer re-applying it is a no-op


async def test_sim_state_reads_are_ledger_derived(h: Harness) -> None:
    gw, intent = await _queued(h)
    assert await gw.get_order_by_client_id(intent.client_order_id) is None  # no engine evidence yet
    claim_next_intents(h.ledger, "lean-1", 1)
    ingest_engine_event(h.ledger, _event(intent, "Filled", 1, fill=("10", "100"), cum="10"), gateway_kind=SIM)
    acct = await gw.get_account()
    assert acct.cash == Decimal("99000") and acct.source == "LEDGER_LEAN_PAPER_SIM"
    [pos] = await gw.get_positions()
    assert pos.source == "LEDGER_LEAN_PAPER_SIM"
    assert [o.client_order_id for o in await gw.list_orders(status="closed")] == [intent.client_order_id]
    assert "SIMULATED" in gw.source_label
    report = await Reconciler(h.db, h.ledger, gw, h.entries, h.clock, confirm_delay_s=0).reconcile()
    assert report.status == ReconciliationStatus.CLEAN


async def test_reconcile_keeps_lock_while_engine_heartbeat_stale(h: Harness) -> None:
    wall = ManualClock(T0)
    gw, _ = _lean(h, wall)
    record_heartbeat(h.db, {"engine_id": "lean-1", "mode": "PAPER", "gateway": "LEAN_PAPER_SIM",
                            "ts": "2026-11-04T15:00:00Z", "status": "RUNNING"}, clock=wall)
    rec = Reconciler(h.db, h.ledger, gw, h.entries, h.clock, confirm_delay_s=0)
    assert (await rec.reconcile()).entries_locked_after is False
    wall.advance(timedelta(minutes=10).total_seconds())  # engine silent (Mac slept / engine died)
    report = await rec.reconcile()
    assert report.entries_locked_after and h.entries.is_flag_active(Flag.ENGINE_HEARTBEAT_STALE)
    record_heartbeat(h.db, {"engine_id": "lean-1", "mode": "PAPER", "gateway": "LEAN_PAPER_SIM",
                            "ts": "2026-11-04T15:10:00Z", "status": "RUNNING"}, clock=wall)
    assert (await rec.reconcile()).entries_locked_after is False


async def test_queued_intents_are_withdrawn_not_handed_out_while_entries_locked(h: Harness) -> None:
    """Stop-new-orders pressed after an intent was queued: the engine gets nothing, the order is REJECTED (never
    sent) and the approval cannot be released later, even once the halt is cleared."""
    gw, intent = await _queued(h)
    halt = h.entries.lock("operator pressed stop-new-orders", "operator")
    assert claim_next_intents(h.ledger, "lean-1", 5) == []
    assert state_of(h, intent.client_order_id) == S.REJECTED
    assert h.ledger.engine_queue_status(intent.client_order_id) == "WITHDRAWN"
    assert h.db.query_one("SELECT COUNT(*) n FROM audit_events WHERE event_type='ENGINE_INTENTS_WITHDRAWN'")["n"] == 1  # type: ignore[index]
    report = await Reconciler(h.db, h.ledger, gw, h.entries, h.clock, confirm_delay_s=0).reconcile()
    assert report.status == ReconciliationStatus.CLEAN
    assert h.entries.clear_halt(halt, by="operator", reason="resume")
    assert not h.entries.is_locked()[0]
    assert claim_next_intents(h.ledger, "lean-1", 5) == []
    assert state_of(h, intent.client_order_id) == S.REJECTED
