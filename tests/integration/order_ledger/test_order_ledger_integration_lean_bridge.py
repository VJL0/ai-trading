"""LEAN engine bridge round trip, server side, with a scripted stand-in for the C# engine (SIMULATED).

These tests ran against ``LeanBridgeGateway(LEAN_PAPER_SIM)`` built by ``build_gateway`` and a deterministic Python
stand-in for the LEAN process -- NOT against the real LEAN engine and NOT against a real Alpaca paper broker. The
stand-in does what docs/architecture/api-contract.md requires of the engine: it holds only the engine token and the
risk engine's *public* key, claims intents, verifies the ECDSA signature over ``signing_string``, executes the
parameters parsed from that signed string, and posts JSON events/heartbeats/rejections that are parsed exactly like
the API does (``json.loads(body, parse_float=Decimal)``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import GatewayKind, OrderState, ReconciliationStatus, TradingMode
from packages.contracts.models import ApprovedOrderIntent
from packages.core.audit import verify_chain
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.signing import IntentSigner, IntentVerifier, load_verifier
from services.order_ledger.bridge import (
    EngineTokenAuth,
    LeanBridgeGateway,
    claim_next_intents,
    ingest_engine_event,
    record_engine_rejection,
    record_heartbeat,
)
from services.order_ledger.entries import EntriesLock, Flag
from services.order_ledger.execution import ExecutionService
from services.order_ledger.ledger import OrderLedger
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import T0, IntentFactory

S = OrderState
SIM = GatewayKind.LEAN_PAPER_SIM


async def until(pred: Callable[[], bool], within_s: float = 3.0) -> None:
    for _ in range(int(within_s / 0.005)):
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached in time")


def _api_parse(body: bytes) -> dict[str, Any]:
    """What services/api does with an engine request body."""
    return json.loads(body, parse_float=Decimal)


@dataclass
class ScriptedEngine:
    """Deterministic stand-in for the C# LEAN process (SIMULATED). Holds no private key and no broker credentials."""

    ledger: OrderLedger
    db: Database
    verifier: IntentVerifier
    engine_id: str = "lean-sim-1"
    lean_order_seq: int = 0
    lean_event_seq: int = 0
    sent: list[bytes] = field(default_factory=list)

    def poll(self, max_n: int = 5) -> list[dict[str, Any]]:
        return claim_next_intents(self.ledger, self.engine_id, max_n)

    def verify_and_parse(self, item: dict[str, Any]) -> dict[str, str] | None:
        """GatedBrokerageModel: verify the signature, then execute only the parameters inside the signed string."""
        if not self.verifier.verify(item["signing_string"], item["signature"]):
            return None
        head, *parts = item["signing_string"].split("|")
        assert head == "lat-intent-v1"
        return dict(p.split("=", 1) for p in parts)

    def post_event(self, order: dict[str, str], lean_order_id: int, status: str, *, fill: tuple[str, str] | None = None,
                   cum: str = "0") -> dict[str, Any]:
        self.lean_event_seq += 1
        body = json.dumps({
            "engine_id": self.engine_id, "client_order_id": order["client_order_id"], "intent_id": order["intent_id"],
            "lean_order_id": lean_order_id, "broker_order_id": None, "status": status,
            "fill_quantity": fill[0] if fill else "0", "fill_price": fill[1] if fill else "0",
            "cum_filled_quantity": cum, "event_ts": "2026-11-04T15:00:05Z", "lean_event_id": self.lean_event_seq,
            "message": "",
        }).encode()
        self.sent.append(body)
        return ingest_engine_event(self.ledger, _api_parse(body), gateway_kind=SIM)

    def replay_all(self) -> list[dict[str, Any]]:
        """An engine restart re-sending every event it ever sent."""
        return [ingest_engine_event(self.ledger, _api_parse(b), gateway_kind=SIM) for b in self.sent]

    def heartbeat(self, *, age_s: float = 0.0) -> dict[str, Any]:
        body = json.dumps({"engine_id": self.engine_id, "mode": "PAPER", "gateway": SIM.value, "algorithm": "LatGated",
                           "lean_commit": "d1c09284", "ts": datetime.now(UTC).isoformat(), "status": "RUNNING"}).encode()
        received = ManualClock(datetime.now(UTC) - timedelta(seconds=age_s))
        return record_heartbeat(self.db, _api_parse(body), clock=received)

    def reject(self, item: dict[str, Any], reason: str) -> dict[str, Any]:
        body = json.dumps({"engine_id": self.engine_id, "client_order_id": item["client_order_id"],
                           "intent_id": item["intent_id"], "reason": reason}).encode()
        return record_engine_rejection(self.db, _api_parse(body))


async def test_lean_bridge_round_trip_with_scripted_engine(tmp_path: Path) -> None:
    """Scripted LEAN stand-in (SIMULATED): entries locked without a heartbeat; an approved intent is queued, claimed
    exactly once, its signature verified with the public key only; LEAN status events drive the ledger through
    SUBMITTED -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED; an engine restart replaying every event changes nothing;
    a tampered signing string is refused by the engine and the order REJECTED; a stale heartbeat (Mac sleep) locks
    entries until a fresh heartbeat plus a CLEAN reconciliation; a cancel before pickup withdraws the intent."""
    settings = Settings(project_root=tmp_path, use_keychain=False, gateway=SIM)
    settings.ensure_dirs()
    signer = IntentSigner.load_or_create(settings.keys_dir)
    auth = EngineTokenAuth(settings.keys_dir)
    token_path = auth.ensure()
    assert auth.verify(token_path.read_text()) and not auth.verify("not-the-token")
    clock = ManualClock(T0)
    db = Database(settings.db_path)
    try:
        ledger = OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier)
        entries = EntriesLock(db, TradingMode.PAPER, clock=clock)
        gw = LeanBridgeGateway(ledger, SIM, clock)  # isolated legacy research harness, not a runtime gateway
        assert isinstance(gw, LeanBridgeGateway) and gw.label.value == "SIMULATED"

        def final_gate(intent: ApprovedOrderIntent) -> list[str]:
            return [] if signer.verifier.verify(intent.signing_string(), intent.signature) else ["bad signature"]

        svc = ExecutionService(ledger, gw, final_gate, clock, entries_lock=entries, watchdog_interval_s=3600)
        rec = Reconciler(db, ledger, gw, entries, clock, confirm_delay_s=0)
        factory = IntentFactory(ledger, signer, clock, mode=TradingMode.PAPER, gateway=SIM, ttl_s=300)
        engine = ScriptedEngine(ledger, db, load_verifier(settings.keys_dir))
        stop = asyncio.Event()
        consumer = asyncio.create_task(svc.run_event_consumer(stop))
        await asyncio.wait_for(svc.consumer_ready.wait(), 2)

        # ---- no engine heartbeat yet: reconciliation is CLEAN against the ledger but entries stay locked
        first = await rec.startup_recovery()
        assert first.entries_locked_after and entries.is_flag_active(Flag.ENGINE_HEARTBEAT_STALE)
        assert engine.heartbeat() == {"ok": True}
        ready = await rec.reconcile()
        assert ready.status == ReconciliationStatus.CLEAN and not ready.entries_locked_after
        assert "SIMULATED" in ready.broker_source

        # ---- submit: queued for the engine (still SUBMITTING), handed out exactly once
        intent = factory.make(qty="10", limit="99.50")
        order = await svc.submit(intent)
        assert order is not None and order.state == S.SUBMITTING and order.broker_status == "queued_for_engine"
        [item] = engine.poll()
        assert engine.poll() == [] and engine.poll() == []
        signed = engine.verify_and_parse(item)
        assert signed is not None
        assert (signed["client_order_id"], signed["symbol"], signed["side"], signed["quantity"], signed["limit_price"],
                signed["gateway"], signed["mode"]) == (intent.client_order_id, "SPY", "BUY", "10", "99.50", SIM.value, "PAPER")

        # ---- engine lifecycle events (LEAN OrderStatus strings), fills as decimal strings
        engine.lean_order_seq += 1
        oid = engine.lean_order_seq
        results = [
            engine.post_event(signed, oid, "New"),
            engine.post_event(signed, oid, "Submitted"),
            engine.post_event(signed, oid, "PartiallyFilled", fill=("4", "99.45"), cum="4"),
            engine.post_event(signed, oid, "Filled", fill=("6", "99.44"), cum="10"),
        ]
        assert [r["state"] for r in results] == ["SUBMITTED", "ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED"]
        assert all(r["accepted"] and not r["duplicate"] for r in results)
        await until(lambda: svc.events_applied >= 4)
        assert ledger.duplicate_count() == 0  # the consumer does not re-apply engine events it streams

        # ---- engine restart: every event re-sent; nothing changes, nothing is booked twice
        replayed = engine.replay_all()
        assert all(r["duplicate"] and not r["applied"] for r in replayed)
        fills = ledger.fills(intent.client_order_id)
        assert [(f["quantity"], f["price"]) for f in fills] == [("4", "99.45"), ("6", "99.44")]
        [pos] = ledger.positions()
        assert pos.quantity == Decimal(10) and pos.avg_entry_price == Decimal("99.444")
        account = await gw.get_account()
        assert account.cash == Decimal("100000") - Decimal("397.80") - Decimal("596.64")
        assert (await rec.reconcile()).status == ReconciliationStatus.CLEAN

        # ---- a signing string altered in transit fails verification: the engine refuses and reports it
        second = factory.make(qty="2", limit="99.00")
        assert await svc.submit(second) is not None
        [item2] = engine.poll()
        tampered = {**item2, "signing_string": item2["signing_string"].replace("quantity=2", "quantity=200")}
        assert engine.verify_and_parse(tampered) is None
        assert engine.reject(item2, "signature verification failed") == {"ok": True, "order_rejected": True}
        assert ledger.get_order(second.client_order_id).state == S.REJECTED  # type: ignore[union-attr]

        # ---- Mac sleep: the heartbeat goes stale -> entries locked; submissions refused until reconciled
        assert engine.heartbeat(age_s=120) == {"ok": True}
        assert svc.watchdog_tick() == [Flag.ENGINE_HEARTBEAT_STALE.value]
        assert entries.is_locked()[0]
        blocked = factory.make(qty="1", limit="99.00")
        assert await svc.submit(blocked) is None
        assert ledger.get_order(blocked.client_order_id).state == S.REJECTED  # type: ignore[union-attr]
        assert ledger.engine_queue_status(blocked.client_order_id) is None and engine.poll() == []
        still = await rec.reconcile()
        assert still.entries_locked_after and entries.is_flag_active(Flag.ENGINE_HEARTBEAT_STALE)
        engine.heartbeat()
        resumed = await rec.reconcile()
        assert resumed.status == ReconciliationStatus.CLEAN and not resumed.entries_locked_after

        # ---- cancel before the engine picked it up: withdrawn, never handed out
        later = factory.make(qty="3", limit="98.00")
        assert await svc.submit(later) is not None
        assert await svc.cancel(later.client_order_id) == S.CANCELED
        assert ledger.engine_queue_status(later.client_order_id) == "WITHDRAWN" and engine.poll() == []

        ok, _, detail = verify_chain(db)
        assert ok, detail
        stop.set()
        await asyncio.wait_for(consumer, 2)
    finally:
        db.close()
