"""LEAN engine bridge, server side (docs/architecture/api-contract.md "Engine bridge").

The integrator mounts these plain functions in FastAPI behind the engine token. Data flow:

* ``ExecutionService.submit`` (final gate passed, order claimed => SUBMITTING) calls
  ``LeanBridgeGateway.submit`` which *queues* the signed intent (``engine_intent_queue``).
* ``claim_next_intents`` hands queued intents to the engine with a compare-and-set QUEUED -> CLAIMED: never
  twice, never after expiry. The engine's C# ``GatedBrokerageModel`` verifies the ECDSA signature over
  ``signing_string`` and executes exactly the parsed signed parameters.
* ``ingest_engine_event`` maps LEAN ``OrderStatus`` updates onto ledger events (dedupe by engine id, client
  order id, LEAN order id and LEAN event id; JSON floats rejected; quantities parsed as Decimal).
* ``record_heartbeat`` / ``record_engine_rejection`` keep engine liveness and gate refusals auditable.

``LeanBridgeGateway`` implements ``ExecutionGateway`` for LEAN_PAPER_SIM / LEAN_ALPACA_PAPER. Broker state
reads delegate to the Alpaca PAPER reader (LEAN_ALPACA_PAPER) or, for LEAN_PAPER_SIM, to ledger-derived
state (explicitly labelled: LEAN's PaperBrokerage keeps no state outside the engine process).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from packages.contracts.enums import (
    EventSource,
    ExecutionLabel,
    GatewayKind,
    OrderEventType,
    OrderState,
    Side,
)
from packages.contracts.interfaces import BrokerStateReader, GatewayRejected
from packages.contracts.models import (
    AccountSnapshot,
    ApprovedOrderIntent,
    BrokerOrder,
    OrderEvent,
    PositionSnapshot,
)
from packages.core import modes
from packages.core.audit import append_audit, get_state, record_failure, set_state
from packages.core.clock import Clock, SystemClock
from packages.core.db import Database, utc_iso
from packages.core.ids import new_id

from .entries import EntriesLock, Flag, set_flag_conn
from .errors import BrokerPayloadError, CancelNotSupported, LedgerError, MissingPaperCredentials
from .ledger import OrderLedger, reject_submitting_conn
from .state_machine import IN_DOUBT, PRE_SUBMISSION, TERMINAL
from .timeutil import parse_iso

ENGINE_TOKEN_FILE = "engine.token"  # noqa: S105 - a file name, not a secret
LEAN_GATEWAYS = frozenset({GatewayKind.LEAN_PAPER_SIM, GatewayKind.LEAN_ALPACA_PAPER})
HEALTHY_ENGINE_STATUSES = frozenset({"RUNNING", "UP", "OK", "READY"})
_ENGINE_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,100}$")

LEAN_STATUS_MAP: dict[str, tuple[OrderEventType, OrderState | None]] = {
    "new": (OrderEventType.PENDING_NEW, OrderState.SUBMITTED),
    "submitted": (OrderEventType.NEW, OrderState.ACKNOWLEDGED),
    "partiallyfilled": (OrderEventType.PARTIAL_FILL, OrderState.PARTIALLY_FILLED),
    "filled": (OrderEventType.FILL, OrderState.FILLED),
    "canceled": (OrderEventType.CANCELED, OrderState.CANCELED),
    "invalid": (OrderEventType.REJECTED, OrderState.REJECTED),
    "cancelpending": (OrderEventType.PENDING_CANCEL, OrderState.CANCEL_PENDING),
    "updatesubmitted": (OrderEventType.REPLACED, None),  # unsupported: never requested => anomaly
    "none": (OrderEventType.UNKNOWN, None),  # LEAN placeholder status: recorded, no state change
}
"""LEAN ``OrderStatus`` (case-insensitive) -> (event type, implied ledger state)."""


# ------------------------------------------------------------------------------------------ auth


class EngineTokenAuth:
    """Engine bearer token at ``var/<ns>/keys/engine.token`` (0600), compared in constant time."""

    def __init__(self, keys_dir: Path, filename: str = ENGINE_TOKEN_FILE) -> None:
        self.path = keys_dir / filename

    def ensure(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w") as fh:
                    fh.write(secrets.token_urlsafe(32))
        if self.path.stat().st_mode & 0o077:
            self.path.chmod(0o600)
        return self.path

    def _expected(self) -> str:
        try:
            return self.path.read_text().strip()
        except OSError:
            return ""

    def verify(self, presented: str | None) -> bool:
        expected = self._expected()
        if not presented or not expected:
            return False
        return hmac.compare_digest(presented.strip().encode(), expected.encode())


# ------------------------------------------------------------------------------------------ parsing


def _reject_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise BrokerPayloadError(f"binary float at {path} rejected; send decimal strings")
    if isinstance(obj, dict):
        for k, v in obj.items():  # pyright: ignore[reportUnknownVariableType]
            _reject_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            _reject_floats(v, f"{path}[{i}]")


def parse_engine_payload(payload: dict[str, Any] | str | bytes) -> dict[str, Any]:
    """JSON (numbers -> Decimal via ``parse_float=Decimal``) or an already-parsed dict; floats refused."""
    if isinstance(payload, (str, bytes)):
        try:
            data = json.loads(payload, parse_float=Decimal)
        except (ValueError, TypeError) as exc:
            raise BrokerPayloadError(f"engine payload is not valid JSON: {exc}") from exc
    else:
        data = payload
    if not isinstance(data, dict):
        raise BrokerPayloadError("engine payload must be a JSON object")
    _reject_floats(data)
    return data  # pyright: ignore[reportUnknownVariableType]


def _decimal(data: dict[str, Any], key: str) -> Decimal | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool | float):
        raise BrokerPayloadError(f"{key}: binary float/bool rejected")
    try:
        out = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise BrokerPayloadError(f"{key}: not a decimal ({value!r})") from exc
    if not out.is_finite():
        raise BrokerPayloadError(f"{key}: non-finite")
    return out


def _required_str(data: dict[str, Any], key: str, maxlen: int = 200) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > maxlen:
        raise BrokerPayloadError(f"{key}: required non-empty string")
    return value.strip()


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise BrokerPayloadError(f"{key}: integer required")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise BrokerPayloadError(f"{key}: integer required")


def _engine_id(data: dict[str, Any]) -> str:
    eid = _required_str(data, "engine_id", 100)
    if not _ENGINE_ID.match(eid):
        raise BrokerPayloadError("engine_id has illegal characters")
    return eid


def _ts(data: dict[str, Any], key: str) -> datetime:
    raw = data.get(key)
    if not isinstance(raw, str):
        raise BrokerPayloadError(f"{key}: ISO-8601 UTC timestamp required")
    try:
        return parse_iso(raw)
    except ValueError as exc:
        raise BrokerPayloadError(f"{key}: {exc}") from exc


# ------------------------------------------------------------------------------------------ bridge functions


def claim_next_intents(ledger: OrderLedger, engine_id: str, max_n: int = 1) -> list[dict[str, Any]]:
    """``GET /api/v1/engine/intents/next``: atomic claim; each intent is handed out at most once.

    Fail closed: while entries are locked (operator halt, mismatch, unknown orders, stale state, restart) nothing is
    handed out and every queued intent is withdrawn (order REJECTED, never sent), so an approval granted before the
    lock is never released to the engine afterwards.
    """
    if not isinstance(engine_id, str) or not _ENGINE_ID.match(engine_id):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise BrokerPayloadError("engine_id has illegal characters")
    locked, reasons = EntriesLock(ledger.db, ledger.mode, clock=ledger.clock).is_locked()
    if locked:
        why = "entries locked at engine poll: " + "; ".join(reasons[:3])
        withdrawn = ledger.withdraw_queued_engine_intents(why)
        if withdrawn:
            at = ledger.clock.now()
            ledger.db.write(lambda c: append_audit(
                c, event_type="ENGINE_INTENTS_WITHDRAWN", source="lean_bridge", mode=ledger.mode.value, ts=at,
                payload={"engine_id": engine_id, "client_order_ids": withdrawn, "reason": why[:500]},
            ))
        return []
    intents = ledger.claim_engine_intents(engine_id, max(1, min(int(max_n), 10)))
    return [
        {
            "intent_id": i.intent_id, "client_order_id": i.client_order_id, "signing_string": i.signing_string(),
            "signature": i.signature, "key_id": i.key_id, "signature_alg": i.signature_alg,
        }
        for i in intents
    ]


def _bridge_failure(ledger: OrderLedger, kind: str, detail: str, *, lock: bool) -> None:
    at = ledger.clock.now()

    def _w(conn: sqlite3.Connection) -> None:
        record_failure(conn, component="lean_bridge", kind=kind, detail=detail)
        append_audit(conn, event_type="ENGINE_EVENT_REFUSED", source="lean_bridge", mode=ledger.mode.value, ts=at,
                     payload={"kind": kind, "detail": detail[:500]})
        if lock:
            set_flag_conn(conn, Flag.LEDGER_ANOMALY, reason=f"engine bridge: {detail}", source="lean_bridge",
                          mode=ledger.mode.value, at=at)

    ledger.db.write(_w)


def ingest_engine_event(
    ledger: OrderLedger, payload: dict[str, Any] | str | bytes, *, gateway_kind: GatewayKind
) -> dict[str, Any]:
    """``POST /api/v1/engine/events`` -> ``{"accepted", "state", "duplicate"}`` (plus ``note``)."""
    try:
        data = parse_engine_payload(payload)
        engine_id = _engine_id(data)
        coid = _required_str(data, "client_order_id", 128)
        intent_id = _required_str(data, "intent_id", 128)
        lean_order_id = _required_int(data, "lean_order_id")
        lean_event_id = _required_int(data, "lean_event_id")
        status_raw = _required_str(data, "status", 40)
        event_ts = _ts(data, "event_ts")
        fill_qty = _decimal(data, "fill_quantity")
        fill_px = _decimal(data, "fill_price")
        cum = _decimal(data, "cum_filled_quantity")
    except BrokerPayloadError as exc:
        _bridge_failure(ledger, "malformed_engine_event", str(exc), lock=False)
        return {"accepted": False, "state": None, "duplicate": False, "note": f"malformed: {exc}"}
    if gateway_kind not in LEAN_GATEWAYS:
        _bridge_failure(ledger, "engine_event_wrong_gateway", f"engine event while gateway is {gateway_kind.value}", lock=True)
        return {"accepted": False, "state": None, "duplicate": False,
                "note": f"gateway {gateway_kind.value} is not a LEAN gateway"}
    mapped = LEAN_STATUS_MAP.get(status_raw.replace("_", "").lower())
    if mapped is None:
        _bridge_failure(ledger, "unknown_lean_status", f"{coid}: unknown LEAN status {status_raw!r}", lock=True)
        return {"accepted": False, "state": None, "duplicate": False, "note": f"unknown status {status_raw!r}"}
    etype, implied = mapped
    rec = ledger.get_record(coid)
    if rec is not None and (rec.intent_id != intent_id or rec.gateway != gateway_kind):
        _bridge_failure(ledger, "engine_event_mismatch",
                        f"{coid}: engine intent {intent_id}/{gateway_kind.value} != ledger {rec.intent_id}/{rec.gateway.value}",
                        lock=True)
        return {"accepted": False, "state": rec.state.value, "duplicate": False, "note": "intent/gateway mismatch"}
    is_fill = etype in (OrderEventType.PARTIAL_FILL, OrderEventType.FILL)
    side = rec.side if rec is not None else None
    if is_fill:
        if fill_qty is None or fill_px is None or fill_qty == 0:
            fill_qty = fill_px = None  # status-only update (no execution in this message)
        else:
            if side == Side.BUY and fill_qty < 0:
                _bridge_failure(ledger, "malformed_engine_event", f"{coid}: negative fill quantity on a BUY", lock=True)
                return {"accepted": False, "state": rec.state.value if rec else None, "duplicate": False, "note": "bad fill sign"}
            fill_qty = abs(fill_qty)
            if fill_px <= 0:
                _bridge_failure(ledger, "malformed_engine_event", f"{coid}: non-positive fill price", lock=True)
                return {"accepted": False, "state": rec.state.value if rec else None, "duplicate": False,
                        "note": "bad fill price"}
    else:
        fill_qty = fill_px = None
    if cum is not None:
        cum = abs(cum)
    state = implied if implied is not None else (rec.state if rec is not None else OrderState.UNKNOWN)
    if implied is None and etype == OrderEventType.UNKNOWN and rec is None:
        state = OrderState.UNKNOWN
    broker_oid = data.get("broker_order_id")
    event_id = f"lean-{engine_id}-{coid}-{lean_order_id}-{lean_event_id}"
    ev = OrderEvent(
        event_id=event_id[:300], client_order_id=coid,
        broker_order_id=str(broker_oid)[:128] if isinstance(broker_oid, str) and broker_oid.strip() else None,
        event_type=etype, state=state, source=EventSource.STREAM, gateway=gateway_kind, fill_quantity=fill_qty,
        fill_price=fill_px, cum_filled_quantity=cum,
        broker_exec_id=f"lean:{engine_id}:{coid}:{lean_order_id}:{lean_event_id}" if fill_qty is not None else None,
        broker_ts=event_ts, received_ts=ledger.clock.now(), sequence=lean_event_id,
        reason=str(data.get("message") or "")[:500] or None,
        raw={"status": status_raw, "engine_id": engine_id, "lean_order_id": lean_order_id, "lean_event_id": lean_event_id},
    )
    res = ledger.apply_event(ev)
    accepted = rec is not None and (implied is not None or etype == OrderEventType.UNKNOWN)
    if etype == OrderEventType.REPLACED:
        accepted = False
    return {
        "accepted": bool(accepted), "state": res.state.value if res.state else None, "duplicate": res.duplicate,
        "applied": res.applied, "note": res.note[:500],
    }


def record_heartbeat(db: Database, payload: dict[str, Any] | str | bytes, *, clock: Clock | None = None) -> dict[str, Any]:
    """``POST /api/v1/engine/heartbeat`` -> system_state ``engine_heartbeat`` (wall-clock ``received_at``)."""
    wall = clock or SystemClock()
    now = wall.now()
    try:
        data = parse_engine_payload(payload)
        engine_id = _engine_id(data)
        mode = _required_str(data, "mode", 40).upper()
        gateway = _required_str(data, "gateway", 40).upper()
        status = _required_str(data, "status", 40).upper()
        engine_ts = _ts(data, "ts")
    except BrokerPayloadError as exc:
        problem = str(exc)
        db.write(lambda c: record_failure(c, component="lean_bridge", kind="malformed_heartbeat", detail=problem))
        return {"ok": False, "note": f"malformed: {problem}"}
    if "LIVE" in mode or gateway == GatewayKind.LEAN_ALPACA_LIVE.value or gateway not in {g.value for g in LEAN_GATEWAYS}:
        detail = f"engine {engine_id} reports mode={mode} gateway={gateway}: refused (paper/simulated LEAN gateways only)"

        def _deny(conn: sqlite3.Connection) -> None:
            record_failure(conn, component="lean_bridge", kind="engine_mode_refused", detail=detail)
            append_audit(conn, event_type="ENGINE_HEARTBEAT_REFUSED", source="lean_bridge", mode=mode, ts=now,
                         payload={"detail": detail})
            set_flag_conn(conn, Flag.ENGINE_HEARTBEAT_STALE, reason=detail, source="lean_bridge", mode=mode, at=now)

        db.write(_deny)
        return {"ok": False, "note": detail}
    value = {
        "engine_id": engine_id, "mode": mode, "gateway": gateway, "status": status,
        "algorithm": str(data.get("algorithm") or "")[:200], "lean_commit": str(data.get("lean_commit") or "")[:80],
        "ts": utc_iso(engine_ts), "received_at": utc_iso(now),
    }
    db.write(lambda c: set_state(c, "engine_heartbeat", value))
    return {"ok": True}


def record_engine_rejection(
    db: Database, payload: dict[str, Any] | str | bytes, *, clock: Clock | None = None
) -> dict[str, Any]:
    """``POST /api/v1/engine/rejections``: the C# GatedBrokerageModel refused an order (never sent)."""
    at = (clock or SystemClock()).now()
    try:
        data = parse_engine_payload(payload)
        engine_id = _engine_id(data)
        reason = _required_str(data, "reason", 2000)
    except BrokerPayloadError as exc:
        problem = str(exc)
        db.write(lambda c: record_failure(c, component="lean_bridge", kind="malformed_rejection", detail=problem))
        return {"ok": False, "note": f"malformed: {problem}"}
    coid = data.get("client_order_id") if isinstance(data.get("client_order_id"), str) else None
    intent_id = data.get("intent_id") if isinstance(data.get("intent_id"), str) else None

    def _w(conn: sqlite3.Connection) -> bool:
        record_failure(conn, component="lean_gate", kind="engine_rejection", detail=f"{engine_id} {coid or '-'}: {reason}")
        row = None
        if coid:
            row = conn.execute("SELECT mode, causal_id, intent_id FROM orders WHERE client_order_id=?", (coid,)).fetchone()
        changed = False
        if row is not None and (intent_id is None or intent_id == row["intent_id"]):
            changed = reject_submitting_conn(conn, str(coid), reason, at)
        append_audit(conn, event_type="ENGINE_REJECTION", source=f"engine:{engine_id}"[:200],
                     mode=row["mode"] if row else "UNKNOWN", causal_id=row["causal_id"] if row else None, ts=at,
                     payload={"client_order_id": coid, "intent_id": intent_id, "reason": reason[:500], "order_rejected": changed})
        return changed

    changed = db.write(_w)
    return {"ok": True, "order_rejected": changed}


def engine_heartbeat_status(
    db: Database,
    wall_clock: Clock,
    max_age_s: float,
    *,
    expected_gateway: GatewayKind | None = None,
    expected_mode: str | None = None,
) -> tuple[bool, str, datetime | None]:
    """``(ok, detail, received_at)``. Not ok (fail closed) when missing, stale, unhealthy, or when the engine
    reports a gateway/mode other than the one this process is configured for."""
    hb = get_state(db, "engine_heartbeat")
    if not isinstance(hb, dict) or not hb.get("received_at"):
        return False, "no LEAN engine heartbeat received", None
    try:
        received = parse_iso(str(hb["received_at"]))  # pyright: ignore[reportUnknownArgumentType]
    except ValueError:
        return False, "unparseable LEAN heartbeat", None
    age = (wall_clock.now() - received).total_seconds()
    status = str(hb.get("status", "")).upper()  # pyright: ignore[reportUnknownArgumentType]
    if age < 0 or age > max_age_s:
        return False, f"LEAN heartbeat stale ({age:.0f}s old, max {max_age_s:.0f}s)", received
    if status not in HEALTHY_ENGINE_STATUSES:
        return False, f"LEAN engine status {status or 'missing'}", received
    reported_gateway = str(hb.get("gateway", "")).upper()  # pyright: ignore[reportUnknownArgumentType]
    if expected_gateway is not None and reported_gateway != expected_gateway.value:
        return False, f"LEAN engine runs gateway {reported_gateway or 'missing'}, expected {expected_gateway.value}", received
    reported_mode = str(hb.get("mode", "")).upper()  # pyright: ignore[reportUnknownArgumentType]
    if expected_mode is not None and reported_mode != expected_mode.upper():
        return False, f"LEAN engine runs mode {reported_mode or 'missing'}, expected {expected_mode.upper()}", received
    return True, f"LEAN heartbeat {age:.0f}s ago ({status})", received


def check_engine_heartbeat(entries: EntriesLock, wall_clock: Clock, max_age_s: float = 15.0) -> bool:
    """Raise ``engine_heartbeat_stale`` (entries locked until a CLEAN reconciliation) when stale. Returns ok."""
    ok, detail, _ = engine_heartbeat_status(entries.db, wall_clock, max_age_s)
    if not ok:
        entries.set_flag(Flag.ENGINE_HEARTBEAT_STALE, detail, "lean_bridge")
    return ok


# ------------------------------------------------------------------------------------------ gateway


class _LedgerTail:
    """Yields engine (STREAM) events of one LEAN gateway recorded after subscription."""

    def __init__(self, gw: LeanBridgeGateway) -> None:
        self._gw = gw
        row = gw.ledger.db.query_one("SELECT COALESCE(MAX(seq), 0) AS s FROM order_events")
        self._seq = int(row["s"]) if row else 0
        self._buffer: list[OrderEvent] = []

    def __aiter__(self) -> _LedgerTail:
        return self

    async def __anext__(self) -> OrderEvent:
        while True:
            if self._buffer:
                return self._buffer.pop(0)
            rows = self._gw.ledger.db.query(
                "SELECT seq, payload_json FROM order_events WHERE seq > ? AND gateway=? AND source=? ORDER BY seq LIMIT 200",
                (self._seq, self._gw.kind.value, EventSource.STREAM.value),
            )
            for r in rows:
                self._seq = max(self._seq, int(r["seq"]))
                self._buffer.append(OrderEvent.model_validate_json(r["payload_json"]))
            if not self._buffer:
                await asyncio.sleep(self._gw.poll_interval_s)


class LeanBridgeGateway:
    """``ExecutionGateway`` whose 'broker' is the local LEAN engine (polling the bridge endpoints)."""

    is_paper: bool = True
    events_preapplied: bool = True
    """``order_events()`` yields ledger rows written by ``ingest_engine_event`` (already applied, deduplicated)."""

    def __init__(
        self,
        ledger: OrderLedger,
        kind: GatewayKind,
        clock: Clock,
        *,
        state_reader: BrokerStateReader | None = None,
        wall_clock: Clock | None = None,
        heartbeat_max_age_s: float = 15.0,
        sim_starting_cash: Decimal = Decimal("100000"),
        poll_interval_s: float = 0.25,
    ) -> None:
        if kind in modes.LIVE_GATEWAYS:
            raise modes.LiveActivationDenied(modes.live_activation_failures(ledger.mode))
        if kind not in LEAN_GATEWAYS:
            raise modes.ModeViolation(f"LeanBridgeGateway does not serve {kind.value}")
        modes.assert_gateway_allowed(ledger.mode, kind)
        if kind == GatewayKind.LEAN_ALPACA_PAPER and state_reader is None:
            raise MissingPaperCredentials("LEAN_ALPACA_PAPER needs the Alpaca PAPER state reader (paper keys)")
        if state_reader is not None and not state_reader.is_paper:
            raise modes.ModeViolation("state reader is not a paper reader")
        self.ledger = ledger
        self.kind = kind
        self.clock = clock
        self.state_reader = state_reader
        self.wall_clock: Clock = wall_clock or SystemClock()
        self.heartbeat_max_age_s = heartbeat_max_age_s
        self.sim_starting_cash = sim_starting_cash
        self.poll_interval_s = poll_interval_s
        self.label: ExecutionLabel = modes.execution_label(ledger.mode, kind)

    @property
    def source_label(self) -> str:
        if self.state_reader is not None:
            return f"{self.kind.value} via Alpaca PAPER REST (broker-authoritative)"
        return f"{self.kind.value}: ledger-derived (LEAN PaperBrokerage, SIMULATED; not an external broker)"

    # ------------------------------------------------------------------ execution
    async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder:
        if intent.gateway != self.kind:
            raise GatewayRejected(f"intent gateway {intent.gateway.value} != {self.kind.value}")
        try:
            modes.assert_gateway_allowed(intent.mode, intent.gateway)
        except modes.ModeViolation as exc:
            raise GatewayRejected(str(exc)) from exc
        try:
            return self.ledger.enqueue_for_engine(intent)
        except LedgerError as exc:
            raise GatewayRejected(str(exc)) from exc

    async def cancel(self, client_order_id: str) -> None:
        if self.ledger.engine_queue_status(client_order_id) == "QUEUED" and self.ledger.withdraw_engine_intent(
            client_order_id, "cancel requested before engine pickup"
        ):
            return
        raise CancelNotSupported(
            "order already handed to the LEAN engine; bridge contract v1 has no engine cancel channel "
            "(cancel must be issued by the engine/broker; state follows broker events)"
        )

    def order_events(self) -> _LedgerTail:
        return _LedgerTail(self)

    def engine_health(self) -> tuple[bool, str]:
        ok, detail, _ = engine_heartbeat_status(
            self.ledger.db, self.wall_clock, self.heartbeat_max_age_s,
            expected_gateway=self.kind, expected_mode=self.ledger.mode.value,
        )
        return ok, detail

    # ------------------------------------------------------------------ state reads
    async def get_account(self) -> AccountSnapshot:
        if self.state_reader is not None:
            return await self.state_reader.get_account()
        rows = self.ledger.db.query("SELECT side, quantity, price FROM fills WHERE gateway=?", (self.kind.value,))
        cash = self.sim_starting_cash
        for r in rows:
            notional = Decimal(r["quantity"]) * Decimal(r["price"])
            cash += -notional if r["side"] == Side.BUY.value else notional
        positions = self.ledger.positions()
        equity = cash + sum((p.cost_basis for p in positions), Decimal(0))
        reserved = sum(
            ((o.quantity - o.filled_quantity) * (o.limit_price or Decimal(0)) for o in self.ledger.open_orders()
             if o.side == Side.BUY and o.gateway == self.kind),
            Decimal(0),
        )
        bp = max(cash - reserved, Decimal(0))
        return AccountSnapshot(
            snapshot_id=new_id("acct"), account_id_masked="LEAN-SIM", is_paper=True, status="ACTIVE", equity=equity,
            cash=cash, buying_power=bp, non_marginable_buying_power=bp, multiplier=Decimal(1), trading_blocked=False,
            account_blocked=False, transfers_blocked=True, asof=self.clock.now(), source="LEDGER_LEAN_PAPER_SIM",
        )

    async def get_positions(self) -> list[PositionSnapshot]:
        if self.state_reader is not None:
            return await self.state_reader.get_positions()
        return [p.model_copy(update={"source": "LEDGER_LEAN_PAPER_SIM"}) for p in self.ledger.positions()]

    def _engine_known(self, state: OrderState) -> bool:
        return state not in PRE_SUBMISSION and state not in IN_DOUBT

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        if self.state_reader is not None:
            return await self.state_reader.get_order_by_client_id(client_order_id)
        rec = self.ledger.get_record(client_order_id)
        if rec is None or rec.gateway != self.kind or not self._engine_known(rec.state):
            return None  # no engine evidence: unknown at the 'broker'
        return rec.to_broker_order()

    async def list_orders(self, *, status: str = "all", after: datetime | None = None) -> list[BrokerOrder]:
        if self.state_reader is not None:
            return await self.state_reader.list_orders(status=status, after=after)
        out: list[BrokerOrder] = []
        for rec in self.ledger.records():
            if rec.gateway != self.kind or not self._engine_known(rec.state) or rec.state == OrderState.RECONCILED:
                continue
            if status == "open" and rec.state in TERMINAL:
                continue
            if status == "closed" and rec.state not in TERMINAL:
                continue
            if after is not None and (rec.submitted_at or rec.created_at) <= after:
                continue
            out.append(rec.to_broker_order())
        return out

    async def aclose(self) -> None:
        return None
