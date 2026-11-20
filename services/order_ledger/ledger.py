"""Durable order ledger (spec §7, §11): one SQLite row per client order id, driven by ``state_machine``.

Guarantees
----------
* Every state change is a compare-and-set inside one ``BEGIN IMMEDIATE`` writer transaction and is written
  to ``order_state_transitions`` together with a hash-chained audit event (same transaction).
* Every received ``OrderEvent`` is stored append-only in ``order_events`` with ``applied`` + ``apply_note``.
  Re-deliveries (same dedupe key: broker execution id, else event id) are counted in
  ``order_event_duplicates`` and never re-applied, so replaying the whole event log is a no-op.
* Out-of-order events never regress state (rank-monotonic, terminal states sticky); fills are *always*
  booked, whatever the state (a fill racing a cancel is money truth). Cumulative filled quantity is
  monotonic and can never exceed the order quantity.
* ``ledger_positions`` change only from recorded ``fills``; average cost is ``cost_basis / quantity`` with an
  exact Decimal running cost basis, so the result does not depend on fill arrival order.
* Anything unexpected (orphan events, broker-id conflicts, over-fills, unsupported broker features) raises
  the ``ledger_anomaly`` entries flag: new entries stay locked until a CLEAN broker reconciliation.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from packages.contracts.enums import (
    EventSource,
    GatewayKind,
    OrderEventType,
    OrderState,
    OrderType,
    RiskVerdict,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.contracts.models import (
    ApprovedOrderIntent,
    BrokerOrder,
    HumanApproval,
    MarketSnapshot,
    OrderEvent,
    PositionSnapshot,
    RiskDecision,
    TradeProposal,
)
from packages.core import modes
from packages.core.audit import append_audit
from packages.core.canonical import canonical_json, sha256_hex
from packages.core.clock import Clock
from packages.core.db import Database
from packages.core.ids import new_id
from packages.core.signing import IntentVerifier
from services.risk.context import intent_fingerprint

from . import intervals
from .entries import Flag, set_flag_conn
from .errors import IntentRefused, LedgerConflict, LedgerError
from .state_machine import (
    BROKER_OUTCOMES,
    IN_DOUBT,
    OPEN,
    PRE_SUBMISSION,
    STATE_EVENT_TYPE,
    TERMINAL,
    WORKING,
    decide,
    is_allowed,
)
from .timeutil import iso, parse_iso, trading_day

S = OrderState
ZERO = Decimal(0)
PRICE_Q = Decimal("0.00000001")
MAX_CLOCK_SKEW = timedelta(seconds=5)
FILL_EVENT_TYPES = frozenset({OrderEventType.PARTIAL_FILL, OrderEventType.FILL})
COMPLETION_STATES = frozenset({S.PARTIALLY_FILLED, S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.EXPIRED})
UNSUPPORTED_EVENT_TYPES = frozenset({OrderEventType.REPLACED})


def _d(text: object) -> Decimal:
    return Decimal(str(text))


def _od(text: object | None) -> Decimal | None:
    return None if text is None else Decimal(str(text))


def _fmt(value: Decimal) -> str:
    return format(value, "f")


def _norm(value: Decimal) -> str:
    return "0" if value == 0 else format(value.normalize(), "f")


# ------------------------------------------------------------------------------------------ records


@dataclass(frozen=True)
class OrderRecord:
    """A full ``orders`` row (more than ``BrokerOrder`` exposes)."""

    client_order_id: str
    intent_id: str
    proposal_id: str
    decision_id: str
    causal_id: str
    mode: TradingMode
    gateway: GatewayKind
    symbol: str
    side: Side
    quantity: Decimal
    order_type: OrderType
    limit_price: Decimal | None
    time_in_force: TimeInForce
    state: OrderState
    broker_order_id: str | None
    broker_status: str | None
    filled_quantity: Decimal
    avg_fill_price: Decimal | None
    intent_fingerprint: str
    requires_human_approval: bool
    human_approval_id: str | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None
    claimed_by: str | None
    claimed_at: datetime | None
    unknown_since: datetime | None
    cancel_requested_at: datetime | None
    state_reason: str | None
    strategy_id: str | None
    intent_json: str

    @classmethod
    def from_row(cls, r: sqlite3.Row | dict[str, Any]) -> OrderRecord:
        g = dict(r)

        def ts(name: str) -> datetime | None:
            v = g.get(name)
            return parse_iso(str(v)) if v else None

        created = ts("created_at")
        updated = ts("updated_at")
        assert created is not None and updated is not None
        return cls(
            client_order_id=g["client_order_id"], intent_id=g["intent_id"], proposal_id=g["proposal_id"],
            decision_id=g["decision_id"], causal_id=g["causal_id"], mode=TradingMode(g["mode"]),
            gateway=GatewayKind(g["gateway"]), symbol=g["symbol"], side=Side(g["side"]), quantity=_d(g["quantity"]),
            order_type=OrderType(g["order_type"]), limit_price=_od(g["limit_price"]),
            time_in_force=TimeInForce(g["time_in_force"]), state=OrderState(g["state"]),
            broker_order_id=g["broker_order_id"], broker_status=g.get("broker_status"),
            filled_quantity=_d(g["filled_quantity"]), avg_fill_price=_od(g["avg_fill_price"]),
            intent_fingerprint=g["intent_fingerprint"], requires_human_approval=bool(g.get("requires_human_approval")),
            human_approval_id=g.get("human_approval_id"), expires_at=ts("expires_at"), created_at=created,
            updated_at=updated, submitted_at=ts("submitted_at"), claimed_by=g.get("claimed_by"),
            claimed_at=ts("claimed_at"), unknown_since=ts("unknown_since"),
            cancel_requested_at=ts("cancel_requested_at"), state_reason=g.get("state_reason"),
            strategy_id=g.get("strategy_id"), intent_json=g["intent_json"],
        )

    @property
    def intent(self) -> ApprovedOrderIntent:
        return ApprovedOrderIntent.model_validate_json(self.intent_json)

    def to_broker_order(self) -> BrokerOrder:
        return BrokerOrder(
            client_order_id=self.client_order_id, broker_order_id=self.broker_order_id, intent_id=self.intent_id,
            gateway=self.gateway, symbol=self.symbol, side=self.side, quantity=self.quantity,
            filled_quantity=self.filled_quantity, avg_fill_price=self.avg_fill_price, order_type=self.order_type,
            limit_price=self.limit_price, time_in_force=self.time_in_force,
            broker_status=self.broker_status or self.state.value, state=self.state,
            submitted_at=self.submitted_at, updated_at=self.updated_at,
        )


@dataclass(frozen=True)
class ApplyResult:
    applied: bool
    state: OrderState | None
    duplicate: bool
    note: str
    booked_quantity: Decimal = ZERO
    absorbed_quantity: Decimal = ZERO
    anomalies: tuple[str, ...] = field(default=())


# ------------------------------------------------------------------------------------------ conn helpers


def _order_row(conn: sqlite3.Connection, coid: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM orders WHERE client_order_id=?", (coid,)).fetchone()


def _require_row(conn: sqlite3.Connection, coid: str) -> sqlite3.Row:
    row = _order_row(conn, coid)
    if row is None:
        raise LedgerError(f"order {coid} vanished inside a transaction")
    return row


def _transition_conn(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    to_state: OrderState,
    *,
    reason: str,
    at: datetime,
    event_id: str | None = None,
    extra_sets: dict[str, Any] | None = None,
) -> bool:
    """Compare-and-set ``row.state -> to_state``. Returns False if the row changed concurrently.

    Every write is checked against ``ALLOWED_TRANSITIONS`` (the single authority); a disallowed transition is a
    programming error and aborts the whole transaction.
    """
    if not is_allowed(OrderState(row["state"]), to_state):
        raise LedgerError(f"transition {row['state']}->{to_state.value} is not in the allowed transition table")
    sets = {"state": to_state.value, "updated_at": iso(at), "state_reason": reason[:500]}
    if extra_sets:
        sets.update(extra_sets)
    assignments = ", ".join(f"{k}=?" for k in sets)
    cur = conn.execute(
        f"UPDATE orders SET {assignments}, version=version+1 WHERE client_order_id=? AND state=? AND version=?",  # noqa: S608
        (*sets.values(), row["client_order_id"], row["state"], row["version"]),
    )
    if cur.rowcount != 1:
        return False
    conn.execute(
        "INSERT INTO order_state_transitions(client_order_id, from_state, to_state, at, reason, event_id)"
        " VALUES (?,?,?,?,?,?)",
        (row["client_order_id"], row["state"], to_state.value, iso(at), reason[:1000], event_id),
    )
    append_audit(
        conn, event_type="ORDER_STATE", source="order_ledger", mode=row["mode"], causal_id=row["causal_id"],
        payload={"client_order_id": row["client_order_id"], "from": row["state"], "to": to_state.value,
                 "reason": reason[:500], "event_id": event_id},
        ts=at,
    )
    return True


def _flag_anomaly(conn: sqlite3.Connection, detail: str, *, mode: str, at: datetime) -> None:
    set_flag_conn(conn, Flag.LEDGER_ANOMALY, reason=detail, source="order_ledger", mode=mode, at=at)


def _fill_rows(conn: sqlite3.Connection, coid: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM fills WHERE client_order_id=? ORDER BY ts, fill_id", (coid,)).fetchall()


def _booked_totals(rows: Sequence[sqlite3.Row]) -> tuple[Decimal, Decimal]:
    qty = sum((_d(r["quantity"]) for r in rows), ZERO)
    notional = sum((_d(r["quantity"]) * _d(r["price"]) for r in rows), ZERO)
    return qty, notional


def _coverage(rows: Sequence[sqlite3.Row]) -> tuple[list[intervals.Interval], list[intervals.Interval], bool]:
    """(all covered ranges, synthetic ranges, every row placed?)."""
    covered: list[intervals.Interval] = []
    synthetic: list[intervals.Interval] = []
    placed = True
    for r in rows:
        if r["cum_from"] is None or r["cum_to"] is None:
            placed = False
            continue
        rng = (_d(r["cum_from"]), _d(r["cum_to"]))
        covered.append(rng)
        if r["synthetic"]:
            synthetic.append(rng)
    return covered, synthetic, placed


def _update_position_conn(
    conn: sqlite3.Connection, symbol: str, side: Side, qty: Decimal, price: Decimal, at: datetime
) -> tuple[Decimal, list[str]]:
    """Apply one booked fill to ``ledger_positions``. Returns (realized P&L of this fill, anomalies)."""
    anomalies: list[str] = []
    row = conn.execute("SELECT * FROM ledger_positions WHERE symbol=?", (symbol,)).fetchone()
    pos_qty = _d(row["quantity"]) if row else ZERO
    cost = _d(row["cost_basis"]) if row else ZERO
    realized_total = _d(row["realized_pl"]) if row else ZERO
    realized = ZERO
    if side == Side.BUY:
        if pos_qty < 0:
            anomalies.append(f"BUY fill on negative ledger position {symbol} ({_fmt(pos_qty)})")
        pos_qty += qty
        cost += qty * price
    else:
        closing = min(qty, max(pos_qty, ZERO))
        if closing > 0:
            avg = cost / pos_qty
            realized = (closing * (price - avg)).quantize(PRICE_Q)
            cost -= closing * avg
            pos_qty -= closing
        excess = qty - closing
        if excess > 0:
            anomalies.append(f"SELL fill exceeds ledger long position in {symbol} by {_fmt(excess)} (long-only violated)")
            pos_qty -= excess
            cost -= excess * price
    if pos_qty == 0:
        cost = ZERO
    avg_cost = (cost / pos_qty).quantize(PRICE_Q) if pos_qty != 0 else ZERO
    realized_total += realized
    conn.execute(
        "INSERT INTO ledger_positions(symbol, quantity, avg_cost, realized_pl, updated_at, cost_basis) VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, avg_cost=excluded.avg_cost,"
        " realized_pl=excluded.realized_pl, updated_at=excluded.updated_at, cost_basis=excluded.cost_basis",
        (symbol, _fmt(pos_qty), _fmt(avg_cost), _fmt(realized_total), iso(at), _fmt(cost)),
    )
    return realized, anomalies


def _insert_fill_conn(
    conn: sqlite3.Connection,
    order: sqlite3.Row,
    *,
    exec_id: str,
    qty: Decimal,
    price: Decimal,
    ts: datetime,
    cum_from: Decimal | None,
    cum_to: Decimal | None,
    synthetic: bool,
    event_id: str,
    at: datetime,
) -> list[str]:
    side = Side(order["side"])
    realized, anomalies = _update_position_conn(conn, order["symbol"], side, qty, price, at)
    fill_id = new_id("fill")
    conn.execute(
        "INSERT INTO fills(fill_id, client_order_id, broker_exec_id, symbol, side, quantity, price, ts, gateway,"
        " cum_from, cum_to, synthetic, realized_pl, trading_day, event_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fill_id, order["client_order_id"], exec_id, order["symbol"], side.value, _fmt(qty), _fmt(price), iso(ts),
            order["gateway"], None if cum_from is None else _fmt(cum_from), None if cum_to is None else _fmt(cum_to),
            int(synthetic), _fmt(realized), trading_day(ts).isoformat(), event_id,
        ),
    )
    append_audit(
        conn, event_type="FILL_BOOKED", source="order_ledger", mode=order["mode"], causal_id=order["causal_id"],
        payload={"fill_id": fill_id, "client_order_id": order["client_order_id"], "broker_exec_id": exec_id,
                 "symbol": order["symbol"], "side": side.value, "quantity": _fmt(qty), "price": _fmt(price),
                 "synthetic": synthetic, "realized_pl": _fmt(realized)},
        ts=at,
    )
    return anomalies


def _refresh_order_fill_totals(conn: sqlite3.Connection, coid: str, at: datetime) -> Decimal:
    qty, notional = _booked_totals(_fill_rows(conn, coid))
    avg = (notional / qty).quantize(PRICE_Q) if qty > 0 else None
    conn.execute(
        "UPDATE orders SET filled_quantity=?, avg_fill_price=?, updated_at=?, version=version+1 WHERE client_order_id=?",
        (_fmt(qty), None if avg is None else _fmt(avg), iso(at), coid),
    )
    return qty


def effective_exec_id(ev: OrderEvent) -> str | None:
    """Execution identity used to dedupe fills: broker exec id, else the order's cumulative quantity."""
    if ev.broker_exec_id:
        return ev.broker_exec_id
    if ev.fill_quantity is not None and ev.cum_filled_quantity is not None:
        return f"cum:{ev.client_order_id}:{_norm(ev.cum_filled_quantity)}"
    return None


def dedupe_key(ev: OrderEvent) -> str:
    exec_id = effective_exec_id(ev) if ev.fill_quantity is not None else None
    return f"exec:{exec_id}" if exec_id else f"event:{ev.event_id}"


def _insert_event_conn(conn: sqlite3.Connection, ev: OrderEvent, key: str, *, applied: bool, note: str) -> None:
    conn.execute(
        "INSERT INTO order_events(event_id, dedupe_key, client_order_id, broker_order_id, event_type, state, source,"
        " gateway, broker_ts, received_ts, applied, apply_note, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ev.event_id, key, ev.client_order_id, ev.broker_order_id, ev.event_type.value, ev.state.value,
            ev.source.value, ev.gateway.value, iso(ev.broker_ts) if ev.broker_ts else None, iso(ev.received_ts),
            int(applied), note[:2000], canonical_json(ev),
        ),
    )


def _record_duplicate_conn(conn: sqlite3.Connection, key: str, ev: OrderEvent, first_event_id: str, at: datetime) -> None:
    conn.execute(
        "INSERT INTO order_event_duplicates(dedupe_key, client_order_id, first_event_id, deliveries, first_seen, last_seen)"
        " VALUES (?,?,?,?,?,?) ON CONFLICT(dedupe_key) DO UPDATE SET deliveries=deliveries+1, last_seen=excluded.last_seen",
        (key, ev.client_order_id, first_event_id, 1, iso(at), iso(at)),
    )


# ------------------------------------------------------------------------------------------ the ledger


class OrderLedger:
    """Durable order state machine. All writes go through ``Database.write`` (single writer)."""

    def __init__(
        self,
        db: Database,
        clock: Clock,
        mode: TradingMode,
        *,
        verifier: IntentVerifier | None = None,
        approval_verifier: Callable[[HumanApproval], bool] | None = None,
    ) -> None:
        self.db = db
        self.clock = clock
        self.mode = mode
        self.verifier = verifier
        self.approval_verifier = approval_verifier

    # ================================================================== decision artefacts
    def _record_artifact(
        self, table: str, key_col: str, key: str, columns: dict[str, Any], payload: str, audit_type: str,
        causal_id: str | None, policy_version: str | None = None, dataset_version: str | None = None,
    ) -> bool:
        """Insert-once. Same content again is a no-op (returns False); different content raises LedgerConflict."""
        at = self.clock.now()

        def _w(conn: sqlite3.Connection) -> bool:
            existing = conn.execute(f"SELECT payload_json FROM {table} WHERE {key_col}=?", (key,)).fetchone()  # noqa: S608
            if existing is not None:
                if existing[0] != payload:
                    raise LedgerConflict(f"{table}.{key_col}={key} already recorded with different content")
                return False
            cols = [key_col, *columns, "payload_json"]
            conn.execute(
                f"INSERT INTO {table}({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",  # noqa: S608
                (key, *columns.values(), payload),
            )
            append_audit(
                conn, event_type=audit_type, source="order_ledger", mode=self.mode.value, causal_id=causal_id,
                payload={key_col: key}, policy_version=policy_version, dataset_version=dataset_version, ts=at,
            )
            return True

        try:
            return self.db.write(_w)
        except sqlite3.IntegrityError as exc:
            raise LedgerError(f"cannot record {table} {key}: {exc}") from exc

    def record_proposal(self, proposal: TradeProposal) -> bool:
        return self._record_artifact(
            "proposals", "proposal_id", proposal.proposal_id,
            {"causal_id": proposal.causal_id, "created_at": iso(proposal.created_at), "action": proposal.action.value,
             "proposer": proposal.proposer.value, "symbol": proposal.symbol},
            canonical_json(proposal), "PROPOSAL_RECORDED", proposal.causal_id,
            policy_version=proposal.policy_version_seen, dataset_version=proposal.dataset_version,
        )

    def record_decision(self, decision: RiskDecision) -> bool:
        return self._record_artifact(
            "risk_decisions", "decision_id", decision.decision_id,
            {"proposal_id": decision.proposal_id, "causal_id": decision.causal_id, "verdict": decision.verdict.value,
             "policy_version": decision.policy_version, "evaluated_at": iso(decision.evaluated_at)},
            canonical_json(decision), "RISK_DECISION_RECORDED", decision.causal_id,
            policy_version=decision.policy_version,
        )

    def record_market_snapshot(self, snapshot: MarketSnapshot) -> bool:
        return self._record_artifact(
            "market_snapshots", "snapshot_id", snapshot.snapshot_id,
            {"symbol": snapshot.symbol, "data_asof": iso(snapshot.data_asof), "feed": snapshot.feed.value,
             "is_simulated": int(snapshot.is_simulated)},
            canonical_json(snapshot), "MARKET_SNAPSHOT_RECORDED", None, dataset_version=snapshot.dataset_version,
        )

    # ================================================================== intents
    def record_intent(self, intent: ApprovedOrderIntent) -> OrderRecord:
        """Create the order row (RISK_APPROVED; awaits HUMAN_APPROVED when ``requires_human_approval``).

        Refuses (``IntentRefused`` / ``ModeViolation``) intents for another mode, a gateway not allowed in this
        mode (live always denied), a wrong client-order-id tag, an already expired intent, a bad signature
        (when a verifier is configured) or without a recorded APPROVED risk decision for the same proposal.
        """
        if intent.mode != self.mode:
            raise IntentRefused(f"intent mode {intent.mode.value} != ledger mode {self.mode.value}")
        modes.assert_gateway_allowed(intent.mode, intent.gateway)
        tag = modes.client_order_tag(intent.mode, intent.gateway)
        if not intent.client_order_id.startswith(f"lat-{tag}-"):
            raise IntentRefused(f"client order id is not tagged 'lat-{tag}-' for {intent.mode.value}/{intent.gateway.value}")
        if self.verifier is not None and not self.verifier.verify(intent.signing_string(), intent.signature):
            raise IntentRefused("intent signature does not verify under the risk engine public key")
        now = self.clock.now()
        if intent.expires_at <= now:
            raise IntentRefused("intent already expired")
        intent_json = intent.model_dump_json()
        fingerprint = intent_fingerprint(intent)

        def _w(conn: sqlite3.Connection) -> None:
            existing = conn.execute(
                "SELECT intent_json FROM orders WHERE client_order_id=? OR intent_id=?",
                (intent.client_order_id, intent.intent_id),
            ).fetchall()
            if existing:
                if len(existing) == 1 and existing[0][0] == intent_json:
                    return
                raise LedgerConflict(f"order {intent.client_order_id}/{intent.intent_id} already recorded with different content")
            dec = conn.execute(
                "SELECT verdict, proposal_id, payload_json FROM risk_decisions WHERE decision_id=?", (intent.decision_id,)
            ).fetchone()
            if dec is None:
                raise IntentRefused(f"no recorded risk decision {intent.decision_id}; record_decision first")
            decision = RiskDecision.model_validate_json(dec["payload_json"])
            if decision.verdict != RiskVerdict.APPROVED:
                raise IntentRefused(f"risk decision {intent.decision_id} is {decision.verdict.value}, not APPROVED")
            if decision.proposal_id != intent.proposal_id or decision.causal_id != intent.causal_id:
                raise IntentRefused("intent does not belong to the recorded risk decision")
            if decision.mode != intent.mode or decision.policy_version != intent.policy_version:
                raise IntentRefused("intent mode/policy differs from the recorded risk decision")
            prop = conn.execute("SELECT payload_json FROM proposals WHERE proposal_id=?", (intent.proposal_id,)).fetchone()
            strategy_id = json.loads(prop[0]).get("strategy_id") if prop else None
            conn.execute(
                "INSERT INTO orders(client_order_id, intent_id, proposal_id, decision_id, causal_id, mode, gateway, symbol,"
                " side, quantity, order_type, limit_price, time_in_force, state, filled_quantity, intent_json,"
                " intent_fingerprint, created_at, updated_at, strategy_id, requires_human_approval, expires_at,"
                " broker_status, state_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    intent.client_order_id, intent.intent_id, intent.proposal_id, intent.decision_id, intent.causal_id,
                    intent.mode.value, intent.gateway.value, intent.symbol, intent.side.value, _fmt(intent.quantity),
                    intent.order_type.value, _fmt(intent.limit_price), intent.time_in_force.value,
                    S.RISK_APPROVED.value, "0", intent_json, fingerprint, iso(now), iso(now), strategy_id,
                    int(intent.requires_human_approval), iso(intent.expires_at), None,
                    "awaiting human approval" if intent.requires_human_approval else "risk approved",
                ),
            )
            for frm, to, why in (
                (None, S.PROPOSED, f"proposal {intent.proposal_id} recorded"),
                (S.PROPOSED, S.RISK_APPROVED, f"risk decision {intent.decision_id} APPROVED; intent {intent.intent_id} signed"),
            ):
                conn.execute(
                    "INSERT INTO order_state_transitions(client_order_id, from_state, to_state, at, reason, event_id)"
                    " VALUES (?,?,?,?,?,?)",
                    (intent.client_order_id, None if frm is None else frm.value, to.value, iso(now), why, None),
                )
            append_audit(
                conn, event_type="ORDER_INTENT_RECORDED", source="order_ledger", mode=self.mode.value,
                causal_id=intent.causal_id, policy_version=intent.policy_version, ts=now,
                payload={"client_order_id": intent.client_order_id, "intent_id": intent.intent_id,
                         "symbol": intent.symbol, "side": intent.side.value, "quantity": _fmt(intent.quantity),
                         "limit_price": _fmt(intent.limit_price), "gateway": intent.gateway.value,
                         "requires_human_approval": intent.requires_human_approval, "fingerprint": fingerprint},
            )

        self.db.write(_w)
        rec = self.get_record(intent.client_order_id)
        assert rec is not None
        return rec

    def record_human_approval(self, approval: HumanApproval) -> OrderRecord:
        """RISK_APPROVED -> HUMAN_APPROVED for an intent that requires a human confirmation.

        The approval must be bound to this intent, confirmed *after* the intent was created (fresh, per order),
        not expired, carry a non-empty approver and statement, and pass ``approval_verifier`` when the ledger
        was given one (the API verifies the operator token before calling this).
        """
        now = self.clock.now()
        if not approval.approver.strip() or not approval.statement.strip():
            raise IntentRefused("human approval needs a named approver and the confirmed statement")
        if approval.expires_at <= approval.confirmed_at:
            raise IntentRefused("human approval expires before it was confirmed")
        if self.approval_verifier is not None and not self.approval_verifier(approval):
            raise IntentRefused("human approval signature does not verify")

        def _w(conn: sqlite3.Connection) -> str:
            row = conn.execute("SELECT * FROM orders WHERE intent_id=?", (approval.intent_id,)).fetchone()
            if row is None:
                raise LedgerError(f"no order for intent {approval.intent_id}")
            if not row["requires_human_approval"]:
                raise IntentRefused("intent does not require human approval")
            if row["state"] == S.HUMAN_APPROVED.value and row["human_approval_id"] == approval.approval_id:
                return row["client_order_id"]
            if row["state"] != S.RISK_APPROVED.value:
                raise IntentRefused(f"order is {row['state']}; human approval only applies to RISK_APPROVED orders")
            if approval.confirmed_at > now + MAX_CLOCK_SKEW:
                raise IntentRefused("human approval confirmed in the future")
            if approval.confirmed_at + MAX_CLOCK_SKEW < parse_iso(str(row["created_at"])):
                raise IntentRefused("human approval predates the intent (a fresh confirmation per order is required)")
            if approval.expires_at <= now:
                raise IntentRefused("human approval already expired")
            conn.execute(
                "INSERT INTO human_approvals(approval_id, intent_id, approver, confirmed_at, payload_json) VALUES (?,?,?,?,?)",
                (approval.approval_id, approval.intent_id, approval.approver, iso(approval.confirmed_at),
                 canonical_json(approval)),
            )
            if not _transition_conn(
                conn, row, S.HUMAN_APPROVED, reason=f"human approval {approval.approval_id} by {approval.approver}",
                at=now, extra_sets={"human_approval_id": approval.approval_id},
            ):
                raise LedgerError("order changed concurrently")
            return row["client_order_id"]

        coid = self.db.write(_w)
        rec = self.get_record(coid)
        assert rec is not None
        return rec

    def claim_for_submission(self, client_order_id: str, claimer: str) -> ApprovedOrderIntent | None:
        """Atomic RISK_APPROVED/HUMAN_APPROVED -> SUBMITTING. Returns the intent exactly once, else None.

        Expired intents move to EXPIRED; an intent requiring human approval without a valid recorded
        approval is refused (stays put). Nothing that was claimed once can ever be claimed again.
        """
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> ApprovedOrderIntent | None:
            row = _order_row(conn, client_order_id)
            if row is None or row["state"] not in (S.RISK_APPROVED.value, S.HUMAN_APPROVED.value):
                return None
            intent = ApprovedOrderIntent.model_validate_json(row["intent_json"])
            if intent.mode != self.mode:
                _transition_conn(conn, row, S.REJECTED, reason=f"intent mode {intent.mode.value} != {self.mode.value}", at=now)
                return None
            try:
                modes.assert_gateway_allowed(intent.mode, intent.gateway)
            except modes.ModeViolation as exc:
                _transition_conn(conn, row, S.REJECTED, reason=f"gateway not allowed: {exc}", at=now)
                return None
            if now >= intent.expires_at:
                _transition_conn(conn, row, S.EXPIRED, reason="intent expired before submission (never sent)", at=now)
                return None
            if row["requires_human_approval"]:
                if row["state"] != S.HUMAN_APPROVED.value or not row["human_approval_id"]:
                    return None
                appr = conn.execute(
                    "SELECT payload_json FROM human_approvals WHERE approval_id=? AND intent_id=?",
                    (row["human_approval_id"], intent.intent_id),
                ).fetchone()
                if appr is None or HumanApproval.model_validate_json(appr[0]).expires_at <= now:
                    return None
            ok = _transition_conn(
                conn, row, S.SUBMITTING, reason=f"claimed for submission by {claimer}", at=now,
                extra_sets={"claimed_by": claimer[:200], "claimed_at": iso(now)},
            )
            return intent if ok else None

        return self.db.write(_w)

    def reject_locally(self, client_order_id: str, reasons: Sequence[str]) -> bool:
        """Pre-submission -> REJECTED (final gate / entries lock refused it; nothing was sent)."""
        return self._local_terminal(client_order_id, S.REJECTED, "rejected before submission: " + "; ".join(reasons))

    def cancel_local(self, client_order_id: str, reason: str = "canceled before submission (never sent)") -> bool:
        return self._local_terminal(client_order_id, S.CANCELED, reason)

    def _local_terminal(self, coid: str, to_state: OrderState, reason: str) -> bool:
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> bool:
            row = _order_row(conn, coid)
            if row is None or OrderState(row["state"]) not in PRE_SUBMISSION:
                return False
            return _transition_conn(conn, row, to_state, reason=reason, at=now)

        return self.db.write(_w)

    def expire_stale_intents(self) -> list[str]:
        """Pre-submission intents past ``expires_at`` -> EXPIRED; unclaimed engine-queue entries likewise."""
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> list[str]:
            out: list[str] = []
            rows = conn.execute(
                "SELECT * FROM orders WHERE state IN (?,?,?) AND expires_at IS NOT NULL AND expires_at <= ?",
                (S.PROPOSED.value, S.RISK_APPROVED.value, S.HUMAN_APPROVED.value, iso(now)),
            ).fetchall()
            for row in rows:
                if _transition_conn(conn, row, S.EXPIRED, reason="intent expired before submission (never sent)", at=now):
                    out.append(row["client_order_id"])
            out.extend(_expire_engine_queue_conn(conn, now))
            return out

        return self.db.write(_w)

    # ================================================================== submission outcomes
    def mark_submitted(self, client_order_id: str, broker_order: BrokerOrder) -> ApplyResult:
        """Record the synchronous gateway response (state from the broker; fills come only from executions)."""
        if broker_order.client_order_id != client_order_id:
            raise LedgerError("broker order client id does not match")
        now = self.clock.now()
        ev = OrderEvent(
            event_id=f"gw-submit-{client_order_id}", client_order_id=client_order_id,
            broker_order_id=broker_order.broker_order_id, event_type=STATE_EVENT_TYPE.get(broker_order.state, OrderEventType.NEW),
            state=broker_order.state, source=EventSource.GATEWAY, gateway=broker_order.gateway,
            cum_filled_quantity=broker_order.filled_quantity, broker_ts=broker_order.submitted_at or broker_order.updated_at,
            received_ts=now, reason="synchronous submit response", raw={"broker_status": broker_order.broker_status},
        )
        # submitted_at is set by the first transition out of SUBMITTING into a broker state (see apply_event);
        # a LEAN-bridge response (still SUBMITTING: queued for the engine) therefore sets nothing.
        return self.apply_event(ev)

    def mark_unknown(self, client_order_id: str, reason: str) -> OrderState | None:
        """SUBMITTING -> UNKNOWN (outcome unknown) and raise ``unknown_orders_present``. Never resubmits."""
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> OrderState | None:
            row = _order_row(conn, client_order_id)
            if row is None:
                return None
            state = OrderState(row["state"])
            if state == S.SUBMITTING:
                ev = OrderEvent(
                    event_id=new_id("ev"), client_order_id=client_order_id, event_type=OrderEventType.SUBMIT_TIMEOUT,
                    state=S.UNKNOWN, source=EventSource.LOCAL, gateway=GatewayKind(row["gateway"]), received_ts=now,
                    reason=reason[:500],
                )
                ok = _transition_conn(conn, row, S.UNKNOWN, reason=f"submission outcome unknown: {reason}", at=now,
                                      event_id=ev.event_id, extra_sets={"unknown_since": iso(now)})
                _insert_event_conn(conn, ev, f"event:{ev.event_id}", applied=ok, note="SUBMITTING->UNKNOWN" if ok else "race")
                state = S.UNKNOWN if ok else OrderState(_require_row(conn, client_order_id)["state"])
            if state == S.UNKNOWN:
                set_flag_conn(conn, Flag.UNKNOWN_ORDERS_PRESENT, reason=f"order {client_order_id} outcome unknown: {reason}",
                              source="execution", mode=self.mode.value, at=now)
            return state

        return self.db.write(_w)

    def mark_rejected(self, client_order_id: str, reason: str, *, source: EventSource = EventSource.GATEWAY) -> ApplyResult:
        """The gateway/broker refused the order synchronously (SUBMITTING -> REJECTED). Nothing to retry."""
        rec = self.get_record(client_order_id)
        gateway = rec.gateway if rec else GatewayKind.FAKE
        ev = OrderEvent(
            event_id=new_id("ev"), client_order_id=client_order_id, event_type=OrderEventType.REJECTED,
            state=S.REJECTED, source=source, gateway=gateway, received_ts=self.clock.now(), reason=reason[:500],
        )
        return self.apply_event(ev)

    def request_cancel(self, client_order_id: str) -> OrderState | None:
        """Working order -> CANCEL_PENDING (LOCAL). In-doubt orders only record the request."""
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> OrderState | None:
            row = _order_row(conn, client_order_id)
            if row is None:
                return None
            state = OrderState(row["state"])
            conn.execute("UPDATE orders SET cancel_requested_at=COALESCE(cancel_requested_at, ?) WHERE client_order_id=?",
                         (iso(now), client_order_id))
            row = _require_row(conn, client_order_id)
            if state in (S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED):
                ev = OrderEvent(
                    event_id=new_id("ev"), client_order_id=client_order_id, broker_order_id=row["broker_order_id"],
                    event_type=OrderEventType.PENDING_CANCEL, state=S.CANCEL_PENDING, source=EventSource.LOCAL,
                    gateway=GatewayKind(row["gateway"]), received_ts=now, reason="cancel requested",
                )
                ok = _transition_conn(conn, row, S.CANCEL_PENDING, reason="cancel requested; broker events decide",
                                      at=now, event_id=ev.event_id)
                _insert_event_conn(conn, ev, f"event:{ev.event_id}", applied=ok, note="cancel requested")
                return S.CANCEL_PENDING if ok else state
            append_audit(conn, event_type="ORDER_CANCEL_REQUESTED", source="order_ledger", mode=self.mode.value,
                         causal_id=row["causal_id"], payload={"client_order_id": client_order_id, "state": state.value},
                         ts=now)
            return state

        return self.db.write(_w)

    def revert_cancel_request(self, client_order_id: str, reason: str) -> OrderState | None:
        """CANCEL_PENDING -> the working state it came from, when the cancel was never transmitted (LOCAL).

        Only valid while the last transition is the LOCAL cancel request itself (no broker event since).
        """
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> OrderState | None:
            row = _order_row(conn, client_order_id)
            if row is None:
                return None
            state = OrderState(row["state"])
            if state != S.CANCEL_PENDING:
                return state
            last = conn.execute(
                "SELECT from_state, reason FROM order_state_transitions WHERE client_order_id=? ORDER BY seq DESC LIMIT 1",
                (client_order_id,),
            ).fetchone()
            if last is None or last["from_state"] is None or not str(last["reason"]).startswith("cancel requested"):
                return state
            back = S.PARTIALLY_FILLED if _d(row["filled_quantity"]) > 0 else OrderState(last["from_state"])
            if back not in (S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED):
                return state
            _transition_conn(conn, row, back, reason=f"cancel request reverted: {reason}", at=now)
            return OrderState(_require_row(conn, client_order_id)["state"])

        return self.db.write(_w)

    def mark_reconciled_absent(self, client_order_id: str, note: str = "not present at broker") -> ApplyResult:
        """UNKNOWN -> RECONCILED: the broker has no such order after the grace period. Never invents fills."""
        rec = self.get_record(client_order_id)
        gateway = rec.gateway if rec else GatewayKind.FAKE
        ev = OrderEvent(
            event_id=new_id("ev"), client_order_id=client_order_id, event_type=OrderEventType.RECONCILED,
            state=S.RECONCILED, source=EventSource.RECONCILE, gateway=gateway, received_ts=self.clock.now(), reason=note,
        )
        return self.apply_event(ev)

    def recover_in_doubt_at_startup(self) -> dict[str, str]:
        """Process start: every SUBMITTING order may have been accepted by the broker before a crash.

        Unclaimed engine-queue entries are withdrawn (never handed to the engine, so never sent: EXPIRED);
        everything else becomes UNKNOWN and is resolved by client-order-id lookup during reconciliation.
        """
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> dict[str, str]:
            out: dict[str, str] = {}
            for coid in _expire_engine_queue_conn(conn, now):
                out[coid] = "EXPIRED"
            for row in conn.execute("SELECT * FROM orders WHERE state=?", (S.SUBMITTING.value,)).fetchall():
                coid = row["client_order_id"]
                q = conn.execute("SELECT status FROM engine_intent_queue WHERE client_order_id=?", (coid,)).fetchone()
                if q is not None and q[0] == "QUEUED":
                    conn.execute("UPDATE engine_intent_queue SET status='WITHDRAWN', note=?"
                                 " WHERE client_order_id=? AND status='QUEUED'",
                                 ("withdrawn at restart before engine pickup", coid))
                    if _transition_conn(conn, row, S.EXPIRED, at=now,
                                        reason="withdrawn at restart before engine pickup (never sent)"):
                        out[coid] = "EXPIRED"
                    continue
                since = row["claimed_at"] or iso(now)
                ev = OrderEvent(
                    event_id=new_id("ev"), client_order_id=coid, event_type=OrderEventType.UNKNOWN, state=S.UNKNOWN,
                    source=EventSource.LOCAL, gateway=GatewayKind(row["gateway"]), received_ts=now,
                    reason="process restarted while SUBMITTING (broker may have accepted)",
                )
                if _transition_conn(conn, row, S.UNKNOWN, reason="restart while SUBMITTING: outcome unknown", at=now,
                                    event_id=ev.event_id, extra_sets={"unknown_since": since}):
                    _insert_event_conn(conn, ev, f"event:{ev.event_id}", applied=True, note="SUBMITTING->UNKNOWN at startup")
                    out[coid] = "UNKNOWN"
            stale = conn.execute(
                "SELECT * FROM orders WHERE state IN (?,?,?) AND expires_at IS NOT NULL AND expires_at <= ?",
                (S.PROPOSED.value, S.RISK_APPROVED.value, S.HUMAN_APPROVED.value, iso(now)),
            ).fetchall()
            for row in stale:
                if _transition_conn(conn, row, S.EXPIRED, reason="intent expired before submission (never sent)", at=now):
                    out[row["client_order_id"]] = "EXPIRED"
            if conn.execute("SELECT 1 FROM orders WHERE state=? LIMIT 1", (S.UNKNOWN.value,)).fetchone():
                set_flag_conn(conn, Flag.UNKNOWN_ORDERS_PRESENT, reason="orders with unknown outcome at startup",
                              source="startup", mode=self.mode.value, at=now)
            return out

        return self.db.write(_w)

    # ================================================================== events
    def apply_event(self, ev: OrderEvent) -> ApplyResult:
        """Idempotent, order-tolerant application of one broker/engine/local event."""
        now = self.clock.now()
        return self.db.write(lambda conn: self._apply_event_conn(conn, ev, now, snapshot=None))

    def apply_broker_order(self, order: BrokerOrder, source: EventSource) -> ApplyResult:
        """Apply a broker REST snapshot of an order (poll / reconciliation lookup by client order id).

        With ``source=RECONCILE`` a broker cumulative filled quantity above the ledger's booked quantity is
        booked as a broker-authoritative catch-up fill (only the uncovered quantity ranges; late execution
        reports for the same range are later absorbed, never double-booked).
        """
        now = self.clock.now()
        material = {
            "broker_order_id": order.broker_order_id, "state": order.state.value, "status": order.broker_status,
            "filled": _norm(order.filled_quantity), "avg": None if order.avg_fill_price is None else _norm(order.avg_fill_price),
        }
        ev = OrderEvent(
            event_id=f"{source.value.lower()}-{order.client_order_id}-{sha256_hex(canonical_json(material))[:20]}",
            client_order_id=order.client_order_id, broker_order_id=order.broker_order_id,
            event_type=STATE_EVENT_TYPE.get(order.state, OrderEventType.UNKNOWN), state=order.state, source=source,
            gateway=order.gateway, cum_filled_quantity=order.filled_quantity, broker_ts=order.updated_at, received_ts=now,
            reason=f"broker snapshot ({order.broker_status})", raw={"broker_status": order.broker_status},
        )
        return self.db.write(lambda conn: self._apply_event_conn(conn, ev, now, snapshot=order))

    def _apply_event_conn(
        self, conn: sqlite3.Connection, ev: OrderEvent, now: datetime, *, snapshot: BrokerOrder | None
    ) -> ApplyResult:
        key = dedupe_key(ev)
        dup = conn.execute(
            "SELECT event_id FROM order_events WHERE dedupe_key=? OR event_id=? LIMIT 1", (key, ev.event_id)
        ).fetchone()
        row = _order_row(conn, ev.client_order_id)
        if dup is not None:
            _record_duplicate_conn(conn, key, ev, dup[0], now)
            return ApplyResult(False, OrderState(row["state"]) if row else None, True, f"duplicate of {dup[0]} ({key})")
        mode = row["mode"] if row else self.mode.value
        if row is None:
            note = "order unknown to the ledger (never created here); recorded, not applied"
            _insert_event_conn(conn, ev, key, applied=False, note=note)
            _flag_anomaly(conn, f"broker event for unknown client order id {ev.client_order_id}", mode=mode, at=now)
            append_audit(conn, event_type="ORDER_EVENT_ORPHAN", source="order_ledger", mode=mode, ts=now,
                         payload={"event_id": ev.event_id, "client_order_id": ev.client_order_id})
            return ApplyResult(False, None, False, note, anomalies=(note,))

        current = OrderState(row["state"])
        notes: list[str] = []
        anomalies: list[str] = []
        if ev.gateway.value != row["gateway"]:
            note = f"event gateway {ev.gateway.value} != order gateway {row['gateway']}; not applied"
            _insert_event_conn(conn, ev, key, applied=False, note=note)
            _flag_anomaly(conn, f"{ev.client_order_id}: {note}", mode=mode, at=now)
            return ApplyResult(False, current, False, note, anomalies=(note,))
        if snapshot is not None:
            mismatch = _snapshot_mismatch(row, snapshot)
            if mismatch:
                note = f"broker order differs from ledger ({mismatch}); not applied"
                _insert_event_conn(conn, ev, key, applied=False, note=note)
                _flag_anomaly(conn, f"{ev.client_order_id}: {note}", mode=mode, at=now)
                return ApplyResult(False, current, False, note, anomalies=(note,))
        # broker order id
        if ev.broker_order_id:
            if row["broker_order_id"] and row["broker_order_id"] != ev.broker_order_id:
                note = f"broker order id {ev.broker_order_id} conflicts with recorded {row['broker_order_id']}; not applied"
                _insert_event_conn(conn, ev, key, applied=False, note=note)
                _flag_anomaly(conn, f"{ev.client_order_id}: {note}", mode=mode, at=now)
                return ApplyResult(False, current, False, note, anomalies=(note,))
            if not row["broker_order_id"]:
                try:
                    conn.execute("UPDATE orders SET broker_order_id=? WHERE client_order_id=?",
                                 (ev.broker_order_id, ev.client_order_id))
                except sqlite3.IntegrityError:
                    note = f"broker order id {ev.broker_order_id} already belongs to another order; not applied"
                    _insert_event_conn(conn, ev, key, applied=False, note=note)
                    _flag_anomaly(conn, f"{ev.client_order_id}: {note}", mode=mode, at=now)
                    return ApplyResult(False, current, False, note, anomalies=(note,))
        if ev.event_type in UNSUPPORTED_EVENT_TYPES:
            note = f"unsupported broker feature '{ev.event_type.value}' (never requested); not applied"
            _insert_event_conn(conn, ev, key, applied=False, note=note)
            _flag_anomaly(conn, f"{ev.client_order_id}: {note}", mode=mode, at=now)
            return ApplyResult(False, current, False, note, anomalies=(note,))

        order_qty = _d(row["quantity"])
        booked = ZERO
        absorbed = ZERO
        # ---- executions (always booked, whatever the state)
        if ev.fill_quantity is not None:
            b, a, n, an = self._book_execution_conn(conn, row, ev, now)
            booked, absorbed = b, a
            notes.extend(n)
            anomalies.extend(an)
        elif ev.event_type in FILL_EVENT_TYPES and ev.source == EventSource.STREAM and ev.cum_filled_quantity is None:
            notes.append("fill event without quantity: state only")
        # ---- broker-authoritative catch-up from a reconciliation snapshot
        if snapshot is not None and ev.source == EventSource.RECONCILE:
            b, n, an = self._catch_up_conn(conn, row, snapshot, ev, now)
            booked += b
            notes.extend(n)
            anomalies.extend(an)
        if booked or absorbed:
            filled_total = _refresh_order_fill_totals(conn, ev.client_order_id, now)
        else:
            filled_total = _d(row["filled_quantity"])
        if filled_total > order_qty:
            anomalies.append(f"booked quantity {_fmt(filled_total)} exceeds order quantity {_fmt(order_qty)}")

        # ---- state
        target = ev.state
        reported_cum = ev.cum_filled_quantity
        # Quantity evidence of completion wins over the reported status (an order whose whole quantity executed
        # is FILLED, even if the status says canceled/expired: a fill racing the cancel). Plain ACK/SUBMITTED
        # updates without executions are left alone so they are recognised (and noted) as stale.
        completes = (booked > 0 or absorbed > 0 or target in COMPLETION_STATES) and filled_total >= order_qty
        reports_complete = reported_cum is not None and reported_cum >= order_qty and target in COMPLETION_STATES
        if ev.source != EventSource.LOCAL and (completes or reports_complete):
            target = S.FILLED
        elif target == S.FILLED and reported_cum is not None and reported_cum < order_qty:
            notes.append(f"FILLED reported with cumulative {_fmt(reported_cum)} < {_fmt(order_qty)}")
        row = _require_row(conn, ev.client_order_id)
        new_state, decision_note = decide(current, target, source=ev.source)
        changed = False
        if new_state is not None:
            extra: dict[str, Any] = {}
            if current == S.UNKNOWN:
                extra["unknown_since"] = None
            if ev.broker_ts is not None:
                extra["last_broker_ts"] = iso(ev.broker_ts)
            if row["submitted_at"] is None and new_state in BROKER_OUTCOMES and ev.source != EventSource.LOCAL:
                extra["submitted_at"] = iso(ev.broker_ts or now)
            changed = _transition_conn(conn, row, new_state, reason=f"{ev.source.value} {ev.event_type.value}: {decision_note}",
                                       at=now, event_id=ev.event_id, extra_sets=extra)
            if changed and current == S.RECONCILED:
                anomalies.append(f"order {ev.client_order_id} was RECONCILED as absent but the broker reports {new_state.value}")
        elif current in PRE_SUBMISSION and ev.source != EventSource.LOCAL:
            anomalies.append(f"broker evidence for never-submitted order {ev.client_order_id} ({current.value})")
        status = ev.raw.get("broker_status") or ev.raw.get("status") or ev.event_type.value
        conn.execute("UPDATE orders SET broker_status=? WHERE client_order_id=?", (str(status)[:100], ev.client_order_id))
        notes.insert(0, decision_note)
        if anomalies:
            for a in anomalies:
                _flag_anomaly(conn, f"{ev.client_order_id}: {a}", mode=mode, at=now)
            notes.extend(f"ANOMALY: {a}" for a in anomalies)
        applied = changed or booked > 0 or absorbed > 0
        note = "; ".join(notes)
        _insert_event_conn(conn, ev, key, applied=applied, note=note)
        final = OrderState(_require_row(conn, ev.client_order_id)["state"])
        return ApplyResult(applied, final, False, note, booked, absorbed, tuple(anomalies))

    def _book_execution_conn(
        self, conn: sqlite3.Connection, row: sqlite3.Row, ev: OrderEvent, now: datetime
    ) -> tuple[Decimal, Decimal, list[str], list[str]]:
        q, p = ev.fill_quantity, ev.fill_price
        if q is None or q <= 0 or p is None or p <= 0:
            return ZERO, ZERO, [], [f"malformed execution (qty={q}, price={p}); not booked"]
        exec_id = effective_exec_id(ev)
        if exec_id is None:
            return ZERO, ZERO, [], ["execution without broker exec id or cumulative quantity cannot be deduplicated; not booked"]
        seen = conn.execute(
            "SELECT 1 FROM fills WHERE broker_exec_id=? UNION ALL SELECT 1 FROM fill_absorptions WHERE broker_exec_id=? LIMIT 1",
            (exec_id, exec_id),
        ).fetchone()
        if seen is not None:
            return ZERO, ZERO, [f"execution {exec_id} already booked"], []
        rows = _fill_rows(conn, row["client_order_id"])
        booked_total, _ = _booked_totals(rows)
        order_qty = _d(row["quantity"])
        _, synthetic_ranges, _ = _coverage(rows)
        has_synthetic = any(r["synthetic"] for r in rows)
        cum = ev.cum_filled_quantity
        absorbed = ZERO
        cum_from: Decimal | None = None
        if cum is not None:
            if cum < q:
                return ZERO, ZERO, [], [f"execution {exec_id} cumulative {_fmt(cum)} < fill {_fmt(q)}; not booked"]
            cum_from = cum - q
            absorbed = intervals.overlap((cum_from, cum), synthetic_ranges)
        notes: list[str] = []
        anomalies: list[str] = []
        overfill = booked_total + (q - absorbed) - order_qty
        if overfill > 0:
            absorbed += overfill
            if has_synthetic:
                notes.append(f"absorbed {_fmt(overfill)} already booked by reconciliation catch-up")
            else:
                anomalies.append(f"execution {exec_id} would over-fill the order by {_fmt(overfill)}; excess not booked")
        absorbed = min(absorbed, q)
        book = q - absorbed
        if absorbed > 0:
            conn.execute(
                "INSERT INTO fill_absorptions(client_order_id, broker_exec_id, reported_quantity, absorbed_quantity,"
                " event_id, at)"
                " VALUES (?,?,?,?,?,?)",
                (row["client_order_id"], exec_id, _fmt(q), _fmt(absorbed), ev.event_id, iso(now)),
            )
            notes.append(f"absorbed {_fmt(absorbed)} of execution {exec_id}")
        if book > 0:
            anomalies.extend(_insert_fill_conn(
                conn, row, exec_id=exec_id, qty=book, price=p, ts=ev.broker_ts or ev.received_ts, cum_from=cum_from,
                cum_to=cum, synthetic=False, event_id=ev.event_id, at=now,
            ))
            notes.append(f"booked fill {_fmt(book)} @ {_fmt(p)} ({exec_id})")
        return book, absorbed, notes, anomalies

    def _catch_up_conn(
        self, conn: sqlite3.Connection, row: sqlite3.Row, snap: BrokerOrder, ev: OrderEvent, now: datetime
    ) -> tuple[Decimal, list[str], list[str]]:
        rows = _fill_rows(conn, row["client_order_id"])
        booked_total, booked_notional = _booked_totals(rows)
        broker_cum = snap.filled_quantity
        if broker_cum == booked_total:
            return ZERO, [], []
        if broker_cum < booked_total:
            return ZERO, [], [f"ledger booked {_fmt(booked_total)} but broker reports only {_fmt(broker_cum)} filled"]
        if snap.avg_fill_price is None or snap.avg_fill_price <= 0:
            return ZERO, [], [f"broker reports {_fmt(broker_cum)} filled without an average price; catch-up not booked"]
        delta = broker_cum - booked_total
        covered, _, placed = _coverage(rows)
        missing = intervals.gaps(covered, broker_cum) if placed else []
        if not placed or intervals.measure(missing) != delta:
            missing_ranges: list[tuple[Decimal | None, Decimal | None, Decimal]] = [(None, None, delta)]
        else:
            missing_ranges = [(a, b, b - a) for a, b in missing]
        price = ((snap.avg_fill_price * broker_cum - booked_notional) / delta).quantize(PRICE_Q)
        anomalies: list[str] = []
        if price <= 0:
            anomalies.append("implied catch-up fill price non-positive; using broker average price")
            price = snap.avg_fill_price
        notes: list[str] = []
        for a, b, qty in missing_ranges:
            exec_id = f"recon:{row['client_order_id']}:{'?' if a is None else _norm(a)}-{_norm(broker_cum if b is None else b)}"
            if conn.execute("SELECT 1 FROM fills WHERE broker_exec_id=?", (exec_id,)).fetchone():
                continue
            anomalies.extend(_insert_fill_conn(
                conn, row, exec_id=exec_id, qty=qty, price=price, ts=snap.updated_at, cum_from=a, cum_to=b,
                synthetic=True, event_id=ev.event_id, at=now,
            ))
            notes.append(f"reconciliation catch-up fill {_fmt(qty)} @ {_fmt(price)} (broker cumulative {_fmt(broker_cum)})")
        return delta, notes, anomalies

    # ================================================================== engine queue (LEAN bridge)
    def enqueue_for_engine(self, intent: ApprovedOrderIntent) -> BrokerOrder:
        """Make a claimed (SUBMITTING) intent available to the LEAN engine. Idempotent."""
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> None:
            row = _order_row(conn, intent.client_order_id)
            if row is None or row["state"] != S.SUBMITTING.value:
                raise LedgerError(f"order {intent.client_order_id} is not SUBMITTING; cannot queue for the engine")
            if row["intent_json"] != intent.model_dump_json():
                raise LedgerError("intent differs from the ledger record")
            conn.execute(
                "INSERT INTO engine_intent_queue(client_order_id, intent_id, gateway, status, queued_at, expires_at)"
                " VALUES (?,?,?,?,?,?) ON CONFLICT(client_order_id) DO NOTHING",
                (intent.client_order_id, intent.intent_id, intent.gateway.value, "QUEUED", iso(now), iso(intent.expires_at)),
            )
            append_audit(conn, event_type="ENGINE_INTENT_QUEUED", source="order_ledger", mode=self.mode.value,
                         causal_id=intent.causal_id, ts=now,
                         payload={"client_order_id": intent.client_order_id, "intent_id": intent.intent_id})

        self.db.write(_w)
        rec = self.get_record(intent.client_order_id)
        assert rec is not None
        return rec.to_broker_order().model_copy(update={"broker_status": "queued_for_engine"})

    def claim_engine_intents(self, engine_id: str, max_n: int) -> list[ApprovedOrderIntent]:
        """Atomic QUEUED -> CLAIMED. Expired entries are never handed out; nothing is handed out twice."""
        now = self.clock.now()
        n = max(0, min(int(max_n), 50))

        def _w(conn: sqlite3.Connection) -> list[ApprovedOrderIntent]:
            _expire_engine_queue_conn(conn, now)
            out: list[ApprovedOrderIntent] = []
            rows = conn.execute(
                "SELECT q.client_order_id, o.state, o.intent_json FROM engine_intent_queue q JOIN orders o"
                " ON o.client_order_id = q.client_order_id WHERE q.status='QUEUED' AND q.expires_at > ?"
                " ORDER BY q.queued_at, q.client_order_id LIMIT ?",
                (iso(now), n),
            ).fetchall()
            for r in rows:
                if r["state"] != S.SUBMITTING.value:
                    conn.execute("UPDATE engine_intent_queue SET status='WITHDRAWN', note=? WHERE client_order_id=?",
                                 (f"order is {r['state']}", r["client_order_id"]))
                    continue
                intent = ApprovedOrderIntent.model_validate_json(r["intent_json"])
                if now >= intent.expires_at:
                    continue
                cur = conn.execute(
                    "UPDATE engine_intent_queue SET status='CLAIMED', claimed_by=?, claimed_at=?"
                    " WHERE client_order_id=? AND status='QUEUED'",
                    (engine_id[:200], iso(now), r["client_order_id"]),
                )
                if cur.rowcount != 1:
                    continue
                append_audit(conn, event_type="ENGINE_INTENT_CLAIMED", source=f"engine:{engine_id}"[:200],
                             mode=self.mode.value, causal_id=intent.causal_id, ts=now,
                             payload={"client_order_id": intent.client_order_id, "intent_id": intent.intent_id})
                out.append(intent)
            return out

        return self.db.write(_w)

    def withdraw_engine_intent(self, client_order_id: str, reason: str) -> bool:
        """QUEUED (never handed to the engine) -> WITHDRAWN and the order -> CANCELED (never sent)."""
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE engine_intent_queue SET status='WITHDRAWN', note=? WHERE client_order_id=? AND status='QUEUED'",
                (reason[:500], client_order_id),
            )
            if cur.rowcount != 1:
                return False
            row = _order_row(conn, client_order_id)
            if row is not None and row["state"] == S.SUBMITTING.value:
                _transition_conn(conn, row, S.CANCELED, reason=f"withdrawn before engine pickup (never sent): {reason}", at=now)
            return True

        return self.db.write(_w)

    def withdraw_queued_engine_intents(self, reason: str) -> list[str]:
        """Every QUEUED engine intent -> WITHDRAWN and its order -> REJECTED (never handed out, never sent).

        Used when entries are locked at engine-poll time (halt, mismatch, unknown orders, restart ...): an approval
        granted before the lock must not be released to the engine afterwards.
        """
        now = self.clock.now()

        def _w(conn: sqlite3.Connection) -> list[str]:
            out: list[str] = []
            for r in conn.execute("SELECT client_order_id FROM engine_intent_queue WHERE status='QUEUED'").fetchall():
                coid = str(r[0])
                cur = conn.execute(
                    "UPDATE engine_intent_queue SET status='WITHDRAWN', note=? WHERE client_order_id=? AND status='QUEUED'",
                    (reason[:500], coid),
                )
                if cur.rowcount != 1:
                    continue
                row = _order_row(conn, coid)
                if row is not None and row["state"] == S.SUBMITTING.value and _transition_conn(
                    conn, row, S.REJECTED, reason=f"withdrawn before engine pickup (never sent): {reason}", at=now
                ):
                    out.append(coid)
            return out

        return self.db.write(_w)

    def engine_queue_status(self, client_order_id: str) -> str | None:
        row = self.db.query_one("SELECT status FROM engine_intent_queue WHERE client_order_id=?", (client_order_id,))
        return None if row is None else str(row["status"])

    # ================================================================== queries
    def get_record(self, client_order_id: str) -> OrderRecord | None:
        row = self.db.query_one("SELECT * FROM orders WHERE client_order_id=?", (client_order_id,))
        return None if row is None else OrderRecord.from_row(row)

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        rec = self.get_record(client_order_id)
        return None if rec is None else rec.to_broker_order()

    def records(self, states: Iterable[OrderState] | None = None) -> list[OrderRecord]:
        if states is None:
            rows = self.db.query("SELECT * FROM orders ORDER BY created_at, client_order_id")
        else:
            values = [s.value for s in states]
            if not values:
                return []
            rows = self.db.query(
                f"SELECT * FROM orders WHERE state IN ({','.join('?' for _ in values)}) ORDER BY created_at, client_order_id",  # noqa: S608
                tuple(values),
            )
        return [OrderRecord.from_row(r) for r in rows]

    def open_orders(self) -> list[BrokerOrder]:
        """Every non-terminal order (incl. authorized-but-unsent and in-doubt), minus expired unsent intents."""
        now = self.clock.now()
        out: list[BrokerOrder] = []
        for rec in self.records(sorted(OPEN, key=lambda s: s.value)):
            if rec.state in PRE_SUBMISSION and rec.expires_at is not None and rec.expires_at <= now:
                continue
            out.append(rec.to_broker_order())
        return out

    def in_doubt_orders(self) -> list[OrderRecord]:
        return self.records(sorted(IN_DOUBT, key=lambda s: s.value))

    def working_orders(self) -> list[OrderRecord]:
        return self.records(sorted(WORKING, key=lambda s: s.value))

    def positions(self) -> list[PositionSnapshot]:
        now = self.clock.now()
        out: list[PositionSnapshot] = []
        for r in self.db.query("SELECT * FROM ledger_positions ORDER BY symbol"):
            qty = _d(r["quantity"])
            if qty == 0:
                continue
            out.append(PositionSnapshot(
                symbol=r["symbol"], quantity=qty, avg_entry_price=_d(r["avg_cost"]), cost_basis=_d(r["cost_basis"]),
                asof=now, source="LEDGER",
            ))
        return out

    def position_quantities(self) -> dict[str, Decimal]:
        return {p.symbol: p.quantity for p in self.positions()}

    def realized_pl_total(self, symbol: str | None = None) -> Decimal:
        rows = self.db.query("SELECT symbol, realized_pl FROM ledger_positions")
        return sum((_d(r["realized_pl"]) for r in rows if symbol is None or r["symbol"] == symbol), ZERO)

    def recent_submission_times(self, since: datetime) -> list[datetime]:
        rows = self.db.query("SELECT claimed_at FROM orders WHERE claimed_at IS NOT NULL AND claimed_at >= ? ORDER BY claimed_at",
                             (iso(since),))
        return [parse_iso(str(r["claimed_at"])) for r in rows]

    def recent_fingerprints(self, since: datetime) -> list[str]:
        """One entry per authorized order in the window (claimed, or still awaiting submission)."""
        rows = self.db.query(
            "SELECT intent_fingerprint FROM orders WHERE created_at >= ? AND (claimed_at IS NOT NULL OR state IN (?,?,?))"
            " ORDER BY created_at",
            (iso(since), S.PROPOSED.value, S.RISK_APPROVED.value, S.HUMAN_APPROVED.value),
        )
        return [str(r["intent_fingerprint"]) for r in rows]

    def daily_realized_pl(self, day: date | datetime) -> Decimal:
        """Realized P&L of fills on a U.S. trading day (America/New_York calendar date)."""
        key = trading_day(day).isoformat() if isinstance(day, datetime) else day.isoformat()
        rows = self.db.query("SELECT realized_pl FROM fills WHERE trading_day=?", (key,))
        return sum((_d(r["realized_pl"]) for r in rows), ZERO)

    def events(self, client_order_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in self.db.query("SELECT * FROM order_events WHERE client_order_id=? ORDER BY seq", (client_order_id,)):
            body = json.loads(r["payload_json"])
            body.update({"seq": r["seq"], "applied": bool(r["applied"]), "apply_note": r["apply_note"],
                         "dedupe_key": r["dedupe_key"]})
            out.append(body)
        return out

    def event_log(self, client_order_id: str | None = None) -> list[OrderEvent]:
        """Every stored event, in arrival order, as contract objects (for replay)."""
        if client_order_id is None:
            rows = self.db.query("SELECT payload_json FROM order_events ORDER BY seq")
        else:
            rows = self.db.query("SELECT payload_json FROM order_events WHERE client_order_id=? ORDER BY seq", (client_order_id,))
        return [OrderEvent.model_validate_json(r["payload_json"]) for r in rows]

    def transitions(self, client_order_id: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT from_state, to_state, at, reason, event_id FROM order_state_transitions WHERE client_order_id=? ORDER BY seq",
            (client_order_id,),
        )

    def fills(self, client_order_id: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT fill_id, broker_exec_id, symbol, side, quantity, price, ts, cum_from, cum_to, synthetic, realized_pl"
            " FROM fills WHERE client_order_id=? ORDER BY ts, fill_id",
            (client_order_id,),
        )

    def duplicate_count(self) -> int:
        row = self.db.query_one("SELECT COALESCE(SUM(deliveries), 0) AS n FROM order_event_duplicates")
        return int(row["n"]) if row else 0

    def unknown_orders(self) -> list[str]:
        return [r.client_order_id for r in self.records([S.UNKNOWN])]

    def is_terminal(self, client_order_id: str) -> bool:
        rec = self.get_record(client_order_id)
        return rec is not None and rec.state in TERMINAL


def _snapshot_mismatch(row: sqlite3.Row, snap: BrokerOrder) -> str:
    problems: list[str] = []
    if snap.symbol != row["symbol"]:
        problems.append(f"symbol {snap.symbol}!={row['symbol']}")
    if snap.order_type.value != row["order_type"]:
        problems.append(f"order type {snap.order_type.value}!={row['order_type']}")
    if snap.side.value != row["side"]:
        problems.append(f"side {snap.side.value}!={row['side']}")
    if snap.quantity != _d(row["quantity"]):
        problems.append(f"quantity {_fmt(snap.quantity)}!={row['quantity']}")
    if row["limit_price"] is not None and snap.limit_price is not None and snap.limit_price != _d(row["limit_price"]):
        problems.append(f"limit {_fmt(snap.limit_price)}!={row['limit_price']}")
    if snap.filled_quantity > snap.quantity:
        problems.append("broker filled quantity exceeds order quantity")
    return ", ".join(problems)


def _expire_engine_queue_conn(conn: sqlite3.Connection, now: datetime) -> list[str]:
    """QUEUED engine entries past expiry -> EXPIRED (never handed out => never sent) and order -> EXPIRED."""
    out: list[str] = []
    rows = conn.execute(
        "SELECT client_order_id FROM engine_intent_queue WHERE status='QUEUED' AND expires_at <= ?", (iso(now),)
    ).fetchall()
    for r in rows:
        coid = r[0]
        cur = conn.execute(
            "UPDATE engine_intent_queue SET status='EXPIRED', note='expired before engine pickup'"
            " WHERE client_order_id=? AND status='QUEUED'",
            (coid,),
        )
        if cur.rowcount != 1:
            continue
        row = _order_row(conn, coid)
        if row is not None and row["state"] == S.SUBMITTING.value and _transition_conn(
            conn, row, S.EXPIRED, reason="intent expired before engine pickup (never sent)", at=now
        ):
            out.append(coid)
    return out


def reject_submitting_conn(conn: sqlite3.Connection, client_order_id: str, reason: str, at: datetime) -> bool:
    """SUBMITTING -> REJECTED inside an open transaction (engine-side GatedBrokerageModel refusal)."""
    row = _order_row(conn, client_order_id)
    if row is None or row["state"] != S.SUBMITTING.value:
        return False
    return _transition_conn(conn, row, S.REJECTED, reason=f"engine refused: {reason}", at=at)
