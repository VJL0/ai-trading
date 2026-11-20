"""Order ledger state machine (spec §7).

``ALLOWED_TRANSITIONS`` is the single authority on which state changes the ledger may write; ``decide``
adds the ordering rules that make event application idempotent and order-independent:

* Broker-side progress is monotonic by ``RANK``: an event whose implied state ranks *below* the current
  state is stale (out-of-order) and is recorded but not applied (an ACK after a fill never regresses).
* Terminal states never regress. The only exits are (a) fill evidence that *completes* the order (a racing
  fill booked after CANCELED/EXPIRED -> FILLED) and (b) broker-authoritative correction of RECONCILED.
  Fills themselves are always booked by the ledger, whatever the state (money truth beats state).
* ``UNKNOWN`` is entered only from ``SUBMITTING`` and only locally (submission outcome unknown). It is
  left only by broker evidence (stream, poll, reconciliation lookup by client order id) or by
  reconciliation marking it ``RECONCILED`` ("not present at broker").
* ``CANCEL_PENDING`` returns to a working state only on broker-authoritative REST evidence (RECONCILE
  or POLL: the broker refused the cancel and the order is still working), never on a stale stream event.
  (``OrderLedger.revert_cancel_request`` also undoes it LOCALLY when the cancel was never transmitted.)
* Pre-submission states (PROPOSED/RISK_APPROVED/HUMAN_APPROVED) move only on LOCAL decisions. Broker
  evidence about an order that was never submitted is an anomaly and never changes state.
"""

from __future__ import annotations

from typing import Final

from packages.contracts.enums import TERMINAL_ORDER_STATES, EventSource, OrderEventType, OrderState

S = OrderState

PRE_SUBMISSION: Final[frozenset[OrderState]] = frozenset({S.PROPOSED, S.RISK_APPROVED, S.HUMAN_APPROVED})
WORKING: Final[frozenset[OrderState]] = frozenset({S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.CANCEL_PENDING})
IN_DOUBT: Final[frozenset[OrderState]] = frozenset({S.SUBMITTING, S.UNKNOWN})
TERMINAL: Final[frozenset[OrderState]] = TERMINAL_ORDER_STATES
OPEN: Final[frozenset[OrderState]] = frozenset(set(S) - set(TERMINAL))
BROKER_OUTCOMES: Final[frozenset[OrderState]] = frozenset(
    {S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.REJECTED, S.EXPIRED}
)
LOCAL_ONLY_TARGETS: Final[frozenset[OrderState]] = frozenset(
    {S.PROPOSED, S.RISK_APPROVED, S.HUMAN_APPROVED, S.SUBMITTING, S.UNKNOWN}
)

ALLOWED_TRANSITIONS: Final[dict[OrderState, frozenset[OrderState]]] = {
    S.PROPOSED: frozenset({S.RISK_APPROVED, S.REJECTED, S.EXPIRED, S.CANCELED}),
    S.RISK_APPROVED: frozenset({S.HUMAN_APPROVED, S.SUBMITTING, S.REJECTED, S.EXPIRED, S.CANCELED}),
    S.HUMAN_APPROVED: frozenset({S.SUBMITTING, S.REJECTED, S.EXPIRED, S.CANCELED}),
    # EXPIRED/CANCELED from SUBMITTING: an engine-queued intent expired or was withdrawn before pickup.
    S.SUBMITTING: BROKER_OUTCOMES | {S.UNKNOWN},
    S.SUBMITTED: frozenset(
        {S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.REJECTED, S.EXPIRED}
    ),
    S.ACKNOWLEDGED: frozenset({S.PARTIALLY_FILLED, S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.REJECTED, S.EXPIRED}),
    S.PARTIALLY_FILLED: frozenset({S.FILLED, S.CANCEL_PENDING, S.CANCELED, S.EXPIRED}),
    # Back to SUBMITTED/ACKNOWLEDGED/PARTIALLY_FILLED only on broker REST evidence that the cancel did not take
    # effect, or LOCALLY when the cancel request was never transmitted (gateway has no cancel channel).
    S.CANCEL_PENDING: frozenset({S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.EXPIRED}),
    S.UNKNOWN: BROKER_OUTCOMES | {S.RECONCILED},
    S.FILLED: frozenset(),
    S.CANCELED: frozenset({S.FILLED}),
    S.EXPIRED: frozenset({S.FILLED}),
    S.REJECTED: frozenset(),
    # A RECONCILED ("not present at broker") order that later shows up at the broker is corrected to
    # broker truth, and the ledger raises the ledger_anomaly flag (entries locked).
    S.RECONCILED: BROKER_OUTCOMES,
}

if set(ALLOWED_TRANSITIONS) != set(S):  # pragma: no cover - import-time invariant
    raise RuntimeError("transition table must cover all 14 OrderState values")

RANK: Final[dict[OrderState, int]] = {
    S.PROPOSED: 0,
    S.RISK_APPROVED: 1,
    S.HUMAN_APPROVED: 2,
    S.SUBMITTING: 3,
    S.UNKNOWN: 3,
    S.SUBMITTED: 4,
    S.ACKNOWLEDGED: 5,
    S.PARTIALLY_FILLED: 6,
    S.CANCEL_PENDING: 7,
    S.CANCELED: 8,
    S.EXPIRED: 8,
    S.REJECTED: 8,
    S.RECONCILED: 8,
    S.FILLED: 9,
}

# Event type that best describes a broker-reported state (used when converting broker snapshots to events).
STATE_EVENT_TYPE: Final[dict[OrderState, OrderEventType]] = {
    S.SUBMITTING: OrderEventType.PENDING_NEW,
    S.SUBMITTED: OrderEventType.ACCEPTED,
    S.ACKNOWLEDGED: OrderEventType.NEW,
    S.PARTIALLY_FILLED: OrderEventType.PARTIAL_FILL,
    S.FILLED: OrderEventType.FILL,
    S.CANCEL_PENDING: OrderEventType.PENDING_CANCEL,
    S.CANCELED: OrderEventType.CANCELED,
    S.REJECTED: OrderEventType.REJECTED,
    S.EXPIRED: OrderEventType.EXPIRED,
    S.UNKNOWN: OrderEventType.UNKNOWN,
    S.RECONCILED: OrderEventType.RECONCILED,
}


def is_allowed(current: OrderState, target: OrderState) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def is_terminal(state: OrderState) -> bool:
    return state in TERMINAL


def decide(current: OrderState, target: OrderState, *, source: EventSource) -> tuple[OrderState | None, str]:
    """Return ``(new_state or None when the state must not change, note)``. Pure; never raises."""
    if target == current:
        return None, f"no state change ({current.value})"
    if target in LOCAL_ONLY_TARGETS and source != EventSource.LOCAL:
        return None, f"{target.value} may only be entered by a local ledger decision, not {source.value} evidence"
    if target == S.RECONCILED and source != EventSource.RECONCILE:
        return None, "RECONCILED may only be entered by the reconciliation service"
    if current in PRE_SUBMISSION and source != EventSource.LOCAL:
        return None, f"broker evidence ({source.value}) for an order that was never submitted (state {current.value})"
    if not is_allowed(current, target):
        if current in TERMINAL:
            return None, f"terminal state {current.value} does not regress to {target.value}"
        return None, f"transition {current.value}->{target.value} not allowed"
    if current in IN_DOUBT or current == S.RECONCILED or current in PRE_SUBMISSION:
        # The allowed set already constrains these; broker evidence resolves doubt.
        return target, f"{current.value}->{target.value}"
    if current == S.CANCEL_PENDING and target in (S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED):
        if source in (EventSource.RECONCILE, EventSource.POLL):
            return target, "broker reports the order still working (cancel not effective)"
        return None, f"stale {target.value} event after cancel request"
    if RANK[target] > RANK[current]:
        return target, f"{current.value}->{target.value}"
    return None, f"stale/out-of-order event: {target.value} would regress {current.value}"
