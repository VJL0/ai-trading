"""Order ledger state machine: the transition table and the ordering rules (pure, no I/O)."""

from __future__ import annotations

import itertools

import pytest

from packages.contracts.enums import TERMINAL_ORDER_STATES, EventSource, OrderState
from services.order_ledger.state_machine import (
    ALLOWED_TRANSITIONS,
    BROKER_OUTCOMES,
    LOCAL_ONLY_TARGETS,
    PRE_SUBMISSION,
    RANK,
    decide,
    is_allowed,
)

S = OrderState
BROKER_SOURCES = (EventSource.STREAM, EventSource.POLL, EventSource.RECONCILE, EventSource.GATEWAY)


def test_table_covers_all_14_states() -> None:
    assert len(OrderState) == 14
    assert set(ALLOWED_TRANSITIONS) == set(OrderState)
    assert set(RANK) == set(OrderState)
    for targets in ALLOWED_TRANSITIONS.values():
        assert targets <= set(OrderState)


def test_filled_and_rejected_are_absorbing() -> None:
    assert ALLOWED_TRANSITIONS[S.FILLED] == frozenset()
    assert ALLOWED_TRANSITIONS[S.REJECTED] == frozenset()


def test_terminal_states_only_exit_to_fill_completion_or_broker_correction() -> None:
    assert ALLOWED_TRANSITIONS[S.CANCELED] == {S.FILLED}
    assert ALLOWED_TRANSITIONS[S.EXPIRED] == {S.FILLED}
    # RECONCILED ("not present at broker") is corrected only to broker truth
    assert ALLOWED_TRANSITIONS[S.RECONCILED] == BROKER_OUTCOMES


def test_unknown_entered_only_from_submitting() -> None:
    sources = {s for s, targets in ALLOWED_TRANSITIONS.items() if S.UNKNOWN in targets}
    assert sources == {S.SUBMITTING}


def test_submitting_entered_only_from_approved_states() -> None:
    sources = {s for s, targets in ALLOWED_TRANSITIONS.items() if S.SUBMITTING in targets}
    assert sources == {S.RISK_APPROVED, S.HUMAN_APPROVED}


def test_nothing_returns_to_pre_submission_states() -> None:
    for frm, targets in ALLOWED_TRANSITIONS.items():
        if frm in PRE_SUBMISSION:
            continue
        assert not (targets & PRE_SUBMISSION), frm


@pytest.mark.parametrize("source", BROKER_SOURCES)
@pytest.mark.parametrize("target", sorted(LOCAL_ONLY_TARGETS, key=lambda s: s.value))
def test_broker_evidence_cannot_enter_local_only_states(target: OrderState, source: EventSource) -> None:
    for current in OrderState:
        new, _ = decide(current, target, source=source)
        assert new is None, (current, target, source)


@pytest.mark.parametrize("source", [EventSource.STREAM, EventSource.POLL, EventSource.GATEWAY, EventSource.LOCAL])
def test_only_reconciliation_enters_reconciled(source: EventSource) -> None:
    new, note = decide(S.UNKNOWN, S.RECONCILED, source=source)
    assert new is None and "RECONCILED" in note
    assert decide(S.UNKNOWN, S.RECONCILED, source=EventSource.RECONCILE)[0] == S.RECONCILED


@pytest.mark.parametrize("current", sorted(PRE_SUBMISSION, key=lambda s: s.value))
def test_broker_evidence_for_never_submitted_order_never_changes_state(current: OrderState) -> None:
    for target, source in itertools.product(BROKER_OUTCOMES, BROKER_SOURCES):
        assert decide(current, target, source=source)[0] is None


def test_ack_after_fill_does_not_regress() -> None:
    assert decide(S.FILLED, S.ACKNOWLEDGED, source=EventSource.STREAM)[0] is None
    assert decide(S.PARTIALLY_FILLED, S.ACKNOWLEDGED, source=EventSource.STREAM)[0] is None
    assert decide(S.PARTIALLY_FILLED, S.SUBMITTED, source=EventSource.STREAM)[0] is None


def test_fill_before_ack_is_applied() -> None:
    assert decide(S.SUBMITTING, S.FILLED, source=EventSource.STREAM)[0] == S.FILLED
    assert decide(S.SUBMITTED, S.PARTIALLY_FILLED, source=EventSource.STREAM)[0] == S.PARTIALLY_FILLED


def test_fill_completing_after_cancel_is_applied() -> None:
    assert decide(S.CANCEL_PENDING, S.FILLED, source=EventSource.STREAM)[0] == S.FILLED
    assert decide(S.CANCELED, S.FILLED, source=EventSource.STREAM)[0] == S.FILLED
    assert decide(S.EXPIRED, S.FILLED, source=EventSource.STREAM)[0] == S.FILLED


@pytest.mark.parametrize("terminal", sorted(TERMINAL_ORDER_STATES - {S.RECONCILED}, key=lambda s: s.value))
def test_terminal_states_never_regress_on_stream(terminal: OrderState) -> None:
    for target in OrderState:
        new, _ = decide(terminal, target, source=EventSource.STREAM)
        if new is not None:
            assert terminal in (S.CANCELED, S.EXPIRED) and new == S.FILLED


def test_cancel_pending_reverts_only_on_rest_evidence() -> None:
    assert decide(S.CANCEL_PENDING, S.ACKNOWLEDGED, source=EventSource.STREAM)[0] is None
    assert decide(S.CANCEL_PENDING, S.PARTIALLY_FILLED, source=EventSource.STREAM)[0] is None
    assert decide(S.CANCEL_PENDING, S.ACKNOWLEDGED, source=EventSource.POLL)[0] == S.ACKNOWLEDGED
    assert decide(S.CANCEL_PENDING, S.PARTIALLY_FILLED, source=EventSource.RECONCILE)[0] == S.PARTIALLY_FILLED


def test_unknown_resolved_by_any_broker_outcome() -> None:
    for target in BROKER_OUTCOMES:
        assert decide(S.UNKNOWN, target, source=EventSource.RECONCILE)[0] == target
        assert decide(S.UNKNOWN, target, source=EventSource.STREAM)[0] == target


def test_decide_results_always_respect_the_table() -> None:
    for current, target, source in itertools.product(OrderState, OrderState, EventSource):
        new, note = decide(current, target, source=source)
        assert isinstance(note, str) and note
        if new is not None:
            assert is_allowed(current, new), (current, new, source)


def test_rank_is_monotonic_along_the_happy_path() -> None:
    path = [S.PROPOSED, S.RISK_APPROVED, S.HUMAN_APPROVED, S.SUBMITTING, S.SUBMITTED, S.ACKNOWLEDGED,
            S.PARTIALLY_FILLED, S.FILLED]
    assert [RANK[s] for s in path] == sorted(RANK[s] for s in path)
    assert RANK[S.FILLED] == max(RANK.values())
