"""Property tests (hypothesis): any permutation/duplication of a valid broker event sequence converges.

Each example builds a broker-valid event sequence for one order (full fill in chunks, partial fill then
cancel, partial fill then DAY expiry, cancel/fill race, reject), applies it in broker order to one order and
as an arbitrary permutation with arbitrary re-deliveries to an identical twin order, then checks that both
end in the same state with the same filled quantity, average fill price and position.
"""

from __future__ import annotations

import itertools
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from packages.contracts.enums import OrderEventType, OrderState, Side
from packages.contracts.models import OrderEvent
from tests.unit.order_ledger.ol_support import T0, Harness, claim, make_harness, stream_event

S = OrderState
E = OrderEventType
PRICES = ("99.98", "99.99", "100.00", "100.01", "100.37")


@dataclass(frozen=True)
class Step:
    etype: OrderEventType
    state: OrderState
    fill: tuple[Decimal, str] | None  # (quantity, price)
    cum: Decimal | None


@dataclass(frozen=True)
class Scenario:
    kind: str
    qty: int
    steps: tuple[Step, ...]

    @property
    def final_state(self) -> OrderState:
        return {"fill": S.FILLED, "cancel_race": S.FILLED, "partial_cancel": S.CANCELED,
                "partial_expire": S.EXPIRED, "reject": S.REJECTED}[self.kind]


def _chunks(draw: st.DrawFn, total: int, max_parts: int) -> list[int]:
    if total == 0:
        return []
    parts = draw(st.integers(min_value=1, max_value=min(max_parts, total)))
    cuts = sorted(draw(st.lists(st.integers(1, total - 1), min_size=parts - 1, max_size=parts - 1, unique=True))) \
        if total > 1 and parts > 1 else []
    bounds = [0, *cuts, total]
    return [b - a for a, b in itertools.pairwise(bounds) if b > a]


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    kind = draw(st.sampled_from(["fill", "partial_cancel", "partial_expire", "cancel_race", "reject"]))
    qty = draw(st.integers(min_value=1, max_value=30))
    steps: list[Step] = [Step(E.ACCEPTED, S.SUBMITTED, None, Decimal(0)), Step(E.NEW, S.ACKNOWLEDGED, None, Decimal(0))]
    if kind == "reject":
        return Scenario(kind, qty, (steps[0], Step(E.REJECTED, S.REJECTED, None, Decimal(0))))
    filled_target = qty if kind in ("fill", "cancel_race") else draw(st.integers(min_value=0, max_value=qty - 1))
    if kind == "cancel_race":
        steps.append(Step(E.PENDING_CANCEL, S.CANCEL_PENDING, None, None))
    cum = 0
    chunks = _chunks(draw, filled_target, 4)
    for n, q in enumerate(chunks):
        cum += q
        last = cum == qty
        px = draw(st.sampled_from(PRICES))
        steps.append(Step(E.FILL if last else E.PARTIAL_FILL, S.FILLED if last else S.PARTIALLY_FILLED,
                          (Decimal(q), px), Decimal(cum)))
        if kind == "partial_cancel" and n == len(chunks) - 1:
            pass
    if kind == "partial_cancel":
        steps.append(Step(E.PENDING_CANCEL, S.CANCEL_PENDING, None, Decimal(cum)))
        steps.append(Step(E.CANCELED, S.CANCELED, None, Decimal(cum)))
    elif kind == "partial_expire":
        steps.append(Step(E.EXPIRED, S.EXPIRED, None, Decimal(cum)))
    return Scenario(kind, qty, tuple(steps))


def _events(sc: Scenario, coid: str, tag: str) -> list[OrderEvent]:
    out: list[OrderEvent] = []
    for i, step in enumerate(sc.steps):
        out.append(stream_event(
            coid, step.etype, step.state, T0, event_id=f"{tag}-ev-{i}",
            fill=(str(step.fill[0]), step.fill[1]) if step.fill else None,
            cum=None if step.cum is None else str(step.cum),
            exec_id=f"{tag}-x-{i}" if step.fill else None, broker_order_id=f"bo-{tag}",
        ))
    return out


def _outcome(h: Harness, coid: str, symbol: str) -> tuple[OrderState, Decimal, Decimal | None, Decimal, Decimal]:
    order = h.ledger.get_order(coid)
    assert order is not None
    pos = {p.symbol: p for p in h.ledger.positions()}.get(symbol)
    return (order.state, order.filled_quantity, order.avg_fill_price,
            pos.quantity if pos else Decimal(0), pos.cost_basis if pos else Decimal(0))


PROPERTY_SETTINGS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPERTY_SETTINGS
@given(sc=scenarios(), data=st.data())
def test_permuted_and_duplicated_events_converge(sc: Scenario, data: st.DataObject) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = make_harness(Path(tmp))
        try:
            a = h.factory.make(symbol="SPY", qty=str(sc.qty), limit="101.00")
            b = h.factory.make(symbol="QQQ", qty=str(sc.qty), limit="101.00")
            claim(h, a)
            claim(h, b)
            for ev in _events(sc, a.client_order_id, "a"):
                h.ledger.apply_event(ev)
            twin = _events(sc, b.client_order_id, "b")
            dups = data.draw(st.lists(st.sampled_from(twin), max_size=len(twin)))
            shuffled = data.draw(st.permutations(twin + dups))
            for ev in shuffled:
                h.ledger.apply_event(ev)
            ref = _outcome(h, a.client_order_id, "SPY")
            got = _outcome(h, b.client_order_id, "QQQ")
            assert ref[0] == sc.final_state, (sc, ref)
            assert got == ref, (sc.kind, [e.event_type.value for e in shuffled], ref, got)
            assert sum(Decimal(f["quantity"]) for f in h.ledger.fills(b.client_order_id)) == got[1]
            # replaying the whole stored event log changes nothing (all duplicates)
            before = (_outcome(h, a.client_order_id, "SPY"), _outcome(h, b.client_order_id, "QQQ"))
            for ev in h.ledger.event_log():
                assert h.ledger.apply_event(ev).duplicate
            assert (_outcome(h, a.client_order_id, "SPY"), _outcome(h, b.client_order_id, "QQQ")) == before
        finally:
            h.db.close()


@PROPERTY_SETTINGS
@given(buy=scenarios().filter(lambda s: s.kind in ("fill", "cancel_race")), data=st.data())
def test_buy_then_sell_positions_converge_under_reordering(buy: Scenario, data: st.DataObject) -> None:
    """A buy order fully filled, then a sell of part of it; each order's events permuted independently."""
    sell_qty = data.draw(st.integers(min_value=1, max_value=buy.qty))
    with tempfile.TemporaryDirectory() as tmp:
        h = make_harness(Path(tmp))
        try:
            results = []
            for tag in ("ref", "perm"):
                symbol = "SPY" if tag == "ref" else "QQQ"
                bi = h.factory.make(symbol=symbol, qty=str(buy.qty), limit="101.00")
                claim(h, bi)
                bevs = _events(buy, bi.client_order_id, f"{tag}b")
                if tag == "perm":
                    bevs = data.draw(st.permutations(bevs + bevs[:2]))
                for ev in bevs:
                    h.ledger.apply_event(ev)
                si = h.factory.make(symbol=symbol, side=Side.SELL, qty=str(sell_qty), limit="99.00")
                claim(h, si)
                sell = Scenario("fill", sell_qty, (
                    Step(E.NEW, S.ACKNOWLEDGED, None, Decimal(0)),
                    Step(E.FILL, S.FILLED, (Decimal(sell_qty), "100.50"), Decimal(sell_qty)),
                ))
                sevs = _events(sell, si.client_order_id, f"{tag}s")
                if tag == "perm":
                    sevs = data.draw(st.permutations(sevs + sevs))
                for ev in sevs:
                    h.ledger.apply_event(ev)
                pos = {p.symbol: p for p in h.ledger.positions()}.get(symbol)
                realized = h.ledger.realized_pl_total(symbol)
                results.append((pos.quantity if pos else Decimal(0), pos.avg_entry_price if pos else Decimal(0),
                                realized))
            assert results[0] == results[1]
            assert results[0][0] == Decimal(buy.qty - sell_qty)
        finally:
            h.db.close()
