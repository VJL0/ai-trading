"""Property test (hypothesis): dropped stream events + broker-authoritative catch-up + late re-delivery converge.

A random subset of a broker-valid event sequence is lost (stream outage); reconciliation applies the broker's
REST snapshot (cumulative filled quantity, average price, final status), which books catch-up fills for the
uncovered quantity; the lost events then arrive late, permuted and duplicated. The order must end with exactly the
broker's filled quantity and state, the position must equal the broker's, nothing may be double-booked and no
anomaly may be raised. Deterministic and offline (no broker at all: events and snapshots are constructed).
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from packages.contracts.enums import EventSource, GatewayKind, OrderType, Side, TimeInForce
from packages.contracts.models import BrokerOrder
from services.order_ledger.entries import Flag
from tests.unit.order_ledger.ol_support import T0, claim, make_harness
from tests.unit.order_ledger.test_order_ledger_properties import Scenario, _events, scenarios

WITH_FILLS = scenarios().filter(lambda s: s.kind != "reject" and any(step.fill for step in s.steps))


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(sc=WITH_FILLS, data=st.data())
def test_lost_events_caught_up_by_reconciliation_then_late_delivery_converges(sc: Scenario, data: st.DataObject) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = make_harness(Path(tmp))
        try:
            intent = h.factory.make(symbol="SPY", qty=str(sc.qty), limit="101.00")
            claim(h, intent)
            coid = intent.client_order_id
            evs = _events(sc, coid, "x")
            lost_idx = data.draw(st.sets(st.integers(0, len(evs) - 1)))
            for i, ev in enumerate(evs):
                if i not in lost_idx:
                    h.ledger.apply_event(ev)
            total = sum((step.fill[0] for step in sc.steps if step.fill), Decimal(0))
            notional = sum((step.fill[0] * Decimal(step.fill[1]) for step in sc.steps if step.fill), Decimal(0))
            snapshot = BrokerOrder(
                client_order_id=coid, broker_order_id="bo-x", intent_id=None, gateway=GatewayKind.FAKE, symbol="SPY",
                side=Side.BUY, quantity=Decimal(sc.qty), filled_quantity=total,
                avg_fill_price=(notional / total).quantize(Decimal("0.00000001")), order_type=OrderType.LIMIT,
                limit_price=Decimal("101.00"), time_in_force=TimeInForce.DAY, broker_status=sc.final_state.value.lower(),
                state=sc.final_state, submitted_at=T0, updated_at=T0,
            )
            res = h.ledger.apply_broker_order(snapshot, EventSource.RECONCILE)
            assert res.anomalies == (), res.note
            late = [evs[i] for i in sorted(lost_idx)]
            dups = data.draw(st.lists(st.sampled_from(evs), max_size=len(evs)))
            for ev in data.draw(st.permutations(late + dups)):
                assert h.ledger.apply_event(ev).anomalies == ()
            order = h.ledger.get_order(coid)
            assert order is not None
            assert order.state == sc.final_state
            assert order.filled_quantity == total
            booked = sum((Decimal(f["quantity"]) for f in h.ledger.fills(coid)), Decimal(0))
            assert booked == total
            [pos] = h.ledger.positions()
            assert pos.quantity == total
            assert abs(pos.cost_basis - notional) <= Decimal("0.000001") * sc.qty
            assert Flag.LEDGER_ANOMALY.value not in h.entries.flags()
        finally:
            h.db.close()
