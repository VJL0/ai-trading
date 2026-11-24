"""Exit plumbing below the position manager: the exit lock, FakeBroker stop-limits, Alpaca stop-limit orders."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import EventSource, GatewayKind, OrderState, OrderType, ReconciliationStatus, Side
from packages.contracts.interfaces import GatewayRejected
from services.order_ledger.alpaca_gateway import AlpacaPaperGateway
from services.order_ledger.alpaca_reader import map_order
from services.order_ledger.entries import Flag
from services.order_ledger.errors import BrokerPayloadError
from services.order_ledger.execution import ExecutionService
from services.order_ledger.fake_broker import FakeBroker
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import ORDER, T0, Harness, make_quote, paper_secret_store
from tests.unit.order_ledger.test_alpaca_gateway import BrokerHTTP

D = Decimal


# ------------------------------------------------------------------------------------------------ exit lock
async def test_exit_lock_ignores_entry_halts_and_position_safety_but_not_uncertainty(h: Harness) -> None:
    rec = Reconciler(h.db, h.ledger, h.broker, h.entries, h.clock, confirm_delay_s=0)
    assert (await rec.startup_recovery()).status == ReconciliationStatus.CLEAN
    assert h.entries.exits_locked() == (False, [])
    h.entries.lock("stop new entries", "operator")
    h.entries.set_position_safety("SPY", "UNPROTECTED", "no confirmed stop")
    locked, reasons = h.entries.is_locked()
    assert locked and any("scope=ENTRIES" in r for r in reasons) and any("SPY UNPROTECTED" in r for r in reasons)
    assert h.entries.exits_locked() == (False, [])
    kill = h.entries.lock("kill switch", "operator", scope="ALL")
    xlocked, xreasons = h.entries.exits_locked()
    assert xlocked and any("scope=ALL" in r for r in xreasons)
    h.entries.clear_halt(kill, by="victor", reason="done")
    h.entries.set_flag(Flag.BROKER_DISCONNECTED, "stream down", "execution")
    assert h.entries.exits_locked()[0]  # state uncertainty blocks exits too
    assert h.entries.clear_position_safety("SPY", "protected")
    assert not any("SPY" in r for r in h.entries.is_locked()[1])
    with pytest.raises(ValueError, match="unknown position safety state"):
        h.entries.set_position_safety("SPY", "FINE", "nope")


async def test_execution_service_sends_a_sell_under_an_entries_halt_and_refuses_a_buy(h: Harness) -> None:
    rec = Reconciler(h.db, h.ledger, h.broker, h.entries, h.clock, confirm_delay_s=0)
    await rec.startup_recovery()
    h.broker.seed_position("SPY", D(5), D("100"))
    h.ledger.db.write(lambda c: c.execute(
        "INSERT INTO ledger_positions(symbol, quantity, avg_cost, realized_pl, updated_at, cost_basis)"
        " VALUES ('SPY','5','100','0','2026-11-04T15:00:00.000000Z','500')"))
    h.quotes.set(make_quote("SPY", "100.00", "100.02", h.clock.now()))
    h.entries.lock("stop new entries", "operator")
    ex = ExecutionService(h.ledger, h.broker, lambda _i: [], h.clock, entries_lock=h.entries)
    buy = h.factory.make(side=Side.BUY, qty="1", limit="100.02")
    assert await ex.submit(buy) is None
    assert h.ledger.get_record(buy.client_order_id).state == OrderState.REJECTED
    sell = h.factory.make(side=Side.SELL, qty="5", limit="100.00")
    order = await ex.submit(sell)
    assert order is not None and h.broker.submission_count(sell.client_order_id) == 1


# ------------------------------------------------------------------------------------------------ fake broker
def _stop_intent(h: Harness, qty: str = "5", stop: str = "99.00", limit: str = "98.50") -> Any:
    return h.factory.make(side=Side.SELL, qty=qty, limit=limit, order_type=OrderType.STOP_LIMIT, stop=stop)


async def test_fake_broker_stop_limit_rests_triggers_and_can_gap_unfilled(tmp_path: Path, h: Harness) -> None:
    h.broker.seed_position("SPY", D(5), D("100"))
    h.quotes.set(make_quote("SPY", "100.00", "100.02", h.clock.now()))
    intent = _stop_intent(h)
    h.ledger.claim_for_submission(intent.client_order_id, "test")
    snap = await h.broker.submit(intent)
    assert snap.state == OrderState.ACKNOWLEDGED and snap.order_type == OrderType.STOP_LIMIT
    assert snap.stop_price == D("99.00") and snap.filled_quantity == 0
    h.clock.advance(1)
    h.broker.advance(make_quote("SPY", "97.00", "97.05", h.clock.now()))  # gap below the limit: triggered, unfilled
    o = h.broker.orders[intent.client_order_id]
    assert o.triggered and o.filled == 0 and o.is_open
    h.clock.advance(1)
    h.broker.advance(make_quote("SPY", "98.60", "98.65", h.clock.now()))  # back above the limit: fills at the bid
    assert o.status == "filled" and o.notional == D("493.00")
    # state persists across a broker restart (protective fields included)
    path = tmp_path / "fb.json"
    fb = FakeBroker(h.clock, h.quotes, state_path=path, session_bounds_fn=h.broker.session_bounds_fn)
    fb.orders = h.broker.orders
    fb._save()
    again = FakeBroker(h.clock, h.quotes, state_path=path)
    assert again.orders[intent.client_order_id].order_type == OrderType.STOP_LIMIT
    assert again.orders[intent.client_order_id].stop == D("99.00") and again.orders[intent.client_order_id].triggered


async def test_fake_broker_stop_limit_shares_are_held_against_other_sells(h: Harness) -> None:
    h.broker.seed_position("SPY", D(5), D("100"))
    h.quotes.set(make_quote("SPY", "100.00", "100.02", h.clock.now()))
    stop = _stop_intent(h)
    h.ledger.claim_for_submission(stop.client_order_id, "test")
    await h.broker.submit(stop)
    sell = h.factory.make(side=Side.SELL, qty="1", limit="100.00")
    h.ledger.claim_for_submission(sell.client_order_id, "test")
    with pytest.raises(GatewayRejected, match="insufficient position"):
        await h.broker.submit(sell)


# ------------------------------------------------------------------------------------------------ alpaca
def test_alpaca_maps_only_protective_stop_limit_sells() -> None:
    raw = {**ORDER, "order_type": "stop_limit", "type": "stop_limit", "side": "sell", "stop_price": "99.00",
           "limit_price": "98.50", "status": "new", "filled_qty": "0"}
    bo = map_order(raw, T0, GatewayKind.ALPACA_PAPER)
    assert bo.order_type == OrderType.STOP_LIMIT and bo.stop_price == D("99.00") and bo.state == OrderState.ACKNOWLEDGED
    with pytest.raises(BrokerPayloadError, match="protective SELL"):
        map_order({**raw, "side": "buy"}, T0, GatewayKind.ALPACA_PAPER)
    with pytest.raises(BrokerPayloadError, match="unsupported broker order type"):
        map_order({**raw, "order_type": "stop", "type": "stop"}, T0, GatewayKind.ALPACA_PAPER)
    assert map_order(ORDER, T0, GatewayKind.ALPACA_PAPER).stop_price is None


class StopAwareHTTP(BrokerHTTP):
    """Like Alpaca, echo the submitted order type (and stop price) back in the order object."""

    def request(self, method: str, url: str, **kw: Any) -> Any:
        resp = super().request(method, url, **kw)
        if method == "POST" and resp.status_code == 200:
            data = kw["json"]
            order = self.orders[data["client_order_id"]]
            order["order_type"] = order["type"] = data["type"]
            if "stop_price" in data:
                order["stop_price"] = data["stop_price"]
            resp._content = __import__("json").dumps(order).encode()
        return resp


def _alpaca(h: Harness) -> tuple[AlpacaPaperGateway, BrokerHTTP, ExecutionService, Reconciler]:
    http = StopAwareHTTP()
    gw = AlpacaPaperGateway(paper_secret_store(), ledger=h.ledger, verifier=h.signer.verifier, clock=h.clock,
                            http_session=http, poll_interval_s=0.001)
    return gw, http, ExecutionService(h.ledger, gw, lambda _: [], h.clock, entries_lock=h.entries), Reconciler(
        h.db, h.ledger, gw, h.entries, h.clock)


async def test_alpaca_submits_an_exact_stop_limit_and_refuses_a_sell_the_broker_cannot_cover(h: Harness) -> None:
    gw, http, ex, rec = _alpaca(h)
    assert (await rec.startup_recovery()).status == ReconciliationStatus.CLEAN
    # the broker holds 4 shares (a filled buy); the ledger agrees after reconciliation
    http.orders["lat-paper-01jbuy"] = {**ORDER, "client_order_id": "lat-paper-01jbuy", "status": "filled",
                                       "filled_qty": "4", "qty": "4", "id": "b-1"}
    sell_too_big = h.factory.make(side=Side.SELL, qty="5", limit="98.50", order_type=OrderType.STOP_LIMIT, stop="99.00",
                                  gateway=GatewayKind.ALPACA_PAPER)
    assert await ex.submit(sell_too_big) is None
    assert h.ledger.get_record(sell_too_big.client_order_id).state == OrderState.REJECTED
    assert http.posts == []  # refused in preflight: a short sale never reaches the broker
    h.clock.advance(1)
    stop = h.factory.make(side=Side.SELL, qty="4", limit="98.50", order_type=OrderType.STOP_LIMIT, stop="99.00",
                          gateway=GatewayKind.ALPACA_PAPER)
    order = await ex.submit(stop)
    assert order is not None and order.order_type == OrderType.STOP_LIMIT and order.stop_price == D("99.00")
    post = http.posts[-1]
    assert post["type"] == "stop_limit" and post["side"] == "sell" and post["time_in_force"] == "day"
    assert post["qty"] == "4" and post["limit_price"] == "98.50" and post["stop_price"] == "99.00"
    assert not any(k.startswith("exact_") for k in post)
    await gw.aclose()


async def test_ledger_flags_a_broker_stop_price_that_differs_from_the_signed_one(h: Harness) -> None:
    gw, http, ex, rec = _alpaca(h)
    await rec.startup_recovery()
    http.orders["lat-paper-01jbuy"] = {**ORDER, "client_order_id": "lat-paper-01jbuy", "status": "filled",
                                       "filled_qty": "4", "qty": "4", "id": "b-1"}
    stop = h.factory.make(side=Side.SELL, qty="4", limit="98.50", order_type=OrderType.STOP_LIMIT, stop="99.00",
                          gateway=GatewayKind.ALPACA_PAPER)
    await ex.submit(stop)
    http.orders[stop.client_order_id]["stop_price"] = "90.00"
    found = await gw.get_order_by_client_id(stop.client_order_id)
    assert found is not None
    result = h.ledger.apply_broker_order(found, EventSource.RECONCILE)
    assert any("stop" in a for a in result.anomalies)
    await gw.aclose()
    assert h.clock.now() < T0 + timedelta(minutes=5)
