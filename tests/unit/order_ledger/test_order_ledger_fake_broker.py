"""Deterministic FakeBroker (GatewayKind.FAKE, SIMULATED): validation, fills, expiry, cancel, failure injection."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import ExecutionLabel, GatewayKind, OrderEventType, OrderState, Side, TradingMode
from packages.contracts.interfaces import ExecutionGateway, GatewayRejected, SubmitTimeout
from packages.core.clock import ManualClock
from packages.core.signing import IntentSigner
from services.order_ledger.errors import BrokerUnavailable, RateLimited, SimulatedCrash, StreamDisconnected
from services.order_ledger.fake_broker import FakeBroker, QuoteBook
from tests.unit.order_ledger.ol_support import CLOSE_T0, T0, Harness, always_open, make_quote

E = OrderEventType
S = OrderState


def _types(h: Harness) -> list[OrderEventType]:
    return [e.event_type for e in h.broker.drain_events()]


def test_identity_and_protocol(h: Harness) -> None:
    assert isinstance(h.broker, ExecutionGateway)
    assert h.broker.kind == GatewayKind.FAKE and h.broker.is_paper and h.broker.label == ExecutionLabel.SIMULATED
    with pytest.raises(TypeError):
        FakeBroker(h.clock, starting_cash=100000.0)  # type: ignore[arg-type]


async def test_marketable_limit_fills_at_ask_not_limit(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.95", "99.97", T0))
    order = await h.broker.submit(h.factory.make(qty="10", limit="100.00", record=False))
    assert order.state == S.FILLED and order.avg_fill_price == Decimal("99.97000000")
    evs = h.broker.drain_events()
    assert [e.event_type for e in evs] == [E.ACCEPTED, E.NEW, E.FILL]
    fill = evs[-1]
    assert fill.broker_exec_id and fill.fill_quantity == Decimal(10) and fill.cum_filled_quantity == Decimal(10)
    acct = await h.broker.get_account()
    assert acct.cash == Decimal("100000") - Decimal("999.70")
    [pos] = await h.broker.get_positions()
    assert pos.quantity == Decimal(10) and pos.source == "FAKE_BROKER"


async def test_slippage_never_worse_than_limit(tmp_path: Path) -> None:
    from tests.unit.order_ledger.ol_support import make_harness

    h = make_harness(tmp_path, slippage_bps=Decimal("50"))
    try:
        h.quotes.set(make_quote("SPY", "99.95", "99.97", T0))
        order = await h.broker.submit(h.factory.make(qty="1", limit="100.10", record=False))
        assert order.avg_fill_price == Decimal("100.10000000")  # 99.97 * 1.005 = 100.47 capped at the limit
    finally:
        h.db.close()


async def test_resting_order_fills_on_later_quote_and_partials(tmp_path: Path) -> None:
    from tests.unit.order_ledger.ol_support import make_harness

    h = make_harness(tmp_path, max_fill_per_match=Decimal("4"))
    try:
        h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
        intent = h.factory.make(qty="10", limit="99.90", record=False)
        order = await h.broker.submit(intent)
        assert order.state == S.ACKNOWLEDGED
        h.clock.advance(1)
        h.broker.advance(make_quote("SPY", "99.80", "99.85", h.clock.now()))
        h.broker.advance(make_quote("SPY", "99.80", "99.88", h.clock.now()))
        h.broker.advance(make_quote("SPY", "99.80", "99.86", h.clock.now()))
        evs = h.broker.drain_events()
        fills = [e for e in evs if e.fill_quantity]
        assert [e.fill_quantity for e in fills] == [Decimal(4), Decimal(4), Decimal(2)]
        assert [e.cum_filled_quantity for e in fills] == [Decimal(4), Decimal(8), Decimal(10)]
        assert len({e.broker_exec_id for e in fills}) == 3
        snap = await h.broker.get_order_by_client_id(intent.client_order_id)
        assert snap is not None and snap.state == S.FILLED
    finally:
        h.db.close()


async def test_validation_rules(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.95", "99.97", T0))
    with pytest.raises(GatewayRejected, match="insufficient position"):
        await h.broker.submit(h.factory.make(side=Side.SELL, qty="1", record=False))
    with pytest.raises(GatewayRejected, match="buying power"):
        await h.broker.submit(h.factory.make(qty="2000", limit="100.00", record=False))
    with pytest.raises(GatewayRejected, match="tick"):
        await h.broker.submit(h.factory.make(qty="1", limit="100.001", record=False))
    with pytest.raises(GatewayRejected, match="whole-share"):
        await h.broker.submit(h.factory.make(qty="1.5", record=False))
    with pytest.raises(GatewayRejected, match="FAKE-gateway"):
        await h.broker.submit(h.factory.make(record=False, gateway=GatewayKind.LEAN_PAPER_SIM))
    forged = h.factory.make(record=False).model_copy(update={"quantity": Decimal("11")})
    with pytest.raises(GatewayRejected, match="signature"):
        await h.broker.submit(forged)
    ok = h.factory.make(qty="1", record=False)
    await h.broker.submit(ok)
    with pytest.raises(GatewayRejected, match="unique"):
        await h.broker.submit(ok)
    late = h.factory.make(record=False, ttl_s=5)
    h.clock.advance(6)
    with pytest.raises(GatewayRejected, match="expired"):
        await h.broker.submit(late)
    assert h.broker.accepted_submissions == [ok.client_order_id]


async def test_regular_session_only(tmp_path: Path) -> None:
    clock = ManualClock(CLOSE_T0 + timedelta(hours=1))  # after the close
    signer = IntentSigner.generate()
    broker = FakeBroker(clock, QuoteBook())  # real XNYS calendar
    from tests.unit.order_ledger.ol_support import IntentFactory

    db_h = __import__("tests.unit.order_ledger.ol_support", fromlist=["make_harness"]).make_harness(tmp_path)
    try:
        factory = IntentFactory(db_h.ledger, signer, clock, mode=TradingMode.PAPER)
        with pytest.raises(GatewayRejected, match="regular session"):
            await broker.submit(factory.make(record=False))
        clock.set(T0)  # Wednesday 10:00 New York
        await broker.submit(factory.make(record=False))
    finally:
        db_h.db.close()


async def test_sell_reduces_position_and_is_long_only(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.95", "99.97", T0))
    await h.broker.submit(h.factory.make(qty="5", record=False))
    await h.broker.submit(h.factory.make(side=Side.SELL, qty="3", limit="99.90", record=False))
    [pos] = await h.broker.get_positions()
    assert pos.quantity == Decimal(2)
    with pytest.raises(GatewayRejected, match="long-only"):
        await h.broker.submit(h.factory.make(side=Side.SELL, qty="3", limit="99.90", record=False))


async def test_day_orders_expire_at_close_and_positions_carry(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "100.00", "100.02", T0))
    await h.broker.submit(h.factory.make(qty="5", limit="100.02", record=False))  # fills
    resting = h.factory.make(qty="5", limit="99.00", record=False)
    await h.broker.submit(resting)
    h.broker.drain_events()
    h.clock.set(CLOSE_T0 + timedelta(minutes=1))
    assert h.broker.expire_day_orders() == [resting.client_order_id]
    assert _types(h) == [E.EXPIRED]
    h.clock.set(T0 + timedelta(days=1))
    [pos] = await h.broker.get_positions()
    assert pos.quantity == Decimal(5)
    assert (await h.broker.list_orders(status="open")) == []


async def test_cancel_and_cancel_fill_race(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "100.00", "100.05", T0))
    a = h.factory.make(qty="3", limit="99.00", record=False)
    await h.broker.submit(a)
    h.broker.drain_events()
    await h.broker.cancel(a.client_order_id)
    assert _types(h) == [E.PENDING_CANCEL, E.CANCELED]
    with pytest.raises(GatewayRejected, match="not cancelable"):
        await h.broker.cancel(a.client_order_id)
    b = h.factory.make(qty="3", limit="99.00", record=False)
    await h.broker.submit(b)
    h.broker.drain_events()
    h.broker.injector.fill_on_cancel = 1
    await h.broker.cancel(b.client_order_id)
    assert _types(h) == [E.PENDING_CANCEL, E.FILL]
    snap = await h.broker.get_order_by_client_id(b.client_order_id)
    assert snap is not None and snap.state == S.FILLED


async def test_failure_injection_hooks(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    inj = h.broker.injector
    inj.reject_next = 1
    with pytest.raises(GatewayRejected):
        await h.broker.submit(h.factory.make(qty="1", record=False))
    inj.rate_limit_next = 1
    with pytest.raises(RateLimited):
        await h.broker.submit(h.factory.make(qty="1", record=False))
    assert isinstance(RateLimited("x"), GatewayRejected)
    inj.partition_before_accept = 1
    lost = h.factory.make(qty="1", record=False)
    with pytest.raises(ConnectionError):
        await h.broker.submit(lost)
    assert await h.broker.get_order_by_client_id(lost.client_order_id) is None
    inj.timeout_after_accept = 1
    accepted = h.factory.make(qty="1", record=False)
    with pytest.raises(SubmitTimeout):
        await h.broker.submit(accepted)
    assert await h.broker.get_order_by_client_id(accepted.client_order_id) is not None
    inj.crash_after_accept = 1
    crashed = h.factory.make(qty="1", record=False)
    with pytest.raises(SimulatedCrash):
        await h.broker.submit(crashed)
    assert not isinstance(SimulatedCrash("x"), Exception)
    assert crashed.client_order_id in h.broker.accepted_submissions
    inj.fail_reads = 1
    with pytest.raises(BrokerUnavailable):
        await h.broker.get_account()
    inj.fail_lookups = 1
    with pytest.raises(BrokerUnavailable):
        await h.broker.get_order_by_client_id(crashed.client_order_id)


async def test_event_drop_duplicate_and_reorder(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    inj = h.broker.injector
    inj.duplicate_next_events = 1
    await h.broker.submit(h.factory.make(qty="1", record=False))
    evs = h.broker.drain_events()
    assert [e.event_type for e in evs] == [E.ACCEPTED, E.ACCEPTED, E.NEW, E.FILL]
    assert evs[0] is evs[1]
    inj.drop_event_types = frozenset({E.FILL})
    await h.broker.submit(h.factory.make(qty="1", record=False))
    assert [e.event_type for e in h.broker.drain_events()] == [E.ACCEPTED, E.NEW]
    assert [e.event_type for e in h.broker.dropped_events] == [E.FILL]
    inj.drop_event_types = frozenset()
    inj.reorder_window = 3
    await h.broker.submit(h.factory.make(qty="1", record=False))
    assert [e.event_type for e in h.broker.drain_events()] == [E.FILL, E.NEW, E.ACCEPTED]


async def test_stream_delivers_live_events_and_disconnects(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "99.98", "100.00", T0))
    stream = h.broker.order_events()
    await h.broker.submit(h.factory.make(qty="1", record=False))
    got = [await asyncio.wait_for(stream.__anext__(), 1) for _ in range(3)]
    assert [e.event_type for e in got] == [E.ACCEPTED, E.NEW, E.FILL]
    h.broker.disconnect_stream()
    with pytest.raises(StreamDisconnected):
        await asyncio.wait_for(stream.__anext__(), 1)
    h.broker.reconnect_stream()
    fresh = h.broker.order_events()  # a new subscription does not replay history (like a websocket)
    await h.broker.submit(h.factory.make(qty="1", record=False))
    nxt = await asyncio.wait_for(fresh.__anext__(), 1)
    assert nxt.event_type == E.ACCEPTED and nxt.client_order_id != got[0].client_order_id


async def test_stale_or_future_quotes_never_fill(h: Harness) -> None:
    h.quotes.set(make_quote("SPY", "100.00", "100.02", T0 + timedelta(seconds=30)))  # not yet available
    order = await h.broker.submit(h.factory.make(qty="1", limit="100.05", record=False))
    assert order.state == S.ACKNOWLEDGED
    h.quotes.set(make_quote("SPY", "100.00", "100.02", T0 - timedelta(minutes=10)))  # stale
    order2 = await h.broker.submit(h.factory.make(qty="1", limit="100.05", record=False))
    assert order2.state == S.ACKNOWLEDGED
    # quotes pushed through advance() are held to the same rule (no look-ahead fill on a future quote)
    h.broker.advance(make_quote("SPY", "99.90", "99.95", T0 + timedelta(seconds=5)))
    h.broker.advance(make_quote("SPY", "99.90", "99.95", T0 - timedelta(minutes=5)))
    assert all(o.filled == 0 for o in h.broker.orders.values())
    h.clock.advance(5)  # the first pushed quote is now available: it fills both resting orders
    h.broker.advance()
    assert all(o.status == "filled" for o in h.broker.orders.values())


async def test_state_persists_across_instances(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    quotes = QuoteBook()
    quotes.set(make_quote("SPY", "100.00", "100.02", T0))
    path = tmp_path / "fb.json"
    b1 = FakeBroker(clock, quotes, session_bounds_fn=always_open, state_path=path)
    from tests.unit.order_ledger.ol_support import make_harness

    hh = make_harness(tmp_path / "h")
    try:
        intent = hh.factory.make(qty="2", limit="100.05", record=False)
        await b1.submit(intent)
        b2 = FakeBroker(clock, quotes, session_bounds_fn=always_open, state_path=path)
        snap = await b2.get_order_by_client_id(intent.client_order_id)
        assert snap is not None and snap.state == S.FILLED
        assert (await b2.get_positions())[0].quantity == Decimal(2)
        assert (await b2.get_account()).cash == (await b1.get_account()).cash
    finally:
        hh.db.close()


async def test_empty_account_state(h: Harness) -> None:
    acct = await h.broker.get_account()
    assert acct.equity == acct.cash == acct.buying_power == Decimal("100000")
    assert acct.is_paper and acct.account_id_masked.startswith("FAKE-")
    assert await h.broker.get_positions() == []
    assert await h.broker.list_orders() == []
