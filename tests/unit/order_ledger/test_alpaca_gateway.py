"""Real alpaca-py serialization + execution/ledger integration, no network or credentials."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
import requests

from packages.contracts.enums import GatewayKind, OrderState, ReconciliationStatus
from packages.contracts.interfaces import GatewayRejected
from services.order_ledger.alpaca_gateway import AlpacaPaperGateway, _PaperOrderSession
from services.order_ledger.errors import ReadOnlyViolation
from services.order_ledger.execution import ExecutionService
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import ACCOUNT, ORDER, POSITION, Harness, _response, paper_secret_store


class BrokerHTTP:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.posts: list[dict[str, Any]] = []
        self.deletes = 0
        self.fail_read = False
        self.lose_response = False
        self.reject = False
        self.cancel_timeout = False

    def close(self) -> None:
        pass

    def request(self, method: str, url: str, **kw: Any) -> requests.Response:
        assert url.startswith('https://paper-api.alpaca.markets/v2/')
        assert kw['allow_redirects'] is False
        if method == 'POST':
            data = kw['json']
            self.posts.append(data)
            if self.reject:
                return _response(422, {'code': 42210000, 'message': 'rejected'}, url)
            self.orders[data['client_order_id']] = {
                **ORDER, **data, 'status': 'new', 'filled_qty': '0', 'filled_avg_price': None}
            if self.lose_response:
                raise requests.ConnectionError('lost after acceptance')
            return _response(200, self.orders[data['client_order_id']], url)
        if method == 'DELETE':
            self.deletes += 1
            if self.cancel_timeout:
                raise requests.Timeout('uncertain cancel')
            for order in self.orders.values():
                if url.endswith(order['id']):
                    order['status'] = 'canceled'
            return _response(200, None, url)
        if self.fail_read:
            raise requests.ConnectionError('offline')
        if url.endswith('/account'):
            return _response(200, ACCOUNT, url)
        if url.endswith('/positions'):
            qty = sum(Decimal(o['filled_qty']) for o in self.orders.values())
            return _response(200, [{**POSITION, 'qty': str(qty), 'avg_entry_price': '100',
                                    'cost_basis': str(qty * 100)}] if qty else [], url)
        if url.endswith('/orders'):
            orders = list(self.orders.values())
            if kw['params'].get('status') == 'open':
                orders = [o for o in orders if o['status'] not in ('filled', 'canceled')]
            return _response(200, orders, url)
        coid = kw['params']['client_order_id']
        return _response(200, self.orders[coid], url) if coid in self.orders else _response(
            404, {'code': 40410000, 'message': 'not found'}, url)


def setup(h: Harness) -> tuple[AlpacaPaperGateway, BrokerHTTP, ExecutionService, Reconciler]:
    http = BrokerHTTP()
    gateway = AlpacaPaperGateway(paper_secret_store(), ledger=h.ledger, verifier=h.signer.verifier,
                                clock=h.clock, http_session=http, poll_interval_s=0.001)
    execution = ExecutionService(h.ledger, gateway, lambda _: [], h.clock, entries_lock=h.entries)
    reconciler = Reconciler(h.db, h.ledger, gateway, h.entries, h.clock)
    return gateway, http, execution, reconciler


async def test_sdk_submit_partial_fill_cancel_restart_and_audit(h: Harness) -> None:
    gateway, http, execution, reconciler = setup(h)
    assert (await reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    order = await execution.submit(intent)
    assert order and order.state == OrderState.ACKNOWLEDGED
    assert http.posts[0]['qty'] == '10' and http.posts[0]['limit_price'] == '100.00'
    assert http.posts[0]['client_order_id'] == intent.client_order_id
    assert 'exact_price' not in http.posts[0]
    assert await execution.submit(intent) is None
    assert len(http.posts) == 1
    http.orders[intent.client_order_id].update(status='partially_filled', filled_qty='4', filled_avg_price='100')
    stream = gateway.order_events()
    await anext(stream)
    await stream.aclose()
    assert h.ledger.position_quantities() == {'SPY': Decimal(4)}
    assert await execution.cancel(intent.client_order_id) == OrderState.CANCELED
    assert http.deletes == 1
    assert (await reconciler.reconcile()).status == ReconciliationStatus.CLEAN
    # Repeated snapshots and a restart do not duplicate partial fills.
    assert (await reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
    assert h.ledger.position_quantities() == {'SPY': Decimal(4)}
    from packages.core.audit import verify_chain
    assert verify_chain(h.db)[0]


@pytest.mark.parametrize('lost', [False, True])
async def test_unknown_acceptance_recovers_without_resubmission(h: Harness, lost: bool) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    http.lose_response = lost
    if lost:
        assert await execution.submit(intent)
    else:
        # Process died after durable claim and remote accept, before recording response.
        h.ledger.claim_for_submission(intent.client_order_id, 'crashed')
        http.orders[intent.client_order_id] = {**ORDER, 'client_order_id': intent.client_order_id,
            'status': 'filled', 'limit_price': '100.00', 'filled_qty': '10', 'filled_avg_price': '100'}
        assert (await reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
        assert h.ledger.position_quantities() == {'SPY': Decimal(10)}
    before = len(http.posts)
    assert await execution.submit(intent) is None
    assert len(http.posts) == before


@pytest.mark.parametrize('failure', ['read', 'reject', 'cancel'])
async def test_uncertain_state_fails_closed(h: Harness, failure: str) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    if failure == 'cancel':
        await execution.submit(intent)
        http.cancel_timeout = True
        await execution.cancel(intent.client_order_id)
    else:
        http.fail_read = failure == 'read'
        http.reject = failure == 'reject'
        assert await execution.submit(intent) is None
        assert h.ledger.get_record(intent.client_order_id).state == OrderState.UNKNOWN
    assert h.entries.is_locked()[0]
    assert len(http.posts) == (0 if failure == 'read' else 1)


async def test_unclaimed_and_tampered_intents_never_reach_sdk(h: Harness) -> None:
    gateway, http, _, _ = setup(h)
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    with pytest.raises(GatewayRejected, match='claimed'):
        await gateway.submit(intent)
    h.ledger.claim_for_submission(intent.client_order_id, 'test')
    with pytest.raises(GatewayRejected, match='signature'):
        await gateway.submit(intent.model_copy(update={'quantity': Decimal(99)}))
    assert not http.posts


@pytest.mark.parametrize(('method', 'url'), [
    ('POST', 'https://api.alpaca.markets/v2/orders'),
    ('POST', 'https://paper-api.alpaca.markets.evil/v2/orders'),
    ('DELETE', 'https://paper-api.alpaca.markets/v2/positions'),
    ('POST', 'https://paper-api.alpaca.markets/v2/orders?x=1'),
])
def test_endpoint_guard(method: str, url: str) -> None:
    with pytest.raises(ReadOnlyViolation):
        _PaperOrderSession(BrokerHTTP()).request(method, url)


async def test_disconnect_raises_for_execution_reconnect(h: Harness) -> None:
    gateway, http, _, _ = setup(h)
    http.fail_read = True
    with pytest.raises(Exception, match='ConnectionError'):
        await asyncio.wait_for(anext(gateway.order_events()), 1)


async def test_existing_client_id_is_verified_and_never_posted(h: Harness) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    http.orders[intent.client_order_id] = {**ORDER, 'client_order_id': intent.client_order_id,
        'status': 'new', 'limit_price': '100.00', 'filled_qty': '0', 'filled_avg_price': None}
    assert await execution.submit(intent)
    assert not http.posts


async def test_client_id_collision_locks_instead_of_accepting_wrong_order(h: Harness) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    http.orders[intent.client_order_id] = {**ORDER, 'client_order_id': intent.client_order_id, 'symbol': 'QQQ'}
    await execution.submit(intent)
    assert h.entries.is_locked()[0]
    assert not http.posts
    assert not h.ledger.fills(intent.client_order_id)


async def test_simultaneous_submission_has_one_durable_claim(h: Harness) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    results = await asyncio.gather(execution.submit(intent), execution.submit(intent))
    assert sum(r is not None for r in results) == 1
    assert len(http.posts) == 1


async def test_cancel_fill_race_books_fill_and_does_not_delete_terminal(h: Harness) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    await execution.submit(intent)
    http.orders[intent.client_order_id].update(status='filled', filled_qty='10', filled_avg_price='100')
    assert await execution.cancel(intent.client_order_id) == OrderState.FILLED
    assert http.deletes == 0
    assert h.ledger.position_quantities() == {'SPY': Decimal(10)}


async def test_reconnect_requires_clean_reconciliation(h: Harness) -> None:
    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    execution.reconnect_backoff_s = (0.01,)
    execution.on_reconnect = reconciler.reconcile
    stop = asyncio.Event()
    http.fail_read = True
    consumer = asyncio.create_task(execution.run_event_consumer(stop))
    try:
        for _ in range(100):
            if execution.disconnects:
                break
            await asyncio.sleep(0.005)
        assert execution.disconnects > 0 and h.entries.is_locked()[0]
        http.fail_read = False
        for _ in range(100):
            if not h.entries.is_locked()[0]:
                break
            await asyncio.sleep(0.005)
        assert not h.entries.is_locked()[0]
        assert not http.posts
    finally:
        stop.set()
        await asyncio.wait_for(consumer, 1)


async def test_expiry_during_preflight_refuses_post(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    original = gateway.get_order_by_client_id

    async def slow_lookup(coid: str) -> Any:
        found = await original(coid)
        h.clock.set(intent.expires_at)
        return found

    monkeypatch.setattr(gateway, 'get_order_by_client_id', slow_lookup)
    assert await execution.submit(intent) is None
    assert not http.posts
    assert h.ledger.get_record(intent.client_order_id).state == OrderState.REJECTED


async def test_timeout_before_delayed_acceptance_never_reposts(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    _gateway, http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    release = threading.Event()
    finished = threading.Event()
    original = http.request

    def delayed(method: str, url: str, **kw: Any) -> requests.Response:
        if method == 'POST':
            assert release.wait(2)
        response = original(method, url, **kw)
        if method == 'POST':
            finished.set()
        return response

    monkeypatch.setattr(http, 'request', delayed)
    execution.submit_timeout_s = 0.05
    try:
        assert await execution.submit(intent) is None
        assert h.ledger.get_record(intent.client_order_id).state == OrderState.UNKNOWN
        assert h.entries.is_locked()[0]
        assert await execution.submit(intent) is None
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
    assert (await reconciler.reconcile()).status == ReconciliationStatus.CLEAN
    assert len(http.posts) == 1
    assert h.ledger.get_record(intent.client_order_id).state == OrderState.ACKNOWLEDGED


async def test_interrupted_cancel_locks_entries(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway, _http, execution, reconciler = setup(h)
    await reconciler.startup_recovery()
    intent = h.factory.make(gateway=GatewayKind.ALPACA_PAPER)
    await execution.submit(intent)

    async def interrupted(_coid: str) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(gateway, 'cancel', interrupted)
    with pytest.raises(asyncio.CancelledError):
        await execution.cancel(intent.client_order_id)
    assert h.entries.is_locked()[0]
    assert h.ledger.get_record(intent.client_order_id).state == OrderState.CANCEL_PENDING
