"""AlpacaPaperStateReader with the alpaca-py HTTP layer replaced by an in-memory fake session (no network).

The fake session receives exactly what alpaca-py's ``TradingClient`` would send over the wire; credentials are
obviously fake test strings (``PK``-prefixed placeholders), never real keys.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
import requests

from packages.contracts.enums import GatewayKind, OrderState, OrderType, Side
from packages.contracts.interfaces import BrokerStateReader
from packages.core.secrets import SecretStore
from services.order_ledger.alpaca_reader import (
    PAPER_BASE_URL,
    AlpacaPaperStateReader,
    map_account,
    map_order,
    map_position,
    mask_account,
)
from services.order_ledger.errors import (
    BrokerAuthError,
    BrokerPayloadError,
    BrokerRateLimited,
    BrokerUnavailable,
    LiveKeyRefused,
    MissingPaperCredentials,
    ReadOnlyViolation,
)
from tests.unit.order_ledger.ol_support import (
    ACCOUNT,
    FAKE_KEY,
    FAKE_SECRET,
    ORDER,
    POSITION,
    T0,
    FakeSession,
    alpaca_routes,
    paper_secret_store,
)


def _store(key: str = FAKE_KEY, secret: str = FAKE_SECRET) -> SecretStore:
    return paper_secret_store(key, secret)


def _reader(session: FakeSession, **kw: Any) -> AlpacaPaperStateReader:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    reader = AlpacaPaperStateReader(_store(), http_session=session, sleep=fake_sleep, **kw)
    reader.sleeps = sleeps  # type: ignore[attr-defined]
    return reader


def _default_routes() -> dict[str, list[tuple[int, Any]]]:
    return alpaca_routes()


async def test_reads_map_to_decimal_contracts() -> None:
    session = FakeSession(_default_routes())
    reader = _reader(session)
    assert isinstance(reader, BrokerStateReader)
    acct = await reader.get_account()
    assert acct.is_paper and acct.source == "ALPACA_PAPER"
    assert acct.equity == Decimal("100250.5") and acct.multiplier == Decimal(2)
    assert acct.account_id_masked == "PA****CT42" and "TESTACCT" not in acct.account_id_masked
    [pos] = await reader.get_positions()
    assert pos.quantity == Decimal(10) and pos.cost_basis == Decimal("5001.20") and pos.source == "BROKER"
    order = await reader.get_order_by_client_id("lat-paper-01jtest")
    assert order is not None
    assert order.state == OrderState.PARTIALLY_FILLED and order.filled_quantity == Decimal(4)
    assert order.avg_fill_price == Decimal("500.05") and order.gateway == GatewayKind.LEAN_ALPACA_PAPER
    assert order.updated_at.microsecond == 500000
    [listed] = await reader.list_orders(status="open")
    assert listed.client_order_id == order.client_order_id
    # every request was a GET to the paper endpoint
    assert {m for m, _, _ in session.calls} == {"GET"}
    assert all(u.startswith(PAPER_BASE_URL + "/v2/") for _, u, _ in session.calls)
    lookup = next(c for c in session.calls if c[1].endswith("by_client_order_id"))
    assert lookup[2]["params"] == {"client_order_id": "lat-paper-01jtest"}
    assert lookup[2]["allow_redirects"] is False


async def test_numbers_as_json_floats_are_converted_exactly() -> None:
    routes = _default_routes()
    routes["/account"] = [(200, {**ACCOUNT, "equity": 100250.55, "cash": 1e5})]
    acct = await _reader(FakeSession(routes)).get_account()
    assert acct.equity == Decimal("100250.55") and acct.cash == Decimal("100000.0")


async def test_lookup_404_is_none() -> None:
    routes = _default_routes()
    routes["/orders:by_client_order_id"] = [(404, {"code": 40410000, "message": "order not found"})]
    assert await _reader(FakeSession(routes)).get_order_by_client_id("lat-paper-missing") is None


async def test_rate_limit_backoff_is_bounded() -> None:
    routes = _default_routes()
    routes["/account"] = [(429, {"code": 42910000, "message": "rate limit exceeded"})] * 2 + [(200, ACCOUNT)]
    reader = _reader(FakeSession(routes), backoff_s=(0.1, 0.2))
    acct = await reader.get_account()
    assert acct.status == "ACTIVE"
    assert reader.rate_limited_retries == 2 and reader.sleeps == [0.1, 0.2]  # type: ignore[attr-defined]
    routes["/positions"] = [(429, {"code": 42910000, "message": "rate limit exceeded"})]
    stubborn = _reader(FakeSession(routes), max_attempts=3, backoff_s=(0.01,))
    with pytest.raises(BrokerRateLimited, match="429 after 3 attempts"):
        await stubborn.get_positions()
    assert stubborn.sleeps == [0.01, 0.01]  # type: ignore[attr-defined]


@pytest.mark.parametrize(("status", "exc"), [(401, BrokerAuthError), (403, BrokerAuthError), (500, BrokerUnavailable)])
async def test_http_errors_fail_closed_without_leaking_credentials(status: int, exc: type[Exception]) -> None:
    routes = _default_routes()
    routes["/account"] = [(status, {"code": status, "message": "nope"})]
    with pytest.raises(exc) as info:
        await _reader(FakeSession(routes)).get_account()
    text = str(info.value) + repr(info.value)
    assert FAKE_KEY not in text and FAKE_SECRET not in text


async def test_network_error_is_unavailable() -> None:
    class Down(FakeSession):
        def request(self, method: str, url: str, **kw: Any) -> requests.Response:
            raise requests.exceptions.ConnectionError("connection refused")

    with pytest.raises(BrokerUnavailable):
        await _reader(Down({})).get_account()


def test_credential_rules() -> None:
    with pytest.raises(MissingPaperCredentials):
        AlpacaPaperStateReader(_store(key=""))
    with pytest.raises(LiveKeyRefused, match="LIVE"):
        AlpacaPaperStateReader(_store(key="AKLIVELOOKING00000000"))
    with pytest.raises(LiveKeyRefused, match="PAPER"):
        AlpacaPaperStateReader(_store(key="XXUNKNOWN0000000000"))
    reader = AlpacaPaperStateReader(_store())
    assert FAKE_KEY not in repr(reader) and FAKE_SECRET not in repr(reader)


def test_reader_exposes_no_order_creation_or_cancel_surface() -> None:
    forbidden = ("submit", "place", "create", "replace", "cancel", "close_", "close_position", "post", "delete")
    for name in dir(AlpacaPaperStateReader):
        assert not any(word in name.lower() for word in forbidden), name
    reader = _reader(FakeSession(_default_routes()))
    for name in vars(reader):
        assert not any(word in name.lower() for word in forbidden), name


async def test_sdk_order_methods_are_blocked_before_the_network() -> None:
    """Even reaching into the SDK client (``submit_order``/``cancel_orders``) cannot send anything."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    session = FakeSession(_default_routes())
    reader = _reader(session)
    client = reader._api._client
    req = LimitOrderRequest(symbol="SPY", qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, limit_price=1.0)
    with pytest.raises(ReadOnlyViolation):
        client.submit_order(req)
    with pytest.raises(ReadOnlyViolation):
        client.cancel_orders()
    assert session.calls == []
    assert reader.refused_requests and all("?" not in r for r in reader.refused_requests)


async def test_paper_credentials_can_never_reach_the_live_endpoint() -> None:
    from alpaca.common.enums import BaseURL

    session = FakeSession(_default_routes())
    reader = _reader(session)
    reader._api._client._base_url = BaseURL.TRADING_LIVE
    with pytest.raises(ReadOnlyViolation, match="non-paper"):
        await reader.get_account()
    assert session.calls == []


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"status": "replaced"}, "status"),
        ({"order_type": "stop", "type": "stop"}, "order type"),
        ({"time_in_force": "gtc"}, "time in force"),
        ({"qty": None, "notional": "100"}, "notional"),
        ({"symbol": "BTC/USD"}, "symbol"),
        ({"filled_qty": "abc"}, "decimal"),
        ({"updated_at": "yesterday", "submitted_at": None}, "timestamp"),
        ({"side": "sell_short"}, "side"),
    ],
)
def test_malformed_or_unsupported_orders_are_refused(patch: dict[str, Any], match: str) -> None:
    with pytest.raises(BrokerPayloadError, match=match):
        map_order({**ORDER, **patch}, T0)


def test_mapping_edge_cases() -> None:
    with pytest.raises(BrokerPayloadError):
        map_account({**ACCOUNT, "equity": None}, T0)
    with pytest.raises(BrokerPayloadError):
        map_account(["not", "an", "object"], T0)
    with pytest.raises(BrokerPayloadError, match=r"only U\.S\. equities"):
        map_position({**POSITION, "asset_class": "crypto"}, T0)
    short = map_position({**POSITION, "side": "short"}, T0)
    assert short.quantity == Decimal(-10)
    done = map_order({**ORDER, "status": "done_for_day"}, T0)
    assert done.state == OrderState.PARTIALLY_FILLED
    new = map_order({**ORDER, "status": "new", "filled_qty": "0", "filled_avg_price": None}, T0)
    assert new.state == OrderState.ACKNOWLEDGED and new.avg_fill_price is None
    sell = map_order({**ORDER, "side": "sell"}, T0)
    assert sell.side == Side.SELL and sell.order_type == OrderType.LIMIT
    blocked = map_account({k: v for k, v in ACCOUNT.items() if k != "trading_blocked"}, T0)
    assert blocked.trading_blocked is True  # missing flag => treated as blocked (fail closed)
    assert mask_account("PA3TESTACCT42") == "PA****CT42"
    assert mask_account("1234") == "****1234"


async def test_malformed_list_payload_is_refused() -> None:
    routes = _default_routes()
    routes["/positions"] = [(200, {"unexpected": "object"})]
    with pytest.raises(BrokerPayloadError):
        await _reader(FakeSession(routes)).get_positions()
    with pytest.raises(ValueError, match="status filter"):
        await _reader(FakeSession(_default_routes())).list_orders(status="weird")
