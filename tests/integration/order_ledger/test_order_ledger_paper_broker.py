"""REAL Alpaca PAPER endpoint, read-only (``@pytest.mark.paper_broker``).

Runs only when Alpaca PAPER credentials exist (macOS Keychain service 'local-ai-trader', the environment, or the
git-ignored .env); otherwise it is skipped. When it runs it talks to https://paper-api.alpaca.markets with GET
requests only (the reader's session guard refuses anything else before a socket is opened) and never places,
cancels or modifies an order. A pass here means: it ran against the real Alpaca PAPER broker.
"""

from __future__ import annotations

import pytest

from packages.contracts.enums import GatewayKind
from packages.core.config import PROJECT_ROOT
from packages.core.secrets import SecretStore
from services.order_ledger.alpaca_reader import AlpacaPaperStateReader

pytestmark = pytest.mark.paper_broker


@pytest.fixture
def reader() -> AlpacaPaperStateReader:
    store = SecretStore(PROJECT_ROOT)
    if not store.alpaca_paper_available():
        pytest.skip("Alpaca PAPER credentials absent (Keychain/env/.env): real paper-broker read test not run")
    return AlpacaPaperStateReader(store, max_attempts=3)


async def test_real_paper_account_positions_and_orders_read_only(reader: AlpacaPaperStateReader) -> None:
    """REAL Alpaca PAPER broker (read-only): account/positions/orders map to Decimal contracts; no non-GET request."""
    account = await reader.get_account()
    assert account.is_paper and account.source == "ALPACA_PAPER"
    assert "****" in account.account_id_masked
    positions = await reader.get_positions()
    assert all(p.source == "BROKER" for p in positions)
    orders = await reader.list_orders(status="all")
    assert all(o.gateway == GatewayKind.LEAN_ALPACA_PAPER for o in orders)
    assert await reader.get_order_by_client_id("lat-paper-does-not-exist-0000") is None
    assert reader.refused_requests == []
