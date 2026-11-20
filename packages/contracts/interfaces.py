"""Service boundaries as Protocols. Implementations live in services/*; the pipeline depends only on these.

Invariant: ``ExecutionGateway.submit`` is the *only* order-creation path and accepts only a signed
``ApprovedOrderIntent``. ``BrokerStateReader`` is read-only (account/positions/orders) and is what
reconciliation and alpaca-py based connectivity checks use.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from .enums import FeedMode, GatewayKind
from .models import (
    AccountSnapshot,
    ApprovedOrderIntent,
    Bar,
    BrokerOrder,
    OrderEvent,
    PositionSnapshot,
    Quote,
)


@runtime_checkable
class MarketDataProvider(Protocol):
    feed_mode: FeedMode
    source_id: str

    async def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> list[Bar]: ...

    async def get_latest_quote(self, symbol: str) -> Quote | None: ...

    def stream_quotes(self, symbols: Sequence[str]) -> AsyncIterator[Quote]: ...


@runtime_checkable
class BrokerStateReader(Protocol):
    """Read-only broker state. Must never expose an order-creation method."""

    kind: GatewayKind
    is_paper: bool

    async def get_account(self) -> AccountSnapshot: ...

    async def get_positions(self) -> list[PositionSnapshot]: ...

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None: ...

    async def list_orders(self, *, status: str = "all", after: datetime | None = None) -> list[BrokerOrder]: ...


class SubmitTimeout(Exception):
    """Submission outcome unknown. Caller MUST mark the order UNKNOWN and query; never blindly retry."""


class GatewayRejected(Exception):
    """Gateway refused the intent before or at the broker (unsupported feature, bad signature, ...)."""


@runtime_checkable
class ExecutionGateway(BrokerStateReader, Protocol):
    async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder:
        """Submit a verified intent. Raises SubmitTimeout (outcome unknown) or GatewayRejected."""
        ...

    async def cancel(self, client_order_id: str) -> None: ...

    def order_events(self) -> AsyncIterator[OrderEvent]:
        """Stream of broker order updates (trade_updates / LEAN OnOrderEvent)."""
        ...
