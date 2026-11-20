"""Trade prints (time & sales) from the market-data feed.

``packages/contracts`` defines ``Bar`` and ``Quote`` only; trades are consumed solely inside market data (feed
health, last-trade price, raw-event preservation), so the model is local, immutable and follows the same rules:
``Decimal`` price/size (floats rejected), timezone-aware UTC timestamps and an explicit ``available_at_ts``.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.contracts.enums import DataFeed
from packages.contracts.types import Dec, UtcDatetime


class Trade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.]{0,9}$")
    price: Dec
    size: Dec
    exchange: str | None = None
    trade_id: str | None = None
    conditions: tuple[str, ...] = ()
    tape: str | None = None
    exchange_ts: UtcDatetime
    provider_ts: UtcDatetime | None = None
    received_ts: UtcDatetime
    available_at_ts: UtcDatetime
    feed: DataFeed
    source_id: str
    is_synthetic: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.price <= 0:
            raise ValueError("non-positive trade price")
        if self.size < 0:
            raise ValueError("negative trade size")
        if self.available_at_ts < self.exchange_ts:
            raise ValueError("available_at_ts precedes exchange_ts")
        if self.is_synthetic != (self.feed == DataFeed.SYNTHETIC):
            raise ValueError("is_synthetic must be true exactly when feed is SYNTHETIC")
        return self
