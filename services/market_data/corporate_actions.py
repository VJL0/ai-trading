"""Point-in-time corporate action records (splits, dividends, renames, delistings).

Not part of ``packages/contracts`` (no other service consumes them yet); kept local and immutable.
``available_at_ts`` is the earliest decision time allowed to use the record. When the announcement date is
unknown (LEAN factor/map files) it is conservatively the ex-date session open: the latest instant at which the
action was certainly public.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.contracts.enums import DataFeed
from packages.contracts.types import Dec, UtcDatetime


class CorporateActionType(StrEnum):
    SPLIT = "SPLIT"
    DIVIDEND = "DIVIDEND"
    RENAME = "RENAME"
    DELISTING = "DELISTING"


class CorporateAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.]{0,9}$")
    action_type: CorporateActionType
    ex_date: date
    available_at_ts: UtcDatetime
    split_factor: Dec | None = Field(default=None, description="price multiplier (0.25 = 4-for-1 forward split)")
    cash_amount: Dec | None = Field(default=None, description="dividend per share, USD")
    reference_price: Dec | None = Field(default=None, description="raw close before the action took effect")
    old_symbol: str | None = None
    new_symbol: str | None = None
    feed: DataFeed
    source_id: str
    raw: str | None = Field(default=None, description="original source row(s), verbatim")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.action_type == CorporateActionType.SPLIT and (self.split_factor is None or self.split_factor <= 0):
            raise ValueError("SPLIT requires a positive split_factor")
        if self.action_type == CorporateActionType.DIVIDEND and (self.cash_amount is None or self.cash_amount <= 0):
            raise ValueError("DIVIDEND requires a positive cash_amount")
        if self.action_type == CorporateActionType.RENAME and not (self.old_symbol and self.new_symbol):
            raise ValueError("RENAME requires old_symbol and new_symbol")
        return self
