"""Risk policy contract (spec §6).

A policy is immutable once approved. Its version id is the SHA-256 of its canonical JSON, so any
edit necessarily produces a new version. Models get read-only access to the serialized policy and
can never write it; edits go through the authenticated local settings flow with an audit record.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from .enums import OrderType
from .types import Dec

POLICY_SCHEMA = "risk-policy/1"

# Leveraged / inverse / volatility ETPs that are always excluded (spec: no leveraged/inverse ETFs).
DEFAULT_DENYLIST: tuple[str, ...] = (
    "TQQQ", "SQQQ", "SPXL", "SPXS", "SPXU", "UPRO", "SDS", "SSO", "QLD", "QID", "UDOW", "SDOW",
    "TNA", "TZA", "LABU", "LABD", "SOXL", "SOXS", "TECL", "TECS", "FAS", "FAZ", "NUGT", "DUST",
    "JNUG", "JDST", "UVXY", "SVXY", "VXX", "VIXY", "SH", "PSQ", "DOG", "RWM", "TMF", "TMV",
    "TSLL", "NVDL", "NVDU", "BOIL", "KOLD", "UCO", "SCO", "YINN", "YANG", "ERX", "ERY",
)

# Liquid large-cap stocks and standard (non-leveraged) ETFs eligible in the initial paper build.
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "V", "JNJ", "PG", "KO", "XOM",
)


class RiskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_schema: str = POLICY_SCHEMA
    name: str = "paper-default"
    long_only: bool = True
    allow_margin: bool = False
    allow_short: bool = False
    allow_extended_hours: bool = False
    regular_session_only: bool = True
    allowed_order_types: tuple[OrderType, ...] = (OrderType.LIMIT,)
    symbol_allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST
    symbol_denylist: tuple[str, ...] = DEFAULT_DENYLIST
    min_price: Dec = Decimal("5.00")
    max_positions: int = Field(default=1, ge=0)
    max_pending_orders: int = Field(default=1, ge=0)
    max_orders_per_minute: int = Field(default=3, ge=0)
    max_orders_per_day: int = Field(default=20, ge=0)
    max_order_notional: Dec = Decimal("2000.00")
    max_position_notional: Dec = Decimal("2500.00")
    max_concentration_pct: Dec = Field(default=Decimal("0.25"), description="position notional / equity")
    max_daily_loss: Dec = Field(default=Decimal("250.00"), description="realized + unrealized, positive number")
    max_planned_loss_per_trade: Dec = Decimal("50.00")
    max_spread_bps: Dec = Decimal("25")
    max_quote_age_seconds: Dec = Decimal("15")
    max_limit_deviation_bps: Dec = Field(default=Decimal("30"), description="limit vs mid, BUY above / SELL below")
    min_account_equity: Dec = Decimal("2000.00")
    min_buying_power_buffer: Dec = Field(default=Decimal("1.02"), description="notional * buffer <= cash BP")
    default_stop_loss_bps: Dec = Field(default=Decimal("150"), description="used when proposal has no stop")
    intent_ttl_seconds: int = Field(default=60, gt=0, le=600)
    duplicate_window_seconds: int = Field(default=300, gt=0)
    require_human_approval: bool = Field(default=False, description="always forced true in LIVE_APPROVAL")

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def version(self) -> str:
        return "pol_" + hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]
