"""Shared constants and helpers for the market-data service (labels, timeframes, Decimal handling)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from functools import lru_cache
from typing import Any

from packages.contracts.enums import DataFeed, FeedMode
from packages.core.money import D, from_external_float

PENDING_DATASET_VERSION = "ds_pending"
"""Placeholder carried by Bars/Quotes until :class:`MarketDataStore` assigns the content-derived version."""

PRICE_SCALE = 6
SIZE_SCALE = 6
_Q6 = Decimal(1).scaleb(-PRICE_SCALE)

# When a load is not restricted to one feed, a symbol's rows always come from exactly ONE feed,
# chosen in this order. Feeds are never mixed within a symbol (a synthetic bar can never be spliced
# into a real series or vice versa); every returned record still carries its own feed label.
FEED_PRIORITY: tuple[DataFeed, ...] = (DataFeed.SIP, DataFeed.IEX, DataFeed.LEAN_SAMPLE, DataFeed.SYNTHETIC)

IEX_LIMITATION = "Real-time quotes/trades from IEX only (~2-3% of consolidated volume); NBBO not guaranteed"
SIMULATED_LIMITATION = "Synthetic data — not market prices"

FEED_MODE_LIMITATIONS: dict[FeedMode, tuple[str, ...]] = {
    FeedMode.IEX_FREE: (
        IEX_LIMITATION,
        "Minute bars aggregate IEX trades only; minutes without an IEX trade are absent (gaps are reported, never filled)",
        "Alpaca Basic (free) plan; SIP consolidated data requires a paid plan that is never purchased automatically",
    ),
    FeedMode.SIP_PAID: (
        "Consolidated SIP feed (Alpaca Algo Trader Plus, paid); explicitly configured, never auto-upgraded or purchased",
        "Historical REST bars may include late-reported trades that were not knowable at bar close",
    ),
    FeedMode.SIMULATED: (
        SIMULATED_LIMITATION,
        "Deterministic replay at a simulated clock; spreads, volumes and prices are modelled, not observed",
    ),
}

FEED_LIMITATIONS: dict[DataFeed, tuple[str, ...]] = {
    DataFeed.SYNTHETIC: (SIMULATED_LIMITATION,),
    DataFeed.LEAN_SAMPLE: (
        "Historical QuantConnect LEAN repository sample data (old vintage, adjustment provenance unknown) "
        "— not current market prices",
    ),
    DataFeed.IEX: (IEX_LIMITATION,),
    DataFeed.SIP: ("Consolidated SIP data (paid subscription)",),
}

FEED_MODE_TO_FEED: dict[FeedMode, DataFeed] = {
    FeedMode.IEX_FREE: DataFeed.IEX,
    FeedMode.SIP_PAID: DataFeed.SIP,
    FeedMode.SIMULATED: DataFeed.SYNTHETIC,
}

_TF_RE = re.compile(r"^\s*(\d+)\s*(min|minute|t|h|hour|d|day)s?\s*$", re.IGNORECASE)


@lru_cache(maxsize=64)
def timeframe_delta(timeframe: str) -> timedelta:
    """'1Min' -> 1 minute, '5Min', '1Hour', '1Day'. Raises ValueError for anything else."""
    m = _TF_RE.match(timeframe)
    if not m:
        raise ValueError(f"unsupported timeframe {timeframe!r} (expected e.g. 1Min, 5Min, 1Hour, 1Day)")
    n = int(m.group(1))
    if n <= 0:
        raise ValueError("timeframe multiple must be positive")
    unit = m.group(2).lower()
    if unit in {"min", "minute", "t"}:
        return timedelta(minutes=n)
    if unit in {"h", "hour"}:
        return timedelta(hours=n)
    return timedelta(days=n)


def canonical_timeframe(timeframe: str) -> str:
    """Normalise spellings to the stored form: '1Min', '5Min', '1Hour', '1Day'."""
    delta = timeframe_delta(timeframe)
    if delta >= timedelta(days=1):
        return f"{delta.days}Day"
    minutes = int(delta.total_seconds() // 60)
    if minutes % 60 == 0:
        return f"{minutes // 60}Hour"
    return f"{minutes}Min"


def is_daily(timeframe: str) -> bool:
    return timeframe_delta(timeframe) >= timedelta(days=1)


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("naive datetime rejected; timestamps must be timezone-aware")
    return value.astimezone(UTC)


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return to_utc(value).isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return to_utc(dt)


def quantize_price(value: Decimal) -> Decimal:
    """Storage precision for normalised DECIMAL(18,6) columns (banker's rounding)."""
    return value.quantize(_Q6, rounding=ROUND_HALF_EVEN)


def clean_decimal(value: Decimal) -> Decimal:
    """Strip trailing zeros (``Decimal('452.130000') -> Decimal('452.13')``) without exponent notation."""
    if value == value.to_integral_value():
        return value.quantize(Decimal(1))
    return value.normalize()


def dec_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def to_decimal(value: Any) -> Decimal:
    """Exact Decimal for an I/O-boundary value. Floats (third-party JSON) go through ``repr``."""
    if isinstance(value, float):
        out = from_external_float(value)
        assert out is not None
        return out
    return D(value)


def feed_limitations(feed_mode: FeedMode, feed: DataFeed | None = None) -> tuple[str, ...]:
    out = list(FEED_MODE_LIMITATIONS.get(feed_mode, ()))
    if feed is not None:
        for item in FEED_LIMITATIONS.get(feed, ()):
            if item not in out:
                out.append(item)
    return tuple(out)
