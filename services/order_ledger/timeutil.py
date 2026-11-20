"""Fixed-width UTC timestamps for the ledger.

``packages.core.db.utc_iso`` omits microseconds when they are zero, so its strings do not sort
lexically in time order (``...00Z`` sorts after ``...00.5Z``). Every timestamp the ledger compares in
SQL (fills ordering, ``since`` windows) is written with this fixed-width formatter instead.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def iso(dt: datetime) -> str:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("naive datetime rejected")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"naive timestamp rejected: {text!r}")
    return dt.astimezone(UTC)


def trading_day(dt: datetime) -> date:
    """U.S. equities trading day (America/New_York calendar date) of an aware timestamp."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime rejected")
    return dt.astimezone(NY_TZ).date()
