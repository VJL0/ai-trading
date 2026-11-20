"""XNYS regular-session checks via ``exchange_calendars`` (holidays and early closes included).

The calendar is built once per process (~1 s) and cached. Any calendar error — including a timestamp
outside the calendar's loaded range — is reported as "not open", so the risk gate fails closed.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, cast
from zoneinfo import ZoneInfo

from packages.contracts.enums import MarketSession

CALENDAR_NAME = "XNYS"
NY_TZ = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def _calendar() -> Any:
    import exchange_calendars as xcals  # pyright: ignore[reportMissingTypeStubs]  # heavy; lazy import

    with warnings.catch_warnings():  # pandas deprecation noise from inside exchange_calendars
        warnings.simplefilter("ignore")
        cal = cast(Any, xcals).get_calendar(CALENDAR_NAME)
        cal.is_open_at_time(cal.first_session_open, side="left")  # build lazily-cached arrays now
    return cal


def _ts(now: datetime) -> Any:
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

    if now.tzinfo is None:
        raise ValueError("naive datetime")
    return cast(Any, pd).Timestamp(now.astimezone(UTC))


def is_regular_session_open(now: datetime) -> tuple[bool, str]:
    """(open?, detail). Open on the session open instant, closed on the close instant."""
    try:
        cal = _calendar()
        is_open = bool(cal.is_open_at_time(_ts(now), side="left"))
    except Exception as exc:  # calendar out of range / parse error => unknown => closed
        return False, f"XNYS calendar could not evaluate {now.isoformat()}: {type(exc).__name__}"
    if is_open:
        return True, "XNYS regular session open"
    return False, "XNYS regular session closed (outside regular hours, weekend, holiday or early close)"


def session_bounds(now: datetime) -> tuple[datetime, datetime] | None:
    """(open, close) in UTC of the XNYS session on ``now``'s New York date, or None if not a session."""
    try:
        import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

        cal = _calendar()
        ny_date = _ts(now).tz_convert("America/New_York").date()
        label = cast(Any, pd).Timestamp(ny_date)
        if not bool(cal.is_session(label)):
            return None
        open_ts = cal.session_open(label).to_pydatetime()
        close_ts = cal.session_close(label).to_pydatetime()
    except Exception:
        return None
    return cast(datetime, open_ts).astimezone(UTC), cast(datetime, close_ts).astimezone(UTC)


def market_session_at(now: datetime) -> MarketSession:
    """Classify ``now`` as PRE / REGULAR / POST / CLOSED for the XNYS session (fail closed => CLOSED)."""
    bounds = session_bounds(now)
    if bounds is None:
        return MarketSession.CLOSED
    open_ts, close_ts = bounds
    if open_ts <= now < close_ts:
        ok, _ = is_regular_session_open(now)
        return MarketSession.REGULAR if ok else MarketSession.CLOSED
    ny = now.astimezone(NY_TZ)
    if now < open_ts and ny.hour >= 4:
        return MarketSession.PRE
    if now >= close_ts and ny.hour < 20:
        return MarketSession.POST
    return MarketSession.CLOSED


def ny_date(now: datetime) -> str:
    """New York calendar date (YYYY-MM-DD) used for per-day order-rate limits."""
    return now.astimezone(NY_TZ).date().isoformat()
