"""XNYS regular-session bounds (exchange_calendars; handles holidays and early closes)."""

from __future__ import annotations

import threading
import warnings
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any

from .timeutil import trading_day

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _calendar() -> Any:
    import exchange_calendars as xcals

    with _lock:
        return xcals.get_calendar("XNYS")


@lru_cache(maxsize=512)
def _bounds_for_day(day_iso: str) -> tuple[datetime, datetime] | None:
    import pandas as pd

    ts = pd.Timestamp(day_iso)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pandas deprecation noise from inside exchange_calendars
            cal = _calendar()
            if not cal.is_session(ts):
                return None
            return (
                cal.session_open(ts).to_pydatetime().astimezone(UTC),
                cal.session_close(ts).to_pydatetime().astimezone(UTC),
            )
    except Exception:  # outside the calendar's range: unknown => closed (fail closed)
        return None


def session_bounds(now: datetime) -> tuple[datetime, datetime] | None:
    """(open, close) of the XNYS session on ``now``'s New York calendar date, or None if not a session."""
    return _bounds_for_day(trading_day(now).isoformat())


def is_regular_session(now: datetime) -> bool:
    b = session_bounds(now)
    return b is not None and b[0] <= now < b[1]


def session_dates(start: date, count: int) -> list[date]:
    """``count`` consecutive XNYS sessions starting at ``start`` (or the next session if ``start`` is not one).
    Raises ``ValueError`` outside the calendar range (callers fail closed)."""
    import pandas as pd

    if count < 1:
        raise ValueError("count must be >= 1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = _calendar()
        ts = pd.Timestamp(start.isoformat())
        first = ts if cal.is_session(ts) else cal.date_to_session(ts, direction="next")
        sessions = cal.sessions_window(first, count)
    return [d.date() for d in sessions]


def session_offset(start: date, sessions: int) -> date:
    """The XNYS session ``sessions - 1`` sessions after ``start`` (1 => ``start``'s own session)."""
    return session_dates(start, sessions)[-1]


def sessions_between(start: date, end: date) -> int:
    """Number of XNYS sessions in [start, end] (0 when end < start). Raises outside the calendar range."""
    import pandas as pd

    if end < start:
        return 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(_calendar().sessions_in_range(pd.Timestamp(start.isoformat()), pd.Timestamp(end.isoformat())))
