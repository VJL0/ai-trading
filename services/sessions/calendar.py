"""XNYS regular-hours helpers (exchange_calendars), America/New_York wall time and UTC timestamp parsing.

Everything fails closed: a timestamp outside the calendar's range, or any calendar error, counts as "market
closed" / "no session", so a session never runs a cycle when regular hours cannot be established.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from typing import Any, Final
from zoneinfo import ZoneInfo

ET: Final = ZoneInfo("America/New_York")
CALENDAR_NAME: Final = "XNYS"
WEEKDAYS: Final[tuple[str, ...]] = ("MON", "TUE", "WED", "THU", "FRI")
_ALL_DAYS: Final[tuple[str, ...]] = (*WEEKDAYS, "SAT", "SUN")


@cache
def xnys() -> Any:
    import exchange_calendars as xcals

    return xcals.get_calendar(CALENDAR_NAME)


def _ts(dt: datetime) -> Any:
    import pandas as pd

    return pd.Timestamp(require_utc(dt))


def require_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("timezone-aware datetime required")
    return dt.astimezone(UTC)


def parse_utc(text: str | None) -> datetime | None:
    if text is None or text == "":
        return None
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return require_utc(dt)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return require_utc(dt).isoformat().replace("+00:00", "Z")


def et_date(dt: datetime) -> date:
    return require_utc(dt).astimezone(ET).date()


def et_hhmm(dt: datetime) -> str:
    return require_utc(dt).astimezone(ET).strftime("%H:%M")


def et_label(dt: datetime) -> str:
    """Human label in New York time, e.g. 'Mon 2026-11-09 09:45 ET'."""
    local = require_utc(dt).astimezone(ET)
    return f"{_ALL_DAYS[local.weekday()].title()} {local:%Y-%m-%d %H:%M} ET"


def weekday_code(d: date) -> str:
    return _ALL_DAYS[d.weekday()]


def et_datetime(d: date, hhmm: str) -> datetime:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return datetime.combine(d, time(hh, mm), ET).astimezone(UTC)


def is_market_open(now: datetime) -> bool:
    """True only inside an XNYS regular session (early closes included). Unknown -> False (fail closed)."""
    try:
        cal = xnys()
        return bool(cal.is_open_on_minute(_ts(now).floor("min")))
    except Exception:
        return False


def session_bounds(d: date) -> tuple[datetime, datetime] | None:
    """(open, close) in UTC of the XNYS regular session on ``d``, or None (holiday / weekend / out of range)."""
    try:
        import pandas as pd

        cal = xnys()
        label = pd.Timestamp(d)
        if not cal.is_session(label):
            return None
        return (cal.session_open(label).to_pydatetime().astimezone(UTC),
                cal.session_close(label).to_pydatetime().astimezone(UTC))
    except Exception:
        return None


def next_open(now: datetime, *, horizon_days: int = 10) -> datetime | None:
    """Next regular-session open strictly after ``now`` (None if unknown)."""
    d = et_date(now)
    for k in range(horizon_days + 1):
        bounds = session_bounds(d + timedelta(days=k))
        if bounds is not None and bounds[0] > require_utc(now):
            return bounds[0]
    return None


def regular_minutes_between(start: datetime, end: datetime) -> bool:
    """Whether any regular-hours minute exists in [start, end)."""
    if end <= start:
        return False
    if is_market_open(start):
        return True
    nxt = next_open(start, horizon_days=max(2, (end - start).days + 2))
    return nxt is not None and nxt < end
