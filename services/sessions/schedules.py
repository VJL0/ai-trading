"""Recurring session schedules: validation and occurrence arithmetic (pure functions, no I/O besides the calendar).

A schedule names weekdays (MON..FRI only) and a New York wall-clock start time. An *occurrence* exists on a date
when that date is an XNYS session on one of the weekdays and the start time lies inside that day's regular
session (holidays are skipped; on an early-close day a start after the close is skipped). Each occurrence has a
window ``[start, start + duration)``; a run whose window has passed is MISSED and is never started late.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final

from .calendar import WEEKDAYS, et_date, et_datetime, require_utc, session_bounds, weekday_code
from .errors import SessionError

START_TIME_RE: Final = re.compile(r"^([01][0-9]|2[0-3]):[0-5][0-9]$")
MAX_LOOKBACK_DAYS: Final = 7
HORIZON_DAYS: Final = 21
REGULAR_SESSION_MINUTES: Final = 390  # 09:30-16:00 ET on a full trading day


@dataclass(frozen=True)
class Occurrence:
    run_date: date
    start: datetime
    end: datetime


def normalize_weekdays(weekdays: Sequence[str]) -> tuple[str, ...]:
    if isinstance(weekdays, str) or not weekdays:
        raise SessionError(422, "weekdays must be a non-empty list such as ['MON', 'WED']")
    out: set[str] = set()
    for w in weekdays:
        code = str(w).strip().upper()[:3]
        if code not in WEEKDAYS:
            raise SessionError(422, f"weekday {w!r} is not a trading weekday; use MON, TUE, WED, THU or FRI")
        out.add(code)
    return tuple(d for d in WEEKDAYS if d in out)


def validate_start_time(text: str) -> str:
    if not isinstance(text, str) or not START_TIME_RE.fullmatch(text.strip()):
        raise SessionError(422, "start_time_et must be New York wall time 'HH:MM' (24h), e.g. '09:45'")
    return text.strip()


def occurrence_on(d: date, start_time_et: str, duration_minutes: int) -> Occurrence | None:
    bounds = session_bounds(d)
    if bounds is None:
        return None
    open_, close = bounds
    start = et_datetime(d, start_time_et)
    if not (open_ <= start < close):
        return None
    return Occurrence(d, start, start + timedelta(minutes=duration_minutes))


def validate_window(weekdays: Sequence[str], start_time_et: str, duration_minutes: int, *, today: date) -> None:
    """Each weekday's window must fit a full XNYS regular session (via exchange_calendars), e.g. 09:30-16:00 ET."""
    for code in weekdays:
        found = False
        for k in range(1, 35):
            d = today + timedelta(days=k)
            if weekday_code(d) != code:
                continue
            bounds = session_bounds(d)
            if bounds is None or (bounds[1] - bounds[0]) < timedelta(minutes=REGULAR_SESSION_MINUTES):
                continue  # holiday or early close: validate against a full regular session
            open_, close = bounds
            start = et_datetime(d, start_time_et)
            end = start + timedelta(minutes=duration_minutes)
            if start < open_:
                raise SessionError(422, f"{start_time_et} ET is before the XNYS regular open (09:30 ET); sessions "
                                        "run only during regular hours")
            if end > close:
                raise SessionError(422, f"a {duration_minutes}-minute session starting {start_time_et} ET would run "
                                        "past the XNYS regular close (16:00 ET); shorten it or start earlier")
            found = True
            break
        if not found:
            raise SessionError(422, f"no upcoming full XNYS regular session found for {code} (calendar unavailable)")


def occurrences(weekdays: Collection[str], start_time_et: str, duration_minutes: int, first: date,
                last: date) -> list[Occurrence]:
    out: list[Occurrence] = []
    d = first
    while d <= last:
        if weekday_code(d) in weekdays:
            occ = occurrence_on(d, start_time_et, duration_minutes)
            if occ is not None:
                out.append(occ)
        d += timedelta(days=1)
    return out


def pending_occurrences(weekdays: Collection[str], start_time_et: str, duration_minutes: int, *,
                        created_at: datetime, now: datetime, recorded: Collection[date]) -> list[Occurrence]:
    """Occurrences that are due (start <= now), created after the schedule, and not yet recorded; oldest first."""
    now = require_utc(now)
    first = max(et_date(created_at), et_date(now) - timedelta(days=MAX_LOOKBACK_DAYS))
    return [o for o in occurrences(weekdays, start_time_et, duration_minutes, first, et_date(now))
            if o.start <= now and o.start >= created_at and o.run_date not in recorded]


def next_occurrence(weekdays: Collection[str], start_time_et: str, duration_minutes: int, *, created_at: datetime,
                    now: datetime, recorded: Collection[date]) -> Occurrence | None:
    """Earliest unrecorded occurrence whose window has not passed (it may already have started)."""
    now = require_utc(now)
    first = max(et_date(created_at), et_date(now) - timedelta(days=1))
    for o in occurrences(weekdays, start_time_et, duration_minutes, first, et_date(now) + timedelta(days=HORIZON_DAYS)):
        if o.start >= created_at and o.end > now and o.run_date not in recorded:
            return o
    return None
