"""XNYS exchange calendar helpers built on ``exchange_calendars`` (holidays and early closes included).

All inputs and outputs are timezone-aware UTC datetimes (session *dates* are New York trading dates).
The calendar is loaded once per process from 1995 so the bundled LEAN sample data (1998+) is covered.
A timestamp outside the loaded calendar range raises :class:`CalendarOutOfRange` instead of guessing,
so callers fail closed.

Session classification (U.S. equities, America/New_York):

* ``REGULAR`` — ``session_open <= ts < session_close`` on an XNYS session (09:30-16:00, early closes 13:00)
* ``PRE``     — 04:00 <= ts < session_open on a session day
* ``POST``    — session_close <= ts < 20:00 (17:00 on early-close days) on a session day
* ``CLOSED``  — everything else (nights, weekends, holidays)
"""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from packages.contracts.enums import MarketSession

NY = ZoneInfo("America/New_York")
CALENDAR_NAME = "XNYS"
CALENDAR_START = "1995-01-01"
PRE_MARKET_OPEN = time(4, 0)
POST_MARKET_CLOSE = time(20, 0)
EARLY_CLOSE_POST_MARKET_CLOSE = time(17, 0)
REGULAR_CLOSE = time(16, 0)


class CalendarOutOfRange(ValueError):
    """The requested date is outside the loaded exchange calendar; the caller must fail closed."""


def _require_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("naive datetime rejected; timestamps must be timezone-aware")
    return ts.astimezone(UTC)


class MarketCalendar:
    """Thin, cached view over the XNYS schedule. Construct once (see :func:`get_calendar`)."""

    def __init__(self, name: str = CALENDAR_NAME, start: str = CALENDAR_START) -> None:
        import warnings

        import exchange_calendars as xcals

        with warnings.catch_warnings():  # exchange_calendars 4.13 trips a pandas FutureWarning-style deprecation
            warnings.filterwarnings("ignore", message=r"Series\.values returning an ndarray")
            cal = cast(Any, xcals).get_calendar(name, start=start)
        schedule = cal.schedule
        self.name = name
        self._open: dict[date, datetime] = {}
        self._close: dict[date, datetime] = {}
        for label, op, cl in zip(schedule.index, schedule["open"], schedule["close"], strict=True):
            d = label.date()
            self._open[d] = op.to_pydatetime().astimezone(UTC)
            self._close[d] = cl.to_pydatetime().astimezone(UTC)
        self._sessions: list[date] = sorted(self._open)
        self._index = {d: i for i, d in enumerate(self._sessions)}
        self.first_session: date = self._sessions[0]
        self.last_session: date = self._sessions[-1]

    # ------------------------------------------------------------------ dates
    def _check_range(self, d: date) -> None:
        if d < self.first_session or d > self.last_session:
            raise CalendarOutOfRange(
                f"{d.isoformat()} is outside the loaded {self.name} calendar "
                f"({self.first_session.isoformat()}..{self.last_session.isoformat()})"
            )

    def covers(self, d: date) -> bool:
        return self.first_session <= d <= self.last_session

    def is_session(self, d: date) -> bool:
        self._check_range(d)
        return d in self._open

    def session_open(self, d: date) -> datetime:
        self._check_range(d)
        try:
            return self._open[d]
        except KeyError:
            raise ValueError(f"{d.isoformat()} is not an {self.name} session") from None

    def session_close(self, d: date) -> datetime:
        self._check_range(d)
        try:
            return self._close[d]
        except KeyError:
            raise ValueError(f"{d.isoformat()} is not an {self.name} session") from None

    def is_early_close(self, d: date) -> bool:
        return self.session_close(d).astimezone(NY).time() < REGULAR_CLOSE

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        """Sessions with ``start <= d <= end`` (inclusive)."""
        if end < start:
            return []
        self._check_range(start)
        self._check_range(end)
        from bisect import bisect_left, bisect_right

        return self._sessions[bisect_left(self._sessions, start) : bisect_right(self._sessions, end)]

    def sessions_ending(self, end: date, count: int) -> list[date]:
        """The last ``count`` sessions on or before ``end`` (chronological)."""
        if count <= 0:
            return []
        self._check_range(end)
        from bisect import bisect_right

        hi = bisect_right(self._sessions, end)
        lo = hi - count
        if lo < 0:
            raise CalendarOutOfRange(f"fewer than {count} sessions before {end.isoformat()} in the calendar")
        return self._sessions[lo:hi]

    def next_session(self, d: date) -> date:
        """First session strictly after ``d``."""
        self._check_range(d)
        from bisect import bisect_right

        i = bisect_right(self._sessions, d)
        if i >= len(self._sessions):
            raise CalendarOutOfRange(f"no session after {d.isoformat()} in the loaded calendar")
        return self._sessions[i]

    def previous_session(self, d: date) -> date:
        """Last session strictly before ``d``."""
        self._check_range(d)
        from bisect import bisect_left

        i = bisect_left(self._sessions, d) - 1
        if i < 0:
            raise CalendarOutOfRange(f"no session before {d.isoformat()} in the loaded calendar")
        return self._sessions[i]

    # ------------------------------------------------------------------ instants
    @staticmethod
    def ny_date(ts: datetime) -> date:
        return _require_aware(ts).astimezone(NY).date()

    def classify(self, ts: datetime) -> MarketSession:
        """Market session at instant ``ts``."""
        ts = _require_aware(ts)
        local = ts.astimezone(NY)
        d = local.date()
        if not self.is_session(d):
            return MarketSession.CLOSED
        op, cl = self._open[d], self._close[d]
        if op <= ts < cl:
            return MarketSession.REGULAR
        pre_start = datetime.combine(d, PRE_MARKET_OPEN, tzinfo=NY).astimezone(UTC)
        post_end_t = EARLY_CLOSE_POST_MARKET_CLOSE if self.is_early_close(d) else POST_MARKET_CLOSE
        post_end = datetime.combine(d, post_end_t, tzinfo=NY).astimezone(UTC)
        if pre_start <= ts < op:
            return MarketSession.PRE
        if cl <= ts < post_end:
            return MarketSession.POST
        return MarketSession.CLOSED

    def classify_bar(self, start_ts: datetime, end_ts: datetime) -> MarketSession:
        """REGULAR only when the whole bar ``[start_ts, end_ts)`` lies inside one regular session.

        A daily bar spanning exactly ``[open, close]`` of a session is REGULAR. Otherwise the session of
        the bar's start instant is returned (PRE/POST/CLOSED); a bar straddling the open is PRE and one
        straddling the close is POST, i.e. never REGULAR.
        """
        start_ts = _require_aware(start_ts)
        end_ts = _require_aware(end_ts)
        d = start_ts.astimezone(NY).date()
        if self.is_session(d):
            op, cl = self._open[d], self._close[d]
            if op <= start_ts and end_ts <= cl:
                return MarketSession.REGULAR
        session = self.classify(start_ts)
        # starts inside the regular session but ends after the close (or the date's session is unknown): POST
        return MarketSession.POST if session == MarketSession.REGULAR else session

    def regular_bar_starts(self, d: date, step: timedelta = timedelta(minutes=1)) -> list[datetime]:
        """Start instants of every regular-session bar of width ``step`` on session ``d``."""
        op, cl = self.session_open(d), self.session_close(d)
        out: list[datetime] = []
        t = op
        while t + step <= cl:
            out.append(t)
            t += step
        return out

    def regular_minutes_count(self, d: date) -> int:
        return int((self.session_close(d) - self.session_open(d)).total_seconds() // 60)


_lock = threading.Lock()
_instance: MarketCalendar | None = None


def get_calendar() -> MarketCalendar:
    """Process-wide cached XNYS calendar."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MarketCalendar()
    return _instance
