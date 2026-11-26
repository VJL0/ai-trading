"""The autopilot's trading-day plan on the XNYS calendar (holidays and early closes included).

Built only from the existing helpers: ``services.sessions.calendar`` (session bounds, New York dates) and
``services.risk.market_hours`` (regular-hours check), plus the risk engine's entry cutoff constant. Everything
fails closed: an unknown calendar answer means "no session", so the autopilot never requests trading it cannot
place inside a known regular session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Final

from services.risk.limits import ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS
from services.risk.market_hours import is_regular_session_open
from services.sessions.calendar import et_date, require_utc, session_bounds

FULL_SESSION: Final = timedelta(minutes=390)
QUIET_BEFORE_OPEN: Final = timedelta(minutes=90)  # same quiet window as services.quantlab.daily
MIN_SESSION_MINUTES: Final = 15
LOOKBACK_DAYS: Final = 10


class Phase(StrEnum):
    CLOSED = "CLOSED"            # weekend, holiday, overnight before the pre-open window
    PREOPEN = "PREOPEN"          # [open - lead, open + grace): health checks and today's session request
    REGULAR = "REGULAR"          # regular hours: bounded status checks only
    POST_CLOSE = "POST_CLOSE"    # [close, close + settle): wait for data to settle
    QUIET = "QUIET"              # after close + settle until 90 min before the next open: research


@dataclass(frozen=True)
class TradingDay:
    day: date
    open: datetime
    close: datetime

    @property
    def early_close(self) -> bool:
        return (self.close - self.open) < FULL_SESSION

    @property
    def entry_cutoff(self) -> datetime:
        return self.close - timedelta(seconds=ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS)

    def preopen_start(self, lead_min: int) -> datetime:
        return self.open - timedelta(minutes=lead_min)

    def start_deadline(self, grace_min: int) -> datetime:
        return self.open + timedelta(minutes=grace_min)

    def research_start(self, settle_min: int) -> datetime:
        return self.close + timedelta(minutes=settle_min)

    def public(self) -> dict[str, object]:
        return {"day": self.day.isoformat(), "open": self.open.isoformat(), "close": self.close.isoformat(),
                "early_close": self.early_close, "entry_cutoff": self.entry_cutoff.isoformat()}


def trading_day(d: date) -> TradingDay | None:
    bounds = session_bounds(d)
    if bounds is None:
        return None
    return TradingDay(d, bounds[0], bounds[1])


def today(now: datetime) -> date:
    return et_date(now)


def session_today(now: datetime) -> TradingDay | None:
    return trading_day(et_date(now))


def last_completed(now: datetime) -> TradingDay | None:
    """The latest session whose close is at or before ``now``."""
    now = require_utc(now)
    d = et_date(now)
    for k in range(LOOKBACK_DAYS + 1):
        td = trading_day(d - timedelta(days=k))
        if td is not None and td.close <= now:
            return td
    return None


def next_session(after: date) -> TradingDay | None:
    for k in range(1, LOOKBACK_DAYS + 1):
        td = trading_day(after + timedelta(days=k))
        if td is not None:
            return td
    return None


def quiet_until(td: TradingDay) -> datetime | None:
    """End of the research window that follows ``td``: 90 minutes before the next session's open."""
    nxt = next_session(td.day)
    return None if nxt is None else nxt.open - QUIET_BEFORE_OPEN


def phase_at(now: datetime, *, lead_min: int, grace_min: int, settle_min: int) -> tuple[Phase, TradingDay | None]:
    """Phase of ``now`` and the trading day it refers to (today's session, or the last completed one)."""
    now = require_utc(now)
    td = session_today(now)
    if td is not None:
        if td.preopen_start(lead_min) <= now < td.start_deadline(grace_min):
            return Phase.PREOPEN, td
        if td.open <= now < td.close:
            return Phase.REGULAR, td
    last = last_completed(now)
    if last is not None:
        if now < last.research_start(settle_min):
            return Phase.POST_CLOSE, last
        end = quiet_until(last)
        if end is not None and now < end:
            return Phase.QUIET, last
    return Phase.CLOSED, td or last


def market_open(now: datetime) -> bool:
    ok, _ = is_regular_session_open(require_utc(now))
    return ok


def session_minutes(td: TradingDay, start_at: datetime, cap: int | None) -> int:
    """Minutes from ``max(start_at, open)`` to the entry cutoff, capped. 0 when no usable window remains.

    The session manager clamps a session at the close itself; ending at the entry cutoff instead keeps the
    session out of the last 30 minutes, where the risk engine refuses every new entry anyway."""
    start = max(require_utc(start_at), td.open)
    minutes = math.floor((td.entry_cutoff - start).total_seconds() / 60)
    if cap is not None:
        minutes = min(minutes, cap)
    return minutes if minutes >= MIN_SESSION_MINUTES else 0
