"""XNYS session classification (holidays, early closes, extended hours) and timeframe helpers."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from packages.contracts.enums import MarketSession
from services.market_data.calendar import CalendarOutOfRange, get_calendar
from services.market_data.common import canonical_timeframe, timeframe_delta
from tests.unit.market_data.md_helpers import utc

CAL = get_calendar()


def test_regular_session_bounds_est() -> None:
    d = date(2026, 11, 3)  # after the DST change: 09:30 ET = 14:30Z
    assert CAL.session_open(d) == utc(2026, 11, 3, 14, 30)
    assert CAL.session_close(d) == utc(2026, 11, 3, 21, 0)
    assert CAL.regular_minutes_count(d) == 390


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        (utc(2026, 11, 3, 9, 0), MarketSession.PRE),  # 04:00 ET
        (utc(2026, 11, 3, 8, 59), MarketSession.CLOSED),  # 03:59 ET
        (utc(2026, 11, 3, 14, 29, 59), MarketSession.PRE),
        (utc(2026, 11, 3, 14, 30), MarketSession.REGULAR),
        (utc(2026, 11, 3, 20, 59, 59), MarketSession.REGULAR),
        (utc(2026, 11, 3, 21, 0), MarketSession.POST),
        (utc(2026, 11, 4, 0, 59), MarketSession.POST),  # 19:59 ET
        (utc(2026, 11, 4, 1, 0), MarketSession.CLOSED),  # 20:00 ET
        (utc(2026, 11, 7, 15, 0), MarketSession.CLOSED),  # Saturday
        (utc(2026, 11, 26, 15, 0), MarketSession.CLOSED),  # Thanksgiving
    ],
)
def test_classify(ts: object, expected: MarketSession) -> None:
    assert CAL.classify(ts) == expected  # type: ignore[arg-type]


def test_early_close_day_after_thanksgiving() -> None:
    d = date(2026, 11, 27)
    assert CAL.is_session(d) and CAL.is_early_close(d)
    assert CAL.regular_minutes_count(d) == 210
    assert CAL.classify(utc(2026, 11, 27, 18, 30)) == MarketSession.POST  # 13:30 ET
    assert CAL.classify(utc(2026, 11, 27, 22, 30)) == MarketSession.CLOSED  # 17:30 ET (post ends 17:00)


def test_classify_bar_straddles_are_never_regular() -> None:
    op, cl = CAL.session_open(date(2026, 11, 3)), CAL.session_close(date(2026, 11, 3))
    assert CAL.classify_bar(op, op + timedelta(minutes=1)) == MarketSession.REGULAR
    assert CAL.classify_bar(op - timedelta(seconds=30), op + timedelta(seconds=30)) == MarketSession.PRE
    assert CAL.classify_bar(cl - timedelta(seconds=30), cl + timedelta(seconds=30)) == MarketSession.POST
    assert CAL.classify_bar(op, cl) == MarketSession.REGULAR  # the daily bar


def test_sessions_and_out_of_range_fail_closed() -> None:
    days = CAL.sessions_in_range(date(2026, 11, 23), date(2026, 11, 30))
    assert date(2026, 11, 26) not in days and len(days) == 5
    assert CAL.previous_session(date(2026, 11, 9)) == date(2026, 11, 6)
    with pytest.raises(CalendarOutOfRange):
        CAL.is_session(date(1990, 1, 2))
    with pytest.raises(ValueError, match="naive"):
        CAL.classify(utc(2026, 11, 3, 15, 0).replace(tzinfo=None))


def test_timeframes() -> None:
    assert canonical_timeframe("1min") == "1Min" and canonical_timeframe("60Min") == "1Hour"
    assert canonical_timeframe("1Day") == "1Day" and timeframe_delta("5Min") == timedelta(minutes=5)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        timeframe_delta("1Fortnight")
