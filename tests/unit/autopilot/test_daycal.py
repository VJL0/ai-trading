"""XNYS day plan: holidays, early closes, phases and the session length (entry cutoff)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from services.autopilot.daycal import Phase, last_completed, phase_at, session_minutes, trading_day

from .fakes import et

KW = {"lead_min": 60, "grace_min": 5, "settle_min": 30}


def test_thanksgiving_has_no_session_and_friday_closes_early() -> None:
    assert trading_day(date(2026, 11, 26)) is None
    fri = trading_day(date(2026, 11, 27))
    assert fri is not None and fri.early_close
    assert fri.close == et(2026, 11, 27, 13)
    assert fri.entry_cutoff == et(2026, 11, 27, 12, 30)
    wed = trading_day(date(2026, 11, 25))
    assert wed is not None and not wed.early_close


def test_phases_through_thanksgiving_week() -> None:
    assert phase_at(et(2026, 11, 25, 8, 45), **KW)[0] is Phase.PREOPEN
    assert phase_at(et(2026, 11, 25, 9, 34), **KW)[0] is Phase.PREOPEN  # inside the start grace
    assert phase_at(et(2026, 11, 25, 9, 36), **KW)[0] is Phase.REGULAR
    assert phase_at(et(2026, 11, 25, 16, 10), **KW)[0] is Phase.POST_CLOSE
    phase, td = phase_at(et(2026, 11, 25, 17), **KW)
    assert phase is Phase.QUIET and td is not None and td.day == date(2026, 11, 25)
    # Thanksgiving: the whole day belongs to Wednesday's research window, no pre-open, no regular hours.
    phase, td = phase_at(et(2026, 11, 26, 9, 0), **KW)
    assert phase is Phase.QUIET and td is not None and td.day == date(2026, 11, 25)
    # Friday: the quiet window ends 90 minutes before the open; pre-open starts 60 minutes before.
    assert phase_at(et(2026, 11, 27, 8, 15), **KW)[0] is Phase.CLOSED
    assert phase_at(et(2026, 11, 27, 8, 30), **KW)[0] is Phase.PREOPEN
    assert phase_at(et(2026, 11, 27, 12, 59), **KW)[0] is Phase.REGULAR
    assert phase_at(et(2026, 11, 27, 13, 10), **KW)[0] is Phase.POST_CLOSE
    assert phase_at(et(2026, 11, 27, 13, 40), **KW)[0] is Phase.QUIET


def test_session_minutes_end_at_the_entry_cutoff() -> None:
    wed = trading_day(date(2026, 11, 25))
    fri = trading_day(date(2026, 11, 27))
    assert wed is not None and fri is not None
    assert session_minutes(wed, et(2026, 11, 25, 8, 45), None) == 360  # 09:30 -> 15:30
    assert session_minutes(fri, et(2026, 11, 27, 8, 45), None) == 180  # 09:30 -> 12:30 (early close)
    assert session_minutes(fri, et(2026, 11, 27, 9, 33), None) == 177  # late in the grace: still ends at cutoff
    assert session_minutes(wed, et(2026, 11, 25, 8, 45), 90) == 90
    assert session_minutes(wed, et(2026, 11, 25, 15, 20), None) == 0  # under 15 minutes left


def test_last_completed_skips_the_holiday() -> None:
    td = last_completed(et(2026, 11, 27, 8, 0))
    assert td is not None and td.day == date(2026, 11, 25)


def test_open_stays_at_0930_eastern_across_dst_transitions() -> None:
    for month, day, utc_hour in ((3, 6, 14), (3, 9, 13), (10, 30, 13), (11, 2, 14)):
        td = trading_day(date(2026, month, day))
        assert td is not None
        assert td.open == datetime(2026, month, day, utc_hour, 30, tzinfo=UTC)
        assert phase_at(et(2026, month, day, 8, 45), **KW)[0] is Phase.PREOPEN
