"""After-close quantlab maintenance scheduling (no subprocesses, no network: the step runner is faked)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from services.market_data.calendar import get_calendar
from services.quantlab.daily import STEPS, DailyJob, StepResult, due_session, next_check


def _at(d: date, hh: int, mm: int) -> datetime:
    from zoneinfo import ZoneInfo

    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)


def test_due_session_respects_close_settle_weekends_and_early_closes() -> None:
    cal = get_calendar()
    mon = date(2026, 11, 23)
    assert cal.is_session(mon)
    assert due_session(_at(mon, 16, 10), None) == cal.previous_session(mon)  # before close + 30 min
    assert due_session(_at(mon, 16, 31), None) == mon
    assert due_session(_at(mon, 16, 31), mon) is None  # already done
    sat = date(2026, 11, 28)
    assert due_session(_at(sat, 12, 0), None) == date(2026, 11, 27)  # Friday 2026-11-27 (13:00 early close)
    assert due_session(_at(date(2026, 11, 27), 13, 31), None) == date(2026, 11, 27)
    nxt = next_check(_at(mon, 12, 0))
    assert nxt == cal.session_close(mon) + timedelta(minutes=30)
    assert next_check(_at(mon, 17, 0)) == cal.session_close(cal.next_session(mon)) + timedelta(minutes=30)


class FakeRunner:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail_on = fail_on

    async def __call__(self, args: Sequence[str], cwd: Path, timeout_s: float) -> StepResult:
        self.calls.append(tuple(args))
        rc = 1 if self.fail_on and args[0] == self.fail_on else 0
        return StepResult(" ".join(args), rc, 0.1, "boom" if rc else "ok")


def test_runs_each_session_once_and_survives_restarts(tmp_path: Path) -> None:
    mon = date(2026, 11, 23)
    runner = FakeRunner()
    done: list[dict] = []  # type: ignore[type-arg]
    job = DailyJob(tmp_path, clock=lambda: _at(mon, 17, 0), runner=runner, on_done=done.append)
    first = asyncio.run(job.run_once())
    assert first is not None and first["ok"] and first["session"] == mon.isoformat()
    assert runner.calls == list(STEPS) and len(done) == 1
    assert asyncio.run(job.run_once()) is None  # same session: nothing to do
    restarted = DailyJob(tmp_path, clock=lambda: _at(mon, 18, 0), runner=runner)
    assert asyncio.run(restarted.run_once()) is None  # durable state: a restart does not repeat the session
    later = DailyJob(tmp_path, clock=lambda: _at(date(2026, 11, 26), 9, 0), runner=runner)  # Thanksgiving
    res = asyncio.run(later.run_once())
    assert res is not None and res["session"] == "2026-11-25"  # missed Tuesday+Wednesday: caught up ONCE
    assert len(runner.calls) == 2 * len(STEPS)


def test_failed_step_stops_the_sequence_and_is_retried(tmp_path: Path) -> None:
    mon = date(2026, 11, 23)
    failures: list[tuple[str, str]] = []
    runner = FakeRunner(fail_on="build-panel")
    job = DailyJob(tmp_path, clock=lambda: _at(mon, 17, 0), runner=runner,
                   on_failure=lambda s, d: failures.append((s, d)))
    res = asyncio.run(job.run_once())
    assert res is not None and not res["ok"]
    assert [c[0] for c in runner.calls] == ["fetch-delta", "build-panel"]  # scan/shadow never ran on a bad panel
    assert failures and failures[0][0].startswith("build-panel")
    assert job.last_done() is None  # not marked done -> retried
    job.runner = FakeRunner()
    assert asyncio.run(job.run_once())["ok"]  # type: ignore[index]
    assert job.last_done() == mon


def test_never_runs_in_premarket_or_regular_hours() -> None:
    from services.quantlab.daily import in_quiet_window

    tue = date(2026, 11, 24)
    assert in_quiet_window(_at(tue, 17, 0)) and in_quiet_window(_at(tue, 23, 0))
    assert in_quiet_window(_at(date(2026, 11, 25), 7, 59))
    assert not in_quiet_window(_at(date(2026, 11, 25), 8, 1))  # 90 minutes before the open
    assert not in_quiet_window(_at(tue, 12, 0)) and not in_quiet_window(_at(tue, 16, 15))  # before close + settle
    assert in_quiet_window(_at(date(2026, 11, 28), 12, 0))  # Saturday


def test_catch_up_due_during_the_day_waits_for_the_close_and_retries_are_capped(tmp_path: Path) -> None:
    from services.quantlab.daily import MAX_ATTEMPTS

    runner = FakeRunner()
    day = DailyJob(tmp_path, clock=lambda: _at(date(2026, 11, 24), 11, 0), runner=runner)
    assert asyncio.run(day.run_once()) is None and runner.calls == []  # Monday is due, but it is market hours
    failing = FakeRunner(fail_on="fetch-delta")
    night = DailyJob(tmp_path, clock=lambda: _at(date(2026, 11, 24), 2, 0), runner=failing)
    results = [asyncio.run(night.run_once()) for _ in range(MAX_ATTEMPTS + 2)]
    assert sum(r is not None for r in results) == MAX_ATTEMPTS  # then it stops hammering the data API
    assert len(failing.calls) == MAX_ATTEMPTS


def test_cancelled_step_kills_its_child(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os
    import signal

    from services.quantlab import daily

    spawned: list[int] = []
    real = asyncio.create_subprocess_exec

    async def scenario() -> None:
        started = asyncio.Event()

        async def fake_exec(*_args, **kwargs):  # type: ignore[no-untyped-def]
            proc = await real("sleep", "30", **kwargs)
            spawned.append(proc.pid)
            started.set()
            return proc

        monkeypatch.setattr(daily.asyncio, "create_subprocess_exec", fake_exec)
        task = asyncio.create_task(daily.run_step(("scan",), tmp_path, 600))
        await started.wait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    import time

    t0 = time.monotonic()
    asyncio.run(scenario())
    assert time.monotonic() - t0 < 5.0  # killed promptly, not left to run out its 30 s
    try:
        os.kill(spawned[0], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:  # reaped zombie check: a killed child must not keep running
        os.kill(spawned[0], signal.SIGKILL)
    assert not alive


def test_steps_stop_when_the_quiet_window_ends_mid_run(tmp_path: Path) -> None:
    times = iter([_at(date(2026, 11, 25), 7, 55), _at(date(2026, 11, 25), 7, 58), _at(date(2026, 11, 25), 8, 5)])
    last = [_at(date(2026, 11, 25), 8, 5)]

    def clock() -> datetime:
        with contextlib.suppress(StopIteration):
            last[0] = next(times)
        return last[0]

    runner = FakeRunner()
    job = DailyJob(tmp_path, clock=clock, runner=runner)
    res = asyncio.run(job.run_once())
    assert res is not None and not res["ok"]
    assert [c[0] for c in runner.calls] == ["fetch-delta"]  # the 08:05 re-check deferred build-panel/scan/shadow
    assert job.last_done() is None
