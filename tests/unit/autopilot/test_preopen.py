"""Pre-open with a fake clock: start, idempotency, holiday, early close, halts, kill switch, failed health checks,
crash resume, unknown outcomes and missed windows."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from services.autopilot.store import Store
from services.autopilot.supervisor import PREOPEN
from services.tools.handlers import ToolError

from .fakes import _session, et, make


def tick(ap: object) -> list[dict[str, object]]:
    return asyncio.run(ap.tick())  # type: ignore[attr-defined]


def test_preopen_requests_one_bounded_paper_session(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    tick(ap)
    starts = [a for n, a in api.calls if n == "start_paper_session"]
    assert starts == [{"duration_minutes": 360, "symbols": None, "cadence_minutes": None, "use_ai": False}]
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "DONE" and job["detail"]["outcome"] == "SCHEDULED"
    assert (tmp_path / "ap" / "reports" / "2026-11-25.json").is_file()
    assert (tmp_path / "ap" / "reports" / "2026-11-25.md").read_text().startswith("# Autopilot daily report")


def test_duplicate_ticks_never_create_a_second_session(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    tick(ap)
    for _ in range(5):
        clock.advance(minutes=6)
        tick(ap)
    assert api.count("start_paper_session") == 1


def test_existing_session_or_operator_schedule_is_adopted_not_duplicated(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path / "a", et(2026, 11, 25, 8, 45))
    api.active = _session("sess_op", "RUNNING", requested_by="operator")
    tick(ap)
    assert api.count("start_paper_session") == 0
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["detail"]["outcome"] == "ALREADY_ACTIVE"

    ap2, _, api2, _, _ = make(tmp_path / "b", et(2026, 11, 25, 8, 45))
    api2.schedules = [{"schedule_id": "sched_1", "enabled": True, "next_run_at": "2026-11-25T14:45:00Z",
                       "created_by": "operator"}]
    tick(ap2)
    assert api2.count("start_paper_session") == 0
    job2 = ap2.store.job("2026-11-25", PREOPEN)
    assert job2 is not None and job2["detail"]["outcome"] == "OPERATOR_SCHEDULE"


def test_holiday_does_nothing(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 26, 8, 45), research=False)
    plan = ap.decide(et(2026, 11, 26, 8, 45))
    assert plan.actions == []
    assert any("not an XNYS session" in n for n in plan.notes)
    tick(ap)
    assert api.calls == []


def test_early_close_session_ends_at_its_entry_cutoff(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 27, 8, 45))
    tick(ap)
    starts = [a for n, a in api.calls if n == "start_paper_session"]
    assert starts and starts[0]["duration_minutes"] == 180  # 09:30 -> 12:30 ET, cutoff 30 min before 13:00


def test_entries_halt_stands_down_until_operator_clears_it(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    api.halts = [{"halt_id": "halt_1", "reason": "operator pause", "source": "operator", "created_at": None}]
    tick(ap)
    assert api.count("start_paper_session") == 0
    sd = ap.store.get("standdown")
    assert sd and not sd["kill_switch"]
    assert ap.store.events(kind="STANDDOWN")
    clock.advance(minutes=6)
    api.halts = []  # the operator cleared it (the autopilot never can)
    tick(ap)
    assert ap.store.get("standdown") is None
    assert ap.store.events(kind="STANDDOWN_CLEARED")
    assert api.count("start_paper_session") == 1


def test_kill_switch_stands_down_and_blocks_the_day(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    api.kill_switch = True
    tick(ap)
    assert ap.store.get("standdown")["kill_switch"] is True
    clock.set(et(2026, 11, 25, 9, 40))  # the window closes while still killed
    tick(ap)
    assert api.count("start_paper_session") == 0
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "BLOCKED"
    assert any("kill switch" in e["message"] for e in ap.store.events(kind="ALERT"))
    # Clearing the kill switch after the window never trades late.
    api.kill_switch = False
    clock.set(et(2026, 11, 25, 10, 0))
    tick(ap)
    assert api.count("start_paper_session") == 0


def test_failed_health_check_blocks_start_alerts_once_and_rechecks_bounded(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 40))
    api.health_overrides = {"reconciliation": "DEGRADED"}
    tick(ap)
    assert api.count("start_paper_session") == 0
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "PENDING"
    assert any("reconciliation" in r for r in job["detail"]["blocked_reasons"])
    alerts = ap.store.events(kind="ALERT")
    assert len(alerts) == 1
    n = api.count("get_system_health")
    clock.advance(minutes=2)  # inside the re-check interval: no new calls
    tick(ap)
    assert api.count("get_system_health") == n
    clock.advance(minutes=4)
    tick(ap)  # same failure: re-checked, but no duplicate alert
    assert api.count("get_system_health") == n + 1
    assert len(ap.store.events(kind="ALERT")) == 1
    api.health_overrides = {}
    clock.advance(minutes=6)
    tick(ap)
    assert api.count("start_paper_session") == 1


def test_feed_disconnected_and_entries_locked_block(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path / "f", et(2026, 11, 25, 8, 45))
    api.feed_status = "DISCONNECTED"
    tick(ap)
    assert api.count("start_paper_session") == 0
    ap2, _, api2, _, _ = make(tmp_path / "e", et(2026, 11, 25, 8, 45))
    api2.entries_locked, api2.lock_reasons = True, ["no CLEAN broker reconciliation has completed yet"]
    tick(ap2)
    assert api2.count("start_paper_session") == 0
    ap3, _, api3, _, _ = make(tmp_path / "s", et(2026, 11, 25, 8, 45))
    api3.feed_status = "STALE"  # tolerated before the open (warning), the risk gate re-checks every cycle
    tick(ap3)
    assert api3.count("start_paper_session") == 1


def test_api_unreachable_blocks_without_start(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    api.unreachable = True
    tick(ap)
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "PENDING" and api.count("start_paper_session") == 0


def test_unknown_outcome_is_reconciled_by_inspection_never_blind_resend(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    api.start_error = ToolError(504, "timed out; the outcome is UNKNOWN")
    api.start_side_effect_before_error = True  # the session was in fact created
    tick(ap)
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "UNKNOWN"
    api.start_error = None
    clock.advance(minutes=1)
    tick(ap)
    assert api.count("start_paper_session") == 1  # inspected and adopted, not resubmitted
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["detail"]["outcome"] == "ADOPTED"


def test_crash_resume_after_start_request(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    # Simulate a crash right after the request reached the API: job RUNNING in the store, session exists.
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "RUNNING", attempt=True)
    api.active = _session("sess_9", "SCHEDULED")
    ap2, *_ = make(tmp_path, et(2026, 11, 25, 8, 50))  # a fresh process on the same store
    ap2.backend = ap.backend
    ap2.clock = clock
    clock.advance(minutes=5)
    asyncio.run(ap2.run(asyncio.Event(), once=True))
    assert ap2.store.events(kind="RESUME")
    assert api.count("start_paper_session") == 0
    job = ap2.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["detail"]["outcome"] == "ADOPTED" and job["detail"]["session_id"] == "sess_9"


def test_crash_resume_without_session_retries_after_inspection(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "RUNNING", attempt=True)
    tick(ap)
    names = [n for n, _ in api.calls]
    assert names.index("get_session_status") < names.index("start_paper_session")
    assert api.count("start_paper_session") == 1


def test_missed_window_never_trades_late(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 25, 9, 40))
    tick(ap)
    assert api.count("start_paper_session") == 0
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "MISSED"
    assert ap.store.events(kind="ALERT")


def test_sleep_gap_is_recorded_and_the_window_is_missed(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 7, 0), research=False)
    asyncio.run(ap.run(asyncio.Event(), once=True))  # nothing due at 07:00
    assert api.count("start_paper_session") == 0
    clock.set(et(2026, 11, 25, 11, 0))  # the Mac slept through pre-open (monotonic clock stopped)
    asyncio.run(ap.run(asyncio.Event(), once=True))
    assert ap.store.events(kind="SLEEP_GAP")
    assert api.count("start_paper_session") == 0
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "MISSED"


def test_409_refusal_is_bounded(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 35))
    api.start_error = ToolError(409, "the regular-hours session window could not be established")
    for _ in range(6):
        tick(ap)
        clock.advance(minutes=6)
    assert api.count("start_paper_session") == 3
    job = ap.store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "BLOCKED"


def test_sleep_during_preopen_checks_cannot_start_late(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 9, 34))
    call = api.call

    async def delayed(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "get_feed_quality":
            clock.advance(minutes=2)
        return await call(name, args)

    monkeypatch.setattr(api, "call", delayed)
    tick(ap)
    assert api.count("start_paper_session") == 0
    assert (ap.store.job("2026-11-25", PREOPEN) or {})["status"] == "MISSED"


def test_completed_operator_session_prevents_second_start(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 25, 9, 33))
    api.recent = [_session("sess_op", "STOPPED", requested_by="operator", started_at="2026-11-25T14:30:00Z")]
    tick(ap)
    assert api.count("start_paper_session") == 0
    assert (ap.store.job("2026-11-25", PREOPEN) or {})["detail"]["outcome"] == "ALREADY_RAN"


def test_unknown_start_keeps_its_state_when_inspection_fails(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "UNKNOWN", attempt=True)
    api.unreachable = True
    tick(ap)
    assert (ap.store.job("2026-11-25", PREOPEN) or {})["status"] == "UNKNOWN"


def test_uncertain_start_is_rechecked_after_health_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 8, 45))
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "UNKNOWN", attempt=True)
    call = api.call

    async def delayed(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "get_feed_quality":
            api.recent = [_session("sess_finished", "STOPPED")]
        return await call(name, args)

    monkeypatch.setattr(api, "call", delayed)
    tick(ap)
    assert api.count("start_paper_session") == 0
    assert (ap.store.job("2026-11-25", PREOPEN) or {})["detail"]["session_id"] == "sess_finished"


def test_store_jobs_are_idempotent_and_events_append_only(tmp_path: Path) -> None:
    import sqlite3

    import pytest

    store = Store(tmp_path / "s.sqlite3")
    now = et(2026, 11, 25, 8, 0)
    store.set_job(now, "2026-11-25", PREOPEN, "DONE", detail={"outcome": "SCHEDULED"})
    store.set_job(now, "2026-11-25", PREOPEN, "RUNNING", detail={"outcome": "X"})
    job = store.job("2026-11-25", PREOPEN)
    assert job is not None and job["status"] == "DONE" and job["detail"]["outcome"] == "SCHEDULED"
    store.event(now, "TICK", "x")
    with sqlite3.connect(store.path) as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE events SET message='y'")
    with sqlite3.connect(store.path) as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("DELETE FROM events")
    with sqlite3.connect(store.path) as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("INSERT OR REPLACE INTO events SELECT seq, ts, day, kind, job, 'replacement', data_json FROM events")


def test_store_round_trips_large_json_and_literal_path(tmp_path: Path) -> None:
    store = Store(tmp_path / "state#archive%20?.sqlite3")
    now = et(2026, 11, 25, 8)
    detail = {"provider_tail": "é" * 60_000, "outcome": "test"}
    store.event(now, "RESULT", "long result", data=detail)
    store.set_job(now, "2026-11-25", PREOPEN, "DONE", detail=detail)
    reader = Store(store.path, readonly=True)
    assert reader.events()[0]["data"] == detail
    assert (reader.job("2026-11-25", PREOPEN) or {})["detail"] == detail
