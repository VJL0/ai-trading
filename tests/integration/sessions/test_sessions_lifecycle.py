"""Persisted sessions through real runtime/tool dispatch; no credentials or external services."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from packages.contracts.enums import ExecutionLabel
from services.sessions.errors import SessionError
from services.sessions.manager import SessionManager
from services.tools.handlers import dispatch


async def test_agent_session_completes_and_reports(rt: Any) -> None:
    await rt.sessions.recover(clean=True)
    started = await dispatch(rt, "start_paper_session", {
        "duration_minutes": 3, "cadence_minutes": 1, "symbols": ["SPY"], "use_ai": False,
    }, actor="codex-test")
    sid = started["session"]["session_id"]
    for _ in range(4):
        await rt.sessions.tick()
        rt.sim_clock.step(60)
        rt._on_sim_tick(rt.now())
    report = await dispatch(rt, "get_session_report", {"session_id": sid}, actor="codex-test")
    assert report["session"]["state"] == "COMPLETED"
    assert report["cycles"] == 3
    assert sum(report["decisions"].values()) == 3
    assert report["api_cost_usd"] == "0"
    assert report["execution_label"] == "SIMULATED"
    assert "SIMULATED" in report["narrative"]
    assert rt.db.query("SELECT * FROM activity_events WHERE session_id=?", (sid,))
    assert rt.db.query("SELECT * FROM quant_signals")


@pytest.mark.parametrize("clean", [True, False])
async def test_restart_requires_clean_reconciliation(rt: Any, clean: bool) -> None:
    s = rt.sessions.start(10, symbols=["SPY"], use_ai=False, requested_by="test")
    replacement = SessionManager(rt)
    result = await replacement.recover(clean=clean)
    assert s["session_id"] in result["resumed" if clean else "interrupted"]
    assert replacement.get(s["session_id"])["state"] == ("RUNNING" if clean else "INTERRUPTED")


async def test_recovery_after_window_is_terminal(rt: Any) -> None:
    s = rt.sessions.start(1, use_ai=False, requested_by="test")
    rt.sim_clock.step(120)
    replacement = SessionManager(rt)
    await replacement.recover(clean=True)
    assert replacement.get(s["session_id"])["state"] == "INTERRUPTED"


async def test_stop_in_flight_finishes_cycle_before_stopping(rt: Any, monkeypatch: Any) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    async def cycle(**_: Any) -> dict[str, Any]:
        entered.set()
        await release.wait()
        return {"outcome": "NO_TRADE", "reason": "test hold"}
    monkeypatch.setattr(rt, "run_cycle", cycle)
    await rt.sessions.recover(clean=True)
    s = rt.sessions.start(10, use_ai=False, requested_by="test")
    task = asyncio.create_task(rt.sessions.tick())
    await entered.wait()
    assert rt.sessions.stop(reason="stop", requested_by="test")["outcome"] == "STOPPING"
    release.set()
    await task
    assert rt.sessions.get(s["session_id"])["state"] == "STOPPED"
    assert len(rt.sessions.cycles(s["session_id"])) == 1


async def test_sleep_skips_slots_and_failures_end_session(rt: Any, monkeypatch: Any) -> None:
    calls = []
    async def cycle(**_: Any) -> dict[str, Any]:
        calls.append(rt.now())
        raise RuntimeError("injected cycle failure")
    monkeypatch.setattr(rt, "run_cycle", cycle)
    await rt.sessions.recover(clean=True)
    s = rt.sessions.start(30, cadence_minutes=1, use_ai=False, requested_by="test")
    for _ in range(3):
        await rt.sessions.tick()
        rt.sim_clock.step(300)
    await rt.sessions.tick()
    assert len(calls) == 3
    assert rt.sessions.get(s["session_id"])["state"] == "FAILED"


async def test_single_active_session_and_validation(rt: Any) -> None:
    rt.sessions.start(10, use_ai=False, requested_by="test")
    with pytest.raises(SessionError, match="already active"):
        rt.sessions.start(10, requested_by="test")
    with pytest.raises(SessionError, match="allowlist"):
        rt.sessions.start(10, symbols=["BAD"], requested_by="test")


async def test_plain_start_queues_once_and_runs_only_in_the_next_regular_window(rt: Any, monkeypatch: Any) -> None:
    now = datetime.fromisoformat("2026-11-26T14:00:00+00:00")  # Thanksgiving, XNYS closed
    opening = datetime.fromisoformat("2026-11-27T14:30:00+00:00")
    monkeypatch.setattr(rt, "now", lambda: now)
    monkeypatch.setattr(rt, "policy", rt.policy.model_copy(update={"symbol_allowlist": ("SPY",)}))
    calls = []

    async def cycle(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"outcome": "NO_TRADE", "reason": "test"}

    monkeypatch.setattr(rt, "run_cycle", cycle)
    await rt.sessions.recover(clean=True)
    result = await dispatch(rt, "start_paper_session", {}, actor="test")
    assert result["outcome"] == "SCHEDULED"
    scheduled = result["session"]
    assert scheduled["state"] == "SCHEDULED" and scheduled["started_at"] is None
    assert scheduled["next_cycle_at"] == "2026-11-27T14:30:00Z"
    assert scheduled["symbols"] == ["SPY"]
    sid = scheduled["session_id"]
    with pytest.raises(SessionError, match="already active"):
        rt.sessions.start(requested_by="test")

    rt.sessions = SessionManager(rt)
    assert sid in (await rt.sessions.recover(clean=True))["resumed"]
    await rt.sessions.tick()
    assert not calls and rt.sessions.get(sid)["state"] == "SCHEDULED"
    now = opening
    await rt.sessions.tick()
    assert len(calls) == 1 and calls[0]["use_ai"] is False
    assert rt.sessions.get(sid)["state"] == "RUNNING"


async def test_queued_session_cannot_move_to_a_different_execution_environment(rt: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(rt, "now", lambda: datetime.fromisoformat("2026-11-26T14:00:00+00:00"))
    result = await dispatch(rt, "start_paper_session", {}, actor="test")
    assert result["outcome"] == "SCHEDULED"
    sid = result["session"]["session_id"]
    monkeypatch.setattr(rt, "execution_label", ExecutionLabel.PAPER)
    replacement = SessionManager(rt)
    assert sid in (await replacement.recover(clean=True))["interrupted"]
    assert replacement.get(sid)["state"] == "INTERRUPTED"


async def test_schedule_once_and_cancel(rt: Any) -> None:
    from services.sessions.calendar import et_date, et_hhmm, weekday_code
    now = rt.now()
    due = now + timedelta(minutes=1)
    sch = rt.sessions.add_schedule([weekday_code(et_date(due))], et_hhmm(due), 2, ["SPY"], 1, False, "test")
    await rt.sessions.recover(clean=True)
    rt.sim_clock.step(60)
    await rt.sessions.tick()
    await rt.sessions.tick()
    runs = rt.sessions.schedule_runs(sch["schedule_id"])
    assert len(runs) == 1 and runs[0]["status"] == "STARTED"
    rt.sessions.cancel_schedule(sch["schedule_id"], requested_by="test")
    assert not rt.sessions.get_schedule(sch["schedule_id"])["enabled"]


async def test_missed_schedule_never_runs_late(rt: Any) -> None:
    from services.sessions.calendar import et_date, et_hhmm, weekday_code
    due = rt.now() + timedelta(minutes=1)
    sch = rt.sessions.add_schedule([weekday_code(et_date(due))], et_hhmm(due), 1, ["SPY"], 1, False, "test")
    rt.sim_clock.step(180)
    await rt.sessions.recover(clean=True)
    await rt.sessions.tick()
    assert rt.sessions.active() is None
    assert rt.sessions.schedule_runs(sch["schedule_id"])[0]["status"] == "MISSED"


async def test_ai_failure_falls_back_then_backs_off(rt: Any, monkeypatch: Any) -> None:
    calls = []
    async def cycle(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["use_ai"])
        if kwargs["use_ai"]:
            rt.activity.begin_cycle("cz_ai_failure", "cz_ai_failure", kwargs["session_id"])
            rt.activity.stage("GPT_REVIEW", "error", detail="injected usage limit")
            rt.activity.end_cycle()
            return {"outcome": "ERROR", "causal_id": "cz_ai_failure", "reason": "usage limit"}
        return {"outcome": "NO_TRADE", "reason": "no net edge"}
    monkeypatch.setattr(rt, "run_cycle", cycle)
    await rt.sessions.recover(clean=True)
    s = rt.sessions.start(10, cadence_minutes=1, use_ai=True, requested_by="test")
    await rt.sessions.tick()
    rt.sim_clock.step(60)
    await rt.sessions.tick()
    assert calls == [True, False, False]
    assert rt.sessions.get(s["session_id"])["ai_suspended_until"] is not None
    assert len(rt.sessions.cycles(s["session_id"])) == 3


async def test_concurrent_agent_entries_cannot_exceed_one_position(rt: Any) -> None:
    from tests.integration.tools.test_tools_real_runtime import _proposal

    spy = await dispatch(rt, "get_market_snapshot", {"symbol": "SPY"}, actor="concurrent")
    qqq = await dispatch(rt, "get_market_snapshot", {"symbol": "QQQ"}, actor="concurrent")
    results = await asyncio.gather(*(
        dispatch(rt, "propose_paper_trade", _proposal(snap), actor="concurrent") for snap in (spy, qqq)
    ))
    assert sorted(r["outcome"] for r in results) == ["REJECTED", "SUBMITTED"]


async def test_completed_report_keeps_its_cutoff_after_replay_clock_restart(rt: Any) -> None:
    before = rt.now()
    s = rt.sessions.start(1, use_ai=False, requested_by="test")
    rt.sim_clock.step(60)
    await rt.sessions.recover(clean=True)
    await rt.sessions.tick()
    first = rt.sessions.report(s["session_id"])
    rt.sim_clock.set(before - timedelta(minutes=5))
    historical = rt.sessions.report(s["session_id"])
    assert historical["performance"]["asof"] == first["performance"]["asof"]
    assert historical["narrative"] == first["narrative"]
