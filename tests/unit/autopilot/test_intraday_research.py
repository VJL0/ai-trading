"""Intraday observation and the after-close research sequence (fake clock, fake API, fake subprocess runner)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from services.autopilot import supervisor
from services.autopilot.research import StepOutcome
from services.autopilot.supervisor import CREW, DAILY, HOLDOUT, PREOPEN, REPORT, SHADOW

from .fakes import _session, et, make

MUTATING = {"start_paper_session", "stop_session", "cancel_paper_order", "propose_paper_trade", "request_halt",
            "run_quant_cycle", "schedule_session", "cancel_schedule", "run_research_experiment"}


def tick(ap: object) -> list[dict[str, object]]:
    return asyncio.run(ap.tick())  # type: ignore[attr-defined]


def test_intraday_one_bounded_check_per_interval_and_no_actions(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 10, 0))
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "DONE", detail={"outcome": "STARTED", "session_id": "sess_1"})
    api.active = _session("sess_1", "RUNNING", started_at="2026-11-25T14:30:00Z")
    for _ in range(30):  # 30 ticks a minute apart -> two checks (10:00, 10:15), never a storm
        tick(ap)
        clock.advance(minutes=1)
    assert api.count("get_session_status") == 2
    assert api.count("get_risk_status") == 2
    assert not [n for n, _ in api.calls if n in MUTATING]


def test_intraday_halt_stands_down_and_never_touches_orders(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 11, 0))
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "DONE", detail={"outcome": "STARTED"})
    api.kill_switch = True
    tick(ap)
    assert ap.store.get("standdown")["kill_switch"] is True
    assert not [n for n, _ in api.calls if n in MUTATING]
    # After the close, still killed: no research either.
    clock.set(et(2026, 11, 25, 17, 0))
    tick(ap)
    assert ap.store.job("2026-11-25", DAILY) is None
    api.kill_switch = False
    clock.advance(minutes=16)
    tick(ap)  # read-only stand-down check sees the operator cleared it
    assert ap.store.get("standdown") is None


def test_after_close_checks_for_new_halts_before_running_research(tmp_path: Path) -> None:
    ap, clock, api, runner, daily = make(tmp_path, et(2026, 11, 25, 17))
    api.halts = [{"halt_id": "halt_after_close", "source": "operator", "reason": "operator pause"}]
    tick(ap)
    assert daily.runs == 0 and runner.calls == []
    assert ap.store.get("standdown")
    api.halts = []
    clock.advance(minutes=16)
    tick(ap)
    assert ap.store.get("standdown") is None
    tick(ap)
    assert daily.runs == 1


def test_research_waits_when_risk_status_is_unavailable(tmp_path: Path) -> None:
    ap, _, api, runner, daily = make(tmp_path, et(2026, 11, 25, 17))
    api.unreachable = True
    tick(ap)
    assert daily.runs == 0 and runner.calls == []


def test_new_halt_cancels_running_research_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 25, 17))
    ap.store.set_job(clock.now(), "2026-11-25", DAILY, "DONE")
    monkeypatch.setattr(supervisor, "RESEARCH_RISK_POLL_S", 0.01)
    cancelled = False

    async def long_child(argv: Sequence[str], cwd: Path, timeout_s: float) -> StepOutcome:
        nonlocal cancelled
        api.kill_switch = True
        try:
            await asyncio.Event().wait()
        finally:
            cancelled = True
        return StepOutcome(tuple(argv), 0, 0)

    ap.runner = long_child
    asyncio.run(asyncio.wait_for(ap.tick(), timeout=1))
    assert cancelled and ap.store.get("standdown")["kill_switch"] is True


def test_daily_runner_recalculates_budget_between_children(tmp_path: Path) -> None:
    from datetime import date

    from services.autopilot.daycal import trading_day

    ap, clock, _, runner, _ = make(tmp_path, et(2026, 11, 27, 7), preopen_lead_min=120)
    td = trading_day(date(2026, 11, 25))
    assert td is not None
    run = ap._daily_runner(td)
    asyncio.run(run(("fetch-delta",), tmp_path, 7200))
    clock.set(et(2026, 11, 27, 7, 26))
    outcome = asyncio.run(run(("scan",), tmp_path, 7200))
    assert outcome.returncode is None
    assert runner.commands() == ["fetch-delta"]


def test_report_keeps_simulation_label_when_api_reads_fail(tmp_path: Path) -> None:
    ap, _, api, _, _ = make(tmp_path, et(2026, 11, 25, 10), research=False)
    ap.simulated_feed = True
    api.unreachable = True
    tick(ap)
    report = json.loads((ap.cfg.report_dir / "2026-11-25.json").read_text())
    assert report["simulated"] is True
    assert "SIMULATED" in (ap.cfg.report_dir / "2026-11-25.md").read_text()


def test_report_uses_safe_temporary_file(tmp_path: Path) -> None:
    ap, _, _, _, _ = make(tmp_path, et(2026, 11, 25, 10), research=False)
    target = tmp_path / "other-file"
    target.write_text("preserve")
    ap.cfg.report_dir.mkdir(parents=True)
    (ap.cfg.report_dir / "2026-11-25.json.tmp").symlink_to(target)
    tick(ap)
    assert target.read_text() == "preserve"
    assert (ap.cfg.report_dir / "2026-11-25.json").is_file()


def test_historical_report_does_not_use_todays_performance(tmp_path: Path) -> None:
    ap, clock, api, _, _ = make(tmp_path, et(2026, 11, 27, 8, 15))
    ap.store.set_job(et(2026, 11, 25, 9), "2026-11-25", PREOPEN, "DONE")
    tick(ap)
    assert api.count("get_performance") == 0
    report = json.loads((ap.cfg.report_dir / "2026-11-25.json").read_text())
    assert report["generated_at"] == clock.now().isoformat()
    assert report["pnl"]["daily_pnl"] is None


def test_research_sequence_runs_in_order_one_step_per_tick(tmp_path: Path) -> None:
    ap, clock, api, runner, daily = make(tmp_path, et(2026, 11, 25, 17, 0), crew_ticks=2)
    ap.store.set_job(clock.now(), "2026-11-25", PREOPEN, "DONE", detail={"outcome": "SCHEDULED", "session_id": "s1"})
    api.families = [{"family": "J_volman", "passed_dev_gate": True, "registry_state": "DEV_PASSED"},
                    {"family": "K_rank", "passed_dev_gate": False, "registry_state": "REJECTED"}]
    runner.results["holdout"] = StepOutcome((), 0, 5.0, json.dumps({"strategy_id": "J_volman:abc", "passed": False,
                                                                     "fail_reasons": ["sharpe"]}))
    shadow_dir = tmp_path / "experiments" / "quantlab"
    shadow_dir.mkdir(parents=True)
    (shadow_dir / "latest_shadow.json").write_text(json.dumps({"equity": {"promotion_checks": [{"sid": "x"}]}}))
    for _ in range(8):
        tick(ap)
        clock.advance(minutes=1)
    assert daily.runs == 1
    assert runner.commands() == ["crew", "crew", "shadow", "holdout"]
    holdout_argv = runner.calls[3][0]
    assert holdout_argv == ("-m", "services.quantlab.cli", "holdout", "--family", "J_volman")
    for job in (DAILY, CREW, SHADOW, HOLDOUT, REPORT):
        row = ap.store.job("2026-11-25", job)
        assert row is not None and row["status"] == "DONE", job
    rep = json.loads((tmp_path / "ap" / "reports" / "2026-11-25.json").read_text())
    assert rep["status"] == "FINAL" and rep["simulated"] is True
    assert rep["crew_runs"]["ticks_done"] == 2
    assert rep["promotions"]["holdout"]["family"] == "J_volman"
    assert rep["promotions"]["shadow_promotion_checks"] == [{"sid": "x"}]
    assert rep["rejections_by_check"] == {"spread": 2}
    assert rep["data_provenance"]["execution_labels"] == ["SIMULATED"]
    assert "SIMULATED" in (tmp_path / "ap" / "reports" / "2026-11-25.md").read_text()


def test_holdout_look_is_taken_at_most_once_per_family(tmp_path: Path) -> None:
    ap, clock, api, runner, _ = make(tmp_path, et(2026, 11, 24, 17, 0), crew_ticks=0)
    api.families = [{"family": "J_volman", "passed_dev_gate": True, "registry_state": "DEV_PASSED"}]
    for _ in range(6):
        tick(ap)
        clock.advance(minutes=1)
    # The registry still says DEV_PASSED next evening (e.g. the look crashed): the autopilot never retries it.
    clock.set(et(2026, 11, 25, 17, 0))
    for _ in range(6):
        tick(ap)
        clock.advance(minutes=1)
    assert [c for c in runner.commands() if c == "holdout"] == ["holdout"]
    row = ap.store.job("2026-11-25", HOLDOUT)
    assert row is not None and "no DEV_PASSED family" in row["detail"]["note"]


def test_interrupted_holdout_is_never_replaced_with_a_second_look_that_day(tmp_path: Path) -> None:
    ap, clock, api, runner, _ = make(tmp_path, et(2026, 11, 25, 17))
    for job in (DAILY, CREW, SHADOW):
        ap.store.set_job(clock.now(), "2026-11-25", job, "DONE")
    ap.store.put(clock.now(), "holdout_attempts", {"J_volman": {"day": "2026-11-25", "at": clock.now().isoformat()}})
    api.families = [{"family": "K_rank", "passed_dev_gate": True, "registry_state": "DEV_PASSED"}]
    tick(ap)
    assert runner.commands() == []
    row = ap.store.job("2026-11-25", HOLDOUT)
    assert row is not None and row["status"] == "FAILED" and row["detail"]["family"] == "J_volman"


def test_observed_simulated_feed_prevents_download_and_holdout_after_restart(tmp_path: Path) -> None:
    ap, _, api, runner, daily = make(tmp_path, et(2026, 11, 25, 17), crew_ticks=0)
    api.feed_mode = "SIMULATED"
    api.families = [{"family": "J_volman", "passed_dev_gate": True, "registry_state": "DEV_PASSED"}]
    for _ in range(5):
        tick(ap)
    assert runner.commands() == [] and daily.runs == 0
    assert (ap.store.job("2026-11-25", HOLDOUT) or {})["status"] == "SKIPPED"


def test_failed_crew_ticks_retry_hourly_and_never_report_success(tmp_path: Path) -> None:
    ap, clock, _, runner, _ = make(tmp_path, et(2026, 11, 25, 17), crew_ticks=1)
    ap.store.set_job(clock.now(), "2026-11-25", DAILY, "DONE")
    runner.results["crew"] = StepOutcome((), None, 1, "timeout")
    tick(ap)
    tick(ap)
    assert runner.commands() == ["crew"]
    for _ in range(3):
        clock.advance(minutes=61)
        tick(ap)
    job = ap.store.job("2026-11-25", CREW)
    assert runner.commands().count("crew") == 3
    assert job is not None and job["status"] == "FAILED" and job["detail"].get("ticks_done", 0) == 0


def test_uncaught_daily_failure_consumes_a_retry(tmp_path: Path) -> None:
    ap, clock, _, _, _ = make(tmp_path, et(2026, 11, 25, 17))

    def broken_factory(*args: object, **kwargs: object) -> object:
        raise OSError("daily job initialization failed")

    ap.daily_job_factory = broken_factory
    tick(ap)
    tick(ap)
    assert (ap.store.job("2026-11-25", DAILY) or {})["attempts"] == 1
    for _ in range(2):
        clock.advance(minutes=61)
        tick(ap)
    assert (ap.store.job("2026-11-25", DAILY) or {})["status"] == "FAILED"


def test_api_owned_or_simulated_pipeline_is_skipped(tmp_path: Path) -> None:
    ap, _, _, runner, daily = make(tmp_path / "a", et(2026, 11, 25, 17, 0), crew_ticks=1)
    ap.api_owns_quantlab = True
    for _ in range(5):
        tick(ap)
    assert daily.runs == 0
    assert ap.store.job("2026-11-25", DAILY)["status"] == "SKIPPED"  # type: ignore[index]
    assert ap.store.job("2026-11-25", SHADOW)["status"] == "SKIPPED"  # type: ignore[index]
    assert "shadow" not in runner.commands()

    ap2, _, _, runner2, daily2 = make(tmp_path / "b", et(2026, 11, 25, 17, 0), crew_ticks=1)
    ap2.simulated_feed = True
    for _ in range(5):
        tick(ap2)
    assert daily2.runs == 0 and "holdout" not in runner2.commands()
    assert runner2.commands() == ["crew"]  # the crew still researches on development data


def test_failed_daily_retries_bounded_then_fails(tmp_path: Path) -> None:
    ap, clock, _, runner, daily = make(tmp_path, et(2026, 11, 25, 17, 0), crew_ticks=0)
    daily.ok = False
    for _ in range(12):
        tick(ap)
        clock.advance(minutes=61)
        if clock.now() >= et(2026, 11, 26, 6, 0):
            break
    assert daily.runs == 3
    assert ap.store.job("2026-11-25", DAILY)["status"] == "FAILED"  # type: ignore[index]
    assert ap.store.job("2026-11-25", SHADOW)["status"] == "SKIPPED"  # type: ignore[index]
    assert "shadow" not in runner.commands()


def test_missed_research_catches_up_once_and_older_day_is_superseded(tmp_path: Path) -> None:
    ap, clock, _, _, daily = make(tmp_path, et(2026, 11, 24, 17, 0), crew_ticks=0, holdout=False)
    daily.ok = False
    tick(ap)  # Tuesday's daily fails once, then the Mac sleeps through the rest of the window
    clock.set(et(2026, 11, 25, 17, 0))
    daily.ok = True
    for _ in range(6):
        tick(ap)
        clock.advance(minutes=1)
    assert ap.store.job("2026-11-24", DAILY)["status"] == "SUPERSEDED"  # type: ignore[index]
    assert ap.store.job("2026-11-25", DAILY)["status"] == "DONE"  # type: ignore[index]
    assert daily.runs == 2  # one failed attempt for Tuesday, one catch-up for Wednesday; no per-day replay
    assert (tmp_path / "ap" / "reports" / "2026-11-24.json").is_file()


def test_research_day_with_no_steps_started_is_superseded(tmp_path: Path) -> None:
    ap, clock, _, _, daily = make(tmp_path, et(2026, 11, 25, 17), crew_ticks=0)
    ap.store.set_job(et(2026, 11, 24, 9), "2026-11-24", PREOPEN, "DONE")
    tick(ap)
    assert (ap.store.job("2026-11-24", DAILY) or {})["status"] == "SUPERSEDED"
    assert daily.runs == 1
    assert (ap.store.job("2026-11-25", DAILY) or {})["status"] == "DONE"
    assert clock.now() == et(2026, 11, 25, 17)


def test_research_children_never_run_into_the_next_preopen(tmp_path: Path) -> None:
    ap, _, _, runner, _ = make(tmp_path, et(2026, 11, 27, 6, 55), crew_ticks=1, crew_timeout_s=3600)
    ap.store.set_job(et(2026, 11, 25, 17, 0), "2026-11-25", DAILY, "DONE")
    tick(ap)
    (_, timeout), = runner.calls
    # next pre-open 08:30 ET minus a 5-minute margin -> at most 90 minutes from 06:55
    assert timeout <= 90 * 60
    ap2, _, _, _, _ = make(tmp_path / "late", et(2026, 11, 27, 8, 0), crew_ticks=1)
    ap2.store.set_job(et(2026, 11, 25, 17, 0), "2026-11-25", DAILY, "DONE")
    # 08:00 is outside the quiet window: no research, only Wednesday's final report (its window ended)
    assert [a.kind for a in ap2.decide(et(2026, 11, 27, 8, 0)).actions] == ["REPORT"]
    plan = ap2.decide(et(2026, 11, 27, 7, 20))  # 65 min of budget left -> allowed, bounded
    assert [a.kind for a in plan.actions] == ["CREW"]
    ap2.cfg = ap2.cfg.__class__(**{**ap2.cfg.__dict__, "preopen_lead_min": 85})  # pre-open 08:05 -> 5 min budget
    assert not [a for a in ap2.decide(et(2026, 11, 27, 7, 55)).actions if a.kind == "CREW"]


def test_crew_locked_store_is_not_retried_in_a_loop(tmp_path: Path) -> None:
    ap, clock, _, runner, _ = make(tmp_path, et(2026, 11, 25, 17, 0), crew_ticks=3, holdout=False)
    runner.results["crew"] = StepOutcome((), 4, 0.1, '{"error": "crew store locked"}')
    for _ in range(6):
        tick(ap)
        clock.advance(minutes=1)
    assert runner.commands().count("crew") == 1
    assert ap.store.job("2026-11-25", CREW)["status"] == "DONE"  # type: ignore[index]
