from __future__ import annotations

import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from services.agents.contracts import Invocation, Provider, RunOutcome, RunResult, TaskKind, TaskState
from services.agents.store import TaskRow

from .test_crew_support import T0, Clock, FakeRunner, auto_ok, make_crew, ok


def _only(crew, kind: TaskKind, key: str, payload: dict, **kw) -> TaskRow:  # type: ignore[no-untyped-def]
    tid, _ = crew.store.enqueue(kind, key, payload, now=crew.clock(), priority=99, **kw)
    row = crew.store.get_task(tid)
    assert row is not None
    return row


def _no_planning(crew) -> None:  # type: ignore[no-untyped-def]
    crew.plan = lambda now: []  # isolate scheduler behaviour from the planner


# --------------------------------------------------------------------------------------- provider choice
def test_provider_choice_cooldown_cap_exclude_and_balance(tmp_path: Path) -> None:
    crew, _, _, clock = make_crew(tmp_path, daily_runs_per_provider=2)
    t = _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum"})
    assert crew.choose_provider(t, clock()) is Provider.CLAUDE  # stable tie-break: config order
    crew.store.start_run("x", Provider.CLAUDE, clock())
    assert crew.choose_provider(t, clock()) is Provider.CODEX  # fewest runs today
    crew.store.start_run("x", Provider.CODEX, clock())
    crew.store.start_run("x", Provider.CODEX, clock())
    assert crew.choose_provider(t, clock()) is Provider.CLAUDE  # codex at daily cap
    crew.store.start_run("x", Provider.CLAUDE, clock())
    assert crew.choose_provider(t, clock()) is None  # both capped
    clock.advance(days=1)
    review = _only(crew, TaskKind.REVIEW, "review:h", {}, exclude_provider=Provider.CLAUDE)
    assert crew.choose_provider(review, clock()) is Provider.CODEX  # never the author
    from services.agents.store import ProviderState

    crew.store.put_provider_state(ProviderState(Provider.CODEX, clock() + timedelta(hours=1), 1), clock())
    assert crew.choose_provider(review, clock()) is None  # other provider cooling down -> the review waits
    assert crew.choose_provider(t, clock()) is Provider.CLAUDE


# ------------------------------------------------------------------------------------------ usage limits
def test_usage_limit_cools_down_requeues_and_other_provider_takes_it(tmp_path: Path) -> None:
    crew, fakes, _, clock = make_crew(tmp_path, [RunResult(RunOutcome.USAGE_LIMIT, detail="limit reached")],
                                      [ok(TaskKind.MARKET_ANALYSIS)])
    _no_planning(crew)
    t = _only(crew, TaskKind.MARKET_ANALYSIS, "ma", {"scan_asof": "2026-11-24"})
    r1 = crew.tick()
    assert r1.ran[0]["provider"] == "claude" and r1.ran[0]["outcome"] == "USAGE_LIMIT"
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.QUEUED and row.attempts == 0
    st = crew.store.provider_state(Provider.CLAUDE)
    assert st.cooldown_until == clock() + timedelta(minutes=60) and st.consecutive_limits == 1
    r2 = crew.tick()
    assert r2.ran[0]["provider"] == "codex" and r2.ran[0]["outcome"] == "OK"
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.DONE
    assert len(fakes[Provider.CLAUDE].calls) == 1


def test_retry_at_is_honoured(tmp_path: Path) -> None:
    clock = Clock()
    reset = T0 + timedelta(hours=3, minutes=7)
    crew, _, _, _ = make_crew(tmp_path, [RunResult(RunOutcome.USAGE_LIMIT, retry_at=reset)], clock=clock,
                              providers=(Provider.CLAUDE,))
    _no_planning(crew)
    _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    crew.tick()
    assert crew.store.provider_state(Provider.CLAUDE).cooldown_until == reset
    clock.advance(hours=3)
    assert crew.tick().idle_reason is not None  # still cooling down: nothing runs, no quota spent
    assert crew.next_wake(clock()) <= 7 * 60 + 1


def test_backoff_doubles_per_consecutive_limit_and_caps(tmp_path: Path) -> None:
    limits = [RunResult(RunOutcome.USAGE_LIMIT) for _ in range(7)]
    crew, _, _, clock = make_crew(tmp_path, limits, providers=(Provider.CLAUDE,))
    _no_planning(crew)
    _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    waits = []
    for _ in range(7):
        start = clock()
        crew.tick()
        st = crew.store.provider_state(Provider.CLAUDE)
        assert st.cooldown_until is not None
        waits.append(st.cooldown_until - start)
        clock.now = st.cooldown_until
    assert waits[:5] == [timedelta(minutes=m) for m in (60, 120, 240, 480, 720)]
    assert waits[5] == waits[6] == timedelta(hours=12)
    t = crew.store.tasks()[0]
    assert t.attempts == 0 and t.state is TaskState.QUEUED


def test_auth_required_disables_until_availability_ok(tmp_path: Path) -> None:
    crew, fakes, _, clock = make_crew(tmp_path, [RunResult(RunOutcome.AUTH_REQUIRED, detail="not logged in"),
                                                 ok(TaskKind.IMPROVEMENT)], providers=(Provider.CLAUDE,))
    _no_planning(crew)
    t = _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    crew.tick()
    st = crew.store.provider_state(Provider.CLAUDE)
    assert st.disabled_reason and "AUTH_REQUIRED" in st.disabled_reason
    fakes[Provider.CLAUDE].available = (False, "not signed in")
    clock.advance(hours=13)  # past any cooldown: still disabled while availability() is not OK
    assert crew.tick().ran == []
    assert crew.store.provider_state(Provider.CLAUDE).disabled_reason
    fakes[Provider.CLAUDE].available = (True, "ok")
    r = crew.tick()
    assert crew.store.provider_state(Provider.CLAUDE).disabled_reason is None
    assert r.ran and r.ran[0]["outcome"] == "OK"
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.DONE and row.attempts == 1


# ------------------------------------------------------------------------------------------ bad outputs
def test_invalid_output_retries_on_other_provider_then_fails(tmp_path: Path) -> None:
    crew, fakes, _, _ = make_crew(tmp_path, [RunResult(RunOutcome.INVALID_OUTPUT, detail="no json")],
                                  [RunResult(RunOutcome.OK, output={"proposals": "not a list"})])
    _no_planning(crew)
    t = _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    r1 = crew.tick()
    assert r1.ran[0]["provider"] == "claude"
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.QUEUED and row.attempts == 1
    r2 = crew.tick()  # retried on the other provider; its OK output fails the crew's own schema check
    assert r2.ran[0]["provider"] == "codex" and r2.ran[0]["outcome"] == "INVALID_OUTPUT"
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.FAILED and row.attempts == 2
    assert row.result and row.result["reason"] == "INVALID_OUTPUT"
    assert crew.tick().ran == []
    assert len(fakes[Provider.CLAUDE].calls) == len(fakes[Provider.CODEX].calls) == 1


def test_provider_exception_maps_to_error(tmp_path: Path) -> None:
    def boom(inv: Invocation) -> RunResult:
        raise RuntimeError("kaboom")

    crew, _, _, _ = make_crew(tmp_path, [boom], providers=(Provider.CLAUDE,))
    _no_planning(crew)
    _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    assert crew.tick().ran[0]["outcome"] == "ERROR"


# ------------------------------------------------------------------------------------------- hypotheses
def _hyp_output(specs: list[dict], mechanism: str = "momentum") -> RunResult:
    return RunResult(RunOutcome.OK, output={
        "mechanism": mechanism, "economic_rationale": "slow diffusion", "why_novel": "residual not raw",
        "falsification": "no edge after costs", "specs": specs, "data_requests": ["earnings dates"]})


def _spec(expr: str, **kw: object) -> dict:
    return {"expression": expr, "universe": "liquid1000", "n": 20, "rebalance": 21, "regime": "none",
            "intent": "test", **kw}


def test_hypothesis_specs_validated_and_ledger_result_stored(tmp_path: Path) -> None:
    runner = FakeRunner({"passed_dev_gate": True, "dsr": 0.97, "wf_oos_sharpe": 0.8, "fail_reasons": []})
    specs = [_spec("rank(resid_mom(126,21))"), _spec("__import__('os').system('rm -rf /')"),
             _spec("rank(mom(9999))"), _spec("rank(mom(126))", n=3)]
    crew, _, _, _ = make_crew(tmp_path, [_hyp_output(specs)], providers=(Provider.CLAUDE,), runner=runner,
                              specs_per_task=5)
    _no_planning(crew)
    t = _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum", "slots": 5})
    res = crew.tick().ran[0]
    assert res["outcome"] == "OK"
    # only the valid canonical spec reached the runner; mom(9999) is caught when evaluated, which the runner owns
    exprs = [c[1]["expression"] for c in runner.calls]
    assert "rank(resid_mom(126, 21))" in exprs
    assert not any("import" in e for e in exprs)
    assert all("intent" not in c[1] for c in runner.calls)
    assert all(c[2] == f"crew-claude-{t.id[:8]}" for c in runner.calls)
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.DONE and row.result is not None
    exp = row.result["experiments"][0]
    assert exp["ledger"]["dsr"] == 0.97 and exp["passed_dev_gate"] is True  # the ledger's numbers
    refusals = {r["status"] for r in row.result["refusals"]}
    assert "INVALID_SPEC" in refusals
    notes = crew.store.notes(task_id=t.id)
    assert {n.kind for n in notes} >= {"hypothesis", "experiment", "refusal", "data_request"}
    assert all(n.evidence_status == "LEDGER_DEV" for n in notes if n.kind == "experiment")
    assert all(n.evidence_status == "AGENT_NOTE" for n in notes if n.kind == "hypothesis")


def test_invalid_dsl_spec_never_reaches_runner(tmp_path: Path) -> None:
    runner = FakeRunner()
    bad = [_spec("open('/etc/passwd').read()"), _spec("mom(20).real"), _spec("rank(mom(20))", rebalance=7),
           _spec("x + 1"), _spec("rank(mom(20, skip=5))")]
    crew, _, _, _ = make_crew(tmp_path, [_hyp_output(bad)], providers=(Provider.CLAUDE,), runner=runner,
                              specs_per_task=5)
    _no_planning(crew)
    t = _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum", "slots": 5})
    crew.tick()
    assert runner.calls == []
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.DONE and row.result is not None
    assert row.result["experiments"] == []
    assert [r["status"] for r in row.result["refusals"]] == ["INVALID_SPEC"] * 5


def test_schema_invalid_spec_field_makes_output_invalid(tmp_path: Path) -> None:
    runner = FakeRunner()
    crew, _, _, _ = make_crew(tmp_path, None, [_hyp_output([_spec("rank(mom(20))", universe="sp500")])],
                              providers=(Provider.CODEX,), runner=runner)
    _no_planning(crew)
    _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum"})
    assert crew.tick().ran[0]["outcome"] == "INVALID_OUTPUT"
    assert runner.calls == []


def test_non_expressible_mechanism_yields_data_requests_only(tmp_path: Path) -> None:
    runner = FakeRunner()
    crew, _, _, _ = make_crew(tmp_path, [_hyp_output([_spec("rank(mom(63))")], mechanism="insider")],
                              providers=(Provider.CLAUDE,), runner=runner)
    _no_planning(crew)
    t = _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "insider"})
    crew.tick()
    assert runner.calls == []
    row = crew.store.get_task(t.id)
    assert row is not None and row.result is not None
    assert [r["status"] for r in row.result["refusals"]] == ["MECHANISM_NOT_EXPRESSIBLE"]
    assert crew.store.notes(kind="data_request", task_id=t.id)


def test_missing_edgar_panel_coverage_creates_data_request_without_experiment(tmp_path: Path) -> None:
    from services.market_data.edgar.parsers import Filing
    from services.market_data.edgar.store import Disclosure, EdgarStore, store_path

    filed = date(2024, 7, 1)
    filing = Filing(7, "8k", "8-K", datetime(2024, 7, 1, 12, tzinfo=UTC), filed, filed, earnings=True)
    EdgarStore(store_path(tmp_path)).ingest([Disclosure(filing, tickers=("TST",))])
    runner = FakeRunner()
    detail = "development EDGAR data unavailable for sue"
    runner.responses = [{"accepted": False, "status": "DATA_UNAVAILABLE", "error": detail}]
    result = _hyp_output([_spec("rank(sue())")], mechanism="earnings")
    crew, providers, _, _ = make_crew(tmp_path, [result], providers=(Provider.CLAUDE,), runner=runner)
    _no_planning(crew)
    task = _only(crew, TaskKind.HYPOTHESIS, "earnings", {"mechanism": "earnings"})

    assert crew.tick().ran[0]["outcome"] == "OK"
    assert runner.calls[0][1]["expression"] == "rank(sue())"
    assert "Relevant DSL functions: sue" in providers[Provider.CLAUDE].calls[0].prompt
    row = crew.store.get_task(task.id)
    assert row is not None and row.result is not None
    assert row.result["experiments"] == []
    assert [r["status"] for r in row.result["refusals"]] == ["DATA_UNAVAILABLE"]
    notes = crew.store.notes(kind="data_request", task_id=task.id)
    assert any(note.body == {"request": detail, "source": "experiment"} and note.mechanism == "earnings" for note in notes)


def test_experiments_stop_when_headroom_runs_out(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.responses = [{"accepted": True, "status": "DONE", "spec": {}, "result": {"passed_dev_gate": False}},
                        {"accepted": False, "error": "agent experiment budget exhausted (6/hour)"}]
    specs = [_spec("rank(mom(126,21))"), _spec("-rank(vol(63))"), _spec("rank(high52())")]
    crew, _, _, _ = make_crew(tmp_path, [_hyp_output(specs)], providers=(Provider.CLAUDE,), runner=runner)
    _no_planning(crew)
    t = _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum", "slots": 3})
    crew.tick()
    assert len(runner.calls) == 2  # third spec not attempted
    row = crew.store.get_task(t.id)
    assert row is not None and row.result is not None
    assert [r["status"] for r in row.result["refusals"]] == ["NO_HEADROOM", "NO_HEADROOM"]


def test_crew_local_headroom_limits_specs(tmp_path: Path) -> None:
    runner = FakeRunner()
    specs = [_spec("rank(mom(126,21))"), _spec("-rank(vol(63))"), _spec("rank(high52())")]
    crew, _, _, _ = make_crew(tmp_path, [_hyp_output(specs)], providers=(Provider.CLAUDE,), runner=runner,
                              per_hour=2)
    _no_planning(crew)
    _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum", "slots": 3})
    crew.tick()
    assert len(runner.calls) == 2


# ---------------------------------------------------------------------------------------------- reviews
def test_review_never_assigned_to_author_and_expires_cancelled(tmp_path: Path) -> None:
    hyp = _hyp_output([_spec("rank(resid_mom(126,21))")])
    # only claude is configured: there is no independent reviewer for claude's hypothesis
    crew, fakes, _, clock = make_crew(tmp_path, [hyp], providers=(Provider.CLAUDE,), review_wait_hours=24)
    fakes[Provider.CLAUDE].default = auto_ok()
    _no_planning(crew)
    tid, _ = crew.store.enqueue(TaskKind.HYPOTHESIS, "hypothesis:momentum:1", {"mechanism": "momentum", "slots": 1},
                                now=clock(), priority=99)
    r = crew.tick()
    assert r.ran[0]["task_id"] == tid and r.ran[0]["provider"] == "claude"
    hyp_row = crew.store.get_task(tid)
    assert hyp_row is not None and hyp_row.author_provider is Provider.CLAUDE  # the crew stamps the author
    del crew.plan  # re-enable the real planner
    clock.advance(minutes=1)
    crew.tick()
    reviews = crew.store.tasks(TaskKind.REVIEW)
    assert len(reviews) == 1 and reviews[0].exclude_provider is Provider.CLAUDE
    assert reviews[0].state is TaskState.QUEUED
    assert not any("verdict" in inv.output_schema.get("properties", {}) for inv in fakes[Provider.CLAUDE].calls)
    clock.advance(hours=25)
    crew.tick()
    rv = crew.store.get_task(reviews[0].id)
    assert rv is not None and rv.state is TaskState.CANCELLED and rv.result == {"reason": "no independent reviewer"}
    assert not any("verdict" in inv.output_schema.get("properties", {}) for inv in fakes[Provider.CLAUDE].calls)


def test_full_cycle_hypothesis_then_cross_provider_review(tmp_path: Path) -> None:
    crew, fakes, _, clock = make_crew(tmp_path, [_hyp_output([_spec("rank(resid_mom(126,21))")])],
                                           [ok(TaskKind.REVIEW, verdict="REFUTED")])
    fakes[Provider.CLAUDE].default = auto_ok()
    fakes[Provider.CODEX].default = auto_ok()
    first = crew.tick()
    assert first.ran[0]["kind"] == "hypothesis" and first.ran[0]["provider"] == "claude"
    clock.advance(minutes=1)
    second = crew.tick()
    assert second.ran[0]["kind"] == "review" and second.ran[0]["provider"] == "codex"
    assert second.ran[0]["result"]["verdict"] == "REFUTED"
    review_note = crew.store.notes(kind="review")[0]
    assert review_note.mechanism == "momentum" and review_note.provider == "codex"


# ------------------------------------------------------------------------------------------- lifecycle
def test_run_once_recovers_running_task_after_crash(tmp_path: Path) -> None:
    crew, _, _, clock = make_crew(tmp_path, [ok(TaskKind.IMPROVEMENT)], providers=(Provider.CLAUDE,))
    _no_planning(crew)
    t = _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    crew.store.claim(t.id, clock())  # a previous process died while running it
    hist = crew.run(once=True, install_signals=False)
    assert hist[0] == {"recovered": [t.id]}
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.DONE and row.attempts == 1


def test_stop_requeues_in_flight_task(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import services.agents.crew as crew_mod

    monkeypatch.setattr(crew_mod, "SHUTDOWN_GRACE_S", 0.05)
    release = threading.Event()

    def slow(inv: Invocation) -> RunResult:
        release.wait(5)
        return RunResult(RunOutcome.OK, output={"proposals": []})

    crew, _, _, _ = make_crew(tmp_path, [slow], providers=(Provider.CLAUDE,))
    _no_planning(crew)
    t = _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    threading.Timer(0.2, crew.stop).start()
    r = crew.tick()
    release.set()
    assert r.ran[0]["outcome"] == "INTERRUPTED"
    row = crew.store.get_task(t.id)
    assert row is not None and row.state is TaskState.QUEUED and row.attempts == 0


def test_queued_hypothesis_waits_for_headroom_and_prompt_uses_remaining_slots(tmp_path: Path) -> None:
    crew, fakes, _, clock = make_crew(tmp_path, [_hyp_output([])], providers=(Provider.CLAUDE,), per_hour=2)
    _no_planning(crew)
    _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum", "slots": 3})
    crew.store.add_note("experiment", {"status": "DONE"}, now=clock() - timedelta(minutes=30))
    crew.store.add_note("experiment", {"status": "DONE"}, now=clock())
    assert crew.tick().ran == [] and fakes[Provider.CLAUDE].calls == []
    clock.advance(minutes=31)
    assert crew.tick().ran[0]["outcome"] == "OK"
    assert "Return at most 1 specs" in fakes[Provider.CLAUDE].calls[0].prompt


def test_hypothesis_for_a_different_mechanism_does_not_reach_quantlab(tmp_path: Path) -> None:
    crew, _, runner, _ = make_crew(tmp_path, [_hyp_output([_spec("rank(mom(126))")], "reversal")])
    _no_planning(crew)
    _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum"})
    assert crew.tick().ran[0]["outcome"] == "INVALID_OUTPUT"
    assert runner.calls == []


def test_credential_like_agent_output_does_not_enter_store_or_tick_result(tmp_path: Path) -> None:
    value = "sk-ant-" + "fixture-not-a-real-key"
    answer = _hyp_output([_spec("rank(mom(126))", intent=value)])
    assert answer.output is not None
    answer.output["economic_rationale"] = value
    crew, _, _, _ = make_crew(tmp_path, [answer], providers=(Provider.CLAUDE,))
    _no_planning(crew)
    task = _only(crew, TaskKind.HYPOTHESIS, "h", {"mechanism": "momentum"})
    assert value not in str(crew.tick().public())
    assert value not in str(crew.store.get_task(task.id))
    assert value not in str(crew.store.notes(task_id=task.id))


def test_auth_probe_not_binary_presence_controls_reenable(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    crew, fakes, _, clock = make_crew(tmp_path, [RunResult(RunOutcome.AUTH_REQUIRED), ok(TaskKind.IMPROVEMENT)],
                                     providers=(Provider.CLAUDE,))
    _no_planning(crew)
    _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    fake = fakes[Provider.CLAUDE]
    monkeypatch.setattr(fake, "auth_available", lambda: (False, "sign-in required"), raising=False)
    crew.tick()
    clock.advance(hours=2)
    assert crew.tick().ran == []
    assert len(fake.calls) == 1
    monkeypatch.setattr(fake, "auth_available", lambda: (True, ""), raising=False)
    assert crew.tick().ran[0]["outcome"] == "OK"


def test_stop_terminates_real_provider_process_group(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os
    import signal
    import subprocess
    import sys
    import time
    from contextlib import suppress

    import services.agents.crew as crew_mod
    from services.agents.providers import ClaudeCodeProvider

    monkeypatch.setattr(crew_mod, "SHUTDOWN_GRACE_S", 1)
    crew, _, _, _ = make_crew(tmp_path, providers=(Provider.CLAUDE,))
    _no_planning(crew)
    task = _only(crew, TaskKind.IMPROVEMENT, "imp", {})
    stub = tmp_path / "sleeping-provider"
    stub.write_text(f"""#!{sys.executable}
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
Path({str(tmp_path / 'pids')!r}).write_text(f"{{os.getpid()}} {{child.pid}}")
time.sleep(30)
""")
    stub.chmod(0o755)
    crew.providers[Provider.CLAUDE] = ClaudeCodeProvider(str(stub))
    tick = threading.Thread(target=crew.tick, daemon=True)
    tick.start()
    pids = tmp_path / "pids"
    deadline = time.monotonic() + 5
    while not pids.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pids.is_file()
    group, child = pids.read_text().split()
    try:
        crew.stop()
        tick.join(timeout=3)
        assert not tick.is_alive()
        for pid in (group, child):
            state = subprocess.run(["ps", "-o", "stat=", "-p", pid], capture_output=True, text=True, timeout=5, check=False)
            assert state.returncode != 0 or state.stdout.strip().startswith("Z")
        row = crew.store.get_task(task.id)
        assert row is not None and row.state is TaskState.QUEUED and row.attempts == 0
    finally:
        with suppress(ProcessLookupError):
            os.killpg(int(group), signal.SIGKILL)


def test_experiment_runner_refuses_conflicting_child_project_root(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from services.agents.crew import default_experiment_runner
    from services.quantlab import views

    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path / "separate-root"))

    def wrong_root(*args: object) -> dict[str, object]:
        return {"accepted": True, "status": "DONE"}

    monkeypatch.setattr(views, "run_agent_experiment", wrong_root)
    assert default_experiment_runner(tmp_path, {}, "crew-test")["status"] == "PROJECT_ROOT_MISMATCH"
