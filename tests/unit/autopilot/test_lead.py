"""The lead agent under the autopilot: trading never waits for it, a failing or hung lead changes nothing on the
trading path, its session requests are fail-safe only and validated, and its pacing is bounded."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from services.autopilot.lead import LeadState, lead_due, parse_cycle
from services.autopilot.research import StepOutcome
from services.autopilot.supervisor import CREW, PREOPEN, Autopilot

from .fakes import FakeApi, FakeClock, FakeRunner, _session, et, make

D = "2026-11-25"  # a Wednesday; Thanksgiving follows


def lead_output(actions: Sequence[dict[str, Any]] = (), *, wake: int | None = 30, outcome: str = "OK",
                rc: int | None = 0) -> StepOutcome:
    body = {"mode": "CLI subscriptions", "lead_cycle": {
        "lead": {"outcome": outcome, "provider": "claude"},
        "decision": {"situation": "session healthy", "priorities": ["watch rejections"]},
        "delegations": {"accepted": [{"task_id": "t1", "kind": "analysis", "role": "rejection analyst"}],
                        "refused": []},
        "ran": [{"task_id": "t1", "kind": "analysis", "provider": "codex", "outcome": "OK"}],
        "session_actions": list(actions), "next_wake_minutes": wake}}
    return StepOutcome(("-m", "services.agents", "lead"), rc, 4.0, json.dumps(body))


class LeadRunner:
    """Delegates research children to FakeRunner; lead children hang until released (or forever)."""

    def __init__(self, inner: FakeRunner) -> None:
        self.inner = inner
        self.lead_calls: list[tuple[str, ...]] = []
        self.outcomes: list[StepOutcome] = []
        self.hang = False
        self._release: asyncio.Event | None = None

    def release(self) -> None:
        assert self._release is not None
        self._release.set()

    async def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> StepOutcome:
        args = tuple(argv)
        if args[2] != "lead":
            return await self.inner(args, cwd, timeout_s)
        self.lead_calls.append(args)
        self._release = asyncio.Event()
        if self.hang:
            await asyncio.Event().wait()  # never returns; only cancellation ends it
        await self._release.wait()
        return self.outcomes.pop(0) if self.outcomes else lead_output()


def setup(tmp: Path, now: Any, **kw: Any) -> tuple[Autopilot, FakeClock, FakeApi, LeadRunner]:
    ap, clock, api, runner, _ = make(tmp, now, lead=True, **kw)
    lead = LeadRunner(runner)
    ap.runner = lead
    return ap, clock, api, lead


def run(body: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(body())  # one event loop: the background lead child lives across ticks


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def lead_events(ap: Autopilot, kind: str) -> list[dict[str, Any]]:
    return [e for e in ap.store.events(kind=kind) if e["job"] in ("lead", "lead_action")]


# ------------------------------------------------------------------------------------------------ pacing (pure)
def test_lead_due_rules() -> None:
    now = et(2026, 11, 25, 10, 0)
    kw = {"max_cycles": 3, "min_interval": timedelta(minutes=20), "max_interval": timedelta(minutes=180)}
    assert lead_due(LeadState(), day=D, now=now, evidence_seq=0, **kw) == (True, f"first lead turn for {D}")
    st = LeadState(D, 1, now, now + timedelta(minutes=60), evidence_seq=5)
    assert not lead_due(st, day=D, now=now + timedelta(minutes=10), evidence_seq=9, **kw)[0]  # min interval wins
    assert not lead_due(st, day=D, now=now + timedelta(minutes=30), evidence_seq=5, **kw)[0]  # nothing new
    assert lead_due(st, day=D, now=now + timedelta(minutes=30), evidence_seq=6, **kw)[1].startswith("new evidence")
    assert lead_due(st, day=D, now=now + timedelta(minutes=60), evidence_seq=5, **kw)[1] == "the lead's requested wake time"
    nowake = LeadState(D, 1, now, None, 5)
    assert lead_due(nowake, day=D, now=now + timedelta(minutes=180), evidence_seq=5, **kw)[0]
    spent = LeadState(D, 3, now - timedelta(hours=5), None, 0)
    assert not lead_due(spent, day=D, now=now, evidence_seq=99, **kw)[0]  # daily turn cap
    assert lead_due(spent, day="2026-11-27", now=now + timedelta(days=2), evidence_seq=99, **kw)[0]  # new day resets


# ------------------------------------------------------------------------------- trading independence
def test_session_is_requested_and_observed_while_the_lead_hangs(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 8, 45))
    lead.hang = True

    async def body() -> None:
        results = await ap.tick()
        kinds = [r["kind"] for r in results]
        assert kinds.index("PREOPEN") < kinds.index("LEAD")  # the session request never waits for the lead
        assert api.count("start_paper_session") == 1
        await settle()
        assert ap.lead_running() and len(lead.lead_calls) == 1
        for minutes in (15, 30, 60, 75, 90, 120):  # into regular hours: observation continues on schedule
            clock.set(et(2026, 11, 25, 8, 45) + timedelta(minutes=minutes))
            await ap.tick()
        assert len(lead.lead_calls) == 1  # one cycle at a time, never a second child while one runs
        assert api.count("start_paper_session") == 1
        intraday = [e for e in ap.store.events(kind="TICK") if e["job"] == "intraday"]
        assert len(intraday) >= 3
        await ap._finish_lead(wait=False)  # pyright: ignore[reportPrivateUsage]
        assert not ap.lead_running()

    run(body)
    assert any("cancelled" in e["message"] for e in lead_events(ap, "RESULT"))
    job = ap.store.job(D, PREOPEN)
    assert job is not None and job["status"] == "DONE"


def test_failed_or_unavailable_lead_alerts_and_changes_nothing(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 8, 45))
    lead.outcomes = [StepOutcome(("x",), 1, 2.0, "Traceback: boom"),
                     lead_output([{"action": "request_halt", "reason": "x"}], outcome="UNAVAILABLE", wake=None,
                                 rc=1)]

    async def body() -> None:
        await ap.tick()
        await settle()
        lead.release()
        await settle()
        clock.advance(minutes=1)
        await ap.tick()
        assert not ap.lead_running()
        alerts = [e["message"] for e in ap.store.events(kind="ALERT") if e["job"] == "lead"]
        assert alerts and "trading and the deterministic research steps continue" in alerts[-1]
        st = ap.lead_state()
        assert st.next_wake is not None and st.next_wake - clock.now() == timedelta(minutes=ap.cfg.lead_max_interval_min)
        clock.advance(minutes=ap.cfg.lead_max_interval_min + 1)
        await ap.tick()
        await settle()
        lead.release()
        await settle()
        clock.advance(minutes=1)
        await ap.tick()

    run(body)
    assert len(lead.lead_calls) == 2
    assert api.count("request_halt") == 0  # a failed child's output is never acted on
    assert api.count("start_paper_session") == 1


# ------------------------------------------------------------------------------------ fail-safe requests
def _one_cycle(ap: Autopilot, clock: FakeClock, lead: LeadRunner, out: StepOutcome) -> None:
    lead.outcomes = [out]

    async def body() -> None:
        await ap.tick()
        await settle()
        assert ap.lead_running()
        lead.release()
        await settle()
        clock.advance(minutes=1)
        await ap.tick()
        await ap._finish_lead(wait=False)  # pyright: ignore[reportPrivateUsage]

    run(body)


def test_stand_down_before_the_session_skips_todays_request(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 8, 40))
    api.health_overrides["reconciliation"] = "DEGRADED"  # pre-open blocked: no session requested yet
    _one_cycle(ap, clock, lead, lead_output([{"action": "stand_down_today",
                                              "reason": "reconciliation degraded twice this week"}]))
    job = ap.store.job(D, PREOPEN)
    assert job is not None and job["status"] == "SKIPPED" and "lead agent stand-down" in job["detail"]["reason"]
    api.health_overrides.clear()
    for minutes in (10, 20, 30, 45):
        clock.set(et(2026, 11, 25, 8, 40) + timedelta(minutes=minutes))
        asyncio.run(ap.tick())
    assert api.count("start_paper_session") == 0
    assert any("stand_down_today" in e["message"] for e in ap.store.events(kind="ALERT"))


def test_stand_down_during_the_session_stops_only_the_autopilot_session(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 8, 45))
    act = {"action": "stand_down_today", "reason": "risk engine rejected every entry on stale quotes"}
    _one_cycle(ap, clock, lead, lead_output([act, act]))
    stops = [a for n, a in api.calls if n == "stop_session"]
    assert stops == [{"session_id": "sess_1", "reason": "lead agent: " + act["reason"]}]
    refused = [e["message"] for e in lead_events(ap, "REFUSED")]
    assert refused == ["lead request stand_down_today refused: already taken for this trading day"]
    assert api.count("start_paper_session") == 1

    ap2, clock2, api2, lead2 = setup(tmp_path / "op", et(2026, 11, 25, 8, 45))
    api2.active = _session("sess_op", "SCHEDULED", requested_by="operator")
    _one_cycle(ap2, clock2, lead2, lead_output([act]))
    assert api2.count("stop_session") == 0  # an operator's session is never stopped by the lead
    assert "not requested by the autopilot" in lead_events(ap2, "REFUSED")[0]["message"]


def test_request_halt_is_executed_once_and_puts_the_autopilot_and_lead_in_stand_down(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 10, 0))
    halt = {"action": "request_halt", "reason": "position safety BREACHED with no broker stop"}
    _one_cycle(ap, clock, lead, lead_output([halt, halt]))
    assert [a for n, a in api.calls if n == "request_halt"] == [{"reason": "lead agent: " + halt["reason"]}]
    assert ap.store.get("standdown")
    clock.advance(hours=2)
    plan = ap.decide(clock.now())
    assert "LEAD" not in [a.kind for a in plan.actions]
    assert "lead agent paused: stand-down until the operator clears the halt" in plan.notes
    assert api.count("start_paper_session") == 0  # 10:00 is past the start window: the day was MISSED anyway


def test_disabled_requests_and_unknown_actions_are_never_executed(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 10, 0), lead_session_actions=False)
    _one_cycle(ap, clock, lead, lead_output([{"action": "request_halt", "reason": "x"},
                                            {"action": "start_paper_session", "reason": "buy the dip"}]))
    assert api.count("request_halt") == 0 and api.count("start_paper_session") == 0
    assert "disabled" in lead_events(ap, "REFUSED")[0]["message"]
    parsed = parse_cycle(lead_output([{"action": "resume_trading", "reason": "x"},
                                      {"action": "request_halt", "reason": ""}]), locked_exit=4)
    assert parsed.session_actions == []


# --------------------------------------------------------------------------------------- research + pacing
def test_lead_replaces_crew_ticks_and_deterministic_research_still_runs(tmp_path: Path) -> None:
    ap, clock, _api, lead = setup(tmp_path, et(2026, 11, 25, 17, 0))
    lead.hang = True

    async def body() -> None:
        for _ in range(8):
            await ap.tick()
            await settle()
            clock.advance(minutes=2)
        await ap._finish_lead(wait=False)  # pyright: ignore[reportPrivateUsage]

    run(body)
    crew = ap.store.job(D, CREW)
    assert crew is not None and crew["status"] == "SKIPPED" and "lead agent" in crew["detail"]["reason"]
    assert "crew" not in lead.inner.commands()
    assert {j["job"]: j["status"] for j in ap.store.jobs(D)}["report"] == "DONE"
    assert len(lead.lead_calls) == 1


def test_lead_wakes_on_new_evidence_but_never_before_the_min_interval(tmp_path: Path) -> None:
    ap, clock, _api, lead = setup(tmp_path, et(2026, 11, 25, 10, 0))
    _one_cycle(ap, clock, lead, lead_output(wake=120))
    t0 = clock.now()
    assert ap.lead_state().next_wake == t0 + timedelta(minutes=120)
    clock.set(t0 + timedelta(minutes=10))
    ap.store.event(clock.now(), "ALERT", "intraday: session FAILED", day=D, job="intraday")
    assert "LEAD" not in [a.kind for a in ap.decide(clock.now()).actions]
    clock.set(t0 + timedelta(minutes=25))
    assert [a.reason for a in ap.decide(clock.now()).actions if a.kind == "LEAD"] == [
        "lead-agent cycle: new evidence since the last lead turn"]


def test_briefing_is_compact_labelled_and_free_of_credentials(tmp_path: Path) -> None:
    ap, clock, _api, lead = setup(tmp_path, et(2026, 11, 25, 10, 0))
    _one_cycle(ap, clock, lead, lead_output())
    brief = json.loads((tmp_path / "ap" / "lead" / "briefing.json").read_text())
    assert brief["provenance"]["is_simulated"] is True and brief["provenance"]["execution_labels"] == ["SIMULATED"]
    assert brief["risk"]["mode"] == "PAPER" and brief["session"] is not None
    assert brief["lead_limits"]["max_turns"] == ap.cfg.lead_max_cycles
    assert "token" not in json.dumps(brief).lower()
    assert (tmp_path / "ap" / "lead" / "briefing.json").stat().st_mode & 0o077 == 0
    argv = lead.lead_calls[0]
    assert argv[:4] == ("-m", "services.agents", "lead", "--json") and argv[-1].endswith("lead/briefing.json")


def test_daily_report_summarizes_the_lead(tmp_path: Path) -> None:
    ap, clock, _api, lead = setup(tmp_path, et(2026, 11, 25, 10, 0))
    _one_cycle(ap, clock, lead, lead_output())
    asyncio.run(ap._report(D, clock.now(), final=False))  # pyright: ignore[reportPrivateUsage]
    rep = json.loads((tmp_path / "ap" / "reports" / f"{D}.json").read_text())
    la = rep["lead_agent"]
    assert (la["cycles"], la["delegated"], la["subagent_runs"]) == (1, 1, 1)
    assert la["last"]["lead_outcome"] == "OK" and la["last"]["situation"] == "session healthy"
    assert "Lead agent: 1 cycles" in (tmp_path / "ap" / "reports" / f"{D}.md").read_text()


def test_session_requests_are_ignored_unless_the_lead_turn_succeeded(tmp_path: Path) -> None:
    ap, clock, api, lead = setup(tmp_path, et(2026, 11, 25, 10, 0))
    _one_cycle(ap, clock, lead, lead_output([{"action": "request_halt", "reason": "x"}], outcome="FAILED"))
    assert api.count("request_halt") == 0
    assert not lead_events(ap, "REFUSED") and not ap.store.get("standdown")
