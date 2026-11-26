"""The deterministic daily state machine.

``decide(now)`` is a pure function of the wall clock, the XNYS calendar and the autopilot's own store: it returns
the actions due now and why (``plan`` prints it without executing anything). ``tick()`` re-reads the wall clock,
decides and executes those actions through the guarded tool backend and the research subprocess allowlist.

Daily sequence for a trading day D (New York time, early closes included; holidays have no session):

1. PREOPEN ``[open - lead, open + grace)``: inspect the API first (adopt an existing session, never duplicate),
   then risk/halts, system health, broker reconciliation and feed; if all pass, request today's PAPER session for
   the regular session up to the entry cutoff via ``start_paper_session``. A failed check blocks the start, is
   recorded and alerted, and is re-checked at most every ``PREOPEN_RECHECK_MIN`` inside the window. When the window
   closes without a session the day is MISSED/BLOCKED: no late trading.
2. REGULAR: one bounded status check per ``INTRADAY_INTERVAL_MIN`` (session status + risk status). Observation
   only: the session manager and the position manager own orders, cancels and exits.
3. QUIET (close + settle until 90 min before the next open): quantlab daily pipeline (unless the API runs it),
   research-crew ticks, shadow journal (whose existing ``promotion_check`` is the only SHADOW -> PAPER path), then
   at most one protected-holdout look for a DEV_PASSED family via the existing ``holdout`` command. One step per
   tick, each child bounded by its timeout and by the next pre-open. A day whose research never ran is superseded
   by the next day's run (the steps are incremental): research catches up once, never per missed day.
4. REPORT: a partial report after pre-open and each intraday check, the final one after research.
5. LEAD (``services.autopilot.lead``): from the pre-open until the research window ends, one lead-agent cycle at a
   time runs as a background child. The lead decides what to investigate or research and delegates it to parallel
   subagents; it replaces the fixed crew ticks. Its only effect on trading is a validated fail-safe request (stand
   down today, request a halt). Steps 1-4 never wait for it and run the same when it is unavailable.

Any active halt (ENTRIES scope or the operator's ALL-scope kill switch) puts the autopilot into stand-down: no
session request and no research step until a later read-only check shows the operator has cleared it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol, cast

from services.sessions.calendar import et_date, parse_utc

from . import report as reports
from .backend import REQUESTED_BY, AuthorityError, ToolPort
from .checks import Verdict, intraday_verdict, preopen_verdict
from .config import AutopilotConfig
from .daycal import (
    Phase,
    TradingDay,
    last_completed,
    market_open,
    next_session,
    phase_at,
    quiet_until,
    session_minutes,
    session_today,
    trading_day,
)
from .lead import (
    EVIDENCE_KINDS,
    K_LEAD,
    K_LEAD_ACTIONS,
    LEAD,
    LEAD_ACTION_JOB,
    PERFORMANCE_KEYS,
    LeadState,
    compact_health,
    compact_report,
    compact_risk,
    compact_session,
    lead_due,
    parse_cycle,
    write_briefing,
)
from .research import (
    CREW_ARGV,
    CREW_LOCKED_EXIT,
    Runner,
    StepOutcome,
    holdout_argv,
    lead_argv,
    quantlab_argv,
    read_json,
    run_child,
)
from .research import summarize_crew as _summarize_crew
from .store import TERMINAL, Store

log = logging.getLogger("lat.autopilot")

PREOPEN: Final = "preopen"
DAILY: Final = "quantlab_daily"
CREW: Final = "crew"
SHADOW: Final = "shadow"
HOLDOUT: Final = "holdout"
REPORT: Final = "report"
RESEARCH_SEQUENCE: Final = (DAILY, CREW, SHADOW, HOLDOUT)
MAX_START_ATTEMPTS: Final = 3
MAX_RESEARCH_ATTEMPTS: Final = 3
RESEARCH_RETRY: Final = timedelta(minutes=60)
PREOPEN_MARGIN: Final = timedelta(minutes=5)
MIN_RESEARCH_BUDGET_S: Final = 600.0  # a child that could not get 10 minutes before the next pre-open waits
RESEARCH_RISK_POLL_S: Final = 30.0
ACTIVE_STATES: Final = frozenset({"SCHEDULED", "RUNNING", "STOPPING"})
K_STANDDOWN: Final = "standdown"
K_LAST_CHECK: Final = "last_intraday_check"
K_LAST_STANDDOWN_CHECK: Final = "last_standdown_check"
K_HOLDOUT_ATTEMPTS: Final = "holdout_attempts"
K_LAST_TICK: Final = "last_tick"


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True)
class Action:
    kind: str
    day: str | None
    reason: str

    def public(self) -> dict[str, Any]:
        return {"kind": self.kind, "day": self.day, "reason": self.reason}


@dataclass
class Plan:
    now: datetime
    phase: Phase
    day: TradingDay | None
    actions: list[Action] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    standdown: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {"now": self.now.isoformat(), "phase": self.phase.value,
                "trading_day": self.day.public() if self.day else None,
                "actions": [a.public() for a in self.actions], "notes": self.notes, "standdown": self.standdown}


def _d(v: Any) -> dict[str, Any]:
    return cast(dict[str, Any], v) if isinstance(v, dict) else {}


def _l(v: Any) -> list[Any]:
    return cast(list[Any], v) if isinstance(v, list) else []


def _ts(v: Any) -> datetime | None:
    try:
        return parse_utc(str(v)) if v else None
    except ValueError:
        return None


def _err(exc: BaseException) -> tuple[int | None, str]:
    status = getattr(exc, "status", None)
    msg_fn = getattr(exc, "message", None)
    text = msg_fn() if callable(msg_fn) else str(exc)
    return (status if isinstance(status, int) else None), f"{type(exc).__name__}: {str(text)[:300]}"


class Autopilot:
    def __init__(self, cfg: AutopilotConfig, store: Store, backend: ToolPort | None, *, clock: Clock | None = None,
                 runner: Runner = run_child, api_owns_quantlab: bool = False, simulated_feed: bool = False,
                 ensure_api: Callable[[], Any] | None = None,
                 daily_job_factory: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg
        self.store = store
        self.backend = backend
        self.clock: Clock = clock or SystemClock()
        self.runner = runner
        self.api_owns_quantlab = api_owns_quantlab
        self.simulated_feed = simulated_feed
        self.ensure_api = ensure_api
        self.daily_job_factory = daily_job_factory
        self._lead_task: asyncio.Future[StepOutcome] | None = None
        self._lead_day: str | None = None

    # ================================================================== decide (pure: clock + calendar + store)
    def decide(self, now: datetime) -> Plan:
        cfg = self.cfg
        phase, td = phase_at(now, lead_min=cfg.preopen_lead_min, grace_min=cfg.start_grace_min,
                             settle_min=cfg.settle_min)
        sd = self.store.get(K_STANDDOWN)
        plan = Plan(now, phase, td, standdown=sd)
        today = session_today(now)
        if today is None and phase in (Phase.CLOSED, Phase.QUIET, Phase.POST_CLOSE):
            plan.notes.append(f"{et_date(now).isoformat()} is not an XNYS session (weekend or holiday)")
        if today is not None:
            self._decide_preopen(plan, today, now)
            if phase is Phase.REGULAR or (phase is Phase.PREOPEN and today.open <= now):
                last = _ts(self.store.get(K_LAST_CHECK))
                if last is None or now - last >= timedelta(minutes=cfg.intraday_interval_min):
                    plan.actions.append(Action("INTRADAY_CHECK", today.day.isoformat(),
                                               f"bounded status check every {cfg.intraday_interval_min} min"))
        if sd and phase not in (Phase.PREOPEN, Phase.REGULAR):
            last = _ts(self.store.get(K_LAST_STANDDOWN_CHECK))
            if last is None or now - last >= timedelta(minutes=cfg.intraday_interval_min):
                plan.actions.append(Action("STANDDOWN_CHECK", None, "stand-down: read-only check whether the operator "
                                                                    "cleared the halt"))
        if phase is Phase.QUIET and td is not None:
            self._decide_research(plan, td, now, sd)
        self._decide_lead(plan, phase, td, now, sd)
        self._decide_reports(plan, now)
        if not plan.actions:
            plan.notes.append("nothing due now")
        return plan

    def _decide_preopen(self, plan: Plan, td: TradingDay, now: datetime) -> None:
        cfg, d = self.cfg, td.day.isoformat()
        job = self.store.job(d, PREOPEN)
        if job is not None and job["status"] in TERMINAL:
            return
        if td.early_close:
            plan.notes.append(f"{d} is an early close ({td.close.isoformat()}); entry cutoff {td.entry_cutoff.isoformat()}")
        last = _ts(_d(job["detail"] if job else {}).get("last_check_at"))
        if job and job["status"] == "UNKNOWN" and last is not None and now - last < timedelta(minutes=cfg.preopen_recheck_min):
            return
        if now >= td.start_deadline(cfg.start_grace_min):
            plan.actions.append(Action("MISS_PREOPEN", d, "the pre-open window has closed: no late session request "
                                                          "(missed windows never trade late)"))
            return
        if now < td.preopen_start(cfg.preopen_lead_min):
            return
        status = job["status"] if job else None
        if status in ("RUNNING", "UNKNOWN"):
            plan.actions.append(Action("PREOPEN", d, "resume: the last start outcome is unknown; inspect the API before "
                                                     "any request"))
            return
        if last is not None and now - last < timedelta(minutes=cfg.preopen_recheck_min):
            plan.notes.append(f"pre-open checks blocked at {last.isoformat()}; re-check after "
                              f"{(last + timedelta(minutes=cfg.preopen_recheck_min)).isoformat()}")
            return
        minutes = session_minutes(td, now, cfg.session_minutes)
        plan.actions.append(Action("PREOPEN", d, f"health checks, then request today's PAPER session "
                                                 f"({minutes} min, ending by the entry cutoff)"))

    def _decide_research(self, plan: Plan, td: TradingDay, now: datetime, sd: Any) -> None:
        d = td.day.isoformat()
        if not self.cfg.research:
            plan.notes.append("research steps disabled (LAT_AUTOPILOT_RESEARCH=false)")
            return
        if sd:
            plan.notes.append("research deferred: stand-down until the operator clears the halt")
            return
        previous: dict[str, dict[str, str]] = {}
        for row in self.store.jobs():
            if row["day"] < d:
                previous.setdefault(row["day"], {})[row["job"]] = row["status"]
        for old, jobs in previous.items():
            if not any(job in jobs for job in (PREOPEN, *RESEARCH_SEQUENCE)) or all(
                jobs.get(job) in TERMINAL for job in RESEARCH_SEQUENCE
            ):
                continue
            plan.actions.append(Action("SUPERSEDE", old, f"research for {old} never completed; {d}'s incremental run "
                                                         "catches up once"))
        if self._research_budget(td, now) < MIN_RESEARCH_BUDGET_S:
            plan.notes.append("research deferred: too close to the next pre-open")
            return
        for job in RESEARCH_SEQUENCE:
            row = self.store.job(d, job)
            if row is not None and row["status"] in TERMINAL:
                continue
            retry = _ts(_d(row["detail"] if row else {}).get("retry_after"))
            if retry is not None and now < retry:
                plan.notes.append(f"{job} retry after {retry.isoformat()}")
                return
            plan.actions.append(Action(job.upper(), d, f"after-close research step {job} for {d}"))
            return

    def lead_state(self) -> LeadState:
        return LeadState.load(self.store.get(K_LEAD), _ts)

    def lead_running(self) -> bool:
        return self._lead_task is not None and not self._lead_task.done()

    def _decide_lead(self, plan: Plan, phase: Phase, td: TradingDay | None, now: datetime, sd: Any) -> None:
        cfg = self.cfg
        if not cfg.lead:
            return
        if td is None or phase is Phase.CLOSED:
            return
        if self.lead_running():
            plan.notes.append(f"lead cycle for {self._lead_day} in progress (background child)")
            return
        if sd:
            plan.notes.append("lead agent paused: stand-down until the operator clears the halt")
            return
        due, reason = lead_due(self.lead_state(), day=td.day.isoformat(), now=now,
                               evidence_seq=self.store.last_seq(EVIDENCE_KINDS, exclude_jobs=(LEAD, LEAD_ACTION_JOB)),
                               max_cycles=cfg.lead_max_cycles, min_interval=timedelta(minutes=cfg.lead_min_interval_min),
                               max_interval=timedelta(minutes=cfg.lead_max_interval_min))
        if due:
            plan.actions.append(Action("LEAD", td.day.isoformat(), f"lead-agent cycle: {reason}"))
        else:
            plan.notes.append(reason)

    def _decide_reports(self, plan: Plan, now: datetime) -> None:
        by_day: dict[str, dict[str, str]] = {}
        for j in self.store.jobs():
            by_day.setdefault(j["day"], {})[j["job"]] = j["status"]
        for d, jobs in sorted(by_day.items()):
            if jobs.get(REPORT) in TERMINAL:
                continue
            td = trading_day(date.fromisoformat(d))
            if td is None or now < td.research_start(self.cfg.settle_min):
                continue
            research_done = not self.cfg.research or all(jobs.get(j) in TERMINAL for j in RESEARCH_SEQUENCE)
            end = quiet_until(td)
            if research_done or (end is not None and now >= end):
                plan.actions.append(Action("REPORT", d, "final daily report"))

    def _research_budget(self, td: TradingDay, now: datetime) -> float:
        """Seconds a research child may run before it would reach the next pre-open (minus a margin)."""
        nxt = next_session(td.day)
        if nxt is None:
            return 0.0
        deadline = nxt.preopen_start(self.cfg.preopen_lead_min) - PREOPEN_MARGIN
        return (deadline - now).total_seconds()

    # ================================================================== tick / run
    async def tick(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            polled = await self._poll_lead()
        except AuthorityError:
            raise
        except Exception as exc:  # a lead result must never stop the deterministic loop
            self._alert(self.clock.now(), self._lead_day, LEAD, f"lead result handling failed: {_err(exc)[1]}")
            polled = None
        if polled is not None:
            results.append({"kind": "LEAD_RESULT", "day": self._lead_day, "result": polled})
        now = self.clock.now()
        plan = self.decide(now)
        for action in plan.actions:
            at = self.clock.now()  # wall clock re-read before every action
            try:
                res = await self._execute(action, at)
            except AuthorityError:
                raise
            except Exception as exc:  # the loop survives; the failure is recorded and alerted
                _, msg = _err(exc)
                self._alert(at, action.day, action.kind.lower(), f"{action.kind} failed: {msg}")
                res = {"error": msg}
            results.append({**action.public(), "result": res})
        self.store.put(self.clock.now(), K_LAST_TICK, self.clock.now().isoformat())
        return results

    async def run(self, stop: asyncio.Event, *, once: bool = False) -> None:
        now = self.clock.now()
        self.store.event(now, "START", "autopilot started", data={"once": once, "config": self.cfg.public()})
        for job in self.store.open_jobs():
            self.store.event(now, "RESUME", f"resuming {job['job']} for {job['day']} (was {job['status']})",
                             day=job["day"], job=job["job"])
        last = _ts(self.store.get(K_LAST_TICK))
        try:
            while not stop.is_set():
                now = self.clock.now()
                if last is not None and (now - last).total_seconds() > self.cfg.tick_s * 3 + 60:
                    self.store.event(now, "SLEEP_GAP", f"wall clock advanced {int((now - last).total_seconds())} s "
                                                       "since the last tick (sleep, suspend or downtime); missed trading "
                                                       "windows are not caught up",
                                     data={"last_tick": last.isoformat()})
                await self.tick()
                last = self.clock.now()
                if once:
                    break
                with contextlib.suppress(TimeoutError):  # wakes early on stop; the wall clock is re-read
                    await asyncio.wait_for(stop.wait(), timeout=self.cfg.tick_s)
        finally:
            await self._finish_lead(wait=once and not stop.is_set())
            self.store.event(self.clock.now(), "STOP", "autopilot stopped")

    async def _execute(self, a: Action, now: datetime) -> dict[str, Any]:
        if a.kind in ("PREOPEN", "MISS_PREOPEN", "INTRADAY_CHECK"):
            assert a.day is not None
            td = trading_day(date.fromisoformat(a.day))
            assert td is not None
            if a.kind == "PREOPEN":
                return await self._preopen(td, now)
            if a.kind == "MISS_PREOPEN":
                return await self._miss_preopen(td, now)
            if not td.open <= now < td.close:
                return {"deferred": "outside the observation window"}
            return await self._intraday(td, now)
        if a.kind == "STANDDOWN_CHECK":
            return await self._standdown_check(now)
        if a.kind == "SUPERSEDE":
            assert a.day is not None
            return self._supersede(a.day, now)
        if a.kind == "REPORT":
            assert a.day is not None
            return await self._report(a.day, now, final=True)
        if a.kind == "LEAD":
            assert a.day is not None
            td = trading_day(date.fromisoformat(a.day))
            assert td is not None
            return await self._lead_launch(td, now)
        assert a.day is not None
        td = trading_day(date.fromisoformat(a.day))
        assert td is not None
        handlers = {"QUANTLAB_DAILY": self._daily, "CREW": self._crew, "SHADOW": self._shadow,
                    "HOLDOUT": self._holdout}
        blocked = await self._check_research_risk(td, now)
        if blocked is not None:
            return {"deferred": blocked}
        now = self.clock.now()
        if not self._research_due(td, now):
            return {"deferred": "research window ended or insufficient time before pre-open"}
        return await handlers[a.kind](td, now)

    # ================================================================== helpers
    async def _call(self, name: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.backend is None:
            raise RuntimeError("no tool backend configured")
        return await self.backend.call(name, dict(args or {}))

    def _alert(self, now: datetime, day: str | None, job: str | None, message: str, *, dedupe: bool = True) -> None:
        key = f"alert:{day}:{job}"
        if dedupe and self.store.get(key) == message:
            return
        self.store.put(now, key, message)
        self.store.event(now, "ALERT", message, day=day, job=job)
        log.warning("autopilot alert: %s", message)

    def _enter_standdown(self, now: datetime, verdict: Verdict, day: str | None) -> None:
        cur = self.store.get(K_STANDDOWN)
        info = {"since": (cur or {}).get("since", now.isoformat()), "kill_switch": verdict.kill_switch,
                "reasons": verdict.reasons[:5]}
        self.store.put(now, K_STANDDOWN, info)
        self.store.put(now, K_LAST_STANDDOWN_CHECK, now.isoformat())
        if not cur:
            what = "operator kill switch (ALL scope)" if verdict.kill_switch else "an active halt"
            self.store.event(now, "STANDDOWN", f"standing down: {what}; no session request or research until the "
                                               "operator clears it", day=day, data=info)
            self._alert(now, day, "standdown", f"stand-down: {what}: {'; '.join(verdict.reasons[:3])}")

    def _clear_standdown(self, now: datetime, day: str | None) -> None:
        if self.store.get(K_STANDDOWN):
            self.store.delete(K_STANDDOWN)
            self.store.event(now, "STANDDOWN_CLEARED", "no active halt any more (cleared by the operator); resuming",
                             day=day)

    def _block(self, now: datetime, day: str, reasons: list[str], *, alert: bool = True,
               extra: dict[str, Any] | None = None, unresolved: bool = False) -> dict[str, Any]:
        self.store.set_job(now, day, PREOPEN, "UNKNOWN" if unresolved else "PENDING",
                           detail={"blocked_reasons": reasons[:10], "last_check_at": now.isoformat(), **(extra or {})})
        self.store.event(now, "DECISION", "pre-open blocked: " + "; ".join(reasons[:3]), day=day, job=PREOPEN,
                         data={"reasons": reasons[:10]})
        if alert:
            self._alert(now, day, PREOPEN, "no session start yet: " + "; ".join(reasons[:3]))
        return {"blocked": reasons[:10]}

    # ================================================================== pre-open
    @staticmethod
    def existing_session(status: Mapping[str, Any], td: TradingDay) -> tuple[str, dict[str, Any]] | None:
        """An existing session or schedule that already covers ``td`` (never create a duplicate)."""
        sessions = [_d(status.get("active")), *(_d(x) for x in _l(status.get("recent")))]
        for s in sessions:
            if s.get("state") in ACTIVE_STATES:
                return ("ADOPTED" if s.get("requested_by") == REQUESTED_BY else "ALREADY_ACTIVE"), s
        for s in sessions:
            when = _ts(s.get("started_at")) or _ts(s.get("ends_at"))
            if when is not None and et_date(when) == td.day:
                return "ALREADY_RAN", s
        for sch in (_d(x) for x in _l(status.get("schedules"))):
            nra = _ts(sch.get("next_run_at"))
            last_run = _d(sch.get("last_run"))
            if last_run.get("run_date") == td.day.isoformat() and last_run.get("status") == "STARTED":
                return "OPERATOR_SCHEDULE", sch
            if sch.get("enabled") and nra is not None and et_date(nra) == td.day:
                return "OPERATOR_SCHEDULE", sch
        return None

    def _adopt(self, now: datetime, d: str, outcome: str, found: dict[str, Any],
               status: Mapping[str, Any]) -> dict[str, Any]:
        detail = {"outcome": outcome, "session_id": found.get("session_id") or _d(found.get("last_run")).get("session_id"),
                  "schedule_id": found.get("schedule_id"), "execution_label": status.get("execution_label"),
                  "is_simulated": status.get("is_simulated"),
                  "session_state": found.get("state"), "requested_by": found.get("requested_by") or found.get("created_by"),
                  "reason": {"ADOPTED": "the API already holds today's autopilot session",
                             "ALREADY_ACTIVE": "another session is already queued or running; not creating a second",
                             "ALREADY_RAN": "a session already ran today",
                             "OPERATOR_SCHEDULE": "an operator schedule already covers today"}[outcome]}
        self.store.set_job(now, d, PREOPEN, "DONE", detail=detail)
        self.store.event(now, "RESULT", f"pre-open: {outcome} ({detail['reason']})", day=d, job=PREOPEN, data=detail)
        return detail

    async def _preopen(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        job = self.store.job(d, PREOPEN) or self.store.set_job(now, d, PREOPEN, "PENDING")
        if now >= td.start_deadline(self.cfg.start_grace_min):
            return await self._miss_preopen(td, now)
        if now < td.preopen_start(self.cfg.preopen_lead_min):
            return {"deferred": "pre-open window has not started"}
        unresolved = job["status"] in ("RUNNING", "UNKNOWN")
        if self.cfg.start_api and self.ensure_api is not None:
            try:
                await asyncio.to_thread(self.ensure_api)
            except Exception as exc:
                return self._block(now, d, [f"local API not available: {_err(exc)[1]}"], unresolved=unresolved)
        try:
            status = await self._call("get_session_status")
        except Exception as exc:
            if isinstance(exc, AuthorityError):
                raise
            return self._block(now, d, [f"get_session_status failed: {_err(exc)[1]}"], unresolved=unresolved)
        found = self.existing_session(status, td)
        if found is not None:
            res = self._adopt(now, d, *found, status)
            await self._standdown_check(now)
            await self._report(d, now, final=False)
            return res
        try:
            health = await self._call("get_system_health")
            feed = await self._call("get_feed_quality")
            if int(job["attempts"]) > 0:
                # An earlier request can become visible while the health checks are in flight.
                status = await self._call("get_session_status")
                found = self.existing_session(status, td)
            risk = await self._call("get_risk_status")
        except Exception as exc:
            if isinstance(exc, AuthorityError):
                raise
            return self._block(now, d, [f"health check call failed: {_err(exc)[1]}"], unresolved=unresolved)
        now = self.clock.now()
        if now >= td.start_deadline(self.cfg.start_grace_min):
            return await self._miss_preopen(td, now)
        if now < td.preopen_start(self.cfg.preopen_lead_min):
            return {"deferred": "pre-open window has not started"}
        verdict = preopen_verdict(risk=risk, health=health, feed=feed, market_open=market_open(now))
        if verdict.stand_down:
            self._enter_standdown(now, verdict, d)
        else:
            self._clear_standdown(now, d)
        if found is not None:
            res = self._adopt(now, d, *found, status)
            await self._report(d, now, final=False)
            return res
        if verdict.stand_down:
            return self._block(now, d, ["stand-down: " + r for r in verdict.reasons], alert=False,
                               extra={"provenance": verdict.provenance}, unresolved=unresolved)
        if not verdict.ok:
            return self._block(now, d, verdict.reasons, extra={"provenance": verdict.provenance}, unresolved=unresolved)
        minutes = session_minutes(td, now, self.cfg.session_minutes)
        if minutes == 0:
            self.store.set_job(now, d, PREOPEN, "SKIPPED", detail={"reason": "no usable regular-hours window left before "
                                                                             "the entry cutoff"})
            return {"skipped": "no usable window"}
        if int(job["attempts"]) >= MAX_START_ATTEMPTS:
            self.store.set_job(now, d, PREOPEN, "BLOCKED", detail={"reason": "start attempts exhausted"})
            self._alert(now, d, PREOPEN, f"no session today: {MAX_START_ATTEMPTS} start attempts did not produce a session")
            return {"blocked": "attempts exhausted"}
        args: dict[str, Any] = {"duration_minutes": minutes, "symbols": list(self.cfg.symbols) if self.cfg.symbols else None,
                                "cadence_minutes": self.cfg.cadence_minutes, "use_ai": self.cfg.use_ai}
        self.store.set_job(now, d, PREOPEN, "RUNNING", attempt=True,
                           detail={"start_requested_at": now.isoformat(), "request": args, "warnings": verdict.warnings,
                                   "provenance": verdict.provenance, "blocked_reasons": []})
        self.store.event(now, "ACTION", f"requesting today's PAPER session ({minutes} min)", day=d, job=PREOPEN, data=args)
        send_at = self.clock.now()
        if not td.preopen_start(self.cfg.preopen_lead_min) <= send_at < td.start_deadline(self.cfg.start_grace_min):
            return {"deferred": "pre-open window changed before dispatch"}
        try:
            res = await self._call("start_paper_session", args)
        except AuthorityError:
            raise
        except Exception as exc:
            code, msg = _err(exc)
            if code in (403, 422):
                self.store.set_job(now, d, PREOPEN, "BLOCKED", detail={"reason": f"start refused: {msg}"})
                self._alert(now, d, PREOPEN, f"no session today: start refused ({msg})")
                return {"refused": msg}
            if code == 409:  # refused by the session manager; re-inspected next tick, never resent blindly
                self.store.set_job(now, d, PREOPEN, "PENDING", detail={"last_error": msg, "last_check_at": now.isoformat()})
                self._alert(now, d, PREOPEN, f"session start refused (409): {msg}")
                return {"refused": msg}
            self.store.set_job(now, d, PREOPEN, "UNKNOWN", detail={"last_error": msg})
            self.store.event(now, "RESULT", f"start outcome UNKNOWN ({msg}); the next tick inspects the API before any "
                                            "retry", day=d, job=PREOPEN)
            return {"unknown": msg}
        sess = _d(res.get("session"))
        detail = {"outcome": res.get("outcome"), "session_id": sess.get("session_id"), "session_state": sess.get("state"),
                  "ends_at": sess.get("ends_at"), "execution_label": res.get("execution_label"),
                  "is_simulated": res.get("is_simulated")}
        self.store.set_job(now, d, PREOPEN, "DONE", detail=detail)
        self.store.event(now, "RESULT", f"session {detail['outcome']}: {detail['session_id']}", day=d, job=PREOPEN,
                         data=detail)
        await self._report(d, now, final=False)
        return detail

    async def _miss_preopen(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        if now < td.start_deadline(self.cfg.start_grace_min):
            return {"deferred": "pre-open window has not ended"}
        d = td.day.isoformat()
        job = self.store.job(d, PREOPEN)
        if job is not None and job["status"] in ("RUNNING", "UNKNOWN"):
            try:  # read-only reconciliation of an unknown start outcome
                status_snapshot = await self._call("get_session_status")
            except AuthorityError:
                raise
            except Exception as exc:
                return self._block(now, d, [f"start outcome remains unknown: {_err(exc)[1]}"], unresolved=True)
            found = self.existing_session(status_snapshot, td)
            if found is not None:
                res = self._adopt(now, d, *found, status_snapshot)
                await self._report(d, now, final=False)
                return res
        blocked = _d(job["detail"] if job else {}).get("blocked_reasons") or []
        status = "BLOCKED" if blocked else "MISSED"
        if job and job["status"] in ("RUNNING", "UNKNOWN"):
            reason = "the pre-open window closed and no session was visible after an uncertain start attempt"
        else:
            reason = ("pre-open checks never passed: " + "; ".join(blocked[:3])) if blocked else \
                "the pre-open window passed without a session request"
        self.store.set_job(now, d, PREOPEN, status, detail={"reason": reason})
        self._alert(now, d, PREOPEN, f"no autopilot session on {d}: {reason}")
        await self._report(d, now, final=False)
        return {"status": status, "reason": reason}

    # ================================================================== intraday / stand-down
    async def _intraday(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        self.store.put(now, K_LAST_CHECK, now.isoformat())  # first: a failing check never turns into a storm
        try:
            status = await self._call("get_session_status")
            risk = await self._call("get_risk_status")
        except AuthorityError:
            raise
        except Exception as exc:
            self._alert(now, d, "intraday", f"status check failed: {_err(exc)[1]}")
            await self._report(d, now, final=False)
            return {"error": _err(exc)[1]}
        v = intraday_verdict(risk=risk, status=status)
        if v.stand_down:
            self._enter_standdown(now, v, d)
        else:
            self._clear_standdown(now, d)
        for w in v.warnings:
            self._alert(now, d, "intraday", w)
        active = _d(status.get("active"))
        summary = {"session_id": active.get("session_id"), "state": active.get("state"),
                   "cycles_run": active.get("cycles_run"), "trades": active.get("trades"),
                   "entries_locked": risk.get("entries_locked"), "stand_down": v.stand_down,
                   "provenance": v.provenance}
        self.store.event(now, "TICK", "intraday status check (observation only)", day=d, job="intraday", data=summary)
        await self._report(d, now, final=False)
        return summary

    async def _standdown_check(self, now: datetime) -> dict[str, Any]:
        self.store.put(now, K_LAST_STANDDOWN_CHECK, now.isoformat())
        try:
            risk = await self._call("get_risk_status")
        except AuthorityError:
            raise
        except Exception as exc:
            return {"error": _err(exc)[1]}
        v = intraday_verdict(risk=risk, status={})
        if v.stand_down:
            self._enter_standdown(now, v, None)
            return {"stand_down": True, "reasons": v.reasons}
        self._clear_standdown(now, None)
        return {"stand_down": False}

    # ================================================================== research (subprocess steps)
    async def _check_research_risk(self, td: TradingDay, now: datetime) -> str | None:
        d = td.day.isoformat()
        try:
            risk = await self._call("get_risk_status")
        except AuthorityError:
            raise
        except Exception as exc:
            reason = f"research deferred until risk status is available: {_err(exc)[1]}"
            self._alert(now, d, "research", reason)
            return reason
        verdict = intraday_verdict(risk=risk, status={})
        if verdict.stand_down:
            self._enter_standdown(now, verdict, d)
            return "active halt"
        if risk.get("mode") != "PAPER" or risk.get("live_execution_compiled") is not False:
            raise AuthorityError("research requires a Paper runtime with Live compiled off")
        return None

    def _research_due(self, td: TradingDay, now: datetime) -> bool:
        end = quiet_until(td)
        return end is not None and td.research_start(self.cfg.settle_min) <= now < end and (
            self._research_budget(td, now) >= MIN_RESEARCH_BUDGET_S
        )

    def _timeout(self, td: TradingDay, now: datetime, configured: float) -> float:
        end = quiet_until(td)
        if end is None:
            return 0.0
        return max(0.0, min(configured, self._research_budget(td, now), (end - now).total_seconds()))

    async def _run_research_child(self, td: TradingDay, argv: Sequence[str], cwd: Path,
                                  configured: float) -> StepOutcome:
        now = self.clock.now()
        if not self._research_due(td, now):
            return StepOutcome(tuple(argv), None, 0, "research window ended or insufficient time before pre-open")
        child = asyncio.ensure_future(self.runner(argv, cwd, self._timeout(td, now, configured)))
        try:
            while not child.done():
                await asyncio.wait({child}, timeout=RESEARCH_RISK_POLL_S)
                if child.done():
                    break
                blocked = await self._check_research_risk(td, self.clock.now())
                if blocked is not None:
                    return StepOutcome(tuple(argv), None, (self.clock.now() - now).total_seconds(), blocked)
            return await child
        finally:
            if not child.done():
                child.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await child

    def _research_fail(self, now: datetime, d: str, job: str, msg: str, detail: dict[str, Any],
                       *, attempts: int | None = None) -> dict[str, Any]:
        if attempts is None:
            attempts = int((self.store.job(d, job) or {}).get("attempts", 0))
        if attempts >= MAX_RESEARCH_ATTEMPTS:
            self.store.set_job(now, d, job, "FAILED", detail={**detail, "error": msg})
            self._alert(now, d, job, f"{job} failed after {attempts} attempts: {msg}")
            return {"failed": msg}
        self.store.set_job(now, d, job, "PENDING", detail={**detail, "error": msg,
                                                           "retry_after": (now + RESEARCH_RETRY).isoformat()})
        return {"retry": msg}

    def _skip(self, now: datetime, d: str, job: str, reason: str) -> dict[str, Any]:
        self.store.set_job(now, d, job, "SKIPPED", detail={"reason": reason})
        self.store.event(now, "RESULT", f"{job} skipped: {reason}", day=d, job=job)
        return {"skipped": reason}

    def _simulated(self, d: str) -> bool:
        if self.simulated_feed:
            return True
        pre = self.store.job(d, PREOPEN)
        prov = _d(_d(pre["detail"] if pre else {}).get("provenance"))
        return prov.get("feed_mode") == "SIMULATED"

    async def _simulated_research_feed(self, d: str) -> bool:
        if self._simulated(d):
            return True
        feed = await self._call("get_feed_quality")
        return feed.get("feed_mode") == "SIMULATED"

    async def _daily(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        if self.api_owns_quantlab:
            return self._skip(now, d, DAILY, "the API runs the quantlab daily pipeline (LAT_QUANTLAB_DAILY_ENABLED)")
        if (self.store.job(d, DAILY) or {}).get("attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
            return self._research_fail(now, d, DAILY, "attempt budget spent", {})
        self.store.set_job(now, d, DAILY, "RUNNING", attempt=True)
        self.store.event(now, "ACTION", "quantlab daily pipeline (fetch-delta, build-panel, scan)", day=d, job=DAILY)
        try:
            if await self._simulated_research_feed(d):
                return self._skip(self.clock.now(), d, DAILY, "simulated feed: no real market-data downloads")
            factory = self.daily_job_factory or _default_daily_job
            job = factory(self.cfg.project_root, clock=self.clock.now, runner=self._daily_runner(td))
            summary = await job.run_once()
            last = job.last_done() if summary is None else None
        except AuthorityError:
            raise
        except Exception as exc:
            return self._research_fail(self.clock.now(), d, DAILY, _err(exc)[1], {})
        at = self.clock.now()
        if summary is None:
            if last is not None and last >= td.day:
                self.store.set_job(at, d, DAILY, "DONE", detail={"note": f"already up to date through {last}"})
                return {"done": "already up to date"}
            return self._research_fail(at, d, DAILY, "not due or attempt budget spent in daily_state.json", {})
        if summary.get("ok"):
            self.store.set_job(at, d, DAILY, "DONE", detail={"summary": summary})
            self.store.event(at, "RESULT", "quantlab daily pipeline done", day=d, job=DAILY, data=summary)
            return summary
        return self._research_fail(at, d, DAILY, "a pipeline step failed", {"summary": summary})

    def _daily_runner(self, td: TradingDay) -> Callable[[Sequence[str], Path, float], Any]:
        from services.quantlab.daily import StepResult

        async def run(args: Sequence[str], cwd: Path, timeout_s: float) -> StepResult:
            name = " ".join(args)
            blocked = await self._check_research_risk(td, self.clock.now())
            if blocked is not None:
                return StepResult(name, None, 0, blocked)
            out = await self._run_research_child(td, quantlab_argv(*args), cwd, min(timeout_s, self.cfg.step_timeout_s))
            return StepResult(name, out.returncode, out.seconds, out.tail[-800:])

        return run

    async def _crew(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        if self.cfg.lead:
            return self._skip(now, d, CREW, "the lead agent decides and delegates the crew's work (LAT_AUTOPILOT_LEAD)")
        if self.cfg.crew_ticks == 0:
            return self._skip(now, d, CREW, "LAT_AUTOPILOT_CREW_TICKS=0")
        row = self.store.job(d, CREW)
        detail = _d(row["detail"] if row else {})
        done = int(detail.get("ticks_done", 0))
        ticks = _l(detail.get("ticks"))
        attempts = int(detail.get("tick_attempts", 0))
        if done >= self.cfg.crew_ticks:
            self.store.set_job(now, d, CREW, "DONE")
            return {"done": done}
        if attempts >= MAX_RESEARCH_ATTEMPTS:
            return self._research_fail(now, d, CREW, "crew tick attempt budget spent", {}, attempts=attempts)
        self.store.set_job(now, d, CREW, "RUNNING", attempt=True, detail={"tick_attempts": attempts + 1})
        self.store.event(now, "ACTION", f"research-crew tick {done + 1}/{self.cfg.crew_ticks} (subprocess)", day=d,
                         job=CREW)
        try:
            out = await self._run_research_child(td, CREW_ARGV, self.cfg.project_root, self.cfg.crew_timeout_s)
        except AuthorityError:
            raise
        except Exception as exc:
            return self._research_fail(self.clock.now(), d, CREW, _err(exc)[1], {}, attempts=attempts + 1)
        at = self.clock.now()
        brief = _summarize_crew(out)
        ticks = [*ticks, {"at": at.isoformat(), **brief}][-12:]
        if out.returncode == CREW_LOCKED_EXIT:
            self.store.set_job(at, d, CREW, "DONE", detail={"ticks_done": done, "ticks": ticks,
                                                           "note": "crew store busy (another crew loop runs)"})
            return brief
        if not out.ok:
            return self._research_fail(at, d, CREW, f"crew tick exited {out.returncode}", {"ticks": ticks},
                                       attempts=attempts + 1)
        done += 1
        status = "DONE" if done >= self.cfg.crew_ticks else "PENDING"
        self.store.set_job(at, d, CREW, status, detail={"ticks_done": done, "ticks": ticks, "tick_attempts": 0,
                                                       "retry_after": None})
        self.store.event(at, "RESULT", f"crew tick {done}/{self.cfg.crew_ticks} exit {out.returncode}", day=d, job=CREW,
                         data=brief)
        return brief

    async def _shadow(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        if self.api_owns_quantlab:
            return self._skip(now, d, SHADOW, "the API's quantlab daily pipeline updates the shadow journal")
        daily = self.store.job(d, DAILY)
        if daily is None or daily["status"] != "DONE":
            return self._skip(now, d, SHADOW, "the daily pipeline did not complete; the shadow journal needs a fresh panel")
        if (self.store.job(d, SHADOW) or {}).get("attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
            return self._research_fail(now, d, SHADOW, "attempt budget spent", {})
        self.store.set_job(now, d, SHADOW, "RUNNING", attempt=True)
        self.store.event(now, "ACTION", "shadow journal update (existing promotion checks)", day=d, job=SHADOW)
        try:
            out = await self._run_research_child(td, quantlab_argv("shadow"), self.cfg.project_root,
                                                 self.cfg.step_timeout_s)
        except AuthorityError:
            raise
        except Exception as exc:
            return self._research_fail(self.clock.now(), d, SHADOW, _err(exc)[1], {})
        at = self.clock.now()
        if not out.ok:
            return self._research_fail(at, d, SHADOW, f"exit {out.returncode}: {out.tail[-200:]}", {})
        shadow = read_json(self.cfg.project_root / "experiments" / "quantlab" / "latest_shadow.json")
        checks = [c for k in ("equity", "crypto") for c in _l(_d(_d(shadow).get(k)).get("promotion_checks"))]
        self.store.set_job(at, d, SHADOW, "DONE", detail={"promotion_checks": checks[:20]})
        self.store.event(at, "RESULT", "shadow journal updated", day=d, job=SHADOW, data={"checks": len(checks)})
        return {"promotion_checks": len(checks)}

    async def _holdout(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        attempted = _d(self.store.get(K_HOLDOUT_ATTEMPTS, {}))
        for family, attempt in attempted.items():
            if _d(attempt).get("day") == d:
                detail = {"family": family, "error": "holdout outcome unknown after interruption"}
                self.store.set_job(now, d, HOLDOUT, "FAILED", detail=detail)
                self._alert(now, d, HOLDOUT, f"holdout outcome for {family} is unknown; inspect the vault")
                return detail
        if not self.cfg.holdout:
            return self._skip(now, d, HOLDOUT, "LAT_AUTOPILOT_HOLDOUT=false")
        if (self.store.job(d, HOLDOUT) or {}).get("attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
            return self._research_fail(now, d, HOLDOUT, "attempt budget spent", {})
        try:
            if await self._simulated_research_feed(d):
                return self._skip(self.clock.now(), d, HOLDOUT,
                                  "simulated workspace: the protected holdout is never evaluated here")
            res = await self._call("get_research_results", {"family": ""})
        except AuthorityError:
            raise
        except Exception as exc:
            self.store.set_job(now, d, HOLDOUT, "RUNNING", attempt=True)
            return self._research_fail(now, d, HOLDOUT, f"research results unavailable: {_err(exc)[1]}", {})
        candidates = sorted(str(_d(f).get("family")) for f in _l(res.get("families"))
                            if _d(f).get("registry_state") == "DEV_PASSED" and _d(f).get("passed_dev_gate") is True
                            and str(_d(f).get("family")) not in attempted)
        if not candidates:
            self.store.set_job(now, d, HOLDOUT, "DONE", detail={"note": "no DEV_PASSED family awaiting its holdout look"})
            return {"candidates": 0}
        family = candidates[0]
        argv = holdout_argv(family)
        blocked = await self._check_research_risk(td, self.clock.now())
        if blocked is not None:
            return {"deferred": blocked}
        now = self.clock.now()
        if not self._research_due(td, now):
            return {"deferred": "research window ended or insufficient time before pre-open"}
        # Recorded BEFORE the child starts: the autopilot never launches a second look for this family, even after a
        # crash mid-evaluation (the vault independently refuses a second look).
        self.store.put(now, K_HOLDOUT_ATTEMPTS, {**attempted, family: {"day": d, "at": now.isoformat()}})
        self.store.set_job(now, d, HOLDOUT, "RUNNING", attempt=True, detail={"family": family})
        self.store.event(now, "ACTION", f"protected-holdout look for DEV_PASSED family {family} (existing holdout "
                                        "command; the vault enforces one look)", day=d, job=HOLDOUT)
        out = await self._run_research_child(td, argv, self.cfg.project_root, self.cfg.step_timeout_s)
        at = self.clock.now()
        data = _d(out.json())
        detail = {"family": family, "returncode": out.returncode, "passed": data.get("passed"),
                  "refused": data.get("refused"), "strategy_id": data.get("strategy_id"),
                  "fail_reasons": _l(data.get("fail_reasons"))[:5]}
        self.store.set_job(at, d, HOLDOUT, "DONE" if out.ok or data.get("refused") else "FAILED", detail=detail)
        self.store.event(at, "RESULT", f"holdout {family}: passed={detail['passed']} refused={detail['refused']}",
                         day=d, job=HOLDOUT, data=detail)
        if not out.ok and not data.get("refused"):
            self._alert(at, d, HOLDOUT, f"holdout look for {family} exited {out.returncode}; it is never retried by "
                                        "the autopilot (inspect and resume by hand)")
        return detail

    def _supersede(self, day: str, now: datetime) -> dict[str, Any]:
        n = 0
        for j in self.store.jobs(day):
            if j["job"] in RESEARCH_SEQUENCE and j["status"] not in TERMINAL:
                self.store.set_job(now, day, j["job"], "SUPERSEDED",
                                   detail={"reason": "a later session's incremental research run catches up"})
                n += 1
        for job in RESEARCH_SEQUENCE:
            if self.store.job(day, job) is None:
                self.store.set_job(now, day, job, "SUPERSEDED", detail={"reason": "never ran; caught up later"})
        self.store.event(now, "DECISION", f"research for {day} superseded ({n} open steps)", day=day)
        return {"superseded": n}


    # ================================================================== lead agent (background child)
    async def _read_optional(self, name: str, args: dict[str, Any] | None, errors: list[str]) -> dict[str, Any]:
        try:
            return await self._call(name, args)
        except AuthorityError:
            raise
        except Exception as exc:
            errors.append(f"{name}: {_err(exc)[1]}")
            return {}

    async def _lead_briefing(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        """Read-only snapshot for the lead: tool outputs (with execution labels) plus the autopilot's own state."""
        d = td.day.isoformat()
        errors: list[str] = []
        status = await self._read_optional("get_session_status", None, errors)
        risk = await self._read_optional("get_risk_status", None, errors)
        perf = await self._read_optional("get_performance", None, errors)
        health = await self._read_optional("get_system_health", None, errors)
        feed = await self._read_optional("get_feed_quality", None, errors)
        sid = _d(_d(self.store.job(d, PREOPEN) or {}).get("detail")).get("session_id")
        rep = await self._read_optional("get_session_report", {"session_id": sid}, errors) if sid else {}
        outputs = [o for o in (status, risk, perf, health, feed, rep) if o]
        phase, _ = phase_at(now, lead_min=self.cfg.preopen_lead_min, grace_min=self.cfg.start_grace_min,
                            settle_min=self.cfg.settle_min)
        health_c, feed_c = compact_health(health, feed)
        state = self.lead_state()
        nxt = next_session(td.day)
        own = (LEAD, LEAD_ACTION_JOB)
        return {
            "asof": now.isoformat(), "phase": phase.value, "trading_day": td.public(),
            "next_session": nxt.public() if nxt else None,
            "provenance": {"execution_labels": sorted({str(o["execution_label"]) for o in outputs
                                                       if o.get("execution_label")}),
                           "is_simulated": any(bool(o.get("is_simulated")) for o in outputs) or self._simulated(d)},
            "standdown": self.store.get(K_STANDDOWN),
            "jobs": [{"job": j["job"], "status": j["status"], "attempts": j["attempts"],
                      "detail": {k: v for k, v in _d(j["detail"]).items() if k not in ("cycles",)}}
                     for j in self.store.jobs(d) if j["job"] != LEAD],
            "alerts": [{k: e[k] for k in ("ts", "job", "message")} for e in self.store.events(day=d, kind="ALERT",
                                                                                              limit=10)],
            "recent_events": [{k: e[k] for k in ("ts", "kind", "job", "message")}
                              for e in self.store.events(limit=40) if e["job"] not in own][-15:],
            "session": compact_session(status) if status else None,
            "risk": compact_risk(risk) if risk else None,
            "performance": {k: perf.get(k) for k in PERFORMANCE_KEYS} if perf else None,
            "session_report": compact_report(rep) if rep else None,
            "health": health_c if health else None, "feed": feed_c if feed else None,
            "lead_limits": {"turn": (state.cycles if state.day == d else 0) + 1, "max_turns": self.cfg.lead_max_cycles,
                            "session_actions_enabled": self.cfg.lead_session_actions,
                            "session_actions_taken_today": _l(_d(self.store.get(K_LEAD_ACTIONS, {})).get(d)),
                            "min_interval_min": self.cfg.lead_min_interval_min,
                            "max_interval_min": self.cfg.lead_max_interval_min},
            "errors": errors[:10],
        }

    async def _lead_launch(self, td: TradingDay, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        if self.lead_running():
            return {"deferred": "a lead cycle is already running"}
        briefing = await self._lead_briefing(td, now)
        path = write_briefing(self.cfg.lead_briefing_path, briefing)
        now = self.clock.now()
        prev = self.lead_state()
        cycles = (prev.cycles if prev.day == d else 0) + 1
        state = LeadState(d, cycles, now, None,
                          self.store.last_seq(EVIDENCE_KINDS, exclude_jobs=(LEAD, LEAD_ACTION_JOB)))
        self.store.put(now, K_LEAD, state.dump())
        self.store.set_job(now, d, LEAD, "RUNNING", attempt=True,
                           detail={"cycles_launched": cycles, "last_launch": now.isoformat()})
        self.store.event(now, "ACTION", f"lead-agent cycle {cycles}/{self.cfg.lead_max_cycles} launched (background "
                                        "child; trading never waits for it)", day=d, job=LEAD,
                         data={"briefing": str(path), "errors": briefing["errors"]})
        argv = lead_argv(self.cfg.lead_timeout_s, path, fake=self.cfg.lead_fake)
        self._lead_day = d
        self._lead_task = asyncio.ensure_future(self.runner(argv, self.cfg.project_root, self.cfg.lead_timeout_s + 120))
        return {"launched": cycles}

    async def _poll_lead(self) -> dict[str, Any] | None:
        task = self._lead_task
        if task is None or not task.done():
            return None
        self._lead_task = None
        d = self._lead_day or et_date(self.clock.now()).isoformat()
        now = self.clock.now()
        try:
            out = task.result()
        except asyncio.CancelledError:
            self.store.set_job(now, d, LEAD, "PENDING", detail={"last_error": "cancelled"})
            return {"cancelled": True}
        except AuthorityError:
            raise
        except Exception as exc:
            out = StepOutcome((), 1, 0.0, f"runner raised {type(exc).__name__}")
        summary = parse_cycle(out, locked_exit=CREW_LOCKED_EXIT)
        state = self.lead_state()
        minutes = summary.next_wake_minutes if summary.next_wake_minutes is not None else self.cfg.lead_max_interval_min
        minutes = max(self.cfg.lead_min_interval_min, min(self.cfg.lead_max_interval_min, minutes))
        if state.day == d:
            self.store.put(now, K_LEAD, LeadState(state.day, state.cycles, state.last_launch,
                                                  now + timedelta(minutes=minutes), state.evidence_seq).dump())
        row = self.store.job(d, LEAD)
        history = [*_l(_d(row["detail"] if row else {}).get("cycles")), {"at": now.isoformat(), **summary.public()}]
        self.store.set_job(now, d, LEAD, "PENDING", detail={"cycles": history[-12:], "last": summary.public(),
                                                           "next_wake": (now + timedelta(minutes=minutes)).isoformat()})
        self.store.event(now, "RESULT", f"lead cycle: {summary.lead_outcome or summary.error} via "
                                        f"{summary.lead_provider or '-'}; {len(summary.delegated)} delegated, "
                                        f"{len(summary.ran)} subagent runs, next turn in {minutes} min",
                         day=d, job=LEAD, data=summary.public())
        if not summary.ok or summary.lead_outcome not in ("OK", None):
            self._alert(now, d, LEAD, f"lead agent did not complete a turn ({summary.lead_outcome or summary.error}); "
                                      "trading and the deterministic research steps continue without it")
        actions = []
        for sa in summary.session_actions if summary.ok and summary.lead_outcome == "OK" else ():
            actions.append(await self._lead_session_action(d, sa, self.clock.now()))
        return {**summary.public(), "session_action_results": actions}

    async def _finish_lead(self, *, wait: bool) -> None:
        task = self._lead_task
        if task is None:
            return
        if wait:
            with contextlib.suppress(Exception):
                await asyncio.wait({task})
            await self._poll_lead()
            return
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self.store.event(self.clock.now(), "RESULT", "lead cycle cancelled: the autopilot is stopping (its "
                                                          "subagent tasks return to the queue)", day=self._lead_day,
                             job=LEAD)
        self._lead_task = None

    async def _lead_session_action(self, d: str, sa: Mapping[str, str], now: datetime) -> dict[str, Any]:
        """Execute one validated fail-safe request of the lead. Never starts, resumes or enlarges trading."""
        action, reason = sa["action"], sa["reason"]

        def refuse(why: str) -> dict[str, Any]:
            self.store.event(now, "REFUSED", f"lead request {action} refused: {why}", day=d, job=LEAD_ACTION_JOB,
                             data={"reason": reason})
            return {"action": action, "refused": why}

        if not self.cfg.lead_session_actions:
            return refuse("lead session requests are disabled (LAT_AUTOPILOT_LEAD_SESSION_ACTIONS=false)")
        taken = _d(self.store.get(K_LEAD_ACTIONS, {}))
        if action in _l(taken.get(d)):
            return refuse("already taken for this trading day")
        today = session_today(now)
        if action == "stand_down_today":
            td = trading_day(date.fromisoformat(d))
            if today is None or td is None or today.day != td.day:
                return refuse("the lead turn was not for today's session")
            if now >= today.close:
                return refuse("today's session is over")
            res = await self._stand_down_today(today, reason, now)
            if "refused" in res:
                return refuse(str(res["refused"]))
        elif action == "request_halt":
            await self._call("request_halt", {"reason": f"lead agent: {reason}"[:200]})
            res = {"halt_requested": True}
        else:
            return refuse("unknown request")
        self.store.put(now, K_LEAD_ACTIONS, {k: v for k, v in taken.items() if k >= d} | {d: [*_l(taken.get(d)), action]})
        self.store.event(now, "ACTION", f"lead request {action} executed: {reason}", day=d, job=LEAD_ACTION_JOB,
                         data=res)
        self._alert(now, d, LEAD_ACTION_JOB, f"lead agent: {action} ({reason})", dedupe=False)
        if action == "request_halt":
            await self._standdown_check(self.clock.now())
        return {"action": action, **res}

    async def _stand_down_today(self, td: TradingDay, reason: str, now: datetime) -> dict[str, Any]:
        d = td.day.isoformat()
        job = self.store.job(d, PREOPEN)
        status = job["status"] if job else None
        if status is None or status == "PENDING":
            self.store.set_job(now, d, PREOPEN, "SKIPPED", detail={"reason": f"lead agent stand-down: {reason}"})
            return {"session": "not requested today"}
        if status in ("RUNNING", "UNKNOWN"):
            return {"refused": "the session start outcome is still being resolved; decide again next turn"}
        sid = _d(job["detail"] if job else {}).get("session_id")
        if status != "DONE" or not sid:
            return {"refused": f"no autopilot session today (pre-open {status})"}
        snapshot = await self._call("get_session_status")
        active = _d(snapshot.get("active"))
        if active.get("session_id") != sid or active.get("state") not in ACTIVE_STATES:
            return {"refused": "today's autopilot session is not active"}
        if active.get("requested_by") != REQUESTED_BY:
            return {"refused": "the active session was not requested by the autopilot"}
        res = await self._call("stop_session", {"session_id": sid, "reason": f"lead agent: {reason}"[:200]})
        return {"session": "stop requested", "session_id": sid, "outcome": _d(res).get("outcome")}

    # ================================================================== report
    async def _report(self, day: str, now: datetime, *, final: bool) -> dict[str, Any]:
        errors: list[str] = []
        pre = self.store.job(day, PREOPEN)
        sid = _d(pre["detail"] if pre else {}).get("session_id")

        async def read(name: str, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
            if self.backend is None:
                return None
            try:
                return await self._call(name, args)
            except AuthorityError:
                raise
            except Exception as exc:
                errors.append(f"{name}: {_err(exc)[1]}")
                return None

        session_report = await read("get_session_report", {"session_id": sid}) if sid else None
        performance = await read("get_performance") if day == et_date(now).isoformat() else None
        if performance is None and day != et_date(now).isoformat():
            errors.append("daily performance unavailable for the historical report date")
        orders: list[dict[str, Any]] = []
        order_output = None
        sess = _d(_d(session_report).get("session"))
        start = _ts(sess.get("started_at"))
        if start is not None:
            order_output = await read("get_orders", {"limit": 50, "open_only": False})
            end = _ts(sess.get("ended_at")) or min(_ts(sess.get("ends_at")) or now, now)
            orders = [o for o in (_d(x) for x in _l(_d(order_output).get("orders")))
                      if (t := _ts(o.get("created_at"))) is not None and start <= t <= end]
        research = await read("get_research_results", {"family": ""}) if final else None
        shadow = None
        if final and (self.api_owns_quantlab or (self.store.job(day, SHADOW) or {}).get("status") == "DONE"):
            shadow = read_json(self.cfg.project_root / "experiments" / "quantlab" / "latest_shadow.json")
        td = trading_day(date.fromisoformat(day))
        rep = reports.build(day=day, now=now, final=final, jobs=self.store.jobs(day),
                            alerts=self.store.events(day=day, kind="ALERT"), session_report=session_report,
                            performance=performance, orders=orders, order_output=order_output,
                            simulated_feed=self._simulated(day), research=research, shadow=_d(shadow) or None,
                            standdown=self.store.get(K_STANDDOWN), trading_day=td.public() if td else None,
                            errors=errors)
        jp, mp = reports.write(self.cfg.report_dir, rep)
        if final:
            self.store.set_job(now, day, REPORT, "DONE", detail={"json": str(jp), "md": str(mp)})
            self.store.event(now, "REPORT", f"final report {jp.name}", day=day, job=REPORT)
        return {"json": str(jp), "md": str(mp), "status": rep["status"], "errors": errors}


def _default_daily_job(root: Path, *, clock: Callable[[], datetime], runner: Callable[..., Any]) -> Any:
    from services.quantlab.daily import STEPS, DailyJob

    return DailyJob(root, clock=clock, runner=runner, steps=tuple(s for s in STEPS if s[0] != "shadow"))


def status_snapshot(cfg: AutopilotConfig, store: Store | None, now: datetime, *, lock_pid: int | None) -> dict[str, Any]:
    """Read-only status: lock holder, phase, today's jobs, stand-down, recent alerts and the next actions."""
    out: dict[str, Any] = {"asof": now.isoformat(), "enabled": cfg.enabled, "running_pid": lock_pid,
                           "state_dir": str(cfg.state_dir), "report_dir": str(cfg.report_dir)}
    if store is None:
        phase, td = phase_at(now, lead_min=cfg.preopen_lead_min, grace_min=cfg.start_grace_min,
                             settle_min=cfg.settle_min)
        out.update({"phase": phase.value, "trading_day": td.public() if td else None, "jobs": [], "store": "absent"})
        return out
    ap = Autopilot(cfg, store, None)
    plan = ap.decide(now)
    days = sorted({j["day"] for j in store.jobs()})[-3:]
    last = last_completed(now)
    out.update({
        "phase": plan.phase.value, "trading_day": plan.day.public() if plan.day else None,
        "last_completed_session": last.day.isoformat() if last else None,
        "standdown": store.get(K_STANDDOWN), "last_tick": store.get(K_LAST_TICK),
        "last_intraday_check": store.get(K_LAST_CHECK),
        "holdout_attempts": store.get(K_HOLDOUT_ATTEMPTS, {}),
        "lead": {"enabled": cfg.lead, "state": store.get(K_LEAD), "actions": store.get(K_LEAD_ACTIONS, {})},
        "jobs": [{k: j[k] for k in ("day", "job", "status", "attempts", "updated_at")}
                 for d in days for j in store.jobs(d)],
        "recent_alerts": [{k: e[k] for k in ("ts", "day", "job", "message")} for e in store.events(kind="ALERT", limit=10)],
        "recent_events": [{k: e[k] for k in ("ts", "kind", "job", "message")} for e in store.events(limit=10)],
        "next": plan.public(),
    })
    return out
