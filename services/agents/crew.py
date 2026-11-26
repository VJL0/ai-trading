"""The crew loop: plan -> pick a task -> pick a provider -> run one headless agent turn -> record.

Boundaries (enforced by ``tests/unit/agents/test_import_boundaries.py``): this module never imports
``services.api``, the order ledger, risk, positions, sessions, broker or market-data execution code, never opens
the holdout or promotes a strategy, and never starts the API. Agent factor specs are validated with
``services.quantlab.factors.validate_spec`` and run through ``run_agent_experiment`` (development data only, hourly
budget, duplicate/redundancy refusal, the same gates as every family). The crew records the ledger's result, never
numbers an agent reports.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any, cast

from pydantic import ValidationError

from . import mechanisms
from .config import MAX_RETRY_AT, CrewConfig, default_project_root
from .context import build_prompt, clean, clean_tree, read_ledger_digest
from .contracts import (
    REQUEUE_OUTCOMES,
    AgentProvider,
    Invocation,
    McpServerSpec,
    Provider,
    RunOutcome,
    RunResult,
    TaskKind,
    TaskState,
)
from .planner import TaskRequest, gather_state, plan, used_slots
from .schemas import (
    HypothesisOutput,
    ImprovementOutput,
    MarketAnalysisOutput,
    ReviewOutput,
    output_schema,
    parse_output,
)
from .store import AGENT_NOTE, CREW_EVENT, INTERRUPTED, LEDGER_DEV, CrewStore, ProviderState, TaskRow

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]
ExperimentRunner = Callable[[Path, dict[str, Any], str], dict[str, Any]]
McpSpecFactory = Callable[[Path, str], McpServerSpec]

AGENT_EXPERIMENTS_PER_HOUR_FALLBACK = 6
SHUTDOWN_GRACE_S = 5.0


def utcnow() -> datetime:
    return datetime.now(UTC)


def default_experiment_runner(root: Path, spec: dict[str, Any], actor: str) -> dict[str, Any]:
    from services.quantlab.views import run_agent_experiment

    raw_root = os.environ.get("LAT_PROJECT_ROOT")
    child_root = Path(raw_root) if raw_root else default_project_root({})
    if not child_root.is_absolute() or child_root.resolve() != root.resolve():
        return {"accepted": False, "status": "PROJECT_ROOT_MISMATCH",
                "error": "QuantLab subprocess project root differs from the crew project root"}
    return run_agent_experiment(root, spec, actor)


def default_mcp_spec(root: Path, agent_name: str) -> McpServerSpec:
    from .providers import research_mcp_spec

    return research_mcp_spec(root, agent_name)


def experiments_per_hour() -> int:
    try:
        from services.quantlab.views import AGENT_EXPERIMENTS_PER_HOUR
    except Exception:  # pragma: no cover - quantlab import failure must not stop the crew from idling safely
        return AGENT_EXPERIMENTS_PER_HOUR_FALLBACK
    return int(AGENT_EXPERIMENTS_PER_HOUR)


def build_providers(config: CrewConfig) -> dict[Provider, AgentProvider]:
    """The real CLI providers (imported lazily so tests and ``--fake`` never need them)."""
    from .providers import ClaudeCodeProvider, CodexProvider

    out: dict[Provider, AgentProvider] = {}
    for p in config.providers:
        out[p] = ClaudeCodeProvider(binary=config.claude_bin) if p is Provider.CLAUDE else \
            CodexProvider(binary=config.codex_bin)
    return out


@dataclass
class TickResult:
    enqueued: list[str] = field(default_factory=list)
    ran: list[dict[str, Any]] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    idle_reason: str | None = None

    def public(self) -> dict[str, Any]:
        return {"enqueued": self.enqueued, "ran": self.ran, "expired": self.expired, "idle_reason": self.idle_reason}


@dataclass
class _InFlight:
    task: TaskRow
    provider: Provider
    run_id: str
    invocation: Invocation
    thread: threading.Thread
    result: list[RunResult] = field(default_factory=list)


class Crew:
    def __init__(self, config: CrewConfig, store: CrewStore, providers: Mapping[Provider, AgentProvider], *,
                 experiment_runner: ExperimentRunner | None = None, clock: Clock | None = None,
                 mcp_spec_factory: McpSpecFactory | None = None, experiments_per_hour: int | None = None) -> None:
        self.config = config
        self.store = store
        self.providers = dict(providers)
        self.experiment_runner = experiment_runner or default_experiment_runner
        self.clock = clock or utcnow
        self.mcp_spec_factory = mcp_spec_factory or default_mcp_spec
        self._per_hour = experiments_per_hour
        self._stop = threading.Event()

    # -------------------------------------------------------------------------------------------- helpers
    @property
    def per_hour(self) -> int:
        if self._per_hour is None:
            self._per_hour = experiments_per_hour()
        return self._per_hour

    @property
    def root(self) -> Path:
        return self.config.project_root

    def stop(self) -> None:
        self._stop.set()

    def provider_order(self) -> list[Provider]:
        return [p for p in self.config.providers if p in self.providers]

    def refresh_providers(self, now: datetime) -> None:
        """Probe a disabled provider after cooldown, using login status for authentication failures."""
        for p in self.provider_order():
            st = self.store.provider_state(p)
            if not st.disabled_reason or st.cooling(now):
                continue
            try:
                agent = self.providers[p]
                probe = agent.availability
                if st.disabled_reason.startswith(RunOutcome.AUTH_REQUIRED.value):
                    probe = getattr(agent, "auth_available", probe)
                ok, reason = probe()
            except Exception as exc:  # availability must not raise, but a broken provider stays disabled
                ok, reason = False, f"availability raised {type(exc).__name__}"
            if ok:
                self.store.put_provider_state(ProviderState(p, st.cooldown_until, st.consecutive_limits, None), now)
                self.store.add_note("event", {"event": "PROVIDER_ENABLED", "reason": clean(reason, 200)[0]}, now=now,
                                    provider=p, evidence_status=CREW_EVENT)

    def choose_provider(self, task: TaskRow, now: datetime, busy: set[Provider] | None = None) -> Provider | None:
        """Available, not cooling down, under the daily cap, not excluded (the author of a reviewed hypothesis),
        not already running; prefer one that has not failed this task, then the fewest runs today, then config order."""
        busy = busy or set()
        failed: dict[str, int] = {}
        for r in self.store.task_runs(task.id):
            if r.get("outcome") in (RunOutcome.INVALID_OUTPUT.value, RunOutcome.ERROR.value, RunOutcome.TIMEOUT.value):
                failed[r["provider"]] = failed.get(r["provider"], 0) + 1
        cands: list[tuple[int, int, int, Provider]] = []
        for i, p in enumerate(self.provider_order()):
            if p in busy or p == task.exclude_provider:
                continue
            st = self.store.provider_state(p)
            if st.disabled_reason or st.cooling(now):
                continue
            today = self.store.runs_today(p, now)
            if today >= self.config.daily_runs_per_provider:
                continue
            cands.append((failed.get(p.value, 0), today, i, p))
        return min(cands)[3] if cands else None

    def _enqueue(self, req: TaskRequest, now: datetime) -> str | None:
        tid, created = self.store.enqueue(req.kind, req.key, req.payload, now=now, priority=req.priority,
                                          author_provider=req.author_provider, exclude_provider=req.exclude_provider,
                                          max_attempts=self.config.max_attempts, expires_at=req.expires_at,
                                          parent_id=req.parent_id)
        return tid if created else None

    # ------------------------------------------------------------------------------------------------ tick
    def plan(self, now: datetime) -> list[str]:
        ledger = read_ledger_digest(self.root, now)
        state = gather_state(self.store, self.root, ledger, now, self.per_hour)
        return [tid for req in plan(state, self.config, now) if (tid := self._enqueue(req, now))]

    def tick(self) -> TickResult:
        now = self.clock()
        res = TickResult()
        res.expired = self.store.expire(now)
        self.refresh_providers(now)
        res.enqueued = self.plan(now)
        runnable = self.store.runnable(now)
        if not runnable:
            res.idle_reason = "no runnable task"
            return res
        busy: set[Provider] = set()
        flights: list[_InFlight] = []
        reserved_slots = 0
        waiting_for_slots = False
        for task in runnable:
            if self._stop.is_set() or len(flights) >= self.config.max_concurrent:
                break
            p = self.choose_provider(task, self.clock(), busy)
            if p is None:
                continue
            slots = 0
            if task.kind is TaskKind.HYPOTHESIS and mechanisms.is_expressible(
                    str(task.payload.get("mechanism", "")), self.root):
                slots = min(self._spec_budget(task), self._free_slots(self.clock()) - reserved_slots)
                if slots < 1:
                    waiting_for_slots = True
                    continue
            claimed_at = self.clock()
            claimed = self.store.claim(task.id, claimed_at)
            if claimed is None:
                continue  # another crew process took it
            if slots:
                claimed = replace(claimed, payload=claimed.payload | {"slots": slots})
            flight = self._launch(claimed, p, claimed_at)
            if flight is None:
                continue
            busy.add(p)
            flights.append(flight)
            reserved_slots += slots
        if not flights:
            res.idle_reason = ("waiting for experiment headroom" if waiting_for_slots else
                               "no provider available (cooldown, daily cap, disabled or excluded)")
            return res
        for f in flights:
            res.ran.append(self._await(f))
        return res

    def _launch(self, task: TaskRow, provider: Provider, now: datetime) -> _InFlight | None:
        run_id = self.store.start_run(task.id, provider, now)
        try:
            workdir = self.config.runs_dir / run_id
            workdir.mkdir(parents=True, exist_ok=True)
            ledger = read_ledger_digest(self.root, now)
            prompt = build_prompt(task, self.store, ledger, specs_per_task=self._spec_budget(task), project_root=self.root)
            inv = Invocation(provider=provider, prompt=prompt,
                             output_schema=output_schema(task.kind, specs_per_task=self._spec_budget(task)),
                             mcp_server=self.mcp_spec_factory(self.root, f"crew-{provider.value}"),
                             workdir=workdir, timeout_s=self.config.run_timeout_s, max_turns=self.config.max_turns,
                             model=self.config.model_for(provider))
        except Exception as exc:
            log.exception("could not prepare run for task %s", task.id)
            self.store.end_run(run_id, RunOutcome.ERROR.value, now=self.clock(), detail=f"prepare: {type(exc).__name__}")
            self._fail_attempt(task, provider, RunResult(RunOutcome.ERROR, detail=f"prepare: {type(exc).__name__}"))
            return None
        holder: list[RunResult] = []
        agent = self.providers[provider]

        def work() -> None:
            try:
                cancellable = getattr(agent, "run_cancellable", None)
                if callable(cancellable):
                    runner = cast(Callable[[Invocation, threading.Event], RunResult], cancellable)
                    holder.append(runner(inv, self._stop))
                else:
                    holder.append(agent.run(inv))
            except Exception as exc:  # providers must not raise; map defensively
                holder.append(RunResult(RunOutcome.ERROR, detail=f"provider raised {type(exc).__name__}"))

        th = threading.Thread(target=work, name=f"crew-{provider.value}-{run_id}", daemon=True)
        th.start()
        return _InFlight(task, provider, run_id, inv, th, holder)

    def _spec_budget(self, task: TaskRow) -> int:
        slots = task.payload.get("slots")
        n = int(slots) if isinstance(slots, int) and slots > 0 else self.config.specs_per_task
        return max(1, min(n, self.config.specs_per_task))

    def _await(self, f: _InFlight) -> dict[str, Any]:
        while f.thread.is_alive():
            f.thread.join(timeout=0.5)
            if self._stop.is_set() and f.thread.is_alive():
                f.thread.join(timeout=SHUTDOWN_GRACE_S)
                if f.thread.is_alive():
                    return self._interrupt(f)
        result = f.result[0] if f.result else RunResult(RunOutcome.ERROR, detail="no result")
        if self._stop.is_set():
            return self._interrupt(f)
        return self.record(f.task, f.provider, f.run_id, result)

    def _interrupt(self, f: _InFlight) -> dict[str, Any]:
        now = self.clock()
        self.store.end_run(f.run_id, INTERRUPTED, now=now, detail="crew stopping (signal)")
        self.store.requeue(f.task.id, now=now, consume_attempt=False)
        self.store.add_note("event", {"event": "INTERRUPTED", "detail": "requeued; attempt not consumed"}, now=now,
                            task_id=f.task.id, provider=f.provider, evidence_status=CREW_EVENT)
        return {"task_id": f.task.id, "kind": f.task.kind.value, "provider": f.provider.value, "outcome": INTERRUPTED}

    # --------------------------------------------------------------------------------------------- record
    def record(self, task: TaskRow, provider: Provider, run_id: str, result: RunResult) -> dict[str, Any]:
        now = self.clock()
        outcome = result.outcome
        parsed: Any = None
        detail = clean(result.detail or "", 600)[0]
        if outcome is RunOutcome.OK:
            try:
                parsed = parse_output(task.kind, result.output)
            except ValidationError as exc:
                outcome = RunOutcome.INVALID_OUTPUT
                detail = f"crew schema check: {exc.error_count()} error(s)"
            if isinstance(parsed, HypothesisOutput) and parsed.mechanism != task.payload.get("mechanism"):
                outcome = RunOutcome.INVALID_OUTPUT
                detail = "hypothesis mechanism does not match its assignment"
        self.store.end_run(run_id, outcome.value, now=now, usage=dict(result.usage), detail=detail)
        summary: dict[str, Any] = {"task_id": task.id, "kind": task.kind.value, "provider": provider.value,
                                   "outcome": outcome.value, "run_id": run_id}
        if outcome is RunOutcome.OK and parsed is not None:
            st = self.store.provider_state(provider)
            if st.consecutive_limits or st.cooldown_until:
                self.store.put_provider_state(ProviderState(provider, None, 0, st.disabled_reason), now)
            task_result = self._handle_ok(task, provider, parsed)
            self.store.finish(task.id, TaskState.DONE, task_result, now=self.clock(), author_provider=provider)
            summary["result"] = task_result
        elif outcome in REQUEUE_OUTCOMES:
            until = self._cool_down(provider, outcome, result, detail, now)
            self.store.requeue(task.id, now=now, consume_attempt=False)
            summary["cooldown_until"] = until.isoformat()
        else:
            row = self._fail_attempt(task, provider, RunResult(outcome, detail=detail))
            summary["task_state"] = row.state.value if row else None
        return summary

    def _cool_down(self, provider: Provider, outcome: RunOutcome, result: RunResult, detail: str,
                   now: datetime) -> datetime:
        st = self.store.provider_state(provider)
        n = st.consecutive_limits + 1
        until = now + self.config.backoff(n)
        if outcome is RunOutcome.USAGE_LIMIT and result.retry_at is not None:
            retry_at = result.retry_at if result.retry_at.tzinfo else result.retry_at.replace(tzinfo=UTC)
            if now < retry_at <= now + MAX_RETRY_AT:
                until = retry_at
        disabled = f"{outcome.value}: {detail}"[:300] if outcome in (RunOutcome.AUTH_REQUIRED,
                                                                     RunOutcome.UNAVAILABLE) else st.disabled_reason
        self.store.put_provider_state(ProviderState(provider, until, n, disabled), now)
        self.store.add_note("event", {"event": "COOLDOWN", "outcome": outcome.value, "until": until.isoformat(),
                                      "consecutive_limits": n, "detail": detail[:200]},
                            now=now, provider=provider, evidence_status=CREW_EVENT)
        return until

    def _fail_attempt(self, task: TaskRow, provider: Provider, result: RunResult) -> TaskRow | None:
        now = self.clock()
        row = self.store.requeue(task.id, now=now, consume_attempt=True)
        if row is not None and row.state is TaskState.FAILED:
            self.store.set_result(task.id, {"reason": result.outcome.value, "detail": result.detail[:300],
                                            "last_provider": provider.value}, now=now)
        self.store.add_note("event", {"event": "RUN_FAILED", "outcome": result.outcome.value,
                                      "detail": result.detail[:300],
                                      "task_state": row.state.value if row else None},
                            now=now, task_id=task.id, provider=provider, evidence_status=CREW_EVENT)
        return row

    # ------------------------------------------------------------------------------------------ per kind
    def _handle_ok(self, task: TaskRow, provider: Provider, out: Any) -> dict[str, Any]:
        if isinstance(out, HypothesisOutput):
            return self._handle_hypothesis(task, provider, out)
        now = self.clock()
        mech = str(task.payload.get("mechanism") or "") or None
        if isinstance(out, ReviewOutput):
            self.store.add_note("review", clean_tree(out.model_dump(mode="json"), 2000) | {"hypothesis_task_id": task.parent_id},
                                now=now, task_id=task.id, provider=provider, mechanism=mech)
            return {"verdict": out.verdict.value, "hypothesis_task_id": task.parent_id, "reviewer": provider.value}
        if isinstance(out, MarketAnalysisOutput):
            safe = clean_tree(out.model_dump(mode="json"), 2000)
            self.store.add_note("market_analysis", {k: safe[k] for k in ("regime_summary", "observations", "data_gaps")},
                                now=now, task_id=task.id, provider=provider)
            seeds = 0
            for s in out.hypothesis_seeds:
                if mechanisms.get(s.mechanism) is None:
                    continue
                rationale, withheld = clean(s.rationale, 2000)
                if withheld:
                    continue
                self.store.add_note("seed", {"rationale": rationale}, now=now, task_id=task.id, provider=provider,
                                    mechanism=s.mechanism)
                seeds += 1
            for gap in safe["data_gaps"]:
                self.store.add_note("data_request", {"request": gap, "source": "market_analysis"}, now=now,
                                    task_id=task.id, provider=provider)
            return {"seeds": seeds, "observations": len(out.observations)}
        if isinstance(out, ImprovementOutput):
            for prop in out.proposals:
                self.store.add_note("improvement", clean_tree(prop.model_dump(mode="json"), 2000), now=now, task_id=task.id,
                                    provider=provider)
            return {"proposals": len(out.proposals)}
        raise TypeError(type(out).__name__)

    def _handle_hypothesis(self, task: TaskRow, provider: Provider, out: HypothesisOutput) -> dict[str, Any]:
        from services.quantlab.factors import SpecError, validate_spec

        now = self.clock()
        mech = str(task.payload.get("mechanism") or out.mechanism)
        hyp = clean_tree(out.model_dump(mode="json"), 2000)
        self.store.add_note("hypothesis", hyp, now=now, task_id=task.id, provider=provider, mechanism=mech)
        for req in hyp["data_requests"]:
            self.store.add_note("data_request", {"request": req, "source": "hypothesis"}, now=now, task_id=task.id,
                                provider=provider, mechanism=mech)
        experiments: list[dict[str, Any]] = []
        refusals: list[dict[str, Any]] = []

        def refuse(spec: dict[str, Any], status: str, error: str, prior: Any = None) -> None:
            body = clean_tree({"spec": spec, "status": status, "error": error, "prior": prior}, 400)
            refusals.append(body)
            self.store.add_note("refusal", body, now=self.clock(), task_id=task.id, provider=provider, mechanism=mech,
                                evidence_status=CREW_EVENT)

        budget = self._spec_budget(task)
        specs = [s.model_dump(mode="json") for s in out.specs]
        if len(specs) > budget:
            for s in specs[budget:]:
                refuse(s, "OVER_SPEC_BUDGET", f"only {budget} specs per task are considered")
            specs = specs[:budget]
        mechanism = mechanisms.get(mech, self.root)
        if mechanism is not None and not mechanism.expressible_now and not hyp["data_requests"]:
            self.store.add_note("data_request", {"request": mechanism.data_needed, "source": "mechanism"},
                                now=now, task_id=task.id, provider=provider, mechanism=mech, evidence_status=CREW_EVENT)
        if specs and (mechanism is None or not mechanism.expressible_now):
            for s in specs:
                refuse(s, "MECHANISM_NOT_EXPRESSIBLE", f"{mech} needs additional DSL features or development data. "
                       "Submit a data request.")
            specs = []
        actor = f"crew-{provider.value}-{task.id[:8]}"
        exhausted = False
        for raw in specs:
            intent = raw.pop("intent", "")
            try:
                clean_spec = validate_spec(dict(raw))
            except SpecError as exc:
                refuse(raw, "INVALID_SPEC", str(exc))
                continue
            if exhausted or self._free_slots(self.clock()) < 1:
                exhausted = True
                refuse(clean_spec, "NO_HEADROOM", "agent experiment budget for this hour is used up")
                continue
            try:
                res = self.experiment_runner(self.root, clean_spec, actor)
            except Exception as exc:
                res = {"accepted": True, "status": "ERROR", "spec": clean_spec, "error": type(exc).__name__}
            if not res.get("accepted"):
                status = str(res.get("status") or "REFUSED")
                err = str(res.get("error") or "")
                if "budget exhausted" in err:
                    exhausted = True
                    status = "NO_HEADROOM"
                if status == "DATA_UNAVAILABLE":
                    request = err or (mechanism.data_needed if mechanism else "") or "Development data for the factor"
                    body = clean_tree({"request": request, "source": "experiment"}, 400)
                    self.store.add_note("data_request", body, now=self.clock(), task_id=task.id, provider=provider,
                                        mechanism=mech, evidence_status=CREW_EVENT)
                refuse(clean_spec, status, err, res.get("prior"))
                continue
            ledger = res.get("result") if isinstance(res.get("result"), dict) else {}
            body = clean_tree({"spec": clean_spec, "intent": intent, "status": str(res.get("status")), "actor": actor,
                               "passed_dev_gate": ledger.get("passed_dev_gate") if ledger else None,
                               "ledger": ledger, "error": str(res.get("error") or "")}, 600)
            experiments.append(body)
            self.store.add_note("experiment", body, now=self.clock(), task_id=task.id, provider=provider,
                                mechanism=mech, evidence_status=LEDGER_DEV)
        return {"mechanism": mech, "hypothesis": {k: hyp[k] for k in ("economic_rationale", "why_novel",
                                                                       "falsification", "data_requests", "specs")},
                "experiments": experiments, "refusals": refusals, "evidence_status": AGENT_NOTE}

    def _free_slots(self, now: datetime) -> int:
        ledger = read_ledger_digest(self.root, now)
        return self.per_hour - used_slots(self.store, ledger, now)

    # ------------------------------------------------------------------------------------------------- run
    def next_wake(self, now: datetime) -> float:
        """Seconds to sleep when idle: IDLE_SLEEP_S, or sooner when a cooldown ends."""
        wait = float(self.config.idle_sleep_s)
        for p in self.provider_order():
            st = self.store.provider_state(p)
            if st.cooldown_until and st.cooldown_until > now:
                wait = min(wait, max(1.0, (st.cooldown_until - now).total_seconds()))
        return wait

    def run(self, *, once: bool = False, install_signals: bool = True) -> list[dict[str, Any]]:
        from .store import SingleInstanceLock

        history: list[dict[str, Any]] = []
        prev: dict[int, Any] = {}
        if install_signals and threading.current_thread() is threading.main_thread():
            def _handler(signum: int, frame: FrameType | None) -> None:
                self._stop.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                prev[sig] = signal.signal(sig, _handler)
        try:
            with SingleInstanceLock(self.config.lock_path):
                recovered = self.store.recover(self.clock())
                if recovered:
                    history.append({"recovered": recovered})
                while not self._stop.is_set():
                    res = self.tick()
                    history.append(res.public())
                    if once:
                        break
                    if not res.ran:
                        self._stop.wait(self.next_wake(self.clock()))
        finally:
            for sig, h in prev.items():
                signal.signal(sig, h)
        return history


def cooldown_remaining(store: CrewStore, provider: Provider, now: datetime) -> timedelta | None:
    st = store.provider_state(provider)
    return st.cooldown_until - now if st.cooling(now) and st.cooldown_until else None
