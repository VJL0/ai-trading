"""Planner: state -> research tasks to enqueue. :func:`plan` is a pure function; :func:`gather_state` reads the crew
store, the trial-ledger digest and artifact timestamps.

Rules (each produces an idempotent key, so re-planning the same state enqueues nothing):

a. REVIEW for each finished hypothesis task with >= 1 spec the trial ledger accepted (status DONE) and no review
   yet. Excludes the author provider and expires after ``REVIEW_WAIT_HOURS`` (no self-review).
b. HYPOTHESIS when the agent experiment budget has >= 1 free slot this hour, no hypothesis task is queued or
   running, and there is an untried mechanism or fresh evidence. The mechanism is chosen by a UCB-style score
   over the expressible mechanisms (attempts, dev-gate passes,
   refusals and refuted reviews), tilted by fresh market-analysis seeds.
c. MARKET_ANALYSIS once per new opportunities/scan ``asof``.
d. IMPROVEMENT when >= 10 failed/refused agent trials or failed crew runs accumulated since the last improvement
   task, at most one per UTC day.

Nothing new -> no tasks, so the crew idles without spending quota.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from . import mechanisms
from .config import CrewConfig
from .context import LedgerDigest, mechanism_record, results_root
from .contracts import Provider, RunOutcome, TaskKind, TaskState
from .store import CrewStore

PRIORITY: Final[dict[TaskKind, int]] = {TaskKind.REVIEW: 40, TaskKind.HYPOTHESIS: 30,
                                        TaskKind.MARKET_ANALYSIS: 20, TaskKind.IMPROVEMENT: 10}
IMPROVEMENT_THRESHOLD: Final = 10
SEED_BONUS: Final = 0.3
FAILED_RUN_OUTCOMES: Final = frozenset({RunOutcome.INVALID_OUTPUT.value, RunOutcome.ERROR.value,
                                        RunOutcome.TIMEOUT.value})


@dataclass(frozen=True)
class TaskRequest:
    kind: TaskKind
    key: str
    payload: dict[str, Any]
    priority: int
    author_provider: Provider | None = None
    exclude_provider: Provider | None = None
    expires_at: datetime | None = None
    parent_id: str | None = None


@dataclass(frozen=True)
class HypothesisRecord:
    task_id: str
    author: Provider | None
    mechanism: str
    accepted_experiments: int
    review_payload: dict[str, Any]


@dataclass(frozen=True)
class PlannerState:
    existing_keys: frozenset[str] = frozenset()
    finished_hypotheses: tuple[HypothesisRecord, ...] = ()
    open_hypotheses: int = 0
    agent_trials_last_hour: int = 0
    experiments_per_hour: int = 6
    mechanism_record: dict[str, dict[str, int]] = field(default_factory=dict)
    seeds: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hypothesis_tasks_by_mechanism: dict[str, int] = field(default_factory=dict)
    hypothesis_evidence: str = ""
    tried_on_evidence: frozenset[str] = frozenset()
    scan_asof: str | None = None
    failures_since_improvement: int = 0
    last_improvement_at: datetime | None = None


def free_slots(state: PlannerState) -> int:
    return max(0, state.experiments_per_hour - state.agent_trials_last_hour)


def mechanism_scores(record: dict[str, dict[str, int]], seeds: dict[str, tuple[str, ...]]) -> dict[str, float]:
    """UCB1-style: mean reward per hypothesis task + exploration bonus (+ a bonus for fresh seeds)."""
    total = sum(int(record.get(m, {}).get("tasks", 0)) for m in mechanisms.EXPRESSIBLE)
    scores: dict[str, float] = {}
    for m in mechanisms.EXPRESSIBLE:
        r = record.get(m, {})
        n = int(r.get("tasks", 0))
        reward = (1.0 * r.get("dev_passes", 0) + 0.5 * r.get("supported", 0) - 0.5 * r.get("refuted", 0)
                  - 0.25 * r.get("refusals", 0))
        mean = reward / n if n else 0.0
        bonus = math.sqrt(2.0 * math.log(total + 1.0) / (n + 1.0)) if total else 1.0
        scores[m] = round(mean + bonus + (SEED_BONUS if seeds.get(m) else 0.0), 6)
    return scores


def choose_mechanism(record: dict[str, dict[str, int]], seeds: dict[str, tuple[str, ...]],
                     tried_on_evidence: frozenset[str] = frozenset()) -> str | None:
    scores = mechanism_scores(record, seeds)
    scores = {m: score for m, score in scores.items() if m not in tried_on_evidence or seeds.get(m)}
    if not scores:
        return None
    order = {m: i for i, m in enumerate(mechanisms.EXPRESSIBLE)}
    return max(scores, key=lambda m: (scores[m], -order[m]))


def plan(state: PlannerState, config: CrewConfig, now: datetime) -> list[TaskRequest]:
    out: list[TaskRequest] = []
    keys = set(state.existing_keys)

    def add(req: TaskRequest) -> None:
        if req.key not in keys:
            keys.add(req.key)
            out.append(req)

    # (a) independent review of hypotheses with accepted ledger results
    for h in state.finished_hypotheses:
        if h.author is None or h.accepted_experiments < 1:
            continue
        add(TaskRequest(TaskKind.REVIEW, f"review:{h.task_id}", h.review_payload, PRIORITY[TaskKind.REVIEW],
                        author_provider=None, exclude_provider=h.author,
                        expires_at=now + timedelta(hours=config.review_wait_hours), parent_id=h.task_id))

    # (b) new hypothesis only when the hourly experiment budget has room and none is in flight
    slots = free_slots(state)
    if slots >= 1 and state.open_hypotheses == 0:
        mech = choose_mechanism(state.mechanism_record, state.seeds, state.tried_on_evidence)
        if mech is not None:
            k = state.hypothesis_tasks_by_mechanism.get(mech, 0) + 1
            add(TaskRequest(TaskKind.HYPOTHESIS, f"hypothesis:{mech}:{k}",
                            {"mechanism": mech, "slots": min(config.specs_per_task, slots),
                             "seeds": list(state.seeds.get(mech, ()))[:3], "evidence": state.hypothesis_evidence},
                            PRIORITY[TaskKind.HYPOTHESIS]))

    # (c) one market analysis per new scan/opportunities asof
    if state.scan_asof:
        add(TaskRequest(TaskKind.MARKET_ANALYSIS, f"market_analysis:{state.scan_asof}",
                        {"scan_asof": state.scan_asof}, PRIORITY[TaskKind.MARKET_ANALYSIS]))

    # (d) improvement pass after enough new failures, at most once per UTC day
    if state.failures_since_improvement >= IMPROVEMENT_THRESHOLD:
        today = now.astimezone(UTC).date()
        if state.last_improvement_at is None or state.last_improvement_at.astimezone(UTC).date() < today:
            add(TaskRequest(TaskKind.IMPROVEMENT, f"improvement:{today.isoformat()}",
                            {"failures_since_last": state.failures_since_improvement,
                             "since": state.last_improvement_at.isoformat() if state.last_improvement_at else None},
                            PRIORITY[TaskKind.IMPROVEMENT]))
    return out


# ------------------------------------------------------------------------------------------- state gathering
def scan_asof(root: Path) -> str | None:
    """The opportunities/scan ``asof`` (or the artifacts' mtime when the file carries none)."""
    out = results_root(root)
    for name in ("latest_scan.json", "latest_selection.json"):
        p = out / name
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
            asof = data.get("asof") if isinstance(data, dict) else None
        except (OSError, ValueError):
            asof = None
        if asof:
            return str(asof)[:32]
        try:
            return datetime.fromtimestamp(p.stat().st_mtime, UTC).strftime("mtime-%Y%m%dT%H%M%SZ")
        except OSError:
            continue
    return None


def _review_payload(task_id: str, author: Provider | None, result: dict[str, Any]) -> dict[str, Any]:
    exps = [e for e in result.get("experiments", []) if isinstance(e, dict)]
    return {"hypothesis_task_id": task_id, "author_provider": author.value if author else None,
            "mechanism": result.get("mechanism"), "hypothesis": result.get("hypothesis", {}),
            "experiments": exps[:6], "refusals": result.get("refusals", [])[:6]}


def completed_experiments(result: dict[str, Any]) -> int:
    return sum(1 for exp in result.get("experiments", []) if isinstance(exp, dict) and exp.get("status") == "DONE")


def used_slots(store: CrewStore, ledger: LedgerDigest, now: datetime) -> int:
    """Agent experiments in the last hour: the ledger's F_agent DEV rows, or the crew's own accepted experiments
    when that is larger (a stale or unreadable ledger must not make the crew overspend)."""
    crew = sum(1 for n in store.notes(kind="experiment", since=now - timedelta(hours=1), limit=1000)
               if n.body.get("status") in ("DONE", "ERROR", "TIMEOUT"))
    return max(ledger.agent_dev_last_hour, crew)


def gather_state(store: CrewStore, root: Path, ledger: LedgerDigest, now: datetime,
                 experiments_per_hour: int) -> PlannerState:
    tasks = store.tasks(limit=-1)
    hyps = [t for t in tasks if t.kind is TaskKind.HYPOTHESIS]
    scan = scan_asof(root)
    evidence = hashlib.sha256(json.dumps(
        [ledger.agent_dev_total, ledger.families, ledger.research.get("updated_at"), scan], sort_keys=True,
    ).encode()).hexdigest()
    finished: list[HypothesisRecord] = []
    for t in hyps:
        if t.state is not TaskState.DONE or not t.result:
            continue
        accepted = completed_experiments(t.result)
        finished.append(HypothesisRecord(t.id, t.author_provider, str(t.payload.get("mechanism", "")), accepted,
                                         _review_payload(t.id, t.author_provider, t.result)))
    by_mech: dict[str, int] = {}
    last_hyp: dict[str, datetime] = {}
    for t in hyps:
        m = str(t.payload.get("mechanism", ""))
        by_mech[m] = by_mech.get(m, 0) + 1
        last_hyp[m] = max(last_hyp.get(m, t.created), t.created)
    seeds: dict[str, list[str]] = {}
    for n in store.notes(kind="seed", limit=200):
        m = n.mechanism or ""
        if not mechanisms.is_expressible(m):
            continue
        if m in last_hyp and n.at <= last_hyp[m]:
            continue  # already consumed by a later hypothesis task on this mechanism
        seeds.setdefault(m, []).append(str(n.body.get("rationale", ""))[:400])
    impr = [t for t in tasks if t.kind is TaskKind.IMPROVEMENT]
    last_impr = max((t.created for t in impr), default=None)
    since = last_impr or datetime(1970, 1, 1, tzinfo=UTC)
    failures = 0
    for n in store.notes(kind=("experiment", "refusal"), since=since, limit=100_000):
        if n.kind == "refusal" or n.body.get("passed_dev_gate") is not True:
            failures += 1
    failures += sum(1 for r in store.runs_since(since) if r.get("outcome") in FAILED_RUN_OUTCOMES)
    return PlannerState(
        agent_trials_last_hour=used_slots(store, ledger, now),
        existing_keys=frozenset(t.key for t in tasks) |
        frozenset(f"review:{t.parent_id}" for t in tasks if t.kind is TaskKind.REVIEW and t.parent_id),
        finished_hypotheses=tuple(finished),
        open_hypotheses=sum(1 for t in hyps if t.state in (TaskState.QUEUED, TaskState.RUNNING)),
        experiments_per_hour=experiments_per_hour,
        mechanism_record=mechanism_record(store),
        seeds={k: tuple(v) for k, v in seeds.items()},
        hypothesis_tasks_by_mechanism=by_mech,
        hypothesis_evidence=evidence,
        tried_on_evidence=frozenset(str(t.payload.get("mechanism", "")) for t in hyps
                                    if t.payload.get("evidence") == evidence),
        scan_asof=scan,
        failures_since_improvement=failures,
        last_improvement_at=last_impr,
    )
