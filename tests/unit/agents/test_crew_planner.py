from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from services.agents import mechanisms
from services.agents.context import LedgerDigest, read_ledger_digest
from services.agents.contracts import Provider, TaskKind, TaskState
from services.agents.planner import (
    IMPROVEMENT_THRESHOLD,
    HypothesisRecord,
    PlannerState,
    choose_mechanism,
    gather_state,
    mechanism_scores,
    plan,
)
from services.agents.store import CrewStore

from .test_crew_support import T0, make_config


def _keys(reqs: list) -> set[str]:
    return {r.key for r in reqs}


def _no_headroom(**kw: object) -> PlannerState:
    return PlannerState(agent_trials_last_hour=6, experiments_per_hour=6, **kw)  # type: ignore[arg-type]


def test_idle_when_nothing_new(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    assert plan(_no_headroom(), cfg, T0) == []
    # everything already enqueued -> nothing
    st = PlannerState(existing_keys=frozenset({"market_analysis:2026-11-24"}), scan_asof="2026-11-24",
                      agent_trials_last_hour=6)
    assert plan(st, cfg, T0) == []


def test_review_rule_excludes_author_and_expires(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, review_wait_hours=24)
    h = HypothesisRecord("abc", Provider.CLAUDE, "momentum", 1, {"hypothesis_task_id": "abc"})
    refused_only = HypothesisRecord("def", Provider.CODEX, "reversal", 0, {})
    reqs = plan(_no_headroom(finished_hypotheses=(h, refused_only)), cfg, T0)
    assert len(reqs) == 1
    r = reqs[0]
    assert r.kind is TaskKind.REVIEW and r.key == "review:abc" and r.exclude_provider is Provider.CLAUDE
    assert r.parent_id == "abc" and r.expires_at == T0 + timedelta(hours=24)
    # idempotent: once the review exists, nothing new
    assert plan(_no_headroom(finished_hypotheses=(h,), existing_keys=frozenset({"review:abc"})), cfg, T0) == []


def test_hypothesis_rule_needs_headroom_and_no_open_hypothesis(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, specs_per_task=3)
    reqs = plan(PlannerState(agent_trials_last_hour=4, experiments_per_hour=6), cfg, T0)
    assert [r.kind for r in reqs] == [TaskKind.HYPOTHESIS]
    assert reqs[0].payload["slots"] == 2  # min(SPECS_PER_TASK, free slots)
    assert reqs[0].payload["mechanism"] in mechanisms.EXPRESSIBLE
    assert plan(PlannerState(agent_trials_last_hour=6), cfg, T0) == []
    assert plan(PlannerState(agent_trials_last_hour=0, open_hypotheses=1), cfg, T0) == []


def test_market_analysis_once_per_asof(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    reqs = plan(_no_headroom(scan_asof="2026-11-24"), cfg, T0)
    assert _keys(reqs) == {"market_analysis:2026-11-24"}
    assert plan(_no_headroom(scan_asof="2026-11-24", existing_keys=frozenset(_keys(reqs))), cfg, T0) == []
    assert _keys(plan(_no_headroom(scan_asof="2026-11-25", existing_keys=frozenset(_keys(reqs))), cfg, T0)) == {
        "market_analysis:2026-11-25"}


def test_improvement_threshold_and_daily_limit(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    assert plan(_no_headroom(failures_since_improvement=IMPROVEMENT_THRESHOLD - 1), cfg, T0) == []
    reqs = plan(_no_headroom(failures_since_improvement=IMPROVEMENT_THRESHOLD), cfg, T0)
    assert _keys(reqs) == {"improvement:2026-11-25"}
    # one already ran today -> none until tomorrow
    st = _no_headroom(failures_since_improvement=25, last_improvement_at=T0 - timedelta(hours=2))
    assert plan(st, cfg, T0) == []
    assert _keys(plan(st, cfg, T0 + timedelta(days=1))) == {"improvement:2026-11-26"}


def test_ucb_prefers_untried_then_rewards_passes_and_penalizes_refuted() -> None:
    base = {m: {"tasks": 0, "attempts": 0, "refusals": 0, "dev_passes": 0, "refuted": 0, "supported": 0}
            for m in mechanisms.KEYS}
    rec = json.loads(json.dumps(base))
    rec["momentum"].update(tasks=3, attempts=6)
    assert choose_mechanism(rec, {}) != "momentum"  # untried mechanisms explored first
    rec = json.loads(json.dumps(base))
    for m in mechanisms.EXPRESSIBLE:
        rec[m].update(tasks=4, attempts=8)
    rec["volatility_risk"]["dev_passes"] = 2
    assert choose_mechanism(rec, {}) == "volatility_risk"
    rec["volatility_risk"]["refuted"] = 6
    assert choose_mechanism(rec, {}) != "volatility_risk"
    # refusals count against; a fresh market-analysis seed tilts the choice
    rec = json.loads(json.dumps(base))
    for m in mechanisms.EXPRESSIBLE:
        rec[m].update(tasks=2, attempts=4)
    rec["momentum"]["refusals"] = 4
    scores = mechanism_scores(rec, {})
    assert scores["momentum"] < scores["reversal"]
    assert choose_mechanism(rec, {"attention": ("abnormal volume in small caps",)}) == "attention"
    # only expressible mechanisms are ever chosen
    assert set(mechanism_scores(base, {})) == set(mechanisms.EXPRESSIBLE)


def _ledger_with_agent_trials(root: Path, n: int) -> None:
    from services.quantlab.ledger import TrialLedger
    from services.quantlab.paths import results_root

    led = TrialLedger(results_root(root) / "trials.sqlite3")
    for i in range(n):
        spec = {"expression": f"rank(mom({20 + i}))", "universe": "liquid1000", "n": 20, "rebalance": 21,
                "regime": "none"}
        led.record(family="F_agent", variant=f"v{i}", spec=spec, data_fingerprint="fp", stage="DEV",
                   metrics={"sharpe": 0.3}, status="OK")
        led.record(family="F_agent", variant=f"WALK_FORWARD:v{i}", spec={"wf_of": "F_agent"}, data_fingerprint="fp",
                   stage="WF", metrics={"sharpe": 0.1}, status="FAIL", note="dsr 0.2 < 0.95; spa p 0.4 > 0.1")


def test_gather_state_reads_ledger_headroom(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _ledger_with_agent_trials(tmp_path, 6)
    store = CrewStore(tmp_path / "crew.sqlite3")
    led = read_ledger_digest(tmp_path, now)
    assert led.available and led.agent_dev_last_hour == 6 and led.families["F_agent"]["wf_fail"] == 6
    st = gather_state(store, tmp_path, led, now, 6)
    assert st.agent_trials_last_hour == 6
    assert plan(st, make_config(tmp_path), now) == []  # no headroom -> no hypothesis, nothing else new
    assert read_ledger_digest(tmp_path, now + timedelta(hours=2)).agent_dev_last_hour == 0


def test_gather_state_from_store_end_to_end(tmp_path: Path) -> None:
    store = CrewStore(tmp_path / "crew.sqlite3")
    cfg = make_config(tmp_path)
    tid, _ = store.enqueue(TaskKind.HYPOTHESIS, "hypothesis:momentum:1", {"mechanism": "momentum"}, now=T0,
                           author_provider=Provider.CODEX)
    store.claim(tid, T0)
    store.finish(tid, TaskState.DONE, {"mechanism": "momentum", "experiments": [{"status": "DONE"}]}, now=T0)
    (tmp_path / "experiments" / "quantlab").mkdir(parents=True)
    (tmp_path / "experiments" / "quantlab" / "latest_scan.json").write_text(json.dumps({"asof": "2026-11-24"}))
    st = gather_state(store, tmp_path, LedgerDigest(available=False), T0, 6)
    reqs = plan(st, cfg, T0)
    kinds = {r.kind: r for r in reqs}
    assert set(kinds) == {TaskKind.REVIEW, TaskKind.HYPOTHESIS, TaskKind.MARKET_ANALYSIS}
    assert kinds[TaskKind.REVIEW].exclude_provider is Provider.CODEX
    assert kinds[TaskKind.HYPOTHESIS].payload["mechanism"] != "momentum"
    for r in reqs:
        store.enqueue(r.kind, r.key, r.payload, now=T0)
    st2 = gather_state(store, tmp_path, LedgerDigest(available=False), T0, 6)
    assert plan(st2, cfg, T0) == []  # idempotent: a queued hypothesis blocks another; keys exist


def test_hypotheses_stop_after_exploration_without_new_evidence(tmp_path: Path) -> None:
    store = CrewStore(tmp_path / "crew.sqlite3")
    cfg = make_config(tmp_path)
    ledger = LedgerDigest(available=True)
    for _ in mechanisms.EXPRESSIBLE:
        req, = plan(gather_state(store, tmp_path, ledger, T0, 6), cfg, T0)
        task_id, _ = store.enqueue(req.kind, req.key, req.payload, now=T0)
        store.claim(task_id, T0)
        store.finish(task_id, TaskState.DONE, {"experiments": []}, now=T0)
    assert plan(gather_state(store, tmp_path, ledger, T0, 6), cfg, T0) == []
    ledger.agent_dev_total += 1
    assert [req.kind for req in plan(gather_state(store, tmp_path, ledger, T0, 6), cfg, T0)] == [TaskKind.HYPOTHESIS]
