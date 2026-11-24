"""Every tool dispatched against the REAL ``services.api.runtime.Runtime`` (SIMULATED: synthetic replay feed,
deterministic FakeBroker, no model, no network). This catches drift between the runtime facade's real data
shapes and the strict tool output contracts, and shows that mutating tools go through the real risk engine,
the real per-hour agent proposal limit and the real audit trail."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import FeedMode, GatewayKind, TradingMode
from packages.core.config import Settings
from services.tools.handlers import ToolError, dispatch
from services.tools.registry import READ_ONLY_TOOL_NAMES, TOOL_REGISTRY

pytestmark = pytest.mark.timeout(180)

ACTOR = "it-real"


def _settings(root: Path, **over: Any) -> Settings:
    base: dict[str, Any] = dict(project_root=root, use_keychain=False, mode=TradingMode.PAPER,
                                gateway=GatewayKind.FAKE, feed_mode=FeedMode.SIMULATED, reconcile_interval_s=3600)
    base.update(over)
    return Settings(**base)


async def _runtime(root: Path, **over: Any) -> Any:
    from services.api.runtime import Runtime

    rt = Runtime(_settings(root, **over))
    report = await rt.reconciler.startup_recovery()
    rt._after_reconcile(report)  # pyright: ignore[reportPrivateUsage]
    assert rt.sim_clock is not None, "SIMULATED feed mode must run on the replay clock"
    for _ in range(2):  # let the replay feed deliver fresh point-in-time quotes
        rt.sim_clock.step(60)
        rt._on_sim_tick(rt.now())  # pyright: ignore[reportPrivateUsage]
    return rt


@pytest.fixture
async def rt(tmp_path: Path) -> AsyncIterator[Any]:
    runtime = await _runtime(tmp_path)
    try:
        yield runtime
    finally:
        await runtime.aclose()


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _proposal(snap: dict[str, Any], qty: str = "1", **patch: Any) -> dict[str, Any]:
    return {"symbol": snap["symbol"], "side": "BUY", "quantity": qty, "limit_price": snap["quote"]["ask"],
            "stop_price": None, "thesis": f"Point-in-time snapshot {snap['snapshot_id']} shows a tight spread.",
            "evidence_ids": [snap["snapshot_id"]], "counterarguments": ["synthetic data: no evidence of an edge"],
            **patch}


async def test_every_read_tool_validates_against_the_real_runtime(rt: Any) -> None:
    args = {"get_market_snapshot": {"symbol": "SPY"}, "get_bars": {"symbol": "SPY", "limit": 5},
            "search_memory": {"query": "edge"}, "get_feed_quality": {}, "get_predictions": {},
            "get_model_report": {}, "get_signals": {"limit": 5},
            "get_portfolio": {}, "get_orders": {}, "get_risk_status": {},
            "get_session_status": {}, "list_schedules": {}, "explain_decisions": {"limit": 5}, "get_performance": {},
            "get_activity": {"limit": 5}, "get_system_health": {}, "get_capabilities": {},
            "get_research_results": {}, "get_opportunities": {}}
    # get_order needs an order (covered below); get_session_report needs a session
    # (tests/integration/sessions/test_sessions_api_e2e.py).
    assert set(args) == set(READ_ONLY_TOOL_NAMES) - {"get_order", "get_session_report"}
    for name, a in args.items():
        out = await dispatch(rt, name, a, actor=ACTOR)  # dispatch validates against the strict output model
        assert out["execution_label"] == "SIMULATED" and out["is_simulated"] is True, name
        assert out["asof"] == rt.now().isoformat().replace("+00:00", "Z"), name
        TOOL_REGISTRY[name].output_model.model_validate(out)
    snap = await dispatch(rt, "get_market_snapshot", {"symbol": "SPY"}, actor=ACTOR)
    assert snap["quote"] is not None and snap["feed"] == "SYNTHETIC" and _ts(snap["data_asof"]) <= _ts(snap["asof"])
    assert _ts(snap["quote"]["available_at_ts"]) <= _ts(snap["asof"])
    bars = await dispatch(rt, "get_bars", {"symbol": "SPY", "limit": 5}, actor=ACTOR)
    assert 0 < len(bars["bars"]) <= 5
    assert all(_ts(b["t"]) < _ts(bars["asof"]) for b in bars["bars"])  # nothing from after the decision clock
    assert bars["last_available_at"] is not None and _ts(bars["last_available_at"]) <= _ts(bars["asof"])
    quality = await dispatch(rt, "get_feed_quality", {}, actor=ACTOR)
    assert quality["status"] == "OK"
    risk = await dispatch(rt, "get_risk_status", {}, actor=ACTOR)
    assert risk["mode"] == "PAPER" and risk["live_execution_compiled"] is False and risk["entries_locked"] is False


async def test_agent_proposal_flows_through_real_risk_engine_and_audit(rt: Any) -> None:
    snap = await dispatch(rt, "get_market_snapshot", {"symbol": "SPY"}, actor=ACTOR)
    ok = await dispatch(rt, "propose_paper_trade", _proposal(snap), actor=ACTOR)
    assert ok["outcome"] == "SUBMITTED", ok
    assert ok["decision"]["verdict"] == "APPROVED" and ok["client_order_id"]
    coid = ok["client_order_id"]

    detail = await dispatch(rt, "get_order", {"client_order_id": coid}, actor=ACTOR)
    assert detail["order"]["client_order_id"] == coid and detail["order"]["execution_label"] == "SIMULATED"
    assert detail["intent"] is not None and "signature" not in detail["intent"]
    assert detail["transitions"], detail

    cancel = await dispatch(rt, "cancel_paper_order", {"client_order_id": coid}, actor=ACTOR)
    assert cancel["outcome"] in ("CANCEL_REQUESTED", "ALREADY_TERMINAL")

    stored = rt.db.query_one("SELECT payload_json FROM proposals WHERE proposal_id=?", (ok["proposal_id"],))
    assert stored is not None and '"proposer":"AGENT_TOOL"' in stored["payload_json"].replace(" ", "")
    assert f'"strategy_id":"agent:{ACTOR}"' in stored["payload_json"].replace(" ", "")
    audit = rt.db.query("SELECT source FROM audit_events WHERE event_type='AGENT_PROPOSAL'")
    assert [a["source"] for a in audit] == [f"agent:{ACTOR}"]

    risky = await dispatch(rt, "propose_paper_trade", _proposal(snap, qty="100000"), actor=ACTOR)
    assert risky["outcome"] == "REJECTED" and risky["client_order_id"] is None
    assert "max_order_notional" in risky["decision"]["failed_checks"]


async def test_real_hourly_agent_proposal_limit(tmp_path: Path) -> None:
    rt = await _runtime(tmp_path, agent_max_proposals_per_hour=2)
    try:
        snap = await dispatch(rt, "get_market_snapshot", {"symbol": "QQQ"}, actor=ACTOR)
        outcomes = [(await dispatch(rt, "propose_paper_trade", _proposal(snap, qty="100000"), actor=ACTOR))["outcome"]
                    for _ in range(3)]
        assert outcomes == ["REJECTED", "REJECTED", "RATE_LIMITED"]
    finally:
        await rt.aclose()


async def test_halt_then_cycle_and_proposals_are_blocked(rt: Any) -> None:
    halt = await dispatch(rt, "request_halt", {"reason": "agent saw abnormal spreads"}, actor=ACTOR)
    assert halt["entries_locked"] is True
    risk = await dispatch(rt, "get_risk_status", {}, actor=ACTOR)
    assert risk["entries_locked"] is True and risk["halts"][0]["halt_id"] == halt["halt_id"]
    assert risk["halts"][0]["source"] == f"agent:{ACTOR}"
    snap = await dispatch(rt, "get_market_snapshot", {"symbol": "SPY"}, actor=ACTOR)
    blocked = await dispatch(rt, "propose_paper_trade", _proposal(snap), actor=ACTOR)
    assert blocked["outcome"] == "REJECTED" and blocked["client_order_id"] is None
    assert "health_entries_unlocked" in blocked["decision"]["failed_checks"]  # the halt, not something else
    cycle = await dispatch(rt, "run_quant_cycle", {}, actor=ACTOR)
    assert cycle["outcome"] != "SUBMITTED" and cycle["client_order_id"] is None
    assert rt.db.query_one("SELECT COUNT(*) n FROM orders")["n"] == 0


async def test_quant_cycle_signals_become_citable_evidence(rt: Any) -> None:
    cycle = await dispatch(rt, "run_quant_cycle", {"symbols": ["SPY", "QQQ"]}, actor=ACTOR)
    assert cycle["outcome"] in ("NO_TRADE", "REJECTED", "SUBMITTED", "BLOCKED"), cycle
    sigs = await dispatch(rt, "get_signals", {"limit": 10}, actor=ACTOR)
    assert sigs["signals"] and sigs["skipped_invalid"] == 0
    assert {s["symbol"] for s in sigs["signals"]} <= {"SPY", "QQQ"}
    assert all(_ts(s["features_asof"]) <= _ts(s["generated_at"]) <= _ts(sigs["asof"]) for s in sigs["signals"])


async def test_research_only_runtime_refuses_every_tool(tmp_path: Path) -> None:
    from services.api.runtime import Runtime

    rt = Runtime(_settings(tmp_path, mode=TradingMode.RESEARCH_ONLY, gateway=GatewayKind.FAKE))
    try:
        for name in ("get_portfolio", "request_halt"):
            with pytest.raises(ToolError) as ei:
                await dispatch(rt, name, {} if name == "get_portfolio" else {"reason": "should be refused"},
                               actor=ACTOR)
            assert ei.value.status == 403
    finally:
        await rt.aclose()


def _train_tiny_registry(experiments_dir: Path) -> str:
    """A real (tiny, SYNTHETIC-data) walk-forward model registry, exactly as `make models` would persist it."""
    from services.research.costs import CostModel
    from services.research.models.config import SuiteConfig
    from services.research.models.registry import registry_root, save_registry
    from services.research.models.walkforward import train_suite
    from services.research.synthetic import make_synthetic_bars

    small = {"num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 40}
    cfg = SuiteConfig(lgbm_params={**small, "num_boost_round": 40}, rank_params={**small, "num_boost_round": 20},
                      cost_params={**small, "num_boost_round": 20, "min_data_in_leaf": 100},
                      catboost_params={"iterations": 60, "depth": 3, "learning_rate": 0.1}, n_folds=3)
    bars = make_synthetic_bars(("SPY", "QQQ", "AAPL"), timeframe="1Min", n_sessions=8, seed=11, drift=0.0, vol=0.012,
                               latency_s=1.0, dataset_version="synthetic-noise-tools-test")
    base = registry_root(experiments_dir)
    return save_registry(train_suite(bars, CostModel(), base, config=cfg), base).registry_version


async def test_model_tools_against_a_real_trained_registry(tmp_path: Path) -> None:
    rt = await _runtime(tmp_path)
    try:
        version = _train_tiny_registry(rt.settings.experiments_dir)
        report = await dispatch(rt, "get_model_report", {}, actor=ACTOR)
        assert report["registry_version"] == version and report["selection_metric"] == "net_sharpe"
        assert report["is_simulated"] is True and report["targets"]
        for tgt in report["targets"]:
            assert tgt["n_candidates"] >= 1
            assert tgt["selected_model"] is None or tgt["selected"] is None or \
                tgt["selected"]["model_id"] == tgt["selected_model"]
        assert report["walk_forward"] is not None and "purge_minutes" in report["walk_forward"]
        assert report["multiple_testing"] is not None and isinstance(report["multiple_testing"]["survivors"], list)

        preds = await dispatch(rt, "get_predictions", {"symbol": "SPY"}, actor=ACTOR)
        assert preds["registry_version"] == version and len(preds["predictions"]) == 1, preds
        for p in preds["predictions"]:
            assert p["symbol"] == "SPY" and _ts(p["features_asof"]) <= _ts(p["asof"]) <= _ts(preds["asof"])
        # Prediction and registry ids handed out here are citable evidence for this session.
        snap = await dispatch(rt, "get_market_snapshot", {"symbol": "SPY"}, actor=ACTOR)
        cited = [version, *[p["prediction_id"] for p in preds["predictions"]][:1]]
        out = await dispatch(rt, "propose_paper_trade",
                             _proposal(snap, evidence_ids=[*cited, snap["snapshot_id"]]), actor=ACTOR)
        assert out["outcome"] in ("SUBMITTED", "REJECTED"), out
    finally:
        await rt.aclose()
