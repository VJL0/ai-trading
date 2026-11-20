"""propose_paper_trade: evidence provenance, deterministic (untrusted-arithmetic-free) proposal construction,
and the REAL deterministic risk engine deciding."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from packages.contracts.enums import DecisionAction, Proposer, Side
from services.tools.handlers import ToolError, dispatch

from .fake_runtime import FakeToolRuntime
from .test_tools_schemas import GOOD_PROPOSAL


def _p(**patch: Any) -> dict[str, Any]:
    return {**GOOD_PROPOSAL, **patch}


async def _primed(actor: str = "t") -> FakeToolRuntime:
    rt = FakeToolRuntime()
    await dispatch(rt, "get_signals", {}, actor=actor)
    await dispatch(rt, "get_predictions", {}, actor=actor)
    return rt


async def test_evidence_must_come_from_this_sessions_tool_outputs() -> None:
    rt = FakeToolRuntime()
    for fake_ids in (["pred_SPY_1"], ["snap_SPY_99"], ["ev_made_up"]):
        with pytest.raises(ToolError) as ei:
            await dispatch(rt, "propose_paper_trade", _p(evidence_ids=fake_ids), actor="t")
        assert ei.value.status == 422 and "not obtained" in ei.value.message()
    assert "submit_agent_proposal" not in rt.calls
    await dispatch(rt, "get_predictions", {"symbol": "SPY"}, actor="t")
    out = await dispatch(rt, "propose_paper_trade", _p(evidence_ids=["pred_SPY_1"]), actor="t")
    assert out["outcome"] == "SUBMITTED"


async def test_ids_issued_to_one_session_are_not_valid_for_another() -> None:
    rt = await _primed("alice")
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, "propose_paper_trade", _p(evidence_ids=["pred_SPY_1"]), actor="mallory")
    assert ei.value.status == 422


async def test_persisted_signal_ids_are_citable_without_listing() -> None:
    rt = FakeToolRuntime()  # never called get_signals in this session
    out = await dispatch(rt, "propose_paper_trade", _p(evidence_ids=["sig_SPY_1"]), actor="fresh")
    assert out["outcome"] == "SUBMITTED"
    assert rt.proposals[-1].signal_id == "sig_SPY_1"


async def test_at_least_one_evidence_id_must_belong_to_the_symbol() -> None:
    rt = await _primed()
    await dispatch(rt, "get_model_report", {}, actor="t")
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, "propose_paper_trade", _p(evidence_ids=["sig_QQQ_1", "reg_test_1"]), actor="t")
    assert ei.value.status == 422 and "belong to SPY" in ei.value.message()
    ok = await dispatch(rt, "propose_paper_trade", _p(evidence_ids=["reg_test_1", "pred_SPY_1"]), actor="t")
    assert ok["outcome"] == "SUBMITTED"


async def test_proposal_is_built_deterministically_and_never_trusts_agent_arithmetic() -> None:
    rt = await _primed()
    out = await dispatch(rt, "propose_paper_trade", _p(limit_price="500.049", stop_price="495.001"), actor="claude-code")
    prop = rt.proposals[-1]
    assert prop.proposer == Proposer.AGENT_TOOL and prop.strategy_id == "agent:claude-code"
    assert prop.action == DecisionAction.TRADE and prop.side == Side.BUY
    assert prop.limit_price == Decimal("500.04")  # BUY limit rounded to the passive (lower) tick
    assert prop.stop_price == Decimal("495.01")  # stop rounded toward the limit (smaller cushion, never larger)
    assert prop.quantity == Decimal(3) and isinstance(prop.quantity, Decimal)
    # Conservative deterministic loss: min(agent stop, policy default stop floored to tick) -> never understated.
    default_stop = Decimal("492.53")  # floor(500.04 * (1 - 150/10000), 0.01)
    assert prop.max_planned_loss == Decimal(3) * (Decimal("500.04") - default_stop)
    assert out["decision"]["recomputed_max_loss"] is not None
    assert prop.max_planned_loss is not None
    assert Decimal(out["decision"]["recomputed_max_loss"]) <= prop.max_planned_loss
    assert prop.evidence_ids[0].startswith("snap_SPY_")  # fresh system snapshot is always attached
    assert prop.data_asof == rt.now() and prop.dataset_version == rt.dataset_version
    assert prop.policy_version_seen == rt.policy.version
    assert "thesis:" in prop.rationale and "counterarguments:" in prop.rationale
    assert out["requested"] == {"symbol": "SPY", "side": "BUY", "quantity": "3", "limit_price": "500.04",
                                "stop_price": "495.01"}


async def test_agent_cannot_supply_its_own_loss_or_sizing_fields() -> None:
    rt = await _primed()
    for extra in ({"max_planned_loss": "0.01"}, {"notional": "1"}, {"proposer": "HUMAN"}, {"strategy_id": "x"},
                  {"order_type": "MARKET"}, {"extended_hours": True}, {"requires_human_approval": False}):
        with pytest.raises(ToolError) as ei:
            await dispatch(rt, "propose_paper_trade", _p(**extra), actor="t")
        assert ei.value.status == 422
    assert rt.proposals == []


async def test_risky_proposals_are_rejected_by_the_deterministic_risk_engine() -> None:
    rt = await _primed()
    big = await dispatch(rt, "propose_paper_trade", _p(quantity="10"), actor="t")  # 5000 notional > 2000 limit
    assert big["outcome"] == "REJECTED" and big["client_order_id"] is None
    assert "max_order_notional" in big["decision"]["failed_checks"]
    assert big["decision"]["verdict"] == "REJECTED"
    far = await dispatch(rt, "propose_paper_trade", _p(quantity="1", limit_price="520.00", stop_price="515.00"), actor="t")
    assert far["outcome"] == "REJECTED" and "limit_vs_mid" in far["decision"]["failed_checks"]
    penny = await dispatch(rt, "propose_paper_trade",
                           _p(symbol="SPY", quantity="1", side="SELL", stop_price=None), actor="t")
    assert penny["outcome"] == "REJECTED"  # SELL without an existing long: long-only enforced by risk engine
    assert "long_only" in penny["decision"]["failed_checks"] or "positions_consistent" in penny["decision"]["failed_checks"]


async def test_stop_validation() -> None:
    rt = await _primed()
    for stop in ("500.04", "600.00", "0"):
        with pytest.raises(ToolError) as ei:
            await dispatch(rt, "propose_paper_trade", _p(stop_price=stop), actor="t")
        assert ei.value.status == 422
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, "propose_paper_trade", _p(side="SELL", stop_price="495.00"), actor="t")
    assert ei.value.status == 422 and "BUY" in ei.value.message()
    assert rt.proposals == []


async def test_sell_limit_rounds_up_and_claims_zero_entry_loss() -> None:
    rt = await _primed()
    await dispatch(rt, "propose_paper_trade", _p(side="SELL", limit_price="500.041", stop_price=None), actor="t")
    prop = rt.proposals[-1]
    assert prop.side == Side.SELL and prop.limit_price == Decimal("500.05") and prop.max_planned_loss == Decimal(0)
    assert prop.stop_price is None


async def test_runtime_refusal_is_403_and_rate_limit_is_reported() -> None:
    rt = await _primed()
    rt.refuse = True
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, "propose_paper_trade", GOOD_PROPOSAL, actor="t")
    assert ei.value.status == 403
    rt.refuse, rt.rate_limited = False, True
    out = await dispatch(rt, "propose_paper_trade", GOOD_PROPOSAL, actor="t")
    assert out["outcome"] == "RATE_LIMITED" and out["decision"] is None and "6/hour" in out["reason"]


async def test_each_proposal_gets_fresh_ids() -> None:
    rt = await _primed()
    a = await dispatch(rt, "propose_paper_trade", _p(quantity="1"), actor="t")
    b = await dispatch(rt, "propose_paper_trade", _p(quantity="2"), actor="t")
    assert a["proposal_id"] != b["proposal_id"] and a["causal_id"] != b["causal_id"]
