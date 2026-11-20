"""Failure injection for the tool interface: runtime exceptions, contract-violating runtime data, session expiry,
no blind retries, fail-closed outputs."""

from __future__ import annotations

import time
from typing import Any

import pytest

from services.tools import session as session_mod
from services.tools.handlers import ToolError, dispatch
from services.tools.session import IssuedIds, WindowLimiter, state_for
from tests.unit.tools.fake_runtime import FakeToolRuntime
from tests.unit.tools.test_tools_schemas import GOOD_PROPOSAL, VALID_ARGS


@pytest.mark.parametrize(("tool", "method"), [
    ("get_market_snapshot", "market_snapshot"), ("get_bars", "market_bars"), ("get_feed_quality", "market_quality"),
    ("get_predictions", "predictions"), ("get_model_report", "model_report"), ("get_signals", "signals"),
    ("get_portfolio", "portfolio"), ("get_orders", "orders"), ("get_order", "order_detail"), ("get_risk_status", "status"),
])
async def test_read_failure_is_503_without_internal_detail(tool: str, method: str) -> None:
    rt = FakeToolRuntime()
    rt.fail.add(method)
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, tool, VALID_ARGS[tool], actor="t")
    assert ei.value.status == 503
    assert "injected" not in ei.value.message() and "RuntimeError" in ei.value.message()


@pytest.mark.parametrize(("tool", "method"), [
    ("propose_paper_trade", "submit_agent_proposal"), ("request_halt", "halt"), ("cancel_paper_order", "cancel"),
    ("run_quant_cycle", "run_cycle"),
])
async def test_mutating_failure_is_unknown_and_never_retried(tool: str, method: str) -> None:
    rt = FakeToolRuntime()
    await dispatch(rt, "get_signals", {}, actor="t")
    rt.fail.add(method)
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, tool, VALID_ARGS[tool], actor="t")
    assert ei.value.status == 500 and "UNKNOWN" in ei.value.message() and "do not retry" in ei.value.message()
    assert rt.calls.count(method) == 1


async def test_float_in_runtime_money_field_is_withheld() -> None:
    rt = FakeToolRuntime()
    rt.corrupt_portfolio = True
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, "get_portfolio", {}, actor="t")
    assert ei.value.status == 502 and "withheld" in ei.value.message()


class _OddRuntime(FakeToolRuntime):
    def model_report(self) -> dict[str, Any]:
        self._rec("model_report")
        return {"registry_version": None, "targets": "not-a-list", "walk_forward": [1, 2], "notes": None}

    def predictions(self, symbol: str | None = None) -> dict[str, Any]:
        self._rec("predictions")
        return {"predictions": [{"prediction_id": "bad", "symbol": "SPY"}, 7, None], "is_simulated": True,
                "detail": "no trained model registry yet (make models)"}

    def market_quality(self) -> dict[str, Any]:
        self._rec("market_quality")
        return {"feed_mode": "SIMULATED", "status": None, "symbols": None, "events": None}


async def test_malformed_runtime_shapes_degrade_to_empty_validated_outputs() -> None:
    rt = _OddRuntime()
    rep = await dispatch(rt, "get_model_report", {}, actor="t")
    assert rep["targets"] == [] and rep["walk_forward"] is None and rep["notes"] == []
    preds = await dispatch(rt, "get_predictions", {}, actor="t")
    assert preds["predictions"] == [] and "make models" in preds["detail"]
    q = await dispatch(rt, "get_feed_quality", {}, actor="t")
    assert q["status"] == "UNKNOWN" and q["symbols"] == []  # unknown feed state is reported, never "OK"


async def test_risk_status_defaults_to_locked_when_unknown() -> None:
    class _NoLock(FakeToolRuntime):
        def status(self) -> dict[str, Any]:
            base = super().status()
            base.pop("entries_locked")
            return base

    out = await dispatch(_NoLock(), "get_risk_status", {}, actor="t")
    assert out["entries_locked"] is True  # fail closed


async def test_issued_evidence_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = FakeToolRuntime()
    await dispatch(rt, "get_predictions", {}, actor="t")
    real = time.monotonic
    monkeypatch.setattr(session_mod.time, "monotonic", lambda: real() + session_mod.ISSUED_ID_TTL_S + 1)
    with pytest.raises(ToolError) as ei:
        await dispatch(rt, "propose_paper_trade", {**GOOD_PROPOSAL, "evidence_ids": ["pred_SPY_1"]}, actor="t")
    assert ei.value.status == 422
    assert "submit_agent_proposal" not in rt.calls


def test_issued_ids_are_bounded_and_sessions_bounded() -> None:
    ids = IssuedIds(max_ids=3)
    for i in range(5):
        ids.add(f"id{i}", "signal", "SPY")
    assert len(ids) == 3 and ids.get("id0") is None and ids.get("id4") is not None
    ids.add(None, "signal")
    assert len(ids) == 3
    st = state_for(object.__new__(FakeToolRuntime))
    for i in range(session_mod.MAX_SESSIONS + 5):
        st.ids_for(f"actor{i}")
    assert len(st.sessions) == session_mod.MAX_SESSIONS


def test_window_limiter() -> None:
    lim = WindowLimiter(window_s=3600)
    assert [lim.try_acquire(2) for _ in range(3)] == [True, True, False]
    assert WindowLimiter().try_acquire(0) is False  # a zero limit means "never"


async def test_state_isolated_per_runtime() -> None:
    a, b = FakeToolRuntime(), FakeToolRuntime()
    await dispatch(a, "get_predictions", {}, actor="t")
    with pytest.raises(ToolError):
        await dispatch(b, "propose_paper_trade", {**GOOD_PROPOSAL, "evidence_ids": ["pred_SPY_1"]}, actor="t")
