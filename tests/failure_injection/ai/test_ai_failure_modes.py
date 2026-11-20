"""Adverse provider/transport behaviour: every failure ends in NO_TRADE, is recorded, and is never blindly retried."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import httpx2
import pytest

from packages.contracts import RiskPolicy
from packages.contracts.enums import DecisionAction, ModelRunStatus
from packages.core.db import Database
from services.ai.config import AiConfig
from services.ai.evidence import build_internal_evidence
from services.ai.fake_transport import FakeOpenAIHandler, FakeScenario, fake_http_client
from services.ai.pipeline import ResearchPipeline, classify_exception
from tests.unit.ai.conftest import make_signal, make_snapshot


def _pipeline(settings: Any, secrets: Any, clock: Any, openai_handler: Any, *, config: AiConfig | None = None,
              db: Database | None = None, anthropic_client: httpx2.AsyncClient | None = None) -> ResearchPipeline:
    clients: dict[str, httpx2.AsyncClient] = {"openai": httpx2.AsyncClient(transport=httpx2.MockTransport(openai_handler))}
    if anthropic_client is not None:
        clients["anthropic"] = anthropic_client
    cfg = config or AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0)
    return ResearchPipeline(settings, secrets, transport="fake", db=db, clock=clock, config=cfg, wall_clock=clock,
                            http_clients=clients)


async def _run(p: ResearchPipeline) -> Any:
    sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
    return await p.run(causal_id="cz_fi", candidate=sig, snapshot=snap, evidence=build_internal_evidence(sig, snap, pol),
                       policy=pol)


@pytest.mark.parametrize(("name", "handler", "expected"), [
    ("non-json 200", lambda r: httpx2.Response(200, content=b"<html>gateway</html>"), {ModelRunStatus.ERROR}),
    ("truncated wire object", lambda r: httpx2.Response(200, json={"id": "x", "object": "response"}),
     {ModelRunStatus.ERROR, ModelRunStatus.SCHEMA_INVALID}),
    ("rate limited", lambda r: httpx2.Response(429, json={"error": {"message": "slow down", "type": "rate_limit"}}),
     {ModelRunStatus.ERROR}),
    ("auth error", lambda r: httpx2.Response(401, json={"error": {"message": "bad key", "type": "auth"}}),
     {ModelRunStatus.ERROR}),
])
async def test_malformed_or_rejected_provider_responses(settings, secrets, clock, name, handler, expected) -> None:
    calls: list[int] = []

    def h(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        return handler(request)

    r = await _run(_pipeline(settings, secrets, clock, h))
    assert r.proposal.action == DecisionAction.NO_TRADE, name
    assert {x.status for x in r.model_runs} <= expected, (name, [x.status for x in r.model_runs])
    assert len(calls) == 1, name  # SDK retries disabled: nothing is silently re-sent


async def test_connection_error_is_error(settings, secrets, clock) -> None:
    def h(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("network partition", request=request)

    r = await _run(_pipeline(settings, secrets, clock, h))
    assert r.proposal.action == DecisionAction.NO_TRADE and r.model_runs[0].status == ModelRunStatus.ERROR


async def test_slow_provider_hits_the_run_deadline(settings, secrets, clock) -> None:
    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(5)
        return httpx2.Response(500)

    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0, run_timeout_s=0.2)
    r = await _run(_pipeline(settings, secrets, clock, slow, config=cfg))
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert r.model_runs[0].status == ModelRunStatus.TIMEOUT and r.model_runs[0].latency_ms < 3000


async def test_runaway_tool_loop_is_cut_by_request_limit(settings, secrets, clock) -> None:
    n = {"i": 0}

    def looping(request: httpx2.Request) -> httpx2.Response:
        n["i"] += 1
        call = {"type": "function_call", "id": f"fc_{n['i']}", "call_id": f"c_{n['i']}", "name": "list_evidence",
                "arguments": "{}", "status": "completed"}
        return httpx2.Response(200, json={
            "id": f"resp_{n['i']}", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-6.1-sol",
            "output": [call], "parallel_tool_calls": True, "tool_choice": "required", "tools": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 1}}})

    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0, sol_request_limit=3)
    r = await _run(_pipeline(settings, secrets, clock, looping, config=cfg))
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert r.model_runs[0].status == ModelRunStatus.ERROR and "UsageLimitExceeded" in (r.model_runs[0].error or "")
    assert n["i"] == 3
    assert r.model_runs[0].input_tokens == 30  # tokens of the failed run still accounted (from the wire)


async def test_internal_bug_fails_closed_and_charges_reservation(settings, secrets, clock, db, monkeypatch) -> None:
    import services.ai.pipeline as pl

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("conversion bug")

    monkeypatch.setattr(pl, "proposal_from_draft", boom)
    p = _pipeline(settings, secrets, clock, FakeOpenAIHandler(FakeScenario.TRADE), db=db)
    r = await _run(p)
    assert r.proposal.action == DecisionAction.NO_TRADE and "internal error" in (r.proposal.no_trade_reason or "")
    assert db.query("SELECT kind FROM failures WHERE component='ai'")[0]["kind"] == "RuntimeError"
    row = db.query("SELECT status, actual_usd FROM ai_invocations")[0]
    assert row["status"] == "SETTLED" and Decimal(row["actual_usd"]) == 0  # FAKE reserves nothing
    assert p.gate.status()["in_flight"] == []  # reservation released even on failure


async def test_claude_timeout_inside_review_still_settles_and_records(settings, secrets, clock, db) -> None:
    from services.ai.fake_transport import FakeAnthropicHandler

    claude = FakeAnthropicHandler(FakeScenario.TIMEOUT)
    p = _pipeline(settings, secrets, clock, FakeOpenAIHandler(FakeScenario.TRADE), db=db,
                  anthropic_client=fake_http_client(claude))
    r = await _run(p)
    assert r.proposal.action == DecisionAction.NO_TRADE
    statuses = {row["model_id"]: row["status"] for row in db.query("SELECT model_id, status FROM model_runs")}
    assert statuses == {"gpt-6.1-sol": "OK", "claude-opus-5-5": "TIMEOUT"}
    assert len(claude.exchanges) == 1


def test_classify_exception_walks_cause_chain() -> None:
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    inner = httpx2.ReadTimeout("t")
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner
    assert classify_exception(outer) == ModelRunStatus.TIMEOUT
    assert classify_exception(UnexpectedModelBehavior("Exceeded maximum output retries")) == ModelRunStatus.SCHEMA_INVALID
    assert classify_exception(ValueError("x")) == ModelRunStatus.ERROR
