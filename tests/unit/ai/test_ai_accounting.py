"""Pricing, token accounting and persistence of model runs / findings / evidence."""

from __future__ import annotations

import json
from decimal import Decimal

from packages.contracts import ModelRunRecord, RiskPolicy
from packages.contracts.enums import ModelProvider, ModelRunStatus, ModelTransport
from packages.core.db import Database
from services.ai.accounting import (
    PRICING,
    RequestTokens,
    estimate_cost,
    persist_evidence,
    persist_findings,
    persist_run,
    request_cost,
    usage_from_wire,
)
from services.ai.evidence import build_internal_evidence
from services.ai.fake_transport import FakeScenario

from .conftest import T0, make_signal, make_snapshot


def test_claude_pricing_from_claude_api_skill() -> None:
    p = PRICING["claude-opus-5-5"]
    assert (p.input_per_mtok, p.output_per_mtok, p.cache_read_per_mtok, p.cache_write_per_mtok) == (
        Decimal("4.00"), Decimal("20.00"), Decimal("0.20"), Decimal("5.00"))
    assert "claude-api skill" in p.source and p.verified
    # 1M uncached input + 1M output = $24; cache tokens priced separately (input_tokens is the total incl. cache)
    r = RequestTokens(input_tokens=1_300_000, output_tokens=1_000_000, cache_read_tokens=200_000, cache_write_tokens=100_000)
    assert request_cost(p, r) == Decimal("4") + Decimal("20") + Decimal("0.04") + Decimal("0.5")


def test_sol_pricing_and_long_context_tier() -> None:
    p = PRICING["gpt-6.1-sol"]
    assert "developers.openai.com/api/docs/models/gpt-6.1-sol" in p.source and p.verified
    short = RequestTokens(input_tokens=100_000, output_tokens=10_000, cache_read_tokens=50_000)
    assert request_cost(p, short) == (Decimal(50_000) * 2 + Decimal(50_000) * Decimal("0.10") + Decimal(10_000) * 10) / 1_000_000
    long = RequestTokens(input_tokens=300_000, output_tokens=10_000)
    assert request_cost(p, long) == (Decimal(300_000) * 4 + Decimal(10_000) * 15) / 1_000_000  # whole request repriced


def test_estimate_cost_transport_labels() -> None:
    reqs = [RequestTokens(1000, 100)]
    real, src = estimate_cost("gpt-6.1-sol", reqs, ModelTransport.REAL)
    assert real == Decimal("0.003") and src == PRICING["gpt-6.1-sol"].source
    fake, fsrc = estimate_cost("gpt-6.1-sol", reqs, ModelTransport.FAKE)
    assert fake == 0 and fsrc.startswith("FAKE_TRANSPORT_NOT_BILLED") and "$0.003000" in fsrc
    unknown, usrc = estimate_cost("some-other-model", reqs, ModelTransport.REAL)
    assert unknown == 0 and usrc.startswith("UNVERIFIED")
    tiny, _ = estimate_cost("claude-opus-5-5", [RequestTokens(1, 0)], ModelTransport.REAL)
    assert tiny == Decimal("0.000004")  # rounded up to the micro-dollar, never down
    assert isinstance(tiny, Decimal)


def test_wire_usage_parsing_for_both_providers() -> None:
    a = usage_from_wire(ModelProvider.ANTHROPIC, {"usage": {"input_tokens": 100, "output_tokens": 50,
                                                            "cache_creation_input_tokens": 10,
                                                            "cache_read_input_tokens": 20}})
    assert a == RequestTokens(130, 50, 20, 10, None)
    o = usage_from_wire(ModelProvider.OPENAI, {"usage": {"input_tokens": 200, "output_tokens": 80,
                                                         "input_tokens_details": {"cached_tokens": 30},
                                                         "output_tokens_details": {"reasoning_tokens": 40}}})
    assert o == RequestTokens(200, 80, 30, 0, 40)
    assert usage_from_wire(ModelProvider.OPENAI, {"error": {}}) is None


async def test_pipeline_tokens_match_the_wire(harness_factory) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    r = await h.run()
    for rec in r.model_runs:
        handler = h.sol if rec.provider == ModelProvider.OPENAI else h.claude
        wire = [usage_from_wire(rec.provider, x.response or {}) for x in handler.exchanges]
        assert rec.input_tokens == sum(w.input_tokens for w in wire if w)
        assert rec.output_tokens == sum(w.output_tokens for w in wire if w)
        assert rec.cache_read_tokens == sum(w.cache_read_tokens for w in wire if w)


def _record(status: ModelRunStatus = ModelRunStatus.OK) -> ModelRunRecord:
    return ModelRunRecord(
        run_id="mrun_test1", causal_id="cz_t", provider=ModelProvider.OPENAI, model_id="gpt-6.1-sol",
        adapter_version="a", prompt_version="p", effort="medium", transport=ModelTransport.FAKE, status=status,
        started_at=T0, finished_at=T0, latency_ms=5, input_tokens=10, output_tokens=2, estimated_cost_usd=Decimal("0.000123"),
        pricing_source="x", request_sha256="0" * 64, output_schema="TradeProposalDraft@x",
    )


def test_persistence_roundtrip_and_idempotency(db: Database, harness_factory) -> None:
    rec = _record()
    persist_run(db, rec)
    persist_run(db, rec)  # idempotent (same run id)
    rows = db.query("SELECT * FROM model_runs")
    assert len(rows) == 1 and rows[0]["estimated_cost_usd"] == "0.000123" and rows[0]["transport"] == "FAKE"
    assert ModelRunRecord.model_validate_json(rows[0]["payload_json"]) == rec
    sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
    ev = build_internal_evidence(sig, snap, pol)
    persist_evidence(db, ev, "cz_t")
    persist_evidence(db, ev, "cz_t")
    erows = db.query("SELECT evidence_id, trust, injection_flags, causal_id FROM evidence ORDER BY evidence_id")
    assert len(erows) == 3 and all(r["trust"] == "TRUSTED_INTERNAL" and json.loads(r["injection_flags"]) == [] for r in erows)


async def test_pipeline_persists_runs_findings_evidence_and_audit(db: Database, harness_factory) -> None:
    from packages.core.audit import verify_chain

    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE, db=db)
    r = await h.run(causal_id="cz_persist")
    assert {x["run_id"] for x in db.query("SELECT run_id FROM model_runs WHERE causal_id='cz_persist'")} == {
        x.run_id for x in r.model_runs}
    assert [x["finding_id"] for x in db.query("SELECT finding_id FROM research_findings")] == [r.findings[0].finding_id]
    assert len(db.query("SELECT 1 FROM evidence WHERE causal_id='cz_persist'")) == 3
    audit = db.query("SELECT * FROM audit_events WHERE event_type='AI_PIPELINE_RESULT'")
    assert len(audit) == 1 and audit[0]["causal_id"] == "cz_persist" and audit[0]["policy_version"] == RiskPolicy().version
    assert "gpt-6.1-sol" in audit[0]["model_version"] and "claude-opus-5-5" in audit[0]["model_version"]
    assert verify_chain(db)[0]
    persist_findings(db, r.findings)  # re-persisting is a no-op
    assert len(db.query("SELECT 1 FROM research_findings")) == 1


async def test_failed_runs_are_persisted_with_failure_rows(db: Database, harness_factory) -> None:
    h = harness_factory(FakeScenario.TIMEOUT, FakeScenario.TRADE, db=db)
    r = await h.run(causal_id="cz_fail")
    assert r.model_runs[0].status == ModelRunStatus.TIMEOUT
    assert db.query("SELECT status FROM model_runs")[0]["status"] == "TIMEOUT"
    kinds = [x["kind"] for x in db.query("SELECT kind FROM failures WHERE causal_id='cz_fail'")]
    assert kinds == ["MODEL_TIMEOUT"]
