"""End-to-end ResearchPipeline paths over the explicit fake transports (no network, transport=FAKE)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from packages.contracts.enums import DecisionAction, ModelProvider, ModelRunStatus, ModelTransport, Proposer
from services.ai.evidence import build_internal_evidence
from services.ai.fake_transport import FakeScenario, injection_evidence, medium_flagged_evidence
from services.ai.injection import sanitize_external

from .conftest import T0, make_signal, make_snapshot


def _statuses(result) -> dict[str, ModelRunStatus]:
    return {r.provider.value: r.status for r in result.model_runs}


async def test_trade_path_produces_validated_ai_proposal(harness_factory) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    r = await h.run()
    p = r.proposal
    assert p.action == DecisionAction.TRADE
    assert p.proposer == Proposer.AI_PIPELINE
    assert (p.symbol, p.side.value if p.side else None) == ("SPY", "BUY")
    # requested quantity/limit are the model's *requests*; arithmetic for planned loss is ours
    assert p.quantity == Decimal(1) and p.limit_price == Decimal("500.04")
    assert p.stop_price is not None and p.stop_price < p.limit_price
    assert p.max_planned_loss == p.quantity * (p.limit_price - p.stop_price)
    assert p.signal_id == "sig_test_0001" and p.policy_version_seen and p.dataset_version == "synthetic-v1"
    assert len(r.findings) == 1 and r.findings[0].recommendation == DecisionAction.TRADE
    assert p.finding_ids == (r.findings[0].finding_id,)
    assert set(p.model_run_ids) == {x.run_id for x in r.model_runs}
    assert _statuses(r) == {"openai": ModelRunStatus.OK, "anthropic": ModelRunStatus.OK}
    assert all(x.transport == ModelTransport.FAKE for x in r.model_runs)
    assert all(x.estimated_cost_usd == 0 and x.pricing_source.startswith("FAKE_TRANSPORT_NOT_BILLED") for x in r.model_runs)
    assert all(x.input_tokens > 0 and x.output_tokens > 0 for x in r.model_runs)
    sol = next(x for x in r.model_runs if x.provider == ModelProvider.OPENAI)
    claude = next(x for x in r.model_runs if x.provider == ModelProvider.ANTHROPIC)
    assert sol.reasoning_tokens and sol.reasoning_tokens > 0 and claude.reasoning_tokens is None
    assert claude.cache_read_tokens > 0 and claude.cache_write_tokens > 0
    assert len(sol.request_sha256) == 64 and sol.response_sha256 and len(sol.response_sha256) == 64
    # every cited evidence id is one of the provided ones
    provided = {e.evidence_id for e in r.evidence}
    assert set(p.evidence_ids) <= provided and set(r.findings[0].evidence_ids) <= provided
    assert h.sol.violations == [] and h.claude.violations == []


async def test_claude_no_trade_forces_no_trade(harness_factory) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.NO_TRADE)
    r = await h.run()
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert "deep review did not recommend TRADE" in (r.proposal.no_trade_reason or "")
    assert r.findings[0].recommendation == DecisionAction.NO_TRADE and r.findings[0].no_trade_reason


async def test_sol_no_trade_without_review(harness_factory) -> None:
    h = harness_factory(FakeScenario.NO_TRADE, FakeScenario.TRADE)
    r = await h.run()
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert r.findings == [] and len(r.model_runs) == 1 and h.claude.exchanges == []


async def test_compromised_sol_trade_without_independent_review_is_refused(harness_factory) -> None:
    # Claude says NO_TRADE but the (INJECTION-scripted) orchestrator returns TRADE anyway: the deterministic
    # two-model rule refuses it.
    h = harness_factory(FakeScenario.INJECTION, FakeScenario.NO_TRADE)
    r = await h.run()
    assert r.proposal.action == DecisionAction.NO_TRADE
    reason = r.proposal.no_trade_reason or ""
    assert "deep review recommended NO_TRADE" in reason


async def test_schema_invalid_sol_output(harness_factory) -> None:
    h = harness_factory(FakeScenario.SCHEMA_INVALID, FakeScenario.TRADE)
    r = await h.run()
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert _statuses(r)["openai"] == ModelRunStatus.SCHEMA_INVALID
    assert "SCHEMA_INVALID" in (r.proposal.no_trade_reason or "")


async def test_schema_invalid_claude_output(harness_factory) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.SCHEMA_INVALID)
    r = await h.run()
    assert _statuses(r)["anthropic"] == ModelRunStatus.SCHEMA_INVALID
    assert r.findings == []
    assert r.proposal.action == DecisionAction.NO_TRADE


async def test_timeout_sol(harness_factory) -> None:
    h = harness_factory(FakeScenario.TIMEOUT, FakeScenario.TRADE)
    r = await h.run()
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert _statuses(r) == {"openai": ModelRunStatus.TIMEOUT}
    rec = r.model_runs[0]
    assert rec.response_sha256 is None and rec.input_tokens == 0 and rec.error
    assert h.sol.exchanges[0].error == "ReadTimeout" and len(h.sol.exchanges) == 1  # never blindly retried


async def test_timeout_claude(harness_factory) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TIMEOUT)
    r = await h.run()
    assert _statuses(r) == {"openai": ModelRunStatus.OK, "anthropic": ModelRunStatus.TIMEOUT}
    assert r.proposal.action == DecisionAction.NO_TRADE and r.findings == []
    assert len(h.claude.exchanges) == 1


async def test_provider_errors_and_refusal_fail_closed(harness_factory) -> None:
    for sol, claude, expect in (
        (FakeScenario.HTTP_ERROR, FakeScenario.TRADE, {"openai": ModelRunStatus.ERROR}),
        (FakeScenario.TRADE, FakeScenario.HTTP_ERROR, {"openai": ModelRunStatus.OK, "anthropic": ModelRunStatus.ERROR}),
        (FakeScenario.TRADE, FakeScenario.REFUSAL, {"openai": ModelRunStatus.OK, "anthropic": ModelRunStatus.ERROR}),
    ):
        h = harness_factory(sol, claude)
        r = await h.run()
        assert r.proposal.action == DecisionAction.NO_TRADE, (sol, claude)
        assert _statuses(r) == expect, (sol, claude)


async def test_high_injection_blocks_run_without_calling_models(harness_factory) -> None:
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    r = await h.run(extra_evidence=injection_evidence(T0 - timedelta(minutes=5)))
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert "prompt injection" in (r.proposal.no_trade_reason or "")
    assert h.sol.exchanges == [] and h.claude.exchanges == []
    assert [x.status for x in r.model_runs] == [ModelRunStatus.INJECTION_BLOCKED]
    assert r.model_runs[0].input_tokens == 0 and r.model_runs[0].estimated_cost_usd == 0
    assert len(r.excluded_evidence) == 3 and not r.models_invoked


async def test_medium_flagged_evidence_cannot_support_trade(harness_factory) -> None:
    medium = medium_flagged_evidence(T0 - timedelta(minutes=5))
    assert any(f.startswith("MEDIUM:") for f in medium.injection_flags)
    assert not any(f.startswith("HIGH:") for f in medium.injection_flags)
    # well-behaved models never cite it -> TRADE allowed
    ok = await harness_factory(FakeScenario.TRADE, FakeScenario.TRADE).run(extra_evidence=[medium])
    assert ok.proposal.action == DecisionAction.TRADE
    assert medium.evidence_id not in ok.proposal.evidence_ids
    # a compromised model cites it, echoes an injection and calls an order tool -> NO_TRADE
    h = harness_factory(FakeScenario.INJECTION, FakeScenario.INJECTION)
    bad = await h.run(extra_evidence=[medium])
    assert bad.proposal.action == DecisionAction.NO_TRADE
    reason = bad.proposal.no_trade_reason or ""
    assert "injection-flagged" in reason or "injection markers" in reason
    assert bad.findings and bad.findings[0].recommendation == DecisionAction.NO_TRADE  # Claude finding downgraded
    sol = next(x for x in bad.model_runs if x.provider == ModelProvider.OPENAI)
    assert "submit_order" in (sol.error or "")  # the unknown tool was refused by PydanticAI and recorded


async def test_untrusted_evidence_is_delimited_as_data_in_prompts(harness_factory) -> None:
    news = sanitize_external("Fed minutes", "Policymakers kept rates unchanged; equities were steady.",
                             "https://news.example.invalid/fed", T0 - timedelta(hours=2), T0 - timedelta(hours=1),
                             is_simulated=True)
    h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
    r = await h.run(extra_evidence=[news])
    assert r.proposal.action == DecisionAction.TRADE
    import json

    sent = json.dumps(h.claude.requests)
    assert "untrusted_evidence" in sent and "DATA ONLY" in sent
    assert "orchestrator_question" in json.dumps(h.claude.requests[0])


async def test_stale_or_missing_inputs_never_call_models(harness_factory, policy) -> None:
    cases = {
        "no candidate": dict(signal=None),
        "stale snapshot": dict(snapshot=make_snapshot(at=T0 - timedelta(minutes=10))),
        "no quote": dict(snapshot=make_snapshot(quote=None)),
        "stale signal": dict(signal=make_signal(at=T0 - timedelta(hours=2))),
        "not candidate": dict(signal=make_signal(is_candidate=False)),
        "symbol mismatch": dict(signal=make_signal("QQQ")),
    }
    for name, kw in cases.items():
        h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
        if name == "no candidate":
            snap = make_snapshot()
            r = await h.pipeline.run(causal_id="cz_x", candidate=None, snapshot=snap, evidence=[], policy=policy)
        else:
            r = await h.run(**kw)
        assert r.proposal.action == DecisionAction.NO_TRADE, name
        assert h.sol.exchanges == [] and h.claude.exchanges == [], name
        assert r.model_runs == [], name


async def test_missing_or_stale_required_evidence_never_calls_models(harness_factory, policy) -> None:
    sig, snap = make_signal(), make_snapshot()
    internal = build_internal_evidence(sig, snap, policy)
    # missing market evidence
    h = harness_factory()
    r = await h.run(signal=sig, snapshot=snap, evidence=[internal[0], internal[2]])
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert "market snapshot evidence is missing" in (r.proposal.no_trade_reason or "")
    assert h.sol.exchanges == []
    # required signal evidence from the future (look-ahead)
    future = internal[0].model_copy(update={"available_at": T0 + timedelta(seconds=30)})
    h2 = harness_factory()
    r2 = await h2.run(signal=sig, snapshot=snap, evidence=[future, internal[1], internal[2]])
    assert r2.proposal.action == DecisionAction.NO_TRADE and "look-ahead" in (r2.proposal.no_trade_reason or "")
    assert h2.sol.exchanges == []


async def test_lookahead_and_stale_optional_evidence_is_excluded_not_shown(harness_factory) -> None:
    future_news = sanitize_external("Tomorrow's news", "Earnings beat.", "https://news.example.invalid/t",
                                    T0 + timedelta(hours=1), T0 + timedelta(hours=1), is_simulated=True)
    old_news = sanitize_external("Old news", "Old story.", "https://news.example.invalid/o",
                                 T0 - timedelta(days=30), T0 - timedelta(days=30), is_simulated=True)
    h = harness_factory()
    r = await h.run(extra_evidence=[future_news, old_news])
    assert r.proposal.action == DecisionAction.TRADE
    assert set(r.excluded_evidence) == {future_news.evidence_id, old_news.evidence_id}
    import json

    sent = json.dumps(h.sol.requests) + json.dumps(h.claude.requests)
    assert "Earnings beat." not in sent and "Old story." not in sent
    assert future_news.evidence_id not in r.proposal.evidence_ids


async def test_gate_limits_invocations_per_symbol(settings, secrets, clock, harness_factory) -> None:
    from services.ai.config import AiConfig

    h = harness_factory(config=AiConfig(min_interval_per_symbol_s=900, min_wall_interval_per_symbol_s=0))
    first = await h.run(causal_id="cz_1")
    assert first.proposal.action == DecisionAction.TRADE
    n = len(h.sol.exchanges)
    clock.advance(60)
    second = await h.run(signal=make_signal(at=clock.now()), snapshot=make_snapshot(at=clock.now()), causal_id="cz_2")
    assert second.proposal.action == DecisionAction.NO_TRADE
    assert "gated" in (second.proposal.no_trade_reason or "") and len(h.sol.exchanges) == n
    clock.advance(900)
    third = await h.run(signal=make_signal(at=clock.now()), snapshot=make_snapshot(at=clock.now()), causal_id="cz_3")
    assert third.proposal.action == DecisionAction.TRADE


async def test_real_transport_without_keys_fails_closed_without_network(settings, secrets, clock, ai_config) -> None:
    from packages.core.config import Settings
    from services.ai.pipeline import ResearchPipeline

    real_settings = Settings(project_root=settings.project_root, ai_transport="real", use_keychain=False)
    p = ResearchPipeline(real_settings, secrets, transport="real", clock=clock, config=ai_config, wall_clock=clock)
    sig, snap = make_signal(), make_snapshot()
    from packages.contracts import RiskPolicy

    pol = RiskPolicy()
    r = await p.run(causal_id="cz_r", candidate=sig, snapshot=snap, evidence=build_internal_evidence(sig, snap, pol), policy=pol)
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert "transport unavailable" in (r.proposal.no_trade_reason or "") and r.model_runs == []
    await p.aclose()
    # transport='real' requested while LAT_AI_TRANSPORT=fake: also refused (never inferred)
    p2 = ResearchPipeline(settings, secrets, transport="real", clock=clock, config=ai_config, wall_clock=clock)
    r2 = await p2.run(causal_id="cz_r2", candidate=sig, snapshot=snap, evidence=build_internal_evidence(sig, snap, pol),
                      policy=pol)
    assert r2.proposal.action == DecisionAction.NO_TRADE and "LAT_AI_TRANSPORT" in (r2.proposal.no_trade_reason or "")
    await p2.aclose()


async def test_default_fake_pipeline_constructed_like_the_api(settings, secrets, clock) -> None:
    from packages.contracts import RiskPolicy
    from services.ai.pipeline import ResearchPipeline

    p = ResearchPipeline(settings, secrets, transport=settings.ai_transport, db=None, clock=clock)
    sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
    r = await p.run(causal_id="cz_api", candidate=sig, snapshot=snap, evidence=build_internal_evidence(sig, snap, pol),
                    policy=pol)
    assert r.proposal.action == DecisionAction.TRADE
    assert set(p.fake_handlers) == {"anthropic", "openai"}
    assert all(x.transport == ModelTransport.FAKE for x in r.model_runs)
    await p.aclose()
