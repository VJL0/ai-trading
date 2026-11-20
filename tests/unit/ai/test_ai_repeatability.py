"""Same fake inputs => identical proposal content (modulo generated ids/timestamps), five times."""

from __future__ import annotations

import json
import re
from typing import Any

from packages.contracts import RiskPolicy
from services.ai.evidence import build_internal_evidence
from services.ai.fake_transport import FakeScenario

from .conftest import make_signal, make_snapshot

_ULID = re.compile(r"\b(?:mrun|find)_[0-9A-HJKMNP-TV-Z]{26}\b")
_ID_FIELDS = {"proposal_id", "created_at", "model_run_ids", "finding_ids"}


def _content(result: Any) -> dict[str, Any]:
    d = result.proposal.model_dump(mode="json")
    for k in _ID_FIELDS:
        d.pop(k)
    findings = [f.model_dump(mode="json", exclude={"finding_id", "model_run_id", "created_at"}) for f in result.findings]
    runs = [r.model_dump(mode="json", include={"provider", "model_id", "status", "effort", "transport", "input_tokens",
                                                "output_tokens", "cache_read_tokens", "reasoning_tokens", "prompt_version",
                                                "output_schema"})
            for r in result.model_runs]
    return {"proposal": d, "findings": findings, "runs": runs}


async def test_trade_repeatable_five_times(harness_factory) -> None:
    sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
    evidence = build_internal_evidence(sig, snap, pol)
    contents = []
    for i in range(5):
        h = harness_factory(FakeScenario.TRADE, FakeScenario.TRADE)
        r = await h.run(signal=sig, snapshot=snap, policy=pol, evidence=evidence, causal_id="cz_repeat")
        assert r.proposal.action.value == "TRADE", i
        c = _content(r)
        c["claude_request_sha256"] = next(x.request_sha256 for x in r.model_runs if x.provider.value == "anthropic")
        c["sol_requests_normalized"] = _ULID.sub("<id>", json.dumps(h.sol.requests, sort_keys=True))
        contents.append(c)
    assert all(c == contents[0] for c in contents[1:])
    # wire-level determinism too: Claude requests are byte-identical; Sol requests are identical once the generated
    # finding/run ids returned by request_deep_review are normalized
    assert len({c["claude_request_sha256"] for c in contents}) == 1
    assert len({c["sol_requests_normalized"] for c in contents}) == 1


async def test_no_trade_repeatable_five_times(harness_factory) -> None:
    sig, snap, pol = make_signal(), make_snapshot(), RiskPolicy()
    evidence = build_internal_evidence(sig, snap, pol)
    contents = []
    for _ in range(5):
        h = harness_factory(FakeScenario.TRADE, FakeScenario.NO_TRADE)
        r = await h.run(signal=sig, snapshot=snap, policy=pol, evidence=evidence, causal_id="cz_repeat")
        contents.append(_content(r))
    assert contents[0]["proposal"]["action"] == "NO_TRADE"
    assert all(c == contents[0] for c in contents[1:])
