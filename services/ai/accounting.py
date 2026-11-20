"""Model-run accounting: token usage, latency, Decimal cost estimates, request/response hashes, persistence.

Token semantics recorded on ``ModelRunRecord`` (both providers, as normalized by PydanticAI):
* ``input_tokens``  = total prompt tokens, *including* cache reads/writes (Anthropic's wire ``input_tokens`` is
  only the uncached remainder; PydanticAI adds the cache fields back, which is what we store).
* ``cache_read_tokens`` / ``cache_write_tokens`` = the cached subsets.
* ``output_tokens`` includes thinking/reasoning tokens (billed as output by both providers).
* ``reasoning_tokens`` = OpenAI ``output_tokens_details.reasoning_tokens``; ``None`` for Claude (not reported).

Costs are computed per HTTP request (GPT-6.1 Sol has a per-request long-context tier) and rounded *up* to the
micro-dollar so budgets never under-count. FAKE-transport runs bill nothing: ``estimated_cost_usd`` is 0 and the
list-price equivalent is stated in ``pricing_source``.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from typing import Any

import httpx2

from packages.contracts import Evidence, ModelRunRecord, ResearchFinding
from packages.contracts.enums import ModelProvider, ModelTransport
from packages.core.canonical import canonical_json
from packages.core.db import Database, utc_iso

MTOK = Decimal(1_000_000)
MICRO = Decimal("0.000001")


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    cache_write_per_mtok: Decimal
    source: str
    long_context_threshold: int | None = None  # prompt tokens above which the whole request is repriced
    long_input_multiplier: Decimal = Decimal(1)  # applies to input, cache read and cache write
    long_output_multiplier: Decimal = Decimal(1)

    @property
    def verified(self) -> bool:
        return not self.source.startswith("UNVERIFIED")


PRICING: dict[str, ModelPricing] = {
    "claude-opus-5-5": ModelPricing(
        model_id="claude-opus-5-5",
        input_per_mtok=Decimal("4.00"),
        output_per_mtok=Decimal("20.00"),
        cache_read_per_mtok=Decimal("0.20"),
        cache_write_per_mtok=Decimal("5.00"),  # 5-minute TTL write = 1.25x input (1-hour TTL, unused here: 2x)
        source=(
            "claude-api skill (Anthropic model table cached 2026-10-06 + prompt-caching guide): claude-opus-5-5 "
            "$4/MTok input, $20/MTok output, $0.20/MTok cache read, 1.25x ($5/MTok) 5-minute cache write; flat across 1M"
        ),
    ),
    "gpt-6.1-sol": ModelPricing(
        model_id="gpt-6.1-sol",
        input_per_mtok=Decimal("2.00"),
        output_per_mtok=Decimal("10.00"),
        cache_read_per_mtok=Decimal("0.10"),
        cache_write_per_mtok=Decimal("2.50"),
        long_context_threshold=272_000,
        long_input_multiplier=Decimal(2),
        long_output_multiplier=Decimal("1.5"),
        source=(
            "https://developers.openai.com/api/docs/models/gpt-6.1-sol (read 2026-11-06): $2/MTok input, $0.10 cached "
            "input, $2.50 cache writes, $10 output; prompts >272K input tokens: 2x input+cache, 1.5x output"
        ),
    ),
}


def pricing_for(model_id: str) -> ModelPricing | None:
    return PRICING.get(model_id)


@dataclass(frozen=True)
class RequestTokens:
    input_tokens: int  # total prompt tokens incl. cache read + write
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int | None = None

    @property
    def uncached_input(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens - self.cache_write_tokens)


@dataclass(frozen=True)
class TokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int | None = None
    requests: int = 0


def total_tokens(requests: Sequence[RequestTokens]) -> TokenTotals:
    reasoning = [r.reasoning_tokens for r in requests if r.reasoning_tokens is not None]
    return TokenTotals(
        input_tokens=sum(r.input_tokens for r in requests),
        output_tokens=sum(r.output_tokens for r in requests),
        cache_read_tokens=sum(r.cache_read_tokens for r in requests),
        cache_write_tokens=sum(r.cache_write_tokens for r in requests),
        reasoning_tokens=sum(reasoning) if reasoning else None,
        requests=len(requests),
    )


def request_cost(pricing: ModelPricing, r: RequestTokens) -> Decimal:
    in_mult = out_mult = Decimal(1)
    if pricing.long_context_threshold is not None and r.input_tokens > pricing.long_context_threshold:
        in_mult, out_mult = pricing.long_input_multiplier, pricing.long_output_multiplier
    cost = (
        Decimal(r.uncached_input) * pricing.input_per_mtok * in_mult
        + Decimal(r.cache_read_tokens) * pricing.cache_read_per_mtok * in_mult
        + Decimal(r.cache_write_tokens) * pricing.cache_write_per_mtok * in_mult
        + Decimal(r.output_tokens) * pricing.output_per_mtok * out_mult
    )
    return cost / MTOK


def estimate_cost(model_id: str, requests: Sequence[RequestTokens], transport: ModelTransport) -> tuple[Decimal, str]:
    """``(estimated_cost_usd, pricing_source)``; FAKE transport bills nothing."""
    pricing = pricing_for(model_id)
    if pricing is None:
        return Decimal(0), f"UNVERIFIED: no pricing table entry for {model_id!r}"
    cost = sum((request_cost(pricing, r) for r in requests), Decimal(0)).quantize(MICRO, rounding=ROUND_CEILING)
    if transport == ModelTransport.FAKE:
        return Decimal(0), f"FAKE_TRANSPORT_NOT_BILLED (list-price equivalent ${cost}; {pricing.source})"
    return cost, pricing.source


# ------------------------------------------------------------------------------------------------
# usage extraction
# ------------------------------------------------------------------------------------------------


def usage_from_messages(messages: Iterable[Any]) -> list[RequestTokens]:
    """Per-request usage from PydanticAI ``ModelResponse.usage`` (RequestUsage) objects."""
    from pydantic_ai.messages import ModelResponse

    out: list[RequestTokens] = []
    for m in messages:
        if not isinstance(m, ModelResponse):
            continue
        u = m.usage
        details = dict(u.details or {})
        reasoning = details.get("reasoning_tokens")
        out.append(RequestTokens(
            input_tokens=int(u.input_tokens or 0),
            output_tokens=int(u.output_tokens or 0),
            cache_read_tokens=int(u.cache_read_tokens or 0),
            cache_write_tokens=int(u.cache_write_tokens or 0),
            reasoning_tokens=int(reasoning) if reasoning is not None else None,
        ))
    return out


def usage_from_wire(provider: ModelProvider, body: dict[str, Any]) -> RequestTokens | None:
    """Fallback for failed runs: parse the provider's raw ``usage`` object from a captured response body."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    u: dict[str, Any] = usage  # pyright: ignore[reportUnknownVariableType]
    if provider == ModelProvider.ANTHROPIC:
        cr = int(u.get("cache_read_input_tokens") or 0)
        cw = int(u.get("cache_creation_input_tokens") or 0)
        return RequestTokens(int(u.get("input_tokens") or 0) + cr + cw, int(u.get("output_tokens") or 0), cr, cw, None)
    in_details: Any = u.get("input_tokens_details") or {}
    out_details: Any = u.get("output_tokens_details") or {}
    return RequestTokens(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_read_tokens=int(in_details.get("cached_tokens") or 0),
        cache_write_tokens=int(in_details.get("cache_write_tokens") or 0),
        reasoning_tokens=int(out_details.get("reasoning_tokens") or 0),
    )


# ------------------------------------------------------------------------------------------------
# HTTP capture (request/response hashes + wire usage), scoped per model run with a context variable
# ------------------------------------------------------------------------------------------------


@dataclass
class HttpCapture:
    provider: ModelProvider
    request_hashes: list[str] = field(default_factory=list[str])
    response_hashes: list[str] = field(default_factory=list[str])
    wire_usage: list[RequestTokens] = field(default_factory=list[RequestTokens])
    statuses: list[int] = field(default_factory=list[int])

    @staticmethod
    def _combine(hashes: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(hashes).encode()).hexdigest()

    @property
    def request_sha256(self) -> str:
        return self._combine(self.request_hashes)

    @property
    def response_sha256(self) -> str | None:
        return self._combine(self.response_hashes) if self.response_hashes else None


_CURRENT: contextvars.ContextVar[HttpCapture | None] = contextvars.ContextVar("lat_ai_http_capture", default=None)


def current_capture() -> HttpCapture | None:
    return _CURRENT.get()


def start_capture(provider: ModelProvider) -> tuple[HttpCapture, contextvars.Token[HttpCapture | None]]:
    cap = HttpCapture(provider)
    return cap, _CURRENT.set(cap)


def end_capture(token: contextvars.Token[HttpCapture | None]) -> None:
    _CURRENT.reset(token)


async def _on_request(request: httpx2.Request) -> None:
    cap = _CURRENT.get()
    if cap is not None:
        cap.request_hashes.append(hashlib.sha256(request.content or b"").hexdigest())


async def _on_response(response: httpx2.Response) -> None:
    cap = _CURRENT.get()
    if cap is None:
        return
    data = await response.aread()
    cap.statuses.append(response.status_code)
    cap.response_hashes.append(hashlib.sha256(data).hexdigest())
    try:
        body = json.loads(data or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(body, dict):
        tokens = usage_from_wire(cap.provider, body)  # pyright: ignore[reportUnknownArgumentType]
        if tokens is not None:
            cap.wire_usage.append(tokens)


def install_capture_hooks(client: httpx2.AsyncClient) -> None:
    hooks = client.event_hooks
    if _on_request not in hooks["request"]:
        hooks["request"].append(_on_request)
    if _on_response not in hooks["response"]:
        hooks["response"].append(_on_response)
    client.event_hooks = hooks


# ------------------------------------------------------------------------------------------------
# persistence (tables from packages/core/migrations/0001_core.sql)
# ------------------------------------------------------------------------------------------------


def insert_run(conn: sqlite3.Connection, record: ModelRunRecord) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO model_runs(run_id, causal_id, provider, model_id, transport, status, started_at,"
        " latency_ms, input_tokens, output_tokens, estimated_cost_usd, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            record.run_id, record.causal_id, record.provider.value, record.model_id, record.transport.value,
            record.status.value, utc_iso(record.started_at), record.latency_ms, record.input_tokens,
            record.output_tokens, format(record.estimated_cost_usd, "f"), canonical_json(record),
        ),
    )


def insert_finding(conn: sqlite3.Connection, finding: ResearchFinding) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO research_findings(finding_id, causal_id, model_run_id, symbol, recommendation,"
        " created_at, payload_json) VALUES (?,?,?,?,?,?,?)",
        (
            finding.finding_id, finding.causal_id, finding.model_run_id, finding.symbol, finding.recommendation.value,
            utc_iso(finding.created_at), canonical_json(finding),
        ),
    )


def insert_evidence(conn: sqlite3.Connection, evidence: Evidence, causal_id: str | None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO evidence(evidence_id, causal_id, source_type, source_id, trust, content_sha256,"
        " available_at, injection_flags, payload_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            evidence.evidence_id, causal_id, evidence.source_type.value, evidence.source_id, evidence.trust.value,
            evidence.content_sha256, utc_iso(evidence.available_at), json.dumps(list(evidence.injection_flags)),
            canonical_json(evidence),
        ),
    )


def persist_run(db: Database, record: ModelRunRecord) -> None:
    db.write(lambda c: insert_run(c, record))


def persist_findings(db: Database, findings: Sequence[ResearchFinding]) -> None:
    def _w(c: sqlite3.Connection) -> None:
        for f in findings:
            insert_finding(c, f)

    if findings:
        db.write(_w)


def persist_evidence(db: Database, evidence: Sequence[Evidence], causal_id: str | None = None) -> None:
    def _w(c: sqlite3.Connection) -> None:
        for e in evidence:
            insert_evidence(c, e, causal_id)

    if evidence:
        db.write(_w)
