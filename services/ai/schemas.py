"""LLM-facing draft contracts and their deterministic conversion into decision contracts.

Models only ever produce a *draft* containing fields they are allowed to choose (symbol, side, requested
quantity/limit/stop, citations, thesis, counterarguments, risks, an uncalibrated confidence label, TRADE or
NO_TRADE and a reason). Everything else -- ids, causal id, timestamps, proposer, policy version, dataset
version, model run ids, finding ids and the planned-loss arithmetic -- is assigned here, deterministically.
There is no draft field for limits, policy or a calibrated probability, and ``extra="forbid"`` rejects any
attempt to add one.

Drafts deliberately avoid JSON-schema constraints (patterns, lengths) so both providers' strict structured
output modes accept the schema unchanged; the equivalent checks run as validators after parsing. A validator
failure is a schema-invalid output (the run fails closed as ``SCHEMA_INVALID``); a semantic failure (unknown
evidence, stale evidence, symbol mismatch, ...) converts the draft into ``NO_TRADE`` with the reason.

The proposal's quantity and limit price are REQUESTS: the independent risk engine recomputes all quantities,
notionals and losses from broker and market state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.contracts import MarketSnapshot, QuantSignal, ResearchFinding, RiskPolicy, TradeProposal
from packages.contracts.enums import ConfidenceLabel, DecisionAction, OrderType, Proposer, Side, Stance
from packages.core.ids import new_id
from packages.core.money import BPS, round_price

from .evidence import EvidenceIndex
from .injection import Severity, scan_text

DRAFT_SCHEMA_VERSION = "ai-drafts/1.0.0"

MAX_THESIS_CHARS = 4000
MAX_ITEM_CHARS = 1000
MAX_LIST_ITEMS = 20
MAX_CITATIONS = 50
MAX_WHOLE_SHARES = 1_000_000

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_QTY_RE = re.compile(r"^[0-9]{1,7}$")
_PRICE_RE = re.compile(r"^[0-9]{1,7}(?:\.[0-9]{1,4})?$")

TradeAction = Literal["TRADE", "NO_TRADE"]
ConfidenceText = Literal["LOW", "MEDIUM", "HIGH"]


class _Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Citation(_Draft):
    """A reference to one evidence item, copied exactly from list_evidence/get_evidence."""

    evidence_id: str = Field(description="evidence_id exactly as listed")
    source_id: str = Field(description="source_id exactly as listed")
    available_at: str = Field(description="available_at exactly as listed (ISO-8601 UTC, e.g. 2026-11-05T15:30:00Z)")

    @field_validator("evidence_id", "source_id", "available_at")
    @classmethod
    def _short(cls, v: str) -> str:
        if not v or len(v) > 300:
            raise ValueError("citation fields must be non-empty and at most 300 characters")
        return v


def _check_texts(values: list[str], name: str) -> list[str]:
    if len(values) > MAX_LIST_ITEMS:
        raise ValueError(f"{name}: at most {MAX_LIST_ITEMS} items")
    for v in values:
        if not v.strip() or len(v) > MAX_ITEM_CHARS:
            raise ValueError(f"{name}: items must be non-empty and at most {MAX_ITEM_CHARS} characters")
    return values


def _check_citations(citations: list[Citation], evidence_ids: list[str]) -> None:
    if len(citations) > MAX_CITATIONS or len(evidence_ids) > MAX_CITATIONS:
        raise ValueError(f"at most {MAX_CITATIONS} citations")
    cited = {c.evidence_id for c in citations}
    missing = [e for e in evidence_ids if e not in cited]
    if missing:
        raise ValueError(f"every evidence_id must be cited; uncited: {missing}")


class ResearchFindingDraft(_Draft):
    """Claude Opus research review of one candidate (output schema of the research agent)."""

    symbol: str = Field(description="the candidate symbol")
    recommendation: TradeAction = Field(description="TRADE or NO_TRADE; NO_TRADE is always acceptable")
    stance: Literal["SUPPORT", "OPPOSE", "NEUTRAL"] = Field(description="stance toward the trade")
    thesis: str = Field(description="concise reasoning grounded in the cited evidence")
    citations: list[Citation] = Field(description="evidence supporting the analysis (copy ids/timestamps exactly)")
    evidence_ids: list[str] = Field(description="ids of the cited evidence; every id must appear in citations")
    counterarguments: list[str] = Field(description="genuine arguments against the trade (at least one)")
    risks: list[str] = Field(description="main risks")
    confidence_label: ConfidenceText = Field(description="qualitative and UNCALIBRATED; never a probability")
    no_trade_reason: str | None = Field(description="required when recommendation is NO_TRADE, else null")

    @field_validator("thesis")
    @classmethod
    def _thesis(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_THESIS_CHARS:
            raise ValueError(f"thesis must be non-empty and at most {MAX_THESIS_CHARS} characters")
        return v

    @field_validator("counterarguments", "risks")
    @classmethod
    def _lists(cls, v: list[str]) -> list[str]:
        return _check_texts(v, "list")

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_citations(self.citations, self.evidence_ids)
        if self.recommendation == "NO_TRADE" and not (self.no_trade_reason or "").strip():
            raise ValueError("NO_TRADE requires no_trade_reason")
        if self.no_trade_reason is not None and len(self.no_trade_reason) > MAX_ITEM_CHARS:
            raise ValueError("no_trade_reason too long")
        if not self.counterarguments:
            raise ValueError("at least one counterargument is required")
        return self


class TradeProposalDraft(_Draft):
    """GPT-6.1 Sol orchestrator output. quantity / limit_price / stop_price are REQUESTS only."""

    recommendation: TradeAction = Field(description="TRADE or NO_TRADE")
    symbol: str | None = Field(description="candidate symbol for TRADE, else null")
    side: Literal["BUY", "SELL"] | None = Field(description="candidate side for TRADE, else null")
    quantity: str | None = Field(description='whole shares as a decimal string, e.g. "3"; null for NO_TRADE')
    limit_price: str | None = Field(description='limit price decimal string, max 4 decimals, e.g. "412.35"')
    stop_price: str | None = Field(description="optional protective stop decimal string (BUY: below limit)")
    citations: list[Citation] = Field(description="evidence relied on (copy ids/timestamps exactly)")
    evidence_ids: list[str] = Field(description="ids of the cited evidence; every id must appear in citations")
    thesis: str = Field(description="concise reasoning")
    counterarguments: list[str] = Field(description="arguments against the trade")
    risks: list[str] = Field(description="main risks")
    confidence_label: ConfidenceText = Field(description="qualitative and UNCALIBRATED; never a probability")
    no_trade_reason: str | None = Field(description="required when recommendation is NO_TRADE, else null")

    @field_validator("thesis")
    @classmethod
    def _thesis(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_THESIS_CHARS:
            raise ValueError(f"thesis must be non-empty and at most {MAX_THESIS_CHARS} characters")
        return v

    @field_validator("counterarguments", "risks")
    @classmethod
    def _lists(cls, v: list[str]) -> list[str]:
        return _check_texts(v, "list")

    @field_validator("quantity")
    @classmethod
    def _qty(cls, v: str | None) -> str | None:
        if v is not None and not _QTY_RE.match(v):
            raise ValueError('quantity must be a whole number of shares as a decimal string, e.g. "3"')
        return v

    @field_validator("limit_price", "stop_price")
    @classmethod
    def _price(cls, v: str | None) -> str | None:
        if v is not None and not _PRICE_RE.match(v):
            raise ValueError('prices must be decimal strings with at most 4 decimal places, e.g. "412.35"')
        return v

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_citations(self.citations, self.evidence_ids)
        if self.recommendation == "TRADE":
            missing = [n for n in ("symbol", "side", "quantity", "limit_price") if getattr(self, n) is None]
            if missing:
                raise ValueError(f"TRADE requires {missing}")
        elif not (self.no_trade_reason or "").strip():
            raise ValueError("NO_TRADE requires no_trade_reason")
        if self.no_trade_reason is not None and len(self.no_trade_reason) > MAX_ITEM_CHARS:
            raise ValueError("no_trade_reason too long")
        return self


def schema_fingerprint(model: type[BaseModel]) -> str:
    """``Name@<sha256-16>`` of the JSON schema, recorded as ModelRunRecord.output_schema."""
    schema = json.dumps(model.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return f"{model.__name__}@{DRAFT_SCHEMA_VERSION}#{hashlib.sha256(schema.encode()).hexdigest()[:16]}"


# ------------------------------------------------------------------------------------------------
# deterministic conversion
# ------------------------------------------------------------------------------------------------


def _parse_ts(text: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt


def citation_problems(
    citations: Sequence[Citation], evidence_ids: Sequence[str], index: EvidenceIndex, *, for_trade: bool
) -> list[str]:
    """Every citation must reference visible evidence with the exact source id and available_at timestamp."""
    problems: list[str] = []
    for c in citations:
        a = index.get(c.evidence_id)
        if a is None:
            problems.append(f"cited evidence {c.evidence_id!r} is not in the provided evidence set")
            continue
        e = a.evidence
        if not a.visible:
            problems.append(f"cited evidence {c.evidence_id} is excluded ({'; '.join(a.problems)})")
        if c.source_id != e.source_id:
            problems.append(f"citation {c.evidence_id}: source_id {c.source_id!r} != {e.source_id!r}")
        ts = _parse_ts(c.available_at)
        if ts is None:
            problems.append(f"citation {c.evidence_id}: available_at {c.available_at!r} is not an exact UTC timestamp")
        elif ts != e.available_at:
            problems.append(f"citation {c.evidence_id}: available_at {c.available_at} != {e.available_at.isoformat()}")
        if e.available_at > index.data_asof:
            problems.append(f"citation {c.evidence_id}: available_at after data_asof (look-ahead)")
        if for_trade and a.visible and not a.supports_trade:
            problems.append(f"evidence {c.evidence_id} is injection-flagged ({a.severity.name}) and cannot support a TRADE")
    for eid in evidence_ids:
        if index.get(eid) is None:
            problems.append(f"evidence id {eid!r} is not in the provided evidence set")
    return problems


def output_text_problems(texts: Sequence[str | None]) -> list[str]:
    """Model output echoing injection markers (instruction overrides, tool-call syntax, order requests) is not
    trusted to support a TRADE."""
    flags: set[str] = set()
    for t in texts:
        if t:
            flags.update(f"{f.code}" for f in scan_text(t) if f.severity >= Severity.HIGH)
    return [f"model output contains injection markers: {sorted(flags)}"] if flags else []


def _dedupe(ids: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for i in ids:
        seen.setdefault(i, None)
    return tuple(seen)


@dataclass(frozen=True)
class FindingConversion:
    finding: ResearchFinding
    downgraded: tuple[str, ...]  # reasons a model TRADE was converted to NO_TRADE


def finding_from_draft(
    draft: ResearchFindingDraft,
    *,
    causal_id: str,
    model_run_id: str,
    candidate: QuantSignal,
    index: EvidenceIndex,
    now: datetime,
    extra_problems: Sequence[str] = (),
) -> FindingConversion:
    problems: list[str] = list(extra_problems)
    if draft.symbol != candidate.symbol:
        problems.append(f"finding symbol {draft.symbol!r} != candidate {candidate.symbol}")
    trade = draft.recommendation == "TRADE"
    problems += citation_problems(draft.citations, draft.evidence_ids, index, for_trade=trade)
    if trade:
        if not draft.citations:
            problems.append("TRADE finding cites no evidence")
        problems += output_text_problems([draft.thesis, *draft.counterarguments, *draft.risks])
    known_ids = [e for e in draft.evidence_ids if index.get(e) is not None]
    recommendation = DecisionAction.TRADE if trade and not problems else DecisionAction.NO_TRADE
    reason = draft.no_trade_reason
    if trade and problems:
        reason = "downgraded from TRADE: " + "; ".join(problems)
    elif not trade and problems:
        reason = f"{draft.no_trade_reason} (citation issues: {'; '.join(problems)})"
    finding = ResearchFinding(
        finding_id=new_id("find"),
        causal_id=causal_id,
        model_run_id=model_run_id,
        symbol=candidate.symbol,
        recommendation=recommendation,
        stance=Stance(draft.stance),
        thesis=draft.thesis[:MAX_THESIS_CHARS],
        evidence_ids=_dedupe(known_ids),
        counterarguments=tuple(draft.counterarguments),
        risks=tuple(draft.risks),
        confidence_label=ConfidenceLabel(draft.confidence_label),
        no_trade_reason=(reason or None) if recommendation == DecisionAction.NO_TRADE else None,
        data_asof=index.data_asof,
        created_at=now,
    )
    return FindingConversion(finding=finding, downgraded=tuple(problems) if trade else ())


def _dec(text: str) -> Decimal | None:
    try:
        d = Decimal(text)
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


@dataclass(frozen=True)
class ProposalContext:
    causal_id: str
    candidate: QuantSignal
    snapshot: MarketSnapshot
    policy: RiskPolicy
    index: EvidenceIndex
    now: datetime
    model_run_ids: tuple[str, ...]
    findings: tuple[ResearchFinding, ...]
    # Two-model rule for the API-key pipeline (Sol TRADE needs a Claude TRADE review). The single-model
    # Sign-in-with-ChatGPT research turn sets this False; every other rule (and the risk engine) still applies.
    require_deep_review: bool = True


def no_trade_proposal(
    reason: str,
    *,
    causal_id: str,
    now: datetime,
    snapshot: MarketSnapshot,
    policy: RiskPolicy,
    candidate: QuantSignal | None,
    evidence_ids: Sequence[str] = (),
    finding_ids: Sequence[str] = (),
    model_run_ids: Sequence[str] = (),
    rationale: str = "",
) -> TradeProposal:
    return TradeProposal(
        proposal_id=new_id("prop"),
        causal_id=causal_id,
        created_at=now,
        data_asof=snapshot.data_asof,
        action=DecisionAction.NO_TRADE,
        no_trade_reason=reason[:4000],
        proposer=Proposer.AI_PIPELINE,
        strategy_id=candidate.strategy_id if candidate else "ai-research",
        symbol=candidate.symbol if candidate else None,
        evidence_ids=_dedupe(evidence_ids),
        signal_id=candidate.signal_id if candidate else None,
        finding_ids=_dedupe(finding_ids),
        model_run_ids=_dedupe(model_run_ids),
        policy_version_seen=policy.version,
        dataset_version=snapshot.dataset_version,
        rationale=rationale[:4000],
    )


def _rationale(draft: TradeProposalDraft) -> str:
    parts = [draft.thesis.strip()]
    if draft.counterarguments:
        parts.append("Counterarguments: " + " | ".join(draft.counterarguments))
    if draft.risks:
        parts.append("Risks: " + " | ".join(draft.risks))
    parts.append(f"Confidence (uncalibrated): {draft.confidence_label}")
    return "\n".join(parts)[:4000]


def proposal_from_draft(draft: TradeProposalDraft, ctx: ProposalContext) -> TradeProposal:
    """Deterministic conversion. Any failed rule yields NO_TRADE with every reason listed."""
    cand = ctx.candidate
    finding_ids = [f.finding_id for f in ctx.findings]
    known_ids = [e for e in draft.evidence_ids if ctx.index.get(e) is not None]
    base_ids = [*ctx.index.required_ids(), *known_ids]

    def _no(reason: str) -> TradeProposal:
        return no_trade_proposal(
            reason, causal_id=ctx.causal_id, now=ctx.now, snapshot=ctx.snapshot, policy=ctx.policy, candidate=cand,
            evidence_ids=base_ids, finding_ids=finding_ids, model_run_ids=ctx.model_run_ids, rationale=_rationale(draft),
        )

    if ctx.index.blocking:
        return _no("required evidence unavailable: " + "; ".join(ctx.index.blocking))
    if draft.recommendation == "NO_TRADE":
        return _no(draft.no_trade_reason or "orchestrator returned NO_TRADE")

    problems: list[str] = []
    # -- two-model rule: a TRADE needs an independent deep review that recommended TRADE, and no NO_TRADE review
    if not ctx.findings and ctx.require_deep_review:
        problems.append("no independent deep review (Claude) finding supports this TRADE")
    elif any(f.recommendation == DecisionAction.NO_TRADE for f in ctx.findings):
        reasons = [f.no_trade_reason or "" for f in ctx.findings if f.recommendation == DecisionAction.NO_TRADE]
        problems.append("deep review recommended NO_TRADE: " + "; ".join(reasons))
    # -- identity
    if draft.symbol != cand.symbol or not _SYMBOL_RE.match(draft.symbol or ""):
        problems.append(f"draft symbol {draft.symbol!r} != candidate {cand.symbol}")
    if draft.side != cand.side.value:
        problems.append(f"draft side {draft.side!r} != candidate side {cand.side.value}")
    if draft.side == Side.SELL.value:
        problems.append("SELL proposals are not produced by the AI pipeline (long-only entries)")
    # -- citations / evidence
    if not draft.citations:
        problems.append("TRADE cites no evidence")
    problems += citation_problems(draft.citations, draft.evidence_ids, ctx.index, for_trade=True)
    problems += output_text_problems([draft.thesis, *draft.counterarguments, *draft.risks])
    # -- requested order parameters (requests only; the risk engine recomputes everything)
    qty = _dec(draft.quantity or "")
    limit = _dec(draft.limit_price or "")
    stop = _dec(draft.stop_price) if draft.stop_price is not None else None
    if qty is None or qty <= 0 or qty != qty.to_integral_value() or qty > MAX_WHOLE_SHARES:
        problems.append(f"requested quantity {draft.quantity!r} is not a positive whole number of shares")
    if limit is None or limit <= 0:
        problems.append(f"requested limit_price {draft.limit_price!r} is not a positive price")
    if draft.stop_price is not None and (stop is None or stop <= 0):
        problems.append(f"requested stop_price {draft.stop_price!r} is not a positive price")
    if problems:
        return _no("; ".join(problems))
    assert qty is not None and limit is not None
    limit = round_price(limit, side_buy=True)
    if stop is None:
        stop = round_price(limit * (1 - ctx.policy.default_stop_loss_bps / BPS), side_buy=False)
    else:
        stop = round_price(stop, side_buy=False)
    if stop >= limit:
        return _no(f"protective stop {stop} must be below the BUY limit {limit}")
    max_loss = qty * (limit - stop)  # informational: the risk engine recomputes from its own reference price
    return TradeProposal(
        proposal_id=new_id("prop"),
        causal_id=ctx.causal_id,
        created_at=ctx.now,
        data_asof=ctx.snapshot.data_asof,
        action=DecisionAction.TRADE,
        proposer=Proposer.AI_PIPELINE,
        strategy_id=cand.strategy_id,
        symbol=cand.symbol,
        side=cand.side,
        quantity=qty,
        order_type=OrderType.LIMIT,
        limit_price=limit,
        stop_price=stop,
        max_planned_loss=max_loss,
        evidence_ids=_dedupe(base_ids),
        signal_id=cand.signal_id,
        finding_ids=_dedupe(finding_ids),
        model_run_ids=_dedupe(ctx.model_run_ids),
        policy_version_seen=ctx.policy.version,
        dataset_version=ctx.snapshot.dataset_version,
        rationale=_rationale(draft),
    )
