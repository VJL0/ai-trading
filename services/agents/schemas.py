"""Structured outputs of each task kind: pydantic models (crew-side validation) and the strict JSON Schema passed to
the CLIs (Claude ``--json-schema``, Codex ``--output-schema``).

The exported schema is a plain draft 2020-12 subset that both CLIs accept: ``object`` / ``array`` / ``string`` /
``integer`` / ``enum`` only, every object ``additionalProperties: false`` with every property required, no
``$ref``, ``oneOf``/``anyOf``, defaults or numeric/length bounds. Bounds that matter are enforced by the pydantic
models and by the crew (for example only the first ``SPECS_PER_TASK`` specs are considered).

Agent output is a proposal. It contains no performance numbers: the crew records the trial ledger's numbers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import SessionAction, TaskKind
from .mechanisms import KEYS as MECHANISM_KEYS

_TXT = 2000
_ITEM = 600
_LIST = 12


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorSpec(_Strict):
    """Shape only: value checks (expression grammar, n, rebalance) belong to ``validate_spec``, so one bad spec is
    refused on its own instead of invalidating the whole answer."""

    expression: str = Field(description="factor DSL expression; higher score = more attractive long")
    universe: Literal["liquid1000", "large300"] = Field(description="stock universe")
    n: int = Field(description="number of names held (5..100), equal weight, long only")
    rebalance: int = Field(description="rebalance interval in sessions: one of 1, 5, 10, 21, 63")
    regime: Literal["none", "sma200", "volscale"] = Field(description="market exposure overlay")
    intent: str = Field(max_length=_ITEM, description="which part of the mechanism this spec tests and why")


class HypothesisOutput(_Strict):
    mechanism: str = Field(description="catalog key of the economic mechanism (use the one assigned)")
    economic_rationale: str = Field(max_length=_TXT, description="who is on the other side and why the effect "
                                    "should persist after costs")
    why_novel: str = Field(max_length=_TXT, description="how this differs from the listed prior trials")
    falsification: str = Field(max_length=_TXT, description="the observable result that would refute it")
    specs: list[FactorSpec] = Field(max_length=10, description="declarative factor specs (empty when the "
                                    "mechanism is not expressible with the DSL)")
    data_requests: list[str] = Field(max_length=_LIST, description="data that would allow a sharper test")


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    WEAK = "WEAK"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ReviewOutput(_Strict):
    verdict: Verdict = Field(description="independent verdict on the hypothesis given the ledger results")
    concerns: list[str] = Field(max_length=_LIST, description="methodological or economic concerns")
    leakage_or_bias_risks: list[str] = Field(max_length=_LIST, description="look-ahead, survivorship, selection, "
                                             "data-snooping or overfitting-to-the-gate risks")
    mechanism_plausibility: str = Field(max_length=_TXT, description="is the stated mechanism what the spec tests?")
    suggested_tests: list[str] = Field(max_length=_LIST, description="cheap tests that would change the verdict")


class HypothesisSeed(_Strict):
    mechanism: str = Field(description="catalog key of the mechanism")
    rationale: str = Field(max_length=_TXT, description="what in the scan or results suggests it")


class MarketAnalysisOutput(_Strict):
    regime_summary: str = Field(max_length=_TXT, description="current market regime in two or three sentences")
    observations: list[str] = Field(max_length=_LIST, description="observations from the research tools")
    hypothesis_seeds: list[HypothesisSeed] = Field(max_length=6, description="mechanisms worth a hypothesis task")
    data_gaps: list[str] = Field(max_length=_LIST, description="missing or stale data that limited the analysis")


class Proposal(_Strict):
    area: str = Field(max_length=200, description="component or process, e.g. 'factor DSL', 'cost model', 'crew'")
    problem: str = Field(max_length=_TXT, description="what goes wrong")
    evidence: str = Field(max_length=_TXT, description="ledger rows, refusals or run failures that show it")
    proposed_change: str = Field(max_length=_TXT, description="the change an engineer would make")
    expected_benefit: str = Field(max_length=_ITEM, description="what improves")
    risk: str = Field(max_length=_ITEM, description="what could go wrong, including validation risks")


class ImprovementOutput(_Strict):
    proposals: list[Proposal] = Field(max_length=6, description="proposals for the operator (never applied by the crew)")


class Confidence(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AnalysisOutput(_Strict):
    """A lead-assigned open question: read-only investigation, advisory findings."""

    answer: str = Field(max_length=_TXT, description="direct answer to the assigned objective")
    findings: list[str] = Field(max_length=_LIST, description="evidence-backed findings (cite tool outputs or digest rows)")
    confidence: Confidence = Field(description="how well the available evidence supports the answer")
    recommendations: list[str] = Field(max_length=_LIST, description="what the lead should do next, if anything")
    follow_ups: list[str] = Field(max_length=_LIST, description="questions worth a later delegation")
    data_gaps: list[str] = Field(max_length=_LIST, description="missing or stale data that limited the analysis")


class Delegation(_Strict):
    """One unit of work the lead hands to a subagent. The role and objective are the lead's own; the kind fixes the
    output contract and the deterministic handling (only hypothesis specs are evaluated, through the trial ledger)."""

    kind: Literal["hypothesis", "review", "market_analysis", "improvement", "analysis"] = Field(
        description="capability: hypothesis (factor specs, run on DEV data), review (independent critique of a "
                    "finished hypothesis task), market_analysis (scan digest), improvement (proposals for the "
                    "operator), analysis (any read-only question)")
    role: str = Field(max_length=120, description="the subagent's role or persona, chosen by you")
    objective: str = Field(max_length=_ITEM, description="what the subagent must find out or produce")
    instructions: str = Field(max_length=1500, description="method, scope and what to ignore")
    rationale: str = Field(max_length=_ITEM, description="the evidence that makes this worth subscription quota now")
    priority: int = Field(description="1 (low) .. 5 (urgent)")
    provider: Literal["any", "claude", "codex"] = Field(description="preferred provider (a hint; budgets decide)")
    mechanism: str = Field(description="catalog mechanism key for kind=hypothesis, else empty")
    subject_task_id: str = Field(description="the hypothesis task id for kind=review, else empty")


class LeadSessionAction(_Strict):
    action: SessionAction = Field(description="stand_down_today (no session today, or stop today's autopilot "
                                  "session) or request_halt (halt new entries until the operator lifts it)")
    reason: str = Field(max_length=300, description="why, in one sentence")
    evidence: list[str] = Field(max_length=6, description="the briefing facts that justify it")


class LeadDecision(_Strict):
    situation: str = Field(max_length=_TXT, description="your assessment of the day: session, risk, research state")
    priorities: list[str] = Field(max_length=8, description="ranked priorities for the rest of the day")
    delegations: list[Delegation] = Field(max_length=12, description="work to hand to parallel subagents now "
                                          "(empty when nothing is worth the quota)")
    session_actions: list[LeadSessionAction] = Field(max_length=2, description="fail-safe session requests; "
                                                     "usually empty")
    memory_notes: list[str] = Field(max_length=5, description="durable lessons worth remembering on later days")
    next_wake_minutes: int = Field(description="when you want your next turn (the scheduler clamps it)")


OUTPUT_MODELS: dict[TaskKind, type[_Strict]] = {
    TaskKind.HYPOTHESIS: HypothesisOutput,
    TaskKind.REVIEW: ReviewOutput,
    TaskKind.MARKET_ANALYSIS: MarketAnalysisOutput,
    TaskKind.IMPROVEMENT: ImprovementOutput,
    TaskKind.ANALYSIS: AnalysisOutput,
    TaskKind.LEAD: LeadDecision,
}

_KEEP = frozenset({"type", "properties", "required", "additionalProperties", "items", "enum", "description"})


def _strictify(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_strictify(x, defs) for x in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
        return _strictify(merged, defs)
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k not in _KEEP:
            continue
        if k == "properties":
            out[k] = {pk: _strictify(pv, defs) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _strictify(v, defs)
        else:
            out[k] = v
    if out.get("type") == "object" or "properties" in out:
        out["type"] = "object"
        out["additionalProperties"] = False
        out["required"] = list(out.get("properties", {}))
    if "enum" in out and "type" not in out:
        out["type"] = "string"
    return out


def output_schema(kind: TaskKind, *, specs_per_task: int = 3) -> dict[str, Any]:
    """Strict JSON Schema for one task kind's final answer."""
    model = OUTPUT_MODELS[TaskKind(kind)]
    raw = model.model_json_schema()
    schema = _strictify(raw, raw.get("$defs", {}))
    props = schema["properties"]
    for field in ("mechanism",):
        if field in props:
            props[field]["enum"] = list(MECHANISM_KEYS)
            props[field]["type"] = "string"
    if kind is TaskKind.MARKET_ANALYSIS:
        seed = props["hypothesis_seeds"]["items"]["properties"]["mechanism"]
        seed["enum"], seed["type"] = list(MECHANISM_KEYS), "string"
    if kind is TaskKind.HYPOTHESIS:
        spec = props["specs"]["items"]["properties"]
        spec["rebalance"]["enum"] = [1, 5, 10, 21, 63]
        props["specs"]["description"] = (f"at most {specs_per_task} declarative factor specs (empty when the "
                                         "mechanism is not expressible with the DSL)")
    if kind is TaskKind.LEAD:
        mech = props["delegations"]["items"]["properties"]["mechanism"]
        mech["enum"], mech["type"] = ["", *MECHANISM_KEYS], "string"
    return schema


def kind_of_schema(schema: dict[str, Any]) -> TaskKind | None:
    """Recover the task kind from an exported schema (used by the fake provider)."""
    props = set(schema.get("properties", {}))
    for kind, model in OUTPUT_MODELS.items():
        if props == set(model.model_fields):
            return kind
    return None


def parse_output(kind: TaskKind, data: Any) -> _Strict:
    """Validate a provider's structured output; raises ``pydantic.ValidationError``."""
    return OUTPUT_MODELS[TaskKind(kind)].model_validate(data)
