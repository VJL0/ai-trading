"""Strict Pydantic input/output contracts of the ONE validated agent tool interface.

Rules (enforced here, relied on by every transport — HTTP, MCP, OpenAI Responses function tools, PydanticAI):

* Every model is ``extra="forbid"`` and frozen: unknown fields are rejected, never ignored.
* Money / price / quantity inputs are ``packages.contracts.Dec`` behind a *decimal-string-only* gate: a binary
  float (Python ``float``) and a JSON number with a fraction (which the HTTP layer parses to ``Decimal``) are
  both rejected, so the same input is accepted or refused identically on every transport. Whole-share
  quantities may also be JSON integers (exact).
* Symbols match the shared pattern; ids match a conservative id pattern.
* Free text is cleaned (control/zero-width/bidi characters removed), length-capped and screened for
  instruction-shaped content (``services.tools.sanitize``).
* Outputs are compact (token-economical) and *always* carry ``execution_label``, ``is_simulated`` and ``asof``
  (decision-clock time; the simulated clock under replay) plus the data's own timestamps.

Quantities and limit prices in ``propose_paper_trade`` are *requests*: the deterministic risk engine recomputes
reference price, notional, exposure and planned loss and may reject. Nothing here can resume trading, change
limits or mode, place live orders, read secrets or touch the filesystem.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Final, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, WithJsonSchema, model_validator

from packages.contracts import Dec, UtcDatetime
from packages.contracts.enums import ExecutionLabel

from .sanitize import (
    MAX_COUNTERARGUMENT_CHARS,
    MAX_REASON_CHARS,
    MAX_THESIS_CHARS,
    clean_text,
    neutralize_text,
    screen_text,
)

SYMBOL_PATTERN: Final = r"^[A-Z][A-Z0-9.]{0,9}$"
ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
DECIMAL_PATTERN: Final = r"^[0-9]{1,9}(\.[0-9]{1,6})?$"
WHOLE_SHARES_PATTERN: Final = r"^[1-9][0-9]{0,5}$"
MAX_WHOLE_SHARES: Final = 100_000

_DECIMAL_RE: Final = re.compile(DECIMAL_PATTERN)
_SHARES_RE: Final = re.compile(WHOLE_SHARES_PATTERN)


# ------------------------------------------------------------------------------------------------ primitives
def _decimal_text_only(value: Any) -> Any:
    """Gate in front of ``Dec``: only decimal *strings* (or exact ints) reach the Decimal parser."""
    if isinstance(value, bool):
        raise ValueError("booleans are not valid prices")
    if isinstance(value, int):
        value = str(value)  # exact; still subject to the same plain-decimal pattern (no negatives, <= 9 digits)
    if isinstance(value, str):
        text = value.strip()
        if not _DECIMAL_RE.fullmatch(text):
            raise ValueError("expected a plain decimal string such as '412.35'")
        return text
    raise ValueError(
        "money/price fields must be JSON strings such as '412.35'; binary floats and fractional JSON numbers are rejected"
    )


def _shares_text_only(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid quantities")
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str):
        text = value.strip()
        if not _SHARES_RE.fullmatch(text) or int(text) > MAX_WHOLE_SHARES:
            raise ValueError(f"quantity must be a whole number of shares between 1 and {MAX_WHOLE_SHARES}, e.g. '10'")
        return text
    raise ValueError("quantity must be a whole number of shares as a string or integer; floats are rejected")


PriceIn = Annotated[
    Dec,
    BeforeValidator(_decimal_text_only),
    WithJsonSchema({"type": "string", "pattern": DECIMAL_PATTERN,
                    "description": "decimal string in USD, e.g. '412.35' (never a JSON number)"}),
]
QuantityIn = Annotated[
    Dec,
    BeforeValidator(_shares_text_only),
    WithJsonSchema({"type": "string", "pattern": WHOLE_SHARES_PATTERN,
                    "description": "whole shares as a string, e.g. '10'"}),
]
Symbol = Annotated[str, Field(pattern=SYMBOL_PATTERN, description="upper-case US stock/ETF ticker, e.g. 'SPY'")]
RecordId = Annotated[str, Field(pattern=ID_PATTERN)]

ThesisText = Annotated[
    str, BeforeValidator(clean_text), Field(min_length=20, max_length=MAX_THESIS_CHARS), AfterValidator(screen_text)
]
CounterText = Annotated[
    str, BeforeValidator(clean_text), Field(min_length=5, max_length=MAX_COUNTERARGUMENT_CHARS),
    AfterValidator(screen_text),
]
# A halt is the fail-safe direction: its reason is never a reason to refuse it. Instruction-shaped text is
# replaced by an inert placeholder instead of rejecting the call.
HaltReasonText = Annotated[
    str, BeforeValidator(clean_text), Field(min_length=5, max_length=MAX_REASON_CHARS), AfterValidator(neutralize_text)
]


# ------------------------------------------------------------------------------------------------ inputs
class ToolInput(BaseModel):
    """Base of every tool input: unknown fields rejected; ``null`` for an optional field means "default"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data: Any) -> Any:
        if isinstance(data, dict):
            fields = cls.model_fields
            return {k: v for k, v in data.items() if not (v is None and k in fields and not fields[k].is_required())}
        return data


class NoInput(ToolInput):
    """Tool takes no arguments (pass an empty object)."""


class GetMarketSnapshotIn(ToolInput):
    symbol: Symbol


class GetBarsIn(ToolInput):
    symbol: Symbol
    timeframe: Literal["1Min", "1Day"] = Field("1Min", description="bar size; default '1Min'")
    limit: int = Field(30, ge=1, le=200, strict=True, description="most recent N bars (1-200, default 30)")


class GetPredictionsIn(ToolInput):
    symbol: Symbol | None = Field(None, description="one symbol, or null for the whole configured universe")


class GetSignalsIn(ToolInput):
    limit: int = Field(10, ge=1, le=50, strict=True, description="most recent N quant signals (1-50, default 10)")
    symbol: Symbol | None = Field(None, description="filter to one symbol, or null")
    candidates_only: bool = Field(False, strict=True, description="only signals that cleared the net-of-cost threshold")


class GetOrdersIn(ToolInput):
    limit: int = Field(20, ge=1, le=50, strict=True, description="most recent N orders (1-50, default 20)")
    open_only: bool = Field(False, strict=True, description="only non-terminal orders")


class GetOrderIn(ToolInput):
    client_order_id: RecordId = Field(description="client_order_id from get_orders / propose_paper_trade")


class ProposePaperTradeIn(ToolInput):
    symbol: Symbol
    side: Literal["BUY", "SELL"] = Field(description="BUY opens/adds to a long; SELL only reduces an existing long")
    quantity: QuantityIn = Field(description="requested whole shares; the risk engine may reject, never resizes")
    limit_price: PriceIn = Field(description="requested limit price (decimal string); rounded to a valid tick")
    stop_price: PriceIn | None = Field(
        None, description="optional protective stop below a BUY limit (decimal string); null = policy default stop"
    )
    thesis: ThesisText = Field(description="why this trade, citing the evidence ids; plain market reasoning only")
    evidence_ids: list[RecordId] = Field(
        min_length=1, max_length=12,
        description="ids returned by earlier tool calls in this session (signal_id, prediction_id, snapshot_id, "
                    "registry_version); at least one must belong to this symbol",
    )
    counterarguments: list[CounterText] = Field(min_length=1, max_length=5, description="strongest reasons not to trade")


class RequestHaltIn(ToolInput):
    reason: HaltReasonText = Field(description="why new entries should stop (a human must resume)")


class CancelPaperOrderIn(ToolInput):
    client_order_id: RecordId = Field(description="client_order_id of an open PAPER/SIMULATED order")


class RunQuantCycleIn(ToolInput):
    symbols: list[Symbol] | None = Field(
        None, min_length=1, max_length=10, description="subset of the configured universe, or null for all"
    )


# ------------------------------------------------------------------------------------------------ outputs
class OutModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolOutput(OutModel):
    """Every tool output carries where the numbers come from and when."""

    execution_label: ExecutionLabel = Field(description="SIMULATED | PAPER (LIVE never occurs in this build)")
    is_simulated: bool = Field(description="true when data or execution is simulated/synthetic")
    asof: UtcDatetime = Field(description="decision-clock time of this response (simulated clock under replay)")


JsonScalar = str | int | float | bool | None


# -- market
class QuoteOut(OutModel):
    bid: Dec
    ask: Dec
    bid_size: Dec
    ask_size: Dec
    mid: Dec
    spread_bps: Dec
    exchange_ts: UtcDatetime
    available_at_ts: UtcDatetime


class BarOut(OutModel):
    t: UtcDatetime = Field(description="bar start time")
    o: Dec
    h: Dec
    l: Dec  # noqa: E741 - compact OHLCV keys are part of the token-economical wire shape
    c: Dec
    v: Dec


class MarketSnapshotOut(ToolOutput):
    snapshot_id: str
    symbol: str
    feed: str
    feed_mode: str
    session: str
    data_asof: UtcDatetime
    dataset_version: str
    source_id: str
    quote: QuoteOut | None = Field(description="null when no fresh point-in-time quote exists (entries fail closed)")
    last_trade_price: Dec | None
    last_bar: BarOut | None
    limitations: list[str]


class BarsOut(ToolOutput):
    symbol: str
    timeframe: str
    feed: str
    feed_mode: str
    dataset_version: str | None
    source_id: str | None
    last_available_at: UtcDatetime | None
    limitations: list[str]
    bars: list[BarOut]


class FeedSymbolOut(OutModel):
    symbol: str
    last_quote_ts: UtcDatetime | None
    staleness_s: Dec | None
    spread_bps: Dec | None
    gaps_today: int
    rejected_today: int


class FeedEventOut(OutModel):
    ts: UtcDatetime | None
    symbol: str | None
    kind: str | None
    severity: str | None


class FeedQualityOut(ToolOutput):
    feed_mode: str
    provider: str | None
    feed: str | None
    status: str
    detail: str | None
    staleness_s: Dec | None
    max_staleness_s: Dec | None
    last_bar_ts: UtcDatetime | None
    last_quote_ts: UtcDatetime | None
    limitations: list[str]
    symbols: list[FeedSymbolOut]
    recent_events: list[FeedEventOut]


# -- models
class ForecastOut(OutModel):
    horizon_minutes: int
    expected_return_bps: Dec
    model_id: str


class PredictionOut(OutModel):
    prediction_id: str
    symbol: str
    asof: UtcDatetime
    features_asof: UtcDatetime
    forecasts: list[ForecastOut]
    success_probability: Dec | None
    success_horizon_minutes: int | None
    probability_calibrated: bool
    calibration_method: str | None
    rank: int | None
    universe_size: int
    vol_forecast_bps: Dec | None
    regime: str
    est_cost_bps: Dec | None
    is_simulated: bool


class PredictionsOut(ToolOutput):
    registry_version: str | None
    detail: str | None
    predictions: list[PredictionOut]


class CandidateOut(OutModel):
    model_id: str
    metrics: dict[str, JsonScalar]


class RejectedModelOut(OutModel):
    model_id: str
    reason: str | None


class TargetOut(OutModel):
    target: str
    selected_model: str | None
    selected: CandidateOut | None
    n_candidates: int
    rejected: list[RejectedModelOut]
    calibration: dict[str, JsonScalar] | None


class ModelReportOut(ToolOutput):
    registry_version: str | None
    trained_at: UtcDatetime | None
    dataset_version: str | None
    selection_metric: str | None
    targets: list[TargetOut]
    walk_forward: dict[str, JsonScalar | dict[str, JsonScalar]] | None
    multiple_testing: dict[str, JsonScalar | list[str]] | None
    notes: list[str]


class SignalOut(OutModel):
    signal_id: str
    symbol: str
    strategy_id: str
    model_version: str
    side: str
    horizon_minutes: int
    score: Dec
    rank: int
    expected_return_bps: Dec
    est_total_cost_bps: Dec
    net_edge_bps: Dec
    probability_up: Dec | None
    probability_calibrated: bool
    is_candidate: bool
    reason: str
    features_asof: UtcDatetime
    generated_at: UtcDatetime
    is_simulated: bool


class CycleBriefOut(OutModel):
    causal_id: str | None
    ran_at: UtcDatetime | None
    candidate_symbol: str | None
    outcome: str | None
    reason: str | None


class SignalsOut(ToolOutput):
    signals: list[SignalOut]
    latest_cycle: CycleBriefOut | None
    skipped_invalid: int = Field(description="persisted rows that failed contract validation and were withheld")


# -- portfolio
class AccountOut(OutModel):
    status: str
    is_paper: bool
    equity: Dec
    cash: Dec
    buying_power: Dec
    trading_blocked: bool
    account_blocked: bool
    asof: UtcDatetime
    source: str


class BrokerPositionOut(OutModel):
    symbol: str
    quantity: Dec
    avg_entry_price: Dec
    market_value: Dec | None
    unrealized_pl: Dec | None
    asof: UtcDatetime
    source: str


class LedgerPositionOut(OutModel):
    symbol: str
    quantity: Dec
    avg_cost: Dec
    realized_pl: Dec


class PnlOut(OutModel):
    daily_realized: Dec
    daily_unrealized: Dec
    start_of_day_equity: Dec | None


class PortfolioOut(ToolOutput):
    account: AccountOut | None
    broker_positions: list[BrokerPositionOut]
    ledger_positions: list[LedgerPositionOut]
    pnl: PnlOut


class OrderOut(OutModel):
    client_order_id: str
    symbol: str
    side: str
    quantity: Dec
    filled_quantity: Dec
    avg_fill_price: Dec | None
    limit_price: Dec | None
    state: str
    gateway: str
    execution_label: str
    created_at: UtcDatetime
    updated_at: UtcDatetime


class OrdersOut(ToolOutput):
    orders: list[OrderOut]


class TransitionOut(OutModel):
    from_state: str | None
    to_state: str
    at: UtcDatetime
    reason: str


class FillOut(OutModel):
    fill_id: str
    quantity: Dec
    price: Dec
    ts: UtcDatetime


class IntentBriefOut(OutModel):
    intent_id: str
    proposal_id: str
    policy_version: str
    expires_at: UtcDatetime
    requires_human_approval: bool


class OrderDetailOut(ToolOutput):
    order: OrderOut
    intent: IntentBriefOut | None
    transitions: list[TransitionOut]
    fills: list[FillOut]
    n_events: int


class HaltOut(OutModel):
    halt_id: str
    reason: str
    source: str
    created_at: UtcDatetime | None


class UtilizationOut(OutModel):
    name: str
    used: Dec
    limit: Dec
    pct: Dec


class DecisionBriefOut(OutModel):
    decision_id: str
    proposal_id: str
    verdict: str
    reasons: list[str]
    evaluated_at: UtcDatetime


class RiskLimitsOut(OutModel):
    long_only: bool
    regular_session_only: bool
    max_positions: int
    max_pending_orders: int
    max_orders_per_minute: int
    max_orders_per_day: int
    max_order_notional: Dec
    max_position_notional: Dec
    max_concentration_pct: Dec
    max_daily_loss: Dec
    max_planned_loss_per_trade: Dec
    max_spread_bps: Dec
    max_quote_age_seconds: Dec
    max_limit_deviation_bps: Dec
    default_stop_loss_bps: Dec
    min_price: Dec
    min_account_equity: Dec
    symbol_allowlist: list[str]


class RiskStatusOut(ToolOutput):
    mode: str
    live_execution_compiled: bool
    policy_version: str
    entries_locked: bool
    lock_reasons: list[str]
    halts: list[HaltOut]
    limits: RiskLimitsOut = Field(description="read-only; only a human operator can change them")
    utilization: list[UtilizationOut]
    recent_decisions: list[DecisionBriefOut]
    agent_max_proposals_per_hour: int


# -- paper trading (mutating)
class RequestedOrderOut(OutModel):
    symbol: str
    side: str
    quantity: Dec
    limit_price: Dec
    stop_price: Dec | None


class RiskDecisionOut(OutModel):
    decision_id: str
    verdict: str
    failed_checks: list[str]
    reasons: list[str]
    recomputed_quantity: Dec | None
    recomputed_notional: Dec | None
    recomputed_max_loss: Dec | None
    reference_price: Dec | None
    policy_version: str


class OrderStateOut(OutModel):
    client_order_id: str
    state: str
    filled_quantity: Dec | None
    avg_fill_price: Dec | None


ProposalOutcome = Literal[
    "SUBMITTED", "REJECTED", "BLOCKED", "AWAITING_HUMAN_APPROVAL", "RATE_LIMITED", "NO_TRADE", "ERROR", "UNKNOWN"
]


class ProposePaperTradeOut(ToolOutput):
    outcome: ProposalOutcome
    reason: str | None
    proposal_id: str
    causal_id: str
    data_asof: UtcDatetime
    requested: RequestedOrderOut = Field(description="what was requested after tick rounding (a request only)")
    decision: RiskDecisionOut | None = Field(description="the independent deterministic risk decision")
    client_order_id: str | None
    order: OrderStateOut | None


class RequestHaltOut(ToolOutput):
    halt_id: str
    entries_locked: bool
    detail: str


class CancelPaperOrderOut(ToolOutput):
    client_order_id: str
    outcome: Literal["CANCEL_REQUESTED", "ALREADY_TERMINAL"]
    state: str | None


class RunQuantCycleOut(ToolOutput):
    outcome: str = Field(description="NO_TRADE | REJECTED | SUBMITTED | BLOCKED | AWAITING_HUMAN_APPROVAL | ERROR | RATE_LIMITED")
    causal_id: str | None
    reason: str | None
    candidate_symbol: str | None
    signals: int | None
    client_order_id: str | None
    entries_locked: bool | None
    lock_reasons: list[str]
    ran_at: UtcDatetime | None



# ================================================================================================ sessions & ops
# Natural-language operation: autonomous PAPER sessions run by the local service (services.sessions), schedules,
# reports, explanations, performance, activity and health. The four session-control inputs below can only start /
# stop / schedule / cancel PAPER sessions; every order inside a session still goes through the risk engine.
WEEKDAY_CODES: Final[tuple[str, ...]] = ("MON", "TUE", "WED", "THU", "FRI")
START_TIME_PATTERN: Final = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
WeekdayCode = Literal["MON", "TUE", "WED", "THU", "FRI"]
ActivityKind = Literal["scan", "predict", "evaluate", "gpt_call", "risk", "reject", "submit", "fill", "reconcile",
                       "session", "agent_tool", "error", "info", "stage", "decision"]
# Stopping is the fail-safe direction (like a halt): instruction-shaped text is neutralized, never refused.
StopReasonText = Annotated[
    str, BeforeValidator(clean_text), Field(min_length=1, max_length=MAX_REASON_CHARS), AfterValidator(neutralize_text)
]


# Same pattern as ``Symbol`` without the per-item description (keeps the catalog compact).
SessionSymbol = Annotated[str, Field(pattern=SYMBOL_PATTERN)]


class StartPaperSessionIn(ToolInput):
    duration_minutes: int = Field(ge=1, le=480, strict=True, description="minutes; 1 hour = 60")
    symbols: list[SessionSymbol] | None = Field(None, min_length=1, max_length=10, description="tickers; null = all")
    cadence_minutes: int = Field(5, ge=1, le=60, strict=True, description="minutes between cycles (default 5)")
    use_ai: bool = Field(True, strict=True, description="in-app model review if available (default true)")


class StopSessionIn(ToolInput):
    reason: StopReasonText = Field(description="why stop")
    session_id: RecordId | None = Field(None, description="null = the active session")


class ScheduleSessionIn(ToolInput):
    weekdays: list[WeekdayCode] = Field(min_length=1, max_length=5)
    start_time_et: str = Field(pattern=START_TIME_PATTERN, description="New York time, 09:30-16:00, e.g. '09:45'")
    duration_minutes: int = Field(ge=1, le=390, strict=True, description="minutes; must end by 16:00 ET")
    symbols: list[SessionSymbol] | None = Field(None, min_length=1, max_length=10, description="tickers; null = all")
    cadence_minutes: int = Field(5, ge=1, le=60, strict=True, description="minutes between cycles (default 5)")
    use_ai: bool = Field(True, strict=True, description="in-app model review if available (default true)")


class CancelScheduleIn(ToolInput):
    schedule_id: RecordId


class GetSessionReportIn(ToolInput):
    session_id: RecordId | None = Field(None, description="null = active or most recent session")


class ExplainDecisionsIn(ToolInput):
    limit: int = Field(10, ge=1, le=50, strict=True, description="last N cycles (default 10)")


class GetActivityIn(ToolInput):
    limit: int = Field(20, ge=1, le=100, strict=True, description="last N events (default 20)")
    kind: ActivityKind | None = Field(None, description="filter; null = all")


class SessionOut(OutModel):
    session_id: str
    state: str = Field(description="SCHEDULED|RUNNING|STOPPING|COMPLETED|STOPPED|INTERRUPTED|FAILED")
    started_at: UtcDatetime | None
    ends_at: UtcDatetime
    ended_at: UtcDatetime | None
    duration_minutes: int
    symbols: list[str]
    cadence_minutes: int
    use_ai: bool
    requested_by: str
    cycles_run: int
    trades: int
    stop_reason: str | None
    schedule_id: str | None
    next_cycle_at: UtcDatetime | None
    last_cycle_at: UtcDatetime | None
    ai_suspended_until: UtcDatetime | None


class ScheduleRunOut(OutModel):
    run_date: str
    status: str = Field(description="STARTED | MISSED | SKIPPED")
    session_id: str | None
    due_at: UtcDatetime


class ScheduleOut(OutModel):
    schedule_id: str
    weekdays: list[str]
    start_time_et: str
    duration_minutes: int
    symbols: list[str]
    cadence_minutes: int
    use_ai: bool
    enabled: bool
    next_run_at: UtcDatetime | None
    created_by: str
    last_run: ScheduleRunOut | None


class StartPaperSessionOut(ToolOutput):
    outcome: Literal["STARTED"]
    session: SessionOut
    detail: str


class StopSessionOut(ToolOutput):
    outcome: Literal["STOPPED", "STOPPING", "ALREADY_ENDED", "NO_ACTIVE_SESSION"]
    session: SessionOut | None
    detail: str


class SessionStatusOut(ToolOutput):
    active: SessionOut | None
    recent: list[SessionOut]
    schedules: list[ScheduleOut]
    entries_locked: bool
    lock_reasons: list[str]
    market_open: bool | None = Field(description="XNYS regular hours open now (decision clock)")
    next_open: UtcDatetime | None
    scheduler_running: bool | None
    detail: str


class ScheduleSessionOut(ToolOutput):
    outcome: Literal["SCHEDULED"]
    schedule: ScheduleOut
    detail: str


class ListSchedulesOut(ToolOutput):
    schedules: list[ScheduleOut]


class CancelScheduleOut(ToolOutput):
    outcome: Literal["CANCELLED", "ALREADY_CANCELLED"]
    schedule: ScheduleOut
    detail: str


class EquityPointOut(OutModel):
    time: int = Field(description="UNIX seconds")
    equity: Dec


class PerformanceOut(ToolOutput):
    mode_label: Literal["BACKTEST", "PAPER", "LIVE", "SIMULATED"]
    portfolio_value: Dec | None
    cash: Dec | None
    daily_pnl: Dec
    total_pnl: Dec
    realized_pnl: Dec
    unrealized_pnl: Dec
    win_rate: Dec | None = Field(description="0..1 over closed round trips; null when none closed")
    round_trips: int
    max_drawdown: Dec = Field(description="0..1 of the equity peak")
    current_drawdown: Dec
    active_positions: int
    open_orders: int
    api_cost_usd: Dec
    equity_points: int
    equity_curve: list[EquityPointOut] = Field(description="most recent points only (compact)")


class PerformanceSummaryOut(OutModel):
    mode_label: str
    realized_pnl: Dec
    unrealized_pnl: Dec
    total_pnl: Dec
    win_rate: Dec | None
    round_trips: int
    max_drawdown: Dec
    portfolio_value: Dec | None


class ReasonCountOut(OutModel):
    reason: str
    count: int


class CheckCountOut(OutModel):
    check: str
    count: int


class TradeOut(OutModel):
    client_order_id: str
    symbol: str
    side: str
    qty: Dec
    filled_qty: Dec
    avg_price: Dec | None
    state: str
    pnl: Dec | None
    pnl_basis: str | None = Field(description="realized (SELL) | unrealized (open BUY lots) | closed | null")


class SessionReportOut(ToolOutput):
    session: SessionOut
    performance: PerformanceSummaryOut
    cycles: int
    decisions: dict[str, int]
    top_no_trade_reasons: list[ReasonCountOut]
    top_risk_rejections: list[CheckCountOut]
    trades: list[TradeOut]
    api_cost_usd: Dec
    narrative: str


class ExplainCycleOut(OutModel):
    causal_id: str
    ran_at: UtcDatetime | None
    session_id: str | None
    outcome: str
    candidate_symbol: str | None
    no_trade_reason: str | None
    failed_checks: list[str]
    entries_locked: bool | None
    lock_reasons: list[str]
    ai_review: str | None
    client_order_id: str | None


class ExplainAggregateOut(OutModel):
    cycles: int
    by_outcome: dict[str, int]
    top_no_trade_reasons: list[ReasonCountOut]
    top_failed_checks: list[CheckCountOut]
    entries_locked_now: bool | None
    lock_reasons_now: list[str]


class ExplainDecisionsOut(ToolOutput):
    answer: str = Field(description="deterministic plain-English summary of the facts below")
    aggregate: ExplainAggregateOut
    cycles: list[ExplainCycleOut]


class ActivityEventOut(OutModel):
    seq: int
    ts: UtcDatetime
    event: str = Field(description="activity | stage | decision")
    kind: str | None
    message: str
    session_id: str | None
    causal_id: str | None


class ActivityOut(ToolOutput):
    events: list[ActivityEventOut]


class HealthComponentOut(OutModel):
    name: str
    status: str = Field(description="OK | DEGRADED | DOWN | DISABLED")
    detail: str
    last_ok: str | None


class SystemHealthOut(ToolOutput):
    overall: Literal["OK", "DEGRADED", "DOWN"]
    components: list[HealthComponentOut]


class SearchMemoryIn(ToolInput):
    query: str = Field(max_length=2000)
    symbol: str = Field(default="", pattern=r"^$|^[A-Z][A-Z0-9.]{0,9}$")
    kind: Literal["", "episodic", "semantic", "research", "reflection"] = ""
    limit: int = Field(default=10, ge=1, le=50)


class MemorySearchOut(ToolOutput):
    memory: dict[str, Any]
