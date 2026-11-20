"""Versioned decision and execution contracts (spec §5).

Every contract is immutable (``frozen``), rejects unknown fields (``extra="forbid"``), uses
``Decimal`` for all money/quantity/price values and timezone-aware UTC timestamps. Each carries
``schema_version`` so persisted records stay interpretable as contracts evolve.

The risk engine never trusts arithmetic in these objects that originates from a model: quantities,
notionals and planned losses are *recomputed* from market and account state.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    Adjustment,
    ConfidenceLabel,
    DataFeed,
    DecisionAction,
    EventSource,
    EvidenceSourceType,
    ExecutionLabel,
    FeedMode,
    GatewayKind,
    MarketRegime,
    MarketSession,
    ModelProvider,
    ModelRunStatus,
    ModelTransport,
    OrderEventType,
    OrderState,
    OrderType,
    Proposer,
    ReconciliationStatus,
    RiskVerdict,
    SessionConstraint,
    Side,
    Stance,
    TimeInForce,
    TradingMode,
    TrustLevel,
)
from .types import Dec, UtcDatetime

SCHEMA_VERSION = "1.0.0"

_SYMBOL_PATTERN = r"^[A-Z][A-Z0-9.]{0,9}$"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True, use_enum_values=False)

    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------------------


class Bar(Contract):
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    timeframe: str = Field(description="e.g. '1Min', '1Day'")
    start_ts: UtcDatetime = Field(description="bar open time (exchange clock)")
    end_ts: UtcDatetime = Field(description="bar close time; a bar is not knowable before end_ts")
    open: Dec
    high: Dec
    low: Dec
    close: Dec
    volume: Dec
    vwap: Dec | None = None
    trade_count: int | None = None
    feed: DataFeed
    adjustment: Adjustment = Adjustment.RAW
    session: MarketSession
    provider_ts: UtcDatetime | None = None
    received_ts: UtcDatetime
    available_at_ts: UtcDatetime = Field(description="earliest time a decision may use this bar")
    source_id: str
    dataset_version: str
    is_synthetic: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.low <= min(self.open, self.close) and self.high >= max(self.open, self.close)):
            raise ValueError("OHLC inconsistent: low/high do not bound open/close")
        if self.low <= 0:
            raise ValueError("non-positive price")
        if self.volume < 0:
            raise ValueError("negative volume")
        if self.end_ts <= self.start_ts:
            raise ValueError("end_ts must be after start_ts")
        if self.available_at_ts < self.end_ts:
            raise ValueError("available_at_ts precedes bar close (look-ahead)")
        if self.is_synthetic != (self.feed == DataFeed.SYNTHETIC):
            raise ValueError("is_synthetic must be true exactly when feed is SYNTHETIC")
        return self


class Quote(Contract):
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    bid: Dec
    ask: Dec
    bid_size: Dec
    ask_size: Dec
    bid_exchange: str | None = None
    ask_exchange: str | None = None
    exchange_ts: UtcDatetime
    provider_ts: UtcDatetime | None = None
    received_ts: UtcDatetime
    available_at_ts: UtcDatetime
    feed: DataFeed
    conditions: tuple[str, ...] = ()
    tape: str | None = None
    source_id: str
    is_synthetic: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("non-positive bid/ask")
        if self.ask < self.bid:
            raise ValueError("crossed quote (ask < bid)")
        if self.available_at_ts < self.exchange_ts:
            raise ValueError("available_at_ts precedes exchange_ts")
        if self.is_synthetic != (self.feed == DataFeed.SYNTHETIC):
            raise ValueError("is_synthetic must be true exactly when feed is SYNTHETIC")
        return self

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal:
        return (self.ask - self.bid) / self.mid * Decimal(10_000)


class MarketSnapshot(Contract):
    snapshot_id: str
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    feed: DataFeed
    feed_mode: FeedMode
    session: MarketSession
    quote: Quote | None
    last_trade_price: Dec | None = None
    last_bar: Bar | None = None
    data_asof: UtcDatetime = Field(description="decision clock time this snapshot represents")
    source_id: str
    dataset_version: str
    is_simulated: bool
    limitations: tuple[str, ...] = Field(
        default=(), description="human-readable feed caveats, e.g. 'IEX-only quotes'"
    )


# --------------------------------------------------------------------------------------------
# Research / signals
# --------------------------------------------------------------------------------------------


class Evidence(Contract):
    evidence_id: str
    source_type: EvidenceSourceType
    source_id: str
    trust: TrustLevel
    title: str
    uri: str | None = None
    excerpt: str = Field(max_length=8000, description="sanitized excerpt shown to models")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_at: UtcDatetime | None = None
    retrieved_at: UtcDatetime
    available_at: UtcDatetime = Field(description="earliest decision time allowed to use this evidence")
    injection_flags: tuple[str, ...] = ()
    is_simulated: bool = False


class QuantSignal(Contract):
    signal_id: str
    strategy_id: str
    model_version: str
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    side: Side = Side.BUY
    horizon_minutes: int = Field(gt=0)
    score: Dec
    rank: int = Field(ge=1)
    expected_return_bps: Dec
    probability_up: Dec | None = None
    probability_calibrated: bool = False
    est_spread_bps: Dec
    est_fees_bps: Dec
    est_slippage_bps: Dec
    net_edge_bps: Dec = Field(description="expected_return_bps minus total estimated costs")
    is_candidate: bool
    reason: str
    features_asof: UtcDatetime
    generated_at: UtcDatetime
    dataset_version: str
    is_simulated: bool

    @property
    def est_total_cost_bps(self) -> Decimal:
        return self.est_spread_bps + self.est_fees_bps + self.est_slippage_bps


class HorizonForecast(Contract):
    horizon_minutes: int = Field(gt=0)
    expected_return_bps: Dec
    model_id: str = Field(description="e.g. lgbm_reg, ridge")
    model_version: str


class PredictionSet(Contract):
    """Outputs of the local quant model suite for one symbol at one decision time (point-in-time)."""

    prediction_id: str
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    asof: UtcDatetime
    features_asof: UtcDatetime
    dataset_version: str
    registry_version: str = Field(description="model registry snapshot the predictions came from")
    return_forecasts: tuple[HorizonForecast, ...]
    success_probability: Dec | None = None
    success_horizon_minutes: int | None = None
    probability_calibrated: bool = False
    calibration_method: str | None = Field(default=None, description="isotonic | sigmoid | None")
    rank: int | None = None
    rank_score: Dec | None = None
    universe_size: int = 0
    vol_forecast_bps: Dec | None = Field(default=None, description="forecast return volatility over vol_horizon")
    vol_horizon_minutes: int | None = None
    vol_model: str | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_probability: Dec | None = None
    est_cost_bps: Dec | None = Field(default=None, description="predicted spread+fees+slippage, one way")
    est_slippage_bps: Dec | None = None
    cost_model_id: str | None = None
    is_simulated: bool
    notes: tuple[str, ...] = ()


class ResearchFinding(Contract):
    finding_id: str
    causal_id: str
    model_run_id: str
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    recommendation: DecisionAction
    stance: Stance
    thesis: str = Field(max_length=4000)
    evidence_ids: tuple[str, ...]
    counterarguments: tuple[str, ...]
    risks: tuple[str, ...] = ()
    confidence_label: ConfidenceLabel = Field(description="uncalibrated qualitative label")
    no_trade_reason: str | None = None
    data_asof: UtcDatetime
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.recommendation == DecisionAction.NO_TRADE and not self.no_trade_reason:
            raise ValueError("NO_TRADE finding requires no_trade_reason")
        if self.recommendation == DecisionAction.TRADE and not self.evidence_ids:
            raise ValueError("TRADE finding requires at least one evidence id")
        return self


# --------------------------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------------------------


class TradeProposal(Contract):
    proposal_id: str
    causal_id: str
    created_at: UtcDatetime
    data_asof: UtcDatetime
    action: DecisionAction
    no_trade_reason: str | None = None
    proposer: Proposer
    strategy_id: str
    symbol: str | None = Field(default=None, pattern=_SYMBOL_PATTERN)
    side: Side | None = None
    quantity: Dec | None = None
    order_type: OrderType = OrderType.LIMIT
    limit_price: Dec | None = None
    stop_price: Dec | None = Field(default=None, description="protective stop; not a guaranteed price")
    time_in_force: TimeInForce = TimeInForce.DAY
    session_constraint: SessionConstraint = SessionConstraint.REGULAR_ONLY
    max_planned_loss: Dec | None = None
    evidence_ids: tuple[str, ...] = ()
    signal_id: str | None = None
    finding_ids: tuple[str, ...] = ()
    model_run_ids: tuple[str, ...] = ()
    policy_version_seen: str | None = Field(default=None, description="policy the proposer read (info only)")
    dataset_version: str
    rationale: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.action == DecisionAction.NO_TRADE:
            if not self.no_trade_reason:
                raise ValueError("NO_TRADE requires no_trade_reason")
            return self
        missing = [
            name
            for name in ("symbol", "side", "quantity", "limit_price", "max_planned_loss")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"TRADE proposal missing required fields: {missing}")
        if not self.evidence_ids:
            raise ValueError("TRADE proposal requires evidence_ids")
        assert self.quantity is not None and self.limit_price is not None
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be positive")
        if self.order_type != OrderType.LIMIT:
            raise ValueError("only LIMIT orders are supported in the initial build")
        return self


class RiskCheck(Contract):
    name: str
    passed: bool
    detail: str
    observed: str | None = None
    limit: str | None = None


class RiskDecision(Contract):
    decision_id: str
    proposal_id: str
    causal_id: str
    verdict: RiskVerdict
    checks: tuple[RiskCheck, ...]
    reasons: tuple[str, ...]
    policy_version: str
    mode: TradingMode
    recomputed_quantity: Dec | None = None
    recomputed_notional: Dec | None = None
    recomputed_max_loss: Dec | None = None
    reference_price: Dec | None = None
    account_snapshot_id: str | None = None
    market_snapshot_id: str | None = None
    evaluated_at: UtcDatetime

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED


class ApprovedOrderIntent(Contract):
    """The only object an execution gateway accepts. Signed (ECDSA P-256 over signing_string()) by the risk engine."""

    intent_id: str
    client_order_id: str = Field(max_length=128)
    proposal_id: str
    decision_id: str
    causal_id: str
    mode: TradingMode
    gateway: GatewayKind
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    side: Side
    quantity: Dec
    order_type: OrderType
    limit_price: Dec
    time_in_force: TimeInForce
    extended_hours: bool = False
    policy_version: str
    created_at: UtcDatetime
    expires_at: UtcDatetime
    requires_human_approval: bool
    human_approval_id: str | None = None
    key_id: str
    signature_alg: str = "ECDSA_P256_SHA256"
    signature: str = Field(pattern=r"^[0-9a-f]{128}$", description="hex(r||s) over signing_string()")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.extended_hours:
            raise ValueError("extended-hours orders are not permitted")
        if self.quantity <= 0 or self.limit_price <= 0:
            raise ValueError("quantity and limit_price must be positive")
        if self.expires_at <= self.created_at:
            raise ValueError("intent expires before it is created")
        return self

    def signing_string(self) -> str:
        """Exact UTF-8 string the risk engine signs and LEAN verifies (then parses order fields from).

        Pipe-delimited so the C# verifier does not need to reproduce canonical JSON. LEAN executes
        the parameters parsed from *this* string, so what is executed is exactly what was signed.
        """
        return intent_signing_string(self.model_dump(mode="json", exclude={"signature"}))


INTENT_SIGNING_FIELDS: tuple[str, ...] = (
    "intent_id", "client_order_id", "proposal_id", "decision_id", "causal_id", "mode", "gateway",
    "symbol", "side", "quantity", "order_type", "limit_price", "time_in_force", "extended_hours",
    "policy_version", "created_at", "expires_at", "requires_human_approval", "human_approval_id",
    "key_id", "signature_alg",
)


def intent_signing_string(fields: dict[str, Any]) -> str:
    parts = ["lat-intent-v1"]
    for name in INTENT_SIGNING_FIELDS:
        value = fields.get(name)
        if value is None:
            text = ""
        elif isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        if "|" in text or "\n" in text:
            raise ValueError(f"illegal character in signed field {name}")
        parts.append(f"{name}={text}")
    return "|".join(parts)


class HumanApproval(Contract):
    approval_id: str
    intent_id: str
    approver: str
    confirmed_at: UtcDatetime
    expires_at: UtcDatetime
    statement: str = Field(description="exact text the human confirmed")
    signature: str = Field(pattern=r"^[0-9a-f]{64,128}$")


# --------------------------------------------------------------------------------------------
# Broker state
# --------------------------------------------------------------------------------------------


class BrokerOrder(Contract):
    client_order_id: str
    broker_order_id: str | None
    intent_id: str | None
    gateway: GatewayKind
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    side: Side
    quantity: Dec
    filled_quantity: Dec = Decimal(0)
    avg_fill_price: Dec | None = None
    order_type: OrderType
    limit_price: Dec | None
    time_in_force: TimeInForce
    broker_status: str = Field(description="raw broker status string")
    state: OrderState
    submitted_at: UtcDatetime | None = None
    updated_at: UtcDatetime


class OrderEvent(Contract):
    event_id: str
    client_order_id: str
    broker_order_id: str | None = None
    event_type: OrderEventType
    state: OrderState = Field(description="state implied by this event")
    source: EventSource
    gateway: GatewayKind
    fill_quantity: Dec | None = None
    fill_price: Dec | None = None
    cum_filled_quantity: Dec | None = None
    broker_exec_id: str | None = Field(default=None, description="unique execution id for fill dedupe")
    broker_ts: UtcDatetime | None = None
    received_ts: UtcDatetime
    sequence: int | None = None
    reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PositionSnapshot(Contract):
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    quantity: Dec
    avg_entry_price: Dec
    cost_basis: Dec
    market_value: Dec | None = None
    current_price: Dec | None = None
    unrealized_pl: Dec | None = None
    asof: UtcDatetime
    source: str = Field(description="BROKER | LEDGER | FAKE_BROKER")


class AccountSnapshot(Contract):
    snapshot_id: str
    account_id_masked: str
    is_paper: bool
    status: str
    currency: str = "USD"
    equity: Dec
    last_equity: Dec | None = None
    cash: Dec
    buying_power: Dec
    non_marginable_buying_power: Dec | None = None
    multiplier: Dec | None = None
    trading_blocked: bool
    account_blocked: bool
    transfers_blocked: bool = False
    shorting_enabled: bool = False
    asof: UtcDatetime
    source: str


class ModelRunRecord(Contract):
    run_id: str
    causal_id: str
    provider: ModelProvider
    model_id: str
    adapter_version: str
    prompt_version: str
    effort: str
    transport: ModelTransport
    status: ModelRunStatus
    started_at: UtcDatetime
    finished_at: UtcDatetime
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int | None = None
    estimated_cost_usd: Dec
    pricing_source: str = Field(description="where the per-token price came from, or 'UNVERIFIED'")
    request_sha256: str
    response_sha256: str | None = None
    output_schema: str
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None


# --------------------------------------------------------------------------------------------
# Risk inputs, health and reconciliation
# --------------------------------------------------------------------------------------------


class SystemHealth(Contract):
    feed_ok: bool
    feed_detail: str = ""
    broker_connected: bool
    broker_detail: str = ""
    reconciled: bool
    reconcile_detail: str = ""
    engine_ok: bool
    engine_detail: str = ""
    entries_locked: bool
    lock_reasons: tuple[str, ...] = ()
    strategy_state_ok: bool = True
    permissions_ok: bool = True


class RiskContext(Contract):
    """Everything the risk engine may read. Built by the pipeline from broker + ledger state."""

    now: UtcDatetime
    mode: TradingMode
    gateway: GatewayKind
    account: AccountSnapshot | None
    positions: tuple[PositionSnapshot, ...]
    open_orders: tuple[BrokerOrder, ...]
    recent_submission_times: tuple[UtcDatetime, ...] = Field(
        default=(), description="submission timestamps (all orders) for rate limiting"
    )
    recent_intent_fingerprints: tuple[str, ...] = Field(
        default=(), description="fingerprints of recently authorized intents for duplicate rejection"
    )
    start_of_day_equity: Dec | None = None
    daily_realized_pl: Dec = Decimal(0)
    daily_unrealized_pl: Dec = Decimal(0)
    market: MarketSnapshot | None
    health: SystemHealth


class ReconciliationMismatch(Contract):
    kind: str = Field(description="e.g. UNEXPECTED_POSITION, QTY_MISMATCH, UNKNOWN_ORDER, MISSING_ORDER")
    symbol: str | None = None
    client_order_id: str | None = None
    local: str | None = None
    broker: str | None = None
    detail: str


class ReconciliationReport(Contract):
    run_id: str
    started_at: UtcDatetime
    finished_at: UtcDatetime
    status: ReconciliationStatus
    gateway: GatewayKind
    broker_source: str
    mismatches: tuple[ReconciliationMismatch, ...]
    resolved_unknown_orders: tuple[str, ...] = ()
    entries_locked_after: bool
    detail: str = ""


class ExecutionStatus(Contract):
    mode: TradingMode
    label: ExecutionLabel
    gateway: GatewayKind
    feed_mode: FeedMode
    live_execution_compiled: bool
    detail: str = ""
