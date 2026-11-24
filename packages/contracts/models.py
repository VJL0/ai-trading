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
    CRYPTO_TIME_IN_FORCE,
    Adjustment,
    AssetClass,
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

EQUITY_SYMBOL_PATTERN = r"^[A-Z][A-Z0-9.]{0,9}$"
CRYPTO_SYMBOL_PATTERN = r"^[A-Z0-9]{2,10}/(USD|USDT|USDC|BTC)$"
_SYMBOL_PATTERN = r"^(?:[A-Z][A-Z0-9.]{0,9}|[A-Z0-9]{2,10}/(?:USD|USDT|USDC|BTC))$"
"""U.S. equity tickers, or Alpaca crypto pairs ``BASE/QUOTE``. The asset class is derived from this shape."""


def asset_class_of(symbol: str) -> AssetClass:
    """Crypto pairs contain '/', U.S. equity tickers never do (the risk engine re-checks the exact pattern)."""
    return AssetClass.CRYPTO if "/" in symbol else AssetClass.US_EQUITY


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
    net_edge_bps: Dec = Field(description="expected_return_bps (GROSS) minus the live round-trip cost estimate, charged "
                                          "once, minus uncertainty_bps: the quantity the decision threshold applies to")
    uncertainty_bps: Dec = Field(default=Decimal(0), description="explicit forecast-uncertainty buffer (bps)")
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
    holding_sessions: int | None = Field(
        default=None, ge=1, le=20,
        description="multi-day holding request for a BUY entry (sessions incl. the entry session); None/1 = intraday. "
                    "Honoured only when the risk policy allows overnight positions.")
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
        assert self.quantity is not None and self.limit_price is not None and self.symbol is not None
        crypto = asset_class_of(self.symbol) == AssetClass.CRYPTO
        if crypto and self.time_in_force not in CRYPTO_TIME_IN_FORCE:
            raise ValueError("crypto orders require time_in_force GTC or IOC")
        if not crypto and self.time_in_force != TimeInForce.DAY:
            raise ValueError("U.S. equity orders require time_in_force DAY")
        if crypto != (self.session_constraint == SessionConstraint.CONTINUOUS):
            raise ValueError("crypto orders require session_constraint CONTINUOUS; equities REGULAR_ONLY")
        if crypto and self.order_type == OrderType.LIMIT and self.time_in_force != TimeInForce.IOC:
            raise ValueError("crypto LIMIT orders must be IOC (nothing rests unattended; stops are the exception)")
        if crypto and self.order_type == OrderType.STOP_LIMIT and self.time_in_force != TimeInForce.GTC:
            raise ValueError("crypto protective STOP_LIMIT orders must be GTC (they persist 24/7)")
        if self.holding_sessions is not None and self.side != Side.BUY:
            raise ValueError("holding_sessions applies to BUY entries only")
        if crypto and self.holding_sessions is not None:
            raise ValueError("holding_sessions applies to U.S. equities only (crypto trades continuously)")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be positive")
        if self.order_type == OrderType.STOP_LIMIT:
            # Protective exit only: a long-only SELL resting below the market that becomes a LIMIT order once the
            # stop trades. Not guaranteed protection: a gap through the limit leaves it unfilled.
            if self.side != Side.SELL:
                raise ValueError("STOP_LIMIT is allowed only for protective SELL exits")
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("STOP_LIMIT requires a positive stop_price")
            if self.limit_price > self.stop_price:
                raise ValueError("STOP_LIMIT SELL limit_price must be at or below stop_price")
        elif self.order_type != OrderType.LIMIT:
            raise ValueError("only LIMIT orders (and STOP_LIMIT protective exits) are supported")
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
    stop_price: Dec | None = Field(
        default=None, description="STOP_LIMIT protective exits only; signed (lat-intent-v2). Not a guaranteed price."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.extended_hours:
            raise ValueError("extended-hours orders are not permitted")
        if self.quantity <= 0 or self.limit_price <= 0:
            raise ValueError("quantity and limit_price must be positive")
        crypto = asset_class_of(self.symbol) == AssetClass.CRYPTO
        if crypto != (self.time_in_force in CRYPTO_TIME_IN_FORCE):
            raise ValueError("crypto intents require GTC/IOC and equity intents require DAY")
        if crypto and self.order_type == OrderType.LIMIT and self.time_in_force != TimeInForce.IOC:
            raise ValueError("crypto LIMIT intents must be IOC")
        if crypto and self.order_type == OrderType.STOP_LIMIT and self.time_in_force != TimeInForce.GTC:
            raise ValueError("crypto STOP_LIMIT intents must be GTC")
        if self.expires_at <= self.created_at:
            raise ValueError("intent expires before it is created")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.side != Side.SELL or self.stop_price is None or self.stop_price <= 0:
                raise ValueError("STOP_LIMIT intents must be SELL exits with a positive stop_price")
            if self.limit_price > self.stop_price:
                raise ValueError("STOP_LIMIT SELL limit_price must be at or below stop_price")
        elif self.stop_price is not None:
            raise ValueError("stop_price is only valid on STOP_LIMIT intents")
        return self

    @property
    def is_protective(self) -> bool:
        return self.order_type == OrderType.STOP_LIMIT

    @property
    def asset_class(self) -> AssetClass:
        return asset_class_of(self.symbol)

    def signing_string(self) -> str:
        """Exact UTF-8 string the risk engine signs and LEAN verifies (then parses order fields from).

        Pipe-delimited so the C# verifier does not need to reproduce canonical JSON. LEAN executes
        the parameters parsed from *this* string, so what is executed is exactly what was signed.
        LIMIT intents keep the original ``lat-intent-v1`` string byte for byte; STOP_LIMIT protective exits use
        ``lat-intent-v2`` (v1 fields plus ``stop_price``), which the LEAN v1 verifier refuses (fail closed).
        Crypto intents (symbol ``BASE/QUOTE``) use ``lat-intent-v3`` (v2 fields plus ``asset_class``); LEAN never
        executes crypto and refuses v3, and the Python gateway accepts it only for crypto symbols.
        """
        return intent_signing_string(self.model_dump(mode="json", exclude={"signature"}))


INTENT_SIGNING_FIELDS: tuple[str, ...] = (
    "intent_id", "client_order_id", "proposal_id", "decision_id", "causal_id", "mode", "gateway",
    "symbol", "side", "quantity", "order_type", "limit_price", "time_in_force", "extended_hours",
    "policy_version", "created_at", "expires_at", "requires_human_approval", "human_approval_id",
    "key_id", "signature_alg",
)


INTENT_SIGNING_FIELDS_V2: tuple[str, ...] = (*INTENT_SIGNING_FIELDS, "stop_price")
INTENT_SIGNING_FIELDS_V3: tuple[str, ...] = (*INTENT_SIGNING_FIELDS_V2, "asset_class")


def intent_signing_string(fields: dict[str, Any]) -> str:
    symbol = str(fields.get("symbol") or "")
    if asset_class_of(symbol) == AssetClass.CRYPTO:
        fields = {**fields, "asset_class": AssetClass.CRYPTO.value}
        version, names = "lat-intent-v3", INTENT_SIGNING_FIELDS_V3
    elif fields.get("stop_price") is not None or fields.get("order_type") == OrderType.STOP_LIMIT.value:
        version, names = "lat-intent-v2", INTENT_SIGNING_FIELDS_V2
    else:
        version, names = "lat-intent-v1", INTENT_SIGNING_FIELDS
    parts = [version]
    for name in names:
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
    stop_price: Dec | None = None


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
    unrealized_intraday_pl: Dec | None = Field(
        default=None, description="P&L since the previous close (broker field); daily-loss input for multi-day holds")
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
    crypto_status: str | None = Field(default=None, description="Alpaca crypto_status; crypto entries need ACTIVE")
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
    exits_locked: bool | None = Field(
        default=None,
        description="risk-reducing SELL exits locked? None => same as entries (fail closed). Halts and unprotected-"
                    "position safety states lock entries but not exits; state uncertainty locks both.",
    )
    exit_lock_reasons: tuple[str, ...] = ()

    @property
    def exits_blocked(self) -> bool:
        if self.exits_locked is None:
            return self.entries_locked or bool(self.lock_reasons)
        return self.exits_locked or bool(self.exit_lock_reasons)

    @property
    def exit_block_reasons(self) -> tuple[str, ...]:
        if self.exits_locked is None:
            return self.lock_reasons
        return self.exit_lock_reasons


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
