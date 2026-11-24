"""Closed vocabularies used across services. Values are stable wire strings."""

from __future__ import annotations

from enum import StrEnum


class TradingMode(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    REPLAY = "REPLAY"
    PAPER = "PAPER"
    LIVE_APPROVAL = "LIVE_APPROVAL"
    BOUNDED_LIVE = "BOUNDED_LIVE"


class ExecutionLabel(StrEnum):
    """What the operator must be told about where orders/fills come from."""

    SIMULATED = "SIMULATED"  # deterministic fake broker / replay / LEAN PaperBrokerage on fake data
    PAPER = "PAPER"  # Alpaca paper endpoint (still simulated fills, but broker-side)
    LIVE = "LIVE"  # real money (never enabled in this build)


class FeedMode(StrEnum):
    IEX_FREE = "IEX_FREE"  # Alpaca Basic: real-time equities from IEX only
    SIP_PAID = "SIP_PAID"  # Algo Trader Plus: consolidated SIP (paid, never auto-purchased)
    SIMULATED = "SIMULATED"  # explicitly synthetic fixture / replay


class DataFeed(StrEnum):
    IEX = "IEX"
    SIP = "SIP"
    SYNTHETIC = "SYNTHETIC"
    CRYPTO_US = "CRYPTO_US"  # Alpaca crypto feed (location 'us'), 24/7
    LEAN_SAMPLE = "LEAN_SAMPLE"  # real historical sample data bundled in the pinned LEAN repo


class Adjustment(StrEnum):
    RAW = "raw"
    SPLIT = "split"
    DIVIDEND = "dividend"
    ALL = "all"


class MarketSession(StrEnum):
    PRE = "PRE"
    REGULAR = "REGULAR"
    POST = "POST"
    CLOSED = "CLOSED"
    CONTINUOUS = "CONTINUOUS"  # 24/7 crypto market (no XNYS session)


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"  # long-only: SELL may only reduce/close an existing long


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LIMIT = "STOP_LIMIT"  # protective exits only: a SELL that becomes a LIMIT once the stop trades


PROTECTIVE_ORDER_TYPES: frozenset[OrderType] = frozenset({OrderType.STOP_LIMIT})
"""Order types allowed only for risk-reducing (long-only SELL) protective exits, never for entries."""


class TimeInForce(StrEnum):
    DAY = "DAY"  # U.S. equities (regular session)
    GTC = "GTC"  # crypto only (Alpaca crypto accepts gtc / ioc)
    IOC = "IOC"  # crypto only


class AssetClass(StrEnum):
    """Derived from the symbol (crypto pairs contain '/'); never a free-standing field an agent can set."""

    US_EQUITY = "US_EQUITY"
    CRYPTO = "CRYPTO"


CRYPTO_TIME_IN_FORCE: frozenset[TimeInForce] = frozenset({TimeInForce.GTC, TimeInForce.IOC})


class SessionConstraint(StrEnum):
    REGULAR_ONLY = "REGULAR_ONLY"  # U.S. equities: XNYS regular session only
    CONTINUOUS = "CONTINUOUS"  # crypto only: 24/7


class DecisionAction(StrEnum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


class Proposer(StrEnum):
    QUANT_BASELINE = "QUANT_BASELINE"
    QUANT_MODEL = "QUANT_MODEL"  # walk-forward-selected local model suite
    AI_PIPELINE = "AI_PIPELINE"  # in-app model adapters (API keys or Sign in with ChatGPT)
    AGENT_TOOL = "AGENT_TOOL"  # external agent (Claude Code / Codex) via the validated tool interface
    HUMAN = "HUMAN"
    TEST = "TEST"
    POSITION_MANAGER = "POSITION_MANAGER"  # deterministic exits/protection for positions the system already holds


class AiRuntimeMode(StrEnum):
    """How (and whether) an AI participates. The quant engine never depends on any of these."""

    NONE = "NONE"  # quant only
    API_KEYS = "API_KEYS"  # in-app Claude Opus 5.5 + GPT-6.1 Sol adapters (separate API billing)
    CLAUDE_CODE = "CLAUDE_CODE"  # user's Claude Code session operates the app via the local MCP server
    CODEX = "CODEX"  # user's Codex session operates the app via the local MCP server
    CHATGPT_OAUTH = "CHATGPT_OAUTH"  # official Sign in with ChatGPT plan usage (Responses API, function tools)


class MarketRegime(StrEnum):
    CALM = "CALM"
    TURBULENT = "TURBULENT"
    UNKNOWN = "UNKNOWN"


class RiskVerdict(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class OrderState(StrEnum):
    PROPOSED = "PROPOSED"
    RISK_APPROVED = "RISK_APPROVED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    RECONCILED = "RECONCILED"


TERMINAL_ORDER_STATES: frozenset[OrderState] = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED, OrderState.RECONCILED}
)


class OrderEventType(StrEnum):
    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    PARTIAL_FILL = "partial_fill"
    FILL = "fill"
    PENDING_CANCEL = "pending_cancel"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"
    DONE_FOR_DAY = "done_for_day"
    SUBMIT_TIMEOUT = "submit_timeout"
    UNKNOWN = "unknown"
    RECONCILED = "reconciled"


class EventSource(StrEnum):
    GATEWAY = "GATEWAY"  # synchronous submit/cancel response
    STREAM = "STREAM"  # broker trade-updates stream / LEAN OnOrderEvent
    POLL = "POLL"  # REST polling
    RECONCILE = "RECONCILE"  # reconciliation service
    LOCAL = "LOCAL"  # local state machine (e.g. timeout -> UNKNOWN)


class GatewayKind(StrEnum):
    ALPACA_PAPER = "ALPACA_PAPER"  # official alpaca-py SDK, paper endpoint only
    FAKE = "FAKE"  # deterministic in-process fake broker (SIMULATED)
    LEAN_PAPER_SIM = "LEAN_PAPER_SIM"  # LEAN PaperBrokerage on simulated data (SIMULATED)
    LEAN_ALPACA_PAPER = "LEAN_ALPACA_PAPER"  # LEAN + Alpaca brokerage plugin, paper endpoint (PAPER)
    LEAN_ALPACA_LIVE = "LEAN_ALPACA_LIVE"  # interface only; activation always denied in this build


class EvidenceSourceType(StrEnum):
    QUANT_SIGNAL = "QUANT_SIGNAL"
    MARKET_DATA = "MARKET_DATA"
    NEWS = "NEWS"
    FILING = "FILING"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    RISK_POLICY = "RISK_POLICY"
    MODEL_NOTE = "MODEL_NOTE"


class TrustLevel(StrEnum):
    TRUSTED_INTERNAL = "TRUSTED_INTERNAL"
    UNTRUSTED_EXTERNAL = "UNTRUSTED_EXTERNAL"


class Stance(StrEnum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    NEUTRAL = "NEUTRAL"


class ConfidenceLabel(StrEnum):
    """Qualitative, explicitly *uncalibrated* model confidence. Never a trading probability."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ModelProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ModelTransport(StrEnum):
    REAL = "REAL"
    FAKE = "FAKE"  # explicit fake HTTP transport: no network, deterministic canned provider responses


class ModelRunStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    INJECTION_BLOCKED = "INJECTION_BLOCKED"
    SKIPPED = "SKIPPED"


class ReconciliationStatus(StrEnum):
    CLEAN = "CLEAN"
    MISMATCH = "MISMATCH"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
