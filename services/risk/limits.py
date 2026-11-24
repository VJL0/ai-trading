"""Hard ceilings and structural invariants for risk policies (spec §6, §8, §12).

These are *module constants*, not configuration. The operator settings flow can tighten or loosen a
policy only inside these bounds, and can never enable margin, shorting, extended hours, non-LIMIT
orders or leveraged/inverse products. Changing a ceiling requires a code change and review.

``validate_policy`` is used twice: by ``PolicyStore.apply_change`` (reject unsafe edits before they
are stored) and by ``RiskEngine`` at evaluation time (an unsafe policy blocks every trade).
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final

from packages.contracts.enums import OrderType
from packages.contracts.policy import DEFAULT_DENYLIST, POLICY_SCHEMA, RiskPolicy

# ---------------------------------------------------------------------------- hard ceilings
HARD_MAX_ORDER_NOTIONAL: Final = Decimal("25000")
HARD_MAX_POSITION_NOTIONAL: Final = Decimal("50000")
HARD_MAX_CONCENTRATION_PCT: Final = Decimal("0.50")
HARD_MAX_DAILY_LOSS: Final = Decimal("2500")
HARD_MAX_PLANNED_LOSS_PER_TRADE: Final = Decimal("500")
HARD_MAX_SPREAD_BPS: Final = Decimal("100")
HARD_MAX_QUOTE_AGE_SECONDS: Final = Decimal("60")
HARD_MAX_LIMIT_DEVIATION_BPS: Final = Decimal("100")
HARD_MIN_PRICE_FLOOR: Final = Decimal("5.00")  # SEC penny-stock threshold; penny/OTC names excluded
HARD_MIN_ACCOUNT_EQUITY_FLOOR: Final = Decimal("500")
HARD_MIN_BUYING_POWER_BUFFER: Final = Decimal("1.00")
HARD_MAX_BUYING_POWER_BUFFER: Final = Decimal("2.00")
HARD_MIN_DEFAULT_STOP_BPS: Final = Decimal("10")
HARD_MAX_DEFAULT_STOP_BPS: Final = Decimal("2000")
HARD_MAX_POSITIONS: Final = 10
HARD_MAX_PENDING_ORDERS: Final = 10
HARD_MAX_ORDERS_PER_MINUTE: Final = 10
HARD_MAX_ORDERS_PER_DAY: Final = 200
HARD_MAX_INTENT_TTL_SECONDS: Final = 300
HARD_MIN_DUPLICATE_WINDOW_SECONDS: Final = 60
HARD_MAX_DUPLICATE_WINDOW_SECONDS: Final = 86_400
HARD_MAX_SYMBOLS: Final = 500
# ---- multi-day equity holding (opt-in, operator policy change only)
HARD_MAX_HOLDING_SESSIONS: Final = 20
HARD_MIN_OVERNIGHT_GAP_BUDGET_BPS: Final = Decimal("300")
"""A multi-day entry must survive an assumed overnight gap of at least 3% within the daily-loss limit."""
# ---- PAPER crypto spot (opt-in, operator policy change only)
HARD_MAX_CRYPTO_ORDER_NOTIONAL: Final = Decimal("5000")
HARD_MAX_CRYPTO_POSITION_NOTIONAL: Final = Decimal("10000")
HARD_MAX_CRYPTO_POSITIONS: Final = 3
HARD_MAX_CRYPTO_SPREAD_BPS: Final = Decimal("100")
HARD_MAX_CRYPTO_SYMBOLS: Final = 10
HARD_MIN_CRYPTO_ORDER_NOTIONAL: Final = Decimal("1")
HARD_MAX_CRYPTO_STOP_BPS: Final = Decimal("2000")
CRYPTO_SYMBOL: Final = re.compile(r"^[A-Z0-9]{2,10}/USD$")
"""Policy-allowlisted crypto pairs must be quoted in USD (no stablecoin or BTC quotes in this build)."""
STABLECOIN_BASES: Final[frozenset[str]] = frozenset({"USDC", "USDT", "USDG", "DAI", "PYUSD", "TUSD", "USDP"})
HARD_MIN_CRYPTO_FEE_BPS: Final = Decimal("10")
"""Crypto fees may not be configured below 10 bps (understated fees would understate planned losses)."""
HARD_MAX_CRYPTO_FEE_BPS: Final = Decimal("100")
STRATEGY_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")

# ---------------------------------------------------------------------------- engine constants
# Not policy fields (yet): see requested contract changes in services/risk/README.md.
MAX_CLOCK_SKEW_SECONDS: Final = Decimal("2")
"""Tolerated future skew for timestamps produced by other local components."""
MAX_CONTEXT_AGE_SECONDS: Final = Decimal("10")
"""RiskContext.now must be within this many seconds of the engine clock (stale context => reject)."""
MAX_ACCOUNT_SNAPSHOT_AGE_SECONDS: Final = Decimal("60")
"""Account snapshot older than this is unknown state => reject (fail closed)."""
MAX_PROPOSAL_DATA_AGE_SECONDS: Final = Decimal("300")
"""Proposal data_asof older than this is stale research (the live quote is re-checked separately)."""
MIN_STOP_DISTANCE_BPS: Final = Decimal("10")
"""A proposal stop closer than this to the limit is treated as invalid (the default stop is used)."""
PLANNED_LOSS_TOLERANCE: Final = Decimal("0.05")
"""A proposal may not *understate* the recomputed planned loss by more than this fraction."""
PLANNED_LOSS_ABS_TOLERANCE: Final = Decimal("0.05")
"""... nor by more than this many dollars (both tolerances must be exceeded to reject)."""
MAX_THROUGH_TOUCH_BPS: Final = Decimal("10")
"""A BUY limit may exceed the ask (a SELL limit may undercut the bid) by at most this many bps."""

ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS: Final = 1800
"""No new BUY entries in the last 30 minutes of the XNYS session (early closes included), so every position
can be exited during regular hours. Exits are never cut off before the close."""
MAX_STOP_LIMIT_GAP_BPS: Final = Decimal("300")
"""A protective STOP_LIMIT SELL may set its limit at most this far below its stop."""
MAX_PROTECTIVE_STOP_DISTANCE_BPS: Final = Decimal("2000")
"""A protective stop more than this far below the mid is not protection (equals the default-stop ceiling)."""
MAX_OPEN_SELLS_PER_SYMBOL: Final = 2
"""Structural cap on concurrently working risk-reducing SELL orders per symbol (protective + top-up). Their total
remaining quantity can never exceed the long position (long_only). SELLs do not use ``max_pending_orders``."""

US_EQUITY_SYMBOL: Final = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
"""Listed U.S. common stock / ETF symbols (class shares like BRK.B allowed). No digits/OTC suffixes."""

HARD_DENYLIST: Final[frozenset[str]] = frozenset(DEFAULT_DENYLIST)
"""Leveraged/inverse/volatility products that no policy edit can remove from the denylist."""


def _positive(name: str, value: Decimal, out: list[str]) -> None:
    if value <= 0:
        out.append(f"{name} must be positive (got {value})")


def _ceiling(name: str, value: Decimal | int, ceiling: Decimal | int, out: list[str]) -> None:
    if value > ceiling:
        out.append(f"{name}={value} exceeds hard ceiling {ceiling}")


def validate_policy(policy: RiskPolicy) -> list[str]:
    """Return every violation of the structural invariants and hard ceilings (empty == valid)."""
    v: list[str] = []
    if policy.policy_schema != POLICY_SCHEMA:
        v.append(f"unsupported policy_schema {policy.policy_schema!r}")
    if not policy.name.strip():
        v.append("policy name must not be empty")
    # Structural invariants — never configurable in this build.
    if not policy.long_only:
        v.append("long_only cannot be disabled (no shorting in this build)")
    if policy.allow_margin:
        v.append("allow_margin cannot be enabled (no margin borrowing)")
    if policy.allow_short:
        v.append("allow_short cannot be enabled (long-only)")
    if policy.allow_extended_hours:
        v.append("allow_extended_hours cannot be enabled (regular session only)")
    if not policy.regular_session_only:
        v.append("regular_session_only cannot be disabled")
    if tuple(policy.allowed_order_types) != (OrderType.LIMIT,):
        v.append(f"allowed_order_types must be exactly (LIMIT,) (got {list(policy.allowed_order_types)})")
    # Symbols.
    allow = [s for s in policy.symbol_allowlist]
    deny = set(policy.symbol_denylist)
    if not allow:
        v.append("symbol_allowlist must not be empty")
    if len(allow) > HARD_MAX_SYMBOLS:
        v.append(f"symbol_allowlist has {len(allow)} symbols (> {HARD_MAX_SYMBOLS})")
    if len(set(allow)) != len(allow):
        v.append("symbol_allowlist contains duplicates")
    bad = sorted(s for s in allow if not US_EQUITY_SYMBOL.match(s))
    if bad:
        v.append(f"symbol_allowlist has invalid symbols: {bad}")
    missing_hard = sorted(HARD_DENYLIST - deny)
    if missing_hard:
        v.append(f"symbol_denylist must keep every leveraged/inverse product; removed: {missing_hard}")
    both = sorted(set(allow) & (deny | HARD_DENYLIST))
    if both:
        v.append(f"symbols both allowlisted and denylisted: {both}")
    # Money / limits.
    for name, value in (
        ("max_order_notional", policy.max_order_notional),
        ("max_position_notional", policy.max_position_notional),
        ("max_concentration_pct", policy.max_concentration_pct),
        ("max_daily_loss", policy.max_daily_loss),
        ("max_planned_loss_per_trade", policy.max_planned_loss_per_trade),
        ("max_spread_bps", policy.max_spread_bps),
        ("max_quote_age_seconds", policy.max_quote_age_seconds),
        ("max_limit_deviation_bps", policy.max_limit_deviation_bps),
        ("min_price", policy.min_price),
        ("min_account_equity", policy.min_account_equity),
        ("default_stop_loss_bps", policy.default_stop_loss_bps),
    ):
        _positive(name, value, v)
    _ceiling("max_order_notional", policy.max_order_notional, HARD_MAX_ORDER_NOTIONAL, v)
    _ceiling("max_position_notional", policy.max_position_notional, HARD_MAX_POSITION_NOTIONAL, v)
    _ceiling("max_concentration_pct", policy.max_concentration_pct, HARD_MAX_CONCENTRATION_PCT, v)
    _ceiling("max_daily_loss", policy.max_daily_loss, HARD_MAX_DAILY_LOSS, v)
    _ceiling(
        "max_planned_loss_per_trade", policy.max_planned_loss_per_trade, HARD_MAX_PLANNED_LOSS_PER_TRADE, v
    )
    _ceiling("max_spread_bps", policy.max_spread_bps, HARD_MAX_SPREAD_BPS, v)
    _ceiling("max_quote_age_seconds", policy.max_quote_age_seconds, HARD_MAX_QUOTE_AGE_SECONDS, v)
    _ceiling("max_limit_deviation_bps", policy.max_limit_deviation_bps, HARD_MAX_LIMIT_DEVIATION_BPS, v)
    _ceiling("default_stop_loss_bps", policy.default_stop_loss_bps, HARD_MAX_DEFAULT_STOP_BPS, v)
    _ceiling("max_positions", policy.max_positions, HARD_MAX_POSITIONS, v)
    _ceiling("max_pending_orders", policy.max_pending_orders, HARD_MAX_PENDING_ORDERS, v)
    _ceiling("max_orders_per_minute", policy.max_orders_per_minute, HARD_MAX_ORDERS_PER_MINUTE, v)
    _ceiling("max_orders_per_day", policy.max_orders_per_day, HARD_MAX_ORDERS_PER_DAY, v)
    _ceiling("intent_ttl_seconds", policy.intent_ttl_seconds, HARD_MAX_INTENT_TTL_SECONDS, v)
    _ceiling(
        "duplicate_window_seconds", policy.duplicate_window_seconds, HARD_MAX_DUPLICATE_WINDOW_SECONDS, v
    )
    if policy.min_price < HARD_MIN_PRICE_FLOOR:
        v.append(f"min_price={policy.min_price} below hard floor {HARD_MIN_PRICE_FLOOR} (penny stocks excluded)")
    if policy.min_account_equity < HARD_MIN_ACCOUNT_EQUITY_FLOOR:
        v.append(f"min_account_equity={policy.min_account_equity} below hard floor {HARD_MIN_ACCOUNT_EQUITY_FLOOR}")
    if policy.min_buying_power_buffer < HARD_MIN_BUYING_POWER_BUFFER:
        v.append(
            f"min_buying_power_buffer={policy.min_buying_power_buffer} below {HARD_MIN_BUYING_POWER_BUFFER} "
            "(would allow spending more than available cash)"
        )
    _ceiling("min_buying_power_buffer", policy.min_buying_power_buffer, HARD_MAX_BUYING_POWER_BUFFER, v)
    if policy.default_stop_loss_bps < HARD_MIN_DEFAULT_STOP_BPS:
        v.append(f"default_stop_loss_bps={policy.default_stop_loss_bps} below floor {HARD_MIN_DEFAULT_STOP_BPS}")
    if policy.duplicate_window_seconds < HARD_MIN_DUPLICATE_WINDOW_SECONDS:
        v.append(
            f"duplicate_window_seconds={policy.duplicate_window_seconds} below floor "
            f"{HARD_MIN_DUPLICATE_WINDOW_SECONDS}"
        )
    if policy.max_orders_per_minute > policy.max_orders_per_day:
        v.append("max_orders_per_minute cannot exceed max_orders_per_day")
    _validate_extensions(policy, v)
    return v


def _validate_extensions(policy: RiskPolicy, v: list[str]) -> None:
    """Multi-day holding and PAPER crypto: bounded even while disabled, so enabling them is a one-flag change."""
    _ceiling("max_holding_sessions", policy.max_holding_sessions, HARD_MAX_HOLDING_SESSIONS, v)
    if policy.overnight_gap_budget_bps < HARD_MIN_OVERNIGHT_GAP_BUDGET_BPS:
        v.append(f"overnight_gap_budget_bps={policy.overnight_gap_budget_bps} below floor "
                 f"{HARD_MIN_OVERNIGHT_GAP_BUDGET_BPS} (overnight gaps are not bounded by stops)")
    crypto = list(policy.crypto_allowlist)
    if len(crypto) > HARD_MAX_CRYPTO_SYMBOLS:
        v.append(f"crypto_allowlist has {len(crypto)} pairs (> {HARD_MAX_CRYPTO_SYMBOLS})")
    if len(set(crypto)) != len(crypto):
        v.append("crypto_allowlist contains duplicates")
    bad = sorted(s for s in crypto if not CRYPTO_SYMBOL.match(s) or s.split("/")[0] in STABLECOIN_BASES)
    if bad:
        v.append(f"crypto_allowlist has invalid pairs (USD-quoted, non-stablecoin bases only): {bad}")
    if policy.crypto_enabled and not crypto:
        v.append("crypto_enabled requires a non-empty crypto_allowlist")
    for name, value in (
        ("crypto_max_order_notional", policy.crypto_max_order_notional),
        ("crypto_max_position_notional", policy.crypto_max_position_notional),
        ("crypto_max_spread_bps", policy.crypto_max_spread_bps),
        ("crypto_min_order_notional", policy.crypto_min_order_notional),
        ("crypto_default_stop_loss_bps", policy.crypto_default_stop_loss_bps),
        ("overnight_gap_budget_bps", policy.overnight_gap_budget_bps),
    ):
        _positive(name, value, v)
    _ceiling("crypto_max_order_notional", policy.crypto_max_order_notional, HARD_MAX_CRYPTO_ORDER_NOTIONAL, v)
    _ceiling("crypto_max_position_notional", policy.crypto_max_position_notional,
             HARD_MAX_CRYPTO_POSITION_NOTIONAL, v)
    _ceiling("crypto_max_positions", policy.crypto_max_positions, HARD_MAX_CRYPTO_POSITIONS, v)
    _ceiling("crypto_max_spread_bps", policy.crypto_max_spread_bps, HARD_MAX_CRYPTO_SPREAD_BPS, v)
    _ceiling("crypto_default_stop_loss_bps", policy.crypto_default_stop_loss_bps, HARD_MAX_CRYPTO_STOP_BPS, v)
    if policy.crypto_min_order_notional < HARD_MIN_CRYPTO_ORDER_NOTIONAL:
        v.append(f"crypto_min_order_notional below {HARD_MIN_CRYPTO_ORDER_NOTIONAL}")
    if policy.crypto_min_order_notional > policy.crypto_max_order_notional:
        v.append("crypto_min_order_notional cannot exceed crypto_max_order_notional")
    if policy.crypto_default_stop_loss_bps < HARD_MIN_DEFAULT_STOP_BPS:
        v.append(f"crypto_default_stop_loss_bps below floor {HARD_MIN_DEFAULT_STOP_BPS}")
    specs = {i.symbol: i for i in policy.crypto_instruments}
    if len(specs) != len(policy.crypto_instruments):
        v.append("crypto_instruments contains duplicate symbols")
    missing = sorted(s for s in crypto if s not in specs)
    if missing:
        v.append(f"crypto_allowlist pairs without pinned instrument increments: {missing}")
    for inst in policy.crypto_instruments:
        if inst.qty_increment <= 0 or inst.price_increment <= 0:
            v.append(f"crypto instrument {inst.symbol}: increments must be positive")
        if not inst.qty_increment.is_finite() or inst.qty_increment < Decimal("0.000000001"):
            v.append(f"crypto instrument {inst.symbol}: qty_increment finer than 1e-9")
    if policy.crypto_fee_bps < HARD_MIN_CRYPTO_FEE_BPS or policy.crypto_fee_bps > HARD_MAX_CRYPTO_FEE_BPS:
        v.append(f"crypto_fee_bps={policy.crypto_fee_bps} outside [{HARD_MIN_CRYPTO_FEE_BPS}, {HARD_MAX_CRYPTO_FEE_BPS}] "
                 "(fees may not be understated)")
    strategies = list(policy.multi_day_strategy_allowlist)
    if len(set(strategies)) != len(strategies):
        v.append("multi_day_strategy_allowlist contains duplicates")
    bad_ids = sorted(s for s in strategies if not STRATEGY_ID.match(s) or s.startswith(("agent:", "position-manager")))
    if bad_ids:
        v.append(f"multi_day_strategy_allowlist has invalid or non-strategy ids: {bad_ids}")
    if policy.allow_overnight_positions and not strategies:
        v.append("allow_overnight_positions requires a non-empty multi_day_strategy_allowlist")


def hard_ceilings() -> dict[str, str]:
    """JSON-safe view of the hard ceilings/floors (for the read-only policy view and dashboards)."""
    return {
        "max_order_notional": format(HARD_MAX_ORDER_NOTIONAL, "f"),
        "max_position_notional": format(HARD_MAX_POSITION_NOTIONAL, "f"),
        "max_concentration_pct": format(HARD_MAX_CONCENTRATION_PCT, "f"),
        "max_daily_loss": format(HARD_MAX_DAILY_LOSS, "f"),
        "max_planned_loss_per_trade": format(HARD_MAX_PLANNED_LOSS_PER_TRADE, "f"),
        "max_spread_bps": format(HARD_MAX_SPREAD_BPS, "f"),
        "max_quote_age_seconds": format(HARD_MAX_QUOTE_AGE_SECONDS, "f"),
        "max_limit_deviation_bps": format(HARD_MAX_LIMIT_DEVIATION_BPS, "f"),
        "min_price_floor": format(HARD_MIN_PRICE_FLOOR, "f"),
        "min_account_equity_floor": format(HARD_MIN_ACCOUNT_EQUITY_FLOOR, "f"),
        "min_buying_power_buffer": format(HARD_MIN_BUYING_POWER_BUFFER, "f"),
        "max_buying_power_buffer": format(HARD_MAX_BUYING_POWER_BUFFER, "f"),
        "default_stop_loss_bps_range": f"{HARD_MIN_DEFAULT_STOP_BPS}-{HARD_MAX_DEFAULT_STOP_BPS}",
        "max_positions": str(HARD_MAX_POSITIONS),
        "max_pending_orders": str(HARD_MAX_PENDING_ORDERS),
        "max_orders_per_minute": str(HARD_MAX_ORDERS_PER_MINUTE),
        "max_orders_per_day": str(HARD_MAX_ORDERS_PER_DAY),
        "max_intent_ttl_seconds": str(HARD_MAX_INTENT_TTL_SECONDS),
        "duplicate_window_seconds_range": f"{HARD_MIN_DUPLICATE_WINDOW_SECONDS}-{HARD_MAX_DUPLICATE_WINDOW_SECONDS}",
        "max_allowlist_symbols": str(HARD_MAX_SYMBOLS),
        "max_holding_sessions": str(HARD_MAX_HOLDING_SESSIONS),
        "min_overnight_gap_budget_bps": format(HARD_MIN_OVERNIGHT_GAP_BUDGET_BPS, "f"),
        "max_crypto_order_notional": format(HARD_MAX_CRYPTO_ORDER_NOTIONAL, "f"),
        "max_crypto_position_notional": format(HARD_MAX_CRYPTO_POSITION_NOTIONAL, "f"),
        "max_crypto_positions": str(HARD_MAX_CRYPTO_POSITIONS),
        "max_crypto_spread_bps": format(HARD_MAX_CRYPTO_SPREAD_BPS, "f"),
    }


STRUCTURAL_INVARIANTS: Final[tuple[str, ...]] = (
    "long-only: SELL may only reduce an existing long position; no shorting",
    "no margin borrowing: BUY notional x buffer must be covered by settled cash / non-marginable buying power",
    "regular session only (XNYS calendar); extended hours always rejected",
    "LIMIT orders only for entries, time-in-force DAY, whole shares; protective exits may rest as STOP_LIMIT SELLs "
    "(a stop is a trigger, never a guaranteed exit price)",
    "no new entries in the last 30 minutes of the regular session; exits continue until the close",
    "ENTRIES-scope halts and unprotected-position safety states stop new entries but never block risk-reducing exits",
    "leveraged/inverse/volatility products always denylisted",
    "live execution compiled off; LIVE_APPROVAL / BOUNDED_LIVE always denied in this build",
    "multi-day equity holding and PAPER crypto are disabled unless an operator policy change enables them; crypto "
    "is long-only LIMIT (GTC/IOC) on USD pairs, bounded by its own notional ceilings, PAPER mode only",
    "the AI has read-only access to the policy; only the authenticated human operator can change limits",
)
