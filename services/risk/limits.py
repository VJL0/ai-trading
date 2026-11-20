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
    return v


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
    }


STRUCTURAL_INVARIANTS: Final[tuple[str, ...]] = (
    "long-only: SELL may only reduce an existing long position; no shorting",
    "no margin borrowing: BUY notional x buffer must be covered by settled cash / non-marginable buying power",
    "regular session only (XNYS calendar); extended hours always rejected",
    "LIMIT orders only, time-in-force DAY, whole shares",
    "leveraged/inverse/volatility products always denylisted",
    "live execution compiled off; LIVE_APPROVAL / BOUNDED_LIVE always denied in this build",
    "the AI has read-only access to the policy; only the authenticated human operator can change limits",
)
