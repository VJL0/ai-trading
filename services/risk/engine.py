"""Independent deterministic Risk & Policy Gate (spec §5, §6, §8, §12; architectural invariants 1-4).

``RiskEngine`` is the *final authorization* for every order:

* ``evaluate(proposal, ctx)`` -> ``RiskDecision``. Deterministic and pure: it reads only the proposal,
  the ``RiskContext`` snapshot, the immutable policy, the injected clock and the in-memory XNYS
  calendar. It never performs I/O. All money/quantity/price arithmetic is ``Decimal``.
  It *independently recomputes* the reference price (ask for BUY, bid for SELL), notional, post-trade
  position notional, concentration and planned max loss, and never trusts proposal arithmetic
  (``max_planned_loss`` is recorded and cross-checked, never used). It never resizes: an oversized
  order is REJECTED and the maximum allowable quantity is reported.
* ``authorize(proposal, decision, ctx)`` -> ECDSA-P256-signed ``ApprovedOrderIntent``; only for an
  APPROVED decision issued by *this* engine instance, and only if a re-evaluation still approves.
* ``final_gate(intent, ctx, verifier, active_policy_version)`` -> list of failures (empty == pass). Run
  at the final order boundary immediately before gateway submission.

Fail closed everywhere: any missing, stale, unknown or unparseable input fails its check, and any
exception while evaluating becomes a REJECTED decision (or a final-gate failure), never an approval.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Final, TypeVar

from pydantic import BaseModel, ValidationError

from packages.contracts.enums import (
    TERMINAL_ORDER_STATES,
    DataFeed,
    DecisionAction,
    ExecutionLabel,
    MarketSession,
    OrderType,
    RiskVerdict,
    SessionConstraint,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.contracts.models import (
    AccountSnapshot,
    ApprovedOrderIntent,
    BrokerOrder,
    HumanApproval,
    Quote,
    RiskCheck,
    RiskContext,
    RiskDecision,
    TradeProposal,
)
from packages.contracts.policy import RiskPolicy
from packages.core import modes
from packages.core.canonical import canonical_json, sha256_hex
from packages.core.clock import Clock
from packages.core.ids import client_order_id, new_id
from packages.core.money import BPS, ZERO, floor_shares, price_tick
from packages.core.signing import SIGNATURE_ALG, IntentSigner, IntentVerifier

from . import limits, market_hours
from .context import ANY_STRATEGY, fingerprint, fingerprint_variants

__all__ = [
    "REQUIRED_TRADE_CHECKS",
    "Exposure",
    "RiskAuthorizationError",
    "RiskEngine",
    "fingerprint",
    "max_allowable_quantity",
]

# ---------------------------------------------------------------------------- check names
CHECK_POLICY_VALID: Final = "policy_valid"
CHECK_ACTION: Final = "action_trade"
CHECK_CONTEXT_FRESH: Final = "context_fresh"
CHECK_INPUTS_VALID: Final = "input_contracts_valid"
CHECK_MODE: Final = "mode_permits_trading"
CHECK_PROPOSAL_COMPLETE: Final = "proposal_complete"
CHECK_RECOMPUTE: Final = "independent_recompute"
CHECK_SYMBOL_FORMAT: Final = "symbol_format"
CHECK_ALLOWLIST: Final = "symbol_allowlist"
CHECK_DENYLIST: Final = "symbol_denylist"
CHECK_ORDER_TYPE: Final = "order_type_limit_only"
CHECK_TIF: Final = "time_in_force_day"
CHECK_WHOLE_SHARES: Final = "whole_shares"
CHECK_PRICE_TICK: Final = "limit_price_tick"
CHECK_MARKET_PRESENT: Final = "market_snapshot_present"
CHECK_QUOTE_PRESENT: Final = "quote_present"
CHECK_DATA_PROVENANCE: Final = "data_provenance"
CHECK_MIN_PRICE: Final = "min_price"
CHECK_SESSION: Final = "regular_session"
CHECK_LIMIT_BAND: Final = "limit_vs_mid"
CHECK_LIMIT_TOUCH: Final = "limit_vs_touch"
CHECK_SPREAD: Final = "max_spread"
CHECK_QUOTE_AGE: Final = "max_quote_age"
CHECK_DATA_ASOF: Final = "data_asof_fresh"
CHECK_ACCOUNT_PRESENT: Final = "account_present"
CHECK_ACCOUNT_FRESH: Final = "account_fresh"
CHECK_ACCOUNT_STATUS: Final = "account_status"
CHECK_POSITIONS_CONSISTENT: Final = "positions_consistent"
CHECK_LONG_ONLY: Final = "long_only"
CHECK_NO_MARGIN: Final = "no_margin"
CHECK_MAX_POSITIONS: Final = "max_positions"
CHECK_ORDER_NOTIONAL: Final = "max_order_notional"
CHECK_POSITION_NOTIONAL: Final = "max_position_notional"
CHECK_CONCENTRATION: Final = "max_concentration"
CHECK_DAILY_LOSS: Final = "max_daily_loss"
CHECK_DAILY_LOSS_BUDGET: Final = "daily_loss_budget"
CHECK_PLANNED_LOSS: Final = "max_planned_loss"
CHECK_PROPOSAL_ARITHMETIC: Final = "proposal_arithmetic"
CHECK_EQUITY_FLOOR: Final = "account_equity_floor"
CHECK_PENDING: Final = "max_pending_orders"
CHECK_RATE_MINUTE: Final = "order_rate_per_minute"
CHECK_RATE_DAY: Final = "order_rate_per_day"
CHECK_HEALTH_FEED: Final = "health_feed"
CHECK_HEALTH_BROKER: Final = "health_broker_connected"
CHECK_HEALTH_RECONCILED: Final = "health_reconciled"
CHECK_HEALTH_ENGINE: Final = "health_engine"
CHECK_HEALTH_ENTRIES: Final = "health_entries_unlocked"
CHECK_HEALTH_STRATEGY: Final = "health_strategy_state"
CHECK_HEALTH_PERMISSIONS: Final = "health_permissions"
CHECK_DUPLICATE: Final = "duplicate_order"
CHECK_EVIDENCE: Final = "evidence_present"
CHECK_MAX_QTY: Final = "max_allowable_quantity"

REQUIRED_TRADE_CHECKS: Final[frozenset[str]] = frozenset(
    {
        CHECK_POLICY_VALID, CHECK_ACTION, CHECK_INPUTS_VALID, CHECK_CONTEXT_FRESH, CHECK_MODE, CHECK_PROPOSAL_COMPLETE,
        CHECK_RECOMPUTE, CHECK_SYMBOL_FORMAT, CHECK_ALLOWLIST, CHECK_DENYLIST, CHECK_ORDER_TYPE, CHECK_TIF,
        CHECK_WHOLE_SHARES, CHECK_PRICE_TICK, CHECK_MARKET_PRESENT, CHECK_QUOTE_PRESENT,
        CHECK_DATA_PROVENANCE, CHECK_MIN_PRICE, CHECK_SESSION, CHECK_LIMIT_BAND, CHECK_LIMIT_TOUCH,
        CHECK_SPREAD, CHECK_QUOTE_AGE, CHECK_DATA_ASOF, CHECK_ACCOUNT_PRESENT, CHECK_ACCOUNT_FRESH,
        CHECK_ACCOUNT_STATUS, CHECK_POSITIONS_CONSISTENT, CHECK_LONG_ONLY, CHECK_NO_MARGIN,
        CHECK_MAX_POSITIONS, CHECK_ORDER_NOTIONAL, CHECK_POSITION_NOTIONAL, CHECK_CONCENTRATION,
        CHECK_DAILY_LOSS, CHECK_DAILY_LOSS_BUDGET, CHECK_PLANNED_LOSS, CHECK_PROPOSAL_ARITHMETIC,
        CHECK_EQUITY_FLOOR, CHECK_PENDING, CHECK_RATE_MINUTE, CHECK_RATE_DAY, CHECK_HEALTH_FEED,
        CHECK_HEALTH_BROKER, CHECK_HEALTH_RECONCILED, CHECK_HEALTH_ENGINE, CHECK_HEALTH_ENTRIES,
        CHECK_HEALTH_STRATEGY, CHECK_HEALTH_PERMISSIONS, CHECK_DUPLICATE, CHECK_EVIDENCE, CHECK_MAX_QTY,
    }
)
"""Every one of these must have run *and* passed for an APPROVED verdict (guards against code paths
that silently skip a check)."""

LIVE_MODES: Final[frozenset[TradingMode]] = frozenset({TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE})
ACCOUNT_ACTIVE_STATUS: Final = "ACTIVE"
NO_TRADE_REASON: Final = "no trade requested"
_DECISION_REGISTRY_MAX: Final = 4096
_SECONDS_PER_MINUTE: Final = Decimal(60)
_MICROS: Final = Decimal(1_000_000)
_QUANT_DISPLAY: Final = Decimal("0.0001")


class RiskAuthorizationError(PermissionError):
    """``authorize`` refused to sign an intent. Never retried automatically."""


# ---------------------------------------------------------------------------- helpers


def _f(value: Decimal | None) -> str | None:
    """Exact decimal string (no exponent)."""
    return None if value is None else format(value, "f")


def _fq(value: Decimal | None) -> str | None:
    """Display-rounded decimal string for derived ratios (comparisons always use exact values)."""
    if value is None:
        return None
    return format(value.quantize(_QUANT_DISPLAY), "f")


def _age_seconds(now: datetime, ts: datetime) -> Decimal:
    """Exact age in seconds as Decimal (negative == ``ts`` is in the future)."""
    delta = now - ts
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return Decimal(micros) / _MICROS


def _require_aware(dt: datetime, what: str) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{what} is a naive datetime")
    return dt.astimezone(UTC)


def _is_open(order: BrokerOrder) -> bool:
    return order.state not in TERMINAL_ORDER_STATES


def _remaining(order: BrokerOrder) -> Decimal:
    rem = order.quantity - order.filled_quantity
    return rem if rem > ZERO else ZERO


def _on_tick(price: Decimal) -> bool:
    tick = price_tick(price)
    return price == price.quantize(tick)


def _floor_div(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator <= ZERO:
        raise ZeroDivisionError("non-positive denominator")
    return floor_shares(numerator / denominator)


def _policy_violations(policy: RiskPolicy) -> tuple[str, ...]:
    """Hard-limit violations, plus a re-validation of the raw values (defeats ``model_construct`` floats)."""
    try:
        RiskPolicy.model_validate(policy.model_dump(warnings=False))
    except Exception as exc:  # any non-validating policy blocks every trade
        return (f"policy does not validate ({type(exc).__name__})",)
    return tuple(limits.validate_policy(policy))


def _pick(ok: bool, passed: str, failed: str) -> str:
    return passed if ok else failed


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _revalidate_one(obj: _ModelT, what: str) -> tuple[_ModelT, str | None]:
    """Re-validate ``obj`` from its raw python-mode values (never JSON mode, which would launder a float into a
    decimal string). Returns (validated copy, None) or (original, problem). A copy that differs from the
    original means a value was smuggled past validation (``model_construct``/``model_copy``) and was coerced
    (e.g. ``"no"`` -> False): that is a problem too, so the caller fails closed."""
    cls = type(obj)
    try:
        validated = cls.model_validate(obj.model_dump(warnings=False))
    except ValidationError as exc:
        return obj, f"{what} does not validate against its contract ({exc.error_count()} error(s))"
    except Exception as exc:  # unserialisable garbage
        return obj, f"{what} cannot be re-validated ({type(exc).__name__})"
    try:
        same = validated == obj
    except Exception:
        same = False
    if not same:
        return validated, f"{what} is not in validated form (a field was set without validation and coerced)"
    return validated, None


def _revalidated(proposal: TradeProposal, ctx: RiskContext) -> tuple[TradeProposal, RiskContext, str | None]:
    """Validated copies of both evaluation inputs plus the first problem found (None == both canonical)."""
    p, p_problem = _revalidate_one(proposal, "proposal")
    cx, c_problem = _revalidate_one(ctx, "risk context")
    return p, cx, p_problem or c_problem


def _decision_digest(decision: RiskDecision) -> str:
    return sha256_hex(canonical_json(decision))


def _proposal_digest(proposal: TradeProposal) -> str:
    return sha256_hex(canonical_json(proposal))


CheckResult = tuple[bool, str, str | None, str | None]
"""(passed, detail, observed, limit)"""


class _Checks:
    """Ordered collector. Each check runs isolated: an exception fails that check (fail closed)."""

    def __init__(self) -> None:
        self.items: list[RiskCheck] = []

    def add(self, name: str, passed: bool, detail: str, observed: str | None = None, limit: str | None = None) -> bool:
        self.items.append(RiskCheck(name=name, passed=bool(passed), detail=detail, observed=observed, limit=limit))
        return bool(passed)

    def run(self, name: str, fn: Callable[[], CheckResult]) -> bool:
        try:
            passed, detail, observed, limit = fn()
        except Exception as exc:  # fail closed: an evaluation error is a failed check
            return self.add(name, False, f"could not evaluate ({type(exc).__name__}: {exc}); fail closed")
        return self.add(name, passed, detail, observed, limit)

    @property
    def all_passed(self) -> bool:
        return bool(self.items) and all(c.passed for c in self.items)

    def names_passed(self) -> set[str]:
        return {c.name for c in self.items if c.passed}

    def reasons(self) -> tuple[str, ...]:
        return tuple(f"{c.name}: {c.detail}" for c in self.items if not c.passed)


@dataclass(frozen=True, slots=True)
class Exposure:
    """Numbers the engine recomputed itself from market + account state (never from the proposal's
    arithmetic). ``None`` means the input needed was unavailable (and the dependent check fails)."""

    symbol: str
    side: Side
    quantity: Decimal
    limit_price: Decimal
    reference_price: Decimal | None
    mid: Decimal | None
    spread_bps: Decimal | None
    notional: Decimal
    existing_qty: Decimal
    pending_buy_qty_symbol: Decimal
    pending_sell_qty_symbol: Decimal
    pending_buy_notional: Decimal | None
    post_trade_qty: Decimal
    valuation_price: Decimal | None
    post_trade_notional: Decimal | None
    equity: Decimal | None
    concentration: Decimal | None
    stop_price: Decimal | None
    stop_source: str
    planned_loss: Decimal
    claimed_planned_loss: Decimal | None
    daily_pl_worst: Decimal | None
    max_allowable_qty: Decimal | None
    binding_limit: str

    @property
    def is_buy(self) -> bool:
        return self.side == Side.BUY


def max_allowable_quantity(decision: RiskDecision) -> Decimal | None:
    """The largest whole-share quantity every limit allowed, as reported in the decision (never applied).

    ``None`` when the engine could not compute it (missing inputs) or the proposal was NO_TRADE.
    """
    for check in decision.checks:
        if check.name == CHECK_MAX_QTY and check.limit is not None:
            try:
                return Decimal(check.limit)
            except ArithmeticError:
                return None
    return None


# ---------------------------------------------------------------------------- engine


class RiskEngine:
    """Deterministic final authorization. Holds the intent-signing private key (via ``IntentSigner``)."""

    def __init__(
        self,
        policy: RiskPolicy,
        signer: IntentSigner,
        clock: Clock,
        *,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self._policy = policy
        self._policy_version = policy.version
        self._policy_violations: tuple[str, ...] = _policy_violations(policy)
        self._signer = signer
        self._clock = clock
        self._new_id: Callable[[str], str] = id_factory or new_id
        # decision_id -> (decision digest, proposal digest): authorize() only signs what evaluate() approved.
        self._issued: OrderedDict[str, tuple[str, str]] = OrderedDict()
        # decision_id -> the one intent signed for it (authorize is idempotent; never a second order).
        self._intents: dict[str, ApprovedOrderIntent] = {}
        self._lock = threading.Lock()
        # Warm the XNYS calendar (in-memory after the first load) so evaluate() itself does no file I/O.
        market_hours.is_regular_session_open(datetime(2026, 1, 2, 15, 0, tzinfo=UTC))

    # ------------------------------------------------------------------ properties
    @property
    def policy(self) -> RiskPolicy:
        return self._policy

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def key_id(self) -> str:
        return self._signer.key_id

    @property
    def policy_violations(self) -> tuple[str, ...]:
        return self._policy_violations

    @staticmethod
    def fingerprint(item: TradeProposal | ApprovedOrderIntent, strategy_id: str | None = None) -> str | None:
        """Duplicate-detection fingerprint (see ``services.risk.context.fingerprint``)."""
        return fingerprint(item, strategy_id)

    def _now(self) -> datetime:
        return _require_aware(self._clock.now(), "engine clock")

    # ------------------------------------------------------------------ recompute
    def recompute(self, proposal: TradeProposal, ctx: RiskContext) -> Exposure:
        """Independently recompute exposure numbers for a TRADE proposal (raises on missing fields)."""
        if proposal.symbol is None or proposal.side is None or proposal.quantity is None or proposal.limit_price is None:
            raise ValueError("TRADE proposal is missing symbol/side/quantity/limit_price")
        pol = self._policy
        symbol = proposal.symbol
        side = proposal.side
        qty = proposal.quantity
        limit = proposal.limit_price
        if qty <= ZERO or limit <= ZERO:
            raise ValueError("quantity and limit_price must be positive")

        quote = self._quote_for(ctx, symbol)
        reference: Decimal | None = None
        mid: Decimal | None = None
        spread: Decimal | None = None
        if quote is not None:
            reference = quote.ask if side == Side.BUY else quote.bid
            mid = (quote.bid + quote.ask) / Decimal(2)
            spread = (quote.ask - quote.bid) / mid * BPS

        existing = sum((p.quantity for p in ctx.positions if p.symbol == symbol), ZERO)
        open_orders = [o for o in ctx.open_orders if _is_open(o)]
        pend_buy_sym = sum((_remaining(o) for o in open_orders if o.symbol == symbol and o.side == Side.BUY), ZERO)
        pend_sell_sym = sum((_remaining(o) for o in open_orders if o.symbol == symbol and o.side == Side.SELL), ZERO)
        pend_buy_notional: Decimal | None = ZERO
        for o in open_orders:
            if o.side != Side.BUY:
                continue
            if o.limit_price is None or pend_buy_notional is None:
                pend_buy_notional = None  # an open BUY without a limit has unknown cash need => fail closed
                continue
            pend_buy_notional += _remaining(o) * o.limit_price

        notional = qty * limit
        account = ctx.account
        equity = account.equity if account is not None else None

        if side == Side.BUY:
            post_qty = existing + pend_buy_sym + qty
            valuation = max(limit, reference) if reference is not None else None
        else:
            post_qty = existing - pend_sell_sym - qty
            valuation = reference if reference is not None else None
        post_notional = (max(post_qty, ZERO) * valuation) if valuation is not None else None
        concentration: Decimal | None = None
        if post_notional is not None and equity is not None and equity > ZERO:
            concentration = post_notional / equity

        # Planned max loss: proposal stop if valid, else policy default. Never the proposal's number.
        claimed = proposal.max_planned_loss
        stop_price: Decimal | None = None
        stop_source = "not_applicable_exit"
        planned_loss = ZERO
        if side == Side.BUY:
            stop = proposal.stop_price
            if stop is not None and ZERO < stop < limit and (limit - stop) / limit * BPS >= limits.MIN_STOP_DISTANCE_BPS:
                stop_price, stop_source = stop, "proposal_stop"
            else:
                raw = limit * (Decimal(1) - pol.default_stop_loss_bps / BPS)
                tick = price_tick(raw)
                floored = (raw / tick).to_integral_value(rounding=ROUND_FLOOR) * tick  # conservative: lower stop
                stop_price = floored
                stop_source = "policy_default_stop" if stop is None else "policy_default_stop(proposal stop invalid)"
            planned_loss = qty * (limit - stop_price)

        # Daily P&L: the worse of (ledger realized + unrealized) and (equity - start-of-day equity). The
        # equity baseline is required for entries: without it daily P&L is unknown => fail closed.
        ledger_pl = ctx.daily_realized_pl + ctx.daily_unrealized_pl
        baseline = ctx.start_of_day_equity
        if baseline is None and account is not None:
            baseline = account.last_equity
        worst: Decimal | None = None
        if account is not None and baseline is not None:
            worst = min(ledger_pl, account.equity - baseline)

        max_qty, binding = self._max_allowable(
            side=side, limit=limit, valuation=valuation, existing=existing, pend_buy_sym=pend_buy_sym,
            pend_sell_sym=pend_sell_sym, pend_buy_notional=pend_buy_notional, account=account,
            stop_price=stop_price, worst=worst,
        )
        return Exposure(
            symbol=symbol, side=side, quantity=qty, limit_price=limit, reference_price=reference, mid=mid,
            spread_bps=spread, notional=notional, existing_qty=existing, pending_buy_qty_symbol=pend_buy_sym,
            pending_sell_qty_symbol=pend_sell_sym, pending_buy_notional=pend_buy_notional, post_trade_qty=post_qty,
            valuation_price=valuation, post_trade_notional=post_notional, equity=equity, concentration=concentration,
            stop_price=stop_price, stop_source=stop_source, planned_loss=planned_loss, claimed_planned_loss=claimed,
            daily_pl_worst=worst, max_allowable_qty=max_qty, binding_limit=binding,
        )

    def _max_allowable(
        self,
        *,
        side: Side,
        limit: Decimal,
        valuation: Decimal | None,
        existing: Decimal,
        pend_buy_sym: Decimal,
        pend_sell_sym: Decimal,
        pend_buy_notional: Decimal | None,
        account: AccountSnapshot | None,
        stop_price: Decimal | None,
        worst: Decimal | None,
    ) -> tuple[Decimal | None, str]:
        """Largest whole-share quantity every limit allows (reported, never applied)."""
        if side == Side.SELL:
            return max(floor_shares(existing - pend_sell_sym), ZERO), "existing_long_position"
        pol = self._policy
        caps: dict[str, Decimal] = {"max_order_notional": _floor_div(pol.max_order_notional, limit)}
        if valuation is None or account is None or pend_buy_notional is None or stop_price is None or worst is None:
            return None, "unknown (missing quote/account/order inputs)"
        held = existing + pend_buy_sym
        caps["max_position_notional"] = _floor_div(pol.max_position_notional, valuation) - held
        if account.equity <= ZERO:
            return ZERO, "non-positive account equity"
        caps["max_concentration"] = _floor_div(pol.max_concentration_pct * account.equity, valuation) - held
        per_share_loss = limit - stop_price
        caps["max_planned_loss"] = _floor_div(pol.max_planned_loss_per_trade, per_share_loss)
        budget = pol.max_daily_loss + min(worst, ZERO)
        caps["daily_loss_budget"] = _floor_div(budget, per_share_loss) if budget > ZERO else ZERO
        spendable = min(self._spendable_candidates(account)) - pend_buy_notional
        caps["no_margin_cash"] = _floor_div(spendable, limit * pol.min_buying_power_buffer) if spendable > ZERO else ZERO
        binding = min(caps, key=lambda k: caps[k])
        return max(floor_shares(caps[binding]), ZERO), binding

    @staticmethod
    def _spendable_candidates(account: AccountSnapshot) -> list[Decimal]:
        """Cash-like amounts a BUY must fit inside. Margin buying power is only ever a further cap."""
        vals = [account.cash, account.buying_power]
        if account.non_marginable_buying_power is not None:
            vals.append(account.non_marginable_buying_power)
        return vals

    @staticmethod
    def _quote_for(ctx: RiskContext, symbol: str) -> Quote | None:
        market = ctx.market
        if market is None or market.symbol != symbol or market.quote is None:
            return None
        if market.quote.symbol != symbol:
            return None
        return market.quote

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, proposal: TradeProposal, ctx: RiskContext) -> RiskDecision:
        """Evaluate a proposal against the policy and context. Never raises; never approves on error."""
        try:
            decision = self._evaluate(proposal, ctx)
        except Exception as exc:  # fail closed
            decision = self._error_decision(proposal, ctx, exc)
        self._register(decision, proposal)
        return decision

    def _register(self, decision: RiskDecision, proposal: TradeProposal) -> None:
        if not decision.approved:
            return
        try:
            entry = (_decision_digest(decision), _proposal_digest(proposal))
        except Exception:  # undigestable => never registered => authorize() refuses (fail closed)
            return
        with self._lock:
            self._issued[decision.decision_id] = entry
            while len(self._issued) > _DECISION_REGISTRY_MAX:
                evicted, _ = self._issued.popitem(last=False)
                self._intents.pop(evicted, None)

    def _error_decision(self, proposal: TradeProposal, ctx: RiskContext, exc: Exception) -> RiskDecision:
        try:
            now = self._now()
        except Exception:
            now = datetime.now(UTC)
        raw_mode: object = getattr(ctx, "mode", None)  # ctx may not even be a RiskContext here
        mode = raw_mode if isinstance(raw_mode, TradingMode) else TradingMode.RESEARCH_ONLY
        detail = f"risk evaluation error ({type(exc).__name__}: {exc}); fail closed"
        try:
            decision_id = self._new_id("rdec")
        except Exception:
            decision_id = new_id("rdec")
        return RiskDecision(
            decision_id=decision_id,
            proposal_id=str(getattr(proposal, "proposal_id", "unknown")),
            causal_id=str(getattr(proposal, "causal_id", "unknown")),
            verdict=RiskVerdict.REJECTED,
            checks=(RiskCheck(name="evaluation_error", passed=False, detail=detail),),
            reasons=(f"evaluation_error: {detail}",),
            policy_version=self._policy_version,
            mode=mode,
            evaluated_at=now,
        )

    def _decision(
        self,
        proposal: TradeProposal,
        ctx: RiskContext,
        now: datetime,
        checks: _Checks,
        exposure: Exposure | None,
        *,
        force_reject: bool = False,
    ) -> RiskDecision:
        approved = (
            not force_reject
            and checks.all_passed
            and REQUIRED_TRADE_CHECKS.issubset(checks.names_passed())
        )
        reasons = checks.reasons()
        if not approved and not reasons:
            reasons = ("not every required risk check ran; fail closed",)
        market = ctx.market
        return RiskDecision(
            decision_id=self._new_id("rdec"),
            proposal_id=proposal.proposal_id,
            causal_id=proposal.causal_id,
            verdict=RiskVerdict.APPROVED if approved else RiskVerdict.REJECTED,
            checks=tuple(checks.items),
            reasons=() if approved else reasons,
            policy_version=self._policy_version,
            mode=ctx.mode,
            recomputed_quantity=exposure.quantity if exposure is not None else None,
            recomputed_notional=exposure.notional if exposure is not None else None,
            recomputed_max_loss=exposure.planned_loss if exposure is not None else None,
            reference_price=exposure.reference_price if exposure is not None else None,
            account_snapshot_id=ctx.account.snapshot_id if ctx.account is not None else None,
            market_snapshot_id=market.snapshot_id if market is not None else None,
            evaluated_at=now,
        )

    def _evaluate(self, proposal: TradeProposal, ctx: RiskContext) -> RiskDecision:
        now = self._now()
        c = _Checks()
        c.run(CHECK_POLICY_VALID, self._chk_policy_valid)

        if proposal.action != DecisionAction.TRADE:
            why = proposal.no_trade_reason or "proposal action is NO_TRADE"
            c.add(CHECK_ACTION, False, f"{NO_TRADE_REASON}: {why}", observed=str(proposal.action), limit="TRADE")
            return self._decision(proposal, ctx, now, c, None, force_reject=True)
        c.add(CHECK_ACTION, True, "TRADE requested", observed="TRADE", limit="TRADE")

        # Every later check reads the *re-validated* copies, never the caller's objects: a value smuggled in
        # through model_construct/model_copy (e.g. health.feed_ok="no", account.is_paper="false") would be
        # truthy as a raw string. Inputs that do not re-validate to themselves also fail input_contracts_valid.
        proposal, ctx, problem = _revalidated(proposal, ctx)
        c.add(
            CHECK_INPUTS_VALID, problem is None,
            "proposal and risk context re-validate to themselves against their contracts" if problem is None
            else f"{problem}; fail closed",
        )
        c.run(CHECK_CONTEXT_FRESH, lambda: self._chk_context_fresh(ctx, now))
        c.run(CHECK_MODE, lambda: self._chk_mode(ctx))
        c.run(CHECK_EVIDENCE, lambda: self._chk_evidence(proposal))
        self._health_checks(c, ctx)

        missing = [n for n in ("symbol", "side", "quantity", "limit_price") if getattr(proposal, n) is None]
        if missing:
            c.add(CHECK_PROPOSAL_COMPLETE, False, f"TRADE proposal missing {missing}")
            return self._decision(proposal, ctx, now, c, None, force_reject=True)
        c.add(CHECK_PROPOSAL_COMPLETE, True, "symbol, side, quantity and limit present")

        try:
            x = self.recompute(proposal, ctx)
        except Exception as exc:
            c.add(CHECK_RECOMPUTE, False, f"independent recompute failed ({type(exc).__name__}: {exc})")
            return self._decision(proposal, ctx, now, c, None, force_reject=True)
        c.add(
            CHECK_RECOMPUTE, True,
            "reference price, notional, exposure and planned loss recomputed independently "
            f"(claimed max_planned_loss={_f(x.claimed_planned_loss)} recorded, not used)",
            observed=_f(x.notional), limit=None,
        )

        c.run(CHECK_SYMBOL_FORMAT, lambda: self._chk_symbol_format(x.symbol))
        c.run(CHECK_ALLOWLIST, lambda: self._chk_allowlist(x.symbol))
        c.run(CHECK_DENYLIST, lambda: self._chk_denylist(x.symbol))
        c.run(CHECK_ORDER_TYPE, lambda: self._chk_order_type(proposal.order_type))
        c.run(CHECK_TIF, lambda: self._chk_tif(proposal.time_in_force))
        c.run(CHECK_WHOLE_SHARES, lambda: self._chk_whole_shares(x.quantity))
        c.run(CHECK_PRICE_TICK, lambda: self._chk_tick(x.limit_price))
        c.run(CHECK_MARKET_PRESENT, lambda: self._chk_market_present(ctx, x.symbol))
        c.run(CHECK_QUOTE_PRESENT, lambda: self._chk_quote_present(ctx, x.symbol))
        c.run(CHECK_DATA_PROVENANCE, lambda: self._chk_provenance(ctx))
        c.run(CHECK_MIN_PRICE, lambda: self._chk_min_price(x))
        c.run(CHECK_SESSION, lambda: self._chk_session(ctx, proposal.session_constraint, now))
        c.run(CHECK_LIMIT_BAND, lambda: self._chk_limit_band(x))
        c.run(CHECK_LIMIT_TOUCH, lambda: self._chk_limit_touch(ctx, x))
        c.run(CHECK_SPREAD, lambda: self._chk_spread(x))
        c.run(CHECK_QUOTE_AGE, lambda: self._chk_quote_age(ctx, x.symbol, now))
        c.run(CHECK_DATA_ASOF, lambda: self._chk_data_asof(proposal, ctx, now))
        c.run(CHECK_ACCOUNT_PRESENT, lambda: self._chk_account_present(ctx))
        c.run(CHECK_ACCOUNT_FRESH, lambda: self._chk_account_fresh(ctx, now))
        c.run(CHECK_ACCOUNT_STATUS, lambda: self._chk_account_status(ctx))
        c.run(CHECK_POSITIONS_CONSISTENT, lambda: self._chk_positions(ctx, now))
        c.run(CHECK_LONG_ONLY, lambda: self._chk_long_only(x))
        c.run(CHECK_NO_MARGIN, lambda: self._chk_no_margin(ctx, x))
        c.run(CHECK_MAX_POSITIONS, lambda: self._chk_max_positions(ctx, x))
        c.run(CHECK_ORDER_NOTIONAL, lambda: self._chk_order_notional(x))
        c.run(CHECK_POSITION_NOTIONAL, lambda: self._chk_position_notional(x))
        c.run(CHECK_CONCENTRATION, lambda: self._chk_concentration(x))
        c.run(CHECK_DAILY_LOSS, lambda: self._chk_daily_loss(x))
        c.run(CHECK_DAILY_LOSS_BUDGET, lambda: self._chk_daily_loss_budget(x))
        c.run(CHECK_PLANNED_LOSS, lambda: self._chk_planned_loss(x))
        c.run(CHECK_PROPOSAL_ARITHMETIC, lambda: self._chk_proposal_arithmetic(x))
        c.run(CHECK_EQUITY_FLOOR, lambda: self._chk_equity_floor(ctx, x))
        c.run(CHECK_PENDING, lambda: self._chk_pending(ctx))
        c.run(CHECK_RATE_MINUTE, lambda: self._chk_rate_minute(ctx, now))
        c.run(CHECK_RATE_DAY, lambda: self._chk_rate_day(ctx, now))
        c.run(CHECK_DUPLICATE, lambda: self._chk_duplicate(proposal, ctx, x))
        c.run(CHECK_MAX_QTY, lambda: self._chk_max_qty(x))
        return self._decision(proposal, ctx, now, c, x)

    # ------------------------------------------------------------------ individual checks
    def _chk_policy_valid(self) -> CheckResult:
        if self._policy_violations:
            return False, "policy violates hard limits: " + "; ".join(self._policy_violations), None, None
        return True, f"policy {self._policy_version} within hard ceilings", self._policy_version, None

    @staticmethod
    def _chk_context_fresh(ctx: RiskContext, now: datetime) -> CheckResult:
        ctx_now = _require_aware(ctx.now, "ctx.now")
        age = _age_seconds(now, ctx_now)
        if age < -limits.MAX_CLOCK_SKEW_SECONDS:
            return False, "risk context is from the future (clock skew)", _fq(age), _f(-limits.MAX_CLOCK_SKEW_SECONDS)
        if age > limits.MAX_CONTEXT_AGE_SECONDS:
            return False, "risk context is stale", _fq(age), _f(limits.MAX_CONTEXT_AGE_SECONDS)
        return True, "risk context is current", _fq(age), _f(limits.MAX_CONTEXT_AGE_SECONDS)

    @staticmethod
    def _chk_mode(ctx: RiskContext) -> CheckResult:
        observed = f"{ctx.mode.value}/{ctx.gateway.value}"
        if ctx.mode == TradingMode.RESEARCH_ONLY:
            return False, "RESEARCH_ONLY mode never trades", observed, "REPLAY|PAPER"
        if ctx.mode in LIVE_MODES or ctx.gateway in modes.LIVE_GATEWAYS:
            fails = modes.live_activation_failures(ctx.mode)
            return False, "live execution is denied in this build: " + "; ".join(fails), observed, "REPLAY|PAPER"
        try:
            modes.assert_gateway_allowed(ctx.mode, ctx.gateway)
        except modes.ModeViolation as exc:
            return False, str(exc), observed, "REPLAY+FAKE or PAPER+(FAKE|ALPACA_PAPER); legacy LEAN values are research-only"
        return True, "mode and gateway permit trading", observed, None

    @staticmethod
    def _chk_evidence(proposal: TradeProposal) -> CheckResult:
        ids = [e for e in proposal.evidence_ids if isinstance(e, str) and e.strip()]  # pyright: ignore[reportUnnecessaryIsInstance]
        if not ids or len(ids) != len(proposal.evidence_ids):
            return False, "TRADE requires non-empty evidence ids", str(len(ids)), ">=1"
        return True, f"{len(ids)} evidence id(s) present", str(len(ids)), ">=1"

    def _health_checks(self, c: _Checks, ctx: RiskContext) -> None:
        h = ctx.health
        c.add(
            CHECK_HEALTH_FEED, h.feed_ok,
            "market feed OK" if h.feed_ok else f"market feed not OK: {h.feed_detail or 'unknown'}",
        )
        c.add(
            CHECK_HEALTH_BROKER, h.broker_connected,
            "broker connected" if h.broker_connected else f"broker not connected: {h.broker_detail or 'unknown'}",
        )
        c.add(
            CHECK_HEALTH_RECONCILED, h.reconciled,
            "broker state reconciled" if h.reconciled else f"broker state not reconciled: {h.reconcile_detail or 'unknown'}",
        )
        c.add(CHECK_HEALTH_ENGINE, h.engine_ok, "engine OK" if h.engine_ok else f"engine not OK: {h.engine_detail or 'unknown'}")
        locked = h.entries_locked or bool(h.lock_reasons)  # any stated lock reason locks (fail closed)
        c.add(
            CHECK_HEALTH_ENTRIES, not locked,
            "entries unlocked" if not locked
            else "new entries locked: " + ("; ".join(h.lock_reasons) or "reason unknown"),
        )
        c.add(
            CHECK_HEALTH_STRATEGY, h.strategy_state_ok,
            "strategy state OK" if h.strategy_state_ok else "strategy state unknown/bad",
        )
        c.add(CHECK_HEALTH_PERMISSIONS, h.permissions_ok, "permissions OK" if h.permissions_ok else "permissions unknown/bad")

    @staticmethod
    def _chk_symbol_format(symbol: str) -> CheckResult:
        ok = bool(limits.US_EQUITY_SYMBOL.match(symbol))
        return ok, "listed U.S. equity symbol format" if ok else "symbol format not allowed (OTC/odd suffix)", symbol, None

    def _chk_allowlist(self, symbol: str) -> CheckResult:
        ok = symbol in self._policy.symbol_allowlist
        return ok, "symbol is allowlisted" if ok else "symbol is not on the policy allowlist", symbol, None

    def _chk_denylist(self, symbol: str) -> CheckResult:
        denied = symbol in self._policy.symbol_denylist or symbol in limits.HARD_DENYLIST
        detail = "leveraged/inverse/volatility product denied" if denied else "not denylisted"
        return not denied, detail, symbol, None

    def _chk_order_type(self, order_type: OrderType) -> CheckResult:
        ok = order_type == OrderType.LIMIT and order_type in self._policy.allowed_order_types
        return ok, "LIMIT order" if ok else f"order type {order_type} not allowed (LIMIT only)", str(order_type), "LIMIT"

    @staticmethod
    def _chk_tif(tif: TimeInForce) -> CheckResult:
        ok = tif == TimeInForce.DAY
        return ok, "DAY time in force" if ok else f"time in force {tif} not allowed", str(tif), "DAY"

    @staticmethod
    def _chk_whole_shares(qty: Decimal) -> CheckResult:
        ok = qty > ZERO and qty == floor_shares(qty)
        return ok, "whole positive shares" if ok else "quantity must be a positive whole number of shares", _f(qty), None

    @staticmethod
    def _chk_tick(limit: Decimal) -> CheckResult:
        ok = _on_tick(limit)
        tick = price_tick(limit)
        return ok, "limit on a valid tick" if ok else f"limit price not on a {tick} tick", _f(limit), _f(tick)

    @staticmethod
    def _chk_market_present(ctx: RiskContext, symbol: str) -> CheckResult:
        m = ctx.market
        if m is None:
            return False, "market snapshot missing; fail closed", None, symbol
        if m.symbol != symbol:
            return False, "market snapshot is for a different symbol", m.symbol, symbol
        return True, f"market snapshot {m.snapshot_id}", m.symbol, symbol

    @staticmethod
    def _chk_quote_present(ctx: RiskContext, symbol: str) -> CheckResult:
        m = ctx.market
        if m is None or m.quote is None:
            return False, "quote missing; fail closed", None, None
        if m.quote.symbol != symbol:
            return False, "quote is for a different symbol", m.quote.symbol, symbol
        return True, "quote present", m.quote.symbol, symbol

    @staticmethod
    def _chk_provenance(ctx: RiskContext) -> CheckResult:
        m = ctx.market
        if m is None:
            return False, "market snapshot missing", None, None
        label = modes.execution_label(ctx.mode, ctx.gateway)
        synthetic = m.is_simulated or m.feed == DataFeed.SYNTHETIC or (m.quote is not None and m.quote.is_synthetic)
        if label != ExecutionLabel.SIMULATED and synthetic:
            return False, f"synthetic/simulated market data cannot drive {label.value} orders", m.feed.value, "IEX|SIP"
        return True, f"{m.feed.value} data acceptable for {label.value} execution", m.feed.value, None

    def _chk_min_price(self, x: Exposure) -> CheckResult:
        if x.reference_price is None:
            return False, "reference price unavailable (no quote)", None, _f(self._policy.min_price)
        low = min(x.reference_price, x.limit_price)
        ok = low >= self._policy.min_price
        detail = _pick(ok, "price above floor", "price below minimum (penny/low-priced names excluded)")
        return ok, detail, _f(low), _f(self._policy.min_price)

    def _chk_session(self, ctx: RiskContext, constraint: SessionConstraint, now: datetime) -> CheckResult:
        if constraint != SessionConstraint.REGULAR_ONLY or self._policy.allow_extended_hours:
            return False, "only REGULAR_ONLY session orders are allowed", str(constraint), "REGULAR_ONLY"
        m = ctx.market
        if m is None:
            return False, "market session unknown (no snapshot)", None, "REGULAR"
        if m.session != MarketSession.REGULAR:
            return False, f"market session is {m.session.value}; extended/closed sessions rejected", m.session.value, "REGULAR"
        is_open, detail = market_hours.is_regular_session_open(now)
        if not is_open:
            return False, detail, now.isoformat(), "XNYS regular hours"
        return True, detail, m.session.value, "REGULAR"

    def _chk_limit_band(self, x: Exposure) -> CheckResult:
        if x.mid is None or x.mid <= ZERO:
            return False, "mid price unavailable", None, _f(self._policy.max_limit_deviation_bps)
        dev = abs(x.limit_price - x.mid) / x.mid * BPS
        ok = dev <= self._policy.max_limit_deviation_bps
        detail = _pick(ok, "limit within band around mid", "limit deviates too far from mid")
        return ok, detail, _fq(dev), _f(self._policy.max_limit_deviation_bps)

    def _chk_limit_touch(self, ctx: RiskContext, x: Exposure) -> CheckResult:
        quote = self._quote_for(ctx, x.symbol)
        if quote is None:
            return False, "quote unavailable", None, _f(limits.MAX_THROUGH_TOUCH_BPS)
        if x.side == Side.BUY:
            through = (x.limit_price - quote.ask) / quote.ask * BPS
            what = "BUY limit above ask"
        else:
            through = (quote.bid - x.limit_price) / quote.bid * BPS
            what = "SELL limit below bid"
        ok = through <= limits.MAX_THROUGH_TOUCH_BPS
        detail = _pick(ok, f"{what} by {_fq(max(through, ZERO))} bps", f"{what} by too much")
        return ok, detail, _fq(through), _f(limits.MAX_THROUGH_TOUCH_BPS)

    def _chk_spread(self, x: Exposure) -> CheckResult:
        if x.spread_bps is None:
            return False, "spread unavailable (no quote)", None, _f(self._policy.max_spread_bps)
        ok = x.spread_bps <= self._policy.max_spread_bps
        return ok, "spread acceptable" if ok else "spread too wide", _fq(x.spread_bps), _f(self._policy.max_spread_bps)

    def _chk_quote_age(self, ctx: RiskContext, symbol: str, now: datetime) -> CheckResult:
        quote = self._quote_for(ctx, symbol)
        limit = self._policy.max_quote_age_seconds
        if quote is None:
            return False, "quote missing; fail closed", None, _f(limit)
        ex_age = _age_seconds(now, quote.exchange_ts)
        rx_age = _age_seconds(now, quote.received_ts)
        av_age = _age_seconds(now, quote.available_at_ts)
        observed = f"exchange={_fq(ex_age)}s received={_fq(rx_age)}s"
        skew = -limits.MAX_CLOCK_SKEW_SECONDS
        if ex_age < skew or rx_age < skew:
            return False, "quote timestamp in the future (clock skew / bad data)", observed, _f(limit)
        if av_age < skew:
            return False, "quote not yet available at decision time (look-ahead)", observed, _f(limit)
        if ex_age > limit or rx_age > limit:
            return False, "quote is stale", observed, _f(limit)
        return True, "quote is fresh", observed, _f(limit)

    @staticmethod
    def _chk_data_asof(proposal: TradeProposal, ctx: RiskContext, now: datetime) -> CheckResult:
        limit = limits.MAX_PROPOSAL_DATA_AGE_SECONDS
        skew = -limits.MAX_CLOCK_SKEW_SECONDS
        p_age = _age_seconds(now, proposal.data_asof)
        c_age = _age_seconds(now, proposal.created_at)
        if p_age < skew or c_age < skew:
            return False, "proposal data_asof/created_at in the future", _fq(p_age), _f(limit)
        if p_age > limit:
            return False, "proposal data_asof is stale", _fq(p_age), _f(limit)
        m = ctx.market
        if m is None:
            return False, "market snapshot missing", None, _f(limit)
        m_age = _age_seconds(now, m.data_asof)
        if m_age < skew:
            return False, "market snapshot data_asof in the future", _fq(m_age), _f(limit)
        if m_age > limit:
            return False, "market snapshot data_asof is stale", _fq(m_age), _f(limit)
        return True, "proposal and snapshot data are current", f"proposal={_fq(p_age)}s snapshot={_fq(m_age)}s", _f(limit)

    @staticmethod
    def _chk_account_present(ctx: RiskContext) -> CheckResult:
        if ctx.account is None:
            return False, "account snapshot missing; fail closed", None, None
        return True, f"account snapshot {ctx.account.snapshot_id}", ctx.account.source, None

    @staticmethod
    def _chk_account_fresh(ctx: RiskContext, now: datetime) -> CheckResult:
        a = ctx.account
        limit = limits.MAX_ACCOUNT_SNAPSHOT_AGE_SECONDS
        if a is None:
            return False, "account snapshot missing; fail closed", None, _f(limit)
        age = _age_seconds(now, a.asof)
        if age < -limits.MAX_CLOCK_SKEW_SECONDS:
            return False, "account snapshot from the future", _fq(age), _f(limit)
        if age > limit:
            return False, "account snapshot is stale; fail closed", _fq(age), _f(limit)
        return True, "account snapshot is current", _fq(age), _f(limit)

    @staticmethod
    def _chk_account_status(ctx: RiskContext) -> CheckResult:
        a = ctx.account
        if a is None:
            return False, "account snapshot missing; fail closed", None, None
        problems: list[str] = []
        if a.trading_blocked:
            problems.append("trading_blocked")
        if a.account_blocked:
            problems.append("account_blocked")
        if a.status.strip().upper() != ACCOUNT_ACTIVE_STATUS:
            problems.append(f"status={a.status!r} (need ACTIVE)")
        if a.currency.upper() != "USD":
            problems.append(f"currency={a.currency}")
        if not a.is_paper:
            problems.append("account is not a paper account (live accounts are never used in this build)")
        if problems:
            return False, "account cannot trade: " + ", ".join(problems), a.status, ACCOUNT_ACTIVE_STATUS
        return True, "account active, paper, not blocked", a.status, ACCOUNT_ACTIVE_STATUS

    @staticmethod
    def _chk_positions(ctx: RiskContext, now: datetime) -> CheckResult:
        seen: set[str] = set()
        problems: list[str] = []
        for p in ctx.positions:
            if p.symbol in seen:
                problems.append(f"duplicate position rows for {p.symbol}")
            seen.add(p.symbol)
            if p.quantity < ZERO:
                problems.append(f"short position {p.symbol} {p.quantity} in a long-only account")
            age = _age_seconds(now, p.asof)
            if age > limits.MAX_ACCOUNT_SNAPSHOT_AGE_SECONDS or age < -limits.MAX_CLOCK_SKEW_SECONDS:
                problems.append(f"position snapshot for {p.symbol} is stale or from the future ({_fq(age)}s)")
        if problems:
            return False, "unexpected position state: " + "; ".join(problems), str(len(ctx.positions)), None
        return True, "positions consistent with long-only", str(len(ctx.positions)), None

    def _chk_long_only(self, x: Exposure) -> CheckResult:
        if not self._policy.long_only or self._policy.allow_short:
            return False, "policy is not long-only", None, None
        if x.side == Side.BUY:
            return True, "BUY (long entry/add)", None, None
        available = x.existing_qty - x.pending_sell_qty_symbol
        if x.existing_qty <= ZERO:
            return False, "SELL without an existing long position would be a short sale", _f(x.quantity), _f(ZERO)
        if x.quantity > available:
            return False, "SELL exceeds the existing long (net of pending sells); no shorting", _f(x.quantity), _f(available)
        return True, "SELL reduces an existing long", _f(x.quantity), _f(available)

    def _chk_no_margin(self, ctx: RiskContext, x: Exposure) -> CheckResult:
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", None, None
        a = ctx.account
        if a is None:
            return False, "account unknown; fail closed", None, None
        if self._policy.allow_margin:
            return False, "policy allows margin (forbidden)", None, None
        if x.pending_buy_notional is None:
            return False, "an open BUY order has no limit price; cash commitment unknown", None, None
        if a.cash <= ZERO:
            return False, "no positive settled cash (margin debit or empty account)", _f(a.cash), ">0"
        required = x.notional * self._policy.min_buying_power_buffer
        spendable = min(self._spendable_candidates(a)) - x.pending_buy_notional
        ok = required <= spendable
        detail = "covered by cash / non-marginable buying power" if ok else "would require margin borrowing"
        return ok, detail, _f(required), _f(spendable)

    def _chk_max_positions(self, ctx: RiskContext, x: Exposure) -> CheckResult:
        limit = self._policy.max_positions
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", None, str(limit)
        held = {p.symbol for p in ctx.positions if p.quantity != ZERO}
        pending = {o.symbol for o in ctx.open_orders if _is_open(o) and o.side == Side.BUY}
        prospective = held | pending | {x.symbol}
        ok = len(prospective) <= limit
        detail = _pick(ok, "position count within limit", "would exceed max positions (one position at a time)")
        return ok, detail, str(len(prospective)), str(limit)

    def _chk_order_notional(self, x: Exposure) -> CheckResult:
        limit = self._policy.max_order_notional
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", _f(x.notional), _f(limit)
        ok = x.notional <= limit
        return ok, "order notional within limit" if ok else "order notional exceeds limit", _f(x.notional), _f(limit)

    def _chk_position_notional(self, x: Exposure) -> CheckResult:
        limit = self._policy.max_position_notional
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", _f(x.post_trade_notional), _f(limit)
        if x.post_trade_notional is None:
            return False, "post-trade position notional unknown (no quote)", None, _f(limit)
        ok = x.post_trade_notional <= limit
        detail = _pick(ok, "post-trade position within limit", "post-trade position notional exceeds limit")
        return ok, detail, _f(x.post_trade_notional), _f(limit)

    def _chk_concentration(self, x: Exposure) -> CheckResult:
        limit = self._policy.max_concentration_pct
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", _fq(x.concentration), _f(limit)
        if x.concentration is None:
            return False, "concentration unknown (no quote or non-positive equity)", None, _f(limit)
        ok = x.concentration <= limit
        detail = _pick(ok, "concentration within limit", "post-trade concentration exceeds limit")
        return ok, detail, _fq(x.concentration), _f(limit)

    def _chk_daily_loss(self, x: Exposure) -> CheckResult:
        limit = -self._policy.max_daily_loss
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", _f(x.daily_pl_worst), _f(limit)
        if x.daily_pl_worst is None:
            return False, "daily P&L unknown (no account or start-of-day equity); fail closed", None, _f(limit)
        ok = x.daily_pl_worst > limit
        detail = _pick(ok, "daily loss within limit", "max daily loss reached; new entries blocked")
        return ok, detail, _f(x.daily_pl_worst), _f(limit)

    def _chk_daily_loss_budget(self, x: Exposure) -> CheckResult:
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", None, None
        if x.daily_pl_worst is None:
            return False, "daily P&L unknown", _f(x.planned_loss), None
        budget = self._policy.max_daily_loss + min(x.daily_pl_worst, ZERO)
        ok = x.planned_loss <= budget
        detail = _pick(ok, "planned loss fits remaining daily loss budget", "planned loss exceeds remaining daily loss budget")
        return ok, detail, _f(x.planned_loss), _f(budget)

    def _chk_planned_loss(self, x: Exposure) -> CheckResult:
        limit = self._policy.max_planned_loss_per_trade
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", _f(x.planned_loss), _f(limit)
        ok = x.planned_loss <= limit
        detail = f"recomputed qty*(limit-stop) with {x.stop_source} {_f(x.stop_price)}"
        return ok, detail if ok else f"planned loss exceeds per-trade limit ({detail})", _f(x.planned_loss), _f(limit)

    @staticmethod
    def _chk_proposal_arithmetic(x: Exposure) -> CheckResult:
        claimed = x.claimed_planned_loss
        observed = f"claimed={_f(claimed)} recomputed={_f(x.planned_loss)}"
        if claimed is None:
            return False, "proposal did not state max_planned_loss", observed, None
        if claimed < ZERO:
            return False, "proposal stated a negative max_planned_loss", observed, None
        if x.side == Side.SELL:
            return True, "exit order: claimed loss recorded only", observed, None
        under = x.planned_loss - claimed
        tolerance = max(limits.PLANNED_LOSS_ABS_TOLERANCE, x.planned_loss * limits.PLANNED_LOSS_TOLERANCE)
        if under > tolerance:
            detail = "proposal understates planned loss (model arithmetic not trusted)"
            return False, detail, observed, f"understatement<={_fq(tolerance)}"
        return True, "proposal arithmetic consistent with independent recompute", observed, f"understatement<={_fq(tolerance)}"

    def _chk_equity_floor(self, ctx: RiskContext, x: Exposure) -> CheckResult:
        a = ctx.account
        limit = self._policy.min_account_equity
        if a is None:
            return False, "account unknown", None, _f(limit)
        if x.side == Side.SELL:
            return True, "not applicable: SELL reduces an existing long", _f(a.equity), _f(limit)
        ok = a.equity >= limit
        return ok, "equity above floor" if ok else "account equity below floor", _f(a.equity), _f(limit)

    def _chk_pending(self, ctx: RiskContext) -> CheckResult:
        limit = self._policy.max_pending_orders
        n = sum(1 for o in ctx.open_orders if _is_open(o))
        ok = n < limit
        return ok, "pending orders within limit" if ok else "too many pending orders", str(n), str(limit)

    def _chk_rate_minute(self, ctx: RiskContext, now: datetime) -> CheckResult:
        limit = self._policy.max_orders_per_minute
        n = sum(1 for t in ctx.recent_submission_times if _age_seconds(now, t) < _SECONDS_PER_MINUTE)
        ok = n < limit
        return ok, "order rate within per-minute limit" if ok else "per-minute order rate limit reached", str(n), str(limit)

    def _chk_rate_day(self, ctx: RiskContext, now: datetime) -> CheckResult:
        limit = self._policy.max_orders_per_day
        today = market_hours.ny_date(now)
        n = sum(1 for t in ctx.recent_submission_times if t > now or market_hours.ny_date(t) == today)
        ok = n < limit
        return ok, "order rate within per-day limit" if ok else "per-day order limit reached", str(n), str(limit)

    @staticmethod
    def _chk_duplicate(proposal: TradeProposal, ctx: RiskContext, x: Exposure) -> CheckResult:
        variants = fingerprint_variants(x.symbol, x.side, x.quantity, x.limit_price, proposal.strategy_id)
        recent = set(ctx.recent_intent_fingerprints)
        hit = [fp for fp in variants if fp in recent]
        if hit:
            return False, "identical order authorized within the duplicate window", hit[0], "no repeat"
        twins = [
            o.client_order_id for o in ctx.open_orders
            if _is_open(o) and o.symbol == x.symbol and o.side == x.side and o.quantity == x.quantity
            and o.limit_price == x.limit_price
        ]
        if twins:
            return False, f"identical open order {twins[0]}", twins[0], "no repeat"
        return True, "no duplicate", variants[0], None

    @staticmethod
    def _chk_max_qty(x: Exposure) -> CheckResult:
        if x.max_allowable_qty is None:
            return False, f"max allowable quantity unknown: {x.binding_limit}", _f(x.quantity), None
        ok = x.quantity <= x.max_allowable_qty
        if ok:
            return True, f"quantity within max allowable (binding: {x.binding_limit})", _f(x.quantity), _f(x.max_allowable_qty)
        return (
            False,
            f"quantity {_f(x.quantity)} exceeds max allowable {_f(x.max_allowable_qty)} "
            f"(binding: {x.binding_limit}); never resized - resubmit a smaller order",
            _f(x.quantity), _f(x.max_allowable_qty),
        )

    # ------------------------------------------------------------------ authorize
    def authorize(self, proposal: TradeProposal, decision: RiskDecision, ctx: RiskContext) -> ApprovedOrderIntent:
        """Sign an ``ApprovedOrderIntent`` for an APPROVED decision this engine issued.

        Raises ``RiskAuthorizationError`` (or ``ModeViolation``) instead of ever signing on doubt.
        """
        if decision.verdict != RiskVerdict.APPROVED:
            raise RiskAuthorizationError("decision is not APPROVED; nothing to authorize")
        if proposal.action != DecisionAction.TRADE:
            raise RiskAuthorizationError(f"{NO_TRADE_REASON}: NO_TRADE proposals never yield an intent")
        if decision.proposal_id != proposal.proposal_id or decision.causal_id != proposal.causal_id:
            raise RiskAuthorizationError("decision does not belong to this proposal")
        if decision.policy_version != self._policy_version:
            raise RiskAuthorizationError("decision was made under a different policy version")
        if decision.mode != ctx.mode:
            raise RiskAuthorizationError("decision was made in a different mode")
        with self._lock:
            entry = self._issued.get(decision.decision_id)
        if entry is None or entry[0] != _decision_digest(decision):
            raise RiskAuthorizationError("decision was not issued by this risk engine (or was modified)")
        if entry[1] != _proposal_digest(proposal):
            raise RiskAuthorizationError("proposal content differs from the one that was evaluated")
        recheck = self._evaluate(proposal, ctx)
        if not recheck.approved:
            raise RiskAuthorizationError("re-evaluation rejected the proposal: " + "; ".join(recheck.reasons))
        with self._lock:
            already = self._intents.get(decision.decision_id)
        if already is not None:
            if self._now() >= already.expires_at:
                raise RiskAuthorizationError("the intent for this decision expired; evaluate a new proposal")
            return already  # idempotent: a retried authorize never mints a second order
        modes.assert_gateway_allowed(ctx.mode, ctx.gateway)
        if ctx.mode in LIVE_MODES:  # unreachable while assert_gateway_allowed refuses live gateways
            raise RiskAuthorizationError("live modes are compiled off")
        assert proposal.symbol is not None and proposal.side is not None
        assert proposal.quantity is not None and proposal.limit_price is not None

        now = self._now()
        requires_human = bool(self._policy.require_human_approval or ctx.mode == TradingMode.LIVE_APPROVAL)
        unsigned = ApprovedOrderIntent(
            intent_id=self._new_id("intent"),
            client_order_id=client_order_id(modes.client_order_tag(ctx.mode, ctx.gateway)),
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            causal_id=proposal.causal_id,
            mode=ctx.mode,
            gateway=ctx.gateway,
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=proposal.quantity,
            order_type=OrderType.LIMIT,
            limit_price=proposal.limit_price,
            time_in_force=TimeInForce.DAY,
            extended_hours=False,
            policy_version=self._policy_version,
            created_at=now,
            expires_at=now + timedelta(seconds=self._policy.intent_ttl_seconds),
            requires_human_approval=requires_human,
            human_approval_id=None,
            key_id=self._signer.key_id,
            signature_alg=SIGNATURE_ALG,
            signature="0" * 128,
        )
        signature = self._signer.sign(unsigned.signing_string())
        intent = ApprovedOrderIntent.model_validate({**unsigned.model_dump(), "signature": signature})
        if not self._signer.verifier.verify(intent.signing_string(), intent.signature):
            raise RiskAuthorizationError("self-verification of the intent signature failed")
        with self._lock:
            existing = self._intents.setdefault(decision.decision_id, intent)
        return existing

    # ------------------------------------------------------------------ final gate
    def final_gate(
        self,
        intent: ApprovedOrderIntent,
        ctx: RiskContext,
        verifier: IntentVerifier,
        active_policy_version: str,
        *,
        human_approval: HumanApproval | None = None,
        approval_verifier: Callable[[HumanApproval, ApprovedOrderIntent], bool] | None = None,
        active_halts: Sequence[object] = (),
        self_recorded: bool = True,
        strategy_id: str | None = None,
    ) -> list[str]:
        """Last check at the order boundary, immediately before gateway submission.

        Returns every failure (empty list == pass). Never raises. ``ctx`` must be freshly captured.
        ``self_recorded=True`` (default) means the ledger already holds this intent, so its own
        fingerprint may appear once in ``ctx.recent_intent_fingerprints`` and its own row in
        ``ctx.open_orders`` is ignored.
        """
        try:
            return self._final_gate(
                intent, ctx, verifier, active_policy_version, human_approval=human_approval,
                approval_verifier=approval_verifier, active_halts=active_halts, self_recorded=self_recorded,
                strategy_id=strategy_id,
            )
        except Exception as exc:  # fail closed
            return [f"final gate internal error ({type(exc).__name__}: {exc}); fail closed"]

    def _final_gate(
        self,
        intent: ApprovedOrderIntent,
        ctx: RiskContext,
        verifier: IntentVerifier,
        active_policy_version: str,
        *,
        human_approval: HumanApproval | None,
        approval_verifier: Callable[[HumanApproval, ApprovedOrderIntent], bool] | None,
        active_halts: Sequence[object],
        self_recorded: bool,
        strategy_id: str | None,
    ) -> list[str]:
        fails: list[str] = []
        now = self._now()
        pol = self._policy

        # -- inputs: check the validated copies (smuggled/coerced values fail closed, see _revalidate_one)
        intent, intent_problem = _revalidate_one(intent, "intent")
        ctx, ctx_problem = _revalidate_one(ctx, "final-gate context")
        for problem in (intent_problem, ctx_problem):
            if problem is not None:
                fails.append(f"inputs: {problem}")

        # -- authenticity
        if intent.signature_alg != SIGNATURE_ALG:
            fails.append(f"signature: unsupported algorithm {intent.signature_alg!r}")
        if verifier.key_id != self._signer.key_id:
            fails.append("signature: supplied verifier is not this risk engine's public key")
        if intent.key_id != verifier.key_id or intent.key_id != self._signer.key_id:
            fails.append(f"signature: key id {intent.key_id} does not match the risk engine key {self._signer.key_id}")
        try:
            message = intent.signing_string()
        except ValueError as exc:
            fails.append(f"signature: unsignable intent fields ({exc})")
        else:
            if not verifier.verify(message, intent.signature) or not self._signer.verifier.verify(message, intent.signature):
                fails.append("signature: invalid (intent tampered or signed by another key)")

        # -- time
        created = _age_seconds(now, intent.created_at)
        if created < -limits.MAX_CLOCK_SKEW_SECONDS:
            fails.append("expiry: intent created in the future")
        if now >= intent.expires_at or ctx.now >= intent.expires_at:
            fails.append(f"expiry: intent expired at {intent.expires_at.isoformat()}")
        ttl = _age_seconds(intent.expires_at, intent.created_at)
        max_ttl = Decimal(min(pol.intent_ttl_seconds, limits.HARD_MAX_INTENT_TTL_SECONDS))
        if ttl > max_ttl:
            fails.append(f"expiry: intent TTL {_f(ttl)}s exceeds {max_ttl}s")
        ctx_age = _age_seconds(now, ctx.now)
        if ctx_age > limits.MAX_CONTEXT_AGE_SECONDS or ctx_age < -limits.MAX_CLOCK_SKEW_SECONDS:
            fails.append(f"context: final-gate context not fresh (age {_fq(ctx_age)}s)")

        # -- mode / gateway ownership
        if intent.mode != ctx.mode or intent.gateway != ctx.gateway:
            fails.append(
                f"mode: intent {intent.mode.value}/{intent.gateway.value} != runtime {ctx.mode.value}/{ctx.gateway.value}"
            )
        if intent.mode in LIVE_MODES or intent.gateway in modes.LIVE_GATEWAYS or ctx.mode in LIVE_MODES:
            denied = "; ".join(modes.live_activation_failures(intent.mode))
            fails.append("mode: live execution is denied in this build: " + denied)
        if intent.mode == TradingMode.RESEARCH_ONLY:
            fails.append("mode: RESEARCH_ONLY never trades")
        try:
            modes.assert_gateway_allowed(intent.mode, intent.gateway)
        except modes.ModeViolation as exc:
            fails.append(f"mode: {exc}")
        try:
            tag = modes.client_order_tag(intent.mode, intent.gateway)
            if not intent.client_order_id.startswith(f"lat-{tag}-"):
                fails.append(f"mode: client order id is not tagged '{tag}' for {intent.mode.value}/{intent.gateway.value}")
        except Exception as exc:
            fails.append(f"mode: cannot derive client order tag ({exc})")

        # -- policy
        if intent.policy_version != active_policy_version:
            fails.append(f"policy: intent policy {intent.policy_version} is no longer active ({active_policy_version})")
        if self._policy_version != active_policy_version:
            fails.append("policy: risk engine holds a stale policy; rebuild the engine")
        if self._policy_violations:
            fails.append("policy: active policy violates hard limits")

        # -- health, halts, entries lock
        h = ctx.health
        for ok, label in (
            (h.feed_ok, f"feed not OK ({h.feed_detail or 'unknown'})"),
            (h.broker_connected, f"broker not connected ({h.broker_detail or 'unknown'})"),
            (h.reconciled, f"not reconciled ({h.reconcile_detail or 'unknown'})"),
            (h.engine_ok, f"engine not OK ({h.engine_detail or 'unknown'})"),
            (not (h.entries_locked or h.lock_reasons), "entries locked: " + ("; ".join(h.lock_reasons) or "reason unknown")),
            (h.strategy_state_ok, "strategy state not OK"),
            (h.permissions_ok, "permissions not OK"),
        ):
            if not ok:
                fails.append(f"health: {label}")
        if active_halts:
            fails.append(f"halt: {len(active_halts)} active trading halt(s)")

        # -- account
        a = ctx.account
        if a is None:
            fails.append("account: snapshot missing; fail closed")
        else:
            acct_age = _age_seconds(now, a.asof)
            if acct_age > limits.MAX_ACCOUNT_SNAPSHOT_AGE_SECONDS or acct_age < -limits.MAX_CLOCK_SKEW_SECONDS:
                fails.append("account: snapshot stale")
            if a.trading_blocked or a.account_blocked or a.status.strip().upper() != ACCOUNT_ACTIVE_STATUS:
                fails.append("account: blocked or not ACTIVE")
            if not a.is_paper:
                fails.append("account: not a paper account")

        # -- order shape (defence in depth; the intent is signed but the policy may have changed)
        if intent.order_type != OrderType.LIMIT or intent.order_type not in pol.allowed_order_types:
            fails.append("order: LIMIT only")
        if intent.time_in_force != TimeInForce.DAY:
            fails.append("order: DAY only")
        if intent.extended_hours:
            fails.append("order: extended hours not permitted")
        if intent.quantity != floor_shares(intent.quantity):
            fails.append("order: whole shares only")
        if not _on_tick(intent.limit_price):
            fails.append("order: limit price not on a valid tick")
        if not limits.US_EQUITY_SYMBOL.match(intent.symbol) or intent.symbol not in pol.symbol_allowlist:
            fails.append("order: symbol not allowlisted")
        if intent.symbol in pol.symbol_denylist or intent.symbol in limits.HARD_DENYLIST:
            fails.append("order: symbol denylisted (leveraged/inverse)")
        is_open, detail = market_hours.is_regular_session_open(now)
        if not is_open:
            fails.append(f"session: {detail}")
        others = [o for o in ctx.open_orders if _is_open(o) and o.client_order_id != intent.client_order_id]
        if intent.side == Side.BUY:
            if intent.quantity * intent.limit_price > pol.max_order_notional:
                fails.append("order: notional exceeds max_order_notional")
            fails.extend(self._entry_money_failures(intent, ctx, others))
        else:
            held = sum((p.quantity for p in ctx.positions if p.symbol == intent.symbol), ZERO)
            pending_sells = sum(
                (_remaining(o) for o in others if o.symbol == intent.symbol and o.side == Side.SELL), ZERO
            )
            if intent.quantity > held - pending_sells:
                fails.append("long-only: SELL exceeds the existing long position (no shorting)")
        if len(others) >= pol.max_pending_orders:
            fails.append(f"orders: {len(others)} other pending order(s) (max {pol.max_pending_orders})")

        # -- duplicates
        variants = fingerprint_variants(
            intent.symbol, intent.side, intent.quantity, intent.limit_price, strategy_id or ANY_STRATEGY
        )
        allowed = 1 if self_recorded else 0
        recent = list(ctx.recent_intent_fingerprints)
        if any(recent.count(fp) > allowed for fp in variants):
            fails.append("duplicate: identical order already authorized within the duplicate window")
        twins = [
            o.client_order_id for o in others
            if o.symbol == intent.symbol and o.side == intent.side and o.quantity == intent.quantity
            and o.limit_price == intent.limit_price
        ]
        if twins:
            fails.append(f"duplicate: identical open order {twins[0]}")

        # -- human approval
        needs_human = intent.requires_human_approval or pol.require_human_approval or intent.mode == TradingMode.LIVE_APPROVAL
        if needs_human:
            fails.extend(self._human_approval_failures(intent, now, human_approval, approval_verifier))
        return list(dict.fromkeys(fails))

    def _entry_money_failures(
        self, intent: ApprovedOrderIntent, ctx: RiskContext, others: Sequence[BrokerOrder]
    ) -> list[str]:
        """No-margin cash cover and the daily-loss stop, re-checked for a BUY at the order boundary (both can
        change between authorize and submission). Same formulas as evaluate; unknown inputs fail closed."""
        a = ctx.account
        if a is None:
            return []  # already reported as "account: snapshot missing"
        out: list[str] = []
        pol = self._policy
        pending: Decimal | None = ZERO
        for o in others:
            if o.side != Side.BUY:
                continue
            if o.limit_price is None or pending is None:
                pending = None
                continue
            pending += _remaining(o) * o.limit_price
        if pending is None:
            out.append("no-margin: an open BUY order has no limit price; cash commitment unknown")
        else:
            required = intent.quantity * intent.limit_price * pol.min_buying_power_buffer
            spendable = min(self._spendable_candidates(a)) - pending
            if a.cash <= ZERO or required > spendable:
                out.append(f"no-margin: BUY needs {_f(required)} but only {_f(spendable)} cash-like buying power")
        baseline = ctx.start_of_day_equity if ctx.start_of_day_equity is not None else a.last_equity
        if baseline is None:
            out.append("daily-loss: start-of-day equity unknown; fail closed")
        else:
            worst = min(ctx.daily_realized_pl + ctx.daily_unrealized_pl, a.equity - baseline)
            if worst <= -pol.max_daily_loss:
                out.append(f"daily-loss: {_f(worst)} reached the max daily loss {_f(pol.max_daily_loss)}")
        return out

    @staticmethod
    def _human_approval_failures(
        intent: ApprovedOrderIntent,
        now: datetime,
        approval: HumanApproval | None,
        approval_verifier: Callable[[HumanApproval, ApprovedOrderIntent], bool] | None,
    ) -> list[str]:
        if not intent.requires_human_approval:
            return ["human approval: required by policy/mode but the intent was not signed as requiring it"]
        if approval is None:
            return ["human approval: required but absent"]
        out: list[str] = []
        if approval.intent_id != intent.intent_id:
            out.append("human approval: bound to a different intent")
        if intent.human_approval_id is not None and approval.approval_id != intent.human_approval_id:
            out.append("human approval: id does not match the intent")
        if _age_seconds(now, approval.confirmed_at) < -limits.MAX_CLOCK_SKEW_SECONDS:
            out.append("human approval: confirmed in the future")
        if now >= approval.expires_at:
            out.append("human approval: expired")
        if approval_verifier is None:
            out.append("human approval: no verifier supplied; cannot authenticate the approval")
        else:
            try:
                ok = approval_verifier(approval, intent)
            except Exception as exc:
                out.append(f"human approval: verification error ({type(exc).__name__})")
            else:
                if not ok:
                    out.append("human approval: signature invalid")
        return out
