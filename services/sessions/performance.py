"""Portfolio performance from the operational ledger (``GET /api/v1/performance``). Decimal only, nothing invented.

Sources (all local SQLite rows written by other services):

* portfolio value / cash: the latest ``account_snapshots`` row at the cut-off time (broker-authoritative, written by
  reconciliation). ``null`` when no snapshot exists yet.
* realized P&L: every recorded fill replayed FIFO per symbol (long-only: BUY opens lots, SELL closes the oldest
  lots first). A SELL beyond the held quantity is an anomaly and books no P&L for the excess.
* unrealized P&L: the position snapshots of the latest successful (non-ERROR) reconciliation at the cut-off.
* daily P&L: realized on the current America/New_York trading date plus current unrealized (the same measure the
  risk engine's daily-loss limit uses).
* round trips: a symbol's position going flat -> long -> flat; win rate = round trips with P&L > 0 / closed ones.
* drawdowns: from the account equity curve (peak-to-trough fraction 0..1).
* API cost: ``SUM(model_runs.estimated_cost_usd)``.

Timestamps are compared after parsing (never as strings). ``since``/``until`` scope realized P&L, round trips,
the equity curve and API cost to a window (used for session reports); point-in-time state (value, cash,
positions, unrealized) is taken at ``until`` (or ``now``).
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Final

from packages.contracts.enums import TERMINAL_ORDER_STATES
from packages.core.db import Database
from packages.core.money import D

from .calendar import et_date, iso, parse_utc, require_utc

ZERO: Final = Decimal(0)
RATIO_Q: Final = Decimal("0.0001")
MAX_CURVE_POINTS: Final = 1000
TERMINAL_STATES: Final[frozenset[str]] = frozenset(s.value for s in TERMINAL_ORDER_STATES)
MODE_LABELS: Final[frozenset[str]] = frozenset({"BACKTEST", "PAPER", "LIVE", "SIMULATED"})
PERFORMANCE_KEYS: Final[tuple[str, ...]] = (
    "mode_label", "portfolio_value", "cash", "daily_pnl", "total_pnl", "realized_pnl", "unrealized_pnl", "win_rate",
    "round_trips", "max_drawdown", "current_drawdown", "active_positions", "open_orders", "api_cost_usd",
    "equity_curve", "asof", "is_simulated",
)


# ------------------------------------------------------------------------------------------------ strict parsing
def dec(value: Any) -> Decimal:
    """Exact Decimal from a ledger value. Binary floats (a corrupt REAL in a TEXT money column, a JSON float) are
    rejected loudly instead of being rounded into the books."""
    if isinstance(value, float):
        raise ValueError("binary float found in a money/quantity/price field; refusing to compute P&L from it")
    return D(value)


def _no_float(text: str) -> Decimal:
    raise ValueError(f"binary float {text} in a ledger JSON payload; money values must be decimal strings")


def loads(text: str) -> Any:
    return json.loads(text, parse_float=_no_float)


# ------------------------------------------------------------------------------------------------ FIFO
@dataclass(frozen=True)
class FillRow:
    fill_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    ts: datetime


@dataclass
class Lot:
    quantity: Decimal
    price: Decimal
    client_order_id: str


@dataclass(frozen=True)
class RoundTrip:
    symbol: str
    opened_at: datetime
    closed_at: datetime
    pnl: Decimal


@dataclass
class FifoResult:
    realized_by_fill: dict[str, Decimal] = field(default_factory=dict[str, Decimal])
    round_trips: list[RoundTrip] = field(default_factory=list[RoundTrip])
    open_lots: dict[str, list[Lot]] = field(default_factory=dict[str, list[Lot]])
    anomalies: list[str] = field(default_factory=list[str])


def fifo(fills: Iterable[FillRow]) -> FifoResult:
    """Replay fills (already in execution order) FIFO per symbol."""
    lots: dict[str, deque[Lot]] = defaultdict(deque)
    trip_open: dict[str, datetime] = {}
    trip_pnl: dict[str, Decimal] = defaultdict(lambda: ZERO)
    out = FifoResult()
    for f in fills:
        if f.quantity <= 0 or f.price <= 0:
            out.anomalies.append(f"fill {f.fill_id}: non-positive quantity/price ignored")
            continue
        book = lots[f.symbol]
        if f.side == "BUY":
            if not book:
                trip_open[f.symbol] = f.ts
                trip_pnl[f.symbol] = ZERO
            book.append(Lot(f.quantity, f.price, f.client_order_id))
            continue
        if f.side != "SELL":
            out.anomalies.append(f"fill {f.fill_id}: unknown side {f.side!r} ignored")
            continue
        remaining, realized = f.quantity, ZERO
        while remaining > 0 and book:
            lot = book[0]
            take = min(lot.quantity, remaining)
            realized += take * (f.price - lot.price)
            lot.quantity -= take
            remaining -= take
            if lot.quantity == 0:
                book.popleft()
        if remaining > 0:
            out.anomalies.append(f"fill {f.fill_id}: SELL {remaining} {f.symbol} exceeds the long position "
                                 "(long-only violated); no P&L booked for the excess")
        out.realized_by_fill[f.fill_id] = realized
        if f.symbol in trip_open:
            trip_pnl[f.symbol] += realized
            if not book:
                out.round_trips.append(RoundTrip(f.symbol, trip_open.pop(f.symbol), f.ts, trip_pnl.pop(f.symbol)))
    out.open_lots = {s: list(b) for s, b in lots.items() if b}
    return out


def load_fills(db: Database, *, until: datetime | None = None, gateway: str | None = None) -> list[FillRow]:
    rows = db.query("SELECT rowid AS rid, fill_id, client_order_id, symbol, side, quantity, price, ts FROM fills")
    allowed = None if gateway is None else {r["client_order_id"] for r in db.query(
        "SELECT client_order_id FROM orders WHERE gateway=?", (gateway,))}
    parsed: list[tuple[datetime, int, FillRow]] = []
    for r in rows:
        if allowed is not None and r["client_order_id"] not in allowed:
            continue
        ts = parse_utc(r["ts"])
        if ts is None or (until is not None and ts > until):
            continue
        parsed.append((ts, int(r["rid"]), FillRow(str(r["fill_id"]), str(r["client_order_id"]), str(r["symbol"]),
                                                  str(r["side"]).upper(), dec(r["quantity"]), dec(r["price"]), ts)))
    parsed.sort(key=lambda t: (t[0], t[1]))
    return [p[2] for p in parsed]


# ------------------------------------------------------------------------------------------------ helpers
def mode_label_for(label: str, is_simulated: bool, *, backtest: bool = False) -> str:
    """SIMULATED / PAPER / LIVE from the execution label; BACKTEST only for backtest result views."""
    if backtest:
        return "BACKTEST"
    value = str(getattr(label, "value", label)).upper()
    if value not in ("SIMULATED", "PAPER", "LIVE"):
        raise ValueError(f"unknown execution label {label!r}")
    if value == "LIVE":
        if is_simulated:
            raise ValueError("inconsistent labels: LIVE execution cannot be simulated")
        return "LIVE"
    return "SIMULATED" if (is_simulated or value == "SIMULATED") else "PAPER"


def _ratio(x: Decimal) -> Decimal:
    return x.quantize(RATIO_Q, rounding=ROUND_HALF_EVEN)


def _fmt(x: Decimal | None) -> str | None:
    return None if x is None else format(x, "f")


def _in_window(ts: datetime, since: datetime | None, until: datetime) -> bool:
    return (since is None or ts >= since) and ts <= until


def drawdowns(equity: list[Decimal]) -> tuple[Decimal, Decimal]:
    """(max drawdown, current drawdown) as fractions of the running peak (0 when the curve is empty/flat)."""
    peak: Decimal | None = None
    max_dd = ZERO
    for e in equity:
        if peak is None or e > peak:
            peak = e
        if peak is not None and peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    current = ZERO
    if equity and peak is not None and peak > 0:
        current = (peak - equity[-1]) / peak
    return _ratio(max(max_dd, ZERO)), _ratio(max(current, ZERO))


def _downsample(points: list[dict[str, Any]], n: int = MAX_CURVE_POINTS) -> list[dict[str, Any]]:
    if len(points) <= n:
        return points
    step = (len(points) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)} | {len(points) - 1})
    return [points[i] for i in idx]


def latest_positions(db: Database, cut: datetime, gateway: str | None = None) -> list[dict[str, Any]] | None:
    """Position snapshots of the latest successful reconciliation at ``cut`` (None if none exists)."""
    runs = db.query("SELECT run_id, finished_at FROM reconciliation_runs WHERE status != 'ERROR'"
                    " AND (? IS NULL OR gateway=?)", (gateway, gateway))
    best: tuple[datetime, str] | None = None
    for r in runs:
        fin = parse_utc(r["finished_at"])
        if fin is None or fin > cut:
            continue
        key = (fin, str(r["run_id"]))
        if best is None or key > best:
            best = key
    if best is None:
        return None
    rows = db.query("SELECT payload_json FROM position_snapshots WHERE snapshot_group=? ORDER BY seq", (best[1],))
    return [loads(r["payload_json"]) for r in rows]


def _unrealized(positions: list[dict[str, Any]]) -> Decimal:
    total = ZERO
    for p in positions:
        upl = p.get("unrealized_pl")
        if upl is not None:
            total += dec(upl)
        elif p.get("market_value") is not None and p.get("cost_basis") is not None:
            total += dec(p["market_value"]) - dec(p["cost_basis"])
    return total


def position_marks(positions: list[dict[str, Any]]) -> dict[str, Decimal]:
    """Latest broker mark per symbol (current_price, else market_value / quantity)."""
    marks: dict[str, Decimal] = {}
    for p in positions:
        sym = str(p.get("symbol"))
        if p.get("current_price") is not None:
            marks[sym] = dec(p["current_price"])
        elif p.get("market_value") is not None and p.get("quantity") not in (None, "0", 0):
            qty = dec(p["quantity"])
            if qty != 0:
                marks[sym] = dec(p["market_value"]) / qty
    return marks


def _account_rows(db: Database, source: str | None = None) -> list[tuple[datetime, str, Decimal]]:
    out: list[tuple[datetime, str, Decimal]] = []
    for r in db.query("SELECT snapshot_id, asof, equity FROM account_snapshots WHERE (? IS NULL OR source=?)", (source, source)):
        ts = parse_utc(r["asof"])
        if ts is not None:
            out.append((ts, str(r["snapshot_id"]), dec(r["equity"])))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def api_cost(db: Database, *, since: datetime | None = None, until: datetime | None = None,
             causal_ids: Iterable[str] | None = None) -> Decimal:
    total = ZERO
    wanted = set(causal_ids) if causal_ids is not None else None
    for r in db.query("SELECT causal_id, started_at, estimated_cost_usd FROM model_runs"):
        if wanted is not None and r["causal_id"] not in wanted:
            continue
        ts = parse_utc(r["started_at"])
        if ts is None or (since is not None and ts < since) or (until is not None and ts > until):
            continue
        total += dec(r["estimated_cost_usd"])
    return total


def open_order_count(db: Database, gateway: str | None = None) -> int:
    rows = db.query("SELECT state FROM orders WHERE (? IS NULL OR gateway=?)", (gateway, gateway))
    return sum(1 for r in rows if str(r["state"]) not in TERMINAL_STATES)


# ------------------------------------------------------------------------------------------------ main
def compute_performance(db: Database, *, now: datetime, label: str, is_simulated: bool,
                        since: datetime | None = None, until: datetime | None = None,
                        backtest: bool = False, gateway: str | None = None) -> dict[str, Any]:
    """Shape of ``GET /api/v1/performance`` (decimal strings; ratios 0..1 with 4 decimals)."""
    now = require_utc(now)
    cut = require_utc(until) if until is not None else now
    since = require_utc(since) if since is not None else None
    mode_label = mode_label_for(label, is_simulated, backtest=backtest)

    fills = load_fills(db, until=cut, gateway=gateway)
    result = fifo(fills)
    today = et_date(cut)
    realized = ZERO
    daily_realized = ZERO
    for f in fills:
        r = result.realized_by_fill.get(f.fill_id)
        if r is None:
            continue
        if _in_window(f.ts, since, cut):
            realized += r
        if et_date(f.ts) == today:
            daily_realized += r
    trips = [t for t in result.round_trips if _in_window(t.closed_at, since, cut)]
    wins = sum(1 for t in trips if t.pnl > 0)

    positions = latest_positions(db, cut, gateway=gateway)
    unrealized = _unrealized(positions) if positions is not None else ZERO
    if positions is not None:
        active = sum(1 for p in positions if dec(p.get("quantity") or "0") != 0)
    else:
        active = sum(1 for r in db.query("SELECT quantity FROM ledger_positions") if dec(r["quantity"]) != 0)

    source = {"ALPACA_PAPER": "ALPACA_PAPER", "FAKE": "FAKE_BROKER"}.get(gateway or "")
    accounts = [a for a in _account_rows(db, source) if a[0] <= cut]
    portfolio_value: Decimal | None = None
    cash: Decimal | None = None
    if accounts:
        latest = db.query_one("SELECT payload_json FROM account_snapshots WHERE snapshot_id=?", (accounts[-1][1],))
        payload = loads(latest["payload_json"]) if latest else {}
        portfolio_value = accounts[-1][2]
        cash = dec(payload["cash"]) if payload.get("cash") is not None else None
    curve_rows = [a for a in accounts if _in_window(a[0], since, cut)]
    max_dd, cur_dd = drawdowns([a[2] for a in curve_rows])
    curve = _downsample([{"time": int(a[0].timestamp()), "equity": format(a[2], "f")} for a in curve_rows])

    out: dict[str, Any] = {
        "mode_label": mode_label,
        "portfolio_value": _fmt(portfolio_value),
        "cash": _fmt(cash),
        "daily_pnl": format(daily_realized + unrealized, "f"),
        "total_pnl": format(realized + unrealized, "f"),
        "realized_pnl": format(realized, "f"),
        "unrealized_pnl": format(unrealized, "f"),
        "win_rate": _fmt(_ratio(Decimal(wins) / Decimal(len(trips)))) if trips else None,
        "round_trips": len(trips),
        "max_drawdown": format(max_dd, "f"),
        "current_drawdown": format(cur_dd, "f"),
        "active_positions": active,
        "open_orders": open_order_count(db, gateway),
        "api_cost_usd": format(api_cost(db, since=since, until=cut), "f"),
        "equity_curve": curve,
        "asof": iso(cut),
        "is_simulated": bool(is_simulated or mode_label == "SIMULATED"),
    }
    return out
