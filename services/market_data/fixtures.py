"""Deterministic SYNTHETIC market-data fixture (explicitly labeled: *not market data*).

``ensure_synthetic_fixture`` generates seeded minute bars for the REGULAR session of real XNYS trading days
(exchange_calendars, holidays and early closes included), daily bars aggregated from them, and top-of-book
quotes every ``quote_interval_s`` seconds (default 5 s, i.e. 12 per minute, so a point-in-time quote is never
older than the risk policy's 15 s freshness limit during a simulated session). Prices follow a seeded
geometric random walk with U-shaped intraday volatility, overnight gaps, U-shaped intraday volume with
open/close spikes, and plausible penny spreads (wider in the first minutes). Bars are aggregated from the same
5-second trade path as the quotes, so OHLC, VWAP and quotes are mutually consistent.

Determinism: the same parameters (seed, symbols, sessions, end date, intervals, latencies) always produce
identical Parquet bytes and the same content-derived ``dataset_version``. ``end_date`` defaults to the fixed
:data:`DEFAULT_END_DATE` (not "today") so the fixture is reproducible across days.

Validation: bars go through :func:`services.market_data.quality.validate_bars` (the real validator; a clean
fixture has zero rejects and zero gaps). The ~370k clean quotes are checked with equivalent vectorized
invariants (positive, uncrossed, unlocked, strictly increasing, inside the regular session) for speed.
``inject_defects=True`` instead builds per-record dicts with known defects (ground truth in the manifest's
``defects_injected``), runs both validators record by record, stores accepted rows (rejected ones only in the raw
Parquet), and flags the dataset so default loads exclude it. Use small ``symbols``/``sessions`` for that mode.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from packages.contracts.enums import Adjustment, DataFeed, MarketSession
from packages.contracts.models import Bar
from packages.core.canonical import canonical_json

from .calendar import MarketCalendar, get_calendar
from .common import PENDING_DATASET_VERSION, SIMULATED_LIMITATION, iso
from .manifest import SERIES_SNAPSHOT, DatasetSpec
from .quality import QualityPolicy, ValidationReport, validate_bars, validate_quotes
from .store import BAR_SCHEMA, QUOTE_SCHEMA, MarketDataStore

log = logging.getLogger("lat.market_data.fixtures")

GENERATOR = "services.market_data.fixtures"
GENERATOR_VERSION = "1.0.0"
DEFAULT_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT")
DEFAULT_SEED = 20261105
DEFAULT_SESSIONS = 20
DEFAULT_END_DATE = date(2026, 11, 4)  # last XNYS session before the build-spec date; fixed for reproducibility
DEFAULT_QUOTE_INTERVAL_S = 5
DEFAULT_QUOTE_LATENCY_MS = 50
DEFAULT_BAR_LATENCY_MS = 1000
DEFAULT_DAILY_LATENCY_S = 60
SYNTHETIC_LABEL = "SYNTHETIC — not market data"

FIXTURE_LIMITATIONS: tuple[str, ...] = (
    SYNTHETIC_LABEL,
    SIMULATED_LIMITATION,
    "Seeded geometric random walk; price levels, spreads, sizes and volumes are modelled, not observed",
    "Regular session only on real XNYS trading days (holidays/early closes from exchange_calendars)",
    "No corporate actions, halts, auctions, news or market impact are modelled",
)


@dataclass(frozen=True)
class SymbolProfile:
    price: float
    annual_vol: float
    daily_volume: int
    spread_cents: int


PROFILES: dict[str, SymbolProfile] = {
    "SPY": SymbolProfile(680.0, 0.15, 60_000_000, 1),
    "QQQ": SymbolProfile(600.0, 0.20, 40_000_000, 1),
    "AAPL": SymbolProfile(260.0, 0.25, 50_000_000, 1),
    "MSFT": SymbolProfile(520.0, 0.22, 20_000_000, 2),
    "IWM": SymbolProfile(240.0, 0.20, 30_000_000, 1),
    "DIA": SymbolProfile(470.0, 0.15, 3_000_000, 2),
    "NVDA": SymbolProfile(190.0, 0.45, 180_000_000, 1),
}


def _seed_int(seed: int, symbol: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{symbol}".encode()).digest()[:8], "big")


def _profile(symbol: str) -> SymbolProfile:
    if symbol in PROFILES:
        return PROFILES[symbol]
    h = _seed_int(0, symbol)
    return SymbolProfile(price=20.0 + (h % 28000) / 100.0, annual_vol=0.30, daily_volume=5_000_000, spread_cents=2)


def source_id_for(seed: int) -> str:
    return f"synthetic:{GENERATOR}@{GENERATOR_VERSION}:seed={seed}"


# ------------------------------------------------------------------------------------------ generation
@dataclass
class _SymbolArrays:
    symbol: str
    q_ts: np.ndarray  # int64 epoch us
    bid_c: np.ndarray
    ask_c: np.ndarray
    bid_sz: np.ndarray
    ask_sz: np.ndarray
    bar_start: np.ndarray
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    vol: np.ndarray
    vwap_e4: np.ndarray
    trades: np.ndarray
    day_index: np.ndarray  # session index per bar


def _us(dt: datetime) -> int:
    return round(dt.timestamp() * 1_000_000)


def _generate_symbol(symbol: str, sessions: Sequence[date], cal: MarketCalendar, seed: int,
                     quote_interval_s: int) -> _SymbolArrays:
    prof = _profile(symbol)
    rng = np.random.default_rng(_seed_int(seed, symbol))
    per_min = 60 // quote_interval_s
    sigma_day = prof.annual_vol / np.sqrt(252.0)
    log_p = float(np.log(prof.price))
    out: dict[str, list[np.ndarray]] = {k: [] for k in (
        "q_ts", "bid_c", "ask_c", "bid_sz", "ask_sz", "bar_start", "o", "h", "lo", "c", "vol", "vwap_e4", "trades",
        "day_index")}
    for di, d in enumerate(sessions):
        open_us = _us(cal.session_open(d))
        n_min = cal.regular_minutes_count(d)
        n = n_min * per_min
        if di > 0:
            log_p += rng.normal(0.0, sigma_day * 0.35)
        t = np.arange(n) / max(n - 1, 1)
        mult = 0.55 + 1.6 * (2 * t - 1) ** 2
        mult /= np.sqrt(np.mean(mult**2))
        sigma_step = sigma_day / np.sqrt(n) * mult
        steps = rng.standard_t(df=5, size=n) * np.sqrt(3.0 / 5.0) * sigma_step
        path = log_p + np.cumsum(steps)
        log_p = float(path[-1])
        mid_c = np.exp(path) * 100.0
        minute = np.arange(n) // per_min
        spread = np.full(n, prof.spread_cents, dtype=np.int64)
        spread[minute < 5] *= 3
        spread[minute >= n_min - 1] *= 2
        spread += (rng.random(n) < 0.05).astype(np.int64)
        bid_c = np.floor(mid_c - spread / 2.0).astype(np.int64)
        bid_c = np.maximum(bid_c, 1)
        ask_c = bid_c + spread
        trade_c = bid_c + rng.integers(0, spread + 1)
        q_ts = open_us + np.arange(n, dtype=np.int64) * quote_interval_s * 1_000_000
        out["q_ts"].append(q_ts)
        out["bid_c"].append(bid_c)
        out["ask_c"].append(ask_c)
        out["bid_sz"].append(rng.integers(1, 30, size=n) * 100)
        out["ask_sz"].append(rng.integers(1, 30, size=n) * 100)
        tc = trade_c.reshape(n_min, per_min)
        out["o"].append(tc[:, 0])
        out["c"].append(tc[:, -1])
        out["h"].append(tc.max(axis=1))
        out["lo"].append(tc.min(axis=1))
        out["vwap_e4"].append(np.rint(tc.mean(axis=1) * 100).astype(np.int64))
        m = np.arange(n_min) / max(n_min - 1, 1)
        w = 1.0 + 2.2 * (2 * m - 1) ** 2
        w[0] *= 2.5
        w[-1] *= 3.5
        w *= rng.lognormal(0.0, 0.35, size=n_min)
        day_vol = prof.daily_volume * (n_min / 390.0) * rng.lognormal(0.0, 0.2)
        vol = np.maximum(np.rint(w / w.sum() * day_vol / 100.0) * 100, 100).astype(np.int64)
        out["vol"].append(vol)
        out["trades"].append(np.maximum(vol // rng.integers(90, 160, size=n_min), 1).astype(np.int64))
        out["bar_start"].append(open_us + np.arange(n_min, dtype=np.int64) * 60_000_000)
        out["day_index"].append(np.full(n_min, di, dtype=np.int64))
    cat = {k: np.concatenate(v) for k, v in out.items()}
    return _SymbolArrays(symbol=symbol, **cat)


def _cents_to_dec(arr: np.ndarray, scale_pow: int = 2) -> pa.Array:
    base = pa.array(arr.astype(np.int64)).cast(pa.decimal128(19, 0))
    factor = pa.scalar(Decimal(1).scaleb(-scale_pow), type=pa.decimal128(scale_pow + 1, scale_pow))
    return pc.multiply(base, factor).cast(pa.decimal128(18, 6))  # pyright: ignore[reportAttributeAccessIssue]


def _ts_arr(us: np.ndarray) -> pa.Array:
    return pa.array(us.astype(np.int64), type=pa.timestamp("us", tz="UTC"))


def _dec_c(v: int) -> Decimal:
    return Decimal(int(v)).scaleb(-2)


def _dec_e4(v: int) -> Decimal:
    return Decimal(int(v)).scaleb(-4)


def _dt(us: int) -> datetime:
    return datetime.fromtimestamp(0, UTC) + timedelta(microseconds=int(us))


@dataclass
class SyntheticData:
    params: dict[str, Any]
    sessions: list[date]
    minute_bars: list[Bar]
    daily_bars: list[Bar]
    quote_table: pa.Table
    arrays: list[_SymbolArrays]


def _bars_for(a: _SymbolArrays, sessions: Sequence[date], cal: MarketCalendar, source_id: str,
              bar_latency: timedelta, daily_latency: timedelta) -> tuple[list[Bar], list[Bar]]:
    minute: list[Bar] = []
    one = timedelta(minutes=1)
    for i in range(len(a.bar_start)):
        start = _dt(int(a.bar_start[i]))
        end = start + one
        recv = end + bar_latency
        minute.append(Bar(
            symbol=a.symbol, timeframe="1Min", start_ts=start, end_ts=end,
            open=_dec_c(a.o[i]), high=_dec_c(a.h[i]), low=_dec_c(a.lo[i]), close=_dec_c(a.c[i]),
            volume=Decimal(int(a.vol[i])), vwap=_dec_e4(a.vwap_e4[i]), trade_count=int(a.trades[i]),
            feed=DataFeed.SYNTHETIC, adjustment=Adjustment.RAW, session=MarketSession.REGULAR,
            provider_ts=end, received_ts=recv, available_at_ts=recv, source_id=source_id,
            dataset_version=PENDING_DATASET_VERSION, is_synthetic=True,
        ))
    daily: list[Bar] = []
    for di, d in enumerate(sessions):
        idx = np.nonzero(a.day_index == di)[0]
        vols = a.vol[idx]
        vwap_e4 = int(np.rint((a.vwap_e4[idx] * vols).sum() / vols.sum()))
        close_ts = cal.session_close(d)
        recv = close_ts + daily_latency
        daily.append(Bar(
            symbol=a.symbol, timeframe="1Day", start_ts=cal.session_open(d), end_ts=close_ts,
            open=_dec_c(int(a.o[idx[0]])), high=_dec_c(int(a.h[idx].max())), low=_dec_c(int(a.lo[idx].min())),
            close=_dec_c(int(a.c[idx[-1]])), volume=Decimal(int(vols.sum())), vwap=_dec_e4(vwap_e4),
            trade_count=int(a.trades[idx].sum()), feed=DataFeed.SYNTHETIC, adjustment=Adjustment.RAW,
            session=MarketSession.REGULAR, provider_ts=close_ts, received_ts=recv, available_at_ts=recv,
            source_id=source_id, dataset_version=PENDING_DATASET_VERSION, is_synthetic=True,
        ))
    return minute, daily


def _quote_table(arrays: Sequence[_SymbolArrays], source_id: str, latency_us: int) -> pa.Table:
    tables: list[pa.Table] = []
    for a in arrays:
        n = len(a.q_ts)
        recv = a.q_ts + latency_us
        tables.append(pa.table({
            "symbol": pa.array([a.symbol] * n, pa.string()),
            "exchange_ts": _ts_arr(a.q_ts),
            "bid": _cents_to_dec(a.bid_c),
            "ask": _cents_to_dec(a.ask_c),
            "bid_size": pa.array(a.bid_sz.astype(np.int64)).cast(pa.decimal128(19, 0)).cast(pa.decimal128(24, 6)),
            "ask_size": pa.array(a.ask_sz.astype(np.int64)).cast(pa.decimal128(19, 0)).cast(pa.decimal128(24, 6)),
            "bid_exchange": pa.nulls(n, pa.string()),
            "ask_exchange": pa.nulls(n, pa.string()),
            "conditions": pa.nulls(n, pa.string()),
            "tape": pa.nulls(n, pa.string()),
            "provider_ts": _ts_arr(a.q_ts),
            "received_ts": _ts_arr(recv),
            "available_at_ts": _ts_arr(recv),
            "feed": pa.array([DataFeed.SYNTHETIC.value] * n, pa.string()),
            "source_id": pa.array([source_id] * n, pa.string()),
            "is_synthetic": pa.array(np.ones(n, dtype=bool)),
        }, schema=QUOTE_SCHEMA))
    return pa.concat_tables(tables) if tables else QUOTE_SCHEMA.empty_table()


def _check_quote_invariants(arrays: Sequence[_SymbolArrays], sessions: Sequence[date], cal: MarketCalendar,
                            quote_interval_s: int) -> dict[str, Any]:
    """Vectorized equivalent of validate_quotes for the clean generator output (fails loudly)."""
    total = 0
    opens = np.array([_us(cal.session_open(d)) for d in sessions], dtype=np.int64)
    closes = np.array([_us(cal.session_close(d)) for d in sessions], dtype=np.int64)
    for a in arrays:
        if (a.bid_c <= 0).any() or (a.ask_c <= a.bid_c).any():
            raise AssertionError(f"{a.symbol}: non-positive, crossed or locked synthetic quote")
        if (np.diff(a.q_ts) <= 0).any():
            raise AssertionError(f"{a.symbol}: synthetic quote timestamps not strictly increasing")
        k = np.searchsorted(opens, a.q_ts, side="right") - 1
        if (k < 0).any() or (a.q_ts >= closes[np.maximum(k, 0)]).any():
            raise AssertionError(f"{a.symbol}: synthetic quote outside the regular session")
        total += len(a.q_ts)
    return {"quotes_checked": total, "method": "vectorized invariants (positive, uncrossed, unlocked, increasing, "
            f"inside regular session); interval {quote_interval_s}s"}


_CACHE: dict[str, tuple[SyntheticData, ValidationReport, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def _params(symbols: Sequence[str], sessions: int, seed: int, end_date: date, quote_interval_s: int,
            quote_latency_ms: int, bar_latency_ms: int, daily_latency_s: int, inject_defects: bool) -> dict[str, Any]:
    return {
        "symbols": list(symbols), "sessions": sessions, "seed": seed, "end_date": end_date.isoformat(),
        "quote_interval_s": quote_interval_s, "quote_latency_ms": quote_latency_ms,
        "bar_latency_ms": bar_latency_ms, "daily_latency_s": daily_latency_s, "inject_defects": inject_defects,
        "calendar": "XNYS (exchange_calendars)", "session_filter": "REGULAR",
    }


def generate_synthetic(
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    sessions: int = DEFAULT_SESSIONS,
    seed: int = DEFAULT_SEED,
    end_date: date | None = None,
    quote_interval_s: int = DEFAULT_QUOTE_INTERVAL_S,
    quote_latency_ms: int = DEFAULT_QUOTE_LATENCY_MS,
    bar_latency_ms: int = DEFAULT_BAR_LATENCY_MS,
    daily_latency_s: int = DEFAULT_DAILY_LATENCY_S,
    calendar: MarketCalendar | None = None,
) -> SyntheticData:
    """Pure, deterministic generation (no I/O)."""
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    if quote_interval_s <= 0 or 60 % quote_interval_s:
        raise ValueError("quote_interval_s must divide 60")
    syms = tuple(dict.fromkeys(s.upper() for s in symbols))
    if not syms:
        raise ValueError("at least one symbol")
    cal = calendar or get_calendar()
    end = end_date or DEFAULT_END_DATE
    if not cal.is_session(end):
        end = cal.previous_session(end)
    days = cal.sessions_ending(end, sessions)
    source_id = source_id_for(seed)
    arrays = [_generate_symbol(s, days, cal, seed, quote_interval_s) for s in syms]
    minute: list[Bar] = []
    daily: list[Bar] = []
    for a in arrays:
        m, dly = _bars_for(a, days, cal, source_id, timedelta(milliseconds=bar_latency_ms), timedelta(seconds=daily_latency_s))
        minute += m
        daily += dly
    params = _params(syms, sessions, seed, end, quote_interval_s, quote_latency_ms, bar_latency_ms, daily_latency_s, False)
    return SyntheticData(params=params, sessions=list(days), minute_bars=minute, daily_bars=daily,
                         quote_table=_quote_table(arrays, source_id, quote_latency_ms * 1000), arrays=arrays)


# ------------------------------------------------------------------------------------------ defects
def _bar_records(bars: Sequence[Bar]) -> list[dict[str, Any]]:
    return [b.model_dump() for b in bars]


def inject_bar_defects(records: list[dict[str, Any]], symbol: str, cal: MarketCalendar) -> list[dict[str, Any]]:
    """Mutate the first session of ``symbol`` in place; returns ground truth [{kind, symbol, start_ts, action}]."""
    idx = [i for i, r in enumerate(records) if r["symbol"] == symbol and r["timeframe"] == "1Min"]
    if len(idx) < 120:
        raise ValueError("need at least 120 minute bars of the defect symbol")
    truth: list[dict[str, Any]] = []
    cents = Decimal("0.01")

    def at(k: int) -> dict[str, Any]:
        return records[idx[k]]

    def note(kind: str, rec: dict[str, Any], action: str) -> None:
        truth.append({"kind": kind, "symbol": symbol, "start_ts": iso(rec["start_ts"]), "expected_action": action})

    r = at(40)
    r["high"] = r["low"] - cents
    note("OHLC_INCONSISTENT", r, "REJECT")
    r = at(50)
    r["low"] = Decimal(0)
    note("NON_POSITIVE_PRICE", r, "REJECT")
    r = at(60)
    r["close"] = (r["close"] * Decimal("1.30")).quantize(cents)
    r["high"] = max(r["high"], r["close"])
    note("OUTLIER_JUMP", r, "FLAG")
    note("OUTLIER_JUMP", at(61), "FLAG")  # the reversion bar also jumps vs the spiked close
    r = at(70)
    r["provider_ts"] = r["received_ts"] + timedelta(seconds=10)
    note("CLOCK_SKEW", r, "REJECT")
    r = at(80)
    r["volume"] = Decimal(-5)
    note("NEGATIVE_VOLUME", r, "REJECT")
    r = at(90)
    r["available_at_ts"] = r["end_ts"] - timedelta(seconds=30)
    r["received_ts"] = r["end_ts"] + timedelta(seconds=1)
    note("LOOK_AHEAD", r, "REJECT")
    # structural defects (order-changing) last, from the highest index down
    gap_recs = [at(k) for k in range(30, 35)]
    for g in gap_recs:
        truth.append({"kind": "GAP", "symbol": symbol, "start_ts": iso(g["start_ts"]), "expected_action": "REPORT"})
    swap_a, swap_b = idx[20], idx[21]
    note("OUT_OF_ORDER", records[swap_a], "REJECT")
    records[swap_a], records[swap_b] = records[swap_b], records[swap_a]
    dup = dict(at(10))
    note("DUPLICATE", dup, "REJECT")
    first_day = cal.ny_date(at(0)["start_ts"])
    pre_start = cal.session_open(first_day) - timedelta(minutes=90)
    pre = dict(at(0))
    pre.update(start_ts=pre_start, end_ts=pre_start + timedelta(minutes=1),
               received_ts=pre_start + timedelta(minutes=1, seconds=1),
               available_at_ts=pre_start + timedelta(minutes=1, seconds=1), provider_ts=pre_start + timedelta(minutes=1),
               session=MarketSession.PRE)
    note("OUTSIDE_REGULAR_SESSION", pre, "REJECT")
    for k in sorted((idx[c] for c in range(30, 35)), reverse=True):
        del records[k]
    records.insert(idx[10] + 1, dup)
    records.insert(idx[0], pre)
    return truth


def inject_quote_defects(records: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    idx = [i for i, r in enumerate(records) if r["symbol"] == symbol]
    if len(idx) < 200:
        raise ValueError("need at least 200 quotes of the defect symbol")
    truth: list[dict[str, Any]] = []
    cents = Decimal("0.01")

    def at(k: int) -> dict[str, Any]:
        return records[idx[k]]

    def note(kind: str, rec: dict[str, Any], action: str) -> None:
        truth.append({"kind": kind, "symbol": symbol, "exchange_ts": iso(rec["exchange_ts"]), "expected_action": action})

    r = at(100)
    r["ask"] = r["bid"] - cents
    note("CROSSED_QUOTE", r, "REJECT")
    r = at(110)
    r["ask"] = r["bid"]
    note("LOCKED_QUOTE", r, "REJECT")
    r = at(120)
    r["received_ts"] = r["exchange_ts"] + timedelta(seconds=30)
    r["available_at_ts"] = r["received_ts"]
    note("STALE_QUOTE", r, "FLAG")
    r = at(130)
    r["provider_ts"] = r["received_ts"] + timedelta(seconds=10)
    note("CLOCK_SKEW", r, "REJECT")
    r = at(140)
    r["bid"] = Decimal(0)
    note("NON_POSITIVE_PRICE", r, "REJECT")
    swap_a, swap_b = idx[160], idx[161]
    note("OUT_OF_ORDER", records[swap_a], "REJECT")
    records[swap_a], records[swap_b] = records[swap_b], records[swap_a]
    dup = dict(at(150))
    note("DUPLICATE", dup, "REJECT")
    records.insert(idx[150] + 1, dup)
    return truth


# ------------------------------------------------------------------------------------------ entry point
def _find_existing(store: MarketDataStore, params: Mapping[str, Any]) -> dict[str, Any] | None:
    wanted = canonical_json(dict(params))
    for m in reversed(store.list_manifests(DataFeed.SYNTHETIC)):
        if m.get("generator", {}).get("name") == GENERATOR and m.get("generator", {}).get("version") == GENERATOR_VERSION \
                and canonical_json(m.get("params", {})) == wanted:
            return m
    return None


def ensure_synthetic_fixture(
    datasets_dir: Path,
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    sessions: int = DEFAULT_SESSIONS,
    seed: int = DEFAULT_SEED,
    end_date: date | None = None,
    inject_defects: bool = False,
    quote_interval_s: int = DEFAULT_QUOTE_INTERVAL_S,
    quote_latency_ms: int = DEFAULT_QUOTE_LATENCY_MS,
    bar_latency_ms: int = DEFAULT_BAR_LATENCY_MS,
    daily_latency_s: int = DEFAULT_DAILY_LATENCY_S,
    store: MarketDataStore | None = None,
) -> dict[str, Any]:
    """Create (or reuse) the deterministic SYNTHETIC dataset and return its manifest.

    Idempotent: if a dataset with identical parameters exists and its files verify, it is returned unchanged;
    otherwise it is regenerated — same seed => byte-identical Parquet and the same ``dataset_version``.
    """
    cal = get_calendar()
    end = end_date or DEFAULT_END_DATE
    if not cal.is_session(end):
        end = cal.previous_session(end)
    syms = tuple(dict.fromkeys(s.upper() for s in symbols))
    params = _params(syms, sessions, seed, end, quote_interval_s, quote_latency_ms, bar_latency_ms, daily_latency_s,
                     inject_defects)
    st = store or MarketDataStore(datasets_dir)
    existing = _find_existing(st, params)
    if existing is not None:
        check = st.verify_dataset(str(existing["dataset_version"]))
        if check["ok"]:
            if not inject_defects:
                st.activate_dataset(str(existing["dataset_version"]))
            return existing
        log.warning("synthetic fixture %s failed verification (%s); regenerating", existing["dataset_version"],
                    check["errors"])

    key = canonical_json(params)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is None:
        data = generate_synthetic(symbols=syms, sessions=sessions, seed=seed, end_date=end,
                                  quote_interval_s=quote_interval_s, quote_latency_ms=quote_latency_ms,
                                  bar_latency_ms=bar_latency_ms, daily_latency_s=daily_latency_s, calendar=cal)
        if inject_defects:
            return _write_defective(st, data, params, cal)
        report = validate_bars(data.minute_bars + data.daily_bars, cal, QualityPolicy(),
                               validated_at=datetime.combine(end, datetime.min.time(), UTC))
        if report.rejected or report.gaps:
            raise AssertionError(f"synthetic generator produced invalid bars: {report.summary()}")
        quote_check = _check_quote_invariants(data.arrays, data.sessions, cal, quote_interval_s)
        cached = (data, report, quote_check)
        with _CACHE_LOCK:
            if len(_CACHE) >= 2:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = cached
    data, report, quote_check = cached
    spec = DatasetSpec(
        source_id=source_id_for(seed),
        feed=DataFeed.SYNTHETIC,
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        is_synthetic=True,
        kind="synthetic_fixture",
        series_mode=SERIES_SNAPSHOT,
        description=f"{SYNTHETIC_LABEL}: seeded minute+daily bars and {quote_interval_s}s top-of-book quotes",
        limitations=FIXTURE_LIMITATIONS,
        params=params,
        extra={"validation": {"bars": _stable_summary(report), "quotes": quote_check},
               "sessions": [d.isoformat() for d in data.sessions]},
    )
    manifest = st.write_dataset(spec, bar_table=_bars_table_fast(report.accepted), quote_table=data.quote_table,
                                bar_extras=report.raw_extras(), created_at=datetime.now(UTC))
    if existing is not None:  # regenerated after a failed verification: it must verify now, or fail loudly
        check = st.verify_dataset(str(manifest["dataset_version"]))
        if not check["ok"]:
            raise FixtureIntegrityError(
                f"synthetic fixture {manifest['dataset_version']} still fails verification after regeneration: "
                f"{check['errors'][:3]}. The raw Parquet under {st.raw_dir / 'SYNTHETIC'} was modified; move it aside "
                "(and the dataset's DuckDB registry rows) and rerun `python -m services.market_data.cli fixture`.")
    return manifest


class FixtureIntegrityError(RuntimeError):
    """The stored synthetic fixture is corrupt and could not be regenerated in place (never used silently)."""


def _stable_summary(report: ValidationReport) -> dict[str, Any]:
    s = report.summary()
    s.pop("validated_at", None)
    return s


def _bars_table_fast(bars: Sequence[Bar]) -> pa.Table:
    from .store import bars_to_table

    return bars_to_table(bars).cast(BAR_SCHEMA)


def _write_defective(st: MarketDataStore, data: SyntheticData, params: Mapping[str, Any],
                     cal: MarketCalendar) -> dict[str, Any]:
    from .store import _quote_from_row  # pyright: ignore[reportPrivateUsage]

    symbol = str(params["symbols"][0])
    bar_recs = _bar_records(data.minute_bars)
    truth = inject_bar_defects(bar_recs, symbol, cal)
    quote_recs = [_quote_from_row(r).model_dump() for r in data.quote_table.to_pylist()]
    truth += inject_quote_defects(quote_recs, symbol)
    asof = datetime.combine(data.sessions[-1], datetime.min.time(), UTC) + timedelta(days=1)
    breport = validate_bars(bar_recs, cal, QualityPolicy(), validated_at=asof)
    qreport = validate_quotes(quote_recs, cal, QualityPolicy(), validated_at=asof)
    spec = DatasetSpec(
        source_id=source_id_for(int(params["seed"])) + ":defects",
        feed=DataFeed.SYNTHETIC,
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        is_synthetic=True,
        kind="synthetic_fixture_defects",
        series_mode=SERIES_SNAPSHOT,
        description=f"{SYNTHETIC_LABEL}: WITH INJECTED DEFECTS for validator tests (excluded from default loads)",
        limitations=(*FIXTURE_LIMITATIONS, "Contains deliberately injected data defects; never use for research"),
        params=params,
        defects_injected=tuple(truth),
        extra={"validation": {"bars": _stable_summary(breport), "quotes": _stable_summary(qreport)},
               "sessions": [d.isoformat() for d in data.sessions],
               "gaps": [g.as_dict() for g in breport.gaps]},
    )
    from .quality import rejected_raw_json

    bars_ok = [b.model_copy(update={"dataset_version": PENDING_DATASET_VERSION}) for b in breport.accepted]
    manifest = st.write_dataset(
        spec,
        bars=bars_ok,
        quotes=list(qreport.accepted),
        bar_extras=breport.raw_extras(rejected_raw_json(breport)),
        quote_extras=qreport.raw_extras(rejected_raw_json(qreport)),
        created_at=datetime.now(UTC),
    )
    return manifest


def manifest_sessions(manifest: Mapping[str, Any]) -> list[date]:
    return [date.fromisoformat(s) for s in manifest.get("extra", {}).get("sessions", [])]
