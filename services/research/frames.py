"""Bar-frame normalisation and point-in-time helpers shared by every research module.

Input contract (produced by ``services.market_data.store.MarketDataStore.load_bars_df``): a pandas
DataFrame with columns ``[symbol, timeframe, start_ts, end_ts, open, high, low, close, volume,
available_at_ts, dataset_version, is_synthetic]``.

``prepare_bars`` validates and normalises it into a frame sorted by ``(symbol, end_ts)`` with:

* tz-aware UTC timestamps (naive timestamps are *rejected*, never assumed to be UTC);
* exact ``Decimal`` OHLCV columns (``open`` ... ``volume``) for accounting;
* float mirrors (``open_f`` ... ``volume_f``) for features/statistics only.

Point-in-time rule used everywhere in this package: a decision taken at time ``T`` may only use rows
with ``available_at_ts <= T``; an order decided at ``T`` executes at the open of the first bar of the
same symbol whose ``start_ts >= T`` (never at a bar that had already opened when the decision was made).
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timeframe",
    "start_ts",
    "end_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "available_at_ts",
    "dataset_version",
    "is_synthetic",
)
OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")
TS_COLUMNS: tuple[str, ...] = ("start_ts", "end_ts", "available_at_ts")
NY = ZoneInfo("America/New_York")
REGULAR_SESSION_MINUTES = 390
TRADING_DAYS_PER_YEAR = 252


class BarDataError(ValueError):
    """The bar frame violates the input contract. Research code fails closed instead of repairing."""


def to_decimal(value: Any) -> Decimal:
    """Exact Decimal from a bar cell.

    Parquet/DuckDB may hand back float64 for price columns; that is an *I/O boundary*, so floats are
    converted through ``repr`` (shortest round-trip string), never ``Decimal(float)``. NaN/inf rejected.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise BarDataError("non-finite decimal in bar data")
        return value
    if isinstance(value, bool | np.bool_):
        raise BarDataError("boolean is not a price/volume")
    if isinstance(value, int | np.integer):
        return Decimal(int(value))
    if isinstance(value, float | np.floating):
        f = float(value)
        if not math.isfinite(f):
            raise BarDataError("NaN/inf in bar data")
        return Decimal(repr(f))
    if isinstance(value, str):
        try:
            d = Decimal(value.strip())
        except InvalidOperation as exc:
            raise BarDataError(f"invalid decimal string {value!r}") from exc
        if not d.is_finite():
            raise BarDataError("non-finite decimal in bar data")
        return d
    raise BarDataError(f"unsupported price/volume type {type(value).__name__}")


def _to_utc(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        s = series
    else:
        values = list(series)
        if any(isinstance(v, datetime) and (v.tzinfo is None or v.utcoffset() is None) for v in values):
            raise BarDataError(f"column {name} contains naive datetimes; timestamps must be tz-aware")
        try:
            s = pd.to_datetime(series, utc=True)
        except (ValueError, TypeError) as exc:
            raise BarDataError(f"column {name} is not parseable as timestamps: {exc}") from exc
        if any(isinstance(v, str) and not _has_offset(v) for v in values):
            raise BarDataError(f"column {name} contains timestamp strings without a UTC offset")
    tz = getattr(s.dt, "tz", None)
    if tz is None:
        raise BarDataError(f"column {name} is timezone-naive; timestamps must be tz-aware UTC")
    # pandas >= 3 defaults to microsecond resolution; normalise so int64 views are always nanoseconds.
    return s.dt.tz_convert("UTC").dt.as_unit("ns")


_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _has_offset(text: str) -> bool:
    return bool(_OFFSET_RE.search(text.strip()))


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + normalise a bar frame (see module docstring). Returns a new frame; input untouched."""
    if not isinstance(df, pd.DataFrame):
        raise BarDataError("bars must be a pandas DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BarDataError(f"bar frame missing required columns: {missing}")
    if df.empty:
        raise BarDataError("bar frame is empty")
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    if out[list(REQUIRED_COLUMNS)].isna().any().any():
        bad = [c for c in REQUIRED_COLUMNS if out[c].isna().any()]
        raise BarDataError(f"null values in columns {bad}; gaps must not be silently filled")
    out["symbol"] = out["symbol"].astype(str)
    out["timeframe"] = out["timeframe"].astype(str)
    out["dataset_version"] = out["dataset_version"].astype(str)
    out["is_synthetic"] = out["is_synthetic"].map(_to_bool)
    for col in TS_COLUMNS:
        out[col] = _to_utc(out[col], col)
    for col in OHLCV:
        dec = [to_decimal(v) for v in out[col].tolist()]
        out[col] = pd.Series(dec, index=out.index, dtype=object)
        out[f"{col}_f"] = np.array([float(d) for d in dec], dtype=np.float64)

    timeframes = sorted(set(out["timeframe"]))
    if len(timeframes) != 1:
        raise BarDataError(f"exactly one timeframe per research call required, got {timeframes}")
    timeframe_delta(timeframes[0])  # validates the timeframe string

    if (out["end_ts"] <= out["start_ts"]).any():
        raise BarDataError("end_ts must be after start_ts")
    if (out["available_at_ts"] < out["end_ts"]).any():
        raise BarDataError("available_at_ts precedes end_ts (look-ahead) in bar data")
    if (out["low_f"] <= 0).any():
        raise BarDataError("non-positive price in bar data")
    if (out["volume_f"] < 0).any():
        raise BarDataError("negative volume in bar data")
    hi_ok = out["high_f"] >= np.maximum(out["open_f"], out["close_f"])
    lo_ok = out["low_f"] <= np.minimum(out["open_f"], out["close_f"])
    if not (hi_ok & lo_ok).all():
        raise BarDataError("OHLC inconsistent: high/low do not bound open/close")

    out = out.sort_values(["symbol", "end_ts"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["symbol", "end_ts"]).any():
        raise BarDataError("duplicate (symbol, end_ts) rows; dedupe at ingestion, research will not guess")
    for sym, g in out.groupby("symbol", sort=False):
        if not g["available_at_ts"].is_monotonic_increasing:
            raise BarDataError(f"available_at_ts not monotonic for {sym}")
        if (g["start_ts"].iloc[1:].to_numpy() < g["end_ts"].iloc[:-1].to_numpy()).any():
            raise BarDataError(f"overlapping bars for {sym}")
    # Session date from the bar *start* in New York time (robust to daily bars stamped 00:00 ET -> 00:00 ET).
    out["session_date"] = pd.Series(list(out["start_ts"].dt.tz_convert(NY).dt.date), index=out.index, dtype=object)
    return out


def prefilter_available(df: pd.DataFrame, asof: datetime | pd.Timestamp) -> pd.DataFrame:
    """Drop rows not knowable at ``asof`` *before* any validation of their content.

    Used by decision-time code so that data published after the decision (even malformed data) cannot
    influence the result, including by raising. Only ``available_at_ts`` is parsed here.
    """
    if not isinstance(df, pd.DataFrame):
        raise BarDataError("bars must be a pandas DataFrame")
    if df.empty:
        return df.copy()  # nothing knowable: callers report "no bars" (NO_TRADE), never guess
    if "available_at_ts" not in df.columns:
        raise BarDataError("bar frame missing required column available_at_ts")
    avail = _to_utc(pd.Series(df["available_at_ts"]), "available_at_ts")
    if avail.isna().any():
        raise BarDataError("null available_at_ts; gaps must not be silently filled")
    return df.loc[(avail <= utc_ts(asof)).to_numpy()].copy()


def aggregate_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a *prepared* intraday frame into one bar per (symbol, NY session date).

    open=first, high=max, low=min, close=last, volume=sum (exact Decimal), start=first start, end=last end,
    available_at=last available_at (a daily bar is knowable only once its last minute is). Daily input
    is returned unchanged. The result is re-validated through ``prepare_bars``.
    """
    tf = str(bars["timeframe"].iloc[0])
    if not is_intraday(tf):
        return bars
    rows: list[dict[str, Any]] = []
    for keys, g in bars.groupby(["symbol", "session_date"], sort=True):
        sym, day = cast("tuple[Any, Any]", keys)
        opens: list[Decimal] = list(g["open"])
        highs: list[Decimal] = list(g["high"])
        lows: list[Decimal] = list(g["low"])
        closes: list[Decimal] = list(g["close"])
        vols: list[Decimal] = list(g["volume"])
        rows.append(
            {
                "symbol": sym,
                "timeframe": "1Day",
                "start_ts": g["start_ts"].iloc[0],
                "end_ts": g["end_ts"].iloc[-1],
                "open": opens[0],
                "high": max(highs),
                "low": min(lows),
                "close": closes[-1],
                "volume": sum(vols, Decimal(0)),
                "available_at_ts": g["available_at_ts"].iloc[-1],
                "dataset_version": "+".join(sorted(set(g["dataset_version"].astype(str)))),
                "is_synthetic": bool(g["is_synthetic"].any()),
                "_day": day,
            }
        )
    out = pd.DataFrame(rows).drop(columns=["_day"])
    return prepare_bars(out)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise BarDataError(f"is_synthetic must be boolean, got {value!r}")


_TF_RE = re.compile(r"^(\d+)\s*(min|minute|t|h|hour|d|day)s?$", re.IGNORECASE)


@lru_cache(maxsize=64)
def timeframe_delta(timeframe: str) -> timedelta:
    m = _TF_RE.match(timeframe.strip())
    if not m:
        raise BarDataError(f"unsupported timeframe {timeframe!r} (expected e.g. 1Min, 5Min, 1Hour, 1Day)")
    n = int(m.group(1))
    unit = m.group(2).lower()
    if n <= 0:
        raise BarDataError("timeframe multiple must be positive")
    if unit in {"min", "minute", "t"}:
        return timedelta(minutes=n)
    if unit in {"h", "hour"}:
        return timedelta(hours=n)
    return timedelta(days=n)


def is_intraday(timeframe: str) -> bool:
    return timeframe_delta(timeframe) < timedelta(days=1)


def bar_trading_minutes(timeframe: str) -> int:
    """Trading minutes represented by one bar (a daily bar = one 390-minute regular session)."""
    delta = timeframe_delta(timeframe)
    if delta >= timedelta(days=1):
        return REGULAR_SESSION_MINUTES * delta.days
    return max(1, int(delta.total_seconds() // 60))


def bars_per_day(timeframe: str) -> int:
    return max(1, REGULAR_SESSION_MINUTES // bar_trading_minutes(timeframe))


def horizon_bars(horizon_minutes: int, timeframe: str) -> int:
    return max(1, round(horizon_minutes / bar_trading_minutes(timeframe)))


def periods_per_year(timeframe: str) -> float:
    minutes = bar_trading_minutes(timeframe)
    return TRADING_DAYS_PER_YEAR * REGULAR_SESSION_MINUTES / minutes


def next_exec_index(start_ts: np.ndarray, decision_ts: np.ndarray) -> np.ndarray:
    """For each decision i (same symbol, sorted), index of the first bar with start_ts >= decision_ts[i].

    Returns ``len(start_ts)`` when no such bar exists (decision cannot be executed in-sample).
    Both inputs are int64 nanosecond arrays sorted ascending.
    """
    return np.searchsorted(start_ts, decision_ts, side="left")


def ts_ns(series: pd.Series | pd.DataFrame | Any) -> np.ndarray:
    """UTC timestamps -> int64 nanoseconds since epoch (independent of the column's stored resolution)."""
    s = pd.Series(series)
    return s.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


@dataclass(frozen=True)
class SessionInfo:
    session_date: date
    close_utc: datetime


@lru_cache(maxsize=1)
def _xnys() -> Any:
    try:
        import exchange_calendars as xcals  # optional heavy import

        with warnings.catch_warnings():  # third-party pandas deprecation noise inside exchange_calendars
            warnings.simplefilter("ignore")
            return xcals.get_calendar("XNYS")
    except Exception:  # pragma: no cover - calendar package missing/broken: fall back to 16:00 ET
        return None


@lru_cache(maxsize=8192)
def session_close_utc(session_day: date) -> datetime:
    """Scheduled regular-session close for an NYSE session date (known in advance; not look-ahead).

    Uses exchange_calendars XNYS (handles early closes). For dates that are not XNYS sessions (e.g.
    a synthetic fixture on a holiday) falls back to 16:00 America/New_York.
    """
    cal = _xnys()
    if cal is not None:
        label = pd.Timestamp(session_day)
        try:
            if cal.is_session(label):
                close = cal.session_close(label)
                return close.tz_convert("UTC").to_pydatetime()
        except Exception:  # noqa: S110 - outside calendar bounds: use the documented fallback
            pass
    return datetime.combine(session_day, time(16, 0), tzinfo=NY).astimezone(UTC)


def utc_ts(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Aware UTC pandas Timestamp (ns resolution); naive or missing inputs are rejected."""
    ts = pd.Timestamp(value)
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        raise BarDataError("missing timestamp")
    if ts.tzinfo is None:
        raise BarDataError("naive timestamp rejected; pass a timezone-aware datetime")
    return ts.tz_convert("UTC").as_unit("ns")


def point_in_time(bars: pd.DataFrame, asof: datetime | pd.Timestamp) -> pd.DataFrame:
    """Rows knowable at ``asof``: ``available_at_ts <= asof``."""
    cutoff = utc_ts(asof)
    return bars.loc[bars["available_at_ts"] <= cutoff]


def dataset_version_of(bars: pd.DataFrame) -> str:
    versions = sorted(set(bars["dataset_version"].astype(str)))
    if not versions:
        raise BarDataError("no dataset_version in bars")
    return "+".join(versions)


def is_simulated_of(bars: pd.DataFrame) -> bool:
    """True when any bar is synthetic (results must then be labelled SIMULATED)."""
    return bool(bars["is_synthetic"].any())


def symbols_of(bars: pd.DataFrame) -> list[str]:
    return sorted(set(bars["symbol"]))


def iter_symbol_frames(bars: pd.DataFrame, symbols: Iterable[str] | None = None) -> Iterable[tuple[str, pd.DataFrame]]:
    wanted = None if symbols is None else set(symbols)
    for sym, g in bars.groupby("symbol", sort=True):
        if wanted is None or sym in wanted:
            yield str(sym), g.reset_index(drop=True)


def trailing_adv_shares(sym_bars: pd.DataFrame, *, sessions: int = 20) -> np.ndarray:
    """Point-in-time average daily volume (shares) usable at each bar's decision time.

    Daily bars: mean volume of the last ``sessions`` bars *including* the current (closed) bar.
    Intraday bars: mean total volume of the last ``sessions`` *completed* sessions strictly before the
    current bar's session (today's partial volume is not a daily volume). NaN when unavailable.
    """
    vol = sym_bars["volume_f"].to_numpy(dtype=np.float64)
    tf = str(sym_bars["timeframe"].iloc[0])
    if not is_intraday(tf):
        return np.asarray(pd.Series(vol).rolling(sessions, min_periods=1).mean(), dtype=np.float64)
    days: list[Any] = list(sym_bars["session_date"])
    totals: dict[Any, float] = {}
    for d, v in zip(days, vol.tolist(), strict=True):
        totals[d] = totals.get(d, 0.0) + v
    ordered = list(totals)  # insertion order == chronological (frame sorted by end_ts)
    prior = pd.Series([totals[d] for d in ordered], dtype=np.float64).shift(1).rolling(sessions, min_periods=1).mean()
    mapping = dict(zip(ordered, np.asarray(prior, dtype=np.float64).tolist(), strict=True))
    return np.array([mapping[d] for d in days], dtype=np.float64)


@lru_cache(maxsize=1)
def _xnys_session_index() -> dict[date, int] | None:
    cal = _xnys()
    if cal is None:  # pragma: no cover - calendar package missing
        return None
    sessions = cal.sessions  # DatetimeIndex of session labels
    return {ts.date(): i for i, ts in enumerate(sessions)}


def missing_sessions_between(d0: date, d1: date) -> int:
    """XNYS sessions strictly between two session dates (weekday count when outside the calendar)."""
    if d1 <= d0:
        return 0
    idx = _xnys_session_index()
    if idx is not None and d0 in idx and d1 in idx:
        return max(0, idx[d1] - idx[d0] - 1)
    return max(0, int(np.busday_count(d0 + timedelta(days=1), d1)))


@dataclass(frozen=True)
class SegmentPolicy:
    """Data-quality segmentation for research series (never repairs, only cuts and reports).

    A symbol's series is cut where (a) more than ``max_missing_sessions`` XNYS sessions are missing between two
    consecutive bars (multi-session holes, e.g. a vendor sample missing years) or (b) a one-bar move exceeds
    ``max_abs_log_return`` (|ln r| > 1 = x2.7 / -63 %: ticker reuse, an unadjusted corporate action or a bad
    print). Only the LAST clean segment (the one adjacent to the most recent data) is kept.
    """

    max_missing_sessions: int = 5
    max_abs_log_return: float = 1.0
    min_bars: int = 3


def last_clean_segment(bars: pd.DataFrame, policy: SegmentPolicy | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply ``SegmentPolicy`` to a *prepared* frame. Returns (kept frame, per-symbol report)."""
    pol = policy or SegmentPolicy()
    kept: list[pd.DataFrame] = []
    report: dict[str, Any] = {}
    for sym, g in bars.groupby("symbol", sort=True):
        g = g.reset_index(drop=True)
        days: list[date] = list(g["session_date"])
        close = g["close_f"].to_numpy(dtype=np.float64)
        cut = 0
        reason = ""
        for i in range(1, len(g)):
            missing = 0 if days[i] == days[i - 1] else missing_sessions_between(days[i - 1], days[i])
            jump = abs(math.log(close[i] / close[i - 1]))
            if missing > pol.max_missing_sessions:
                cut, reason = i, f"{missing} missing sessions before {days[i].isoformat()}"
            elif jump > pol.max_abs_log_return:
                cut, reason = i, f"one-bar |ln r|={jump:.3f} on {days[i].isoformat()}"
        seg = g.iloc[cut:]
        entry: dict[str, Any] = {"rows_in": len(g), "rows_kept": len(seg), "dropped_before_cut": cut}
        if cut:
            entry["last_cut"] = reason
        if len(seg) < pol.min_bars:
            entry["excluded"] = f"last clean segment has {len(seg)} bars (< {pol.min_bars})"
        else:
            kept.append(seg)
        report[str(sym)] = entry
    if not kept:
        raise BarDataError(f"no symbol has a clean segment of >= {pol.min_bars} bars: {report}")
    out = pd.concat(kept, ignore_index=True)
    summary = {
        "policy": {"max_missing_sessions": pol.max_missing_sessions, "max_abs_log_return": pol.max_abs_log_return,
                   "min_bars": pol.min_bars, "rule": "keep the last clean contiguous segment per symbol"},
        "symbols": report,
        "rows_dropped": int(sum(v["rows_in"] - v["rows_kept"] for v in report.values())
                            + sum(v["rows_kept"] for v in report.values() if "excluded" in v)),
    }
    return out, summary


def float_to_dec(value: float, places: int = 6) -> Decimal:
    """Statistic (float) -> Decimal for a contract field, quantised to ``places``. NaN/inf rejected."""
    if not math.isfinite(value):
        raise ValueError("non-finite statistic")
    return Decimal(f"{value:.{places}f}")
