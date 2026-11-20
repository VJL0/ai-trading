"""Importer for the REAL historical sample data bundled in the pinned QuantConnect LEAN clone.

Source layout (``<lean_root>/Data/equity/usa``)::

    daily/<sym>.zip                      -> <sym>.csv            "yyyyMMdd HH:mm,open,high,low,close,volume"
    minute/<sym>/<yyyyMMdd>_trade.zip    -> ..._minute_trade.csv "ms_since_midnight,open,high,low,close,volume"
    minute/<sym>/<yyyyMMdd>_quote.zip    -> ..._minute_quote.csv "ms,bid o,h,l,c,last bid size,ask o,h,l,c,last ask size"
    factor_files/<sym>.csv               "yyyyMMdd,price_factor,split_factor,reference_price"
    map_files/<sym>.csv                  "yyyyMMdd,mapped_ticker,primary_exchange"

Prices are integers x10000 (deci-cents); minute times are milliseconds since midnight America/New_York; daily
bars are normalized to the XNYS session ``[open, close]``. Records are normalized with ``feed=LEAN_SAMPLE``,
``is_synthetic=False`` and validated with the standard :class:`QualityPolicy` (extended-hours minute bars are
rejected from the normalized tables but preserved in raw Parquet; gaps are reported, never filled).

Corporate actions follow LEAN's own derivation (``CorporateFactorProvider.GetSplitsAndDividends``): for
consecutive factor rows i < i+1, ex-date = next trading day after row i's date, split factor =
sf_i / sf_{i+1}, dividend = round(ref_i - ref_i * pf_i / pf_{i+1}, 2). Map files yield RENAME (ticker change)
and DELISTING (last row before 2050-12-31) events.

Point in time: ``received_ts`` is the vintage of the source snapshot (the LEAN commit timestamp) because the
bytes were obtained from that repository state; ``available_at_ts`` is bar end + latency (when the bar would have
been knowable). The *values* are of unknown adjustment/revision provenance — see the manifest limitations.
"""

from __future__ import annotations

import logging
import subprocess
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import Adjustment, DataFeed, MarketSession
from packages.contracts.models import Bar
from packages.core.config import PROJECT_ROOT
from packages.core.db import Database

from .calendar import NY, CalendarOutOfRange, MarketCalendar, get_calendar
from .common import FEED_LIMITATIONS, PENDING_DATASET_VERSION, iso
from .corporate_actions import CorporateAction, CorporateActionType
from .manifest import SERIES_SNAPSHOT, DatasetSpec
from .quality import (
    Gap,
    QualityPolicy,
    ValidationReport,
    gaps_within_observed_sessions,
    record_quality_events,
    rejected_raw_json,
    validate_bars,
    validate_quotes,
    with_gaps,
)
from .store import MarketDataStore

log = logging.getLogger("lat.market_data.lean_sample")

IMPORTER = "services.market_data.lean_sample"
IMPORTER_VERSION = "1.0.0"
PRICE_SCALE = 4  # LEAN equity prices are integers x10000
DEFAULT_LEAN_ROOT = PROJECT_ROOT / "trading" / "lean" / "Lean"
DEFAULT_MINUTE_LATENCY = timedelta(seconds=2)
DEFAULT_DAILY_LATENCY = timedelta(minutes=15)
FAR_FUTURE = date(2050, 12, 31)

LEAN_SAMPLE_LIMITATIONS: tuple[str, ...] = (
    *FEED_LIMITATIONS[DataFeed.LEAN_SAMPLE],
    "Source: QuantConnect LEAN repository sample data (Data/equity/usa) at the pinned commit; old vintage "
    "(exact minute-file days and daily range are listed in this manifest's extra.coverage) — not current prices",
    "Adjustment provenance unknown: treated as raw; values may include later vendor revisions (PIT of values not guaranteed)",
    "Sample data credited to AlgoSeek in LEAN's Data/equity/usa/readme.md ('all rights reserved'); "
    "local research/testing use only, do not redistribute",
    "Minute quotes are LEAN quote-bar closes (last top of book inside each minute), not tick-level NBBO",
    "Extended-hours minute bars are preserved in raw Parquet but rejected from normalized tables (regular session only)",
)


HISTORICAL_IMPORT_POLICY = QualityPolicy().with_actions(stale_quote="IGNORE")


class LeanFormatError(ValueError):
    pass


@dataclass(frozen=True)
class FactorRow:
    day: date
    price_factor: Decimal
    split_factor: Decimal
    reference_price: Decimal
    raw: str


@dataclass(frozen=True)
class MapRow:
    day: date
    ticker: str
    exchange: str | None
    raw: str


def lean_commit(lean_root: Path) -> tuple[str, datetime | None]:
    """(commit sha, commit time) of the LEAN clone; ('unknown', None) if ``lean_root`` is not itself the top of a git
    checkout (a directory nested in some other repository must not inherit that repository's commit)."""
    try:
        top = subprocess.run(["git", "-C", str(lean_root), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        if Path(top).resolve() != Path(lean_root).resolve():
            return "unknown", None
        sha = subprocess.run(["git", "-C", str(lean_root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        when = subprocess.run(["git", "-C", str(lean_root), "log", "-1", "--format=%cI"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
        return sha, datetime.fromisoformat(when).astimezone(UTC)
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown", None


def read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise LeanFormatError(f"{path}: expected exactly one entry, found {names}")
        return zf.read(names[0]).decode("utf-8")


def _px(field: str) -> Decimal:
    s = field.strip()
    if not s or not (s.lstrip("-").isdigit()):
        raise LeanFormatError(f"non-integer LEAN price field {field!r}")
    return Decimal(int(s)).scaleb(-PRICE_SCALE)


def _px_opt(field: str) -> Decimal | None:
    return None if not field.strip() else _px(field)


def _int_opt(field: str) -> Decimal | None:
    return None if not field.strip() else _int(field)


def _int(field: str) -> Decimal:
    s = field.strip()
    if not s.lstrip("-").isdigit():
        raise LeanFormatError(f"non-integer LEAN volume/size field {field!r}")
    return Decimal(int(s))


def _ny_dt(day: date, ms: int) -> datetime:
    return (datetime.combine(day, time(0), tzinfo=NY) + timedelta(milliseconds=ms)).astimezone(UTC)


def _base(symbol: str, source_id: str, received: datetime) -> dict[str, Any]:
    return {"symbol": symbol, "feed": DataFeed.LEAN_SAMPLE, "adjustment": Adjustment.RAW, "source_id": source_id,
            "received_ts": received, "provider_ts": None, "dataset_version": PENDING_DATASET_VERSION,
            "is_synthetic": False}


def parse_daily_csv(text: str, symbol: str, *, cal: MarketCalendar, source_id: str, received: datetime,
                    latency: timedelta = DEFAULT_DAILY_LATENCY) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise LeanFormatError(f"{symbol} daily line {n}: expected 6 fields")
        day = datetime.strptime(parts[0].strip()[:8], "%Y%m%d").date()  # noqa: DTZ007 - a date, not an instant
        try:
            start, end = cal.session_open(day), cal.session_close(day)
            session = MarketSession.REGULAR
        except (ValueError, CalendarOutOfRange):
            start = datetime.combine(day, time(9, 30), tzinfo=NY).astimezone(UTC)
            end = datetime.combine(day, time(16, 0), tzinfo=NY).astimezone(UTC)
            session = MarketSession.CLOSED
        out.append({**_base(symbol, source_id, received), "timeframe": "1Day", "start_ts": start, "end_ts": end,
                    "open": _px(parts[1]), "high": _px(parts[2]), "low": _px(parts[3]), "close": _px(parts[4]),
                    "volume": _int(parts[5]), "vwap": None, "trade_count": None, "session": session,
                    "available_at_ts": end + latency})
    return out


def parse_minute_trade_csv(text: str, symbol: str, day: date, *, cal: MarketCalendar, source_id: str,
                           received: datetime, latency: timedelta = DEFAULT_MINUTE_LATENCY) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise LeanFormatError(f"{symbol} {day} minute line {n}: expected 6 fields")
        start = _ny_dt(day, int(parts[0]))
        end = start + timedelta(minutes=1)
        try:
            session = cal.classify_bar(start, end)
        except CalendarOutOfRange:
            session = MarketSession.CLOSED
        out.append({**_base(symbol, source_id, received), "timeframe": "1Min", "start_ts": start, "end_ts": end,
                    "open": _px(parts[1]), "high": _px(parts[2]), "low": _px(parts[3]), "close": _px(parts[4]),
                    "volume": _int(parts[5]), "vwap": None, "trade_count": None, "session": session,
                    "available_at_ts": end + latency})
    return out


def parse_minute_quote_csv(text: str, symbol: str, day: date, *, source_id: str, received: datetime,
                           latency: timedelta = DEFAULT_MINUTE_LATENCY) -> list[dict[str, Any]]:
    """Quote-bar closes as top-of-book quotes (exchange_ts = minute end - 1us, available at minute end + latency)."""
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 11:
            raise LeanFormatError(f"{symbol} {day} quote line {n}: expected 11 fields")
        bar_end = _ny_dt(day, int(parts[0])) + timedelta(minutes=1)
        # The quote-bar close is the last quote inside [start, end): stamp it at the inclusive upper bound end-1us so
        # the 15:59 bar stays REGULAR and the 09:29 bar stays PRE; it becomes knowable only at bar end + latency.
        ex = bar_end - timedelta(microseconds=1)
        # A one-sided quote bar (empty bid or ask fields) yields None values: the validator rejects it as
        # INVALID_RECORD and the store preserves it verbatim in raw Parquet (never silently dropped or filled).
        out.append({"symbol": symbol, "bid": _px_opt(parts[4]), "bid_size": _int_opt(parts[5]),
                    "ask": _px_opt(parts[9]), "ask_size": _int_opt(parts[10]), "bid_exchange": None, "ask_exchange": None,
                    "exchange_ts": ex,
                    "provider_ts": None, "received_ts": received, "available_at_ts": bar_end + latency,
                    "feed": DataFeed.LEAN_SAMPLE, "conditions": ("LEAN_QUOTEBAR_CLOSE",), "tape": None,
                    "source_id": source_id, "is_synthetic": False})
    return out


def parse_factor_file(text: str) -> list[FactorRow]:
    rows: list[FactorRow] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or "inf" in s or "e+" in s:  # LEAN skips these rows too (precision loss)
            continue
        p = s.split(",")
        try:
            rows.append(FactorRow(datetime.strptime(p[0], "%Y%m%d").date(), Decimal(p[1]), Decimal(p[2]),  # noqa: DTZ007
                                  Decimal(p[3]) if len(p) > 3 else Decimal(0), s))
        except (ValueError, ArithmeticError, IndexError) as exc:
            raise LeanFormatError(f"bad factor row {s!r}: {exc}") from exc
    rows = [r for r in rows if r.price_factor * r.split_factor > 0]
    return sorted(rows, key=lambda r: r.day)


def parse_map_file(text: str) -> list[MapRow]:
    rows: list[MapRow] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        p = s.split(",")
        rows.append(MapRow(datetime.strptime(p[0], "%Y%m%d").date(), p[1].strip().upper(),  # noqa: DTZ007
                           p[2].strip() if len(p) > 2 and p[2].strip() else None, s))
    return sorted(rows, key=lambda r: r.day)


def corporate_actions_from_factors(symbol: str, rows: Sequence[FactorRow], cal: MarketCalendar,
                                   source_id: str) -> list[CorporateAction]:
    out: list[CorporateAction] = []
    for i in range(len(rows) - 1):
        row, nxt = rows[i], rows[i + 1]
        try:
            ex = cal.next_session(row.day)
            avail = cal.session_open(ex)
        except (CalendarOutOfRange, ValueError):
            continue
        raw = f"{row.raw}\n{nxt.raw}"
        if nxt.split_factor != 0 and row.split_factor != nxt.split_factor:
            out.append(CorporateAction(symbol=symbol, action_type=CorporateActionType.SPLIT, ex_date=ex,
                                       available_at_ts=avail, split_factor=row.split_factor / nxt.split_factor,
                                       reference_price=row.reference_price or None, feed=DataFeed.LEAN_SAMPLE,
                                       source_id=source_id, raw=raw))
        if nxt.price_factor != 0 and row.price_factor != nxt.price_factor and row.reference_price > 0:
            ratio = row.price_factor / nxt.price_factor
            amount = (row.reference_price - row.reference_price * ratio).quantize(Decimal("0.01"))
            if amount > 0:
                out.append(CorporateAction(symbol=symbol, action_type=CorporateActionType.DIVIDEND, ex_date=ex,
                                           available_at_ts=avail, cash_amount=amount, reference_price=row.reference_price,
                                           feed=DataFeed.LEAN_SAMPLE, source_id=source_id, raw=raw))
    return out


def corporate_actions_from_map(symbol: str, rows: Sequence[MapRow], cal: MarketCalendar,
                               source_id: str) -> list[CorporateAction]:
    out: list[CorporateAction] = []
    for i in range(len(rows) - 1):
        row, nxt = rows[i], rows[i + 1]
        if row.ticker != nxt.ticker:
            try:
                ex = cal.next_session(row.day)
                avail = cal.session_open(ex)
            except (CalendarOutOfRange, ValueError):
                continue
            out.append(CorporateAction(symbol=symbol, action_type=CorporateActionType.RENAME, ex_date=ex,
                                       available_at_ts=avail, old_symbol=row.ticker, new_symbol=nxt.ticker,
                                       feed=DataFeed.LEAN_SAMPLE, source_id=source_id, raw=f"{row.raw}\n{nxt.raw}"))
    if rows and rows[-1].day < FAR_FUTURE:
        last = rows[-1]
        try:
            avail = cal.session_close(last.day) if cal.is_session(last.day) else datetime.combine(
                last.day, time(16), tzinfo=NY).astimezone(UTC)
            out.append(CorporateAction(symbol=symbol, action_type=CorporateActionType.DELISTING, ex_date=last.day,
                                       available_at_ts=avail, old_symbol=last.ticker, feed=DataFeed.LEAN_SAMPLE,
                                       source_id=source_id, raw=last.raw))
        except CalendarOutOfRange:
            pass
    return out


def _sym_ok(stem: str) -> str | None:
    s = stem.upper()
    if s and s[0].isalpha() and len(s) <= 10 and all(c.isalnum() or c == "." for c in s):
        return s
    return None


@dataclass
class LeanSampleRead:
    bar_records: list[dict[str, Any]]
    quote_records: list[dict[str, Any]]
    corporate_actions: list[CorporateAction]
    files: list[str]
    minute_days: dict[str, list[date]]


def read_lean_equity_dir(
    equity_dir: Path,
    *,
    symbols: Iterable[str] | None = None,
    resolutions: Sequence[str] = ("daily", "minute"),
    include_quotes: bool = True,
    include_corporate_actions: bool = True,
    source_id: str,
    received: datetime,
    cal: MarketCalendar | None = None,
) -> LeanSampleRead:
    """Parse a LEAN ``equity/usa`` directory (the bundled sample or one written by ``lean_export``)."""
    cal = cal or get_calendar()
    wanted = {s.upper() for s in symbols} if symbols else None
    bars: list[dict[str, Any]] = []
    quotes: list[dict[str, Any]] = []
    files: list[str] = []
    minute_days: dict[str, list[date]] = {}
    if "daily" in resolutions and (equity_dir / "daily").is_dir():
        for z in sorted((equity_dir / "daily").glob("*.zip")):
            sym = _sym_ok(z.stem)
            if sym is None or (wanted is not None and sym not in wanted):
                continue
            bars += parse_daily_csv(read_zip_text(z), sym, cal=cal, source_id=source_id, received=received)
            files.append(z.relative_to(equity_dir).as_posix())
    if "minute" in resolutions and (equity_dir / "minute").is_dir():
        for sym_dir in sorted(p for p in (equity_dir / "minute").iterdir() if p.is_dir()):
            sym = _sym_ok(sym_dir.name)
            if sym is None or (wanted is not None and sym not in wanted):
                continue
            for z in sorted(sym_dir.glob("*_trade.zip")):
                day = datetime.strptime(z.name[:8], "%Y%m%d").date()  # noqa: DTZ007
                bars += parse_minute_trade_csv(read_zip_text(z), sym, day, cal=cal, source_id=source_id, received=received)
                minute_days.setdefault(sym, []).append(day)
                files.append(z.relative_to(equity_dir).as_posix())
            if include_quotes:
                for z in sorted(sym_dir.glob("*_quote.zip")):
                    day = datetime.strptime(z.name[:8], "%Y%m%d").date()  # noqa: DTZ007
                    quotes += parse_minute_quote_csv(read_zip_text(z), sym, day, source_id=source_id, received=received)
                    files.append(z.relative_to(equity_dir).as_posix())
    actions: list[CorporateAction] = []
    if include_corporate_actions:
        for f in sorted((equity_dir / "factor_files").glob("*.csv")) if (equity_dir / "factor_files").is_dir() else []:
            sym = _sym_ok(f.stem)
            if sym is None or (wanted is not None and sym not in wanted):
                continue
            actions += corporate_actions_from_factors(sym, parse_factor_file(f.read_text()), cal, source_id)
            files.append(f.relative_to(equity_dir).as_posix())
        for f in sorted((equity_dir / "map_files").glob("*.csv")) if (equity_dir / "map_files").is_dir() else []:
            sym = _sym_ok(f.stem)
            if sym is None or (wanted is not None and sym not in wanted):
                continue
            actions += corporate_actions_from_map(sym, parse_map_file(f.read_text()), cal, source_id)
            files.append(f.relative_to(equity_dir).as_posix())
    return LeanSampleRead(bars, quotes, actions, files, minute_days)


def lean_sample_gaps(bars: Sequence[Bar], minute_days: Mapping[str, Sequence[date]],
                     cal: MarketCalendar) -> list[Gap]:
    """Gaps for a LEAN sample: minute gaps per *file day* (each file is one full session, so missing minutes inside
    it are real gaps; days without a file are simply not part of the sample) and daily gaps over each symbol's span."""
    return gaps_within_observed_sessions(bars, cal, intraday_days=minute_days)


def import_lean_sample(
    datasets_dir: Path,
    *,
    lean_root: Path | None = None,
    symbols: Iterable[str] | None = None,
    resolutions: Sequence[str] = ("daily", "minute"),
    include_quotes: bool = True,
    policy: QualityPolicy | None = None,
    db: Database | None = None,
    store: MarketDataStore | None = None,
) -> dict[str, Any]:
    """Import the LEAN sample into the store as one ``LEAN_SAMPLE`` dataset; returns its manifest."""
    root = lean_root or DEFAULT_LEAN_ROOT
    equity_dir = root / "Data" / "equity" / "usa"
    if not equity_dir.is_dir():
        raise FileNotFoundError(f"LEAN sample data not found at {equity_dir}; build/clone LEAN first (make lean-build)")
    cal = get_calendar()
    sha, commit_time = lean_commit(root)
    # received_ts = vintage of the bytes: the LEAN commit time for the pinned clone, otherwise the import time
    received = commit_time or datetime.now(UTC).replace(microsecond=0)
    source_id = f"lean-sample:QuantConnect/Lean@{sha[:12]}"
    read = read_lean_equity_dir(equity_dir, symbols=symbols, resolutions=resolutions, include_quotes=include_quotes,
                                source_id=source_id, received=received, cal=cal)
    # Historical bulk import: received_ts is the snapshot vintage (years after the exchange time), so the live
    # 'stale quote' check (receipt delay) is meaningless here and is disabled by default; all other checks apply.
    pol = policy or HISTORICAL_IMPORT_POLICY
    ca_dates: dict[str, set[date]] = {}
    for a in read.corporate_actions:
        if a.action_type in (CorporateActionType.SPLIT, CorporateActionType.DIVIDEND):
            ca_dates.setdefault(a.symbol, set()).add(a.ex_date)
    breport = validate_bars(read.bar_records, cal, replace(pol, report_gaps=False), corporate_action_dates=ca_dates,
                            validated_at=received)
    breport = with_gaps(breport, lean_sample_gaps(breport.accepted, read.minute_days, cal) if pol.report_gaps else [])
    qreport = validate_quotes(read.quote_records, cal, pol, validated_at=received)
    minute_gaps = [g.as_dict() for g in breport.gaps if g.timeframe == "1Min"]
    coverage = _coverage(breport.accepted, read.minute_days)
    spec = DatasetSpec(
        source_id=source_id,
        feed=DataFeed.LEAN_SAMPLE,
        generator=IMPORTER,
        generator_version=IMPORTER_VERSION,
        is_synthetic=False,
        kind="lean_sample_import",
        series_mode=SERIES_SNAPSHOT,
        description="QuantConnect LEAN repository sample equity data (daily + minute trade/quote bars, factor/map files)",
        limitations=(*LEAN_SAMPLE_LIMITATIONS, _period_limitation(coverage)),
        params={"lean_root": "trading/lean/Lean", "resolutions": list(resolutions), "include_quotes": include_quotes,
                "symbols": sorted({s.upper() for s in symbols}) if symbols else "all", "policy": _policy_dict(pol)},
        lean_commit=sha,
        vintage=iso(received),
        extra={
            "source_files": len(read.files),
            "coverage": coverage,
            "validation": {"bars": _stable(breport), "quotes": _stable(qreport)},
            "minute_gaps": minute_gaps[:200],
            "minute_gap_count": len(minute_gaps),
            "daily_gap_count": sum(1 for g in breport.gaps if g.timeframe == "1Day"),
            "received_ts_semantics": "LEAN commit timestamp (vintage of the source snapshot)",
        },
    )
    st = store or MarketDataStore(datasets_dir)
    manifest = st.write_dataset(
        spec,
        bars=list(breport.accepted),
        quotes=list(qreport.accepted),
        corporate_actions=read.corporate_actions,
        bar_extras=breport.raw_extras(rejected_raw_json(breport)),
        quote_extras=qreport.raw_extras(rejected_raw_json(qreport)),
        created_at=datetime.now(UTC),
    )
    if db is not None:
        ctx = {"dataset_version": manifest["dataset_version"], "source_id": source_id}
        record_quality_events(db, breport, DataFeed.LEAN_SAMPLE, context=ctx)
        record_quality_events(db, qreport, DataFeed.LEAN_SAMPLE, context=ctx)
    return manifest


def _coverage(bars: Sequence[Bar], minute_days: Mapping[str, Sequence[date]]) -> dict[str, Any]:
    daily: dict[str, list[date]] = {}
    for b in bars:
        if b.timeframe == "1Day":
            daily.setdefault(b.symbol, []).append(b.start_ts.astimezone(NY).date())
    all_minute = sorted({d for ds in minute_days.values() for d in ds})
    return {
        "minute_file_days": {s: [d.isoformat() for d in sorted(ds)] for s, ds in sorted(minute_days.items())},
        "minute_years": sorted({d.year for d in all_minute}),
        "daily_range": {s: [min(ds).isoformat(), max(ds).isoformat()] for s, ds in sorted(daily.items())},
    }


def _period_limitation(coverage: Mapping[str, Any]) -> str:
    ranges = list(coverage.get("daily_range", {}).values())
    daily = f"daily bars {min(r[0] for r in ranges)}..{max(r[1] for r in ranges)}" if ranges else "no daily bars"
    years = coverage.get("minute_years") or []
    minute = f"minute bars only on a few sample days in {', '.join(str(y) for y in years)}" if years else "no minute bars"
    return f"Data period: {daily}; {minute} — old historical sample, not current market data"


def _stable(report: ValidationReport) -> dict[str, Any]:
    s = report.summary()
    s.pop("validated_at", None)
    return s


def _policy_dict(p: QualityPolicy) -> dict[str, str]:
    from dataclasses import asdict

    return {k: str(v.value if hasattr(v, "value") else v) for k, v in asdict(p).items()}
