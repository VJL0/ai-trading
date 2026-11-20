"""Export stored bars (e.g. the SYNTHETIC fixture) to LEAN's on-disk equity format so a self-hosted LEAN
backtest can run on them.

Writes under ``<lean_data_dir>/equity/usa/``:

* ``minute/<sym>/<yyyyMMdd>_trade.zip`` -> ``<yyyyMMdd>_<sym>_minute_trade.csv`` ``ms,o,h,l,c,v`` (prices x10000)
* ``minute/<sym>/<yyyyMMdd>_quote.zip`` (optional, from quotes) -> quote bars ``ms,bo,bh,bl,bc,bsz,ao,ah,al,ac,asz``
* ``daily/<sym>.zip`` -> ``<sym>.csv`` ``yyyyMMdd 00:00,o,h,l,c,v`` (derived from regular-session minutes if absent)
* ``map_files/<sym>.csv`` and ``factor_files/<sym>.csv`` (identity factors: no corporate actions in synthetic data)

and optionally copies ``market-hours/``, ``symbol-properties/`` and ``alternative/interest-rate/`` from a LEAN
``Data`` folder so the output directory can be used directly as LEAN's ``data-folder``. Verified against the real
engine: LEAN's BasicTemplateAlgorithm on SPY minute bars re-exported by this module produces statistics identical to
the run on LEAN's own bundled sample files. Zip entries use a fixed timestamp so identical
input yields byte-identical archives. Prices that are not representable in LEAN's x10000 integer format, or
fractional volumes, raise :class:`LeanExportError` (never silently rounded).
"""

from __future__ import annotations

import shutil
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import MarketSession
from packages.contracts.models import Bar, Quote

from .calendar import NY
from .common import is_daily

ZIP_DATE = (1980, 1, 1, 0, 0, 0)
DEFAULT_EXCHANGE_CODES: Mapping[str, str] = {"SPY": "P", "IWM": "P", "DIA": "P", "GLD": "P", "TLT": "Q"}
_SCALE = Decimal(10_000)


class LeanExportError(ValueError):
    pass


def _scaled(value: Decimal, what: str) -> int:
    v = value * _SCALE
    if v != v.to_integral_value():
        raise LeanExportError(f"{what}={value} is not representable in LEAN's x10000 integer price format")
    return int(v)


def _whole(value: Decimal, what: str) -> int:
    if value != value.to_integral_value():
        raise LeanExportError(f"{what}={value} is fractional; LEAN equity volumes/sizes are integers")
    return int(value)


def _ms_since_midnight(ts: datetime) -> int:
    local = ts.astimezone(NY)
    midnight = datetime.combine(local.date(), time(0), tzinfo=NY)
    return int((local.replace(tzinfo=None) - midnight.replace(tzinfo=None)) / timedelta(milliseconds=1))


def _write_zip(path: Path, entry: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(entry, date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, text)
    tmp.replace(path)


def _minute_line(b: Bar) -> str:
    return ",".join(str(x) for x in (
        _ms_since_midnight(b.start_ts), _scaled(b.open, "open"), _scaled(b.high, "high"), _scaled(b.low, "low"),
        _scaled(b.close, "close"), _whole(b.volume, "volume")))


def _daily_line(b: Bar) -> str:
    day = b.start_ts.astimezone(NY).date()
    return ",".join([f"{day:%Y%m%d} 00:00", *(str(x) for x in (
        _scaled(b.open, "open"), _scaled(b.high, "high"), _scaled(b.low, "low"), _scaled(b.close, "close"),
        _whole(b.volume, "volume")))])


def derive_daily(minute_bars: Sequence[Bar]) -> list[Bar]:
    """Aggregate REGULAR-session minute bars into one bar per (symbol, NY date)."""
    groups: dict[tuple[str, date], list[Bar]] = defaultdict(list)
    for b in minute_bars:
        if b.session == MarketSession.REGULAR:
            groups[(b.symbol, b.start_ts.astimezone(NY).date())].append(b)
    out: list[Bar] = []
    for (_sym, _day), bars in sorted(groups.items()):
        bars.sort(key=lambda x: x.start_ts)
        first, last = bars[0], bars[-1]
        out.append(first.model_copy(update={
            "timeframe": "1Day", "end_ts": last.end_ts, "high": max(x.high for x in bars),
            "low": min(x.low for x in bars), "close": last.close, "volume": sum((x.volume for x in bars), Decimal(0)),
            "vwap": None, "trade_count": None, "available_at_ts": last.available_at_ts, "received_ts": last.received_ts,
        }))
    return out


def _quote_bar_lines(quotes: Sequence[Quote]) -> dict[tuple[str, date], list[str]]:
    by_min: dict[tuple[str, date, int], list[Quote]] = defaultdict(list)
    for q in quotes:
        local = q.exchange_ts.astimezone(NY)
        minute_start = local.replace(second=0, microsecond=0)
        by_min[(q.symbol, local.date(), _ms_since_midnight(minute_start))].append(q)
    out: dict[tuple[str, date], list[str]] = defaultdict(list)
    for (sym, day, ms), qs in sorted(by_min.items()):
        qs.sort(key=lambda q: q.exchange_ts)
        bids = [q.bid for q in qs]
        asks = [q.ask for q in qs]
        out[(sym, day)].append(",".join(str(x) for x in (
            ms, _scaled(bids[0], "bid"), _scaled(max(bids), "bid"), _scaled(min(bids), "bid"), _scaled(bids[-1], "bid"),
            _whole(qs[-1].bid_size, "bid_size"), _scaled(asks[0], "ask"), _scaled(max(asks), "ask"),
            _scaled(min(asks), "ask"), _scaled(asks[-1], "ask"), _whole(qs[-1].ask_size, "ask_size"))))
    return out


def export_bars_to_lean(
    bars: Sequence[Bar],
    lean_data_dir: Path,
    *,
    quotes: Sequence[Quote] = (),
    market: str = "usa",
    derive_daily_if_missing: bool = True,
    write_map_and_factor_files: bool = True,
    exchange_codes: Mapping[str, str] | None = None,
    copy_metadata_from: Path | None = None,
) -> dict[str, Any]:
    """Write LEAN-format files; returns a summary (files, symbols, row counts)."""
    root = Path(lean_data_dir) / "equity" / market
    codes = {**DEFAULT_EXCHANGE_CODES, **(exchange_codes or {})}
    minute = sorted((b for b in bars if b.timeframe == "1Min"), key=lambda b: (b.symbol, b.start_ts))
    daily = sorted((b for b in bars if is_daily(b.timeframe)), key=lambda b: (b.symbol, b.start_ts))
    other = [b for b in bars if b.timeframe != "1Min" and not is_daily(b.timeframe)]
    if other:
        raise LeanExportError(f"unsupported timeframe(s) for LEAN export: {sorted({b.timeframe for b in other})}")
    daily_syms = {b.symbol for b in daily}
    if derive_daily_if_missing:
        daily += [b for b in derive_daily(minute) if b.symbol not in daily_syms]
        daily.sort(key=lambda b: (b.symbol, b.start_ts))
    files: list[str] = []
    rows = {"minute": 0, "daily": 0, "quote_minutes": 0}

    by_day: dict[tuple[str, date], list[Bar]] = defaultdict(list)
    for b in minute:
        by_day[(b.symbol, b.start_ts.astimezone(NY).date())].append(b)
    for (sym, day), day_bars in sorted(by_day.items()):
        s = sym.lower()
        path = root / "minute" / s / f"{day:%Y%m%d}_trade.zip"
        _write_zip(path, f"{day:%Y%m%d}_{s}_minute_trade.csv", "\n".join(_minute_line(b) for b in day_bars) + "\n")
        files.append(path.relative_to(lean_data_dir).as_posix())
        rows["minute"] += len(day_bars)
    for (sym, day), lines in sorted(_quote_bar_lines(quotes).items()):
        s = sym.lower()
        path = root / "minute" / s / f"{day:%Y%m%d}_quote.zip"
        _write_zip(path, f"{day:%Y%m%d}_{s}_minute_quote.csv", "\n".join(lines) + "\n")
        files.append(path.relative_to(lean_data_dir).as_posix())
        rows["quote_minutes"] += len(lines)
    by_sym: dict[str, list[Bar]] = defaultdict(list)
    for b in daily:
        by_sym[b.symbol].append(b)
    for sym, sym_bars in sorted(by_sym.items()):
        s = sym.lower()
        path = root / "daily" / f"{s}.zip"
        _write_zip(path, f"{s}.csv", "\n".join(_daily_line(b) for b in sym_bars) + "\n")
        files.append(path.relative_to(lean_data_dir).as_posix())
        rows["daily"] += len(sym_bars)

    symbols = sorted({b.symbol for b in minute} | set(by_sym))
    if write_map_and_factor_files:
        for sym in symbols:
            s = sym.lower()
            first = min(b.start_ts.astimezone(NY).date() for b in (*minute, *daily) if b.symbol == sym)
            exch = codes.get(sym, "Q")
            map_path = root / "map_files" / f"{s}.csv"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_text(f"{first:%Y%m%d},{s},{exch}\n20501231,{s},{exch}\n")
            fac_path = root / "factor_files" / f"{s}.csv"
            fac_path.parent.mkdir(parents=True, exist_ok=True)
            fac_path.write_text(f"{first:%Y%m%d},1,1,0\n20501231,1,1,0\n")
            files += [map_path.relative_to(lean_data_dir).as_posix(), fac_path.relative_to(lean_data_dir).as_posix()]
    copied: list[str] = []
    if copy_metadata_from is not None:
        for sub in ("market-hours", "symbol-properties", "alternative/interest-rate"):
            src = Path(copy_metadata_from) / sub
            if src.is_dir():
                dst = Path(lean_data_dir) / sub
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied.append(sub)
    return {"lean_data_dir": str(lean_data_dir), "market": market, "symbols": symbols, "rows": rows,
            "files": sorted(files), "metadata_copied": copied,
            "is_synthetic": any(b.is_synthetic for b in bars),
            "dataset_versions": sorted({b.dataset_version for b in bars})}
