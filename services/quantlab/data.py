"""Bulk, resumable, provenance-recorded research datasets from free Alpaca market data.

* U.S. equities/ETFs: daily SIP bars (historical SIP is free when older than 15 minutes) for every active research
  instrument AND every delisted/renamed symbol recoverable from the asset list and corporate actions (mergers,
  worthless removals, name changes) — this reduces, but does not eliminate, survivorship bias (corporate-action
  coverage starts later than the bar history; see ``delisted_coverage`` in the manifest).
  Both ``raw`` (point-in-time price levels for filters) and ``all`` (split+dividend adjusted, for returns) are kept.
* Crypto: hourly and daily bars for every tradable USD-quoted pair (stablecoin/USD pairs excluded).
* Intraday stock bars for selected symbol-days (stocks-in-play research) via :func:`fetch_intraday_symbol_days`.

Every dataset directory has ``manifest.json``: request parameters, knowledge time (``fetched_at``), per-batch row
counts and SHA-256 file hashes. Re-running resumes completed batches; nothing is overwritten silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.capabilities.http import DATA_BASE, ReadOnlyAlpacaHttp
from services.market_data.calendar import get_calendar

log = logging.getLogger("lat.quantlab.data")

EQUITY_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
STABLE = frozenset({"USDC", "USDT", "USDG", "DAI", "PYUSD", "USD"})
BAR_COLUMNS = ["symbol", "t", "o", "h", "l", "c", "v", "n", "vw"]
DEFAULT_START = date(2016, 1, 1)


def last_completed_session(now: datetime | None = None, settle: timedelta = timedelta(minutes=20)) -> date:
    cal = get_calendar()
    now = now or datetime.now(UTC)
    d = cal.ny_date(now)
    if not cal.is_session(d) or cal.session_close(d) + settle > now:
        d = cal.previous_session(d)
    return d


def _nunique(df: pd.DataFrame) -> int:
    return len(set(df["symbol"].tolist())) if len(df) else 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bars_frame(bars: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = [(sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("n", 0), b.get("vw"))
            for sym, items in bars.items() for b in items]
    df = pd.DataFrame(rows, columns=BAR_COLUMNS)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True)
    for col in ("o", "h", "l", "c", "v", "vw"):
        df[col] = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index).astype("float64")
    df["n"] = pd.Series(pd.to_numeric(df["n"], errors="coerce"), index=df.index).fillna(0).astype("int64")
    return df.sort_values(["symbol", "t"]).drop_duplicates(["symbol", "t"], keep="last").reset_index(drop=True)


class Manifest:
    def __init__(self, directory: Path, params: dict[str, Any]) -> None:
        self.dir = directory
        self.path = directory / "manifest.json"
        directory.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self.data = json.loads(self.path.read_text())
            if self.data.get("params") != params:
                raise ValueError(f"{directory}: existing dataset was fetched with different parameters; "
                                 "use a new directory (datasets are immutable)")
        else:
            self.data = {"params": params, "batches": {}, "created_at": datetime.now(UTC).isoformat()}

    def done(self, key: str) -> bool:
        entry = self.data["batches"].get(key)
        return bool(entry) and (self.dir / entry["file"]).is_file()

    def record(self, key: str, file: str, rows: int, symbols: int, extra: dict[str, Any] | None = None) -> None:
        self.data["batches"][key] = {"file": file, "rows": rows, "symbols": symbols,
                                     "sha256": _sha(self.dir / file) if rows else None,
                                     "fetched_at": datetime.now(UTC).isoformat(), **(extra or {})}
        self.flush()

    def flush(self) -> None:
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)


# ------------------------------------------------------------------------------------------- symbol discovery
def discover_delisted(http: ReadOnlyAlpacaHttp, start: date, end: date) -> pd.DataFrame:
    """Symbols that left the market (acquired, worthless, renamed) per Alpaca corporate actions."""
    types = "cash_merger,stock_merger,stock_and_cash_merger,worthless_removal,name_change"
    rows: list[dict[str, Any]] = []
    cur = start
    while cur <= end:
        nxt = min(date(cur.year, 12, 31), end)
        data = http.paginate(f"{DATA_BASE}/v1/corporate-actions",
                             {"types": types, "start": cur.isoformat(), "end": nxt.isoformat(), "limit": 1000},
                             items_key="corporate_actions")
        assert isinstance(data, dict)
        for kind, items in data.items():
            for it in items:
                sym = it.get("acquiree_symbol") or it.get("old_symbol") or it.get("symbol")
                when = it.get("effective_date") or it.get("process_date")
                if sym and EQUITY_SYMBOL.match(str(sym)):
                    rows.append({"symbol": sym, "event": kind, "date": when,
                                 "new_symbol": it.get("new_symbol") or it.get("acquirer_symbol") or None})
        cur = nxt + timedelta(days=1)
    return pd.DataFrame(rows, columns=["symbol", "event", "date", "new_symbol"])


def research_equity_symbols(assets: pd.DataFrame, delisted: pd.DataFrame | None = None) -> list[str]:
    eq = assets[(assets["asset_class"] == "us_equity") & assets["research_kind"] & (assets["exchange"] != "OTC")]
    syms = {s for s in eq["symbol"].astype(str) if EQUITY_SYMBOL.match(s)}
    if delisted is not None and not delisted.empty:
        known_otc = set(assets.loc[assets["exchange"] == "OTC", "symbol"].astype(str))
        syms |= {s for s in delisted["symbol"].astype(str) if EQUITY_SYMBOL.match(s) and s not in known_otc}
    return sorted(syms)


def exlisted_candidates(assets: pd.DataFrame, known: set[str], delisted: pd.DataFrame | None = None) -> list[str]:
    """Symbols that may hold exchange-listed SIP history under a ticker Alpaca no longer lists as an exchange asset.

    Alpaca keys asset records by the CURRENT ticker: a listed company that failed or moved to OTC shows up as an OTC
    asset (SVB: SIVB -> SIVBQ, First Republic: FRC -> FRCB, Signature: SBNY on OTC) while its exchange-era SIP bars
    remain under the old ticker. SIP carries no OTC trades, so probing these candidates returns only exchange-listed
    history. Candidates: every OTC asset symbol, plus the ticker with a trailing bankruptcy 'Q' removed, plus the
    5-letter OTC form with its trailing class letter removed (FRCB -> FRC)."""
    eq = assets[assets["asset_class"] == "us_equity"]
    otc = eq[(eq["exchange"] == "OTC") | (eq["status"] == "inactive")]
    cands: set[str] = set()
    for sym in otc["symbol"].astype(str):
        if not EQUITY_SYMBOL.match(sym):
            continue
        cands.add(sym)
        if len(sym) >= 4 and sym.endswith("Q"):
            cands.add(sym[:-1])
        if len(sym) in (4, 5) and (eq.loc[eq["symbol"] == sym, "exchange"] == "OTC").any():
            base = sym[:-1]
            if len(base) >= 2:
                cands.add(base)
    if delisted is not None and not delisted.empty:  # e.g. SIVBQ (worthless removal) -> SIVB
        for sym in delisted["symbol"].astype(str):
            if len(sym) >= 4 and sym.endswith("Q"):
                cands.add(sym[:-1])
    return sorted(c for c in cands if c not in known and EQUITY_SYMBOL.match(c))


def crypto_usd_pairs(assets: pd.DataFrame) -> list[str]:
    cr = assets[(assets["asset_class"] == "crypto") & assets["tradable"]]
    out = []
    for s in cr["symbol"].astype(str):
        base, _, quote = s.partition("/")
        if quote == "USD" and base not in STABLE:
            out.append(s)
    return sorted(out)


# ------------------------------------------------------------------------------------------- equity daily
def fetch_equity_daily(http: ReadOnlyAlpacaHttp, symbols: Sequence[str], out_dir: Path, *, adjustment: str,
                       start: date = DEFAULT_START, end: date | None = None, batch_size: int = 100,
                       workers: int = 3, progress: Callable[[int, int], None] | None = None) -> Manifest:
    end = end or last_completed_session()
    params = {"kind": "equity_daily", "feed": "sip", "timeframe": "1Day", "adjustment": adjustment,
              "start": start.isoformat(), "end": end.isoformat(), "batch_size": batch_size,
              "symbols_sha256": hashlib.sha256(",".join(symbols).encode()).hexdigest()}
    man = Manifest(out_dir, params)
    batches = [list(symbols[i:i + batch_size]) for i in range(0, len(symbols), batch_size)]
    end_ts = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)

    def work(idx: int) -> None:
        key = f"b{idx:05d}"
        if man.done(key):
            return
        data = http.paginate(f"{DATA_BASE}/v2/stocks/bars",
                             {"symbols": ",".join(batches[idx]), "timeframe": "1Day", "start": start.isoformat(),
                              "end": end_ts.isoformat(), "feed": "sip", "adjustment": adjustment, "limit": 10000,
                              "sort": "asc"}, items_key="bars")
        df = _bars_frame(data if isinstance(data, dict) else {})
        file = f"{key}.parquet"
        if not df.empty:
            df.to_parquet(out_dir / file, index=False)
        man.record(key, file, len(df), _nunique(df),
                   {"requested": len(batches[idx])})

    pending = [i for i in range(len(batches)) if not man.done(f"b{i:05d}")]
    done = len(batches) - len(pending)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(work, pending):
            done += 1
            if progress:
                progress(done, len(batches))
    man.data["complete"] = all(man.done(f"b{i:05d}") or man.data["batches"].get(f"b{i:05d}", {}).get("rows") == 0
                               for i in range(len(batches)))
    man.flush()
    return man


def load_bars(directory: Path, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no bar files in {directory}")
    want = set(symbols) if symbols is not None else None
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if want is not None:
            df = df[df["symbol"].isin(sorted(want))]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "t"]).drop_duplicates(["symbol", "t"], keep="last").reset_index(drop=True)


# ------------------------------------------------------------------------------------------- crypto
def fetch_crypto_bars(http: ReadOnlyAlpacaHttp, pairs: Sequence[str], out_dir: Path, *, timeframe: str,
                      start: date = date(2021, 1, 1), end: datetime | None = None) -> Manifest:
    end = end or datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    params = {"kind": "crypto", "timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
              "pairs": list(pairs), "location": "us"}
    man = Manifest(out_dir, params)
    for pair in pairs:
        key = pair.replace("/", "_")
        if man.done(key):
            continue
        data = http.paginate(f"{DATA_BASE}/v1beta3/crypto/us/bars",
                             {"symbols": pair, "timeframe": timeframe, "start": start.isoformat(),
                              "end": end.isoformat(), "limit": 10000, "sort": "asc"}, items_key="bars")
        df = _bars_frame(data if isinstance(data, dict) else {})
        file = f"{key}.parquet"
        if not df.empty:
            df.to_parquet(out_dir / file, index=False)
        man.record(key, file, len(df), 1 if len(df) else 0)
    man.data["complete"] = True
    man.flush()
    return man


# ------------------------------------------------------------------------------------------- intraday symbol-days
def fetch_intraday_symbol_days(http: ReadOnlyAlpacaHttp, plan: dict[date, list[str]], out_dir: Path, *,
                               timeframe: str = "5Min", tag: str = "sip5m",
                               progress: Callable[[int, int], None] | None = None) -> Manifest:
    """Regular-session bars for a {session date: [symbols]} plan (one request per day, batched symbols)."""
    cal = get_calendar()
    params = {"kind": "intraday_symbol_days", "timeframe": timeframe, "feed": "sip", "tag": tag}
    man = Manifest(out_dir, params)
    days = sorted(plan)
    for i, d in enumerate(days):
        key = d.isoformat()
        syms = sorted(set(plan[d]))
        if not syms or man.done(key):
            continue
        start, end = cal.session_open(d), cal.session_close(d)
        frames = []
        for j in range(0, len(syms), 100):
            data = http.paginate(f"{DATA_BASE}/v2/stocks/bars",
                                 {"symbols": ",".join(syms[j:j + 100]), "timeframe": timeframe,
                                  "start": start.isoformat(), "end": (end - timedelta(seconds=1)).isoformat(),
                                  "feed": "sip", "adjustment": "raw", "limit": 10000, "sort": "asc"},
                                 items_key="bars")
            frames.append(_bars_frame(data if isinstance(data, dict) else {}))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLUMNS)
        file = f"{key}.parquet"
        if not df.empty:
            df.to_parquet(out_dir / file, index=False)
        man.record(key, file, len(df), _nunique(df), {"requested": len(syms)})
        if progress:
            progress(i + 1, len(days))
    man.data["complete"] = True
    man.flush()
    return man


def merge_with_deltas(base: pd.DataFrame, deltas: list[pd.DataFrame], *, adjusted: bool) -> pd.DataFrame:
    """Append later delta fetches to a base bar set. Adjusted series are rescaled per symbol to the newest basis
    using the median price ratio on overlapping days (a split between fetches changes the back-adjustment)."""
    out = base
    for delta in deltas:
        if delta.empty:
            continue
        if adjusted and not out.empty:
            ov = out.merge(delta[["symbol", "t", "c"]], on=["symbol", "t"], suffixes=("", "_new"))
            if not ov.empty:
                rs = ov["c_new"].to_numpy(dtype=float) / ov["c"].to_numpy(dtype=float)
                osym = ov["symbol"].to_numpy().astype(str)
                ratio = {k: float(np.median(rs[osym == k])) for k in np.unique(osym)}
                f = np.array([ratio.get(str(x), 1.0) for x in out["symbol"].to_numpy()], dtype=float)
                out = out.copy()
                for col in ("o", "h", "l", "c", "vw"):
                    out[col] = out[col].to_numpy(dtype=float) * f
                out["v"] = out["v"].to_numpy(dtype=float) / f
        out = pd.concat([out, delta], ignore_index=True)
        out = out.sort_values(["symbol", "t"]).drop_duplicates(["symbol", "t"], keep="last").reset_index(drop=True)
    return out


EARNINGS_RE = re.compile(r"\b(earnings|EPS|quarterly results|Q[1-4] (results|revenue)|guidance|beats|misses)\b",
                         re.I)


def fetch_news_counts(http: ReadOnlyAlpacaHttp, plan: dict[date, list[str]], out_path: Path) -> pd.DataFrame:
    """Pre-open news per (session, symbol): headlines published between the previous session close and 09:30 ET.
    Point-in-time by construction (only items created before the decision time are counted)."""
    cal = get_calendar()
    done: pd.DataFrame | None = pd.read_parquet(out_path) if out_path.is_file() else None
    seen = set(done["session"].astype(str)) if done is not None else set()
    rows: list[dict[str, Any]] = []
    for d in sorted(plan):
        if d.isoformat() in seen:
            continue
        syms = [s for s in plan[d] if s != "SPY"]
        if not syms:
            continue
        start = cal.session_close(cal.previous_session(d))
        end = cal.session_open(d)
        items = http.paginate(f"{DATA_BASE}/v1beta1/news", {"symbols": ",".join(syms), "start": start.isoformat(),
                                                             "end": end.isoformat(), "limit": 50, "sort": "asc"},
                              items_key="news", max_pages=20)
        counts: dict[str, list[int]] = {s: [0, 0] for s in syms}
        for it in items if isinstance(items, list) else []:
            earn = bool(EARNINGS_RE.search(str(it.get("headline", ""))))
            for s in it.get("symbols") or []:
                if s in counts:
                    counts[s][0] += 1
                    counts[s][1] += int(earn)
        rows.extend({"session": d.isoformat(), "symbol": s, "news": c[0], "earnings_news": c[1]}
                    for s, c in counts.items())
    new = pd.DataFrame(rows, columns=["session", "symbol", "news", "earnings_news"])
    out = pd.concat([done, new], ignore_index=True) if done is not None else new
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
