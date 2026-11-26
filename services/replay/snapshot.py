"""Bar snapshots for the replay harness: validated, content-hashed, point-in-time 5-minute regular-session bars.

Input frame (the research panel's raw format, ``services.quantlab.intraday.IntradayPanel.build``): one row per bar
with ``t`` (bar START, tz-aware or UTC), ``symbol``, ``o``, ``h``, ``l``, ``c``, ``v`` and optional ``vw``. A bar is
knowable only at its END (``t + 5 min``): every consumer here (quotes, the runtime's bar frame, the research panel)
respects that availability, and nothing after the decision time is visible.

The protected QuantLab holdout (``services.quantlab.gates.HOLDOUT_START`` ... ``HOLDOUT_END``) is never replayed from
real data: :meth:`BarSnapshot.assert_outside_holdout` refuses it (fail closed). Synthetic snapshots carry
``is_synthetic=True`` and the SYNTHETIC feed, and are labelled as such everywhere downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.contracts.enums import DataFeed
from services.order_ledger.sessions import session_bounds
from services.order_ledger.timeutil import NY_TZ

BAR_MINUTES = 5
BAR = timedelta(minutes=BAR_MINUTES)
TIMEFRAME = "5Min"
REQUIRED = ("t", "symbol", "o", "h", "l", "c", "v")
DEFAULT_PRICES: dict[str, float] = {"SPY": 500.0, "QQQ": 430.0, "AAPL": 190.0, "MSFT": 410.0, "IWM": 200.0}


class SnapshotError(ValueError):
    """The bar snapshot violates the input contract (fail closed; never repaired)."""


@dataclass(frozen=True)
class Provenance:
    source: str
    feed: DataFeed
    is_synthetic: bool
    dataset_version: str
    content_sha256: str
    timeframe: str = TIMEFRAME
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["feed"] = self.feed.value
        return d


def session_open_close(day: date) -> tuple[datetime, datetime] | None:
    """XNYS (open, close) of ``day`` in UTC (holidays -> None; early closes included)."""
    return session_bounds(datetime.combine(day, time(12), tzinfo=NY_TZ))


def xnys_sessions(start: date, count: int) -> list[date]:
    out: list[date] = []
    d = start
    guard = 0
    while len(out) < count:
        if session_open_close(d) is not None:
            out.append(d)
        d += timedelta(days=1)
        guard += 1
        if guard > 400 + 3 * count:
            raise SnapshotError(f"could not find {count} XNYS sessions from {start} (calendar range?)")
    return out


class BarSnapshot:
    """An immutable set of 5-minute bars plus provenance and per-session daily aggregates."""

    def __init__(self, bars: pd.DataFrame, provenance: Provenance) -> None:
        self._bars = bars.copy()
        self.provenance = provenance
        self.symbols: tuple[str, ...] = tuple(sorted(set(bars["symbol"].astype(str))))
        days = sorted(set(bars["day"]))
        self.sessions: list[date] = [d for d in days if session_open_close(d) is not None]
        daily = bars.groupby(["symbol", "day"], sort=True).agg(volume=("v", "sum"), close=("c", "last"),
                                                              open=("o", "first"))
        self._daily = daily
        self._adv_cache: dict[tuple[str, date, int], float | None] = {}

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars.copy()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_frame(cls, df: pd.DataFrame, *, source: str, feed: DataFeed, note: str = "",
                   dataset_version: str | None = None) -> BarSnapshot:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise SnapshotError(f"bar snapshot missing columns {missing}")
        if df.empty:
            raise SnapshotError("bar snapshot is empty")
        out = df.loc[:, [c for c in (*REQUIRED, "vw") if c in df.columns]].copy()
        if out[list(REQUIRED)].isna().any().any():
            raise SnapshotError("null values in the bar snapshot; gaps must be missing rows, never NaN fields")
        out["t"] = pd.to_datetime(out["t"], utc=True)
        out["symbol"] = out["symbol"].astype(str)
        for col in ("o", "h", "l", "c", "v"):
            out[col] = out[col].astype(float)
        if "vw" not in out.columns:
            out["vw"] = out["c"]
        out["vw"] = out["vw"].astype(float).fillna(out["c"])
        if not np.isfinite(out[["o", "h", "l", "c", "v", "vw"]].to_numpy()).all():
            raise SnapshotError("non-finite price or volume in the bar snapshot")
        if (out[["o", "h", "l", "c", "vw"]] <= 0).any().any():
            raise SnapshotError("non-positive price in the bar snapshot")
        if (out["v"] < 0).any():
            raise SnapshotError("negative volume in the bar snapshot")
        if not ((out["h"] >= out[["o", "c"]].max(axis=1)) & (out["l"] <= out[["o", "c"]].min(axis=1))).all():
            raise SnapshotError("OHLC inconsistent: high/low do not bound open/close")
        if out.duplicated(["t", "symbol"]).any():
            raise SnapshotError("duplicate (t, symbol) bars")
        ny = pd.DatetimeIndex(out["t"]).tz_convert(NY_TZ)
        if ((ny.minute % BAR_MINUTES) != 0).any() or (ny.second != 0).any() or (ny.microsecond != 0).any() \
                or (ny.nanosecond != 0).any():
            raise SnapshotError(f"bars must start on the {BAR_MINUTES}-minute grid")
        out["day"] = [d.date() for d in ny]
        ok = np.zeros(len(out), dtype=bool)
        bounds: dict[date, tuple[datetime, datetime] | None] = {}
        for i, (ts, d) in enumerate(zip(out["t"], out["day"], strict=True)):
            b = bounds.setdefault(d, session_open_close(d))
            start = ts.to_pydatetime()
            ok[i] = b is not None and b[0] <= start and start + BAR <= b[1]
        if not ok.all():
            raise SnapshotError(f"{int((~ok).sum())} bar(s) outside the XNYS regular session (regular session only)")
        out = out.sort_values(["t", "symbol"], kind="mergesort").reset_index(drop=True)
        canon = out[["t", "symbol", "o", "h", "l", "c", "v", "vw"]].copy()
        canon["t"] = canon["t"].map(lambda x: x.isoformat())
        digest = hashlib.sha256(canon.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
        is_synth = feed == DataFeed.SYNTHETIC
        prov = Provenance(source=source, feed=feed, is_synthetic=is_synth,
                          dataset_version=dataset_version or f"replay_{digest[:16]}", content_sha256=digest, note=note)
        return cls(out, prov)

    @classmethod
    def load(cls, path: Path, *, feed: DataFeed | None = None, source: str | None = None) -> BarSnapshot:
        """CSV or parquet bars; an optional ``<file>.provenance.json`` sidecar gives ``feed``/``source``/``note``."""
        side = path.with_name(path.name + ".provenance.json")
        meta: dict[str, Any] = json.loads(side.read_text()) if side.is_file() else {}
        df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
        fd = feed or DataFeed(meta.get("feed", DataFeed.SIP.value))
        return cls.from_frame(df, source=source or str(meta.get("source", path.name)), feed=fd,
                              note=str(meta.get("note", "")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._bars[["t", "symbol", "o", "h", "l", "c", "v", "vw"]].to_csv(path, index=False)
        path.with_name(path.name + ".provenance.json").write_text(json.dumps(
            {"feed": self.provenance.feed.value, "source": self.provenance.source, "note": self.provenance.note},
            indent=1))

    # ------------------------------------------------------------------ holdout
    def holdout_sessions(self) -> list[date]:
        from services.quantlab.gates import HOLDOUT_END, HOLDOUT_START

        return [d for d in self.sessions if HOLDOUT_START <= d <= HOLDOUT_END]

    def assert_outside_holdout(self) -> None:
        """Real data inside the protected holdout is refused (synthetic data has no holdout)."""
        if self.provenance.is_synthetic:
            return
        inside = self.holdout_sessions()
        if inside:
            raise SnapshotError(f"{len(inside)} session(s) inside the protected QuantLab holdout "
                                f"({inside[0]} .. {inside[-1]}); replay refuses holdout data")

    # ------------------------------------------------------------------ point-in-time daily facts
    def prior_sessions(self, symbol: str, day: date, n: int = 20) -> list[date]:
        days = [d for d in self.sessions if d < day]
        have = set(self._daily.loc[symbol].index) if symbol in self._daily.index.get_level_values(0) else set()
        return [d for d in days if d in have][-n:] if n > 0 else []

    def adv_shares(self, symbol: str, day: date, n: int = 20) -> float | None:
        """Mean total volume of the last ``n`` COMPLETED sessions strictly before ``day`` (None when unknown)."""
        key = (symbol, day, n)
        if key not in self._adv_cache:
            prior = self.prior_sessions(symbol, day, n)
            vols = [float(self._daily.loc[(symbol, d), "volume"]) for d in prior]
            self._adv_cache[key] = float(np.mean(vols)) if vols else None
        return self._adv_cache[key]

    def prev_close(self, symbol: str, day: date) -> float | None:
        prior = self.prior_sessions(symbol, day, 1)
        return float(self._daily.loc[(symbol, prior[-1]), "close"]) if prior else None

    def window(self, start: date | None, end: date | None, *, warmup_sessions: int = 1) -> list[date]:
        """Replay sessions in [start, end]; by default the first ``warmup_sessions`` sessions are history only."""
        days = self.sessions[warmup_sessions:] if start is None else [d for d in self.sessions if d >= start]
        if end is not None:
            days = [d for d in days if d <= end]
        return days

    # ------------------------------------------------------------------ research views
    def research_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._bars.loc[:, ["t", "symbol", "o", "h", "l", "c", "v", "vw"]]).copy()


# ====================================================================== synthetic data (tests, demos)
@dataclass(frozen=True)
class BarEvent:
    """Override one synthetic bar: ``gap_pct`` moves the OPEN (and the rest of the session follows from there);
    ``ohlc`` replaces the bar outright (later bars continue from its close)."""

    session: int
    slot: int
    symbol: str
    gap_pct: float | None = None
    ohlc: tuple[float, float, float, float] | None = None
    close_pct: float | None = None
    """Return from the bar's open to its close (deterministic move inside the bar)."""


@dataclass(frozen=True)
class SyntheticSpec:
    symbols: tuple[str, ...] = ("SPY",)
    start: date = date(2024, 3, 4)
    sessions: int = 4
    seed: int = 7
    sigma_bps: float = 6.0
    drift_bps: float = 0.0
    range_bps: float = 3.0
    volume: float = 80_000.0
    overnight_sigma_bps: float = 0.0
    events: tuple[BarEvent, ...] = field(default_factory=tuple)


def synthetic_snapshot(spec: SyntheticSpec | None = None, **kw: Any) -> BarSnapshot:
    """Deterministic seeded random-walk 5-minute bars on real XNYS sessions (SYNTHETIC feed).

    Within a session every bar opens at the previous bar's close (no intrabar gap unless an event says so), so a
    marketable entry at the next bar's open is unambiguous. Prices are rounded to 4 decimals.
    """
    spec = spec or SyntheticSpec(**kw)
    rng = np.random.default_rng(spec.seed)
    days = xnys_sessions(spec.start, spec.sessions)
    events = {(e.session, e.slot, e.symbol): e for e in spec.events}
    rows: list[dict[str, Any]] = []
    last: dict[str, float] = {s: DEFAULT_PRICES.get(s, 100.0) for s in spec.symbols}
    for di, day in enumerate(days):
        b = session_open_close(day)
        assert b is not None
        n_slots = int((b[1] - b[0]).total_seconds() // (BAR_MINUTES * 60))
        for sym in spec.symbols:
            px = last[sym] * (1 + rng.normal(0.0, spec.overnight_sigma_bps / 1e4)) if di else last[sym]
            for s in range(n_slots):
                ev = events.get((di, s, sym))
                o = px
                if ev is not None and ev.gap_pct is not None:
                    o = px * (1 + ev.gap_pct)
                r = rng.normal(spec.drift_bps / 1e4, spec.sigma_bps / 1e4)
                if ev is not None and ev.close_pct is not None:
                    r = ev.close_pct
                c = o * (1 + r)
                wig_h, wig_l = abs(rng.normal(0, spec.range_bps / 1e4)), abs(rng.normal(0, spec.range_bps / 1e4))
                h, lo = max(o, c) * (1 + wig_h), min(o, c) * (1 - wig_l)
                if ev is not None and ev.ohlc is not None:
                    o, h, lo, c = ev.ohlc
                o, h, lo, c = (round(x, 4) for x in (o, h, lo, c))
                h, lo = max(h, o, c), min(lo, o, c)
                v = float(round(spec.volume * rng.uniform(0.8, 1.2)))
                rows.append({"t": b[0] + s * BAR, "symbol": sym, "o": o, "h": h, "l": lo, "c": c, "v": v,
                             "vw": round((o + h + lo + c) / 4, 4)})
                px = c
            last[sym] = px
    df = pd.DataFrame(rows)
    return BarSnapshot.from_frame(df, source=f"synthetic(seed={spec.seed})", feed=DataFeed.SYNTHETIC,
                                  note="deterministic seeded random walk; not market data")

