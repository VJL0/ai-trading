"""Point-in-time data validation and live feed-quality monitoring.

``validate_bars`` / ``validate_quotes`` take raw candidate records (``Bar``/``Quote`` models *or* plain dicts with
the same field names, as produced by provider normalizers) and return a :class:`ValidationReport` with the
accepted models, rejected records (with reasons), every quality issue and the list of gaps versus the XNYS
calendar. Gaps are reported, **never filled**. Each check is configurable per :class:`QualityPolicy` as
``REJECT`` (drop from normalized storage; the raw event is still preserved by the store), ``FLAG`` (keep but
record an issue) or ``IGNORE``.

Checks: out-of-order timestamps, duplicates (identical and conflicting), calendar gaps, OHLC inconsistency,
zero/negative prices, negative volume, outlier jumps (percent and robust z-score), bars/quotes outside the
regular session, crossed/locked quotes, stale quotes (received long after the exchange timestamp), clock skew
(provider/exchange timestamp in the future relative to received_ts, or a bar received before it closed) and
look-ahead (``available_at_ts`` earlier than the event time). Records that cannot be turned into a contract
model are always rejected as ``INVALID_RECORD``.

``FeedQualityMonitor`` tracks the last bar/quote per symbol, staleness, gaps and rejects for the dashboard
(``/status`` ``feed`` object and ``/market/quality``) and is the feed-health input of the entries gate.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import ValidationError

from packages.contracts.enums import DataFeed, FeedMode, MarketSession
from packages.contracts.models import Bar, Quote
from packages.core.clock import Clock
from packages.core.db import Database, utc_iso

from .calendar import CalendarOutOfRange, MarketCalendar, get_calendar
from .common import FEED_MODE_LIMITATIONS, FEED_MODE_TO_FEED, feed_limitations, iso, timeframe_delta
from .store import RawExtras, _raw_record_json
from .trades import Trade

log = logging.getLogger("lat.market_data.quality")


class CheckAction(StrEnum):
    REJECT = "REJECT"
    FLAG = "FLAG"
    IGNORE = "IGNORE"


class IssueKind(StrEnum):
    OUT_OF_ORDER = "OUT_OF_ORDER"
    DUPLICATE = "DUPLICATE"
    DUPLICATE_CONFLICT = "DUPLICATE_CONFLICT"
    GAP = "GAP"
    OHLC_INCONSISTENT = "OHLC_INCONSISTENT"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    OUTLIER_JUMP = "OUTLIER_JUMP"
    OUTSIDE_REGULAR_SESSION = "OUTSIDE_REGULAR_SESSION"
    CLOCK_SKEW = "CLOCK_SKEW"
    LOOK_AHEAD = "LOOK_AHEAD"
    CROSSED_QUOTE = "CROSSED_QUOTE"
    LOCKED_QUOTE = "LOCKED_QUOTE"
    STALE_QUOTE = "STALE_QUOTE"
    INVALID_RECORD = "INVALID_RECORD"


Severity = Literal["ERROR", "WARN", "INFO"]


@dataclass(frozen=True)
class QualityPolicy:
    """Per-check action plus thresholds. Defaults are trading-safe (fail closed)."""

    out_of_order: CheckAction = CheckAction.REJECT
    duplicate: CheckAction = CheckAction.REJECT
    ohlc_inconsistent: CheckAction = CheckAction.REJECT
    non_positive_price: CheckAction = CheckAction.REJECT
    negative_volume: CheckAction = CheckAction.REJECT
    outlier_jump: CheckAction = CheckAction.FLAG
    outside_regular_session: CheckAction = CheckAction.REJECT
    clock_skew: CheckAction = CheckAction.REJECT
    crossed_quote: CheckAction = CheckAction.REJECT
    locked_quote: CheckAction = CheckAction.REJECT
    stale_quote: CheckAction = CheckAction.FLAG
    max_jump_pct: Decimal = Decimal("0.08")
    max_overnight_jump_pct: Decimal = Decimal("0.25")
    jump_zscore: Decimal = Decimal("12")
    zscore_window: int = 60
    zscore_min_obs: int = 30
    zscore_min_abs_return: Decimal = Decimal("0.005")  # z-score flags need a move of at least 50 bps (MAD can be ~0)
    max_clock_skew_s: Decimal = Decimal("2")
    stale_quote_s: Decimal = Decimal("5")
    report_gaps: bool = True

    def action(self, kind: IssueKind) -> CheckAction:
        mapping: dict[IssueKind, CheckAction] = {
            IssueKind.OUT_OF_ORDER: self.out_of_order,
            IssueKind.DUPLICATE: self.duplicate,
            IssueKind.DUPLICATE_CONFLICT: self.duplicate,
            IssueKind.OHLC_INCONSISTENT: self.ohlc_inconsistent,
            IssueKind.NON_POSITIVE_PRICE: self.non_positive_price,
            IssueKind.NEGATIVE_VOLUME: self.negative_volume,
            IssueKind.OUTLIER_JUMP: self.outlier_jump,
            IssueKind.OUTSIDE_REGULAR_SESSION: self.outside_regular_session,
            IssueKind.CLOCK_SKEW: self.clock_skew,
            IssueKind.CROSSED_QUOTE: self.crossed_quote,
            IssueKind.LOCKED_QUOTE: self.locked_quote,
            IssueKind.STALE_QUOTE: self.stale_quote,
        }
        # LOOK_AHEAD and INVALID_RECORD are not configurable: they are always rejected.
        return mapping.get(kind, CheckAction.REJECT)

    def with_actions(self, **actions: CheckAction | str) -> QualityPolicy:
        return replace(self, **{k: CheckAction(v) for k, v in actions.items()})


DEFAULT_POLICY = QualityPolicy()


@dataclass(frozen=True)
class QualityIssue:
    kind: IssueKind
    action: CheckAction
    symbol: str | None
    ts: datetime | None
    detail: str
    index: int | None = None

    @property
    def severity(self) -> Severity:
        if self.action == CheckAction.REJECT:
            return "ERROR"
        return "WARN" if self.kind != IssueKind.GAP else "INFO"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "action": self.action.value, "severity": self.severity,
                "symbol": self.symbol, "record_ts": iso(self.ts), "detail": self.detail, "index": self.index}


@dataclass(frozen=True)
class Gap:
    symbol: str
    timeframe: str
    session_date: date
    start: datetime
    end: datetime
    missing: int

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "timeframe": self.timeframe, "session_date": self.session_date.isoformat(),
                "start": iso(self.start), "end": iso(self.end), "missing": self.missing}


@dataclass(frozen=True)
class RejectedRecord:
    index: int
    record: Mapping[str, Any]
    reasons: tuple[str, ...]


@dataclass
class ValidationReport:
    kind: Literal["bars", "quotes", "trades"]
    accepted: list[Any]
    accepted_index: list[int]
    rejected: list[RejectedRecord]
    issues: list[QualityIssue]
    gaps: list[Gap]
    validated_at: datetime
    input_count: int

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.gaps

    def flags_by_accepted_position(self) -> dict[int, list[str]]:
        pos = {src: i for i, src in enumerate(self.accepted_index)}
        out: dict[int, list[str]] = defaultdict(list)
        for issue in self.issues:
            if issue.action == CheckAction.FLAG and issue.index is not None and issue.index in pos:
                out[pos[issue.index]].append(issue.kind.value)
        return dict(out)

    def raw_extras(self, raw_json_by_input_index: Mapping[int, str] | None = None) -> RawExtras:
        raw = raw_json_by_input_index or {}
        rejected: list[tuple[Mapping[str, Any], Sequence[str]]] = []
        for r in self.rejected:
            rec = dict(r.record)
            if r.index in raw:
                rec["raw_json"] = raw[r.index]
            rejected.append((rec, r.reasons))
        return RawExtras(
            rejected=rejected,
            flags=self.flags_by_accepted_position(),
            raw_json={i: raw[src] for i, src in enumerate(self.accepted_index) if src in raw},
        )

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            out[f"{issue.action.value.lower()}:{issue.kind.value}"] += 1
        return dict(sorted(out.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "input": self.input_count,
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "flagged": sum(1 for i in self.issues if i.action == CheckAction.FLAG),
            "gaps": len(self.gaps),
            "missing_bars": sum(g.missing for g in self.gaps),
            "issue_counts": self.counts(),
            "validated_at": iso(self.validated_at),
        }

    def rejected_by_symbol(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.rejected:
            out[str(r.record.get("symbol"))] += 1
        return dict(out)

    def gaps_by_symbol(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for g in self.gaps:
            out[g.symbol] += 1
        return dict(out)


# ------------------------------------------------------------------------------------------ helpers
def _as_record(item: Any) -> dict[str, Any]:
    if isinstance(item, Bar | Quote | Trade):
        return item.model_dump()
    if isinstance(item, Mapping):
        return dict(item)  # pyright: ignore[reportUnknownArgumentType]
    raise TypeError(f"cannot validate {type(item).__name__}")


def _dec(rec: Mapping[str, Any], key: str) -> Decimal | None:
    v = rec.get(key)
    if v is None:
        return None
    if isinstance(v, bool | float):
        raise TypeError(f"{key}: {type(v).__name__} rejected (Decimal/str/int only)")
    if isinstance(v, Decimal):
        if not v.is_finite():
            raise ValueError(f"{key}: non-finite")
        return v
    if isinstance(v, int | str):
        d = Decimal(str(v))
        if not d.is_finite():
            raise ValueError(f"{key}: non-finite")
        return d
    raise TypeError(f"{key}: unsupported type {type(v).__name__}")


def _ts(rec: Mapping[str, Any], key: str) -> datetime | None:
    v = rec.get(key)
    if v is None:
        return None
    if isinstance(v, str):
        v = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if not isinstance(v, datetime):
        raise TypeError(f"{key}: not a datetime")
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError(f"{key}: naive datetime rejected")
    return v.astimezone(UTC)


def _secs(d: Decimal) -> timedelta:
    return timedelta(seconds=float(d))


class _Collector:
    def __init__(self, policy: QualityPolicy) -> None:
        self.policy = policy
        self.issues: list[QualityIssue] = []

    def hit(self, kind: IssueKind, symbol: str | None, ts: datetime | None, detail: str, index: int,
            rejects: list[str]) -> None:
        action = self.policy.action(kind)
        if action == CheckAction.IGNORE:
            return
        self.issues.append(QualityIssue(kind, action, symbol, ts, detail, index))
        if action == CheckAction.REJECT:
            rejects.append(kind.value)


def _robust_z(value: float, history: Sequence[float]) -> float | None:
    if len(history) < 3:
        return None
    med = statistics.median(history)
    mad = statistics.median(abs(h - med) for h in history)
    if mad <= 0:
        return None
    return abs(value - med) / (1.4826 * mad)


def _group_runs(missing: list[datetime], step: timedelta) -> list[tuple[datetime, datetime, int]]:
    runs: list[tuple[datetime, datetime, int]] = []
    for ts in missing:
        if runs and ts == runs[-1][1]:
            s, _e, n = runs[-1]
            runs[-1] = (s, ts + step, n + 1)
        else:
            runs.append((ts, ts + step, 1))
    return runs


# ------------------------------------------------------------------------------------------ bars
def validate_bars(
    records: Iterable[Bar | Mapping[str, Any]],
    calendar: MarketCalendar | None = None,
    policy: QualityPolicy = DEFAULT_POLICY,
    *,
    expected_range: tuple[datetime, datetime] | None = None,
    expected_symbols: Sequence[str] | None = None,
    corporate_action_dates: Mapping[str, set[date]] | None = None,
    validated_at: datetime | None = None,
) -> ValidationReport:
    """Validate bars in *arrival order* (out-of-order = earlier than the last accepted bar of the same key).

    Gaps: for intraday timeframes, every regular-session bar start of the XNYS calendar inside
    ``expected_range`` (default: the full sessions spanned by each symbol's accepted bars) that has no accepted
    bar is reported; daily timeframes report missing sessions. ``corporate_action_dates`` suppresses outlier
    flags on known split/dividend ex-dates.
    """
    cal = calendar or get_calendar()
    col = _Collector(policy)
    items = list(records)
    accepted: list[tuple[int, Bar]] = []
    rejected: list[RejectedRecord] = []
    skew = _secs(policy.max_clock_skew_s)
    last_start: dict[tuple[str, str], datetime] = {}
    seen: dict[tuple[str, str, datetime], dict[str, Any]] = {}
    last_close: dict[tuple[str, str], tuple[Decimal, date]] = {}
    returns: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=policy.zscore_window))
    ca_dates = corporate_action_dates or {}

    for idx, item in enumerate(items):
        rejects: list[str] = []
        try:
            rec = _as_record(item)
        except TypeError as exc:
            col.hit(IssueKind.INVALID_RECORD, None, None, str(exc), idx, rejects)
            rejected.append(RejectedRecord(idx, {"repr": repr(item)}, tuple(rejects)))
            continue
        symbol = str(rec.get("symbol") or "")
        try:
            tf = str(rec.get("timeframe") or "")
            step = timeframe_delta(tf)
            start = _ts(rec, "start_ts")
            end = _ts(rec, "end_ts")
            received = _ts(rec, "received_ts")
            available = _ts(rec, "available_at_ts")
            provider = _ts(rec, "provider_ts")
            o, h, low, c = (_dec(rec, k) for k in ("open", "high", "low", "close"))
            vol = _dec(rec, "volume")
            if start is None or end is None or received is None or available is None:
                raise ValueError("start_ts/end_ts/received_ts/available_at_ts are required")
            if o is None or h is None or low is None or c is None or vol is None:
                raise ValueError("open/high/low/close/volume are required")
        except (TypeError, ValueError, ArithmeticError) as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol or None, None, str(exc), idx, rejects)
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        key = (symbol, tf)
        daily = step >= timedelta(days=1)

        if min(o, h, low, c) <= 0:
            col.hit(IssueKind.NON_POSITIVE_PRICE, symbol, start, f"o={o} h={h} l={low} c={c}", idx, rejects)
        if vol < 0:
            col.hit(IssueKind.NEGATIVE_VOLUME, symbol, start, f"volume={vol}", idx, rejects)
        if h < max(o, c) or low > min(o, c) or h < low:
            col.hit(IssueKind.OHLC_INCONSISTENT, symbol, start, f"o={o} h={h} l={low} c={c}", idx, rejects)
        if end <= start:
            col.hit(IssueKind.INVALID_RECORD, symbol, start, "end_ts <= start_ts", idx, rejects)
        if available < end:
            col.hit(IssueKind.LOOK_AHEAD, symbol, start, f"available_at_ts {iso(available)} < end_ts {iso(end)}",
                    idx, rejects)
        if provider is not None and provider > received + skew:
            col.hit(IssueKind.CLOCK_SKEW, symbol, start,
                    f"provider_ts {iso(provider)} is after received_ts {iso(received)}", idx, rejects)
        if end > received + skew:
            col.hit(IssueKind.CLOCK_SKEW, symbol, start,
                    f"bar received {iso(received)} before it closed {iso(end)}", idx, rejects)
        try:
            session = cal.classify_bar(start, end)
            if session != MarketSession.REGULAR:
                col.hit(IssueKind.OUTSIDE_REGULAR_SESSION, symbol, start, f"session={session.value}", idx, rejects)
        except CalendarOutOfRange as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol, start, str(exc), idx, rejects)
            session = None
        dup_key = (symbol, tf, start)
        if dup_key in seen:
            prior = seen[dup_key]
            same = all(prior.get(k) == rec.get(k) for k in ("open", "high", "low", "close", "volume"))
            col.hit(IssueKind.DUPLICATE if same else IssueKind.DUPLICATE_CONFLICT, symbol, start,
                    "identical duplicate" if same else "same bar key with different values", idx, rejects)
        elif key in last_start and start < last_start[key]:
            col.hit(IssueKind.OUT_OF_ORDER, symbol, start, f"arrived after {iso(last_start[key])}", idx, rejects)

        if not rejects and key in last_close and c > 0:
            prev_c, prev_day = last_close[key]
            day = cal.ny_date(start)
            if day not in ca_dates.get(symbol, set()):
                new_session = daily or day != prev_day
                limit = policy.max_overnight_jump_pct if new_session else policy.max_jump_pct
                jump = max(abs(c / prev_c - 1), abs(o / prev_c - 1))
                if jump > limit:
                    col.hit(IssueKind.OUTLIER_JUMP, symbol, start, f"jump {jump:.4%} > {limit:.2%} vs prev close {prev_c}",
                            idx, rejects)
                else:
                    lr = math.log(float(c / prev_c))
                    hist = returns[key]
                    if len(hist) >= policy.zscore_min_obs:
                        z = _robust_z(lr, list(hist))
                        if (z is not None and z > float(policy.jump_zscore)
                                and abs(lr) >= float(policy.zscore_min_abs_return)):
                            col.hit(IssueKind.OUTLIER_JUMP, symbol, start, f"robust z={z:.1f} > {policy.jump_zscore}",
                                    idx, rejects)
        if rejects:
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        try:
            model = item if isinstance(item, Bar) else Bar.model_validate(
                {**rec, "session": rec.get("session") or (session.value if session else None)})
        except ValidationError as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol, start, exc.errors()[0].get("msg", str(exc)), idx, rejects)
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        accepted.append((idx, model))
        seen[dup_key] = rec
        last_start[key] = max(start, last_start.get(key, start))
        if key in last_close:
            prev_c, _ = last_close[key]
            if prev_c > 0:
                returns[key].append(math.log(float(c / prev_c)))
        last_close[key] = (c, cal.ny_date(start))

    accepted.sort(key=lambda t: (t[1].symbol, t[1].timeframe, t[1].start_ts))
    gaps = _bar_gaps([b for _, b in accepted], cal, expected_range, expected_symbols) if policy.report_gaps else []
    for g in gaps:
        col.issues.append(QualityIssue(IssueKind.GAP, CheckAction.FLAG, g.symbol, g.start,
                                       f"{g.missing} missing {g.timeframe} bar(s) {iso(g.start)}..{iso(g.end)}", None))
    return ValidationReport(
        kind="bars",
        accepted=[b for _, b in accepted],
        accepted_index=[i for i, _ in accepted],
        rejected=rejected,
        issues=col.issues,
        gaps=gaps,
        validated_at=validated_at or datetime.now(UTC),
        input_count=len(items),
    )


def _bar_gaps(bars: Sequence[Bar], cal: MarketCalendar, expected_range: tuple[datetime, datetime] | None,
              expected_symbols: Sequence[str] | None) -> list[Gap]:
    by_key: dict[tuple[str, str], set[datetime]] = defaultdict(set)
    for b in bars:
        by_key[(b.symbol, b.timeframe)].add(b.start_ts)
    if expected_symbols and expected_range:
        tfs = {tf for _s, tf in by_key} or {"1Min"}
        for s in expected_symbols:
            for tf in tfs:
                by_key.setdefault((s, tf), set())
    gaps: list[Gap] = []
    for (symbol, tf), present in sorted(by_key.items()):
        step = timeframe_delta(tf)
        if expected_range is not None:
            lo, hi = expected_range
            d0, d1 = cal.ny_date(lo), cal.ny_date(hi)
        elif present:
            lo = hi = None
            d0, d1 = cal.ny_date(min(present)), cal.ny_date(max(present))
        else:
            continue
        try:
            sessions = cal.sessions_in_range(d0, d1)
        except CalendarOutOfRange:
            continue
        if step >= timedelta(days=1):
            present_days = {cal.ny_date(t) for t in present}
            missing_days = [d for d in sessions if d not in present_days
                            and (lo is None or cal.session_open(d) >= lo) and (hi is None or cal.session_close(d) <= hi)]
            for d in missing_days:
                gaps.append(Gap(symbol, tf, d, cal.session_open(d), cal.session_close(d), 1))
            continue
        for d in sessions:
            starts = cal.regular_bar_starts(d, step)
            missing = [t for t in starts if t not in present and (lo is None or t >= lo) and (hi is None or t + step <= hi)]
            for s, e, n in _group_runs(missing, step):
                gaps.append(Gap(symbol, tf, d, s, e, n))
    return gaps


def compute_bar_gaps(bars: Sequence[Bar], calendar: MarketCalendar | None = None, *,
                     expected_range: tuple[datetime, datetime] | None = None,
                     expected_symbols: Sequence[str] | None = None) -> list[Gap]:
    """Missing regular-session bars vs the XNYS calendar (reported, never filled). See ``validate_bars``."""
    return _bar_gaps(bars, calendar or get_calendar(), expected_range, expected_symbols)


def gaps_within_observed_sessions(bars: Sequence[Bar], calendar: MarketCalendar | None = None, *,
                                  intraday_days: Mapping[str, Sequence[date]] | None = None) -> list[Gap]:
    """Gaps for a sampled/bundled dataset: intraday gaps inside each session that has data (or each session listed in
    ``intraday_days`` per symbol, e.g. one per source file) plus daily gaps over each symbol's daily span. Sessions
    entirely absent from an intraday sample are not part of the sample and are not counted as intraday gaps."""
    cal = calendar or get_calendar()
    by_day: dict[tuple[str, str, date], list[Bar]] = defaultdict(list)
    daily: list[Bar] = []
    days: dict[tuple[str, str], set[date]] = defaultdict(set)
    for b in bars:
        if timeframe_delta(b.timeframe) >= timedelta(days=1):
            daily.append(b)
            continue
        d = cal.ny_date(b.start_ts)
        by_day[(b.symbol, b.timeframe, d)].append(b)
        days[(b.symbol, b.timeframe)].add(d)
    if intraday_days:
        for sym, ds in intraday_days.items():
            tfs = {tf for (s, tf) in days if s == sym} or {"1Min"}
            for tf in tfs:
                days[(sym, tf)].update(ds)
    out: list[Gap] = []
    for (sym, tf), ds in sorted(days.items()):
        for d in sorted(ds):
            if not cal.covers(d) or not cal.is_session(d):
                continue
            out += _bar_gaps(by_day.get((sym, tf, d), []), cal, (cal.session_open(d), cal.session_close(d)), [sym]) \
                if by_day.get((sym, tf, d)) else [Gap(sym, tf, d, cal.session_open(d), cal.session_close(d),
                                                      len(cal.regular_bar_starts(d, timeframe_delta(tf))))]
    out += _bar_gaps(daily, cal, None, None)
    return out


def only_available(report: ValidationReport, asof: datetime) -> ValidationReport:
    """Copy of ``report`` without accepted records that are not yet knowable at ``asof`` (still-forming bars)."""
    keep = [(i, rec) for i, rec in zip(report.accepted_index, report.accepted, strict=True)
            if getattr(rec, "available_at_ts", asof) <= asof]
    return replace(report, accepted=[r for _, r in keep], accepted_index=[i for i, _ in keep])


def with_gaps(report: ValidationReport, gaps: Sequence[Gap]) -> ValidationReport:
    """Copy of ``report`` whose gap list (and GAP issues) is replaced by ``gaps`` (e.g. per-file-day gaps)."""
    issues = [i for i in report.issues if i.kind != IssueKind.GAP]
    for g in gaps:
        issues.append(QualityIssue(IssueKind.GAP, CheckAction.FLAG, g.symbol, g.start,
                                   f"{g.missing} missing {g.timeframe} bar(s) {iso(g.start)}..{iso(g.end)}", None))
    return replace(report, gaps=list(gaps), issues=issues)


# ------------------------------------------------------------------------------------------ quotes
def validate_quotes(
    records: Iterable[Quote | Mapping[str, Any]],
    calendar: MarketCalendar | None = None,
    policy: QualityPolicy = DEFAULT_POLICY,
    *,
    validated_at: datetime | None = None,
) -> ValidationReport:
    """Validate quotes in arrival order (see module docstring for the checks)."""
    cal = calendar or get_calendar()
    col = _Collector(policy)
    items = list(records)
    accepted: list[tuple[int, Quote]] = []
    rejected: list[RejectedRecord] = []
    skew = _secs(policy.max_clock_skew_s)
    stale = _secs(policy.stale_quote_s)
    last_ts: dict[str, datetime] = {}
    seen: set[tuple[Any, ...]] = set()
    for idx, item in enumerate(items):
        rejects: list[str] = []
        try:
            rec = _as_record(item)
        except TypeError as exc:
            col.hit(IssueKind.INVALID_RECORD, None, None, str(exc), idx, rejects)
            rejected.append(RejectedRecord(idx, {"repr": repr(item)}, tuple(rejects)))
            continue
        symbol = str(rec.get("symbol") or "")
        try:
            ex = _ts(rec, "exchange_ts")
            received = _ts(rec, "received_ts")
            available = _ts(rec, "available_at_ts")
            provider = _ts(rec, "provider_ts")
            bid, ask = _dec(rec, "bid"), _dec(rec, "ask")
            bsz, asz = _dec(rec, "bid_size"), _dec(rec, "ask_size")
            if ex is None or received is None or available is None:
                raise ValueError("exchange_ts/received_ts/available_at_ts are required")
            if bid is None or ask is None or bsz is None or asz is None:
                raise ValueError("bid/ask/bid_size/ask_size are required")
        except (TypeError, ValueError, ArithmeticError) as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol or None, None, str(exc), idx, rejects)
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        if bid <= 0 or ask <= 0:
            col.hit(IssueKind.NON_POSITIVE_PRICE, symbol, ex, f"bid={bid} ask={ask}", idx, rejects)
        elif ask < bid:
            col.hit(IssueKind.CROSSED_QUOTE, symbol, ex, f"ask {ask} < bid {bid}", idx, rejects)
        elif ask == bid:
            col.hit(IssueKind.LOCKED_QUOTE, symbol, ex, f"ask == bid == {bid}", idx, rejects)
        if bsz < 0 or asz < 0:
            col.hit(IssueKind.INVALID_RECORD, symbol, ex, "negative size", idx, rejects)
        if ex > received + skew or (provider is not None and provider > received + skew):
            col.hit(IssueKind.CLOCK_SKEW, symbol, ex, f"exchange/provider ts after received_ts {iso(received)}", idx, rejects)
        if available < ex:
            col.hit(IssueKind.LOOK_AHEAD, symbol, ex, "available_at_ts precedes exchange_ts", idx, rejects)
        if received - ex > stale:
            col.hit(IssueKind.STALE_QUOTE, symbol, ex, f"received {(received - ex).total_seconds():.1f}s after exchange_ts",
                    idx, rejects)
        try:
            if cal.classify(ex) != MarketSession.REGULAR:
                col.hit(IssueKind.OUTSIDE_REGULAR_SESSION, symbol, ex, f"session={cal.classify(ex).value}", idx, rejects)
        except CalendarOutOfRange as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol, ex, str(exc), idx, rejects)
        ident = (symbol, ex, bid, ask, bsz, asz)
        if ident in seen:
            col.hit(IssueKind.DUPLICATE, symbol, ex, "identical duplicate quote", idx, rejects)
        elif symbol in last_ts and ex < last_ts[symbol]:
            col.hit(IssueKind.OUT_OF_ORDER, symbol, ex, f"arrived after {iso(last_ts[symbol])}", idx, rejects)
        if rejects:
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        try:
            model = item if isinstance(item, Quote) else Quote.model_validate(rec)
        except ValidationError as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol, ex, exc.errors()[0].get("msg", str(exc)), idx, rejects)
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        accepted.append((idx, model))
        seen.add(ident)
        last_ts[symbol] = max(ex, last_ts.get(symbol, ex))
    return ValidationReport(
        kind="quotes",
        accepted=[q for _, q in accepted],
        accepted_index=[i for i, _ in accepted],
        rejected=rejected,
        issues=col.issues,
        gaps=[],
        validated_at=validated_at or datetime.now(UTC),
        input_count=len(items),
    )


# ------------------------------------------------------------------------------------------ trades
def validate_trades(
    records: Iterable[Trade | Mapping[str, Any]],
    calendar: MarketCalendar | None = None,
    policy: QualityPolicy = DEFAULT_POLICY,
    *,
    validated_at: datetime | None = None,
) -> ValidationReport:
    """Validate trade prints in arrival order: non-positive price, negative size, clock skew, look-ahead, outside the
    regular session, duplicates (same id/time/price/size) and out-of-order arrival."""
    cal = calendar or get_calendar()
    col = _Collector(policy)
    items = list(records)
    accepted: list[tuple[int, Trade]] = []
    rejected: list[RejectedRecord] = []
    skew = _secs(policy.max_clock_skew_s)
    last_ts: dict[str, datetime] = {}
    seen: set[tuple[Any, ...]] = set()
    for idx, item in enumerate(items):
        rejects: list[str] = []
        try:
            rec = _as_record(item)
        except TypeError as exc:
            col.hit(IssueKind.INVALID_RECORD, None, None, str(exc), idx, rejects)
            rejected.append(RejectedRecord(idx, {"repr": repr(item)}, tuple(rejects)))
            continue
        symbol = str(rec.get("symbol") or "")
        try:
            ex = _ts(rec, "exchange_ts")
            received = _ts(rec, "received_ts")
            available = _ts(rec, "available_at_ts")
            provider = _ts(rec, "provider_ts")
            price, size = _dec(rec, "price"), _dec(rec, "size")
            if ex is None or received is None or available is None or price is None or size is None:
                raise ValueError("exchange_ts/received_ts/available_at_ts/price/size are required")
        except (TypeError, ValueError, ArithmeticError) as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol or None, None, str(exc), idx, rejects)
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        if price <= 0:
            col.hit(IssueKind.NON_POSITIVE_PRICE, symbol, ex, f"price={price}", idx, rejects)
        if size < 0:
            col.hit(IssueKind.INVALID_RECORD, symbol, ex, "negative size", idx, rejects)
        if ex > received + skew or (provider is not None and provider > received + skew):
            col.hit(IssueKind.CLOCK_SKEW, symbol, ex, f"exchange/provider ts after received_ts {iso(received)}", idx, rejects)
        if available < ex:
            col.hit(IssueKind.LOOK_AHEAD, symbol, ex, "available_at_ts precedes exchange_ts", idx, rejects)
        try:
            session = cal.classify(ex)
            if session != MarketSession.REGULAR:
                col.hit(IssueKind.OUTSIDE_REGULAR_SESSION, symbol, ex, f"session={session.value}", idx, rejects)
        except CalendarOutOfRange as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol, ex, str(exc), idx, rejects)
        ident = (symbol, rec.get("trade_id"), ex, price, size)
        if ident in seen:
            col.hit(IssueKind.DUPLICATE, symbol, ex, "duplicate trade print", idx, rejects)
        elif symbol in last_ts and ex < last_ts[symbol]:
            col.hit(IssueKind.OUT_OF_ORDER, symbol, ex, f"arrived after {iso(last_ts[symbol])}", idx, rejects)
        if rejects:
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        try:
            model = item if isinstance(item, Trade) else Trade.model_validate(rec)
        except ValidationError as exc:
            col.hit(IssueKind.INVALID_RECORD, symbol, ex, exc.errors()[0].get("msg", str(exc)), idx, rejects)
            rejected.append(RejectedRecord(idx, rec, tuple(rejects)))
            continue
        accepted.append((idx, model))
        seen.add(ident)
        last_ts[symbol] = max(ex, last_ts.get(symbol, ex))
    return ValidationReport(kind="trades", accepted=[t for _, t in accepted], accepted_index=[i for i, _ in accepted],
                            rejected=rejected, issues=col.issues, gaps=[], validated_at=validated_at or datetime.now(UTC),
                            input_count=len(items))


# ------------------------------------------------------------------------------------------ persistence
def record_quality_events(
    db: Database,
    report: ValidationReport,
    feed: DataFeed | str,
    *,
    max_rows_per_kind: int = 50,
    ts: datetime | None = None,
    context: Mapping[str, Any] | None = None,
) -> int:
    """Write ``data_quality_events`` rows: up to ``max_rows_per_kind`` per issue kind plus one summary row.

    Returns the number of rows written. ``ts`` is the detection time (default: ``report.validated_at``); the
    record's own timestamp is in ``detail_json.record_ts``.
    """
    feed_s = DataFeed(feed).value if not isinstance(feed, DataFeed) else feed.value
    when = utc_iso(ts or report.validated_at)
    per_kind: dict[str, int] = defaultdict(int)
    rows: list[tuple[str, str | None, str, str, str, str]] = []
    for issue in report.issues:
        per_kind[issue.kind.value] += 1
        if per_kind[issue.kind.value] > max_rows_per_kind:
            continue
        detail = issue.as_dict()
        if context:
            detail["context"] = dict(context)
        rows.append((when, issue.symbol, feed_s, issue.kind.value, issue.severity, json.dumps(detail, sort_keys=True)))
    summary = {**report.summary(), "truncated_kinds": {k: n for k, n in per_kind.items() if n > max_rows_per_kind}}
    if context:
        summary["context"] = dict(context)
    rows.append((when, None, feed_s, "VALIDATION_SUMMARY",
                 "ERROR" if report.rejected else ("WARN" if report.issues else "INFO"), json.dumps(summary, sort_keys=True)))

    def _write(conn: Any) -> int:
        conn.executemany(
            "INSERT INTO data_quality_events(ts, symbol, feed, kind, severity, detail_json) VALUES (?,?,?,?,?,?)", rows)
        return len(rows)

    return db.write(_write)


def rejected_raw_json(report: ValidationReport) -> dict[int, str]:
    """Verbatim JSON of rejected records keyed by input index (for raw preservation when no payload exists)."""
    return {r.index: _raw_record_json(r.record) for r in report.rejected}


# ------------------------------------------------------------------------------------------ live monitor
class FeedConnectionState(StrEnum):
    UNSPECIFIED = "UNSPECIFIED"  # no ingestion service attached (e.g. replay): status from staleness only
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    BACKFILLING = "BACKFILLING"
    LIVE = "LIVE"


@dataclass
class _SymbolState:
    last_bar_ts: datetime | None = None
    last_quote_ts: datetime | None = None
    last_quote_received: datetime | None = None
    last_trade_ts: datetime | None = None
    last_trade_price: Decimal | None = None
    spread_bps: Decimal | None = None
    gaps_today: int = 0
    rejected_today: int = 0


@dataclass
class FeedQualityMonitor:
    """Thread-safe live feed-quality state. ``status()`` matches docs/architecture/api-contract.md.

    Fail closed: a symbol never seen, a quote older than ``max_staleness_s`` (by exchange or receive time)
    or a non-LIVE connection state makes the overall status non-OK.
    """

    feed_mode: FeedMode
    provider: str
    clock: Clock
    symbols: Sequence[str] = ()
    max_staleness_s: Decimal = Decimal("15")
    max_clock_skew_s: Decimal = Decimal("2")
    _state: dict[str, _SymbolState] = field(default_factory=dict[str, _SymbolState])
    _conn: FeedConnectionState = FeedConnectionState.UNSPECIFIED
    _conn_detail: str = ""
    _day: date | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        for s in self.symbols:
            self._state.setdefault(s, _SymbolState())

    # ------------------------------------------------------------ observations
    def _roll_day(self) -> None:
        today = MarketCalendar.ny_date(self.clock.now())
        if self._day != today:
            for st in self._state.values():
                st.gaps_today = 0
                st.rejected_today = 0
            self._day = today

    def _sym(self, symbol: str) -> _SymbolState:
        return self._state.setdefault(symbol, _SymbolState())

    def track(self, symbol: str) -> None:
        """Track a symbol before any data arrives (it keeps the status non-OK until a fresh quote is seen)."""
        with self._lock:
            self._sym(symbol)

    def observe_quote(self, quote: Quote) -> None:
        with self._lock:
            self._roll_day()
            st = self._sym(quote.symbol)
            if st.last_quote_ts is None or quote.exchange_ts >= st.last_quote_ts:
                st.last_quote_ts = quote.exchange_ts
                st.last_quote_received = quote.received_ts
                try:
                    st.spread_bps = quote.spread_bps
                except ArithmeticError:
                    st.spread_bps = None

    def observe_trade(self, trade: Trade) -> None:
        with self._lock:
            self._roll_day()
            st = self._sym(trade.symbol)
            if st.last_trade_ts is None or trade.exchange_ts >= st.last_trade_ts:
                st.last_trade_ts = trade.exchange_ts
                st.last_trade_price = trade.price

    def observe_bar(self, bar: Bar) -> None:
        with self._lock:
            self._roll_day()
            st = self._sym(bar.symbol)
            if st.last_bar_ts is None or bar.end_ts > st.last_bar_ts:
                st.last_bar_ts = bar.end_ts

    def record_rejects(self, symbol: str, count: int = 1) -> None:
        with self._lock:
            self._roll_day()
            self._sym(symbol).rejected_today += count

    def record_gaps(self, symbol: str, count: int = 1) -> None:
        with self._lock:
            self._roll_day()
            self._sym(symbol).gaps_today += count

    def observe_report(self, report: ValidationReport) -> None:
        for item in report.accepted:
            if isinstance(item, Quote):
                self.observe_quote(item)
            elif isinstance(item, Bar):
                self.observe_bar(item)
            elif isinstance(item, Trade):
                self.observe_trade(item)
        for s, n in report.rejected_by_symbol().items():
            self.record_rejects(s, n)
        for s, n in report.gaps_by_symbol().items():
            self.record_gaps(s, n)

    def set_connection_state(self, state: FeedConnectionState | str, detail: str = "") -> None:
        with self._lock:
            self._conn = FeedConnectionState(state)
            self._conn_detail = detail

    @property
    def connection_state(self) -> FeedConnectionState:
        return self._conn

    # ------------------------------------------------------------ status
    def _ages(self, st: _SymbolState, now: datetime) -> tuple[Decimal, Decimal] | None:
        """(staleness = oldest of exchange/receive age, clamped >= 0; minimum raw age, negative = from the future)."""
        if st.last_quote_ts is None:
            return None
        ages = [now - st.last_quote_ts]
        if st.last_quote_received is not None:
            ages.append(now - st.last_quote_received)
        secs = [a.total_seconds() for a in ages]
        return max(Decimal(0), Decimal(str(round(max(secs), 3)))), Decimal(str(round(min(secs), 3)))

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock.now()
            self._roll_day()
            rows: list[dict[str, Any]] = []
            worst: Decimal | None = None
            missing: list[str] = []
            skewed: list[str] = []
            for symbol in sorted(self._state):
                st = self._state[symbol]
                ages = self._ages(st, now)
                age = ages[0] if ages is not None else None
                if ages is None:
                    missing.append(symbol)
                else:
                    if ages[1] < -self.max_clock_skew_s:
                        skewed.append(symbol)
                    if worst is None or ages[0] > worst:
                        worst = ages[0]
                rows.append({
                    "symbol": symbol,
                    "last_bar_ts": iso(st.last_bar_ts),
                    "last_quote_ts": iso(st.last_quote_ts),
                    "staleness_s": format(age, "f") if age is not None else None,
                    "spread_bps": format(st.spread_bps.quantize(Decimal("0.01")), "f") if st.spread_bps is not None else None,
                    "gaps_today": st.gaps_today,
                    "rejected_today": st.rejected_today,
                })
            last_bar = max((s.last_bar_ts for s in self._state.values() if s.last_bar_ts), default=None)
            last_quote = max((s.last_quote_ts for s in self._state.values() if s.last_quote_ts), default=None)
            last_trade = max((s.last_trade_ts for s in self._state.values() if s.last_trade_ts), default=None)
            conn = self._conn
            if conn in (FeedConnectionState.DISCONNECTED, FeedConnectionState.CONNECTING):
                status, detail = "DISCONNECTED", self._conn_detail or f"feed {conn.value.lower()}"
            elif conn == FeedConnectionState.BACKFILLING:
                status, detail = "BACKFILLING", self._conn_detail or "backfilling gaps before resuming entries"
            elif not self._state or missing:
                status = "STALE"
                detail = f"no quote observed yet for {', '.join(missing) or 'any symbol'} (fail closed)"
            elif skewed:
                status = "STALE"
                detail = f"quote timestamps ahead of the local clock for {', '.join(skewed)} (clock skew; fail closed)"
            elif worst is not None and worst > self.max_staleness_s:
                status, detail = "STALE", f"worst quote age {worst}s > {self.max_staleness_s}s"
            else:
                status, detail = "OK", "fresh quotes for all tracked symbols"
            feed = FEED_MODE_TO_FEED.get(self.feed_mode)
            return {
                "feed_mode": self.feed_mode.value,
                "provider": self.provider,
                "feed": feed.value if feed else None,
                "is_simulated": self.feed_mode == FeedMode.SIMULATED,
                "last_bar_ts": iso(last_bar),
                "last_quote_ts": iso(last_quote),
                "last_trade_ts": iso(last_trade),
                "staleness_s": format(worst, "f") if worst is not None else None,
                "status": status,
                "detail": detail,
                "connection_state": conn.value,
                "max_staleness_s": format(self.max_staleness_s, "f"),
                "limitations": list(feed_limitations(self.feed_mode, feed) or FEED_MODE_LIMITATIONS.get(self.feed_mode, ())),
                "symbols": rows,
                "checked_at": iso(now),
            }

    def is_ok(self) -> bool:
        return self.status()["status"] == "OK"
