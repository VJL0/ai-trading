"""Derive filing features at each New York session open and align them to an equity panel."""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from services.market_data.calendar import NY, get_calendar

from .panel import EquityPanel

if TYPE_CHECKING:
    from services.market_data.edgar.parsers import Transaction
    from services.market_data.edgar.store import Disclosure, EdgarStore

EARNINGS_FIELDS = ("sue", "earnings_announcement", "earnings_window", "earnings_drift_age")
FUNDAMENTAL_FIELDS = ("accruals", "asset_growth", "gross_profitability", "net_share_issuance")
INSIDER_FIELDS = ("insider_net_buy_ratio", "insider_cluster_buy")
EVENT_FIELDS = EARNINGS_FIELDS + FUNDAMENTAL_FIELDS + INSIDER_FIELDS

_WINDOW = 5
_INSIDER_DAYS = 30
_SEASON_TOLERANCE = 20  # fiscal year boundaries can move under a 52/53-week calendar
_FLOW = dict[tuple[date, date], float]


def _fiscal_day(values: Iterable[date], reference: date, days: int) -> date | None:
    target = reference - timedelta(days=days)
    closest = min(values, key=lambda day: (abs((day - target).days), day), default=None)
    return closest if closest is not None and abs((closest - target).days) <= _SEASON_TOLERANCE else None


def _quarters(flows: _FLOW, *, eps: bool = False) -> dict[date, float]:
    """Prefer a directly reported quarter. Difference fiscal YTD flows where necessary."""
    quarters = {end: value for (start, end), value in sorted(flows.items())
                if 70 <= (end - start).days <= 110}
    ends_by_start: dict[date, list[date]] = defaultdict(list)
    for start, end in flows:
        ends_by_start[start].append(end)
    for (start, end), value in sorted(flows.items(), key=lambda item: item[0][1]):
        if end in quarters or not 155 <= (end - start).days <= 385:
            continue
        if eps:
            # Annual EPS minus the first three reported quarters approximates Q4 when no direct EPS exists.
            elapsed = sorted(t for t in quarters if start <= t < end)
            elapsed = [start, *elapsed, end]
            if len(elapsed) == 5 and all(70 <= (b - a).days <= 110 for a, b in pairwise(elapsed)):
                elapsed = elapsed[1:-1]
                quarters[end] = value - sum(quarters[t] for t in elapsed)
        else:
            previous = _fiscal_day((t for t in ends_by_start[start] if t < end), end, 91)
            if previous is not None:
                quarters[end] = value - flows[start, previous]
    return quarters


def _ttm(flows: _FLOW, end: date) -> float:
    annual = [value for (start, stop), value in flows.items() if stop == end and 345 <= (stop - start).days <= 385]
    if len(annual) == 1:
        return annual[0]
    quarterly = _quarters(flows)
    total = 0.0
    for i in range(4):
        if end not in quarterly:
            return np.nan
        total += quarterly[end]
        if i == 3:
            break
        prior = _fiscal_day((day for day in quarterly if day < end), end, 91)
        if prior is None:
            return np.nan
        end = prior
    return total


def _sue(flows: _FLOW) -> float:
    eps = _quarters(flows, eps=True)
    dates = sorted(eps)
    surprises: list[float] = []
    for i, day in enumerate(dates):
        insertion = bisect_left(dates, day - timedelta(days=365), hi=i)
        prior = _fiscal_day(dates[max(0, insertion - 1):min(i, insertion + 1)], day, 365)
        surprises.append(eps[day] - eps[prior] if prior is not None else np.nan)
    if not surprises or not np.isfinite(surprises[-1]):
        return np.nan
    history = [value for value in surprises[:-1] if np.isfinite(value)][-8:]
    if len(history) < 4:
        return np.nan
    sd = float(np.std(history, ddof=1))
    return surprises[-1] / sd if sd > 0 else np.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


@dataclass
class _Issuer:
    tickers: tuple[str, ...] = ()
    flows: dict[str, _FLOW] = field(default_factory=lambda: defaultdict(dict))
    balances: dict[str, dict[date, float]] = field(default_factory=lambda: defaultdict(dict))
    financial: tuple[float, ...] = (np.nan,) * 5
    financial_dirty: bool = False
    earnings_day: date | None = None
    insider_covered: bool = False
    originals: dict[tuple[date, tuple[str, ...]], set[str]] = field(default_factory=lambda: defaultdict(set))
    active: dict[str, tuple[Transaction, ...]] = field(default_factory=dict)

    def apply(self, record: Disclosure, day: date) -> None:
        filing = record.filing
        self.tickers = record.effective_tickers(self.tickers)
        if filing.earnings and not filing.form.endswith("/A"):
            self.earnings_day = day
        for fact in record.facts:
            if fact.start is None:
                self.balances[fact.metric][fact.end] = fact.value
            else:
                self.flows[fact.metric][fact.start, fact.end] = fact.value
        if record.facts:
            self.financial_dirty = True
        own = record.ownership
        if own is not None:
            if filing.form.endswith("/A"):
                candidates = self.originals.get((own.original_filed, own.owners), set()) if own.original_filed else set()
                if len(candidates) != 1:
                    # The amendment cannot safely be linked to an original. Avoid counting it as a new trade.
                    return
                group = next(iter(candidates))
            else:
                group = filing.accession
                self.originals[filing.filed, own.owners].add(group)
            self.insider_covered = True
            transactions = tuple(tx for tx in own.transactions if tx.day <= filing.filed)
            self.active[group] = transactions

    def financial_features(self) -> tuple[float, ...]:
        if not self.financial_dirty:
            return self.financial
        flows = self.flows
        assets, shares = self.balances["assets"], self.balances["shares"]
        accruals = growth = gross = issuance = np.nan
        if assets:
            end = max(assets)
            year_ago = _fiscal_day(assets, end, 365)
            if year_ago is not None:
                growth = _ratio(assets[end], assets[year_ago]) - 1
                average_assets = (assets[end] + assets[year_ago]) / 2
                accruals = _ratio(_ttm(flows["net_income"], end) - _ttm(flows["operating_cash_flow"], end),
                                  average_assets)
            gp = _ttm(flows["gross_profit"], end)
            if not np.isfinite(gp):
                gp = _ttm(flows["revenue"], end) - _ttm(flows["cost_of_revenue"], end)
            gross = _ratio(gp, assets[end])
        if shares:
            end = max(shares)
            previous = _fiscal_day(shares, end, 365)
            if previous is not None:
                issuance = _ratio(shares[end], shares[previous]) - 1
        self.financial = (_sue(flows["eps"]), accruals, growth, gross, issuance)
        self.financial_dirty = False
        return self.financial

    def insider(self, day: date) -> tuple[float, float]:
        if not self.insider_covered:
            return np.nan, np.nan
        start = day - timedelta(days=_INSIDER_DAYS - 1)
        self.active = {accession: selected for accession, txs in self.active.items()
                       if (selected := tuple(tx for tx in txs if tx.day >= start))}
        buy = sell = 0.0
        buyers: set[str] = set()
        for txs in self.active.values():
            seen: set[int] = set()
            for tx in txs:
                if tx.code == "P":
                    buyers.add(tx.owner)
                if tx.row not in seen:
                    if tx.code == "P":
                        buy += tx.shares
                    else:
                        sell += tx.shares
                    seen.add(tx.row)
        return _ratio(buy - sell, buy + sell) if buy or sell else 0.0, float(len(buyers) >= 3)


def attach_events(panel: EquityPanel, store: EdgarStore) -> EquityPanel:
    """Return an enriched panel; the caller chooses the date range before fitting or evaluating."""
    from services.market_data.edgar.parsers import normalize_ticker

    fields = {name: np.full(panel.shape, np.nan, dtype=np.float32) for name in EVENT_FIELDS}
    if not len(panel.dates):
        return replace(panel, f={**panel.f, **fields})
    cal = get_calendar()
    symbols = [normalize_ticker(str(sym)) for sym in panel.symbols]
    dates = panel.dates.date
    last = dates[-1]
    disclosures = store.as_of(cal.session_open(last), symbols=symbols)
    candidate = next(disclosures, None)
    if candidate is None:
        return replace(panel, f={**panel.f, **fields})
    first = min(dates[0], candidate.available_at.astimezone(NY).date())
    ordinals = {day: i for i, day in enumerate(cal.sessions_in_range(first, last))}
    issuers: dict[int, _Issuer] = {}
    owners: dict[str, set[int]] = defaultdict(set)
    for row, day in enumerate(dates):
        op = cal.session_open(day)
        while candidate is not None and candidate.available_at <= op:
            cik = candidate.filing.cik
            if cik not in issuers:
                issuers[cik] = _Issuer()
            issuer = issuers[cik]
            old_tickers = issuer.tickers
            issuer.apply(candidate, candidate.available_at.astimezone(NY).date())
            for old in set(old_tickers) - set(issuer.tickers):
                owners[old].discard(cik)
            for new in issuer.tickers:
                owners[new].add(cik)
            candidate = next(disclosures, None)
        insider_row: dict[int, tuple[float, float]] = {}
        for col, symbol in enumerate(symbols):
            matches = owners.get(symbol)
            if not matches or len(matches) != 1:
                continue
            cik = next(iter(matches))
            issuer = issuers[cik]
            if issuer.earnings_day is not None:
                age = ordinals[day] - ordinals[issuer.earnings_day]
                fields["earnings_announcement"][row, col] = float(age == 0)
                fields["earnings_window"][row, col] = float(age < _WINDOW)
                fields["earnings_drift_age"][row, col] = age
            for name, value in zip(("sue", *FUNDAMENTAL_FIELDS), issuer.financial_features(), strict=True):
                fields[name][row, col] = value
            if cik not in insider_row:
                insider_row[cik] = issuer.insider(day)
            for name, value in zip(INSIDER_FIELDS, insider_row[cik], strict=True):
                fields[name][row, col] = value
    return replace(panel, f={**panel.f, **fields}, meta={**panel.meta, "event_source": "SEC EDGAR"})


def missing_fields(panel: EquityPanel, fields: Iterable[str]) -> tuple[str, ...]:
    """Fields without overlapping panel data, suitable for an experiment preflight."""
    present = np.isfinite(panel.f["close"]) & (panel.kind == "COMMON_STOCK")
    return tuple(name for name in fields if name not in panel.f or not (np.isfinite(panel.f[name]) & present).any())


def evidence_fingerprint(panel: EquityPanel, fields: Iterable[str] = EVENT_FIELDS) -> str:
    """Hash only the aligned observations in the caller's development slice."""
    h = hashlib.sha256(panel.content_fingerprint().encode())
    present = np.isfinite(panel.f["close"])
    cols = sorted(np.where(present.any(axis=0))[0].tolist(), key=lambda j: str(panel.symbols[j]))
    for name in sorted(fields):
        h.update(name.encode())
        values = np.where(present[:, cols], panel.f[name][:, cols], np.nan).astype("<f4")
        h.update(values.tobytes())
    return h.hexdigest()[:16]
