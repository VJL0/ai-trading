"""Normalize filing metadata, XBRL facts and ownership disclosures."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cached_property
from html.parser import HTMLParser
from typing import Any

from services.market_data.calendar import NY, get_calendar

PERIODIC = frozenset({"10-K", "10-Q"})
FORMS = PERIODIC | {"8-K", "4"}


def first_session_open(accepted: datetime) -> datetime:
    if accepted.utcoffset() is None:
        raise ValueError("filing acceptance must be timezone-aware")
    cal = get_calendar()
    day = accepted.astimezone(NY).date()
    if cal.is_session(day) and accepted <= cal.session_open(day):
        return cal.session_open(day)
    return cal.session_open(cal.next_session(day))


def acceptance(value: str) -> datetime:
    if re.fullmatch(r"\d{14}", value):
        parsed = datetime.strptime(value, "%Y%m%d%H%M%S")  # noqa: DTZ007 - SEC archive fields use New York time
    elif "T" in value or " " in value:
        parsed = datetime.fromisoformat(value)
    else:
        raise ValueError("SEC acceptance requires a timestamp, including time of day")
    return parsed.replace(tzinfo=NY).astimezone(UTC) if parsed.utcoffset() is None else parsed.astimezone(UTC)


def submissions_acceptance(value: str) -> datetime:
    """Conservative reading of the submissions JSON ``acceptanceDateTime``.

    The field carries a ``Z`` suffix, but its wall-clock value has been reported to be New York time. Reading
    an Eastern 16:30 as UTC would place an after-close filing at 11:30 ET, which is lookahead. Both readings are
    computed and the later one is used, so a mislabelled zone can only delay availability, never advance it.
    The fetch CLI resolves acceptance from the filing detail page instead; this path is a fallback.
    """
    as_stated = acceptance(value)
    wall = datetime.fromisoformat(value) if ("T" in value or " " in value) else None
    if wall is None or wall.utcoffset() is None:
        return as_stated
    as_eastern = wall.replace(tzinfo=None).replace(tzinfo=NY).astimezone(UTC)
    return max(as_stated, as_eastern)


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str
    form: str
    accepted_at: datetime
    filed: date
    period: date
    document: str = ""
    earnings: bool = False

    @cached_property
    def available_at(self) -> datetime:
        return first_session_open(self.accepted_at)


def detail_acceptance(payload: bytes) -> datetime:
    """The EDGAR archive detail page displays the authoritative acceptance timestamp in Eastern time."""
    match = re.search(rb'>\s*Accepted\s*</div>\s*<div[^>]*>\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*</div>',
                      payload, re.I)
    if not match:
        raise ValueError("SEC filing detail page has no acceptance timestamp")
    return acceptance(match[1].decode("ascii"))


def filing_index(cik: int, columns: Mapping[str, Any], *, start: date = date.min, end: date = date.max,
                 resolve_acceptance: Callable[[str], datetime] | None = None) -> Iterable[Filing]:
    """EDGAR submissions expose the issuer's 8-K item index alongside other forms."""
    for i, form in enumerate(columns.get("form", [])):
        base = form.removesuffix("/A")
        if base not in FORMS:
            continue
        filed = date.fromisoformat(columns["filingDate"][i])
        if not start <= filed <= end:
            continue
        accession = columns["accessionNumber"][i]
        if resolve_acceptance is not None:
            accepted_at = resolve_acceptance(accession)
        else:
            timestamps = columns.get("acceptanceDateTime", [])
            accepted = timestamps[i] if i < len(timestamps) else None
            if not accepted:
                continue
            try:
                accepted_at = submissions_acceptance(accepted)
            except ValueError:
                continue
        period = columns["reportDate"][i]
        items = columns.get("items", [])
        earnings = base in PERIODIC or (base == "8-K" and i < len(items) and "2.02" in re.split(r"[,\s]+", items[i]))
        yield Filing(cik, accession, form, accepted_at, filed,
                     date.fromisoformat(period) if period else filed, columns["primaryDocument"][i], earnings)


# Ordered aliases express a preference within an individual accession and period.
_CONCEPTS = {
    "revenue": ("USD", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax")),
    "eps": ("USD/shares", ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic")),
    "net_income": ("USD", ("NetIncomeLoss", "ProfitLoss")),
    "operating_cash_flow": ("USD", ("NetCashProvidedByUsedInOperatingActivities",
                                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    "assets": ("USD", ("Assets",)),
    "equity": ("USD", ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    "shares": ("shares", ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")),
    "gross_profit": ("USD", ("GrossProfit",)),
    "cost_of_revenue": ("USD", ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")),
}
INSTANT_METRICS = frozenset({"assets", "equity", "shares"})


@dataclass(frozen=True)
class Fact:
    metric: str
    start: date | None
    end: date
    value: float


def company_facts(payload: Mapping[str, Any], filings: Mapping[str, Filing]) -> dict[str, tuple[Fact, ...]]:
    gaap = payload.get("facts", {}).get("us-gaap", {})
    dei = payload.get("facts", {}).get("dei", {})
    out: dict[str, dict[tuple[str, date | None, date], Fact]] = defaultdict(dict)
    for metric, (unit, aliases) in _CONCEPTS.items():
        for tag in aliases:
            tagged: dict[tuple[str, date | None, date], set[float]] = defaultdict(set)
            for raw in (gaap.get(tag) or dei.get(tag) or {}).get("units", {}).get(unit, []):
                acc = raw["accn"]
                filing = filings.get(acc)
                if filing is None or filing.form.removesuffix("/A") not in PERIODIC:
                    continue
                if not isinstance(raw["val"], int | float) or not math.isfinite(raw["val"]):
                    continue
                start = date.fromisoformat(raw["start"]) if raw.get("start") else None
                end = date.fromisoformat(raw["end"])
                if (metric in INSTANT_METRICS) != (start is None) or end > filing.filed or (start and end < start):
                    continue
                # Share-count comparisons use fiscal-period snapshots, which can be matched year over year.
                if metric == "shares" and tag.startswith("Entity") and end != filing.period:
                    continue
                tagged[acc, start, end].add(float(raw["val"]))
            for (acc, start, end), values in tagged.items():
                if len(values) == 1:
                    out[acc].setdefault((metric, start, end), Fact(metric, start, end, next(iter(values))))
    return {acc: tuple(sorted(facts.values(), key=lambda f: (f.metric, f.end, f.start or f.end)))
            for acc, facts in out.items()}


@dataclass(frozen=True)
class Transaction:
    owner: str
    roles: tuple[str, ...]
    day: date
    code: str
    shares: float
    price: float | None
    row: int


@dataclass(frozen=True)
class Ownership:
    tickers: tuple[str, ...]
    owners: tuple[str, ...]
    original_filed: date | None
    transactions: tuple[Transaction, ...]


def normalize_ticker(symbol: str) -> str:
    return symbol.strip().upper().replace("-", ".")


_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,11}")
_PLACEHOLDERS = frozenset({"NONE", "NA"})


def _listed_symbols(raw: str) -> tuple[str, ...]:
    """Filer-typed Form 4 symbols can list share classes or carry placeholder text. Keep only ticker-shaped values."""
    symbols = {normalize_ticker(part) for part in raw.replace(";", ",").split(",")}
    return tuple(sorted(s for s in symbols if _SYMBOL.fullmatch(s) and s not in _PLACEHOLDERS))


class IssuerMismatch(ValueError):
    """The submissions CIK filed this Form 4 as a reporting owner of a different issuer."""


def form4(payload: bytes, cik: int) -> Ownership:
    # EDGAR owner XML requires no DTD or entity expansion. Reject both before parsing.
    if b"<!" in payload.replace(b"\x00", b"") and re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b",
                                                           payload.replace(b"\x00", b""), re.I):
        raise ValueError("Form 4 XML contains a DTD or entity declaration")
    root = ET.fromstring(payload)  # noqa: S314 - DTD and entity declarations rejected above
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    if int(root.findtext("issuer/issuerCik", "0")) != cik:
        raise IssuerMismatch("Form 4 issuer differs from the submissions CIK")
    symbols = _listed_symbols(root.findtext("issuer/issuerTradingSymbol", ""))
    original = root.findtext("dateOfOriginalSubmission")
    owners: dict[str, tuple[str, ...]] = {}
    for owner in root.findall("reportingOwner"):
        ident = owner.findtext("reportingOwnerId/rptOwnerCik", "").strip().lstrip("0")
        if not ident:
            continue
        rel = owner.find("reportingOwnerRelationship")
        roles: list[str] = []
        if rel is not None:
            for tag, role in (("isDirector", "director"), ("isOfficer", "officer"),
                              ("isTenPercentOwner", "ten_percent_owner"), ("isOther", "other")):
                if rel.findtext(tag, "").strip().lower() in {"1", "true"}:
                    roles.append(role)
            title = rel.findtext("officerTitle", "").strip()
            if title:
                roles.append(title)
        owners[ident] = tuple(roles)
    transactions: list[Transaction] = []
    for i, row in enumerate(root.findall("nonDerivativeTable/nonDerivativeTransaction")):
        code = row.findtext("transactionCoding/transactionCode", "").strip()
        direction = row.findtext("transactionAmounts/transactionAcquiredDisposedCode/value", "").strip()
        if code not in {"P", "S"} or (direction and direction != {"P": "A", "S": "D"}[code]):
            continue
        shares = float(row.findtext("transactionAmounts/transactionShares/value", "nan"))
        raw_price = row.findtext("transactionAmounts/transactionPricePerShare/value")
        price = float(raw_price) if raw_price else None
        day = date.fromisoformat(row.findtext("transactionDate/value", ""))
        if math.isfinite(shares) and shares > 0:
            if price is not None and (not math.isfinite(price) or price < 0):
                price = None
            transactions.extend(Transaction(owner, roles, day, code, shares, price, i) for owner, roles in owners.items())
    return Ownership(symbols, tuple(sorted(owners)), date.fromisoformat(original) if original else None,
                     tuple(transactions))


class _CoverSymbols(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.symbols: set[str] = set()
        self.current: list[str] | None = None
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "ix:hidden":
            self.hidden += 1
        if tag == "ix:nonnumeric" and not self.hidden and (dict(attrs).get("name") or "").lower() == "dei:tradingsymbol":
            self.current = []

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "ix:hidden":
            self.hidden = max(0, self.hidden - 1)
        if tag == "ix:nonnumeric" and self.current is not None:
            symbol = normalize_ticker("".join(self.current))
            if _SYMBOL.fullmatch(symbol):
                self.symbols.add(symbol)
            self.current = None


def cover_tickers(payload: bytes) -> tuple[str, ...]:
    parser = _CoverSymbols()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return tuple(sorted(parser.symbols))
