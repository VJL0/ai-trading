"""Operator-invoked EDGAR fetch. Importing the research store does no network I/O."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import httpx

from packages.core.config import PROJECT_ROOT
from services.market_data.calendar import NY

from .client import EdgarClient
from .parsers import (
    PERIODIC,
    Fact,
    Filing,
    IssuerMismatch,
    company_facts,
    cover_tickers,
    detail_acceptance,
    filing_index,
    form4,
    normalize_ticker,
)
from .store import Disclosure, EdgarStore, store_path


def _filings(client: EdgarClient, cik: int, start: date, end: date) -> dict[str, Filing]:
    index = client.submissions(cik)["filings"]
    columns = [index["recent"]]
    for block in index["files"]:
        if date.fromisoformat(block["filingTo"]) >= start and date.fromisoformat(block["filingFrom"]) <= end:
            columns.append(client.historical_submissions(block["name"]))

    def accepted_at(accession: str) -> datetime:
        try:
            return detail_acceptance(client.filing_detail(cik, accession))
        except ValueError as exc:
            raise ValueError(f"{accession}: {exc}") from exc

    return {f.accession: f for batch in columns for f in filing_index(
        cik, batch, start=start, end=end, resolve_acceptance=accepted_at,
    )}


def _fetch_issuer(client: EdgarClient, store: EdgarStore, cik: int, start: date, end: date) -> int:
    filings = _filings(client, cik, start, end)
    payload: dict[str, tuple[Fact, ...]] = {}
    if any(f.form.removesuffix("/A") in PERIODIC for f in filings.values()):
        try:
            company = client.companyfacts(cik)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
        else:
            payload = company_facts(company, filings)
    for f in sorted(filings.values(), key=lambda filing: filing.accepted_at):
        base = f.form.removesuffix("/A")
        if base == "4":
            try:
                own = form4(client.filing(cik, f.accession, f.document), cik)
            except IssuerMismatch:
                continue  # this CIK reported as an owner of another issuer's securities
            record = Disclosure(f, ownership=own, tickers=own.tickers)
        elif base in PERIODIC:
            symbols = cover_tickers(client.filing(cik, f.accession, f.document))
            record = Disclosure(f, facts=payload.get(f.accession, ()), tickers=symbols)
        else:
            record = Disclosure(f)
        store.ingest((record,))
    return len(filings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download and index the specified issuers")
    fetch.add_argument("issuers", nargs="+", help="CIKs or current SEC tickers")
    fetch.add_argument("--start", type=date.fromisoformat, required=True, help="earliest filing date (YYYY-MM-DD)")
    fetch.add_argument("--end", type=date.fromisoformat, default=datetime.now(NY).date(),
                       help="latest filing date (YYYY-MM-DD)")
    fetch.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root containing datasets/quantlab")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    store = EdgarStore(store_path(args.root))
    try:
        with EdgarClient(store.path.parent / "cache") as client:
            directory: dict[int, set[str]] = defaultdict(set)
            by_symbol: dict[str, set[int]] = defaultdict(set)
            tickers, observed_at = client.company_tickers()
            for row in tickers.values():
                symbol, cik = normalize_ticker(row["ticker"]), int(row["cik_str"])
                directory[cik].add(symbol)
                by_symbol[symbol].add(cik)
            selected: set[int] = set()
            for requested in args.issuers:
                if requested.isascii() and requested.isdecimal():
                    cik = int(requested)
                    if not 1 <= cik <= 9_999_999_999:
                        raise ValueError("CIK must be a positive integer of at most 10 digits")
                    selected.add(cik)
                    continue
                ciks = by_symbol[normalize_ticker(requested)]
                if len(ciks) != 1:
                    raise ValueError(f"ticker {requested!r} has no unique current SEC CIK, use an explicit CIK")
                selected.update(ciks)
            # This is evidence of what the directory showed at retrieval, never a historical ticker map.
            store.snapshot_tickers({cik: tuple(sorted(directory[cik])) for cik in selected if directory[cik]},
                                   observed_at)
            counts = {cik: _fetch_issuer(client, store, cik, args.start, args.end) for cik in sorted(selected)}
    except httpx.HTTPStatusError as exc:
        parser.exit(2, f"EDGAR fetch failed: SEC returned HTTP {exc.response.status_code} for {exc.request.url}\n")
    except httpx.HTTPError as exc:
        parser.exit(2, f"EDGAR fetch failed: {type(exc).__name__} contacting SEC EDGAR\n")
    except ValueError as exc:
        parser.exit(2, f"EDGAR fetch failed: {exc}\n")
    print(json.dumps({"source": "SEC EDGAR", "store": str(store.path), "filings_by_cik": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
