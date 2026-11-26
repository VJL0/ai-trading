"""A local, accession-versioned research store, independent of the trading ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from services.market_data.calendar import NY

from .parsers import Fact, Filing, Ownership, Transaction, normalize_ticker

_SCHEMA = """
CREATE TABLE IF NOT EXISTS disclosures (
    cik INTEGER NOT NULL, accession TEXT NOT NULL, accepted_at TEXT NOT NULL, available_at TEXT NOT NULL,
    form TEXT NOT NULL, filed TEXT NOT NULL, period TEXT NOT NULL, earnings INTEGER NOT NULL,
    facts TEXT NOT NULL, ownership TEXT, tickers TEXT NOT NULL,
    PRIMARY KEY (cik, accession)
);
CREATE INDEX IF NOT EXISTS disclosures_asof ON disclosures(available_at, cik);
"""


@dataclass(frozen=True)
class Disclosure:
    filing: Filing
    facts: tuple[Fact, ...] = ()
    ownership: Ownership | None = None
    tickers: tuple[str, ...] = ()

    @property
    def available_at(self) -> datetime:
        return self.filing.available_at

    def effective_tickers(self, previous: tuple[str, ...]) -> tuple[str, ...]:
        # A one-symbol Form 4 can confirm an existing share class without discarding its siblings.
        if not self.tickers or (self.ownership is not None and set(self.tickers).issubset(previous)):
            return previous
        return self.tickers


def store_path(project_root: Path) -> Path:
    return project_root / "datasets" / "quantlab" / "edgar" / "edgar.sqlite3"


class EdgarStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def ingest(self, records: Iterable[Disclosure]) -> None:
        """Add immutable accession versions. Reruns can fill in missing payload fields."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(_SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            for record in records:
                filing = record.filing
                previous = db.execute("SELECT * FROM disclosures WHERE cik=? AND accession=?",
                                      (filing.cik, filing.accession)).fetchone()
                atoms = (filing.cik, filing.accession, filing.accepted_at.astimezone(UTC).isoformat(),
                         record.available_at.isoformat(), filing.form, filing.filed.isoformat(),
                         filing.period.isoformat(), int(filing.earnings))
                facts = json.dumps([asdict(f) for f in record.facts], default=str)
                ownership = json.dumps(asdict(record.ownership), default=str) if record.ownership is not None else None
                tickers = json.dumps(record.tickers)
                if previous is not None:
                    if previous[:8] != atoms:
                        raise ValueError(f"conflicting filing metadata for accession {filing.accession}")
                    if record.facts:
                        # Preserve first sourced fact values per accession/metric/period. Restatements carry a new accession.
                        key_fields = ("metric", "start", "end")
                        merged = {tuple(f[k] for k in key_fields): f for f in json.loads(facts)}
                        merged.update({tuple(f[k] for k in key_fields): f for f in json.loads(previous[8])})
                        facts = json.dumps(list(merged.values()))
                    else:
                        facts = previous[8]
                    if previous[9] is not None:
                        ownership = previous[9]
                    if previous[10] != "[]":
                        tickers = previous[10]
                db.execute("INSERT OR REPLACE INTO disclosures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (*atoms, facts, ownership, tickers))

    def snapshot_tickers(self, tickers: dict[int, tuple[str, ...]], observed_at: datetime) -> None:
        """Record the dated SEC directory snapshot without assigning it to past sessions."""
        day = observed_at.astimezone(NY).date()
        self.ingest(Disclosure(Filing(cik, f"directory@{observed_at.astimezone(UTC).isoformat()}", "directory",
                                      observed_at, day, day), tickers=symbols) for cik, symbols in sorted(tickers.items()))

    def as_of(self, instant: datetime, *, cik: int | None = None,
              symbols: Iterable[str] | None = None) -> Iterator[Disclosure]:
        """Yield eligible versions, including amendments, in acceptance order within each session."""
        if instant.utcoffset() is None:
            raise ValueError("as-of time must be timezone-aware")
        if not self.path.is_file():
            return
        asof = instant.astimezone(UTC).isoformat()
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            where = "available_at <= ?"
            params: list[str | int] = [asof]
            if cik is not None:
                where += " AND cik = ?"
                params.append(cik)
            if symbols is not None:
                names = sorted({normalize_ticker(symbol) for symbol in symbols})
                # Identify candidate CIKs from any eligible ticker version, then keep their complete prior histories.
                where += (
                    " AND cik IN (SELECT DISTINCT d.cik FROM disclosures AS d, json_each(d.tickers) AS t"
                    " WHERE d.available_at <= ? AND t.value IN (SELECT value FROM json_each(?)))"
                )
                params.extend((asof, json.dumps(names)))
            query = ("SELECT * FROM disclosures WHERE " + where +  # noqa: S608 - fixed predicates above
                     " ORDER BY available_at, accepted_at, accession, cik")
            for row in db.execute(query, params):
                facts = tuple(Fact(f["metric"], date.fromisoformat(f["start"]) if f["start"] else None,
                                   date.fromisoformat(f["end"]), f["value"]) for f in json.loads(row[8]))
                ownership = None
                if row[9] is not None:
                    own = json.loads(row[9])
                    transactions = tuple(Transaction(t["owner"], tuple(t["roles"]), date.fromisoformat(t["day"]),
                                                     t["code"], t["shares"], t["price"], t["row"])
                                         for t in own["transactions"])
                    original = own["original_filed"]
                    ownership = Ownership(tuple(own["tickers"]), tuple(own["owners"]),
                                          date.fromisoformat(original) if original else None, transactions)
                yield Disclosure(Filing(row[0], row[1], row[4], datetime.fromisoformat(row[2]),
                                        date.fromisoformat(row[5]), date.fromisoformat(row[6]), earnings=bool(row[7])),
                                 facts, ownership, tuple(json.loads(row[10])))

    def coverage(self, through: datetime) -> frozenset[str]:
        """Mechanism prerequisites in eligible, symbol-addressable observations."""
        if through.utcoffset() is None:
            raise ValueError("coverage cutoff must be timezone-aware")
        if not self.path.is_file():
            return frozenset()
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            records = db.execute(
                "SELECT max(earnings), max(facts != '[]'), max(ownership IS NOT NULL) FROM disclosures "
                "WHERE available_at <= ? AND cik IN "
                "(SELECT cik FROM disclosures WHERE available_at <= ? AND tickers != '[]')",
                (through.astimezone(UTC).isoformat(),) * 2,
            ).fetchone()
        return frozenset(name for name, available in zip(("earnings", "fundamentals", "insider"), records, strict=True)
                         if available)

    def digest(self, through: datetime) -> str | None:
        """A summary of eligible observations, which changes when a fetch adds or fills in prior disclosures."""
        if through.utcoffset() is None:
            raise ValueError("digest cutoff must be timezone-aware")
        if not self.path.is_file():
            return None
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute(
                "SELECT count(*), max(available_at), total(length(facts)), total(length(coalesce(ownership, ''))), "
                "total(length(tickers)) FROM disclosures WHERE available_at <= ?",
                (through.astimezone(UTC).isoformat(),),
            ).fetchone()
        return hashlib.sha256(json.dumps(list(row)).encode()).hexdigest()[:16]

    def cik_for_ticker(self, symbol: str, instant: datetime) -> int | None:
        """Resolve an active ticker using only dated CIK evidence visible at the requested instant."""
        name = normalize_ticker(symbol)
        current: dict[int, tuple[str, ...]] = {}
        for record in self.as_of(instant, symbols=(name,)):
            cik = record.filing.cik
            current[cik] = record.effective_tickers(current.get(cik, ()))
        matches = [cik for cik, tickers in current.items() if name in tickers]
        return matches[0] if len(matches) == 1 else None
