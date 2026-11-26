from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from services.market_data.edgar import __main__ as cli
from services.market_data.edgar import client
from services.market_data.edgar.parsers import (
    acceptance,
    cover_tickers,
    detail_acceptance,
    filing_index,
    first_session_open,
    form4,
)
from services.market_data.edgar.store import EdgarStore, store_path


def test_client_requires_env_and_retries_then_reads_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAT_SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="LAT_SEC_USER_AGENT"):
        client.EdgarClient(tmp_path)
    agent = "fixture-operator"
    monkeypatch.setenv("LAT_SEC_USER_AGENT", agent)
    pacing: list[bool] = []
    delays: list[float] = []
    requests: list[httpx.Request] = []
    monkeypatch.setattr(client, "_pace", lambda _: pacing.append(True))
    monkeypatch.setattr(client.time, "sleep", delays.append)

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"facts": {}})

    with client.EdgarClient(tmp_path, transport=httpx.MockTransport(serve)) as sec:
        assert sec.companyfacts(123) == sec.companyfacts(123) == {"facts": {}}
        with pytest.raises(ValueError, match="HTTPS origin"):
            sec.get("https://www.sec.gov.invalid/doc")
        sec.filing(123, "0000000123-24-000001", "xslF345X05/form4.xml")
    assert len(requests) == len(pacing) == 3
    assert delays == [3]
    assert all(request.headers["User-Agent"] == agent for request in requests)
    assert str(requests[-1].url).endswith("/123/000000012324000001/form4.xml")


def test_acceptance_uses_next_tradable_open_and_dst() -> None:
    friday = acceptance("2024-03-08T16:01:00")
    assert acceptance("20240308160100") == friday
    assert detail_acceptance(b'<div class="infoHead">Accepted</div>\n<div class="info">2024-03-08 16:01:00</div>') == friday
    with pytest.raises(ValueError, match="time of day"):
        acceptance("2024-03-08")
    assert first_session_open(friday) == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert first_session_open(acceptance("2024-07-03T17:00:01Z")) == datetime(2024, 7, 5, 13, 30, tzinfo=UTC)
    assert first_session_open(acceptance("2024-07-05T09:29:59-04:00")) == datetime(2024, 7, 5, 13, 30, tzinfo=UTC)
    assert first_session_open(acceptance("2024-07-05T09:30:01-04:00")) == datetime(2024, 7, 8, 13, 30, tzinfo=UTC)


def test_issuer_8k_index_and_periodic_dates() -> None:
    forms = ["8-K", "8-K", "10-Q", "10-K/A", "4"]
    rows = list(filing_index(7, {
        "form": forms, "filingDate": ["2024-05-10"] * 5, "reportDate": ["2024-03-31"] * 5,
        "items": ["7.01, 9.01", "2.02,9.01", "", "", ""], "primaryDocument": ["x.xml"] * 5,
        "accessionNumber": [str(i) for i in range(5)],
        "acceptanceDateTime": ["2024-05-10T12:00:00Z"] * 4 + [""],
    }))
    assert [f.earnings for f in rows] == [False, True, True, True]
    assert rows[2].filed == date(2024, 5, 10)
    assert rows[2].period == date(2024, 3, 31)
    assert cover_tickers(b'<ix:nonNumeric name="dei:TradingSymbol"> old-a </ix:nonNumeric>') == ("OLD.A",)


def test_fetch_cli_uses_archive_acceptance_not_submissions_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    accession = "0000000007-24-000001"
    columns = {
        "form": ["8-K"], "filingDate": ["2024-03-08"], "reportDate": ["2024-03-07"],
        "items": ["2.02,9.01"], "primaryDocument": ["earnings.htm"],
        "accessionNumber": [accession], "acceptanceDateTime": ["2024-03-08T10:15:00Z"],
    }

    def serve(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "TST", "cik_str": 7}})
        if path == "/submissions/CIK0000000007.json":
            return httpx.Response(200, json={"filings": {"recent": columns, "files": []}})
        if path.endswith(f"/{accession}-index.htm"):
            return httpx.Response(200, content=b'<div class="infoHead">Accepted</div><div>2024-03-08 16:15:00</div>')
        raise AssertionError(path)

    monkeypatch.setenv("LAT_SEC_USER_AGENT", "fixture-operator")
    monkeypatch.setattr(client, "_pace", lambda _: None)
    monkeypatch.setattr(cli, "EdgarClient", lambda cache: client.EdgarClient(cache, transport=httpx.MockTransport(serve)))
    assert cli.main(["fetch", "TST", "--root", str(tmp_path), "--start", "2024-03-08", "--end", "2024-03-08"]) == 0
    store = EdgarStore(store_path(tmp_path))
    assert list(store.as_of(datetime(2024, 3, 8, 23, tzinfo=UTC), cik=7)) == []
    filings = list(store.as_of(datetime(2024, 3, 11, 13, 30, tzinfo=UTC), cik=7))
    assert len(filings) == 1 and filings[0].filing.earnings and filings[0].filing.accession == accession


def _transaction(code: str, shares: int, price: str = "12.5") -> str:
    direction = "D" if code == "S" else "A"
    return f"""<nonDerivativeTransaction>
      <transactionDate><value>2024-05-08</value></transactionDate>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{direction}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>"""


def test_form4_open_market_roles_and_economics() -> None:
    rows = "".join(_transaction(code, num) for code, num in (("P", 50), ("S", 25), ("A", 10_000), ("M", 12_000)))
    payload = f"""<ownershipDocument>
      <issuer><issuerCik>7</issuerCik><issuerTradingSymbol>ABC</issuerTradingSymbol></issuer>
      <reportingOwner>
        <reportingOwnerId><rptOwnerCik>00044</rptOwnerCik></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>true</isOfficer>
          <officerTitle>CFO</officerTitle>
        </reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>{rows}</nonDerivativeTable>
    </ownershipDocument>""".encode()
    own = form4(payload, 7)
    assert own.tickers == ("ABC",)
    assert [(t.code, t.shares, t.price) for t in own.transactions] == [("P", 50, 12.5), ("S", 25, 12.5)]
    assert own.transactions[0].roles == ("director", "officer", "CFO")
    assert own.transactions[0].owner == "44"
    with pytest.raises(ValueError, match="issuer differs"):
        form4(payload, 8)
    with pytest.raises(ValueError, match="DTD"):
        form4(b"<!DOCTYPE ownershipDocument>" + payload, 7)


def _ownership_xml(issuer_cik: int, symbol: str) -> bytes:
    return f"""<ownershipDocument>
      <issuer><issuerCik>{issuer_cik}</issuerCik><issuerTradingSymbol>{symbol}</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0000000044</rptOwnerCik></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>{_transaction("P", 10)}</nonDerivativeTable>
    </ownershipDocument>""".encode()


def test_form4_symbol_must_be_a_ticker_not_free_text() -> None:
    assert form4(_ownership_xml(7, "googl, goog"), 7).tickers == ("GOOG", "GOOGL")
    for placeholder in ("N/A", "NONE", "  ", "Class A Common Stock"):
        assert form4(_ownership_xml(7, placeholder), 7).tickers == ()


def test_fetch_skips_form4_where_the_issuer_cik_is_only_the_reporting_owner(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    own, held = "0000000007-24-000002", "0000000007-24-000003"
    columns = {
        "form": ["4", "4"], "filingDate": ["2024-03-08"] * 2, "reportDate": ["2024-03-07"] * 2,
        "items": ["", ""], "primaryDocument": ["xslF345X05/own.xml", "xslF345X05/held.xml"],
        "accessionNumber": [own, held], "acceptanceDateTime": ["", ""],
    }

    def serve(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"ticker": "TST", "cik_str": 7}})
        if path == "/submissions/CIK0000000007.json":
            return httpx.Response(200, json={"filings": {"recent": columns, "files": []}})
        if path.endswith("-index.htm"):
            return httpx.Response(200, content=b'<div class="infoHead">Accepted</div><div>2024-03-08 08:15:00</div>')
        if path.endswith("/own.xml"):
            return httpx.Response(200, content=_ownership_xml(7, "TST"))
        if path.endswith("/held.xml"):  # CIK 7 files this Form 4 as a 10% owner of another issuer
            return httpx.Response(200, content=_ownership_xml(99, "OTHR"))
        raise AssertionError(path)

    monkeypatch.setenv("LAT_SEC_USER_AGENT", "fixture-operator")
    monkeypatch.setattr(client, "_pace", lambda _: None)
    monkeypatch.setattr(cli, "EdgarClient", lambda cache: client.EdgarClient(cache, transport=httpx.MockTransport(serve)))
    assert cli.main(["fetch", "7", "--root", str(tmp_path), "--start", "2024-03-08", "--end", "2024-03-08"]) == 0
    stored = list(EdgarStore(store_path(tmp_path)).as_of(datetime(2024, 3, 8, 15, tzinfo=UTC), cik=7))
    assert [(d.filing.accession, d.tickers) for d in stored] == [(own, ("TST",))]


def test_submissions_acceptance_z_suffix_never_advances_availability() -> None:
    from datetime import UTC, datetime

    from services.market_data.edgar.parsers import first_session_open, submissions_acceptance

    # 16:30 wall clock on a Tuesday with a "Z" suffix: if it is really Eastern, the filing came after the close,
    # so it must not be available before the next session's open.
    accepted = submissions_acceptance("2024-02-06T16:30:00.000Z")
    assert accepted == datetime(2024, 2, 6, 21, 30, tzinfo=UTC)
    assert first_session_open(accepted) == datetime(2024, 2, 7, 14, 30, tzinfo=UTC)
