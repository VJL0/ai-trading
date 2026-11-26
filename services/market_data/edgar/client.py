"""Explicit SEC downloads with a shared request budget and a disk response cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

DATA = "https://data.sec.gov"
WWW = "https://www.sec.gov"
_HOSTS = {"data.sec.gov", "www.sec.gov"}
_RETRY = {429, 500, 502, 503, 504}
_REQUEST_INTERVAL_S = 0.125


def _pace(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    # Coordinate both SEC origins and simultaneous fetches using this cache.
    with (cache / "request-clock").open("a+") as clock:
        fcntl.flock(clock, fcntl.LOCK_EX)
        clock.seek(0)
        previous = float(clock.read() or "0")
        elapsed = time.monotonic() - previous
        if 0 <= elapsed < _REQUEST_INTERVAL_S:
            time.sleep(_REQUEST_INTERVAL_S - elapsed)
        clock.seek(0)
        clock.truncate()
        clock.write(str(time.monotonic()))


def _retry_delay(header: str | None, attempt: int) -> float:
    base = float(2**attempt)
    if header:
        if header.isdecimal():
            return max(base, float(header))
        try:
            retry_at = parsedate_to_datetime(header)
            return max(base, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            pass
    return base


class EdgarClient:
    def __init__(self, cache: Path, *, transport: httpx.BaseTransport | None = None) -> None:
        agent = os.environ.get("LAT_SEC_USER_AGENT", "").strip()
        if not agent:
            raise ValueError('SEC downloads require LAT_SEC_USER_AGENT set to "Name email".')
        if not agent.isascii() or not agent.isprintable():
            raise ValueError("LAT_SEC_USER_AGENT must be a printable ASCII HTTP header value.")
        self.cache = cache
        self.http = httpx.Client(headers={"User-Agent": agent}, timeout=30, transport=transport, follow_redirects=False)

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.http.close()

    def get(self, url: str, *, max_age: float | None = None) -> bytes:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc not in _HOSTS:
            raise ValueError("SEC downloads require an approved SEC HTTPS origin")
        path = self._cache_path(url)
        if path.is_file() and (max_age is None or time.time() - path.stat().st_mtime <= max_age):
            return path.read_bytes()
        for attempt in range(4):
            _pace(self.cache)
            try:
                response = self.http.get(url)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                time.sleep(_retry_delay(None, attempt))
                continue
            if response.status_code in _RETRY and attempt < 3:
                time.sleep(_retry_delay(response.headers.get("Retry-After"), attempt))
                continue
            response.raise_for_status()
            self.cache.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as tmp:
                tmp.write(response.content)
                temporary = Path(tmp.name)
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return response.content
        raise AssertionError("retry loop did not terminate")

    def _cache_path(self, url: str) -> Path:
        return self.cache / hashlib.sha256(url.encode()).hexdigest()

    def company_tickers(self) -> tuple[dict[str, Any], datetime]:
        url = f"{WWW}/files/company_tickers.json"
        payload = json.loads(self.get(url, max_age=86400))
        return payload, datetime.fromtimestamp(self._cache_path(url).stat().st_mtime, UTC)

    def submissions(self, cik: int) -> dict[str, Any]:
        return json.loads(self.get(f"{DATA}/submissions/CIK{cik:010d}.json", max_age=3600))

    def historical_submissions(self, name: str) -> dict[str, Any]:
        if not re.fullmatch(r"CIK\d{10}-submissions-\d+\.json", name):
            raise ValueError("invalid SEC submissions archive name")
        return json.loads(self.get(f"{DATA}/submissions/{name}"))

    def companyfacts(self, cik: int) -> dict[str, Any]:
        return json.loads(self.get(f"{DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json", max_age=3600))

    def filing_detail(self, cik: int, accession: str) -> bytes:
        return self.filing(cik, accession, f"{accession}-index.htm")

    def filing(self, cik: int, accession: str, document: str) -> bytes:
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("invalid SEC accession")
        # Form 4 primaryDocument can be an XSL-rendered path. The raw XML is at the filing root.
        name = document.rsplit("/", 1)[-1]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("invalid SEC filing document name")
        return self.get(f"{WWW}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{name}")
