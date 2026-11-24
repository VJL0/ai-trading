"""GET-only Alpaca HTTP client shared by capability probes and research data fetchers.

* Only ``GET`` is possible (there is no other verb method) and only allowlisted hosts are reachable: the PAPER
  trading API (read endpoints: account, assets, option contracts) and the market-data API.
* PAPER key ids (``PK…``) only; live keys are refused. Credentials never appear in ``repr``, logs or errors.
* Client-side rate limiting below Alpaca's free-plan 200 requests/minute, bounded retries on 429/5xx.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from packages.core.secrets import SecretName, SecretStore

PAPER_TRADING_HOST = "paper-api.alpaca.markets"
DATA_HOST = "data.alpaca.markets"
ALLOWED_HOSTS = frozenset({PAPER_TRADING_HOST, DATA_HOST})
TRADING_BASE = f"https://{PAPER_TRADING_HOST}"
DATA_BASE = f"https://{DATA_HOST}"


class ReadOnlyViolation(PermissionError):
    """A request that this client must never make (non-GET, non-allowlisted host, non-paper key)."""


class MissingAlpacaCredentials(RuntimeError):
    pass


class AlpacaHttpError(RuntimeError):
    def __init__(self, status: int, path: str, message: str) -> None:
        super().__init__(f"HTTP {status} for {path}: {message[:300]}")
        self.status = status
        self.path = path


class _RateLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.per_minute = max(1, per_minute)
        self._clock, self._sleep = clock, sleep
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.per_minute:
                    self._stamps.append(now)
                    return
                self._sleep(max(0.01, 60.0 - (now - self._stamps[0])))


class ReadOnlyAlpacaHttp:
    """Minimal GET-only client. ``get(url, params)`` returns parsed JSON or raises :class:`AlpacaHttpError`."""

    def __init__(self, key_id: str, secret_key: str, *, per_minute: int = 180, timeout_s: float = 30.0,
                 max_attempts: int = 8, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        key_id, secret_key = key_id.strip(), secret_key.strip()
        if key_id.startswith("AK"):
            raise ReadOnlyViolation("live Alpaca key id (prefix 'AK') refused; PAPER keys ('PK') only")
        if not key_id.startswith("PK"):
            raise ReadOnlyViolation("Alpaca key id does not look like a PAPER key id (expected prefix 'PK')")
        self._client = httpx.Client(
            headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key, "Accept": "application/json"},
            timeout=timeout_s, follow_redirects=False, transport=transport)
        self._limiter = _RateLimiter(per_minute, sleep=sleep)
        self._sleep = sleep
        self._max_attempts = max(1, max_attempts)
        self.requests = 0
        self.retries = 0

    @classmethod
    def from_secrets(cls, project_root: Path, **kwargs: Any) -> ReadOnlyAlpacaHttp:
        store = SecretStore(project_root)
        key = store.get(SecretName.ALPACA_PAPER_KEY_ID)
        secret = store.get(SecretName.ALPACA_PAPER_SECRET_KEY)
        if key is None or secret is None:
            raise MissingAlpacaCredentials(
                "Alpaca PAPER credentials are absent (Keychain service 'local-ai-trader' or the git-ignored .env)")
        return cls(key.get_secret_value(), secret.get_secret_value(), **kwargs)

    def __repr__(self) -> str:  # never include credentials
        return f"ReadOnlyAlpacaHttp(hosts={sorted(ALLOWED_HOSTS)}, read_only=True)"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ReadOnlyAlpacaHttp:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _check_url(url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
            raise ReadOnlyViolation(f"host not allowlisted for read-only Alpaca access: {parts.hostname!r}")
        return parts.path

    def get_response(self, url: str, params: Mapping[str, Any] | None = None) -> httpx.Response:
        path = self._check_url(url)
        delay = 1.0
        last: httpx.Response | None = None
        for attempt in range(self._max_attempts):
            self._limiter.acquire()
            self.requests += 1
            try:
                resp = self._client.get(url, params=dict(params or {}))
            except httpx.TransportError as exc:
                if attempt + 1 >= self._max_attempts:
                    raise AlpacaHttpError(0, path, f"transport error: {type(exc).__name__}") from None
                self.retries += 1
                self._sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last = resp
                if attempt + 1 >= self._max_attempts:
                    break
                self.retries += 1
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else delay
                self._sleep(min(max(wait, delay), 60.0))
                delay = min(delay * 2, 60.0)
                continue
            return resp
        assert last is not None
        return last

    def get(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        resp = self.get_response(url, params)
        if resp.status_code != 200:
            raise AlpacaHttpError(resp.status_code, urlsplit(url).path, resp.text)
        return resp.json()

    def paginate(self, url: str, params: Mapping[str, Any], *, items_key: str,
                 max_pages: int = 10_000) -> list[Any] | dict[str, Any]:
        """Follow ``next_page_token``. Dict payloads (bars keyed by symbol) are merged per key."""
        p = dict(params)
        merged_list: list[Any] = []
        merged_dict: dict[str, Any] = {}
        is_dict = False
        for _ in range(max_pages):
            data = self.get(url, p)
            chunk = data.get(items_key) if isinstance(data, dict) else data
            if isinstance(chunk, dict):
                is_dict = True
                for k, v in chunk.items():
                    if isinstance(v, list):  # bars/trades keyed by symbol: concatenate pages
                        merged_dict.setdefault(k, []).extend(v)
                    else:  # snapshots keyed by contract: one object per key
                        merged_dict[k] = v
            elif isinstance(chunk, list):
                merged_list.extend(chunk)
            token = data.get("next_page_token") if isinstance(data, dict) else None
            if not token:
                break
            p["page_token"] = token
        return merged_dict if is_dict else merged_list
