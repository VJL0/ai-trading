"""Request guards for the localhost API.

* All state-changing requests need ``X-LAT-Request: 1`` (blocks simple cross-site form posts) and,
  if an ``Origin`` header is present, it must be the local dashboard or the API itself.
* Operator endpoints additionally need the human operator token (var/<ns>/keys/operator.token).
* Engine bridge endpoints need the engine token (var/<ns>/keys/engine.token), given only to LEAN.
Tokens are compared in constant time and never logged. Models and the GUI agent never get either.
"""

from __future__ import annotations

import hmac
import secrets
from pathlib import Path

from fastapi import Header, HTTPException, Request, status

OPERATOR_TOKEN_FILE = "operator.token"  # noqa: S105 - a file name, not a secret
ENGINE_TOKEN_FILE = "engine.token"  # noqa: S105 - a file name, not a secret
AGENT_TOKEN_FILE = "agent.token"  # noqa: S105 - a file name, not a secret


def ensure_token(keys_dir: Path, filename: str) -> Path:
    keys_dir.mkdir(parents=True, exist_ok=True)
    path = keys_dir / filename
    if not path.exists():
        import os

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_urlsafe(32))
    return path


def read_token(keys_dir: Path, filename: str) -> str:
    return (keys_dir / filename).read_text().strip()


def tokens_match(presented: str | None, expected: str) -> bool:
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode(), expected.encode())


def allowed_origins(api_port: int, dashboard_port: int) -> frozenset[str]:
    hosts = ("127.0.0.1", "localhost")
    return frozenset(
        {f"http://{h}:{dashboard_port}" for h in hosts} | {f"http://{h}:{api_port}" for h in hosts}
        | {f"http://{h}:4173" for h in hosts}  # vite preview
    )


def csrf_guard(request: Request, x_lat_request: str | None = Header(default=None)) -> None:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    if x_lat_request != "1":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing X-LAT-Request header")
    origin = request.headers.get("origin")
    if origin is not None and origin not in request.app.state.allowed_origins:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "origin not allowed")


def operator_guard(request: Request, x_lat_operator_token: str | None = Header(default=None)) -> str:
    csrf_guard(request, request.headers.get("x-lat-request"))
    expected: str = request.app.state.operator_token
    if not tokens_match(x_lat_operator_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "operator token required")
    return "operator"


def engine_guard(request: Request, x_lat_engine_token: str | None = Header(default=None)) -> str:
    expected: str = request.app.state.engine_token
    if not tokens_match(x_lat_engine_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "engine token required")
    return "engine"


def agent_guard(request: Request, x_lat_agent_token: str | None = Header(default=None)) -> str:
    """Agent tool interface (MCP server / in-app AI runtimes). Never accepts the operator or engine token."""
    expected: str = request.app.state.agent_token
    if not tokens_match(x_lat_agent_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "agent token required")
    return request.headers.get("x-lat-agent-name", "agent")[:64]
