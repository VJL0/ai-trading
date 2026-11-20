"""Errors raised by the session manager. ``status`` is an HTTP-style code the tool layer passes through:
403 refused by mode/label, 404 unknown id, 409 conflict (another session active, closed market, overlap),
422 invalid arguments, 503 manager not ready."""

from __future__ import annotations


class SessionError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
