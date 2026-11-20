"""Exceptions raised by the order ledger service. All of them fail closed."""

from __future__ import annotations

from packages.contracts.interfaces import GatewayRejected


class LedgerError(Exception):
    """Base class for ledger refusals."""


class LedgerConflict(LedgerError):
    """A record with the same id already exists with *different* content (never silently overwritten)."""


class IntentRefused(LedgerError):
    """An intent was refused by the ledger (mode/gateway/signature/decision checks)."""


class UnlockDenied(PermissionError):
    """An entries-lock flag or halt may not be cleared (e.g. no CLEAN reconciliation since it was raised)."""


class CancelNotSupported(GatewayRejected):
    """The gateway cannot transmit a cancel for this order; the request never left the process."""


class RateLimited(GatewayRejected):
    """Broker refused the request with a rate limit (HTTP 429). The order was NOT accepted; never auto-retried."""


class StreamDisconnected(ConnectionError):
    """The broker order-update stream dropped. Entries stay locked until a CLEAN reconciliation."""


class SimulatedCrash(BaseException):  # deliberately not an Exception: nothing may swallow it
    """Failure injection: the process 'dies' right after the broker accepted an order.

    Derives from ``BaseException`` so that ``except Exception`` handlers in the execution path cannot
    record anything -- exactly like a real crash. Tests then 'restart' on the same SQLite file.
    """


class BrokerPayloadError(ValueError):
    """A broker / engine payload could not be validated (malformed, unsupported feature, float money)."""


class BrokerUnavailable(ConnectionError):
    """Broker state could not be read (connection error, timeout, 5xx)."""


class BrokerAuthError(PermissionError):
    """Broker refused the credentials (401/403). The message never contains the credential."""


class LiveKeyRefused(PermissionError):
    """A key that looks like a LIVE Alpaca key was supplied where only PAPER keys are permitted."""


class MissingPaperCredentials(RuntimeError):
    """Alpaca PAPER credentials are required for this gateway but are absent."""


class ReadOnlyViolation(PermissionError):
    """A read-only broker client attempted a non-GET request or a non-paper endpoint."""


class BrokerRateLimited(BrokerUnavailable):
    """A read-only broker request stayed rate limited (HTTP 429) after bounded backoff."""
