# Alpaca SDK adapter verification — 2026-11-09

**Update:** hosted Alpaca paper buy/sell execution is now verified. See
[hosted verification and remaining ingestion work](alpaca-hosted-paper.md). The initial
missing-credentials results below are retained as historical verification evidence.

## Implemented

- New `ALPACA_PAPER` execution gateway built on the pinned official `alpaca-py` SDK.
- Existing risk authorization, signed intent validation, durable submission claims, lifecycle, reconciliation,
  fill accounting and audit are retained. No live execution is enabled.
- Exact decimal order payloads and stable client order IDs. Preflight lookup and an atomic ledger claim
  prevent duplicate submissions. SDK mutation retries are disabled.
- Uncertain mutation outcomes require broker reconciliation. Timeouts, failed lookups, interrupted cancels,
  and disconnected polling fail closed; no automatic order resubmission.
- Partial fills, cancel/fill races, reconnect reconciliation, and recovery after reopening SQLite are covered.
- Two-second REST order polling feeds the existing ledger/activity/SSE; this is not a WebSocket implementation.
- Runtime LEAN execution choices and old execution launchers are retired. LEAN research/backtests and
  isolated legacy regression harnesses remain. The build no longer clones/builds/bundles the Alpaca plugin
  or `LatAlpacaGate`. Upstream license validation code was not modified.
- `make paper` selects the SDK paper gateway and checks credentials. A running gateway mismatch is refused.

## Evidence and limitations

The SDK pipeline integration test uses the real risk engine, signer, ledger, SDK serialization, reconciliation
and audit against controlled HTTP responses. It closes and reopens SQLite before consuming a broker fill,
then confirms that recovery books the fill exactly once without another submission. Failure injection also
covers delayed acceptance after timeout, concurrent duplicates, wrong client-ID payloads and reconnects.
These are offline results, not evidence of hosted Alpaca execution.

A separate real-process MCP -> HTTP API smoke session completed three scheduled decision cycles with the
operator disconnected. It produced NO_TRADE decisions and verified 29 audit rows. This was explicitly
SIMULATED; it verifies the session infrastructure, not broker fills.

Alpaca paper credentials are absent. The three credential-gated broker/market-data tests skip, and `make paper`
fails before launch with an explicit missing-credentials message. No hosted paper submit, fill, cancel, or
session has been verified. Required locally: `ALPACA_PAPER_KEY_ID` and `ALPACA_PAPER_SECRET_KEY`.
No QuantConnect deployment credentials or license are required for the new adapter.

The reader fails closed on a full 500-order result page rather than accepting truncated reconciliation.
Existing foreign-gateway orders or account/ledger discrepancies remain locked for operator review; signed
intents and history are not rewritten during migration. Fresh `Settings()` still defaults to the safe FAKE
simulator; `make paper` explicitly selects `ALPACA_PAPER`, while `make demo` explicitly selects FAKE.

## Checks

**Final full Python acceptance: 1,828 passed, 8 deselected, 301.87 seconds.**
Command: `uv run --frozen pytest -q -m 'not live_models and not paper_broker and not gui'`.
- Focused SDK lifecycle/failure, real risk/SQLite pipeline and launch checks: 22 passed.
- Fresh LEAN build without Alpaca plugin: passed; 84 .NET tests passed.
- Dashboard: 42 tests, production build, lint/typecheck and 3 Chromium E2E tests passed.
- Ruff: clean; Pyright: zero errors.
- Real broker checks: 3 skipped (missing credentials).
- Existing pinned LEAN dependencies still emit upstream NuGet vulnerability warnings; this change does not
  upgrade them or suppress those warnings.

An earlier full run passed 1,824 tests and failed one obsolete expectation that the LEAN Alpaca plugin be
loaded. That assertion now requires zero plugin copies, and its real-engine rerun passed. The subsequent full acceptance run passed all 1,828 selected tests.

API reference consulted: [official alpaca-py order methods](https://alpaca.markets/sdks/python/api_reference/trading/orders.html).
