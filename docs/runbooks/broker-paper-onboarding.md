# Alpaca paper execution through the official SDK

The runtime's `ALPACA_PAPER` adapter uses `alpaca-py`. LEAN is for backtesting/research only.
No QuantConnect user ID, API token, deployment license, Docker deployment or LEAN engine is required for execution.
The old plugin sources remain archived and unmodified; their license checks have not been bypassed.

Store **paper** credentials in the macOS Keychain (service `local-ai-trader`) or the git-ignored `.env`:

- `ALPACA_PAPER_KEY_ID`
- `ALPACA_PAPER_SECRET_KEY`

The application rejects live (`AK`) keys and uses only `https://paper-api.alpaca.markets`.
Ambiguous `APCA_*` names are ignored. Do not put secrets in chat, source code, or shell command arguments.
Use `.env` for runtime settings. `.env.example` lists the available settings and their defaults.

## Launch

Check existing sessions, working orders and positions before switching gateways. Stop sessions first;
resolve existing broker orders and reconcile before shutting down. Keep the ledger and audit history.
Do not reset the broker account or delete ledger rows to make reconciliation pass.

```sh
make status
make stop       # stops local services; does not cancel broker-held orders
make paper      # same as make paper-sdk: PAPER + ALPACA_PAPER; IEX_FREE by default
```

The supervisor refuses a gateway mismatch with an already-running process; `make paper` does not reconfigure a running demo.
Confirm `gateway=ALPACA_PAPER`, `execution_label=PAPER`, live execution disabled, and CLEAN reconciliation
through `get_risk_status` / `get_system_health` before starting an MCP session.
Missing credentials block startup. `make demo` explicitly selects the simulated FAKE gateway.
SIP or paid model access is not needed or enabled by this workflow.

## Execution and recovery

The existing risk engine signs a LIMIT/DAY, whole-share intent. The ledger atomically claims it before
the adapter checks the signature, exact ledger payload, expiry, mode, and broker client ID. Signed decimal
values reach the SDK wire payload as exact strings. The client ID is retained across all lookups/recovery.
The SDK does not retry mutations. Any uncertain response is looked up by that same ID; unresolved outcomes
lock entries. A missing broker response is never treated as permission to submit again.

Order updates use bounded REST polling (two seconds by default), feeding the existing cumulative-fill
ledger and live activity/SSE. This adapter does not use a WebSocket. Disconnects lock entries; reconnect
runs reconciliation, and only a CLEAN result unlocks. Restart recovery resolves claimed/UNKNOWN orders,
books missed partial or final fills, and preserves the audit chain. Cancellation acknowledgement alone
is not treated as a final cancellation; subsequent broker state decides whether a fill won the race.

Account-wide reconciliation refuses unrecognized or mismatched orders/positions. Existing orders from a
retired gateway require review; changing gateway does not rewrite their signed intents or audit history.
The reader currently fails closed at its 500-order page limit instead of accepting truncated state.
Unsupported asset classes, bracket orders, extended hours, or malformed broker payloads also fail closed.

## Verification

```sh
uv run --frozen pytest -q tests/unit/order_ledger/test_alpaca_gateway.py tests/integration/order_ledger/test_alpaca_sdk_pipeline.py
uv run --frozen pytest -q -m paper_broker
```

The first command tests the actual SDK against controlled HTTP responses, including the real risk engine,
SQLite reopen, reconciliation and audit. It is **offline verification**, not hosted broker execution.
The second command is read-only broker connectivity and skips without credentials; it does not verify fills.
For hosted execution, start a short MCP paper session during regular market hours and inspect orders,
fill records, session report and CLEAN reconciliation. A NO_TRADE session alone does not verify execution.

SDK reference: [TradingClient orders](https://alpaca.markets/sdks/python/api_reference/trading/orders.html).
