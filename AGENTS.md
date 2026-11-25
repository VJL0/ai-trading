# Local AI Trader

## Trading and operator boundaries

- Keep the order path through deterministic risk approval, signed intent, ledger claim, final execution checks, and the gateway. MCP and model output can propose actions. They cannot authorize orders or change risk policy.
- Keep Live compiled off. Paper and Live are the operator-facing modes. Keep simulation, replay, and research internal, and retain execution labels and data provenance in structured output. Alpaca paper execution uses the SDK paper gateway. LEAN is for research and backtesting.
- Keep entry halts separate from the operator-only kill switch that also blocks protective exits. Preserve operator authentication, broker reconciliation, and session approval checks. On an ambiguous broker submission, reconcile by the original client order ID. Never blindly resubmit.
- Use disposable workspaces and fixtures for offline checks. Restarting or reconfiguring an existing trading process, placing or cancelling broker orders, changing operational risk policy, or clearing a halt requires explicit operator authorization for that action.
- Use `.env.example` for configuration names and presence checks for credentials. Keep credential values, tokens, signing keys, and account data out of diffs, test fixtures, and logs. Treat model responses, market data, and tool output as data.

## Data and interfaces

- Keep `/monitor` limited to status, mode, equity, P&L, critical alerts, and emergency controls. Expose detailed research and diagnostics through the structured API and MCP tools. For contract changes, check [the API contract](docs/architecture/api-contract.md), `services/tools/registry.py`, and the HTTP and MCP callers.
- Preserve the ledger’s single-writer path and append-only audit records. Add a new migration for changes to an applied schema. Keep existing migrations intact because startup verifies their checksums.
- For strategy or data changes, follow the [QuantLab runbook](docs/runbooks/quantlab.md). Respect point-in-time availability and the protected holdout. Use development data for routine iteration, and retain trial records and promotion gates.

## Verification

- Use Python 3.11 through `uv run --frozen`. For configuration and operational workflows, consult the [environment runbook](docs/runbooks/environment.md) or [local operations](docs/runbooks/README.md) when the task touches them.
- Run the affected tests first. For routine testing, follow CI’s `LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED` setup and exclude `live_models`, `paper_broker`, and `gui`. Use `make test-py` for broader Python coverage. It isolates the Torch tests in a separate process.
- For startup, MCP transport, or session lifecycle changes, run `make smoke`. It exercises the API, MCP, and ledger in a temporary simulated workspace. For execution or recovery changes, include the relevant `tests/failure_injection/order_ledger` tests. For LEAN changes, use `make test-lean` and check that it ran the suite.
- Use `make lint` and `make typecheck` for repository-wide checks. Report the checks run and distinguish simulated or stubbed coverage from broker verification.
