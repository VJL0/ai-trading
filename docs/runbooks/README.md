# Local operations

Use `make demo` for a no-cost, synthetic, paper-only launch. It starts the API at `127.0.0.1:8765` and the
read-only dashboard at `127.0.0.1:5173`. Existing processes are preserved rather than restarted.
`make dev` uses your configured environment; `make demo` explicitly selects FAKE/SIMULATED/NONE and disables Keychain.

The dashboard shows decision stages, current decisions, explanations, order transitions, session status,
performance, health, and a causal event timeline. It subscribes to SSE, resynchronizes on reconnect, and polls
while disconnected. It has no trading controls or token entry. Animations honor reduced-motion preferences.

Use the AI's MCP tools to start/stop sessions and create/cancel schedules. `make halt` stops new entries;
`stop_session` stops future decision cycles. Neither liquidates positions or cancels existing broker orders.
Only the operator CLI can clear a halt under the existing authentication and reconciliation requirements.

- `make smoke`: isolated temporary API + MCP processes, autonomous session with client disconnection,
  reconnect/report, SSE, and audit-chain verification. No account credentials, external API calls or paid inference.
- `make status`: service status. Logs are under `var/logs`.
- `make stop`: stops local processes. Broker-held orders remain at the broker.
- `make reconcile`: reconcile the ledger before proceeding after unexpected shutdowns.
- `make test-py`: excludes `live_models`, `paper_broker`, and `gui` even if credentials happen to be installed.

Custom ports: `LAT_API_PORT=18766 LAT_DASHBOARD_PORT=5180 make demo`. The supervisor passes the selected API port
to the dashboard proxy. Do not run multiple API processes against one ledger. Use a separate `LAT_PROJECT_ROOT`
for isolated tests (the smoke command does this automatically).

[Sessions](paper-sessions.md) · [MCP](ai-runtime-mcp.md) · [Broker onboarding](broker-paper-onboarding.md)
