# Local operations

Tell the connected AI “start trading” to start or queue one session and launch the local API if needed.

The [local monitor](http://127.0.0.1:8765/monitor) shows trading status, mode, account equity, daily and
total P&L, critical alerts, and emergency controls. `make dev` starts the API with your local settings.
`make demo` selects synthetic data, the fake broker, and no model calls.

**Halt new entries** blocks new entries and leaves protective exits enabled. **Halt entries and exits** also
blocks protective exits and requires the current workspace's operator token from `var/<mode>/keys/operator.token`.
Broker-held orders stay active when either control is used. Inspect analytics, diagnostics, and session reports
through MCP or the structured API and event stream.

Use the AI's MCP tools to start/stop sessions and create/cancel schedules. `make halt` stops new entries;
`stop_session` stops future decision cycles. Neither liquidates positions or cancels existing broker orders.
Only the operator CLI can clear a halt under the existing authentication and reconciliation requirements.

- `make smoke`: isolated temporary API + MCP processes, autonomous session with client disconnection,
  reconnect/report, SSE, and audit-chain verification. No account credentials, external API calls or paid inference.
- `make status`: service status. Logs are under `var/logs`.
- `make agents` / `make agents-once` / `make agents-status`: the research crew (Claude Code + Codex on your own
  sign-in; research only, no execution authority). See [research crew](../architecture/agent-crew.md).
- `make autopilot` / `make autopilot-once` / `make autopilot-status` / `make autopilot-plan`: the autonomous daily
  orchestrator (PAPER session at the open, research crew and quantlab after the close, daily report; needs
  `LAT_AUTOPILOT_ENABLED=true`). See [autopilot runbook](autopilot.md) and [design](../architecture/autopilot.md).
- `make stop`: stops local processes. Broker-held orders remain at the broker.
- `make reconcile`: reconcile the ledger before proceeding after unexpected shutdowns.
- `make validate`: the complete local validation (lint, types, contracts, tests, startup smoke, LEAN, dependency
  audit). There is no remote CI. See [validation](validation.md).
- `make test-py`: excludes `live_models`, `paper_broker`, and `gui` even if credentials happen to be installed.

Set `LAT_API_PORT` in `.env` to customize the API and monitor port. Use a separate `LAT_PROJECT_ROOT` for isolated tests. The smoke command
does this automatically.

[Sessions](paper-sessions.md) · [MCP](ai-runtime-mcp.md) · [Broker onboarding](broker-paper-onboarding.md)
