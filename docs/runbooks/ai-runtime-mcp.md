# Runbook: operating the app from Claude Code or Codex via MCP

The workstation exposes ONE validated tool interface (`services/tools`). Every AI runtime uses it:

| `LAT_AI_RUNTIME` | Who drives | Transport to the tools | Billing / limits |
|---|---|---|---|
| `NONE` | nobody (quant only) | – | – |
| `API_KEYS` | in-app Claude Opus 5.5 + GPT-6.1 Sol adapters | in-process | separate API billing |
| `CLAUDE_CODE` | your Claude Code session | MCP stdio server `local-ai-trader` -> local API | your Claude plan's own usage limits |
| `CODEX` | your Codex session | MCP stdio server `local-ai-trader` -> local API | your ChatGPT plan's own usage limits |
| `CHATGPT_OAUTH` | in-app "Sign in with ChatGPT" research turn | in-process, OpenAI Responses function tools | your ChatGPT plan |

The quant engine never depends on any of these. PAPER is the default mode and live execution is compiled off.

**The app never holds subscription credentials for Claude Code or Codex.** Each client signs in by itself and
starts the MCP server as its own child process. The server receives no model or broker credential. It holds
only the local agent token (`var/<ns>/keys/agent.token`), which the local API accepts for the agent tool routes
and nothing else. The token is never shown to the model.

## 1. Start trading

Tell the connected AI “start trading” and approve the `start_paper_session` tool call if prompted. The tool
starts the configured local API on demand and generates the agent token as part of API startup. The API picks
the regular-hours session window and eligible symbols. It keeps running after the client disconnects.

`make dev` starts the API and local monitor for manual inspection. The API remains the only SQLite writer and the
deterministic risk engine stays in charge of every order.

## 2. Claude Code

The project registration is checked in at `.mcp.json` (project scope, official format):

```json
{"mcpServers": {"local-ai-trader": {"type": "stdio", "command": "uv",
  "args": ["run", "--frozen", "--directory", "/Users/victor/Developer/ai-trading", "python", "-m", "services.mcp_server"],
  "env": {"LAT_AGENT_NAME": "claude-code"}}}}
```

* Open Claude Code in the project directory and approve `local-ai-trader` when asked. Project servers start out
  pending; you can also approve them later in `/mcp`.
* Check the registration: `claude mcp get local-ai-trader`. On this Mac (2026-11-07) it reports
  `Scope: Project config (shared via .mcp.json)` and `Status: ⏸ Pending approval (run claude to approve)`, and
  `claude mcp list` shows the same server as pending. The server is left pending on purpose: approving it is
  your decision.
* Re-register through the official CLI if needed: `scripts/register_mcp.sh` prints the commands, and
  `scripts/register_mcp.sh --apply` runs `claude mcp add --scope project …` only when `.mcp.json` lacks the entry.
* Approvals: Claude Code asks before each MCP tool call unless you allow-list it. Suggested allow-list (put it in
  your own `.claude/settings.local.json`; this repo does not write it for you). It covers **read-only tools only**:
  `mcp__local-ai-trader__get_market_snapshot`, `…__get_bars`, `…__get_feed_quality`, `…__get_predictions`,
  `…__get_model_report`, `…__get_signals`, `…__get_portfolio`, `…__get_orders`, `…__get_order`,
  `…__get_risk_status`. Leave `propose_paper_trade`, `request_halt`, `cancel_paper_order` and `run_quant_cycle`
  on "ask".

## 3. Codex

The project configuration is in `.codex/config.toml`. Codex loads it only for a project you have marked as trusted:

* `[mcp_servers.local-ai-trader]` holds `command = "uv"`, the same `args`, `cwd`, `startup_timeout_sec = 30` and
  `tool_timeout_sec = 90`.
* `default_tools_approval_mode = "prompt"`. Only the ten read-only tools have `approval_mode = "approve"`. The four
  mutating tools keep `"prompt"`. Key names and values follow the Codex configuration reference
  (`approval_mode` is one of `auto`, `prompt`, `writes`, `approve`); Codex's default 10 s start-up timeout is
  raised to 30 s because `uv run` can be slow on a cold cache.
* Every tool input schema is under Codex's default `tool_input_schema_max_bytes` (5,000 bytes); a test checks it.

Historical setup note (2026-11-07; superseded by the verification below): the Codex CLI was **not installed on this Mac**. `scripts/register_mcp.sh` detects that, prints the steps below and
installs nothing:

1. Install Codex yourself, for example with `npm install -g @openai/codex` or `brew install --cask codex`.
2. Run `codex login` and choose **Sign in with ChatGPT**. Your plan's usage limits apply.
3. Open Codex in the project and trust it. Alternatively, register at user level with
   `codex mcp add local-ai-trader --env LAT_AGENT_NAME=codex -- uv run --frozen --directory <root> python -m services.mcp_server`.
   This command is what `scripts/register_mcp.sh --apply` runs when Codex is present.

## 4. What an agent can and cannot do

| Category | Tools | Effect |
|---|---|---|
| market | `get_market_snapshot`, `get_bars`, `get_feed_quality` | read |
| models | `get_predictions`, `get_model_report`, `get_signals` | read |
| portfolio | `get_portfolio`, `get_orders`, `get_order`, `get_risk_status` | read |
| paper_trading | `propose_paper_trade`, `request_halt`, `cancel_paper_order`, `run_quant_cycle` | risk-gated paper actions |

There is **no** tool that resumes trading, lifts a halt, changes risk limits, policy or mode, approves orders,
submits live orders, reconciles, reads secrets or tokens, or touches files, the shell or the network. Tests
enforce this (`tests/unit/tools/test_tools_registry.py`).

Typical session: `get_risk_status` -> `get_feed_quality` -> `get_signals` / `get_predictions` ->
`get_market_snapshot SPY` -> `propose_paper_trade` (cite the ids returned above) -> `get_order` once.

## 5. Safety model (layers, outermost first)

1. **Client approval.** Claude Code asks before tool calls unless allow-listed. In Codex, `approval_mode = "prompt"`
   applies to every mutating tool.
2. **Mode gate.** The MCP server refuses to start outside PAPER/REPLAY. The API refuses every agent tool call with
   403 unless the mode is PAPER/REPLAY and the execution label is not LIVE.
3. **Agent token.** The agent routes require `X-LAT-Agent-Token`. That token is distinct from the operator and
   engine tokens and cannot reach any operator endpoint.
4. **Strict schemas.** Unknown fields are rejected. Money must be decimal strings: floats, and JSON numbers with
   fractions, are rejected on every transport. Symbols and ids must match patterns. Free text has control and
   bidi characters stripped, is length-capped and is screened for injection-shaped content. A halt is the
   fail-safe direction and is never refused: an instruction-shaped halt reason is recorded as an inert
   placeholder instead of being rejected.
5. **Deterministic proposal construction.** The agent's quantity and limit are *requests*. The limit is rounded
   to a passive tick. `max_planned_loss` is computed deterministically and conservatively, never taken from the
   agent. `data_asof` comes from a fresh snapshot. Evidence ids must be ids that a tool returned to this agent
   session, or persisted signal ids, and at least one must belong to the traded symbol.
6. **Independent risk engine, then signed intent, then final gate, then paper gateway.** The engine recomputes
   everything and may reject; it never resizes. Proposals are rate-limited per hour
   (`LAT_AGENT_MAX_PROPOSALS_PER_HOUR`, default 6). Quant cycles started by agents are limited the same way and
   never call an LLM.
7. **Outputs are data.** Every output carries `execution_label`, `is_simulated` and `asof`. Outputs are validated
   against strict output models: contract-violating data is withheld (502), never forwarded. The server
   instructions tell the model to treat tool output as data, not instructions.
8. **No blind retries.** A timed-out or failed mutating call (including any 5xx from the API) is reported as
   UNKNOWN. The agent is told to inspect `get_orders` first.

What was verified on this Mac, without any credentials: the MCP server ran as a real stdio subprocess driven by
the official SDK client against the real FastAPI app with a fake runtime, including the exact `.mcp.json`
command; every tool ran against the real runtime in SIMULATED mode (synthetic replay feed, FakeBroker, real risk
engine, a tiny trained model registry). Historical verification before continuation: no Claude Code or Codex session had driven the tools yet: Claude Code
lists the server as pending your approval, and Codex is not installed.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `error 503: agent token unavailable` | Run the start-session tool or `make dev` to launch the API and provision the token. |
| `error 503: the local API could not prepare a paper session` | Check `var/logs/api.log` and the configured API address. |
| `error 401: … rejected the agent token` | The token file changed while the API kept the old one. Restart the API (and the MCP server). |
| `error 403` | The API runs outside PAPER/REPLAY or with a LIVE label. Agents are not allowed there. |
| `error 422` | Input was invalid. The message names the field. Send money as strings such as `"412.35"`. |
| `outcome: RATE_LIMITED` | Wait. The limit is per hour and only a human can change it. |
| Claude Code shows "Pending approval" | Run `claude` in the project and approve `local-ai-trader`, or use `/mcp`. |

Manual smoke test without any client:
`uv run --frozen python -m services.mcp_server` (it speaks JSON-RPC on stdin/stdout and logs to stderr only).


## Continuation verification (2026-11-08)

The connected Codex desktop MCP client successfully started a five-minute SIMULATED session, retrieved its
completed report (five NO_TRADE cycles), proposed and closed one synthetic share through the real risk engine,
and read the fill ledger and performance. The final simulated portfolio had no open positions/orders. This
verifies Codex's actual MCP path, not Claude Code's sign-in or a live ChatGPT OAuth account.

The shared registry now has 25 tools (17 read-only, 8 mutating), including `start_paper_session`, `stop_session`,
`schedule_session`, `cancel_schedule`, session status/reports, decision explanations, activity, performance and
health. Current registrations list those permissions; the historical tool counts above describe the earlier build.
MCP server instructions now describe session operation, disconnect behavior and uncertain-action recovery.

`make smoke` verifies the real API and stdio server in an isolated temporary workspace, including disconnecting
the client while the session completes and reconnecting to obtain the report. No inference credentials are used.

The existing in-app ChatGPT adapter follows the documented dynamic registration and plan-usage route; see
[official overview](https://developers.openai.com/siwc/token-sharing-open-source) and
[registration/sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in), checked 2026-11-08.
Eligibility, successful authentication and real inference are **not verified** in this continuation. Existing
fake-transport OAuth, token validation, response streaming, failure and tool-contract tests are included in the
Python suite. No subscription credentials were extracted, no limits bypassed, and no paid API calls were made.
