# Environment variable and secret schema

## Secrets (Keychain service `local-ai-trader`, account = name; or git-ignored `.env`)
| Name | Used by | Notes |
|---|---|---|
| `ALPACA_PAPER_KEY_ID` | market data, read-only reconciliation, LEAN paper launcher | must start with `PK` |
| `ALPACA_PAPER_SECRET_KEY` | same | never logged, never in prompts/GUI env |
| `ANTHROPIC_API_KEY` | `services/ai` Claude adapter | used only when `LAT_AI_TRANSPORT=real` |
| `OPENAI_API_KEY` | `services/ai` Sol adapter | used only when `LAT_AI_TRANSPORT=real` |
| `ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET_KEY` | nothing in this build | presence-only; `SecretStore.get` refuses them |
| *(ignored)* `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | — | ambiguous paper/live names are never read |

Generated local secrets (per namespace, `var/<ns>/keys/`, mode 0600, git-ignored):
`intent_signing.pem` (risk engine private key), `intent_signing.pub.pem` (gateway/LEAN verification),
`operator.token` (human operator: resume, policy edits, run-cycle), `engine.token` (local LEAN bridge only).

## Settings (`LAT_` prefix)
| Variable | Default | Values |
|---|---|---|
| `LAT_MODE` | `PAPER` | `RESEARCH_ONLY`, `REPLAY`, `PAPER`, `LIVE_APPROVAL`*, `BOUNDED_LIVE`* (*always denied in this build) |
| `LAT_GATEWAY` | `FAKE` | `FAKE` (SIMULATED), `LEAN_PAPER_SIM` (SIMULATED), `LEAN_ALPACA_PAPER` (PAPER) |
| `LAT_FEED_MODE` | auto | `IEX_FREE` (default with paper keys), `SIP_PAID` (only if you bought it), `SIMULATED` |
| `LAT_AI_TRANSPORT` | `fake` | `fake` (explicit fake HTTP transports), `real` (paid API calls) |
| `LAT_CLAUDE_MODEL` / `LAT_CLAUDE_EFFORT` | `claude-opus-5-5` / `high` | effort `high`, `xhigh`, `max` |
| `LAT_SOL_MODEL` / `LAT_SOL_EFFORT` | `gpt-6.1-sol` / `medium` | effort `low`, `medium`, `high` |
| `LAT_API_HOST` / `LAT_API_PORT` | `127.0.0.1` / `8765` | non-loopback hosts are rejected |
| `LAT_DASHBOARD_PORT` | `5173` | |
| `LAT_SYMBOLS` | `["SPY","QQQ","AAPL","MSFT"]` | JSON list |
| `LAT_RECONCILE_INTERVAL_S` | `30` | |
| `LAT_USE_KEYCHAIN` | `true` | |
| `LAT_LOG_LEVEL` | `INFO` | |

Namespaces: `RESEARCH_ONLY→var/research`, `REPLAY→var/replay`, `PAPER→var/paper`, live modes→`var/live`.
Switching `LAT_MODE` therefore switches ledger DB, keys and tokens — paper state is never reused for live.
