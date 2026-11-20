# ADR 0002 — Local quant model suite and subscription-based AI runtime modes

Status: **Accepted** (owner update request 2026-11-07; extends ADR 0001 without replacing the frozen stack).

## Context
The owner asked to (1) add a local prediction-model suite selected by net risk-adjusted performance, (2) let the
application be operated by AI through the owner's existing subscriptions, and (3) keep the quant engine independent of
AI, keep deterministic risk limits authoritative, default to paper, and require explicit authorization for live orders.

ADR 0001 / spec §"Important cost and access facts" says subscriptions must never be scraped or repurposed. Every mode
below uses only **officially documented** mechanisms (verified against developers.openai.com and Claude Code docs on
2026-11-07); no cookies, session tokens or unofficial endpoints are used.

## Decision
### Quant model suite (`services/research/models`)
LightGBM regressors (5/15/60-minute net returns), CatBoost success classifier with isotonic/logistic calibration,
LightGBM lambdarank opportunity ranker, Ridge statistical baseline, HAR-RV + EWMA volatility (blend chosen out of sample),
2-state Markov-switching regime model using **filtered** probabilities only, and a spread baseline + LightGBM residual
execution-cost model. Leakage-free walk-forward (purge + embargo, frozen final test used once), costs from the existing
cost model, and selection by **net Sharpe** with multiple-testing control; if nothing beats the zero baseline the
selection is the baseline and the signal engine emits NO_TRADE. The quant pipeline runs with `LAT_AI_RUNTIME=NONE`.

### One validated tool interface (`services/tools`)
A single registry of strictly typed tools (market analysis, model predictions, portfolio monitoring, paper-trading
operations) exported identically as MCP tools, OpenAI Responses function tools and PydanticAI tools. Mutating tools are
limited to `propose_paper_trade` (→ `RiskEngine.evaluate` → signed intent → final gate → paper gateway, rate-limited),
`request_halt`, `cancel_paper_order` and `run_quant_cycle`. No tool can resume trading, change limits or mode, submit live
orders, or read secrets. The API process remains the single SQLite writer: the MCP server calls the local API with an
agent token that models never see.

### Runtime modes (`LAT_AI_RUNTIME`)
| Mode | Who runs the model | Billing | Mechanism |
|---|---|---|---|
| `NONE` | nobody | — | quant engine only |
| `API_KEYS` | the app (Claude Opus 5.5 + GPT-6.1 Sol) | separate API billing | Anthropic Messages API, OpenAI Responses API via PydanticAI |
| `CLAUDE_CODE` | the owner's Claude Code session | the owner's Claude plan | local stdio MCP server registered in `.mcp.json` |
| `CODEX` | the owner's Codex session | the owner's ChatGPT plan | local stdio MCP server in `.codex/config.toml` / `codex mcp add` |
| `CHATGPT_OAUTH` | the app, on the owner's ChatGPT plan | ChatGPT plan usage (Plus/Pro, limits apply) | official "Sign in with ChatGPT" open-source/local flow: OAuth 2.0 + PKCE, loopback `http://127.0.0.1:1455/auth/callback`, scope `chatgpt.tokens.use.direct`, Responses API with `store:false`, `stream:true`, function tools only |

In `CLAUDE_CODE`/`CODEX` modes the app never calls a model and never holds subscription credentials; the client's own
usage limits and approval prompts apply (mutating tools default to approval-required in Codex config).

## Consequences
* Usage limits: ChatGPT Plus shares a five-hour usage limit across apps; the app surfaces
  `subscription_sharing_usage_limit_exceeded` and links to ChatGPT usage settings, and never retries in a loop.
* OpenAI's documentation does not address financial or automated use of plan sharing; the owner must confirm eligibility
  and terms before enabling `CHATGPT_OAUTH`. The app invokes it only on explicit operator action or the gated cycle.
* Live trading remains compiled off (`LIVE_EXECUTION_COMPILED=False`); no AI mode can change that.
