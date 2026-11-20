# Claude Opus 5.5 — IMPLEMENTATION DIRECTIVE
## Local Mac Quantitative + AI Trading Workstation
**Architecture frozen by the project owner as of November 5, 2026.**

## Your job
You are the implementation lead and principal engineer. **Do not perform a new competitive technology search or substitute your own preferred architecture.** The architecture, model IDs, broker, tools, and implementation order are decided below. Read cited official documentation for *implementation details* and library APIs only. When something contradicts these constraints, preserve the selected architecture, explain the precise conflict, and implement all independent work.

Build real software on this MacBook, run it, test it, and fix it. Do not end with plans, scaffolds, diagrams, or placeholders. Use maximum available reasoning effort for deep engineering tasks and parallel subagents only when they have clearly bounded deliverables. Do not pretend the project can finish in the background or that external API integration has been tested without credentials.

### Immutable product boundaries
- Local-first **MacBook project**: source, research, storage, dashboard, risk engine, order engine and monitoring all run locally. Docker containers may run locally, but **no AWS/GCP/VPS/Vercel/Kubernetes/hosted application backend**.
- External managed **model, market data, and brokerage APIs are permitted**. No direct external service purchase, account creation, transfer, or funding without user approval.
- Initial markets: long-only U.S. stocks and nonleveraged ETFs. Exclude options, futures, crypto, shorting, leveraged/inverse ETFs, OTC/penny stocks, extended hours, and margin borrowing.
- **PAPER** mode is default. Live API client must not be able to place orders unless a separate human-controlled release gate is passed. No live orders from this coding session.
- The system **may** eventually generate decisions and route real orders through the official broker API. Screen clicking is **never** the live order route.
- The local trading service stops when MacBook sleeps/shuts down. Broker-held orders may remain active; on restart reconciliation is mandatory.

## Fixed technology decisions (do not revisit)

| Layer | Selection | Notes |
|---|---|---|
| Deep analysis model | **Anthropic Claude Opus 5.5** | API ID `claude-opus-5-5`; use `high` for normal detailed research, `xhigh` or `max` for carefully scoped escalation and evaluation |
| Orchestration model | **OpenAI GPT-6.1 Sol** | API ID `gpt-6.1-sol`; **Responses API** for tool calling; default `medium` effort, `high` for difficult workflows |
| Model framework | **PydanticAI** | Strict, versioned Pydantic schemas; adapters for both providers |
| Trading engine | **QuantConnect LEAN (Apache 2.0), self-hosted from source** | Do **not** require paid QuantConnect LEAN CLI; isolate C# engine; use same LEAN strategy code in backtest and paper/live paths |
| Broker | **Alpaca Trading API** | Paper first, live adapter locked; use official LEAN Alpaca brokerage plugin for engine order execution |
| Supporting Alpaca client | `alpaca-py` | Use for standalone data/account/connection tests only; **never create a second uncoordinated order-submission path** |
| Historical/ML research | **Microsoft Qlib v0.9.8**, LightGBM, CatBoost, scikit-learn | Install in compatible isolated Python environment; data snapshots are point-in-time |
| Financial agent reference | **TradingAgents v0.7.1** | Use only as research/reference or bounded read-only agent integration; it must not own broker authorization |
| Native macOS control | **Peekaboo v4.9.7** | App/window/accessibility inspection, screenshots, clicks, typing for allowlisted non-execution apps |
| Browser control | **Microsoft Playwright MCP** | Structured accessibility targeting preferred over pixel clicking |
| API backend | **Python 3.11 + FastAPI + asyncio** | Python 3.11 aligns with published LEAN Python support and Qlib supported classifiers |
| Execution engine language | **C# (LEAN)** | Build and pin repository commit; obtain .NET SDK matching pinned source requirements |
| Local structured store | **SQLite in WAL mode** | Single-writer discipline, migrations, strict transactions; auditable order + approval state |
| Analytical store | **DuckDB + Parquet** | Historical OHLCV/quotes/features/model datasets, with provenance manifests |
| Dashboard | **React 19 + TypeScript + Vite + TradingView Lightweight Charts** | Localhost only; include required TradingView attribution |
| Package/process tools | `uv`, `pnpm`, `make`, local process supervisor, optional local Docker | No cloud services or Redis/NATS by default |
| Testing | `pytest`, `hypothesis`, `ruff`, `pyright`, Vitest, Playwright, .NET tests | Mandatory simulation and failure injection |
| Telemetry | Structured JSON logs, OpenTelemetry where useful, SQLite audit events | No third-party hosted observability requirement |

Current source release baselines: Qlib `v0.9.8` (2026-10-08), TradingAgents `v0.7.1` (2026-10-27), Peekaboo `v4.9.7` (2026-10-23). LEAN and its Alpaca plugin should be pinned to a documented known-good commit pair after cloning, without an open-ended comparative search. Do not choose NautilusTrader v2 release candidates as the initial trading engine. Do not use the proprietary/paid LEAN CLI as a hard dependency. Build/launch the open-source engine and its broker plugin directly. If blocked, document the build-specific obstacle while completing model, market-data, risk and deterministic simulation components; **do not silently substitute an incompatible engine**.

### Important cost and access facts
- Claude Pro and ChatGPT Plus subscriptions are for interactive coding; arbitrary production inference uses separate API billing. Never scrape or repurpose subscription cookies/tokens.
- Configure API keys via macOS Keychain or a locally ignored `.env` for development, with `PAPER`-only credentials by default. No production broker credentials in LLM prompts or desktop-control processes.
- Alpaca Basic market data is free but real-time equities are **IEX-only**. Implement separate `IEX_FREE` versus `SIP_PAID` provider modes and display them clearly. Algo Trader Plus currently costs $99/month and provides consolidated exchange coverage. Do not purchase automatically.
- Trading with less than $25,000 is not governed by the old PDT count at Alpaca, but actual account/buying-power/margin restrictions must be checked from the broker and FINRA's current intraday-margin framework. Do **not** build legacy PDT count rejection from outdated documentation. Do not use borrowed funds.
- Paper fills are simulated; Alpaca warns they omit market impact, latency-related slippage, queue position, fees/dividends and other live phenomena.

## Target system architecture
```
                          LOCAL MACBOOK
 ┌─────────────────────────────────────────────────────────┐
 │  Alpaca market-data WebSocket / historical API          │
 │           ↓ timestamp validation + durable ingestion    │
 │  Parquet / DuckDB + SQLite audit ledger                 │
 │           ↓                                             │
 │  Qlib / LightGBM quantitative signals + cost model      │
 │           ↓ selected candidate only                     │
 │  Claude Opus research + GPT-6.1 Sol orchestration       │
 │           ↓ typed TradeProposal / NO_TRADE              │
 │  Independent deterministic Risk & Policy Gate           │
 │           ↓ signed/authorized order intent              │
 │  Self-hosted LEAN engine → Alpaca brokerage plugin       │
 │           ↓ order lifecycle + streaming/polling         │
 │  Persistent reconciliation and recovery service         │
 │           ↓                                             │
 │  FastAPI localhost → React dashboard + audit logs        │
 │                                                         │
 │  Separate Peekaboo / Playwright screen-control agent    │
 │  (research apps and dashboard, NO live broker clicks)   │
 └─────────────────────────────────────────────────────────┘
          External calls only: AI APIs / data / broker
```

### Architectural invariants
1. **Single broker order gateway.** No route to Alpaca order creation except through the LEAN brokerage integration behind risk/authorization. An independent test client may send to the *paper endpoint only*.
2. **No model-to-broker credentials.** Models get sanitized read-only context and `propose_trade`; never `submit_live_order`.
3. **Risk gate cannot be bypassed** by LEAN strategies, UI tools, agent output, restart/replay, or alternate administrative endpoints. Enforce gateway ownership and policy validation at the final order boundary.
4. **Fail closed.** If price freshness, account state, broker status, strategy state, permissions, or synchronization is unknown, new entries are blocked.
5. **Local persistent evidence.** Every action is associated with a source, timestamp, dataset version, model version, policy version, and causality ID.
6. **Paper and live isolation** at code, configuration, secret, database, and executable permission boundaries—not merely a UI toggle.
7. **No reliance on broker app UI automation** for order placement, authentication bypass, or trading actions.

## Implemented modules and their contracts

### 1. System preflight
Detect macOS version, Intel vs Apple Silicon, RAM, available disk, Docker, dotnet, Python, Node, Homebrew, Claude Code, and requested permissions. Do not ask for hardware details that can be introspected. Fail with actionable preflight messages when minimum requirements fail (Peekaboo requires macOS 15+). All scripts must work from a dedicated project directory.

### 2. Data ingestion
Use official Alpaca market-data endpoints for minute bars, trade and quote streams where entitlements permit. Record `exchange_ts`, `provider_ts`, `received_ts`, `available_at_ts`, source ID, feed (`IEX`/`SIP`), adjustment policy, and session. Preserve raw events in partitioned Parquet and typed normalized bar/quote tables. Reject out-of-order/missing/anomalous data according to configurable policies; don't silently fill gaps. Include corporate action/calendar ingestion, symbol eligibility and research universe snapshots. On reconnection, backfill gaps before resuming new entries. Historical experiments must not use information published after the simulated decision time.

### 3. Quant research and baseline
Implement no-LLM benchmarks first: buy-and-hold, simple momentum, intraday mean reversion, and a liquidity/volatility-filtered baseline. Do not claim any edge. Use Qlib v0.9.8 research pipelines for supervised feature experiments (LightGBM/CatBoost). Predict conditional return over a specified horizon or rank candidates; calibrate probabilities and estimate spread, fees, and slippage. Make walk-forward chronological splits, purging/embargo where labels overlap, point-in-time news and corporate actions, multiple-testing controls, and frozen final test intervals. Keep a simple deterministic baseline alongside every AI pipeline.

### 4. AI research
Use two strict adapters:
- Anthropic Messages API, `model="claude-opus-5-5"`; adaptive thinking and explicit effort settings. It reviews relevant filings/news and candidate signals, lists evidence and counterarguments and may return NO_TRADE.
- OpenAI Responses API, `model="gpt-6.1-sol"`, `reasoning={"effort":"medium"}` (adjustable via evaluated configuration). It retrieves, coordinates, structures and validates the research workflow. Tool calls must use Responses, not a legacy Chat Completions tool-calling assumption.

Use PydanticAI provider adapters and structured Pydantic contracts. Version prompts and adapters. Model agents have **no** order API access. No model may change limits or model its own confidence as calibrated trading probability without evaluation. If any required evidence is stale/unavailable, return `NO_TRADE`. Do not invoke an expensive LLM on every market tick. Model outputs must include source IDs and exact timestamps. Detect prompt injection in retrieved web/news content. Store exact token usage, latency, and estimated cost.

### 5. Decision contract
Define and validate `MarketSnapshot`, `Evidence`, `QuantSignal`, `ResearchFinding`, `TradeProposal`, `RiskDecision`, `ApprovedOrderIntent`, `BrokerOrder`, `OrderEvent`, `PositionSnapshot`, `AccountSnapshot`, and `ModelRunRecord`. Use `Decimal` for money, quantities and prices (no floats in accounting). Include unique causal IDs, `data_asof`, side, symbol, strategy, quantities, limit price, market-session constraints, evidence IDs, max planned loss, policy version, and explicit `NO_TRADE`. The risk engine independently recomputes all exposure/quantity data and never trusts LLM-supplied arithmetic.

### 6. Independent risk engine
Deterministic Python service and final integration gate with unit tests. Initial paper defaults: long-only, liquid U.S. stocks/standard ETFs, no leverage, no margin debit, one position at a time, regular session only, capped order notional, maximum daily realized+unrealized loss, max spread, max quote age, max pending orders, max order rate, max concentration, account equity floor, system health gates, single portfolio source of truth, and duplicate-order rejection. Risk policy is stored as a versioned immutable approved configuration. The AI has read-only access to it and cannot edit it. Users can edit limits only through a separate authenticated local settings flow with auditable changes. Use broker account equity/buying power checks in addition to local controls. Protective stops are not guaranteed execution prices.

### 7. LEAN + Alpaca execution
Build the open-source LEAN engine and the maintained `QuantConnect/Lean.Brokerages.Alpaca` plugin locally, pin exact dependency revisions, configure isolated paper/live environments, and wire approved intents into LEAN as the only order submission path. LEAN algorithms must not be able to bypass the independent policy gateway. Inspect the source and use documented engine extension hooks for control instead of fake wrappers. Implement or integrate a durable order ledger with states `PROPOSED`, `RISK_APPROVED`, `HUMAN_APPROVED`, `SUBMITTING`, `SUBMITTED`, `ACKNOWLEDGED`, `PARTIALLY_FILLED`, `FILLED`, `CANCEL_PENDING`, `CANCELED`, `REJECTED`, `EXPIRED`, `UNKNOWN`, `RECONCILED`. Keep broker IDs and client order IDs. Consume streaming trade updates and reconcile with account/orders/positions APIs. When a submission times out, assume `UNKNOWN`, query by client order ID, and **never blindly retry**. Handle out-of-order events, partial fills, cancel/fill races, rejections, and restart. On unexpected positions or broker data mismatch, block all new entries. For an initially unsupported broker feature, reject rather than degrade silently.

### 8. Operating modes
`RESEARCH_ONLY`, `REPLAY`, `PAPER`, `LIVE_APPROVAL`, `BOUNDED_LIVE`.
- Default `PAPER`, using paper keys only. All sandbox and deterministic replay tests must pass without any credentials.
- `LIVE_APPROVAL` requires separately provided live credentials, a user-enabled release lock, verified broker account identity, a fresh signed human confirmation **per order**, and independent risk approval.
- `BOUNDED_LIVE` must be compiled/configured **off** and be inaccessible through runtime prompting. Implement only a non-executable documented interface and tests for activation denial. Do not activate or place real-money orders in this work session.
- Mode transitions and credential assignment require independent audit and fail-closed design. Changing an environment variable alone must not make paper credentials submit live orders.

### 9. macOS computer use
Integrate Peekaboo 4.9.7 for native Finder, Notes, Calculator, custom dashboard, terminal/IDE windows, and Playwright MCP for authorized browsers. Install with `brew install openclaw/tap/peekaboo` when permitted; validate macOS permissions (`peekaboo permissions status`). Connect Playwright MCP using `claude mcp add playwright npx @playwright/mcp@latest`, under the development account, if allowed. **Do not grant the GUI agent broker order access, production secrets, arbitrary local files or unrestricted shell access**. Use an explicit application allowlist. Verify on Calculator/Notes and a fake paper dashboard: inspect → click → type → confirm persisted result. Deny attempts to click a live brokerage order button.

### 10. Dashboard and operations
Local-only FastAPI endpoint bound to `127.0.0.1` and a React 19/Vite/TypeScript dashboard. Use TradingView Lightweight Charts with attribution. Views: market feed quality/IEX-vs-SIP, scanner/candidates, model evidence and cost, portfolio/P&L, LEAN engine status, proposed vs approved orders, detailed fill lifecycle, risk utilization, replay/backtest performance, broker reconciliation, startup/shutdown state, failures, and big stop-new-orders control. Clearly distinguish simulated/paper/live status. Stream updates via SSE/WebSocket to localhost. No fake live prices. Include transparent data timestamps and feed limitations.

### 11. Persistence and process management
Use SQLite WAL and a single durable writer for operational order/account/audit events; DuckDB + Parquet for analytical data. Keep explicit migrations and backups. Use a local macOS process supervisor or `make` scripts; no hosted queue/service. Only if measured necessary, add an embedded queue with durable outbox semantics. `make setup`, `make preflight`, `make dev`, `make test`, `make lint`, `make typecheck`, `make backtest`, `make replay`, `make paper`, `make status`, `make reconcile`, `make stop`, `make audit` must actually work. The Mac sleeps → local engine does not continue. On resume/restart all new entries remain locked until broker-authoritative reconciliation succeeds; don't invent fills in the gap.

### 12. Quality/security
- Secrets only in environment/Keychain, never repo, client bundle, screenshot logs, or prompts.
- Sanitize retrieved documents and UI content against prompt injection.
- Pin and hash project dependencies with `uv.lock`, `pnpm-lock.yaml`, LEAN commits and .NET packages; record software bill of materials/licenses.
- Test missing/malformed quotes, empty account state, duplicate events, network partitions, rate limits, cross-day settlement, extended session, broker disconnect, LLM timeout, retry replay, Mac sleep, and crash after broker accepts an order.
- Use deterministic replay/fake broker clients and property-based tests where possible; any API test must state whether it ran against a real paper broker.
- Compare backtest assumptions to Alpaca paper fill semantics; do not assume paper P&L predicts live fills.

## Repository layout (selected)
```
local-ai-trader/
  apps/dashboard/                    # React/Vite UI
  apps/mac-agent/                    # Peekaboo / Playwright scopes
  services/api/                      # FastAPI HTTP + streams
  services/market_data/              # Alpaca ingestion and backfill
  services/research/                 # Qlib, features, experiments
  services/ai/                       # Claude + OpenAI/PydanticAI
  services/risk/                     # deterministic authorization
  services/order_ledger/             # reconciliation, audit ledger
  trading/lean/                      # pinned LEAN source integration
  trading/alpaca_plugin/             # pinned brokerage plugin
  packages/contracts/                # Pydantic + generated TS types
  packages/core/                     # policy IDs, timestamps, math
  datasets/raw/                      # Parquet, gitignored
  datasets/processed/                # adjusted, versioned manifests
  experiments/                       # point-in-time test results
  tests/unit/
  tests/integration/
  tests/failure_injection/
  tests/e2e/
  docs/adr/
  docs/architecture/
  docs/runbooks/
  docs/sources/                      # source manifest supplied below
  scripts/
  Makefile
  README.md
  .env.example
```

## Prescribed work sequence — execute, don't re-research

**Stage 0: Preflight and source manifest.** Inspect this Mac and working directory, create `docs/adr/` for the above fixed selections, and record exact upstream source pins. Do not conduct alternative-stack research. If the Mac lacks permissions or dependencies, explain the missing hardware/software capability without switching to cloud.

**Stage 1: Repo and deterministic local test.** Install project runtimes, create config, migrations, health endpoints, typed contracts, SQLite audit log, DuckDB tables, provider interfaces, fake broker, test harness and local start/stop targets. Run `make test`.

**Stage 2: Data and backtest.** Ingest limited Alpaca paper/free IEX bars for selected symbols when keys exist; otherwise load a bundled, explicitly synthetic fixture. Implement bars/quotes and point-in-time validation. Build LEAN source and Alpaca plugin; provide a working LEAN historical backtest with realistic fee/spread approximations and one non-ML baseline. Show results from real execution, not mock printouts.

**Stage 3: AI agents.** Implement provider adapters (with explicit fake transport when keys missing), PydanticAI research, source evidence snapshots, conditional candidate selection, validated `TradeProposal`, NO_TRADE handling, traces and costs. Run repeatability/evidence tests. Do not tell the user to handwire Python files; create them.

**Stage 4: Risk plus paper execution.** Complete the independent policy gate and audited LEAN Alpaca paper path, with safe order submit/cancel/status stream, unknown state recovery, position/account reconciliation. Exercise both test fake broker and genuine Alpaca paper path if keys are available; precisely distinguish the two in all reports.

**Stage 5: Dashboard plus Mac GUI.** Build actual localhost dashboard, streaming status updates, and allowlisted Peekaboo/Playwright macOS workflows. Verify GUI actions on benign apps and custom paper UI.

**Stage 6: Fault injection and recovery.** Run all adverse scenarios; fix defects; test restart and local-sleep/reconnection behavior; confirm default live execution remains impossible.

**Stage 7: Documentation and handoff.** Document exact installation and commands; include architecture Mermaid diagrams, security boundaries, API contract, unit/e2e test results, broker paper onboarding, environment variable schema, data-quality limitations, known issues and live-readiness checklist. Label every component `TESTED`, `IMPLEMENTED_NOT_INTEGRATION_TESTED`, `SIMULATED`, `BLOCKED`, or `PLANNED`. Never claim a live integration was tested without credentials and an actual verified paper outcome.

### Acceptance criteria
All of the following must be **demonstrably true** before claiming an initial complete working system:
1. `make setup`, `make test`, `make dev` work on the current Mac.
2. Real historical bar data (or transparently labeled synthetic fixture) is loaded with source manifest; backtest runs.
3. Feature/quant baseline returns a scored candidate or NO_TRADE, with realistic cost model.
4. Claude and Sol adapter integration exists, with validated schema and independent fake-transport tests.
5. A risky or invalid proposed order is stopped by deterministic risk policy.
6. LEAN engine and Alpaca plugin are genuinely built and run locally, or blocked integration is candidly reported; do not fake completion.
7. An approved **paper** order flows through LEAN → Alpaca paper broker when user has supplied paper keys; otherwise a fully exercised deterministic fake broker path exists.
8. Orders and positions reconcile; unexpected state stops new orders, and unknown outcomes never automatically resubmit.
9. Dashboard uses actual local operational state and labels data feeds/simulation accurately.
10. An allowlisted native Mac app can be inspected and manipulated with Peekaboo; a browser can be controlled through Playwright MCP.
11. The app recovers safely from restarts, failures, missing credentials, and connectivity drops.
12. No live order can be placed by default, via GUI, CLI, model output, or a mistaken environment switch.

### Working rules
Do not ask me to choose an engine, broker, database, model, agent library, UI stack, or screen-control toolkit: **all are decided**. Do not recreate the prior comparative research. You may read the official implementation docs listed below, inspect API source, and solve compatibility issues as engineering tasks. Only ask me for consent when you need money spent, external account setup, a permission affecting sensitive applications, or access to credentials. Never ask me to paste private keys into chat. Do not claim async background work. Proceed with real code, commands, tests, and honest reporting until the active session or tools run out.

## Verified official implementation resources (November 5, 2026)
- Anthropic Opus 5.5: https://platform.claude.com/docs/en/models/opus-5-5/overview
- OpenAI Sol: https://developers.openai.com/api/docs/models/gpt-6.1-sol
- OpenAI Responses reasoning: https://developers.openai.com/api/docs/guides/reasoning
- LEAN source: https://github.com/QuantConnect/Lean
- LEAN source engine reference: https://www.quantconnect.com/docs/v2/lean-engine/getting-started
- LEAN config with `live-alpaca`: https://github.com/QuantConnect/Lean/blob/master/Launcher/config.json
- LEAN Alpaca brokerage plugin: https://github.com/QuantConnect/Lean.Brokerages.Alpaca
- LEAN CLI paid tier caveat: https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-live-deploy
- Alpaca Trading API: https://docs.alpaca.markets/us/docs/trading-api
- Alpaca orders / client IDs: https://docs.alpaca.markets/us/docs/working-with-orders
- Alpaca paper environment: https://docs.alpaca.markets/us/docs/paper-trading
- Alpaca intraday margin: https://docs.alpaca.markets/us/docs/the-intraday-margin-rule
- Alpaca data coverage: https://docs.alpaca.markets/us/docs/about-market-data-api
- alpaca-py SDK: https://github.com/alpacahq/alpaca-py
- Microsoft Qlib v0.9.8: https://github.com/microsoft/qlib/releases
- TradingAgents v0.7.1: https://github.com/TauricResearch/TradingAgents/releases
- PydanticAI: https://github.com/pydantic/pydantic-ai
- Peekaboo v4.9.7: https://github.com/openclaw/Peekaboo/releases
- Playwright MCP: https://github.com/microsoft/playwright-mcp
- TradingView Lightweight Charts: https://github.com/tradingview/lightweight-charts

**START NOW:** Inspect the local repo/host and implement Stage 0 through the first truly working local testable vertical slice. Do not stop after documentation. Do not invent progress. Do not execute real trades or purchase external services.
