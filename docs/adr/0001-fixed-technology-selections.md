# ADR 0001 — Fixed technology selections (owner decision, 2026-11-05)

Status: **Accepted (frozen by the project owner)**. This ADR records decisions; it does not revisit them.

| Layer | Selection | Pin / version in this build |
|---|---|---|
| Deep analysis model | Anthropic Claude Opus 5.5 (`claude-opus-5-5`), Messages API, adaptive thinking, effort `high` (escalation `xhigh`/`max`) | `anthropic` SDK 2.6.0 |
| Orchestration model | OpenAI GPT-6.1 Sol (`gpt-6.1-sol`), **Responses API**, reasoning effort `medium` (`high` for hard workflows) | `openai` SDK 3.46.0 |
| Model framework | PydanticAI (strict versioned Pydantic schemas, adapters for both providers) | `pydantic-ai-slim` 2.66.0 |
| Trading engine | QuantConnect LEAN (Apache-2.0), self-hosted from source, C# engine isolated | commit `d1c09284c438b70eb965b7dc40f44e7dca69e7b5` (2026-11-05), net10.0 |
| Broker | Alpaca Trading API (paper first; live adapter locked) via official LEAN Alpaca plugin | `QuantConnect/Lean.Brokerages.Alpaca` commit `b92d6ecd391d5a79e91b060668d1ea4fafec9b89` (2026-10-19) |
| Supporting Alpaca client | `alpaca-py` — data/account/connection/reconciliation reads only; never a second order path | 0.45.0 |
| Historical/ML research | Microsoft Qlib v0.9.8 (isolated env), LightGBM, CatBoost, scikit-learn | lightgbm 4.7.0, catboost 1.2.10, scikit-learn 1.9.1; Qlib see `services/research/qlib_env` |
| Financial agent reference | TradingAgents v0.7.1 — reference only; never owns broker authorization | not integrated (reference) |
| Native macOS control | Peekaboo v4.9.7 (allowlisted, non-execution apps) | see `apps/mac-agent/README.md` |
| Browser control | Microsoft Playwright MCP | pinned in `apps/mac-agent/README.md` |
| API backend | Python 3.11 + FastAPI + asyncio | CPython 3.11.17, FastAPI 0.162.0 |
| Execution engine language | C# (LEAN) | .NET SDK 10.0.403 (project-local `.tools/dotnet`) |
| Structured store | SQLite WAL, single writer, migrations, strict transactions | stdlib sqlite3 |
| Analytical store | DuckDB + Parquet with provenance manifests | duckdb 2.0.0, pyarrow |
| Dashboard | React 19 + TypeScript + Vite + TradingView Lightweight Charts (attribution shown) | see `apps/dashboard/package.json` |
| Tooling | uv, pnpm, make, local supervisor (`scripts/supervisor.py`), optional local Docker | uv 0.13.0, pnpm 12.11.2 |
| Testing | pytest, hypothesis, ruff, pyright, Vitest, Playwright, .NET tests | lock files |
| Telemetry | structured JSON logs, SQLite hash-chained audit events | no hosted observability |

## Engineering decisions made while implementing (within the frozen architecture)
1. **Intent signatures are asymmetric (ECDSA P-256/SHA-256).** The risk engine holds the private key; the
   Python gateway boundary and the C# `GatedBrokerageModel` inside LEAN hold only the public key, so no
   LEAN strategy (or any other component) can mint an order. The signed payload is a pipe-delimited
   `signing_string` and LEAN executes the fields parsed from that exact string.
2. **Live execution is a compile-time constant** (`packages/core/modes.py: LIVE_EXECUTION_COMPILED=False`,
   `BOUNDED_LIVE_COMPILED=False`). Env vars, prompts, UI and API cannot change it.
3. **Paper/live isolation** by namespace: separate `var/<ns>/` ledger DB, keys, tokens; distinct secret
   names (`ALPACA_PAPER_*` vs `ALPACA_LIVE_*`); ambiguous `APCA_API_*` names are ignored.
4. **Real model calls are opt-in** (`LAT_AI_TRANSPORT=real`); key presence alone never triggers paid calls.
   Default runs use explicit fake HTTP transports and are recorded `transport=FAKE`.
5. **Playwright MCP pinned** to an exact version instead of `@latest` to satisfy the dependency-pinning rule (§12).
6. **Repository root** is the dedicated project directory `ai-trading/` (the spec's `local-ai-trader/` layout
   is used verbatim beneath it).
