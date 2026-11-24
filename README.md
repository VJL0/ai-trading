# Local AI Trader

Local quantitative research and **paper-only** trading, operated through Claude Code, Codex or the existing
ChatGPT integration. The React dashboard is a read-only operations display. Live execution is compiled off.

From this directory, with the existing dependencies installed:

```sh
make demo        # API :8765 + dashboard :5173; synthetic feed, fake broker, AI calls disabled
make smoke       # isolated real-process MCP/API session verification; no credentials or paid calls
make status
make stop
```

Open [Trading operations](http://127.0.0.1:5173). Start a session through the connected `local-ai-trader` MCP tools:
`start_paper_session(duration_minutes=5, cadence_minutes=1, symbols=["SPY"], use_ai=false)`.
Then inspect `get_session_report`, `explain_decisions`, and `get_system_health`. The API runs the session after the
chat disconnects. A NO_TRADE result is expected whenever no candidate clears the existing quantitative threshold.

`make demo` preserves already-running services; it does not reconfigure them. Check `get_risk_status` and
`get_system_health` before operating. The demo advances historical synthetic market time at 30x wall time; a
five-minute session normally takes about ten seconds. These are simulated results, not Alpaca paper executions.

See [local operations](docs/runbooks/README.md), [session lifecycle](docs/runbooks/paper-sessions.md),
[AI/MCP integration](docs/runbooks/ai-runtime-mcp.md), and [verification report](docs/verification/continuation.md).

Verification:

```sh
make test-py             # unit, integration, failure injection, e2e; excludes paid/broker/GUI opt-ins
make test-lean           # .NET gate and bridge suite, if existing LEAN build is available
make test-dashboard
make test-dashboard-e2e  # headless Chromium; local SSE fixture server
make lint
make typecheck
```

Dependencies are pinned in `uv.lock`, dashboard `pnpm-lock.yaml`, and the existing LEAN source pins.
For a fresh environment see [environment setup](docs/runbooks/environment.md). Real Alpaca paper connectivity
uses the official `alpaca-py` SDK with `make paper` (`LAT_GATEWAY=ALPACA_PAPER`).
Only `ALPACA_PAPER_KEY_ID` and `ALPACA_PAPER_SECRET_KEY` are required for broker access.
LEAN is used for research/backtesting; no QuantConnect deployment license or credentials are needed.
See [SDK paper onboarding](docs/runbooks/broker-paper-onboarding.md).

## Roadmap work (2026-11)

* [Roadmap](docs/roadmap/README.md) and [progress / evidence](docs/roadmap/PROGRESS.md).
* [Position lifecycle](docs/architecture/position-lifecycle.md): autonomous protective stops and exits, exit-aware
  locks, explicit unprotected-position safety states. A stop is a trigger, never a guaranteed exit price.
* [Data pipeline, research snapshots and research/runtime parity](docs/architecture/data-pipeline-and-parity.md):
  `make data-pipeline`, `make data-snapshots`, `make compare-feeds`.

## Multi-asset research (2026-11)

* [Design](docs/architecture/multi-asset-research.md), [runbook](docs/runbooks/quantlab.md) and the
  [results report](docs/verification/multi-asset-research-2026-11-23.md).
* `make capabilities` verifies what the Alpaca account and this build's risk policy actually allow per asset
  class (equities, short, margin, leveraged ETPs, crypto, options). Research eligibility never implies execution.
* `make quantlab-data`, `make quantlab-research`, `make quantlab-daily`: point-in-time daily SIP panels including
  delisted names, crypto, stocks-in-play intraday bars; strategy families compared after costs with walk-forward
  selection, deflated Sharpe, SPA, PBO and a one-use protected holdout; evidence-weighted cross-asset selection
  that returns `NO_TRADE` unless a validated, executable opportunity exists.
* Read-only API `GET /api/v1/quantlab/*`, MCP tools `get_capabilities`, `get_research_results`,
  `get_opportunities`, `run_research_experiment`, and the dashboard's "Multi-asset research" panel.

## Persistent trading memory

The API automatically journals decisions and projects searchable history into a local SQLite memory store.
The shared `search_memory` tool exposes point-in-time research, observations and explicitly unvalidated
reflections. Autonomous research runs controlled memory ablations and retains rejected experiments.
See [persistent memory, evaluation and boundaries](services/memory/README.md).
