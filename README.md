# Local AI Trader

Local quantitative research and **paper-only** trading, operated through Claude Code, Codex or the existing
ChatGPT integration. The local monitor shows trading status and emergency controls. Live execution is compiled off.

With the project MCP server connected, tell Claude Code or Codex **“start trading.”** Approve the paper-session
tool call if your client prompts. The tool starts the local API when needed and opens one session for up to
60 minutes during regular market hours. Outside market hours it queues for the next open. The API runs the
session after the chat disconnects.

The operator selects **Paper** or **Live** using `LAT_TRADING_MODE` in `.env`. Live is unavailable in this build.
For Alpaca paper orders, select `PAPER` and provide paper credentials. The session uses the active risk policy
and requires no model API keys.

```sh
make smoke       # isolated real-process MCP/API session verification; no credentials or paid calls
make status
make stop
```

Inspect results with `get_session_report`, `explain_decisions`, or `get_system_health`. Open the [local monitor](http://127.0.0.1:8765/monitor) while the API is running.

The internal `make demo` verification target advances a synthetic market clock at 30× wall time.

See [local operations](docs/runbooks/README.md), [session lifecycle](docs/runbooks/paper-sessions.md),
[AI/MCP integration](docs/runbooks/ai-runtime-mcp.md), and [verification report](docs/verification/continuation.md).

Verification:

```sh
make test-py             # unit, integration, failure injection, e2e; excludes paid/broker/GUI opt-ins
make test-lean           # .NET gate and bridge suite, if existing LEAN build is available
make lint
make typecheck
```

Dependencies are pinned in `uv.lock` and the existing LEAN source pins.
For a fresh environment, copy `.env.example` to `.env` and edit your local settings. See the
[environment runbook](docs/runbooks/environment.md). Real Alpaca paper connectivity
uses the official `alpaca-py` SDK with the Paper selection or the `make paper` convenience command.
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
  `get_opportunities` and `run_research_experiment`.
* `make quantlab-session` (2026-11-24): session-timing research on 5-minute SIP bars for the allowlisted ETFs and
  large caps (intraday momentum, VWAP reversion, opening gaps, overnight, turn-of-month) plus a volatility-managed
  index family, under a simulator that mirrors the runtime lifecycle. No family passed; see the
  [session-timing report](docs/verification/session-timing-research-2026-11-24.md). Opt-in after-close automation:
  `LAT_QUANTLAB_DAILY_ENABLED=1`.

## Persistent trading memory

The API automatically journals decisions and projects searchable history into a local SQLite memory store.
The shared `search_memory` tool exposes point-in-time research, observations and explicitly unvalidated
reflections. Autonomous research runs controlled memory ablations and retains rejected experiments.
See [persistent memory, evaluation and boundaries](services/memory/README.md).
