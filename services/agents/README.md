# Research crew (`services.agents`)

An optional, separate process that has the operator's own **Claude Code** (`claude -p`) and **Codex**
(`codex exec`) CLIs do research work: propose factor hypotheses, review each other's results, analyse new scans,
and propose process improvements. Design: [docs/architecture/agent-crew.md](../../docs/architecture/agent-crew.md).

It has **no execution authority**. It never imports or calls the API, risk, order ledger, positions, sessions or
broker code, never starts the API, never opens the protected holdout and never promotes a strategy. Trading runs
the same with the crew stopped. Tests enforce both import directions (`tests/unit/agents/test_import_boundaries.py`).

## Running

```sh
uv run --frozen python -m services.agents providers        # CLI availability, no quota spent
uv run --frozen python -m services.agents run --once       # one tick
uv run --frozen python -m services.agents run              # loop (SIGINT/SIGTERM: stop and requeue in-flight work)
uv run --frozen python -m services.agents status [--json]  # queue, cooldowns, today's runs, last notes
uv run --frozen python -m services.agents notes [--kind review] [--limit 20] [--json]
uv run --frozen python -m services.agents enqueue hypothesis --payload '{"mechanism": "reversal"}'
uv run --frozen python -m services.agents run --once --fake  # demo: simulated providers, separate store, nothing evaluated
```

Sign in to each CLI yourself (`claude` / `codex login`). The crew never reads, copies or stores tokens. Only one
`run` loop may use a store at a time (lock file next to the store).

## Each tick

1. Expire overdue tasks (a review nobody independent could take becomes `CANCELLED (no independent reviewer)`),
   re-enable providers whose `availability()` is OK again.
2. Plan (`planner.py`) from state: a **review** for every hypothesis with an accepted ledger result (never by its
   author), a **hypothesis** when the hourly agent-experiment budget has a free slot and none is in flight (mechanism
   chosen by a UCB score over the expressible mechanisms), a **market analysis** per new scan `asof`, an
   **improvement** pass after >= 10 new failures (at most daily). No new evidence means no task and no quota spent.
3. Run the highest-priority runnable task on a provider that is not cooling down, under its daily cap and not
   excluded, balancing by fewest runs today. The agent gets the read-only research MCP profile, a disposable workdir
   under `experiments/agents/runs/<run_id>`, a strict JSON output schema and a compact sanitized digest.
4. Record. Hypothesis specs go through `validate_spec`; only accepted specs run, one at a time, through
   `run_agent_experiment` (development data, hourly budget, duplicate/redundancy refusal, the normal gates). The
   crew stores the **trial ledger's** result, never numbers an agent reports.

## Budgets (env, see `.env.example`)

`LAT_AGENTS_PROVIDERS` (claude,codex), `LAT_AGENTS_MAX_CONCURRENT` (1), `LAT_AGENTS_DAILY_RUNS_PER_PROVIDER` (12),
`LAT_AGENTS_RUN_TIMEOUT_S` (900), `LAT_AGENTS_MAX_TURNS` (12), `LAT_AGENTS_SPECS_PER_TASK` (3),
`LAT_AGENTS_COOLDOWN_MINUTES` (60), `LAT_AGENTS_REVIEW_WAIT_HOURS` (24), `LAT_AGENTS_IDLE_SLEEP_S` (600),
`LAT_AGENTS_STORE`, `LAT_AGENTS_CLAUDE_BIN`, `LAT_AGENTS_CODEX_BIN`, `LAT_AGENTS_MODEL_CLAUDE`,
`LAT_AGENTS_MODEL_CODEX`. Agent experiments are additionally capped by quantlab (6 per hour).

## Recovery

* **Usage limit**: the task goes back to the queue without using an attempt; the provider cools down until the
  reset time the CLI printed, else `COOLDOWN_MINUTES` doubled per consecutive limit (cap 12 h). The other provider
  keeps working. A successful run resets the counter.
* **Not signed in / CLI missing**: as above, and the provider stays disabled until `providers` reports it OK.
* **Bad output, error, timeout**: one attempt used; the retry prefers the other provider; `FAILED` after 2.
* **Crash or kill**: on the next start, `RUNNING` tasks return to `QUEUED` (attempt not used) and open runs are
  marked `INTERRUPTED`.

## Memory

`experiments/agents/crew.sqlite3` (WAL; not the trading ledger). `notes` is append-only (SQLite triggers).
`evidence_status`: `AGENT_NOTE` (agent proposals and opinions), `LEDGER_DEV` (copied development trial-ledger
results), `CREW_EVENT` (bookkeeping and deterministic refusals). None of it is validation, a fact or a signal. Every
note that goes back into a prompt is sanitized (`services.ai.injection`) and quoted as data; HIGH-severity text is
withheld.

## What it cannot do

Trade, propose orders, halt, change modes or risk limits, start sessions or the API, run generated code, edit the
repository, use paid model APIs, open the holdout or promote. Mechanisms the price/volume DSL cannot express
(earnings, fundamentals, insider, analyst, ...) only produce data requests. Agents in the Codex sandbox can read
files the process user can read; holdout integrity rests on the deterministic holdout ledger.
