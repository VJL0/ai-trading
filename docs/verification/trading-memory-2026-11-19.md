# Trading journal and persistent memory verification

Implemented in the existing app and activated on November 19, 2026 (New York time). The API and autonomous
research service are running. Execution remains **SIMULATED / FAKE**, with live trading disabled. No risk limits,
security approvals, LEAN order path or Alpaca order path were changed. A pre-migration ledger backup is retained
at `var/paper/backups/pre-memory-2026-11-19.sqlite`.

## Working capabilities

- Transactional, append-only capture of operational observations, forecasts, decisions, rejected opportunities,
  model outputs, executions and activity; existing records backfilled into durable memory.
- FTS5 matching, eligible-candidate BM25, hash-vector retrieval, numerical forecast-state similarity, structured
  filters and rank fusion. Both content visibility and ranking statistics respect the query's knowledge cutoff.
- Shared read-only `search_memory` tool, exercised through the running HTTP backend used by MCP. Existing Codex
  approval defaults are preserved; the new tool inherits the prompt default.
- Persistent development research and failures, evidence-linked hypothesis reflections, and independently
  verified semantic confirmation receipts. Raw holdout rows and predictions are not imported.
- Expected/observed forecast diagnostics when compatible recorded endpoint quotes exist. These explicitly
  identify counterfactual midpoint returns; actual execution P&L remains in the ledger.
- Five fixed, matched memory ablations in the existing autonomous scheduler, with leakage guards, cost stress,
  session-block tests, trial correction and persistent rejection decisions. Existing independent validation
  and promotion remain authoritative for supported models.

## Economic result

The completed real-data development comparison covered **243,220 OOS rows across 193 observed sessions** for
AAPL, MSFT, QQQ and SPY. Baseline, numerical episodes, keyword, hash-vector, hybrid and reflection variants all
selected the same **one losing trade**. Each returned **-1.6574%** under the research simulator. Sharpe is undefined
with one trade. Incremental mean session return was zero and every paired incremental p-value was 1.0.

**All five memory variants were rejected.** This is neither evidence of a profitable strategy nor a general
finding that RAG cannot work. Sparse trading and IEX data limitations constrain the experiment. No protected
confirmation interval was opened and no memory strategy was promoted. The report counts all 148 attempts in
that research journal when setting its correction threshold. Subsequent version-bound campaigns remain logged
and bounded by the existing scheduler.

The memory vector baseline uses deterministic word/bigram hashing, not a learned semantic model. The experiment
tests retrieval of coarse market-state descriptions and outcomes, not unconstrained LLM RAG. Passing future
memory challengers would still require a paired independent-confirmation implementation; their promotion is
currently blocked deliberately.

## Verification

The broad regression run passed 1,868 tests, skipped six external/paid-environment checks, and exposed two
integration issues. Both were repaired and successfully rerun: the new tool retains the existing approval
default, and signal insertion counts now exclude outbox-trigger writes. The focused integration run passed
422 checks; the final memory suite passed 15. Eight advanced-model tests and 16 CPU/Kronos tests passed in
isolated runs after a combined native-library run hung. These counts overlap and must not be added together.

Ruff, targeted Pyright and `git diff --check` pass. Runtime HTTP retrieval returned historical records with
`execution_authority=false`. Three local retrieval smoke measurements were 48.05, 36.68 and 37.32 ms. More than
4,500 memory records persisted by final verification; exact counts and source hashes are in the
[machine-readable report](trading-memory-2026-11-19.json).

Operational details, commands and bounded-search limitations are in the
[memory service documentation](../../services/memory/README.md).
