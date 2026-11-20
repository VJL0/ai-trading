# Persistent trading memory

The operational SQLite database remains authoritative. Migration `0700_memory.sql` adds an append-only,
transactional outbox for audit events, market snapshots, all scored signals (including rejected candidates),
AI model runs, research findings, proposals including NO_TRADE, risk rejections, order events, fills and
pipeline activity. Existing records are backfilled. Runtime forecast calls also retain full PredictionSets.
The existing research journal and prediction artifacts remain the authoritative experiment record.

A background API task projects batches of 500 records every 30 seconds into `var/<namespace>/memory.sqlite3`.
Restarting after a crash replays the last batch idempotently; checkpoints advance only after durable inserts.
Index failure is logged and retried without invoking the broker or changing a risk decision. Projection lag
can accumulate under sustained high volume. The outbox retains records until indexed; it is not a lossy queue.

## Retrieval and evidence

`search_memory` is a read-only tool in the shared registry, available through HTTP, MCP, ChatGPT and the
existing in-app agents. Results carry source identifiers, timestamps and explicit evidence status. Queries
use the runtime clock; callers cannot override the cutoff to read future memory. Historical backfills become
available when captured, not retroactively at their event time. Research results use their recorded completion
time. A reflection never becomes a fact because an agent repeats it.

The local store provides structured kind/symbol/time filters, FTS5 matching with BM25 statistics fitted only on eligible candidates, signed SHA-256 word/bigram vectors,
normalized numerical state distance (a symbol query uses its latest eligible forecast state), and reciprocal-rank fusion. Search bounds candidates to the
latest 2,000 eligible records plus 2,000 lexical matches. It is approximate at larger corpus sizes and reports
its candidate count. The hash-vector encoder is a reproducible lexical baseline, **not a pretrained semantic
embedding model**. No model downloads, external embedding APIs, sqlite native extension or GPU allocation is
needed. Semantic discoveries require a CONFIRMED receipt from the existing independent validator and verified
model artifact hashes. The importer retains the receipt, not holdout rows or predictions. Expiry remains
visible; historical confirmation is not authorization. Ordinary appends cannot assert validation. Experiment development evidence is not independent confirmation.

SQLite WAL/FTS5 reuses the app's durable storage and full-text facilities. DuckDB/Parquet continue to serve the
existing market-data engine; adding a second analytics store or MLflow server would duplicate existing
provenance and experiment tracking. sqlite-vec and learned embeddings are deferred until a benchmark shows a
benefit that justifies their installation and indexing costs. Existing MPS/PyTorch learners are preserved.

## Outcomes and autonomous research

The projector imports completed and failed development jobs, their hypotheses, lineage, metrics and errors.
It emits deterministic, evidence-linked reflections describing failed screening criteria and required tests.
These are research hypotheses, not LLM conclusions promoted to market knowledge. The existing bounded
hypothesis grammar and workers continue to run the experiments; arbitrary generated code is never executed.

Where suitable recorded quotes exist near both endpoints, forecast diagnostics link a signal to its start
and horizon snapshots, expected return, observed midpoint return, estimated costs and prediction error.
These are explicitly **counterfactual mark-to-market diagnostics**, including rejected opportunities, not
realized trading P&L. Actual fills and account P&L remain in the ledger. Missing or mismatched data leaves an
outcome unresolved. The bounded recent-signal scan does not guarantee automatic resolution of older signals.

Autonomous campaigns prioritize six matched ridge variants: baseline, numerical episodes, keyword regime
retrieval, hash-vector regime retrieval, hybrid retrieval and a conservative local overprediction reflection.
All use the same purged walk-forward folds, training-only normalization, at most 256 calibration episodes,
16 neighbors, fixed 50% residual shrinkage, costs, doubled-cost stress and non-overlapping position simulator.
Calibration label endpoints must precede the test interval. Test labels never enter retrieval. Text descriptors
are coarse standardized market features, not news or free-form LLM rationales. Thus this tests bounded
market-state retrieval, not whether arbitrary LLM RAG improves returns.

`memory-comparison.json` reports paired session-return bootstrap tests, Sharpe/drawdown differences and
all-attempt correction (failures included). Memory challengers remain research-only even if a development
comparison passes: promotion requires implementing and passing a matched independent confirmation protocol.
Existing protected holdouts, risk approvals, LEAN and Alpaca execution paths remain authoritative.

Run a reproducible comparison on an existing **development-only** artifact:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python -m services.memory.benchmark --cache experiments/autonomous/cache/<dataset-key>
```

This verifies the numeric cache hash, preserves preprocessing provenance, binds current evaluation code,
records all six attempts, and never opens a holdout. Results persist under
`experiments/autonomous/memory-benchmark/`. The normal autonomous scheduler writes its own paired report
on every completed batch. Continuous projection starts automatically with the API; continuous research uses
the existing `python -m services.research.autonomous` service.

References: [SQLite FTS5](https://www.sqlite.org/fts5.html),
[SQLite WAL](https://www.sqlite.org/wal.html),
[Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
These support storage and selection-bias controls; none establish a profitable trading edge.
