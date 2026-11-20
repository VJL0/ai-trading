# Autonomous research verification — 2026-11-12

The existing application now runs a persistent, bounded autonomous research loop with 12 implemented learner families,
including Metal-accelerated neural candidates. Development screening, independent one-use holdout confirmation,
hash-verified portable inference, reversible paper promotion, and read-only activity reporting are implemented.
The original model suite, independent risk authorization, Alpaca adapter, LEAN, ledger and operator tools remain.

## Observed market-data campaign

- Source: existing Alpaca IEX-only ingestion, no paid feed requested.
- Dataset: `ds_6f37a2401444b6ea`; SPY, QQQ, AAPL, MSFT; two calendar years ending November 12, 2026.
- 758,273 accepted minute bars; 19,403 rejected rows and 13,557 gap entries retained in ingestion diagnostics.
- All 12 families completed development walk-forward evaluation over 193 evaluation sessions.
- Each candidate produced fewer than 100 trades (1–85). None passed the full development screen.
- No independent confirmation was attempted, no final holdout was opened for this campaign, and no model was promoted.
- Results are in `autonomous-research-2026-11-12.json`. These are development benchmarks under a cost model,
  not realized portfolio performance or evidence of a sustainable edge.

A subsequent source revision added history selection: real histories take precedence over synthetic fixtures;
short batches cannot displace long histories. The source hash changed, so the daemon is evaluating that revision
separately. Previous experiments and failures remain recorded rather than being overwritten.

## Verification

- Final broad suite after the history-selection change: **1,866 passed, 1 skipped, 8 deselected** in 322.51 seconds.
  The skip is the missing-credentials case because this Mac has Alpaca paper credentials configured.
- Autonomous research/validation suite after that change: **36 passed**.
- Dashboard: **44 tests passed in 7 files**; production build and bundle secret scan passed.
- Ruff and Pyright passed for autonomous research, runtime integration and performance accounting changes.
- All 12 learner artifact formats exercised; neural training used MPS, with portable CPU prediction parity checked.
- API observed in `PAPER` / `ALPACA_PAPER`, promotion status `NO_PROMOTION`, no active autonomous candidate.
- Tests excluded paid model calls, broker-paper integration submissions and GUI automation.

Local operational logs are in `var/logs/`; jobs, events and numeric caches are in `experiments/autonomous/`.
The application-wide validation book is separate from campaign output directories.

## Remaining scope

This is a functioning extensible research pipeline, not completion of every paradigm in the original brief.
RL, causal discovery, graph/temporal networks, time-series foundation models and full meta-learning are not implemented.
Online SGD is a training candidate, not deployed continual learning. Feature drift and expiry monitoring do not replace
realized-performance monitoring. Incremental ingestion batches are not yet consolidated automatically into immutable
cumulative research histories; a suitable new history must reach the existing store before another data campaign.
No claim of trading profitability is made.
