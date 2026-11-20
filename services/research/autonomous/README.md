# Autonomous local research

This service extends the existing research pipeline; it has **no execution authority**. The existing quantitative
suite, signed risk authorization, Alpaca order path, LEAN integration, operator integrations and ledger are unchanged.

Run a bounded campaign or keep watching the local data store:

```sh
uv sync --extra advanced-research
uv run --frozen --extra advanced-research python -m services.research.autonomous --once
uv run --frozen --extra advanced-research python -m services.research.autonomous
uv run --frozen --extra advanced-research python -m services.research.autonomous --status
```

Continuous mode checks stored intraday histories every five minutes, resumes queued work, explores
available learner families, and mutates the top three development hypotheses. A campaign has 24 trials by default;
when exhausted it waits for new data rather than mining the same history indefinitely. There are no paid model calls,
package installations, remote model downloads, order submissions, or generated-code execution inside the service.
Use the CLI from existing Codex/Claude Code operators; existing MCP tools are preserved. No new MCP mutation tool is
introduced. The existing process supervisor also supports `make research` and `make research-status`. Stop with Ctrl-C; the scheduler terminates its workers and records interrupted jobs.

## What works

* SQLite WAL journal, atomic claims, deterministic job IDs, immutable numeric dataset cache, source/data/quote hashes,
  fold activity events, OOS prediction artifacts, process logs, bounded concurrency and wall-clock/RSS watchdogs.
* Local LightGBM, CatBoost, XGBoost, ridge, Bayesian regression, ExtraTrees, chronological SGD, Gaussian-mixture regime
  experts and PCA representation learning. Models predict net returns, realized volatility and a cost proxy jointly
  when the backend supports it (independent heads for boosting and Bayesian adapters).
* PyTorch multi-task MLP, masked-feature autoencoder pretraining, and feature-token transformer; MPS selected on
  supported Macs, CPU otherwise. These operate on point-in-time tabular features: they are **not temporal sequence
  transformers or pretrained time-series foundation models**. Kronos is available separately through the sequence adapter described below. Missing dependencies are visible, never silently replaced.
* Symbolic feature transformations and capacity mutations form a bounded search grammar; this is not unrestricted
  genetic programming or autonomous source-code generation. PCA/masked modeling use training data only.
* Existing point-in-time data preparation, quotes, cost model, labels, non-overlapping position simulation, purged
  walk-forward folds, and calibration splits. Models fit only inside folds; conservative residual allowances drive
  NO_TRADE. This residual quantile is a heuristic uncertainty buffer, not a coverage guarantee under drift.
* Net return/Sharpe, drawdown, per-trade Sortino, tail loss, target MAE, doubled-cost stress, momentum baseline and
  session-block bootstrap diagnostics. Bonferroni screening counts **all** journal trials, including failures, for both
  positive net return and incremental return over momentum. Passing is eligibility for independent validation only.
* GET `/api/v1/research/autonomous` and a read-only dashboard panel show actual records and failures, with explicit
  SIMULATED labels. No timers masquerade as experiment progress.

## Validation and deployment boundary

The outer final test is excluded entirely from worker caches, including its features and targets. Inner fold results
remain development evidence even when statistically significant. A candidate needs at least 120 development sessions,
100 trades, three positive folds, positive doubled-cost results, acceptable drawdown and corrected bootstrap tests
before it can qualify for independent validation. Synthetic data always fails this screen.

Independent confirmation now runs in a separate process. It reproduces the candidate's development predictions,
locks the candidate and a symbol/time interval in the application-wide `autonomous-validation.sqlite3`, then opens
that interval exactly once. Prior development exposure, overlapping legacy ML/model-suite evaluations and dataset
aliases are checked. The legacy evaluator also honors the new exposure book. Failed/crashed reservations consume the
interval. Confirmation requires recent real data, at least 60 holdout sessions and 50 trades, positive cost-stressed
returns, bounded drawdown and two block-bootstrap tests with summable alpha spending over confirmation attempts.

Confirmed models are exported as hash-bound JSON/numeric NPZ/native booster artifacts; no pickle loader is used.
Prediction parity is checked before holdout scoring. Promotion records a reversible PAPER pointer, separate from the
existing model-suite registry. The runtime accepts it only for `PAPER` + `ALPACA_PAPER`; the existing signal, signed
risk and order paths remain authoritative. A missing first promotion preserves the existing suite. Corrupt, expired
or incompatible promotions yield NO_TRADE. Persistent feature shifts and inference failures can revert to the prior
confirmed candidate; confirmations expire after seven days. A stale model instance cannot roll back a newer model.

The inference adapter conservatively passes a lower net-return estimate to the existing gross-return interface;
that engine subtracts current costs again. This additional haircut can reduce trades and means research returns are
not a forecast of the runtime portfolio. Position sizing and portfolio risk limits remain with the existing engine.
The first real-data campaign used 758,273 accepted IEX minute bars for SPY, QQQ, AAPL and MSFT over two calendar
years. All 12 families completed, but none passed the development screen; no final holdout was opened and no new
candidate was promoted. IEX coverage and recorded gaps limit these results; this is not evidence of a sustainable edge.

Other boundaries: online SGD here is a chronological training candidate, not a deployed continual-learning service;
regime mixtures are fold-local; deployed monitoring currently detects feature shifts and expiry, not every form of concept drift. RL, causal discovery, graph/temporal networks,
semi-supervised learning and full meta-learning are not implemented. They require appropriate
point-in-time data and task-specific adapters/validation; listing a family would not supply those capabilities.

## Data refresh boundary

Automatic selection prefers real histories over synthetic fixtures and newer histories spanning at least 540 calendar
days when available; otherwise it selects the longest available history for diagnostics. Short ingestion batches
cannot displace a longer training history. This is a selection heuristic, not validation evidence. The service does
not yet consolidate incremental batches into immutable cumulative histories or fetch periodic backfills. It waits
for a suitable new stored history, while existing ingestion/operator integrations remain available. Explicit
`--dataset` selection is supported. No data is silently spliced across feeds or revisions.

## Resource behavior

Default workers use two CPU threads each, at most four simultaneous workers and at most one neural/MPS worker.
Each worker has a six-GiB RSS watchdog and 30-minute deadline. These are configurable research budgets, not model-size
caps. RSS does not measure every Metal allocation; MPS also uses its allocator high-watermark. Reserve memory for the
trading service and macOS when increasing budgets. Training is in separate low-priority processes, not API request
handlers. Interrupted jobs are retained as failures rather than silently excluded from the research history.

The data cache avoids repeated feature construction for identical inputs. It excludes the reserved final interval;
source changes invalidate it. A fresh output directory is a fresh research journal, **not a fresh holdout entitlement**.
The existing model-suite inference sources are preserved. Autonomous artifacts pin their own research-source hash.

## Research references

* [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html): Metal backend availability and use.
* [Apple accelerated PyTorch](https://developer.apple.com/metal/pytorch/): Apple GPU training support.
* [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html):
  chronological splitting and gaps; this service uses the repository's stronger label-interval purging.
* [Bailey et al., Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf):
  repeated strategy search requires explicit selection-bias controls. This release uses conservative screening,
  not a claim to implement the paper's PBO estimator or a deflated Sharpe ratio.

Validation: `pytest tests/unit/research/test_autonomous_research.py`; dashboard tests include recorded/simulated output
and failure behavior. Synthetic test profitability is never used as financial evidence.

## Kronos sequence challenger

The [Kronos integration](../kronos/README.md) adds local Kronos-base and optional small, retained forecast paths,
paired multi-horizon benchmarks, fold-local token-head fine-tuning and LEAN OOS score replay. Provisioned models
are discovered by the existing scheduler. Kronos remains research-only; explicit pretraining-overlap and
independent sequence-confirmation blockers prevent the existing tabular promotion route from consuming its holdout.
CPU comparators run in a separate interpreter to avoid conflicting macOS PyTorch/LightGBM OpenMP runtimes.
