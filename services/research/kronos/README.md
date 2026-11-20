# Kronos local research challenger

Kronos-base (102,310,592 parameters) is the default financial time-series foundation-model challenger.
Kronos-small is an optional 24.7M efficiency baseline. Both use the base tokenizer and a maximum 512-token
context. Inference and bounded token-head fine-tuning run locally in float32, using PyTorch MPS when available,
with a recorded CPU retry for unsupported Metal operations. No inference service or paid API is involved.

## Setup and operation

```sh
uv sync --extra advanced-research --extra kronos
uv run --frozen --extra kronos python -m services.research.kronos setup
# Optional: provision both base and small
uv run --frozen --extra kronos python -m services.research.kronos setup --small

# Bounded diagnostic: 8 OOS decisions/fold, not promotion evidence
uv run --frozen --extra kronos python -m services.research.kronos benchmark \
  --dataset ds_6f37a2401444b6ea --context 64 --samples 2 --rows-per-fold 8

# Normal autonomous scheduler discovers installed local checkpoints automatically.
# Its existing process/RSS/time budgets apply; at most one torch/Kronos worker runs at once.
uv run --frozen --extra advanced-research --extra kronos python -m services.research.autonomous --once
```

Only `setup` accesses the public Hugging Face Hub. It resolves immutable revisions, downloads JSON configuration
and safetensors into ignored `var/models/kronos`, and records SHA256 hashes. Loading verifies both config and
weights, uses explicit local classes and safetensors, and never downloads or unpickles model objects. The manifest
hash is part of each autonomous hypothesis identity. Missing checkpoints remain visibly unavailable.

`KronosAdapter.forecast` takes a timezone-aware historical OHLCV frame indexed by bar start, with
`available_at_ts`, an explicit decision timestamp, and future calendar timestamps. Context is truncated BEFORE
normalization and must be published at the decision. Missing amount uses volume times mean OHLC, recorded here
as a proxy, not exchange turnover. `Forecast.paths` retains `[sample, future_bar, OHLCVA]` draws; `features`
produces multi-horizon net open-to-open return mean, standard deviation, quantiles, positive-return frequency,
and realized-variance forecasts. An h-bar return requires h+1 forecast opens. Raw invalid candle/volume fractions
are reported; nonfinite or nonpositive prices fail closed. Sampling frequencies/quantiles are not calibrated
probabilities or coverage guarantees. Two-path runs are plumbing diagnostics only; use more paths and seeds for
research conclusions.

## Research integration

CPU baselines run in a fresh bounded subprocess: loading macOS PyTorch and LightGBM OpenMP runtimes in one
process can deadlock. Inputs/outputs use temporary numeric NPZ and JSON, never pickle.

The existing numeric development cache includes compact per-symbol raw sequences and availability times, plus
suite volatility inputs and 5/15/60-minute labels. No bars, features or labels from the reserved outer holdout
enter a worker cache. Context lookup filters publication time independently for each decision. Future timestamps
are derived from the calendar, never from observed future rows. The autonomous benchmark currently requires
one-minute, same-session bars. The general adapter accepts explicit timestamp grids at other frequencies.

Each Kronos trial evaluates LightGBM, CatBoost and Ridge on identical fitting, calibration and OOS row sets;
row-budget sampling is deterministic before outcomes are inspected. All models share a complete-case label mask
that excludes future missing-minute windows, ensuring an h-bar target really spans h regular minutes. The report
counts exclusions; results therefore describe complete windows and cannot establish performance during data gaps. Each baseline fits the suite's same net
5/15/60-minute labels. HAR-RV is compared on its appropriate task, 60-minute realized variance, using the
repository's recorded daily/weekly/monthly or short-history fallback periods. Current selected registry return
architectures are refitted inside each fold; later-trained deployed weights are never scored on earlier history.
An unavailable/corrupt selected registry is a missing comparison and blocks eligibility.

All economic comparisons use the existing next-open execution labels, reference notional, quote/fallback spread,
fees, impact model, one-position simulation and doubled-cost stress. Return MAE is reported per horizon;
economic scoring uses the existing 60-minute target. Calibration-only residual allowances apply to every return
model. Kronos also applies its lower sample quantile. Paired five-session block bootstrap comparisons report
incremental net return and a lower bound for the session-Sharpe difference, corrected for journal trials,
comparators and horizons. Sparse/synthetic results fail closed.

The agent can propose a larger context and 16-step token-head fine-tuning after a profitable real-data development
parent. Fine-tuning resets pretrained weights for every fold, consumes only fit-window sequences, freezes the
tokenizer/backbone, clips gradients and saves numeric head weights. It does not fine-tune on calibration or test
outcomes. This is a bounded parameter-efficient adaptation, not full-model retraining or unrestricted code search.

Artifacts live under `experiments/autonomous/runs/<job>/`: predictions, retained paths, fold/paired metrics,
weight/source/data hashes, device/fallback details, timing and optional head weights. The existing model registry
adds a separate `models/challengers` catalog with hash-bound artifact references. Its deployed-registry loader
never selects these research entries, so a challenger cannot displace the current model suite.

## Promotion boundary

Kronos remains **RESEARCH_ONLY**. The upstream pretraining data cutoff/overlap is not established by the downloaded
checkpoint. Local leakage-free splits cannot prove that historical observations were absent from pretraining.
The benchmark records this limitation and passes explicit blockers into the existing autonomous screen before
any final-test reservation. No outer holdout is consumed by these trials and no paper/live pointer is written.

A sequence-specific independent confirmation/runtime deployment path is not enabled in this integration. It must
establish post-checkpoint prospective evidence, credible incremental net risk-adjusted value against all required
comparators, sampling stability and LEAN execution validation before this blocker can be removed. Existing
independent confirmation for tabular learners is unchanged. A profitable development score alone cannot promote
Kronos, and the agent cannot remove the blocker through hyperparameter optimization.

## LEAN workflow

All comparator scores share one hash-verified replay format. Export a completed trial:

```sh
uv run --frozen --extra kronos python -m services.research.kronos lean-export \
  --artifacts experiments/autonomous/runs/JOB --model kronos_base --output var/kronos-scores.csv
# Read csv_sha256 from the export result; scores-file must be absolute.
scripts/lean_backtest.sh --algorithm Lat.Lean.Algorithms.LatResearchReplayAlgorithm \
  --resolution Minute --data-folder YOUR_MATCHING_LEAN_EXPORT --dataset YOUR_DATASET \
  --symbol SPY --start YYYY-MM-DD --end YYYY-MM-DD \
  --param scores-file=/absolute/path/to/var/kronos-scores.csv --param scores-sha256=SHA256
```

Build with the existing `scripts/lean_build.sh`. Replay rejects live mode and altered score files, only consumes
scores at/after their decision time, and uses the existing LAT brokerage, fees, spread/slippage and limit-fill
models. It drops stale entries and expires unfilled entry limits after one bar. Export each comparator and use
the same underlying LEAN data and cost parameters for paired execution checks. LEAN's limit fills are a separate
measurement from the research simulator's next-open convention; neither parity nor profitability is assumed.

## Upstream provenance and compatibility patches

Source: https://github.com/shiyu-coder/Kronos at commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Only `model/kronos.py`, `model/module.py`, and the MIT license
are vendored. The license is retained in `vendor/LICENSE`.

Local patches: relative imports replace upstream global `model`/sys.path injection; padding masks (`True` means
padding) are inverted for PyTorch SDPA (`True` means allowed), combined with causal masks explicitly, and passed
with `is_causal=False` when an explicit mask is present. Evaluation disables dropout in both tokenizer and model.
Variable-length inference runs individual unpadded contexts, avoiding padding in normalization/tokenization.
Paths are represented as separate batch rows with upstream `sample_count=1`, preventing upstream averaging.
MPS fallback clears upstream non-buffer RoPE caches before CPU retry. No CUDA-only autocast/DDP scripts are used.

Verification: `uv run --frozen --extra kronos pytest tests/unit/research/test_kronos.py` plus the existing research
and LEAN tests. The verification report distinguishes real checkpoint inference, real historical diagnostics,
synthetic tests and LEAN fixture execution.
