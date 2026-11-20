# Kronos integration verification — November 18, 2026

The pre-existing workspace was committed as `7112918` before integration work began. The integration changes
remain in the working tree. No deployed model or paper-trading pointer was changed.

## Implemented

- Local Kronos-base adapter (102,310,592 parameters), optional small, explicit safetensors provisioning and hashes.
- OHLCV/amount inputs, availability checks, 5/15/60-minute forecast features and retained probabilistic paths.
- Evaluation mode, float32 MPS inference, recorded CPU fallback, corrected padding/causal masks and unpadded ragged inference.
- Autonomous scheduling, resource budgets, separate CPU comparator process, and bounded fold-local token-head fine-tuning.
- Common-row benchmarks against LightGBM, CatBoost, Ridge, HAR-RV volatility, and refitted current selected return models.
- Hash-bound research catalog alongside the existing deployed registry; explicit blockers in the existing promotion screen.
- LEAN backtest-only OOS score replay with the existing brokerage, fees, slippage and limit-fill models.

## Actual checkpoint and data run

The downloaded base checkpoint revision was `2b554741eca47781b64468546e77fef3e85130e6`; tokenizer revision was
`0e0117387f39004a9016484a186a908917e22426`. Source is pinned to upstream commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Checkpoint files are local and ignored by git.

Job `9fa371c51dce43f5e7f8ab0163cf7562c7d717a3060de884be82d615f21a633b` used stored IEX dataset
`ds_6f37a2401444b6ea` for AAPL, MSFT, QQQ and SPY. It completed four purged development folds on MPS in
96.8 seconds, with context 64, two independent paths per forecast, 20 calibration rows and eight OOS decisions
per fold. The entire outer holdout remained excluded from worker storage and unopened.

Of 503,754 development samples, 325,946 had label windows aligned to a complete minute grid. The other 177,808
were excluded from **every comparator's** fit/calibration/test pool. This prevents sparse observed-bar labels
from extending a nominal 60-minute forecast beyond 60 minutes; conclusions do not apply to missing-data windows.

| Model | OOS decisions | Trades after uncertainty and costs | Net return |
|---|---:|---:|---:|
| Kronos-base | 32 | 0 | 0 |
| LightGBM | 32 | 0 | 0 |
| CatBoost | 32 | 0 | 0 |
| Ridge | 32 | 0 | 0 |
| Current selected return architectures | 32 | 0 | 0 |

Kronos's volatility QLIKE was 0.2780 versus HAR-RV's 0.5385; volatility RMSE was 9.41 versus 21.90 bps.
These 32 observations do not establish a reliable volatility advantage or a tradable edge. Across the stored
OOS paths, 28.05% of sampled bars violated candle geometry or nonnegative volume/amount checks; raw draws are
retained and this diagnostic is not hidden by repairing the paths. All generated prices were finite and positive.

All paired incremental-value tests failed eligibility. There was no profitable development parent, so actual
checkpoint fine-tuning was not launched. Tiny-model tests verified that the training implementation updates
only token-head weights and rejects fit/validation overlap.

## Verification

- **220 tests passed** across 19 research/LEAN test files, run in separate interpreters. The similarly named
  `test_research_models_support.py` is a fixture/helper module with no collected tests, not a failing test.
- Final calendar-alignment changes were rechecked with **12 Kronos tests and 36 autonomous tests**, all passing.
- Targeted Ruff and Pyright passed; `git diff --check` passed.
- LEAN C# build completed with zero errors; existing upstream/package warnings remained.
- LEAN deterministic positive-score fixture: two orders, $2.76 fees. This verifies fill plumbing, not Kronos performance.
- LEAN replay of the actual 32 Kronos OOS scores against 11,867 matching historical minute bars: zero orders,
  zero fees, equity unchanged at $100,000. This confirms the no-trade replay; it does not establish fill parity
  for a profitable strategy.

The first real-data attempt exposed a macOS OpenMP deadlock when PyTorch and LightGBM shared an interpreter.
CPU comparators now run in a fresh bounded process. A subsequent attempt exposed the deployed registry loader's
removal of its self-hash from in-memory metadata; the benchmark now hashes the verified registry file. Failed
attempts remain in the journal and count toward the search history.

## Remaining boundary

Kronos stays **RESEARCH_ONLY**. Upstream pretraining overlap is unknown, the benchmark is deliberately small,
and no incremental net risk-adjusted value was demonstrated. Independent sequence confirmation and runtime
promotion are **not enabled** in this integration. Existing tabular promotion continues to operate; its route
cannot promote a Kronos research entry. Enabling sequence deployment requires prospective independent evidence
and execution validation, not merely a better development score.

See [operating instructions](../../services/research/kronos/README.md) and the adjacent
[structured verification record](kronos-integration-2026-11-18.json) for metrics, hashes and exact provenance.
