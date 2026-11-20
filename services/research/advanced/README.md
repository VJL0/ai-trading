# Local ML research

This package extends the existing autonomous journal, purged walk-forward splits, protected confirmation book,
portable artifacts, research registry and paper-only runtime. It adds no UI and has no broker/order authority.

## Models and purposes

| Models | Actual implementation and purpose |
|---|---|
| LightGBM, CatBoost, XGBoost, Ridge, Elastic Net | Fold-local multi-task training: net 5/15/60-minute returns, success, volatility, downside, shortfall and dollar-volume targets. |
| DoubleEnsemble | Microsoft's Qlib implementation, three boosters per target with sample reweighting and feature selection in the isolated Qlib interpreter. |
| TabM | Official TabM, eight parameter-efficient members, supervised MPS training and safe portable numeric weights. |
| PatchTST, iTransformer, ModernTCN, multi-task Transformer | Compact local trainable architecture adaptations operating on true OHLCV histories, with optional masked reconstruction pretraining and Reptile initialization. These are not claims to reproduce upstream paper hyperparameters/results. |
| FinCast, Timestral-209M, t0-gamma, Simons-297M, Chronos-2, TiRex-2, TimesFM 2.5 | Actual pinned pretrained checkpoints, local marginal quantile forecasting, and forecast-derived features for fold-local Ridge heads. |
| FinText | Apache-licensed Chronos Tiny 2023 US checkpoint. Benchmarks reject pre-2024 rows; minute-return inference is an explicit domain transfer from daily excess-return pretraining. |
| Kronos-base / optional small | Existing `../kronos` adapter: OHLCV paths, multi-horizon features, sampling diagnostics and guarded token-head fine-tuning. |
| TabICLv2, Kumo Tabular, Mitra-v2 | Actual local in-context regression. Kumo uses the small checkpoint's quantiles; Mitra uses its 1,000-bin v2 head, not the incompatible v1 scalar head. |
| HAR-RV, EWMA, GJR-GARCH | Out-of-sample realized-variance forecasting. GJR fitting rescales minute returns for numerical stability. |
| HMM, Markov switching | Training-only regime estimation with forward-filtered probabilities; no smoothed future-dependent state labels. |
| Symbolic regression / genetic programming | Bounded expression-tree evolution and regularized multi-task heads; no generated Python execution. |
| Contextual bandits / offline RL | Logged-propensity contextual reward models, doubly robust evaluation and behavior-supported fitted-Q iteration. Research only; real behavior logs are required for meaningful policy evaluation. |

The first three target columns preserve the existing net-60-return / volatility / execution-shortfall contract.
Execution shortfall and future dollar volume are proxies, not observed broker fill costs or order-book depth.
Success probabilities use isotonic maps fitted on the purged calibration block in the autonomous pipeline.
The bounded tournament reports raw clipped head probabilities; it does not claim they have been calibrated.

## Setup and execution

```sh
uv sync --extra kronos --extra advanced-research
uv sync --project services/research/qlib_env
uv sync --project services/research/foundation_env
uv sync --project services/research/rd_env
.venv/bin/python -m services.research.advanced setup chronos2
.venv/bin/python -m services.research.advanced setup tabiclv2
.venv/bin/python -m services.research.advanced benchmark \
  --cache experiments/autonomous/cache/DEVELOPMENT_KEY \
  --output experiments/advanced/UNIQUE_RUN
.venv/bin/python -m services.research.advanced statistics \
  --cache experiments/autonomous/cache/DEVELOPMENT_KEY \
  --paired-report experiments/advanced/UNIQUE_RUN/report.json \
  --output experiments/advanced/STATISTICS_RUN.json
.venv/bin/python -m services.research.autonomous --once --budget 24 --workers 2
```

Setup choices include all pretrained names in `foundation.CATALOG` and `TABULAR_CATALOG`. Artifacts live under
ignored `var/models/foundation`; manifests bind checksums and exact revisions. Timestral's Modified MIT weights
have a $20 million consolidated monthly company/employer revenue restriction. The user confirmed eligibility
on 2026-11-18. Preserve license/notice files and reassess eligibility if that circumstance changes.

The root environment never imports every model. Foundation inference uses an isolated interpreter and one
checkpoint at a time. t0 uses MLX; Kronos, FinCast, Chronos, TiRex and TimesFM use MPS. Timestral and Simons use CPU
because their upstream scalers require float64. TabICLv2/Kumo/Mitra use CPU compatibility paths. FinText samples
in batches of eight; FinCast also batches by eight. Native CPU boosters run separately from Torch to avoid
macOS OpenMP conflicts. The autonomous scheduler serializes accelerator jobs and enforces process-tree limits.

## Autonomous research and selection

`autonomous.learners` registers trainable candidates and foundation challengers. The agent explores cheap models
before pretrained challengers, mutates promising real-data candidates, and can enable masked pretraining or
meta-initialization. Fixed-weight Ridge/LightGBM/CatBoost ensembles are candidates with their own confirmation.
All bounded tournament trials, including failed workers, enter the journal and research-only registry.

Microsoft RD-Agent(Q)'s actual Qlib model/factor hypothesis generators prepare proposals, using an application
transport bridge. Proposed experiments are restricted to registered models and bounded parameters. Qlib
DoubleEnsemble executes the actual Qlib training implementation. Enable the live proposal step with:

```sh
LAT_AI_TRANSPORT=real .venv/bin/python -m services.research.autonomous --once --rd-agent --workers 2
```

This requires the application's existing configured provider credentials. The bridge uses the existing Responses
API adapter and a one-request / 12,000-token limit, records usage, and does not pass broker secrets or holdout data
to RD-Agent. The current application setting is `fake`; no paid live LLM experiment was run during verification.
RD-Agent's unrestricted generated-code/Docker loop is not used: its proposals feed the existing controlled workers.

Only existing independent confirmation can select a trainable model for reversible PAPER deployment. Foundation
feature challengers remain research-only: diagnostic samples, pretraining-overlap uncertainty and absent prospective
sequence confirmation prohibit promotion. Their upstream full-model fine-tuning recipes are not all ported; Kronos's
bounded head tuning and the local supervised/SSL/meta training paths are implemented. FinCast/Mitra inference does
not imply the upstream CUDA training recipes have been reproduced.

No policy is promoted from synthetic trajectories or invented propensities. No live trading deployment is enabled.
See `docs/verification/ml-system-2026-11-18.md` for actual results and limitations.
