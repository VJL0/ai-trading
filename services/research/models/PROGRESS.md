# services/research/models PROGRESS

Attempt 3 (2026-11-07). Inventory at start: services/research/models/ did not exist, no experiments/models, no
tests/*/research/test_research_models_*.py -> fresh implementation. Research core (115 tests) untouched.
Attempt 4 (2026-11-07): inventory: models/{config,features,targets,estimators,vol,regime,strategy,walkforward,registry,
predict}.py + cli train-models/predict + 4 test files exist from attempt 3 (unrecorded). Re-verifying from scratch.
- start: pytest test_research_models_{targets,components,support} 17 passed; test_research_models_suite 16 passed (6.4 s).
- R1 targets.py: next-open net-return/success/rank/vol/cost labels; NEW aligned cross-sectional decisions per bar slot
  (forward_labels with explicit decision), HAR base auto (literal 1-session base if >= 32 sessions else FALLBACK).
  ruff/pyright clean; pytest models targets+components+suite 33 passed. Left: new tests for alignment/HAR choice.
- R5 predict.py: regime now from the TRAINING proxy symbols with staleness checks (was: universe only); registry
  load cross-checks deployed vs selected per target. 33 passed. Left: regime/proxy tests.
- R7 tests added: targets (forward_labels==symbol_labels, aligned decisions w/ per-symbol latency, HAR base choice,
  literal session HAR sums) 12 passed; leakage (frozen-test perturbation leaves dev metrics/BH/deployed models
  identical; frozen test once -> 2nd run REFUSED -> zero baseline) 3 passed; regime (Markov selected on switching
  market, filtered regime tracks hidden state 26/28, proxy missing/stale -> UNKNOWN, future-data invariance) 5 passed.
- R4 registry.py: + inference_code_hash check (frames/features/models features/estimators/vol/regime sources), deployed-vs-
  selected consistency per target incl. calibrator method; regime eval targets restricted to the fold window.
  Tests: CLI unit 4 passed; store integration (real MarketDataStore fixture, API call shape, 2nd run REFUSED) 1 passed;
  semantic registry checks w/ re-hashed self-hash 3 passed. Left: full gate, real CLI runs into experiments/, README.
- R3 walkforward: contaminated frozen-test reuse can no longer confirm a trading model (first look only); vol
  rejection reasons for all candidates. Gate: ruff OK, pyright 0 errors (src+tests), pytest research dirs 168 passed
  (115 core + 53 model suite). tools integration test_model_tools_against_a_real_trained_registry 1 passed.
- R6 README section added. Next: real CLI runs into experiments/models (LEAN first, synthetic last).
- R6 CLI runs (repo store -> experiments/models): LEAN ds_2fa8b3644cc15c28 -> mreg_20261108T035209Z_032be2a9aee7;
  synthetic ds_1c99c07583eae908 -> mreg_20261108T035222Z_73806d805978 (newest). Both: all trading targets
  zero_baseline, frozen test EVALUATED_ONCE; predict on both exit 0, NO_TRADE. README results recorded.
- R6 CLI: train-models default dataset = newest contiguous 1Min dataset (LEAN sample only when named; refuses if none)
  so `make models` picks ds_1c99c07583eae908. Test added (fake store).
- FINAL gate (2026-11-08 UTC): ruff OK + pyright 0 errors on services/research/models, cli.py, model tests;
  pytest tests/{unit,integration,failure_injection,e2e}/research 169 passed (54 model-suite); tests/integration/tools/
  test_tools_real_runtime.py 7 passed. Registries in experiments/models verify against current code. Left: nothing in
  scope; API-side requests (proxy symbols in predictions(), registry reload) reported to integrator.
