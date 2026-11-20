# services/research — no-LLM quant baselines, cost model, signal engine, ML walk-forward, Qlib

Deterministic quantitative research for the local paper-trading workstation (spec §3, §5 `QuantSignal`, §12,
acceptance criterion 3). Nothing in this package places, routes or authorises orders and nothing reads secrets.
**No edge is claimed**: every result file carries `no_edge_note`, the dataset version and `is_simulated`.
Results on synthetic or small sample data are not evidence of a tradable edge.

## Commands

```bash
uv run --frozen python -m services.research.cli datasets                      # stored datasets (MarketDataStore)
uv run --frozen python -m services.research.cli baselines   [--dataset VER]   # -> experiments/baselines/<run_id>/
uv run --frozen python -m services.research.cli signals --asof 2026-11-04T19:30:00Z [--dataset VER]
uv run --frozen python -m services.research.cli walkforward [--dataset VER]   # -> experiments/ml/<run_id>/
uv run --frozen python -m services.research.cli qlib        [--dataset VER]   # -> experiments/qlib/<run_id>/
uv run --frozen python -m services.research.cli train-models [--dataset VER] [--source store|synthetic]  # -> experiments/models/<ver>/
uv run --frozen python -m services.research.cli predict --asof 2026-11-04T19:30:00Z [--dataset VER] [--registry VER]
```

The local quant model suite (`train-models` / `predict`) is documented in *Local quant model suite* below.

Common options: `--source auto|store|synthetic` (default `auto`), `--timeframe 1Day|1Min|...`, `--symbols A,B`,
`--out <experiments dir>`, `--datasets-dir <store root>`. `signals` adds `--min-net-edge-bps`, `--horizon-minutes`,
`--model-run experiments/ml/<run_id>`; `walkforward` adds `--horizon-bars`, `--folds`, `--calibration
isotonic|sigmoid`, `--allow-final-test-reuse`; `qlib` adds `--horizon`.

Data is read through `services.market_data.store.MarketDataStore` (imported lazily inside the command). With
`auto` and no stored dataset the command falls back to the in-memory SYNTHETIC research fixture
(`research-synthetic-v1`) and labels every output so; `--source store` never falls back. Datasets with injected
defects are only used when named with `--dataset`. For stored RAW datasets the CLI loads the dataset's
point-in-time corporate actions (`load_corporate_actions(..., asof=...)`) and applies them (see *Data handling*).
Exit codes: 0 ok, 2 data/usage error (fail closed, JSON error on stderr), 3 Qlib BLOCKED/FAILED.
`make backtest` runs `cli baselines` (then the LEAN backtest owned by trading/).

## Public API (import paths used by services/api and others)

| Name | Signature / purpose |
|---|---|
| `services.research.costs.CostModel` | `CostModel(fees=FeeSchedule(), spread=SpreadModel(), slippage=SlippageModel())`; `.estimate(symbol, side, quantity, price, *, spread_bps=None, adv_shares=None) -> CostEstimate`; `.estimate_round_trip(symbol, quantity, price, *, exit_price=None, spread_bps=None, adv_shares=None) -> RoundTripEstimate`; `.describe()` |
| `services.research.costs.spread_bps_from_quote(bid, ask)` | full quoted spread in bps of mid |
| `services.research.signals.QuantSignalEngine` | `QuantSignalEngine(cost_model, *, min_net_edge_bps=5, horizon_minutes=60, model=None, lookback_bars=20, drift_shrinkage=0.25, order_notional=1000, max_data_age=None, min_adv_usd=10_000_000, clock=None, max_abs_bar_log_return=0.25, max_missing_sessions=1, horizon_within_session=True)`; `.generate(bars_df, asof, universe, causal_id, *, quote_spreads_bps=None) -> list[QuantSignal]`; `.generate_report(...) -> SignalReport` (`.candidate`, `.outcome`, `.no_trade_reason`, `.skipped`) |
| `services.research.signals.persist_signals(db, signals, causal_id) -> int` | idempotent insert into core `quant_signals` via the single writer |
| `services.research.baselines.run_all_baselines(bars_df, cost_model, out_dir, *, strategies=None, initial_cash=Decimal(100000), symbols=None, run_id=None, dividends=None, data_adjustments=None, segment_policy=SegmentPolicy()) -> dict` | writes `<out_dir>/<run_id>/{results.json,manifest.json,runs/<strategy>__<symbol>/summary.json}`; summaries follow `GET /api/v1/backtests` |
| `services.research.baselines.run_backtest(sym_bars, targets, cost_model, *, strategy, params=None, initial_cash=..., liquidate_at_end=True, entries_expire_at_session_end=False, dividends=None) -> BacktestResult` | single-symbol event-driven PIT backtest of a 0/1 target |
| `services.research.adjustments.apply_corporate_actions(bars_df, actions, *, adjustment="raw", asof=None) -> AdjustmentResult` | split adjustment + split-adjusted dividends from PIT records (`.bars`, `.dividends`, `.report`) |
| `services.research.frames.prepare_bars(df)`, `last_clean_segment(bars, policy=SegmentPolicy())` | validate/normalise (fails closed); data-quality segmentation |
| `services.research.ml.run_walk_forward(bars_df, cost_model, out_dir, *, config=WalkForwardConfig(), run_id=None, allow_final_test_reuse=False, evaluate_final=True, data_adjustments=None, segment_policy=SegmentPolicy()) -> WalkForwardOutput` | full ML pipeline |
| `services.research.ml.TrainedSignalModel.from_output(out)`, `load_trained_model(run_dir)` | plug a selected walk-forward model into `QuantSignalEngine(model=...)` |
| `services.research.qlib_bridge.convert_to_qlib(bars_df, provider_dir, *, segment_policy=SegmentPolicy())`, `run_qlib_workflow(bars_df, out_dir, *, horizon=5, seed=..., run_id=None, timeout_s=1200, data_adjustments=None)` | Qlib provider conversion + isolated run |

Bar frame input contract (exactly what `MarketDataStore.load_bars_df` returns): columns
`[symbol, timeframe, start_ts, end_ts, open, high, low, close, volume, available_at_ts, dataset_version,
is_synthetic]`, one timeframe per call, tz-aware UTC timestamps (naive rejected), `available_at_ts >= end_ts`,
no duplicates/overlaps/nulls (gaps are never filled). Float price columns are converted to exact `Decimal`
through `repr` at this I/O boundary; float mirrors (`close_f` …) are used only for features/statistics.

## Design

**Cost model (`costs.py`, Decimal only).** Per leg: commission (Alpaca equities $0; per-share/per-order/minimum
configurable); SELL-only regulatory fees: SEC Section 31 = rate × proceeds (default $27.80 per $1M, the rate
effective 2024-05-22 — **VERIFY CURRENT RATE**; the SEC resets it at least annually and set it to $0 from
2025-05-14, so the default is deliberately conservative) and FINRA TAF $0.000166/share capped at $8.30/trade
(effective 2024-01-01 — **VERIFY CURRENT RATE**), each rounded up to the cent; spread = half the quoted spread,
or a liquidity-bucket estimate from dollar ADV (2/5/15/50 bps full spread), or 30 bps when liquidity is unknown;
slippage = 2 bps fixed + `100 bps × sqrt(qty / ADV)` (10 bps flat when ADV unknown). Floats and non-finite
values are rejected. Not calibrated to fills.

**Baselines (`baselines.py`).** Buy-and-hold; momentum (long iff `C_t / C_{t-L} - 1 > 0`); intraday mean
reversion (z-score of close vs session VWAP, long below `-entry_z`, exit at `exit_z`, always flat before the
scheduled XNYS close, entries never fill in a later session; skipped on daily bars); filtered momentum (momentum
AND trailing dollar ADV ≥ floor AND annualised realised vol ≤ cap). Event-driven, long-only, no leverage, whole
shares, one position at a time per backtest. Decision at bar `t` is taken at `available_at_ts[t]` from bars ≤ t;
execution at the open of the first bar with `start_ts ≥ decision time` plus modelled costs; dividends (when
supplied) are credited to positions held into the ex-date; terminal positions are liquidated at the last close
with SELL costs. Metrics: total return, CAGR (daily bars only), Sharpe (mean/stdev of end-of-session equity
returns × √252, rf = 0), max drawdown, turnover (traded notional / mean equity), trades, total costs (fees,
spread, slippage), dividend income, exposure (fraction of bars ending in a position).

**Signal engine (`signals.py`).** Only rows with `available_at_ts ≤ asof` are even parsed (future rows cannot
influence or break a decision). Baseline score = momentum t-stat `ln(C_t/C_{t-L}) / (σ √L)`; expected return =
shrunk trailing drift projected over the horizon (a heuristic prior, *not* a fitted estimate). Round-trip
spread/fees/slippage from the cost model (quote spread when supplied), `net_edge_bps = expected − costs`, ranked.
**At most one** `is_candidate=True`: the highest-ranked signal that passes every check — `net_edge_bps >
min_net_edge_bps`, its own data fresh (intraday: last bar ≤ 2 bars + 2 min old; daily ≤ 4 days), dollar ADV above
the floor, expected return positive, no one-bar |ln r| > 0.25 and no more than one missing XNYS session inside
the lookback (suspected bad print / unadjusted corporate action / feed hole), valid model output and (with a
model) model horizon = engine horizon, and - for intraday bars - `asof + horizon` no later than the scheduled
XNYS close (regular session only; no overnight hold is planned). Only universe symbols are validated, so a bad
row of an unrelated symbol cannot block a decision; a knowable bad row of a universe symbol fails closed. A stale
or illiquid symbol therefore never blocks, nor becomes, the candidate. Otherwise no candidate → caller emits NO_TRADE. `probability_up` is empty for the baseline;
`probability_calibrated` is True only for a trained classifier whose calibrator passed the frozen-test check
below: the prediction must claim it AND the model must declare `calibration_evaluated is True`. `is_simulated` mirrors `is_synthetic` in the data.

**ML walk-forward (`ml.py`, `features.py`, `splits.py`, `stats.py`).** Features: log returns over 1/5/20 bars,
20-bar realised vol, volume z-score, bar range, distance from the 20-bar mean (trailing windows only). Label:
price-only forward return over `H` bars under the backtest's execution convention; label interval
`[decision, available_at(exit bar)]`. Splits: frozen final test = last 20 % of decision times; expanding
walk-forward folds over the rest; **purging** of training samples whose label interval overlaps the test window;
**embargo** of `E = H` decision steps; held-out calibration block (last 25 % of each training set, itself
purged/embargoed). Variants: deterministic baseline t-stat (no fit), LightGBM classifier/regressor, CatBoost
classifier/regressor (fixed seeds, single thread, deterministic). Calibration: isotonic (default) or sigmoid
(Platt) fitted on the calibration block only; Brier (raw, calibrated, climatology), log loss, ECE, reliability
table. **Multiple testing: Benjamini–Hochberg FDR** (q = 0.10) over the variant family; per-variant one-sided
p-value for H0 "OOS rank IC ≤ 0" with `n_eff = unique OOS decision times / H` (overlapping labels and
cross-sectional dependence make naive n optimistic); checked against `scipy.stats.false_discovery_control`.
Selection = best mean development-fold IC (the frozen test is never consulted — tested by rewriting every
frozen-period bar and asserting identical development results). The frozen test is then evaluated **once**
(selected variant + pre-registered baseline); its sample-key sha256 is recorded and
`<out_dir>/frozen_test_registry.json` refuses a second evaluation — of the same samples, **or of any overlapping
time interval of the same dataset version** (changing horizon/folds/features cannot buy a fresh look).
Check-and-reserve runs under an exclusive `fcntl.flock` and the registry entry is written (`STARTED`) before the
frozen data is touched, so concurrent runs cannot both get a first look and a crashed evaluation still counts.
`--allow-final-test-reuse` labels the result CONTAMINATED and disables the calibration flag.
`probability_calibrated_for_signals` is True only if the selected model is a classifier with a held-out
calibrator, evaluated once, whose calibrated Brier on the frozen test ≤ climatology Brier. Outputs:
`results.json`, `manifest.json` (file sha256s, library versions), `oos_predictions.csv`,
`final_test_predictions.csv`, `model/` (LightGBM text / CatBoost cbm + JSON meta; no pickles).

**Data handling (`adjustments.py`, `frames.last_clean_segment`).** RAW datasets (e.g. the LEAN sample) are
split-adjusted backward from point-in-time records (`split_factor` = price multiplier; prices × factor, volume
÷ factor for sessions before the ex-date; `asof` drops records not yet published). All features/rules are
scale-free, so this changes price levels only, never a return visible at decision time. Dividends become
split-adjusted cash per share credited in backtests. Then every research run keeps, per symbol, only the last
clean contiguous segment: a cut at > 5 missing XNYS sessions or a one-bar |ln r| > 1 (ticker reuse / bad data).
Both steps are recorded in every output (`data_adjustments`, `data_quality`).

**Qlib v0.9.8 (`qlib_bridge.py`, `qlib_env/`).** Separate uv project, `requires-python ==3.11.*`. PyPI's newest
`pyqlib` is 0.9.7, so 0.9.8 is built from upstream tag `v0.9.8` (commit `54355232463878d2eebb91fe0ee5fa7fa1f5976c`,
verified with `git ls-remote` on 2026-11-06; the installed dist's `direct_url.json` records the same commit).
Set up with `cd services/research/qlib_env && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB=0.9.8 uv sync --python 3.11`
(its own `.venv`; the root venv never imports qlib). The bridge writes Qlib's binary provider format directly
(`calendars/day.txt`, `instruments/all.txt`, `features/<sym>/<field>.day.bin` as `<f` float32 with the
calendar start index first), aggregating intraday bars to PIT daily bars, then runs
`qlib_env/lat_qlib_workflow.py` in a subprocess with a minimal environment (no credentials, MLflow telemetry
off): `qlib.init` (region US, 1 kernel, no caches) → `DataHandlerLP`/`QlibDataLoader` with 7 expressions →
label `Ref($open,-(H+1))/Ref($open,-1)-1` → `DatasetH` train/valid/test with `H+1`-session gaps →
`LGBModel` (early stopping on valid) inside `R.start` → `SignalRecord` + `SigAnaRecord` (IC, Rank IC) in a
local file MLflow store. Results in `experiments/qlib/<run_id>/` (`qlib_data/` and `mlruns/` are git-ignored).

## Local quant model suite (`services/research/models/`)

Leakage-free walk-forward training of a local model suite on intraday (1-minute) bars, PERFORMANCE-BASED model
selection by NET Sharpe, a hash-verified registry and point-in-time `PredictionSet`s. Independent of every AI
runtime (it imports no model adapter), reuses this package's splits/purging/embargo (`splits.plan_walk_forward`),
`CostModel`, calibrators (`ml.fit_calibrator`), Benjamini-Hochberg (`stats.benjamini_hochberg`) and the frozen-test
registry (`ml.FrozenTest`). NO EDGE IS CLAIMED; results on synthetic data are not evidence of edge.

```bash
uv run --frozen python -m services.research.cli train-models --dataset ds_1c99c07583eae908 --source store
uv run --frozen python -m services.research.cli train-models --dataset ds_2fa8b3644cc15c28 --source store \
    --timeframe 1Min --symbols SPY,AIG,BAC,IBM --start 2013-10-01T00:00:00Z --end 2013-10-12T00:00:00Z
uv run --frozen python -m services.research.cli predict --dataset ds_1c99c07583eae908 --asof 2026-11-04T18:00:00Z
```

`train-models` options: `--folds`, `--start/--end` (bar window), `--no-quotes`, `--allow-final-test-reuse`
(CONTAMINATED re-evaluation: reported, never confirms a model). `predict`: `--registry <ver>`,
`--signal-horizon-minutes` (default 60). Exit 2 = fail closed (no/invalid registry, bad input).

**Targets (`targets.py`).** Decision time = the latest `available_at_ts` of the bars sharing a bar slot (the
cross-section is complete; strictly increasing across slots). Entry at the open of the first bar starting at/after
the decision, exit H bars later at the open, same regular session. `ret_{5,15,60}m` = gross open-to-open bps minus
the `CostModel` round trip ($10k reference notional, decision-time close, trailing completed-session ADV, last
quote spread published by the decision when quotes are stored, else the liquidity-bucket estimate);
`success` = net 60-minute return > 0; `rank` = cross-sectional percentile of the 60-minute net return per decision
time (lambdarank relevance 0..4); vol = realised variance of the next 60 minutes; cost = one-way arrival shortfall
proxy (half quoted spread at execution + fees + next-open vs decision-close) - SIMULATED on synthetic data. Rows
with unknown liquidity or incomplete features/labels are dropped, never filled.

**Models.** `lgbm_reg_{5,15,60}` (LightGBM regression on net bps), `ridge_{5,15,60}` (scikit-learn Ridge on
standardised features), zero-forecast baseline; `lgbm_rank` (lambdarank, groups = decision timestamps) vs the
deterministic `momentum_rank` t-stat; `catboost_clf` + isotonic AND sigmoid (Platt) calibrators fitted on the
held-out calibration block of every training window, calibrator chosen by validation Brier, vs climatology;
volatility `har_rv` (OLS on daily/weekly/monthly RV components; literal one-session base period when >= 32
sessions, else an explicit FALLBACK to 60-minute periods), `ewma` (RiskMetrics, lambda 0.94 configurable) and
blends (w = 0.25/0.5/0.75) chosen by development QLIKE (vol RMSE reported); `markov_2state` (statsmodels
`MarkovRegression`, switching variance, FILTERED probabilities only, states mapped CALM/TURBULENT by variance) on
the equal-weighted 5-minute market proxy of the training symbols vs the unconditional variance (QLIKE of the
next-block second moment); execution cost `spread_baseline` (half spread + fees) vs `cost_model_full` vs
`spread_plus_lgbm_residual` (LightGBM on spread/vol/participation/ADV/time-of-day predicting slippage bps), by MAE.
Every candidate has a stable `model_id` and `model_version` = sha256(model id, params, model-suite source).

**Walk-forward and selection (`walkforward.py`).** Frozen final test = last 20 % of decision times; 4 expanding
folds on the development period; purge of every training row whose LONGEST label overlaps the test window;
embargo = 60 decision steps; calibration hold-out = last 25 % of each training window (purged against it).
For each return horizon {lgbm_reg, ridge, zero} and for ranking {lgbm_rank, momentum, zero} are scored by the NET
Sharpe of the long-only one-position strategy they imply (`strategy.py`: top score at each decision time, enter
iff score > 0 - rankers always hold the top-ranked symbol -, hold for the horizon, no overlap; annualisation:
mean/stdev of per-trade net returns x sqrt(trades per year), stated in every report). Each candidate's one-sided
t-test "mean net return > 0" enters ONE Benjamini-Hochberg family (FDR 0.10). Eligible = survives BH, net Sharpe >
0, >= 10 trades, fitted in every fold; otherwise the target selects the zero baseline and the signal adapter
returns 0 -> NO_TRADE. Final models are refitted on the development period only. The frozen test is evaluated
exactly once per dataset interval (registry `experiments/models/frozen_test_registry.json`); a selected trading
model must also have net Sharpe > 0 there (first look only - a refused or CONTAMINATED look demotes it to the
baseline). `probability_calibrated` is True only for a held-out calibrator whose Brier on the once-evaluated
frozen test is <= climatology. Reports: net return, net Sharpe, turnover (trades/session), hit rate, IC, Brier,
n_test, per-fold logs, BH details.

**Registry (`registry.py`).** `experiments/models/<registry_version>/{registry.json, artifacts/}` (LightGBM text,
CatBoost `.cbm`, JSON for Ridge/HAR/EWMA/Markov/calibrators; nothing is unpickled). `registry.json` (written last)
holds dataset version, code hashes, config, selections, all metrics, calibration + reliability, walk-forward
parameters, frozen-test usage, `is_simulated`, artifact sha256s and its own canonical sha256. `load_registry`
returns the NEWEST complete registry or `None` when it is missing, unparsable, tampered (self-hash or any artifact
hash), internally inconsistent (deployed model != recorded selection, calibrator method) or trained with different
inference code (`config.INFERENCE_SOURCES`) - an older registry is never silently substituted.

**Predictions (`predict.py`).** `PredictionService(registry, cost_model)` drops every row published after `asof`
before validating anything; a symbol is absent from the result when stale (> 4 minutes), too short, illiquid,
suspect (|1-bar ln r| > 0.25 in the lookback) or of another timeframe. `expected_return_bps` is the selected
model's NET forecast (0 for the zero baseline). The regime uses the TRAINING proxy symbols and is UNKNOWN (with a
note) when one is missing/stale. Also a `signals.SignalModel` (`predict_latest`): `QuantSignalEngine(cost_model,
model=svc)` gets net forecast + round-trip spread/fees/fixed slippage at the bucket spread (the engine charges its
own costs again incl. impact).

| Name | Signature |
|---|---|
| `services.research.models.walkforward.train_suite` | `(bars_df, cost_model, base_dir, *, config=None, quotes=None, allow_final_test_reuse=False, data_adjustments=None, segment_policy=SegmentPolicy(), data_note="", run_id=None) -> TrainResult` (`base_dir` = `experiments/models`) |
| `services.research.models.registry.save_registry` | `(result, base_dir) -> ModelRegistry` |
| `services.research.models.registry.load_registry` | `(experiments_dir, registry_version=None) -> ModelRegistry | None` (looks in `<experiments_dir>/models`) |
| `ModelRegistry` | `.registry_version`, `.report()` (= `GET /api/v1/models`), `.selected(target)`, `.meta`, `.fitted`, `.is_simulated`, `.dataset_version`, `.probability_calibrated` |
| `services.research.models.predict.PredictionService` | `(registry, cost_model, *, signal_horizon_minutes=60, max_data_age=None)`; `.predict(bars_df, asof, universe, *, quote_spreads_bps=None) -> list[PredictionSet]`; `.predict_latest(sym_bars)`; `.registry_version`, `.model_version`, `.strategy_id`, `.horizon_bars`, `.calibration_evaluated` |
| `services.research.models.config.SuiteConfig` | all parameters (`describe()` is hashed into the registry) |

Model-suite limitations: on the 20-session synthetic fixture every trading target selects the zero baseline
(expected: a random walk); the LEAN minute sample has only 6 sessions (5 usable), far too little for any claim; the
cost-label "realised slippage" is a bar-based proxy (no broker fills exist; SIMULATED on synthetic data); the
self-hash is integrity, not a signature; re-running `train-models` on an already evaluated dataset yields a
baseline-only registry (fail closed) which becomes the newest one; the regime needs bars of every training proxy
symbol in the frame passed to `predict`; HAR's literal session base needs >= 32 sessions of history at training
AND 22 sessions in the prediction frame (else no vol forecast is served).

## Tests

```bash
uv run --frozen pytest tests/unit/research tests/integration/research tests/failure_injection/research -q
uv run --frozen pytest tests/unit/research/test_research_models_*.py tests/integration/research/test_research_models_*.py -q
```

Unit tests use deterministic synthetic helpers (`tests/unit/research/bar_factory.py`); integration tests run
the real isolated Qlib interpreter and the real `MarketDataStore` + synthetic fixture in a temp dir (skipped if
unavailable). No test touches the network.

## Limitations (honest)

- Results on the bundled synthetic fixtures are SIMULATED and say nothing about real markets; results on the LEAN
  sample (real, old, vendor-sample data with holes) are benchmarks under a stated cost model, not evidence of edge.
- Bars carry no quotes, so backtest spreads are liquidity-bucket estimates; no partial fills, queue position,
  latency or borrow. Dividends are not reinvested; renames/delistings are reported, not applied.
- The clean-segment rule also cuts at a genuine one-day move beyond ×2.7 / −63 % (reported in `data_quality`);
  dropping such a period can bias results, so inspect the report before reading anything into a run.
- Split adjustment makes pre-split share counts "adjusted shares", which only affects per-share fee terms
  (FINRA TAF; conservative) and whole-share rounding.
- Regulatory fee defaults are dated; verify current SEC Section 31 and FINRA TAF rates.
- Qlib IC with a handful of instruments is extremely noisy (many dates have constant predictions after early
  stopping); Qlib runs a single train/valid/test split, not the full purged walk-forward (that is `ml.py`).
- The frozen-test registry is per experiments base directory; pointing `--out` elsewhere starts a new registry.
- The ML cost diagnostics quantise ADV to 3 significant figures for memoisation.

## Results on stored data (2026-11-06, code `src-sha256:` see each manifest)

Run with `--datasets-dir <scratch store>` holding the LEAN sample import (`ds_e894fdc1b4893649`, real 1998-2021
daily vendor sample, RAW + PIT splits/dividends) and the synthetic fixture (`ds_acc78b861e2ec1fc`, SIMULATED).
NO EDGE CLAIMED; numbers are benchmarks under the stated cost model.

- `baselines` LEAN (SPY,QQQ,IWM,EEM,AAPL,IBM,BAC,AIG): 14 split events applied; EEM cut before 2003-04-11,
  QQQ before 2011-03-23 (holes). Buy-and-hold SPY total return 3.76 (CAGR 6.9 %, Sharpe 0.48), median across
  the 8 symbols 4.02; momentum median 1.26, filtered momentum median 1.62. Synthetic 1Min: all baselines
  except buy-and-hold lose money after costs (as expected on a random walk).
- `signals` LEAN asof 2021-03-31T21:00Z: candidate IBM (net 43.7 bps, heuristic, uncalibrated). Synthetic asof
  2026-11-04T19:00Z: candidate SPY (net 6.8 bps); asof 20:40Z: NO_TRADE (horizon would cross the close).
- `walkforward`: no variant passes Benjamini-Hochberg (q ~ 0.58 LEAN, ~ 0.49 synthetic); the frozen tests had
  already been evaluated once by `wf_01M6RT4CR900K90DFX30D3D7R2` / `wf_01M6RT68SP4B2BQZDTY8QV9XB2`, so the new
  runs were correctly REFUSED_ALREADY_EVALUATED (development results byte-identical to those runs).
- `qlib` LEAN: OK with pyqlib 0.9.8 (best_iteration 5, test Rank IC -0.013, pooled -0.031); synthetic daily
  fixture has only 20 sessions -> SKIPPED.

### Model suite runs (2026-11-08 UTC, `experiments/models/`)

Run with the repository store (`datasets/`), default `SuiteConfig` (4 folds, embargo 60 min, BH FDR 0.10).
Each frozen test has now been evaluated ONCE (`experiments/models/frozen_test_registry.json`); re-running on the
same datasets yields baseline-only registries by design. NO EDGE CLAIMED.

- `mreg_20261108T035209Z_032be2a9aee7` - LEAN sample `ds_2fa8b3644cc15c28` (real 2013 vendor minute sample,
  SPY/AIG/BAC/IBM, 2013-10-04..10-11, 6 558 rows, stored LEAN quotes): every trading target = `zero_baseline`
  (no BH survivor; `ridge_60` dev net Sharpe 19.3 on only 5 trades and `lgbm_rank` 10.3 with p = 0.19 rejected),
  success = climatology, vol = `ewma` (HAR fitted in < 2 folds: FALLBACK periods), regime = `markov_2state`,
  cost = `spread_baseline`. `predict --asof 2013-10-11T17:00:00Z`: 4 PredictionSets, regime CALM, engine NO_TRADE.
- `mreg_20261108T035222Z_73806d805978` - synthetic fixture `ds_1c99c07583eae908` (SIMULATED, 20 sessions,
  SPY/QQQ/AAPL/MSFT, 24 928 rows, stored synthetic quotes): every trading target = `zero_baseline` (dev net Sharpe
  lgbm_reg_60 -7.2, ridge_60 -6.7, lgbm_rank +3.1 with p = 0.30), success = climatology (dev Brier raw 0.287 ->
  isotonic 0.258, still worse than climatology 0.252; `probability_calibrated` false), vol = `blend_0.50`, regime =
  `markov_2state`, cost = `spread_baseline`. This is the NEWEST registry (what `GET /api/v1/models` serves).
  `predict --asof 2026-11-04T18:00:00Z`: 4 PredictionSets (expected returns 0), regime CALM, engine NO_TRADE.

## Autonomous research service

The [autonomous service](autonomous/README.md) adds persistent campaigns, twelve learner families, isolated Metal-capable workers, independent one-use confirmation and reversible PAPER-only model promotion. It preserves the existing suite and execution boundary. See its README for tested capabilities and explicit limitations.

### Local Kronos challenger

See [Kronos setup, paired benchmarks and LEAN replay](kronos/README.md). Kronos-base runs locally on MPS/CPU,
with optional Kronos-small, sampled OHLCV paths, feature generation, and bounded fold-local token-head adaptation.
Research entries are recorded separately from deployed registry versions and cannot bypass promotion gates.
