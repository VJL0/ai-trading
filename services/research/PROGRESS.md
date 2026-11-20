# services/research PROGRESS

Attempt 3 (2026-11-06). Inventory at start: costs.py, frames.py, provenance.py, baselines.py, signals.py,
features.py, splits.py, stats.py, synthetic.py present but UNTESTED; ml.py, cli.py, README, tests missing.
qlib_env/.venv exists with pyqlib 0.9.8 from git tag v0.9.8 (commit 54355232, verified via git ls-remote;
PyPI latest is 0.9.7) - imports OK.
1. costs.py CostModel/estimate/round-trip/spread_bps_from_quote: TESTED (pytest tests/unit/research/test_research_costs.py 13 passed).
2. baselines.py (buy_and_hold, momentum, intraday_mean_reversion, filtered_momentum; event-driven PIT backtester; run_all_baselines JSON): TESTED (tests/unit/research/test_research_baselines.py 17 passed).
3. signals.py QuantSignalEngine: TESTED (tests/unit/research/test_research_signals.py 13 passed). Left: integration test persist_signals.
4. ml.py (walk-forward LightGBM/CatBoost clf+reg + baseline rule, isotonic/sigmoid calibration, Brier/reliability, BH FDR, frozen final test registry, TrainedSignalModel/load_trained_model) + splits.py hardening: TESTED (test_research_ml.py 14 passed, test_research_splits.py 6 passed incl. hypothesis). Defect fixed: ML cost diagnostics used whole-panel medians (frozen-test leak) -> per-sample PIT costs.
5. Qlib: qlib_env (pyqlib 0.9.8 from git tag v0.9.8, uv sync --locked OK), qlib_bridge.py (bin converter, segments, subprocess), qlib_env/lat_qlib_workflow.py (DataHandlerLP+LGBModel+SignalRecord/SigAnaRecord): TESTED (unit test_research_qlib_bridge.py 6 passed; integration test_research_qlib_workflow.py 2 passed, real qlib subprocess).
6. cli.py (datasets/baselines/signals/walkforward/qlib; lazy MarketDataStore; --source auto|store|synthetic; --datasets-dir): TESTED (unit test_research_cli.py 6 passed with fake store; integration test_research_market_data_store.py 3 passed with REAL MarketDataStore + ensure_synthetic_fixture in tmp dir). Left: README, full gate, self-review.
Attempt 4 (2026-11-06): re-ran gate (88 passed, 1 ruff E501, 27 pyright errors in tests only). Self-review fixes:
- signals: candidate = highest-ranked signal passing ALL checks (a stale/illiquid top rank no longer blocks a fresh one); test added.
- baselines: intraday mean reversion entries expire at session end (no next-session fill after a data gap); test added.
- ml: frozen-test registry also refuses overlapping interval on same dataset_version (changing H/folds no longer buys a fresh look); test added.
- costs: explicit non-finite Decimal rejection; test added. Tautological no-leverage assertion in baselines test replaced.
- tests: pyright clean (typed ts/to_dt helpers). Gate: ruff OK, pyright 0 errors (src+tests), pytest 95 passed.
Left: README update, CLI run on stored fixture, final review.
- REAL-DATA defects found running the CLI on the LEAN sample (adjustment=raw): splits (AAPL 7:1, AIG 1:20 reverse) corrupted
  backtests; QQQ has a 2004-11..2011-03 hole; EEM pre-2003-04 is a different security (10x one-bar jump). Fixed:
  adjustments.py (PIT split adjustment + split-adjusted dividends, asof filter), dividend cash credit in run_backtest,
  frames.last_clean_segment (cut at >5 missing XNYS sessions or |ln r|>1, keep last segment; default on in baselines/ML/Qlib),
  signal-engine guards (|1-bar ln r|>0.25 or >1 missing session in lookback -> blocker). CLI applies store corporate actions.
  Tests: test_research_adjustments.py 7, test_research_data_quality.py 6. Gate: ruff OK, pyright 0, pytest 111 passed.
Attempt 5 (2026-11-06): gate re-run at start (ruff OK, pyright 0, pytest 111 passed). Self-review fixes:
- signals: intraday candidate blocked when asof+horizon > scheduled XNYS close; calibrated label requires model.calibration_evaluated;
  only universe symbols validated (unrelated bad row no longer blocks). Tests added (signals 21).
- ml: frozen-test check-and-reserve under fcntl lock, entry written STARTED before evaluation (crash/concurrency counted). Tests added.
- qlib_bridge: run_dir recorded in SKIPPED/BLOCKED manifests.
- CLI run on scratch store (LEAN sample ds_e894fdc1b4893649 + synthetic ds_acc78b861e2ec1fc): baselines/signals/walkforward/qlib all exit 0;
  walkforward correctly REFUSED (frozen tests evaluated in attempt 4). README results section added.
Gate: ruff OK, pyright 0 errors, pytest 115 passed. Left: nothing in scope; e2e api tests blocked by missing services.ai.pipeline (other agent).
