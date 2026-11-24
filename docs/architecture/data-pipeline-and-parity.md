# Historical data pipeline, research snapshots and research/runtime parity

Phases 2 and 3 of the [roadmap](../roadmap/README.md).

## Phase 2: data pipeline

```
make data-pipeline        # = python -m services.market_data.cli pipeline   (real Alpaca, bounded per run)
make data-snapshots       # list research snapshots;  ... snapshot verify --id rs_...
make compare-feeds        # IEX vs historical SIP, last 5 sessions (read-only, free plan)
LAT_DATA_PIPELINE_ENABLED=1 make dev   # also run it in the API process after each close (default off)
```

`services/market_data/pipeline.py` (`HistoryPipeline`):

1. **Plan** per symbol of the eligible universe (default: the active policy allowlist, 24 stocks/ETFs) from what is
   stored for the feed: *increment* windows from the last stored bar, *initial*/*history* windows back to
   `history_days` (730). Only whole sessions that closed at least `settle` (20 min) ago are requested, never a forming
   one, which also keeps historical SIP requests outside Alpaca's most-recent-15-minutes restriction. Newest first,
   chunked (10 sessions), at most `max_requests` per run (rate-limit budget); a 429 ends the run, the next one resumes.
2. **Validate and store** every window as its own incremental, content-addressed dataset through the existing quality
   checks (duplicates, out-of-order, OHLC, outliers, look-ahead; gaps reported, never filled), with one
   `md_ingestion_runs` row and `data_quality_events` per window.
3. **Snapshot** (`services/market_data/research_snapshots.py`): an immutable, cumulative research history defined by
   *constituents* (exact dataset versions: one feed, one adjustment, no defects, no synthetic/real or IEX/SIP mixing,
   only datasets registered by the cutoff) and a *knowledge cutoff* (`available_at_ts <= cutoff`, latest revision
   known at the cutoff wins). It stores lineage, per-symbol coverage (sessions, missing sessions, incomplete sessions,
   revised bars, stale tail) and a SHA-256 over every resolved bar. `snapshot_id = rs_ + sha256(definition)[:16]`;
   `verify_snapshot` recomputes everything. The store resolves a snapshot id wherever it accepts `dataset_version`
   (bars, quotes, corporate actions), so research reads a snapshot exactly like a dataset.
4. **Research trigger**: the snapshot becomes research's input (`research_state.json`, `pick_history`) only when it
   adds at least `min_new_sessions` (5) complete sessions or new symbols over the previous ready snapshot.

### Evidence (2026-11-20, real Alpaca data, read-only)

* First pipeline pass: 200 of 1,024 planned windows, 618,514 bars accepted, 4,149 rejected (kept raw), coverage
  extended from 4 to all 24 eligible symbols (recent history first; the older history fills in over the next runs).
  Snapshot `rs_95135dab40bccd81` marked ready.
* IEX vs SIP, 2026-11-13..19 ([report](../verification/feed-compare-2026-11-20.json)): IEX minute coverage
  0.97–1.00 for SPY/QQQ/AAPL/MSFT/XLF/KO, IEX share of consolidated volume 1.3–8.6 %, median |close diff| 0.4–1.1
  bps (p95 2.3–5.8 bps), 1-minute return correlation 0.90–0.98, realized-volatility ratio 0.98–1.01.
  Reading: for these liquid names IEX minute *prices* track the consolidated tape to about a basis point, so price
  accuracy alone does not justify a paid real-time SIP plan; IEX *volume* is a small, varying slice, so volume-based
  features must come from the same feed in research and runtime. Not evidence of any strategy edge. No purchase was
  made or enabled.

## Phase 3: one prediction contract, costs charged once

`services/research/decision.py` is the shared contract. Every forecast declares its basis: `GROSS` (deterministic
baseline) or `NET` (models trained on `net = gross - round trip at reference_notional`). `decide_edge` charges the
live round trip exactly once (`NET`: re-price the training cost convention with the live inputs, then subtract the
live cost) and applies an explicit `uncertainty_bps` buffer and an explicit threshold.

Defects fixed:

* The autonomous paper model returned its NET forecast (minus the validation allowance) as if it were gross; the
  signal engine then subtracted the round trip a second time. It now returns `basis="NET"` with the allowance as an
  explicit uncertainty buffer (`QuantSignal.uncertainty_bps`).
* The suite adapter added back only spread + fees + fixed slippage, so the impact term was charged twice. It now
  returns `basis="NET"` with its `reference_notional`.

`QuantSignal.net_edge_bps` = gross − live round trip − uncertainty buffer: the number the threshold (5 bps) applies to.
The position manager uses the signal's `horizon_minutes` for time exits, so the 5/15/60-minute horizon a model was
trained for is the horizon the position is held.

Parity tests (`tests/unit/research/test_decision_parity.py`): a fitted LightGBM 60-minute model applied to the
training panel's feature rows equals `PredictionService.predict` on point-in-time truncated bars, and the engine's net
edge equals the research net forecast when the order is sized at the reference notional. Remaining, documented
differences: live quote vs last quote published by the decision time; live order notional vs `reference_notional`
(only impact moves). Open item for Phase 5: walk-forward model selection still evaluates scores at a 0 bps threshold
(it takes more trades than the runtime's 5 bps-after-costs-and-buffer rule), so its statistics describe the model, not
the exact runtime strategy; the full-lifecycle backtest must apply `decide_edge` with the runtime threshold.
