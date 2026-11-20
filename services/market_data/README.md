# services/market_data

Market data for the local paper-trading workstation: Alpaca ingestion (IEX free / SIP paid, PAPER keys only), a
deterministic **SYNTHETIC** fixture, an importer for the **real** sample data bundled in the pinned LEAN clone, a
LEAN-format exporter, point-in-time validation, Parquet + DuckDB storage with provenance manifests, a strict
point-in-time replay provider, an ingestion service that backfills before entries resume, snapshots and
universe/eligibility snapshots. All money/price/size values are `Decimal`; all timestamps are timezone-aware UTC.

## CLI

```
uv run --frozen python -m services.market_data.cli [--datasets-dir DIR] <command>
  fixture [--symbols ..] [--sessions N] [--seed S] [--end-date YYYY-MM-DD] [--inject-defects]
  import-lean-sample [--lean-root trading/lean/Lean] [--symbols ..] [--no-quotes] [--no-db | --db PATH]
  export-lean --dataset <ver> --out trading/lean/data-local [--no-quotes] [--no-metadata]
  validate --dataset <ver> [--no-quotes] [--record]
  ingest-alpaca --symbols SPY QQQ [--feed-mode IEX_FREE|SIP_PAID] [--days 3] [--no-db]
  universe [--asof ISO] [--candidates ..] [--feed F] [--no-db]
  status
```

JSON on stdout; exit 0 ok, 1 validation/integrity failure, 2 configuration problem (e.g. missing keys: the message
explains the Keychain / `.env` names and points to `LAT_FEED_MODE=SIMULATED`). Secrets are never printed, only their
presence. `--no-db` keeps the run from creating/migrating the operational SQLite ledger (data-quality findings then
live only in the manifest).

Datasets currently in `datasets/` (generated on this Mac, 2026-11-06):

| dataset_version | feed | content |
|---|---|---|
| `ds_1c99c07583eae908` | SYNTHETIC | SPY/QQQ/AAPL/MSFT, 20 XNYS sessions to 2026-11-04, 31,200 minute + 80 daily bars, 374,400 quotes (5 s). **SYNTHETIC — not market data** |
| `ds_2fa8b3644cc15c28` | LEAN_SAMPLE | LEAN repo sample at commit `d1c09284c438…`: 70,025 daily + 15,278 regular-session minute bars, 14,977 quotes, 876 corporate actions; old data (daily 1998–2021, minute days 2008/2013/2014/2015/2023) |

## Layout

```
datasets/raw/<FEED>/{bars,quotes,trades}/symbol=<S>/date=<YYYY-MM-DD>/part.parquet   raw events incl. rejected ones
datasets/raw/<FEED>/bars_daily/symbol=<S>/year=<YYYY>/part.parquet                    raw daily bars (yearly files)
datasets/raw/<FEED>/corporate_actions/symbol=<S>/part.parquet
datasets/analytics.duckdb      bars, quotes, trades, corporate_actions (DECIMAL(18,6)/(24,6)), datasets registry,
                               active_series, universe_snapshots
datasets/processed/manifests/<dataset_version>.json
```

Raw rows carry `quality_status` (ACCEPTED/FLAGGED/REJECTED), reasons and the verbatim payload (`raw_json`). Normalized
tables hold accepted rows only. `dataset_version = "ds_" + sha256(canonical manifest)[:16]`; it hashes each
partition's `content_sha256` (this dataset's own rows), so identical content yields the same version regardless of
what else is stored. Manifests list source_id, feed, symbols, start/end, row counts, per-file sha256 at write time,
generator/importer + version, LEAN commit and vintage, created_at, is_synthetic, limitations and validation summary.

## Public API (import paths used by services/api and services/research)

```python
from services.market_data.store import MarketDataStore            # MarketDataStore(datasets_dir)
store.write_bars(bars, *, manifest)  /  write_quotes / write_trades / write_dataset(spec, bars=, quotes=, trades=, corporate_actions=)
store.load_bars(symbols, start, end, timeframe, feed=None, asof=None, *, dataset_version=None, sessions=None) -> list[Bar]
store.load_bars_df(..., prices="float"|"decimal") -> DataFrame[BAR_DF_COLUMNS]   # float64 prices: research only
store.load_quotes(...), latest_quote(symbol, asof), load_trades(...), latest_trade(symbol, asof)
store.load_corporate_actions(symbols, *, asof=None, feed=None, dataset_version=None)
store.latest_manifest(feed=None), list_manifests(feed=None, include_defective=True), get_manifest(ver), verify_dataset(ver)

from services.market_data.fixtures import ensure_synthetic_fixture  # (datasets_dir, *, symbols, sessions=20, seed=20261105, end_date=None, inject_defects=False) -> manifest
from services.market_data.lean_sample import import_lean_sample      # (datasets_dir, *, lean_root=None, symbols=None, db=None) -> manifest
from services.market_data.lean_export import export_bars_to_lean     # (bars, lean_data_dir, *, quotes=(), copy_metadata_from=None) -> summary
from services.market_data.quality import validate_bars, validate_quotes, validate_trades, QualityPolicy, record_quality_events, FeedQualityMonitor
from services.market_data.alpaca_provider import AlpacaMarketDataProvider, MissingCredentials   # (secret_store, feed_mode, clock)
from services.market_data.replay import ReplayMarketDataProvider     # (store, clock, feed, feed_mode, *, dataset_version=None)
from services.market_data.ingestion import IngestionService, IngestionState   # (provider, store, db, clock, quality_monitor, *, symbols=None)
from services.market_data.snapshot import build_market_snapshot      # await (provider, symbol, asof, feed_mode, *, dataset_version)
from services.market_data.universe import build_universe_snapshot, persist_universe_snapshot, eligible_at
```

`FeedQualityMonitor(feed_mode, provider, clock, symbols=())` — `observe_quote/observe_bar/observe_trade`,
`set_connection_state`, `status()` returns the `/status` `feed` object and the `/market/quality` body (minus
`events`) of `docs/architecture/api-contract.md`, with `limitations` (IEX_FREE: *"Real-time quotes/trades from IEX only
(~2-3% of consolidated volume); NBBO not guaranteed"*; SIMULATED: *"Synthetic data — not market prices"*). Status is
non-OK (fail closed) until every tracked symbol has a quote younger than 15 s, while DISCONNECTED/BACKFILLING, and when
quote timestamps are ahead of the local clock by more than 2 s.

## Point-in-time rules

* Every load takes `asof`; only rows with `available_at_ts <= asof` are returned; among revisions of a bar the latest
  one available at `asof` wins. Bars are available at `end_ts + latency` (fixture 1 s, Alpaca 2 s, LEAN minute 2 s,
  daily 15 min); live stream bars never before receipt; quotes/trades at receipt.
* Feeds are never spliced per symbol (priority SIP > IEX > LEAN_SAMPLE > SYNTHETIC unless `feed=` is given), and a
  *snapshot* dataset (fixture/import) becomes the active series of its symbols so two fixtures are never mixed.
  Datasets with injected defects are excluded unless requested by version.
* The replay provider caps every query at `clock.now()`; `stream_quotes` yields only as the clock advances. A
  hypothesis property test checks that no returned record is ever available after `asof`.

## Validation (`QualityPolicy`: REJECT / FLAG / IGNORE per check)

Out-of-order, duplicates (identical/conflicting), calendar gaps (XNYS via exchange_calendars; reported, **never
filled**), OHLC inconsistency, zero/negative price, negative volume, outlier jumps (% intraday/overnight and robust
z-score with a 50 bps floor; suppressed on known split/dividend ex-dates), outside regular session, crossed/locked
quotes, stale quotes (receipt delay), clock skew (provider/exchange time after receipt, bar received before close) and
look-ahead (always rejected). Malformed records are rejected as INVALID_RECORD, never coerced (floats included).

## Ingestion

States `DISCONNECTED -> CONNECTING -> BACKFILLING -> LIVE`. Every (re)connect backfills each symbol from the end of
its last stored bar to now through historical bars before `LIVE`; `entries_allowed()` is true only when LIVE, the
last backfill succeeded, no backfill came back "thin" (> 50 % of expected regular-session bars missing), and the
monitor is OK. Quotes, trades and bars share one websocket (`stream_events`); alpaca-py's silent internal reconnects,
queue overflow or stream end raise `StreamDisconnected` -> DISCONNECTED -> backoff -> reconnect + backfill. A streamed
bar that leaves a hole triggers a backfill before LIVE resumes. Transitions go to `md_feed_state_transitions`,
backfills to `md_ingestion_runs` (migration `0100_market_data.sql`), findings to `data_quality_events`.

## Limitations / honesty

* **No Alpaca keys exist on this Mac**: the Alpaca provider is unit-tested only with handcrafted alpaca-py objects and
  fake clients/streams. `tests/integration/market_data/test_market_data_alpaca_paper.py` (`@pytest.mark.paper_broker`)
  fetches real IEX bars when keys exist and skips otherwise; it has never run here.
* SIP_PAID must be configured explicitly; an entitlement error raises `FeedEntitlementError` and nothing falls back,
  upgrades or purchases.
* The SYNTHETIC fixture is modelled data (seeded random walk, U-shaped volume, penny spreads, 5 s quotes so a PIT
  quote is always < 15 s old in a simulated session); it has no trades, corporate actions, halts or auctions.
* LEAN sample data is old (1998–2023), of unknown adjustment provenance (treated as raw), credited to AlgoSeek in
  LEAN's readme (local use only). Its sparse daily files produce many reported daily gaps (e.g. QQQ, FB, AAA, USO).
  `received_ts` is the LEAN commit time (vintage); a non-git LEAN root uses the import time and `lean_commit=unknown`.
* LEAN export was checked against the real engine: BasicTemplateAlgorithm (2013-10-07..11) on SPY minute bars
  imported -> stored -> re-exported produced statistics identical to LEAN's own files (all 28 lines). A LEAN backtest
  on the synthetic 2026 fixture needs an algorithm with matching dates (owned by the LEAN component).
* DuckDB connections are short-lived and retried on lock contention (`lock_timeout_s`, default 20 s) so the API and
  the CLI can share `analytics.duckdb`; a long-running writer in another process blocks readers until it finishes.
* An unreadable raw partition is quarantined (`part.parquet.corrupt-*`, kept as evidence) on the next write; a
  synthetic fixture whose raw files were falsified fails loudly (`FixtureIntegrityError`) instead of being used.
