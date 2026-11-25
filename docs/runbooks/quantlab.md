# Multi-asset research runbook (quantlab)

Everything below runs locally, uses only free Alpaca endpoints through a GET-only client (150 requests/minute,
leaving headroom for the running paper API on the account's 200/minute limit) and places no orders.

## First build (≈1.5 hours of downloads, once)

```bash
uv run --frozen python -m services.capabilities probe          # capability snapshot + full asset list
uv run --frozen python -m services.quantlab.cli fetch-equities # daily SIP raw (all) + adjusted (ever-liquid)
uv run --frozen python -m services.quantlab.cli fetch-crypto   # 1Hour + 1Day bars, tradable USD pairs
uv run --frozen python -m services.quantlab.cli crypto-spreads # top-of-book spread snapshot
uv run --frozen python -m services.quantlab.cli build-panel    # point-in-time panels (npz + json meta)
uv run --frozen python -m services.quantlab.cli orb-fetch      # 5-minute bars for PIT stocks-in-play
```

Downloads are resumable (per-batch manifests). A dataset directory is immutable: changing its parameters raises.

## Research

```bash
uv run --frozen python -m services.quantlab.cli research --asset equity   # A, A2, C1, C2, C3
uv run --frozen python -m services.quantlab.cli research --asset orb      # B
uv run --frozen python -m services.quantlab.cli research --asset crypto   # D1, D2
uv run --frozen python -m services.quantlab.cli research --asset session  # H1-H3 intraday, I1 overnight, I2 TOM
uv run --frozen python -m services.quantlab.cli research --asset xsrank   # K: learning-to-rank x 1/5/20 x cost-aware
uv run --frozen python -m services.quantlab.cli holdout --family A2_etf_rotation   # ONE look, DEV_PASSED only
```

Every variant is appended to `experiments/quantlab/trials.sqlite3`; re-running a family adds trials, which
raises its deflated-Sharpe bar. Holdout looks are permanent (`holdout.sqlite3`): one per spec, three per family.

Hypothesis memory: a family whose every variant already has a trial on the same data fingerprint and code is
skipped (`skipped_no_new_evidence` in the run summary; `--force` overrides and adds trials). An agent factor spec
already evaluated on the same development data is refused with `DUPLICATE_HYPOTHESIS` and not counted against the
hourly budget; a corrected or extended development panel is new evidence. A delta that only adds rows after the
protected holdout start is not. An agent factor whose mean rank correlation with an already-searched signal or a
recent agent factor reaches |0.9| is refused with `REDUNDANT_HYPOTHESIS` (no trial; the response names the nearest
signal); accepted experiments report their `novelty`.

## Learning-to-rank and cost-aware selection (family K)

```bash
make quantlab-xsrank                                          # = research --asset xsrank (≈ 1 hour, CatBoost dominates)
uv run --frozen python -m services.quantlab.cli xsrank-diagnostics   # re-derive diagnostics only (no trials)
```

Five models × horizons 1/5/20 × {top_n, cost_aware} = 30 pre-registered variants on the liquid-1000 universe,
predicting from 2018 (purged walk-forward per calendar year), trading from 2019, walk-forward selection from 2020.
XGBoost comes from the `advanced-research` extra; without it those variants are recorded as `ERROR` trials. The run
also writes `experiments/quantlab/xs_rank_diagnostics.json`: rank IC (non-overlapping dates), decile spreads of the
executable excess return vs the top decile's round-trip cost, ranking-objective vs regression paired IC tests,
edge calibration and conformal coverage per year, every variant's 2020+ metrics, and SPY / equal-weight universe /
sector-ETF momentum / cash over the same period. Re-running records 30 more trials (raising the DSR bar).

## Session-timing research (5-minute SIP, allowlisted ETFs and large caps)

```bash
make quantlab-session   # = session-fetch, build-panel --what session, session-profile, research --asset session
```

`session-fetch` downloads split/dividend-adjusted 5-minute SIP bars for the 24 allowlisted symbols from 2016-01-04
to the day before the holdout (the end date is clamped in code; the holdout year is never downloaded by research),
resumable per (year, symbol batch), 4 threads under the shared 150/minute limit. `session-profile` writes a
descriptive time-of-day return table (`experiments/quantlab/session_profile.json`; nothing is selected from it).
The simulator mirrors the runtime lifecycle: marketable LIMIT entries with non-fills, no entry in the last 30
minutes, flatten 10 minutes before the close (early closes included), a protective stop on every position, the
runtime cost model once per leg. I1/I2 hold overnight and therefore need the operator's overnight authorization
in the risk policy before they could ever trade.

## Daily operation (after the close)

```bash
uv run --frozen python -m services.quantlab.cli fetch-delta && \
uv run --frozen python -m services.quantlab.cli build-panel --what equity && \
uv run --frozen python -m services.quantlab.cli scan && \
uv run --frozen python -m services.quantlab.cli shadow
uv run --frozen python -m services.quantlab.cli options-scan   # during market hours (indicative quotes)
uv run --frozen python -m services.quantlab.cli crypto-scan
```

`scan` writes the point-in-time scan and the cross-asset selection (TRADE/NO_TRADE with reasons). `shadow`
journals target portfolios of SHADOW/PAPER strategies and settles earlier entries; PAPER promotion is checked
automatically and refused while any blocker remains (shadow sessions, tracking, capability, lifecycle).

## Where results appear

* `GET /api/v1/quantlab/{capabilities|research|opportunities|strategies|trials|options}`; the research detail of
  `K_xs_rank` (`?family=K_xs_rank` / `get_research_results`) adds a compact `diagnostics` digest.
* MCP tools `get_capabilities`, `get_research_results`, `get_opportunities`; agents can submit a factor with
  `run_research_experiment` (development data only, 6 per hour, recorded).
* Reports: `docs/verification/multi-asset-research-2026-11-23.md`,
  `docs/verification/session-timing-research-2026-11-24.md`.
