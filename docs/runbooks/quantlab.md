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
uv run --frozen python -m services.quantlab.cli holdout --family A2_etf_rotation   # ONE look, DEV_PASSED only
```

Every variant is appended to `experiments/quantlab/trials.sqlite3`; re-running a family adds trials, which
raises its deflated-Sharpe bar. Holdout looks are permanent (`holdout.sqlite3`): one per spec, three per family.

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

* `GET /api/v1/quantlab/{capabilities|research|opportunities|strategies|trials|options}` and the dashboard's
  "Multi-asset research" panel (read-only).
* MCP tools `get_capabilities`, `get_research_results`, `get_opportunities`; agents can submit a factor with
  `run_research_experiment` (development data only, 6 per hour, recorded).
* Reports: `docs/verification/multi-asset-research-2026-11-23.md`.
