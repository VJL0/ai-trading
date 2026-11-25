# Family K — learning-to-rank, holding horizons and cost-aware selection (2026-11-25)

Development research on real Alpaca SIP daily data (point-in-time panel incl. delisted names), development period
2016-01-04 … 2025-11-20. The protected holdout (2025-11-21 … 2026-11-20) was **not** loaded (0 vault looks).
Backtested results under the runtime cost model — not realized trading results, not evidence of future profit.

Run `ql_20261125T053958Z` (`make quantlab-xsrank`); artifacts in `experiments/quantlab/runs/ql_20261125T053958Z/`
and `experiments/quantlab/xs_rank_diagnostics.json`. 30 trials recorded (ledger total 1,436 development trials).
Wall clock ≈ 31 min (model fits 29 min, CatBoost dominant), 2,934 names ever eligible (liquid-1000 universe).

## Question

The external research brief of 2026-11-25 proposed, as the highest-priority experiments: (1) explicit ranking
objectives vs return regression, (2) 1/5/20-session horizons vs intraday, (5) cost-aware selection vs "largest
predicted return", with calibrated uncertainty and passive benchmarks. Family G already compared LightGBM and
ridge *regression* at one 21-session horizon; no daily ranking objective, calibration or cost-aware selection
existed. Family K tests all three in one pre-registered grid (see `services/quantlab/strategies/xs_rank.py`).

## Verdict: **fails the development gate; nothing promoted; selection stays `NO_TRADE`**

| Gate item | K_xs_rank | Required |
|---|---|---|
| Deflated Sharpe (30 trials) | 0.195 | ≥ 0.95 |
| Walk-forward OOS Sharpe 2020-01 … 2025-11 | 0.35 (SPY 0.76 same period) | ≥ 0.5 |
| SPA vs SPY / alpha t | p = 1.00 / −1.73 (alpha −7.7 %/yr, beta 1.05) | p ≤ 0.10 or t ≥ 2 |
| PBO | 0.47 | ≤ 0.5 |
| Sharpe at 2× costs | 0.77 | > 0 |
| Walk-forward max drawdown | −45.5 % (2021-02 → 2022-10) | ≥ −35 % |
| Benjamini-Hochberg across 16 families | q = 1.00 | reject |

Walk-forward choice per year: the cost-aware LambdaRank h=5 variant in 2020 and 2022–2025 (it held SPY in 91 % of
rebalances, so those years equal SPY), ridge top-N h=5 in 2021 (−19.6 % vs SPY +28.6 %). The family's best
variants *are* the index: the honest outcome of a rule that buys a stock only when its edge beats its costs.

## Findings

**1. Ranking objectives did not beat regression.** Paired per-date rank-IC differences (executable open→open
excess return, non-overlapping dates; t-statistics are not corrected for the 15 model × horizon cells):

| vs LightGBM regression | h = 1 | h = 5 | h = 20 |
|---|---|---|---|
| LightGBM `lambdarank` | −0.0079 (t −3.69) | −0.0082 (t −1.54) | −0.0146 (t −1.34) |
| XGBoost `rank:pairwise` | −0.0072 (t −3.18) | −0.0118 (t −1.79) | −0.0091 (t −0.88) |
| CatBoost `YetiRank` | +0.0016 (t +1.17) | +0.0061 (t +1.94) | −0.0008 (t −0.12) |
| (LightGBM reg − ridge) | −0.0021 (t −1.16) | −0.0080 (t −1.78) | +0.0035 (t +0.44) |

LambdaRank and pairwise are significantly *worse* on IC at h=1; YetiRank ties regression. LambdaRank (truncation at
the top 30) did produce the highest top-decile excess at h=5 (9.3 bps) — still below its round-trip cost.

**2. There is a small, persistent ranking signal, but it is smaller than the cost of trading it.**

| Model, horizon | Rank IC (t) | Positive years | Top-decile excess | Round-trip cost | Top − cost |
|---|---|---|---|---|---|
| ridge h5 | 0.024 (2.96) | 8/8 | +4.0 bps | 12.7 bps | −8.7 |
| CatBoost YetiRank h5 | 0.023 (3.13) | 8/8 | +6.2 | 12.7 | −6.5 |
| LightGBM reg h5 | 0.016 (3.04) | 7/8 | +7.1 | 12.8 | −5.7 |
| LightGBM reg h1 | 0.010 (3.38) | 7/8 | +0.1 | 13.0 | −13.0 |
| LightGBM reg h20 | 0.018 (1.77) | 6/8 | +17.9 | 12.7 | +5.2 (spread t 0.94) |

At h=1 the IC comes from the middle of the distribution (top-minus-bottom decile ≈ 0 bps). Only LightGBM
regression at h=20 has a top decile above its cost, and that spread is not statistically distinguishable from zero.

**3. Horizon: costs decide.** Top-N net Sharpe 2020+ (gross Sharpe in brackets, full development period):
h=1 −0.46 … +0.18 (gross up to 0.72; turnover 85–177×/yr, cost drag 8–17 %/yr); h=5 0.19–0.43 (cost drag
2–3.5 %/yr); h=20 0.42–0.66 (cost drag ≈ 0.6 %/yr). Best top-N: LightGBM regression h=20, Sharpe 0.66, CAGR
15.4 %, max DD −35.8 % vs SPY 0.76 / 14.4 % / −33.8 %. Consistent with the session-timing program: short-horizon
gross edges exist but do not survive costs.

**4. Cost-aware selection behaved as designed — and reduced to the index.** The calibrated lower bound
(edge − 1 s.e.) of the best bin was below the ≈ 13 bps round trip almost always: 13 of the 15 cost-aware variants
never bought a stock; LambdaRank bought 0.1 names on average at h=5 and 3 names at h=20.
Its value here is negative evidence: it prevented every top-N variant's cost losses (e.g. h=1) without finding
an edge. Top-bin edge estimates (h=5, LambdaRank): 5–21 bps predicted with 8–15 bps standard error; realized top
bin per year −22 … +19 bps — the calibration is not stable enough to trade on.

**5. Conformal bands are honest about single-position risk.** 80 % Mondrian bands covered 0.62–0.84 of the next
year's outcomes (2020 under-covered at ≈ 0.63: a regime break); band widths 4 % (h=1) to 23 % (h=20) per name,
two orders of magnitude wider than the edges.

**6. Passive comparators (2020-01 … 2025-11):** SPY Sharpe 0.76 (CAGR 14.4 %, DD −33.8 %); sector-ETF 6-1
momentum top 3 0.71 (14.9 %, −30.0 %); equal-weight liquid universe 0.51 (9.8 %, −40.1 %); cash 0.

## What this does and does not show

* It does not show that cross-sectional ranking is useless: IC is positive and persistent at h=5. It shows that
  with today's free daily features, a $100k book's costs, and these models, no selection rule here beats SPY.
* The model label (risk-adjusted, sector-relative) differs from the diagnostic and economic quantity (universe-
  relative executable return); the mismatch is deliberate and pre-registered, and may cost the rankers some IC.
* Re-running K adds 30 trials. The next evidence must be *new information* (fundamentals/filings/estimates with
  availability timestamps — not price features again), a longer horizon with lower turnover, or prospective data.

## Not done (and why)

* TimesFM 3.0, Toto 2.0, TabPFN 3.5, TimeH2O: not added — research-only/non-commercial weight licences, no
  incremental evidence yet, and K shows the bottleneck is cost vs edge, not model capacity.
* SEC EDGAR filings/XBRL fundamentals: proposed as a separate task (needs an operator-supplied SEC User-Agent,
  download authorization and its own point-in-time design).

## Verification

* New tests (SIMULATED data): `tests/unit/quantlab/test_ranking_calibration.py` (12),
  `tests/integration/quantlab/test_xs_rank_family.py` (9: walk-forward-only scores, future perturbation cannot
  change past scores, PIT sector groups, calibration windows end before each year, targets, prohibitive costs →
  market only, all five models fit with query groups, trial recording, strict-JSON diagnostics),
  `tests/unit/quantlab/test_novelty.py` (4), `tests/integration/quantlab/test_agent_novelty_cli.py` (1),
  `tests/unit/quantlab/test_xs_rank_views.py` (1).
* `make test-py`: 2,257 passed, 1 skipped (+ 20 torch tests in their own process). `make lint`, `make typecheck`:
  clean. No broker access, no orders, no holdout data.
