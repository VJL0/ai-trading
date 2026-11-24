# Multi-asset research verification — 2026-11-23

What was built, what was measured on real (free) Alpaca data, and what remains research-only. Every number below
is a **backtest after modelled costs** on historical data or a live research scan; none is a realized trading
result, and none is evidence of future profitability. Nothing was enabled for execution: the risk policy,
paper-only mode, compiled-off live trading and the execution stack are unchanged.

## Verdict

**No strategy family passed the pre-registered development gate, so the protected holdout was never opened and
the cross-asset engine selects `NO_TRADE`.** That is the correct outcome under the evidence, not a failure of
the system: the gate requires a deflated Sharpe ≥ 0.95 with every trial counted, a significant edge over the
passive benchmark (SPY / BTC buy-and-hold), PBO ≤ 0.5, positive Sharpe at 2× costs and a bounded drawdown.

The main finding is that, on Alpaca SIP data 2016–2025 with realistic costs, every long-only equity/ETF family
either **was market beta** (momentum: higher CAGR than SPY but beta ≈ 1.4, −58% drawdown, alpha t ≈ 0.7) or
**under-performed buy-and-hold** (rotation, reversal, pullbacks). Crypto time-series momentum produced the best
raw risk-adjusted figure (walk-forward OOS Sharpe 1.04) but had no alpha over holding BTC (t ≈ 0) and a PBO of
0.62. The ML ranker beat the ridge baseline on identical features and folds, yet not SPY.

## What was verified about Alpaca (capability probe, read-only)

* $5,000 paper margin account (multiplier 4, shorting enabled, options level 3, crypto ACTIVE tier 1).
* 13,535 tradable U.S. equities (5,761 common stocks, ~4,800 ETFs, 560 leveraged, 271 inverse, 51 volatility
  ETPs by name inference), 19,147 inactive assets, 73 crypto pairs (33 non-stable USD pairs).
* Options: equity/ETF (American) and **SPX, XSP, VIX** (European) contracts exist; **NDX/RUT none**. Market data:
  indicative only — **OPRA denied** ("agreement is not signed"); real-time SIP denied; historical SIP, IEX, crypto
  bars/order books, news and corporate actions available on the free plan.
* PDT fields are gone: FINRA replaced the pattern-day-trader rule with intraday margin standards (effective
  2026-06-04) and Alpaca deprecated the fields; the build's no-borrowing rule already bounds intraday exposure.
* Execution eligibility today: **U.S. equity long only** (LIMIT DAY, 24-symbol allowlist, 1 position ≤ 25% of
  equity). Short, margin, leveraged/inverse ETPs, crypto, options and extended hours are research-only.

## Data actually used

* Daily SIP bars 2016-01-04 … 2026-11-20: 24.1M raw rows for 20,273 symbols (active + delisted recovered from
  corporate actions), 17.7M adjusted rows for the 10,251 ever-liquid symbols → panel 2,738 sessions × 10,174
  instruments. Survivorship: delisted names with merger/name-change records are included (e.g. TWTR); bank
  failures without a corporate-action record (SIVB, FRC) are missing — residual bias favours long strategies.
* Crypto: 33 USD pairs, daily 2021-01-01 … 2026-11-22 (hourly for 19 pairs incl. BTC/ETH).
* Stocks in play: 5-minute SIP bars for the top-40 opening-gap names per session (plan built point-in-time from
  the daily panel) plus pre-open Alpaca news counts.
* Options: live indicative chains for SPY, QQQ, IWM, NVDA, TSLA, AAPL, AMD, META (≈ 18k quotes).

## Defects found and fixed during the research (all recorded; contaminated trials stay in the ledger)

1. **Instrument classification**: T-bill funds, "exchange-traded fund" names and ProShares/MicroSectors products
   were tagged as common stock (277 instruments), leaking inverse/leveraged ETFs into stock universes. Fixed,
   regression-tested, panel rebuilt, equity research re-run.
2. **Crypto relisting print**: SOL/USD was relisted on 2024-08-26 with a stray $18 open against a $157 close; a
   position carried across the gap booked +767% intraday and inflated the first crypto run (CAGR 76% → 40%
   after the fix). Crypto panels now break segments at gaps, distrust first-bar opens and force exits after 3
   missing days.
3. **Option payoff model**: a flat-vol lognormal model overstated the value of selling skewed OTM puts and the
   delayed indicative quotes were priced against a real-time stock print. Now filtered historical simulation
   (empirical tails) and the put–call-parity spot of the quotes themselves.
4. **Anthropic 20-strict-tool limit**: adding research tools pushed the in-app Claude review to 21 tools and it
   failed closed (NO_TRADE). The in-app per-candidate pipeline no longer receives the research views.
5. **ML score width** and **paginator dict pages** bugs caught by tests before any real run.

## Controlled comparisons (development period)

| Question | Result |
|---|---|
| Advanced vs simple model (G): LightGBM vs ridge, identical PIT features, purged walk-forward folds, same top-N rule | LightGBM better (mean development Sharpe 0.59 vs 0.21; walk-forward chose LightGBM every year from 2021) — but WF OOS Sharpe 0.75 vs SPY 0.76, alpha t 0.70: **no edge over passive exposure**. Foundation models (Kronos, Chronos, TimesFM, …) were not re-run on the daily cross-section: they showed no tradable edge at minute horizons (2026-11-18 report) and the stronger tabular model does not beat SPY here, so their inference cost is not justified yet. |
| Holding period (A, C) | Weekly vs monthly rebalancing within momentum: similar Sharpe, monthly lower cost; reversal holds of 5–10 days all fail. |
| Execution timing (selected variants, next open vs next close) | Momentum/rotation indifferent (Sharpe 0.83/0.84, 0.76/0.76); mean reversion decays with a later fill (C3 0.67 → 0.38, C2 0.59 → 0.47): their small edge is time-sensitive. ML ranker slightly better at the close (lower cost). |
| Intraday (B, ORB on PIT stocks in play, 2021–2025, 5-minute bars) | Best variant: gross CAGR 24.8% / gross Sharpe 1.01, but ~106× annual turnover costs 11.2%/yr; walk-forward-selected variants were negative out of sample (Sharpe −0.07). Costs + instability defeat it. News-confirmed variants could not be tested: Alpaca news history on this plan starts 2026-04 (inside the holdout). |
| Crypto direct vs ETF (same SMA-50 trend rule, 2024-04 … 2025-11) | IBIT Sharpe 0.70 (38 trades) vs BTC/USD 0.54 (104 trades, 25 bps taker): the 24/7 market flips the signal more often and Alpaca crypto fees dominate. Exchange-traded exposure was the cheaper implementation. |
| Crypto trend (D1) vs BTC buy-and-hold | Lower drawdown/vol but alpha t ≈ 0 and PBO 0.62; BTC B&H Sharpe 1.45 over 2023–2025 was higher. |
| Policy feasibility | Under today's policy (allowlist, 1 position ≤ 25% of a $5k account, whole shares) the selected variants would average 1–22% exposure and 0.3–4.7% CAGR: even a validated multi-name strategy would need a deliberate policy change by the operator to matter economically. |
| Options (live indicative scan, 8 underlyings, 30 DTE) | ATM IV ≈ HAR-RV forecast (IV/RV 0.84–1.27). Under filtered historical simulation, spreads that sell OTM index puts (SPY/QQQ put debit spreads ≈ +26% EV of capital) look attractive and spreads that sell OTM calls look poor — the index put-skew premium. Single stocks ≈ 0. ATM spreads are 1–8% of premium. **Unvalidated**: no historical executable quotes; crash risk is the price of that premium. Journaled prospectively for settlement at expiry. |

## Strategy families (development period, walk-forward out-of-sample, after modelled costs)

Holdout (protected, untouched by development): 2025-11-21 → 2026-11-20. Backtested development results on historical data under a cost model. They are not realized trading results and are not evidence of future profitability.

| Family | Asset | Variants | WF OOS period | Sharpe | CAGR | Max DD | Benchmark Sharpe / CAGR / DD | DSR (trials) | SPA p | alpha t | PBO | 2x cost Sharpe | Verdict |
|---|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| C3_etf_pullback | etf | 12 | 2019-01-02…2025-11-20 | 0.72 | 11.6% | -29.9% | 0.89 / 16.7% / -33.8% | 0.878 (12) | 1.00 | 0.36 | 0.28 | 0.64 | FAIL: DSR 0.878 < 0.95; no edge vs passive benchmark (Sharpe-diff SPA p=1.00, alpha t=0.36) |
| G_ml_rank | us_equity | 16 | 2020-01-02…2025-11-20 | 0.75 | 18.1% | -41.2% | 0.76 / 14.4% / -33.8% | 0.742 (16) | 1.00 | 0.70 | 0.16 | 0.96 | FAIL: DSR 0.742 < 0.95; no edge vs passive benchmark (Sharpe-diff SPA p=1.00, alpha t=0.70) |
| D1_crypto_tsmom | crypto | 48 | 2023-01-01…2025-11-20 | 1.04 | 39.7% | -38.4% | 1.45 / 77.3% / -30.6% | 0.714 (96) | 0.82 | -0.02 | 0.62 | 0.93 | FAIL: DSR 0.714 < 0.95; no edge vs passive benchmark (Sharpe-diff SPA p=0.82, alpha t=-0.02) |
| A_momentum | us_equity | 120 | 2019-01-02…2025-11-20 | 0.66 | 22.2% | -58.3% | 0.89 / 16.7% / -33.8% | 0.614 (240) | 1.00 | 0.71 | 0.14 | 1.00 | FAIL: DSR 0.614 < 0.95; no edge vs passive benchmark (Sharpe-diff SPA p=1.00, alpha t=0.71) |
| A2_etf_rotation | etf | 72 | 2019-01-02…2025-11-20 | 0.43 | 6.4% | -28.4% | 0.89 / 16.7% / -33.8% | 0.502 (144) | 0.88 | -0.62 | 0.68 | 0.78 | FAIL: DSR 0.502 < 0.95; walk-forward OOS Sharpe 0.43 < 0.5 |
| C1_weekly_reversal | us_equity | 24 | 2019-01-02…2025-11-20 | 0.41 | 7.6% | -54.2% | 0.89 / 16.7% / -33.8% | 0.495 (48) | 1.00 | -0.70 | 0.61 | 0.50 | FAIL: DSR 0.495 < 0.95; walk-forward OOS Sharpe 0.41 < 0.5 |
| C2_pullback | us_equity | 24 | 2019-01-02…2025-11-20 | 0.20 | 1.3% | -67.0% | 0.89 / 16.7% / -33.8% | 0.277 (37) | 1.00 | -1.00 | 0.17 | 0.34 | FAIL: DSR 0.277 < 0.95; walk-forward OOS Sharpe 0.20 < 0.5 |
| D2_crypto_xsmom | crypto | 36 | 2023-01-01…2025-11-20 | 0.27 | 3.7% | -36.4% | 1.45 / 77.3% / -30.6% | 0.166 (72) | 0.79 | -0.34 | 0.35 | 0.25 | FAIL: DSR 0.166 < 0.95; walk-forward OOS Sharpe 0.27 < 0.5 |
| B_orb | us_equity | 32 | 2022-01-03…2025-11-20 | -0.07 | -3.0% | -21.9% | 0.36 / 4.4% / -13.3% | 0.000 (96) | 0.75 | -0.25 | 0.00 | -0.19 | FAIL: DSR 0.000 < 0.95; walk-forward OOS Sharpe -0.07 < 0.5 |

### Execution tier: selected variant under today's risk policy ($5,000 account, allowlist, 1 position ≤ 25% of equity, whole shares)

| Family | Sharpe | CAGR | Max DD | Avg exposure | Trades |
|---|---:|---:|---:|---:|---:|
| C3_etf_pullback | 0.67 | 2.4% | -8.4% | 9.0% | 1251 |
| G_ml_rank | 0.16 | 0.3% | -5.7% | 1.2% | 12 |
| D1_crypto_tsmom | intraday/crypto families use their own execution model | | | | |
| A_momentum | 0.79 | 4.7% | -7.2% | 7.7% | 172 |
| A2_etf_rotation | 0.68 | 3.7% | -9.0% | 21.8% | 161 |
| C1_weekly_reversal | 0.12 | 0.3% | -8.7% | 3.4% | 177 |
| C2_pullback | 0.69 | 0.6% | -1.3% | 0.8% | 67 |
| D2_crypto_xsmom | intraday/crypto families use their own execution model | | | | |
| B_orb | intraday/crypto families use their own execution model | | | | |

### Protected holdout looks

None. No family passed the development gate, so the holdout was never opened.


## Cross-asset selection (2026-11-23): **NO_TRADE** — no executable candidate with positive evidence-weighted utility

* AAOI (A_momentum, REJECTED): evidence REJECTED: no validated edge (expected return shrunk to 0); not executable: symbol not on the risk-policy allowlist; multi-day holding not supported by the current position lifecycle (EOD flatten)
* AXTI (A_momentum, REJECTED): evidence REJECTED: no validated edge (expected return shrunk to 0); not executable: symbol not on the risk-policy allowlist; multi-day holding not supported by the current position lifecycle (EOD flatten)
* DELL (A_momentum, REJECTED): evidence REJECTED: no validated edge (expected return shrunk to 0); not executable: symbol not on the risk-policy allowlist; multi-day holding not supported by the current position lifecycle (EOD flatten)
* INTC (A_momentum, REJECTED): evidence REJECTED: no validated edge (expected return shrunk to 0); not executable: symbol not on the risk-policy allowlist; multi-day holding not supported by the current position lifecycle (EOD flatten)
* LITE (A_momentum, REJECTED): evidence REJECTED: no validated edge (expected return shrunk to 0); not executable: symbol not on the risk-policy allowlist; multi-day holding not supported by the current position lifecycle (EOD flatten)
* MRNA (A_momentum, REJECTED): evidence REJECTED: no validated edge (expected return shrunk to 0); not executable: symbol not on the risk-policy allowlist; multi-day holding not supported by the current position lifecycle (EOD flatten)

## Options volatility scan (2026-11-23T21:01 UTC, indicative (delayed/derived))

| Underlying | 30d ATM IV | HAR-RV forecast | IV/RV | 25Δ skew | ATM spread % of premium |
|---|---:|---:|---:|---:|---:|
| SPY | 22.2% | 22.2% | 1.00 | 10.8 vol pts | 1.6% |
| QQQ | 31.1% | 32.6% | 0.95 | 12.3 vol pts | 2.1% |
| IWM | 24.3% | 24.2% | 1.00 | 9.9 vol pts | 6.8% |
| NVDA | 61.6% | 59.0% | 1.04 | 9.2 vol pts | 4.0% |
| TSLA | 49.6% | 59.6% | 0.83 | 8.9 vol pts | 5.6% |
| AAPL | 24.8% | 27.2% | 0.91 | 6.4 vol pts | 16.8% |
| AMD | 81.2% | 71.3% | 1.14 | 15.7 vol pts | 6.6% |
| META | 43.9% | 45.9% | 0.96 | 7.6 vol pts | 6.7% |

## Crypto scan (2026-11-22): BTC above 50-day average = False, BTC 90-day return -22.5%


## Capability snapshot

* crypto_spot: research=yes, execution=no (policy NOT_IMPLEMENTED)
* extended_and_overnight_hours: research=yes, execution=no (policy DISABLED_BY_POLICY)
* leveraged_inverse_etp: research=yes, execution=no (policy DENYLISTED)
* margin_borrowing: research=yes, execution=no (policy DISABLED_BY_POLICY)
* us_equity_long: research=yes, execution=yes (policy EXECUTABLE)
* us_equity_short: research=yes, execution=no (policy DISABLED_BY_POLICY)
* us_options: research=yes, execution=no (policy DISABLED_BY_POLICY)

## What is implemented vs empirically tested vs research-only

* **Implemented and tested in code** (unit + integration, synthetic data; full suite 2,056 + 20 torch tests pass):
  capability probe, GET-only client, PIT panels with delisted names and segment rules, cost models, backtester
  (non-fills, forced exits, whole shares, open/close execution), walk-forward/DSR/SPA/PBO/BH, append-only trial
  ledger, one-use holdout vault, strategy registry with reversible promotion and shadow journal, factor DSL,
  opportunity contract and selection, options pricing/IV/early-exercise/FHS/parity spot, scanner, API, MCP tools,
  dashboard panel.
* **Empirically tested on real Alpaca data**: every family above (development period only), the controlled
  comparisons, the live options and crypto scans, the scanner and the cross-asset selection (NO_TRADE).
* **Research-only / not executable**: shorts, margin, leveraged/inverse ETPs, crypto, options, extended hours (by
  policy or missing execution path); news-catalyst intraday (no history); options strategies (no executable quote
  history); the protected holdout (never opened).
* **Not built, evidence-gated**: a multi-day holding position lifecycle and a crypto execution adapter — both
  touch the safety-critical order path and are only justified once a strategy that needs them validates.
* **Operational**: no orders were placed; the paper API was restarted once (no session, no positions, market
  closed) to load the new read-only route; risk policy, keys and modes unchanged.

## Trial accounting

Every evaluated variant is in `experiments/quantlab/trials.sqlite3` (append-only), including runs superseded by
the data fixes above; deflated Sharpe uses the full count per family (e.g. A_momentum 240, A2 144, D1 96, B 96).
Agent-proposed factors (`run_research_experiment`) are counted under `F_agent` (1 so far: volatility-penalized
12-1 momentum — OOS Sharpe 0.59, alpha t −0.49, failed).

## Reproduce

```bash
make capabilities && make quantlab-data && make quantlab-research
uv run --frozen python -m services.quantlab.cli timing        # execution-timing diagnostic
uv run --frozen python -m services.quantlab.cli options-scan  # market hours
uv run --frozen python -m services.quantlab.cli report        # these tables, from the recorded artifacts
```
