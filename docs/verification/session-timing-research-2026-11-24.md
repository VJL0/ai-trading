# Session-timing research (U.S. stocks and ETFs) — 2026-11-24

Research-only. Every number below is a **backtest after modelled costs on historical development data**
(2016-01-04 … 2025-11-20). None is a realized trading result or evidence of future profitability. No order was
placed, the risk policy and paper-only mode are unchanged, and the protected holdout (2025-11-21 … 2026-11-20) was
not opened: the intraday download is clamped in code to end before it.

## Question

When in the U.S. session — opening, midday, afternoon, close, overnight, around month-ends — does an allowlisted
ETF or large cap show an after-cost edge that the current execution stack could actually capture? These are the
first families that fit today's lifecycle without any policy change (H1–H3: intraday, allowlisted symbols), plus
the two classic multi-session effects (I1 overnight, I2 turn-of-month) and a defensive volatility-managed index
family (J) that would need the operator's overnight authorization.

## Method

* **Data**: Alpaca historical SIP 5-minute bars (free plan, split/dividend adjusted), 24 allowlisted symbols,
  2016-01-04 … 2025-11-20; regular-session slots only (early closes kept with their shorter grid). Daily SIP panel
  (raw levels, ADV, trends) as of the previous close for every intraday decision. Consolidated data only — no IEX
  observation is treated as the market.
* **Execution model = runtime lifecycle**: marketable LIMIT BUY at the decision close + 10 bps, filled at the next
  bar's open or not at all; no entry in the last 30 minutes; flatten at the bar 10 minutes before the close
  (12:50 on early closes); every position carries a protective stop (150 bps; I2 also tests 300 bps) with the
  runtime's STOP_LIMIT semantics (a bar trading below the 50 bps limit gap is a `STOP_BREACH`; gap-through exits at the bar open,
  including the next session's open for overnight holds); one position per symbol; concurrency cap; the runtime
  cost model (spread bucket, 2 bps fixed slippage, square-root impact, SEC/FINRA on sells) charged once per leg,
  opening-spread multiplier in the first 15 minutes.
* **Validation**: the unchanged pre-registered development gate (deflated Sharpe ≥ 0.95 with every family trial
  counted, walk-forward OOS Sharpe ≥ 0.5, SPA p ≤ 0.10 vs SPY buy-and-hold or Newey-West alpha t ≥ 2, PBO ≤ 0.5,
  positive Sharpe at 2× costs, drawdown ≥ −35 %, ≥ 50 trades) and Benjamini-Hochberg across families. Walk-forward:
  yearly re-selection from 2019 with ≥ 504 training sessions. Benchmark: SPY held close-to-close (the investor's
  passive alternative), which is stricter than an open-to-close benchmark for intraday rules.
* **Variants were fixed in code before any result was seen** (H1 84, H2 72, H3 48, I1 36, I2 64, J 72); all
  are in `experiments/quantlab/trials.sqlite3`.

## Known limitations

* The 12 large caps in the allowlist were chosen in 2026; universes containing them (H2/H3 `allow24`) carry
  selection/survivorship bias favouring long rules. ETF-only universes are much less affected.
* 5-minute bars cannot resolve the runtime's 15-second entry timeout or queue position; a gap past the limit is a
  non-fill, otherwise the bar open is the fill price before costs. Alpaca paper fills are themselves simulated,
  so real slippage can only be measured with live fills (not authorized).
* The cost model is deliberately conservative for mega-cap ETFs (see the observed-spread table): a rule whose gross
  edge per trade is below the modelled cost but above the observed spread is reported as cost-sensitive, not
  promoted — standards are not relaxed to admit it.

## Verdict

**No family passed the pre-registered development gate, so nothing was registered, the protected holdout was not
opened, and the opportunity engine keeps selecting `NO_TRADE`.** 376 development trials were recorded (H1 84,
H2 72, H3 48, I1 36, I2 64, J 72); holdout vault looks: 0.

* **Intraday timing does not survive costs on allowlisted ETFs and large caps.** Every intraday family is negative
  out of sample (walk-forward Sharpe H1 −0.48, H2 −1.10, H3 −0.96). The selected variants' gross edges are
  3.7 bps (H1), 0.8 bps (H2) and 0.2 bps (H3) per round trip, none statistically distinguishable from zero
  (gross t ≤ 1.44), against ≈ 6.7–7.0 bps of modelled cost. The classic market-intraday-momentum window
  (last half hour into the close) is not executable here, and the executable windows show no edge.
* **The overnight premium is real but too small to harvest.** SPY/QQQ/IWM earn +3.8/+5.0/+5.8 bps per night on
  average (t 2.6–3.1, descriptive). I1 captures 5.9 bps gross per trade (t 1.86), below the ≈ 7 bps cost, and
  the only lifecycle-executable form (exit 15:50 of the next session) also carries the next day's intraday risk:
  walk-forward Sharpe −0.08, max drawdown −39 %.
* **Turn-of-month** shows +21 bps gross per trade (t 1.55) but the walk-forward selection is unstable
  (PBO 0.61) and out-of-sample Sharpe is −0.18.
* **J (volatility-managed QQQ, 15 % target, 10-day lookback, 0.25 band)** is the strongest result of the whole
  research program and still fails. Walk-forward OOS 2019–2025: Sharpe 1.14 vs SPY buy-and-hold 0.89 and QQQ
  buy-and-hold 0.95; CAGR 17.7 % vs 16.7 % / 22.2 %; max drawdown −23.5 % vs −33.8 % / −35.0 %; DSR 0.991 with
  all 72 trials counted; positive at 2× costs (1.13). Failures: no significant edge over the passive benchmark
  (SPA p 0.23, alpha t 1.81 vs SPY, 1.87 vs QQQ — the gate needs p ≤ 0.10 or t ≥ 2), PBO 0.54 > 0.5, and
  Benjamini-Hochberg q 0.128 > 0.10 across all recorded families. The walk-forward chose the same spec every year;
  2022 was −20 %. It is a risk-reducing overlay, not a validated edge, and it would also need the operator's
  overnight authorization to trade. Standards were not relaxed.

## Session-timing families (development period, walk-forward out-of-sample, after modelled costs)

| Family | Variants | WF OOS period | Sharpe | CAGR | Max DD | SPY B&H Sharpe / CAGR | DSR (trials) | SPA p | alpha t | beta | PBO | 2x cost Sharpe | Trades | Verdict |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| I1_overnight | 36 | 2019-01-02…2025-11-20 | -0.08 | -1.1% | -39.1% | 0.89 / 16.7% | 0.043 (36) | 1.00 | -1.51 | 0.23 | 0.17 | -0.60 | 1768 | FAIL: DSR 0.043 < 0.95; walk-forward OOS Sharpe -0.08 < 0.5 |
| I2_turn_of_month | 64 | 2019-01-02…2025-11-20 | -0.18 | -1.1% | -12.8% | 0.89 / 16.7% | 0.013 (64) | 1.00 | -1.21 | 0.08 | 0.61 | 0.45 | 176 | FAIL: DSR 0.013 < 0.95; walk-forward OOS Sharpe -0.18 < 0.5 |
| H3_opening_gap | 48 | 2019-01-02…2025-11-20 | -0.96 | -3.6% | -24.9% | 0.89 / 16.7% | 0.000 (48) | 1.00 | -3.43 | 0.05 | 0.22 | -0.53 | 648 | FAIL: DSR 0.000 < 0.95; walk-forward OOS Sharpe -0.96 < 0.5 |
| H2_vwap_reversion | 72 | 2019-01-02…2025-11-20 | -1.10 | -2.7% | -18.6% | 0.89 / 16.7% | 0.000 (72) | 1.00 | -3.14 | 0.02 | 0.15 | -1.01 | 786 | FAIL: DSR 0.000 < 0.95; walk-forward OOS Sharpe -1.10 < 0.5 |
| H1_intraday_momentum | 84 | 2019-01-02…2025-11-20 | -0.48 | -3.4% | -29.0% | 0.89 / 16.7% | 0.000 (84) | 1.00 | -2.61 | 0.14 | 0.00 | -0.80 | 1572 | FAIL: DSR 0.000 < 0.95; walk-forward OOS Sharpe -0.48 < 0.5 |

Selected variants (walk-forward choice in the final year):

* I1_overnight: `exit=next_eod,filter=intraday_up,universe=SPY`; turnover 179.1x/yr, cost drag 6.2%/yr
* I2_turn_of_month: `enter_k_last=1,exit_m_first=2,regime=sma200,stop_bps=150.0,universe=SPY`; turnover 17.8x/yr, cost drag 0.6%/yr
* H3_opening_gap: `direction=fade_down,entry=10:00,hold=120,k_sd=2.0,universe=index4`; turnover 16.4x/yr, cost drag 0.6%/yr
* H2_vwap_reversion: `entry=14:00,hold=60,k_sd=2.0,regime=none,universe=index4`; turnover 19.9x/yr, cost drag 0.7%/yr
* H1_intraday_momentum: `entry=10:00,exit=eod,signal=open_to_entry,threshold_sd=0.5,universe=SPY`; turnover 159.3x/yr, cost drag 5.3%/yr

### Execution tier under today's policy ($5,000, 1 position ≤ 25%, whole shares)

| Family | Sharpe | CAGR | Max DD | Avg exposure | Trades |
|---|---:|---:|---:|---:|---:|
| I1_overnight | NOT_PERMITTED: today's policy rejects overnight holds (allow_overnight_positions off or strategy not allowlisted) | | | | |
| I2_turn_of_month | NOT_PERMITTED: today's policy rejects overnight holds (allow_overnight_positions off or strategy not allowlisted) | | | | |
| H3_opening_gap | -0.42 | -0.3% | -3.8% | 0.3% | 234 |
| H2_vwap_reversion | -0.37 | -0.2% | -2.0% | 0.2% | 336 |
| H1_intraday_momentum | -0.26 | -0.4% | -6.3% | 5.9% | 1572 |

### Trade-level economics of the selected variants (runtime cost model)

| Family | Filled / attempted | Mean gross bps | t (gross) | Mean cost bps | Mean net bps | Hit rate (net) | Exits |
|---|---|---:|---:|---:|---:|---:|---|
| H1_intraday_momentum | 786 / 786 | 3.70 | 1.44 | 6.67 | -2.96 | 50.9% | EOD 741, STOP 45 |
| H2_vwap_reversion | 393 / 393 | 0.80 | 0.32 | 7.04 | -6.24 | 42.5% | STOP 5, TIME 388 |
| H3_opening_gap | 324 / 324 | 0.20 | 0.04 | 6.91 | -6.71 | 45.7% | STOP 29, STOP_BREACH 1, TIME 294 |
| I1_overnight | 884 / 884 | 5.85 | 1.86 | 6.97 | -1.12 | 49.9% | EOD 792, STOP 74, STOP_BREACH 1, STOP_GAP 17 |
| I2_turn_of_month | 88 / 88 | 21.05 | 1.55 | 6.70 | 14.35 | 58.0% | EOD 72, STOP 14, STOP_GAP 2 |

### Descriptive time-of-day profile (2468 full sessions; mean bps, t-stat)

| Symbol | overnight | →10:00 | →10:30 | →12:00 | →14:00 | →15:00 | →15:30 | →16:00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | 3.8 (2.6) | 0.9 (1.6) | 0.1 (0.2) | 0.6 (1.5) | 0.2 (0.5) | 0.9 (2.0) | 0.1 (0.3) | -0.4 (-0.7) |
| QQQ | 5.0 (2.9) | 1.5 (1.7) | 0.2 (0.2) | 0.7 (1.4) | 0.2 (0.4) | 1.4 (2.7) | -0.3 (-0.5) | -0.1 (-0.2) |
| IWM | 5.8 (3.1) | -0.1 (-0.1) | -0.9 (-1.1) | 0.9 (1.6) | 0.4 (0.9) | 0.8 (1.5) | -0.4 (-0.7) | -1.2 (-1.7) |
| DIA | 3.5 (2.4) | 0.9 (1.4) | -0.0 (-0.1) | 0.6 (1.6) | 0.1 (0.2) | 0.8 (1.8) | 0.3 (0.6) | -0.4 (-0.6) |
| TLT | -1.2 (-0.9) | 0.2 (0.5) | 0.1 (0.3) | -0.3 (-1.0) | 0.3 (1.3) | -0.5 (-1.6) | -0.2 (-0.6) | 1.4 (4.1) |
| GLD | 5.4 (3.8) | -1.1 (-2.2) | 0.6 (1.2) | -0.2 (-0.9) | 0.5 (2.0) | 0.3 (1.1) | -0.2 (-0.9) | 0.1 (0.2) |
| AAPL | 1.5 (0.6) | 9.6 (6.4) | -0.0 (-0.0) | 1.3 (2.0) | 0.0 (0.0) | 0.8 (1.4) | 0.0 (0.0) | -0.6 (-0.6) |
| NVDA | 18.7 (4.8) | 3.0 (1.1) | 0.3 (0.2) | 1.2 (1.1) | 0.8 (0.8) | 3.1 (3.3) | -2.5 (-2.6) | 1.7 (1.4) |

Windows are the half hour ending at the time shown (→10:30 = 10:00-10:30); →10:00 includes the opening 30 minutes from the open, not the overnight gap.

### Observed SIP NBBO spreads vs the cost model (16 development sessions, 09:35/10:00/12:00/15:25)

| Symbol | Observed full spread bps (median) | 09:35 | 15:25 | Model full spread bps | Model fixed slippage bps/leg |
|---|---:|---:|---:|---:|---:|
| SPY | 0.30 | 0.28 | 0.24 | 2.00 | 2.00 |
| QQQ | 0.49 | 0.46 | 0.34 | 2.00 | 2.00 |
| IWM | 0.54 | 0.53 | 0.52 | 2.00 | 2.00 |
| DIA | 0.89 | 0.90 | 0.59 | 2.00 | 2.00 |
| VTI | 1.45 | 2.11 | 1.02 | 5.00 | 2.00 |
| VOO | 0.74 | 0.73 | 0.73 | 2.00 | 2.00 |
| XLK | 0.88 | 1.30 | 0.70 | 2.00 | 2.00 |
| XLF | 2.75 | 2.79 | 2.74 | 2.00 | 2.00 |
| XLE | 1.35 | 1.49 | 1.25 | 2.00 | 2.00 |
| XLV | 0.79 | 0.98 | 0.76 | 2.00 | 2.00 |
| TLT | 0.85 | 0.85 | 0.86 | 2.00 | 2.00 |
| GLD | 0.63 | 0.80 | 0.59 | 2.00 | 2.00 |
| AAPL | 0.75 | 1.48 | 0.63 | 2.00 | 2.00 |
| MSFT | 1.26 | 2.17 | 0.85 | 2.00 | 2.00 |
| AMZN | 2.89 | 4.00 | 1.33 | 2.00 | 2.00 |
| GOOGL | 3.69 | 4.81 | 1.05 | 2.00 | 2.00 |
| META | 2.62 | 4.24 | 1.40 | 2.00 | 2.00 |
| NVDA | 2.87 | 4.86 | 1.72 | 2.00 | 2.00 |
| JPM | 1.79 | 2.80 | 1.16 | 2.00 | 2.00 |
| V | 3.28 | 5.49 | 1.52 | 2.00 | 2.00 |
| JNJ | 2.35 | 4.38 | 1.24 | 2.00 | 2.00 |
| PG | 2.39 | 3.62 | 1.42 | 2.00 | 2.00 |
| KO | 1.69 | 1.88 | 1.67 | 5.00 | 2.00 |
| XOM | 1.68 | 2.84 | 1.16 | 2.00 | 2.00 |

## J_vol_managed (daily panel, next-open execution)

| | J walk-forward OOS | SPY buy-and-hold | QQQ buy-and-hold |
|---|---:|---:|---:|
| Sharpe | 1.14 | 0.89 | 0.95 |
| CAGR | 17.7 % | 16.7 % | 22.2 % |
| Max drawdown | −23.5 % | −33.8 % | −35.0 % |
| alpha (annual) / t / beta of J vs | — | 6.1 % / 1.81 / 0.66 | 4.0 % / 1.87 / 0.59 |

Yearly J returns: 2019 +29.1 %, 2020 +37.2 %, 2021 +18.7 %, 2022 −20.4 %, 2023 +41.3 %, 2024 +17.7 %, 2025 +10.4 %
(to 2025-11-20). Execution tier under today's sizing ($5,000, one position ≤ 25 %, whole shares): Sharpe 0.93,
CAGR 4.8 %, max drawdown −9.1 % — a what-if, since holding across sessions is disabled in today's policy.
Source: `experiments/quantlab/j_benchmark_diagnostic.json` (diagnostic of the already selected spec, no trial).

## Observations that are NOT tested strategies

* **Cost model vs observed spreads.** Observed SIP NBBO spreads for SPY/QQQ/IWM (0.30/0.49/0.54 bps) are a small
  fraction of the model's 2 bps MEGA bucket; for several large caps the model is too LOW at the open (09:35:
  GOOGL 4.8, NVDA 4.9, V 5.5, JNJ 4.4 bps observed vs 2 bps). The fixed 2 bps/leg slippage dominates intraday
  costs and is not measurable from Alpaca paper fills (simulated). Intraday ETF rules with a gross edge of a
  few bps (H1, I1) are therefore cost-sensitive, but their gross edges are also not significant, so a cheaper
  cost model would not turn them into validated strategies. The runtime model was not changed (research/runtime
  cost parity).
* **AAPL open→10:00 drift** (+9.6 bps, t 6.4, positive in every year 2016–2025; the official daily open gives the
  same figure, so it is not an opening-print artifact; `experiments/quantlab/open_drift_check.json`) and NVDA's
  +18.7 bps overnight mean. Both were found by
  descriptive mining of the development set on stocks chosen in 2026 (survivorship), so testing them on the same
  data would be circular. They are recorded as hypotheses that need prospective evidence; nothing was selected
  from the profile.

## Review before any trial was recorded

Research was paused until an independent read-only review (2 rounds, 6 reviewer agents; findings and fixes are
listed in `docs/roadmap/PROGRESS.md`) finished; contaminated trials
would otherwise have stayed in the append-only ledger. Fixed: multi-session P&L booked in one lump on the exit
day (biases beta towards 0 and inflates alpha t; hides drawdowns) → marked to market every session; stops filled
at the stop when a bar traded through the STOP_LIMIT's limit → `STOP_BREACH` at min(limit, bar close); I1's
next-open exit is not a lifecycle the runtime has → executable `next_eod` form + `NOT_EXECUTABLE` /
`NOT_PERMITTED` execution tiers; re-run skip logic that could hide an interrupted family and shrink BH-FDR; an
adjustment-basis mismatch in the (unused) session holdout path; and daily-job safety (no runs in market hours,
bounded waits, child killed on cancellation). Every fix has a regression test.

## Reproduce

```bash
make quantlab-session      # session-fetch, build-panel --what session, session-profile, research --asset session
uv run --frozen python -m services.quantlab.cli research --asset equity --families J_vol_managed
uv run --frozen python -m services.quantlab.cli session-diagnostics
uv run --frozen python -m services.quantlab.cli spread-sample
uv run --frozen python -m services.quantlab.cli report --session
```

A re-run on the same data and code is skipped as carrying no new evidence (`--force` adds trials and raises
every family's deflated-Sharpe bar). Artifacts: `experiments/quantlab/runs/ql_20261125T025430Z` (session),
`experiments/quantlab/runs/ql_20261125T025531Z` (J), `docs/verification/session-timing-research-2026-11-24.json`.
