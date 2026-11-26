# AI Trading — Development Roadmap

**Primary objective:** maximize sustainable net risk-adjusted ROI through reliable, evidence-based improvements.
Local-first (48 GB Apple Silicon), autonomous AI research, Alpaca paper execution, LEAN backtesting.
Progress and evidence per phase: [PROGRESS.md](PROGRESS.md).

| # | Phase | Priority | Depends on |
|---|---|---|---|
| 1 | Complete position and exit management | Critical | existing risk + ledger |
| 2 | Automated historical market-data pipeline | Critical | existing ingestion + storage |
| 3 | Unify research, backtesting and trading decisions (prediction parity) | Critical | model registry |
| 4 | Strategy discovery and opportunity selection | High | 2, 3 |
| 5 | Realistic backtesting and economic validation | High | 1–4 |
| 6 | Autonomous research and model improvement | High | 5 |
| 7 | Reliability, monitoring and maintenance | High | integrated throughout |

## Completion criteria

1. **Positions/exits:** entry-to-exit lifecycle verified in simulation and Alpaca paper tests (partial fills,
   cancellation races, disconnects, market close, restart recovery). A planned stop is never described as
   guaranteed protection. Unprotected positions trigger an explicit safety state instead of silently continuing new
   entries.
2. **Data:** research automatically discovers refreshed datasets and reproduces historical experiments without
   combining incompatible feeds, using future revisions or losing provenance. Coverage beyond SPY/QQQ/AAPL/MSFT from
   the eligible universe; compare IEX-only data against better data before any paid purchase.
3. **Parity:** the same model and point-in-time inputs produce equivalent predictions and trade decisions in research
   and paper runtime, apart from documented differences (e.g. updated quotes). One prediction contract (gross, costs,
   net, uncertainty, thresholds); costs never double-charged; identical features/preprocessing; 5/15/60-minute
   horizons; decision-parity tests.
4. **Discovery:** reproducible comparisons against simple strategies; recorded reasons for candidate failure;
   attribution of missed opportunities to predictions, costs, uncertainty or constraints. Long-only, cash-account and
   risk restrictions preserved.
5. **Backtesting:** full trading lifecycle (non-fills, capital constraints, exits); simulated, historical and actual
   paper results clearly separated; statistical standards never lowered to promote a strategy.
6. **Autonomous research:** generates and evaluates hypotheses, persists failures and successes, promotes only
   evidence-backed improvements into reversible paper trading; agents cannot modify protected evaluation rules,
   access reserved data or bypass risk authorization.
7. **Reliability:** extended paper sessions survive failures without duplicate orders; trustworthy operational and
   economic diagnostics; local validation (`make validate`) for tests/types/security/model contracts; dependency review; docs match the Alpaca SDK
   execution architecture.

## Global engineering rules

Inspect before changing; preserve architecture and completed features; decide technically and independently;
evaluate technologies by measurable after-cost improvement; keep experimental ML separate from broker execution;
preserve immutable audit records, point-in-time rules and protected holdouts; never spend money or enable live
trading without explicit authorization; never weaken risk controls or statistical validation to force trades; real
implementations and comprehensive tests; report measurable evidence, never profitability from passing tests.

A system that produces `NO_TRADE` because no strategy has demonstrated an edge is functioning correctly. Increased
trading frequency is not improved ROI.
