"""Multi-asset quantitative research lab (research-only; no order, risk-policy or broker-write authority).

Point-in-time panels across Alpaca-supported instruments (U.S. equities/ETFs incl. delisted names, crypto,
options metadata/quotes), vectorized after-cost backtests, walk-forward + protected-holdout validation with
multiple-testing corrections, a unified cross-asset opportunity contract, and a reversible strategy registry.
Execution stays with ``services.risk`` -> ``services.order_ledger``; this package only emits research artifacts,
shadow signals and (for PAPER-promoted strategies) ordinary ``TradeProposal`` candidates.
"""
