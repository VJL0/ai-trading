# Journal retrieval and trading ROI: evidence assessment

Research date: 2026-11-19. This is a literature and implementation assessment, not a measured improvement in this system. No trading configuration was changed and no paid experiments were run.

## Decision

Evidence supports a bounded research experiment, not an assumption that journal RAG will improve net returns. Prioritize outcome-linked research memory and evaluation over a general vector database or free-form self-reflection in the trading path. There is no defensible percentage estimate of ROI uplift for this workstation from the evidence reviewed.

## Primary research

1. **FinMem**, section 6.3, studies retrieval bandwidth. Reported cumulative returns for top-1, top-3, top-5 and top-10 retrieval are 52.09%, 29.44%, 54.70% and 79.44%. This demonstrates sensitivity to memory selection, including deterioration when adding context. It does not establish retrieval versus no retrieval, nor transfer to our intraday strategy. Source: https://arxiv.org/html/2311.13743v2

2. **FinAgent**, section 7.1, reports that adding high-level reflection to market intelligence plus low-level reflection raises its reported annualized-return metric from 57.16% to 89.25% on TSLA and 52.33% to 54.80% on ETHUSD. TSLA maximum drawdown also increases from 25.77% to 27.62%. This tests a reflection component, not a clean journal-retrieval-only intervention. These historical system results are not expected returns for our system. Source: https://arxiv.org/html/2402.18485v2

3. **FinSABER** re-evaluates FinMem and FinAgent with broader, longer evaluation and explicit commissions. Reported advantages weaken; both exhibit regime-dependent shortcomings. This is evidence against extrapolating the original backtests into durable profitability, not proof that every retrieval design fails. Commission assumptions also do not establish execution realism for our intraday fills. Source: https://arxiv.org/html/2505.07078v2

4. **LLM Trading Agents Earn Beta, Not Alpha**, a November 14, 2026 preprint, reports 91 runs of six models, no statistically significant factor alpha, and market/style exposure explaining a median 78% of returns. This is broader caution about agent-return claims; it does not isolate journal RAG and should not be treated as a definitive verdict on it. Source: https://arxiv.org/abs/2611.19530

## Fit to this repository

- `services/api/pipeline.py`: quant signals select a candidate before AI review; the AI is skipped when no candidate qualifies. Retrieval in that review cannot create opportunities in those skipped cycles.
- `services/ai/accounting.py`: findings and evidence already persist, providing raw material for memory.
- `services/research/advanced/rd_agent.py`: proposals already receive the last 12 experiment results. Any new retrieval must beat this existing context, not merely an empty prompt.
- `services/research/autonomous/README.md`: the documented initial real-data campaign had no candidate pass development screening. That is a documented campaign result, not a fresh check of current campaign state. It makes demonstrated signal quality a more direct priority than extra narrative context.
- The session started earlier in this chat was explicitly SIMULATED. Its results cannot establish real-market ROI.

## Plausible benefits and failure modes

The most credible initial benefit is research efficiency: avoiding duplicate failed hypotheses and retrieving comparable measured experiments. That can improve research output per dollar without necessarily increasing trading P&L. In trading review, retrieval might reduce repeated costly decisions, but it might also suppress profitable trades, reinforce incorrect narratives, or increase latency and model expense. These are hypotheses, not observed effects here.

## Experiment required before adoption

Freeze model, prompt, candidate stream, sizing, execution model and risk rules. Compare the quant baseline, current AI review without new memory, structured outcome-linked retrieval, and semantic retrieval. Include the current recent-12 experiment context when testing research proposals. Predeclare retrieval settings and count every attempted variant.

Replay chronologically. Each decision may retrieve only entries actually available then; a later realized outcome or later-written lesson must not appear in earlier memory. Track entry availability separately from the original trade timestamp. Do not let evaluation data enter the journal used for development. Audit pretrained historical knowledge separately; timestamp filtering cannot remove a model's memorized future events.

Measure paired net P&L after spread, fees, slippage and incremental inference/retrieval costs, plus turnover, exposure-adjusted performance, drawdown, tail loss and latency. Score rejected candidates at fixed horizons as counterfactual research outcomes, clearly separated from realized executable returns. Run repeated model seeds and session-block uncertainty estimates with multiple-testing correction. Use untouched chronological confirmation and forward paper observation; replay alone cannot remove all contamination concerns.

Economic test: incremental value = P&L with retrieval minus P&L without retrieval minus incremental model, storage and operating costs. Engineering cost must also be recovered for positive project ROI. Require repeatable positive incremental value at comparable risk; do not promote on a more persuasive journal, a single seed, higher gross return, or simulated fills alone.

## Implementation priority

1. Link existing findings to measurable outcomes and searchable experiment identities.
2. Evaluate cheap structured/text retrieval in research workflows against existing recent-history context.
3. Add embeddings only if they improve measured retrieval usefulness and downstream economic results over the simpler baseline.
4. Keep any trading-path memory experimental until independent confirmation supports adoption.
