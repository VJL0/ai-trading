"""One decision cycle, instrumented stage by stage for the live operations UI.

MARKET_DATA -> FEATURES -> MODELS -> GPT_REVIEW -> RISK_ENGINE -> ORDER_DECISION -> BROKER -> RECONCILIATION

Every artifact shares one causal id and is persisted. The LLM is never invoked per tick: only for a selected candidate,
only when entries are not locked, and only in an AI runtime mode where the app itself calls a model (API_KEYS or
CHATGPT_OAUTH). In CLAUDE_CODE / CODEX / NONE modes GPT_REVIEW is skipped and the quant engine decides alone.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from packages.contracts import QuantSignal, TradeProposal
from packages.contracts.enums import AiRuntimeMode, DecisionAction, Proposer, Side
from packages.core.audit import append_audit, record_failure, set_state
from packages.core.canonical import canonical_json
from packages.core.db import utc_iso
from packages.core.ids import new_causal_id, new_id
from packages.core.logging import log_event
from packages.core.money import BPS, floor_shares, round_price

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger("lat.pipeline")

LOOKBACK_DAYS = 10
APP_CALLS_MODEL = frozenset({AiRuntimeMode.API_KEYS, AiRuntimeMode.CHATGPT_OAUTH})


def _f(d: Decimal | None) -> str | None:
    return None if d is None else format(d, "f")


class DecisionCycle:
    def __init__(self, rt: Runtime, *, session_id: str | None = None) -> None:
        self.rt = rt
        self.session_id = session_id

    async def run(self, *, symbols: list[str], use_ai: bool, actor: str) -> dict[str, Any]:
        rt, act = self.rt, self.rt.activity
        causal_id = new_causal_id()
        now = rt.now()
        locked, lock_reasons = rt.entries_status()
        summary: dict[str, Any] = {"causal_id": causal_id, "ran_at": utc_iso(now), "actor": actor,
                                   "session_id": self.session_id, "execution_label": rt.execution_label.value,
                                   "symbols": symbols, "entries_locked": locked, "lock_reasons": lock_reasons}
        act.begin_cycle(causal_id, causal_id, self.session_id)
        stage = "MARKET_DATA"
        try:
            # 1. MARKET_DATA: point-in-time bars + quotes
            act.stage(stage, "running", item=",".join(symbols))
            act.activity("scan", f"Scanning {len(symbols)} symbols: {', '.join(symbols)}", actor=actor, causal_id=causal_id)
            bars_df = rt.store.load_bars_df(symbols, now - timedelta(days=LOOKBACK_DAYS), now, "1Min",
                                            feed=rt.data_feed, asof=now,
                                            dataset_version=rt.dataset_version if rt.sim_clock is not None else None)
            if bars_df is None or len(bars_df) == 0:
                act.stage(stage, "error", item=",".join(symbols), detail="no point-in-time bars available")
                return await self._no_trade(summary, "no point-in-time market data available", stage_error=True)
            spreads = rt.quote_spreads_bps(symbols)
            act.stage(stage, "passed", item=",".join(symbols),
                      detail=f"{len(bars_df)} bars, {len(spreads)} fresh quotes, feed {rt.feed_mode.value}")
            # 2. FEATURES + 3. MODELS (the signal engine computes features, then the selected model suite scores them)
            stage = "FEATURES"
            act.stage(stage, "running", item=",".join(symbols))
            engine = rt.current_signal_engine()
            act.stage(stage, "passed", item=",".join(symbols), detail="point-in-time features computed")
            stage = "MODELS"
            act.stage(stage, "running", item=",".join(symbols))
            act.activity("predict", "Generating predictions with the walk-forward-selected model suite",
                         actor=actor, causal_id=causal_id)
            signals: list[QuantSignal] = engine.generate(bars_df, now, symbols, causal_id, quote_spreads_bps=spreads)
            self._persist_signals(causal_id, signals)
            candidate = next((s for s in signals if s.is_candidate), None)
            preds = rt.prediction_for(candidate.symbol if candidate else None, bars_df)
            summary["candidate_symbol"] = candidate.symbol if candidate else None
            summary["signals"] = len(signals)
            act.stage(stage, "passed", item=candidate.symbol if candidate else None,
                      detail=(f"candidate {candidate.symbol} net edge {candidate.net_edge_bps} bps" if candidate
                              else f"{len(signals)} scored, none cleared the net-of-cost threshold"))
            from services.market_data.snapshot import build_market_snapshot

            snap_symbol = candidate.symbol if candidate else symbols[0]
            if rt.sim_clock is None:
                snapshot = await rt.market_snapshot(snap_symbol)
            else:
                snapshot = await build_market_snapshot(rt.provider, snap_symbol, now, rt.feed_mode,
                                                       dataset_version=rt.dataset_version)
            rt.ledger.record_market_snapshot(snapshot)
            if candidate is not None:
                act.activity("evaluate", f"Evaluating candidate {candidate.symbol}: expected "
                             f"{candidate.expected_return_bps} bps, cost {candidate.est_total_cost_bps} bps",
                             actor=actor, causal_id=causal_id)
            # 4. GPT_REVIEW
            stage = "GPT_REVIEW"
            from services.ai.evidence import build_internal_evidence

            evidence = build_internal_evidence(candidate, snapshot, rt.policy) if candidate else []
            mode = rt.settings.ai_runtime
            if candidate is not None and use_ai and not locked and mode in APP_CALLS_MODEL:
                act.stage(stage, "running", item=candidate.symbol)
                act.activity("gpt_call", f"Requesting model review of {candidate.symbol} ({mode.value}, "
                             f"transport {rt.settings.ai_transport.upper()})", actor=actor, causal_id=causal_id)
                result = await rt.ai_review(causal_id=causal_id, candidate=candidate, snapshot=snapshot, evidence=evidence)
                proposal: TradeProposal = result.proposal
                summary["model_runs"] = [r.run_id for r in result.model_runs]
                act.stage(stage, "passed" if proposal.action == DecisionAction.TRADE else "rejected",
                          item=candidate.symbol, detail=proposal.no_trade_reason or "model review supports the trade")
            else:
                why = ("no candidate" if candidate is None else "AI disabled for this cycle" if not use_ai
                       else "entries locked" if locked else f"runtime {mode.value}: the app does not call a model")
                act.stage(stage, "skipped", item=candidate.symbol if candidate else None, detail=why)
                proposal = self._quant_proposal(causal_id, candidate, snapshot, evidence, why)
            rt.ledger.record_proposal(proposal)
            summary["proposal"] = proposal.model_dump(mode="json")
            if proposal.action == DecisionAction.NO_TRADE:
                for s in ("RISK_ENGINE", "BROKER"):
                    act.stage(s, "skipped", detail="no trade requested")
                act.stage("ORDER_DECISION", "passed", item=snap_symbol, detail=f"NO_TRADE: {proposal.no_trade_reason}")
                self._card(candidate, preds, proposal, "NO_TRADE", proposal.no_trade_reason or "no trade", [])
                return await self._finish(summary | {"outcome": "NO_TRADE", "reason": proposal.no_trade_reason})
            # 5. RISK_ENGINE (independent, deterministic; recomputes everything)
            stage = "RISK_ENGINE"
            act.stage(stage, "running", item=proposal.symbol)
            ctx = await rt.risk_context(snapshot)
            decision = rt.risk.evaluate(proposal, ctx)
            rt.ledger.record_decision(decision)
            summary["decision"] = decision.model_dump(mode="json")
            failures = [f"{c.name}: {c.detail}" for c in decision.checks if not c.passed]
            if not decision.approved:
                act.stage(stage, "rejected", item=proposal.symbol, detail="; ".join(failures)[:500])
                act.activity("reject", f"Risk engine rejected {proposal.side} {proposal.quantity} {proposal.symbol}: "
                             f"{failures[0] if failures else 'rejected'}", actor="risk-engine", causal_id=causal_id)
                act.stage("ORDER_DECISION", "rejected", item=proposal.symbol, detail="risk rejected")
                act.stage("BROKER", "skipped", detail="nothing to submit")
                self._card(candidate, preds, proposal, "REJECTED", "; ".join(decision.reasons), failures)
                return await self._finish(summary | {"outcome": "REJECTED", "reason": "; ".join(decision.reasons)})
            act.stage(stage, "passed", item=proposal.symbol, detail=f"all {len(decision.checks)} checks passed")
            # 6. ORDER_DECISION -> signed intent
            stage = "ORDER_DECISION"
            act.stage(stage, "running", item=proposal.symbol)
            intent = rt.risk.authorize(proposal, decision, ctx)
            rt.ledger.record_intent(intent)
            summary["client_order_id"] = intent.client_order_id
            act.stage(stage, "passed", item=proposal.symbol,
                      detail=f"{intent.side} {intent.quantity} {intent.symbol} @ {intent.limit_price} (signed intent)")
            self._card(candidate, preds, proposal, "APPROVED", proposal.rationale or "risk-approved", [])
            if intent.requires_human_approval:
                act.stage("BROKER", "skipped", detail="awaiting per-order human approval")
                return await self._finish(summary | {"outcome": "AWAITING_HUMAN_APPROVAL"})
            # 7. BROKER (single order path; final gate inside ExecutionService)
            stage = "BROKER"
            act.stage(stage, "running", item=intent.client_order_id)
            act.activity("submit", f"Submitting {intent.side} {intent.quantity} {intent.symbol} limit {intent.limit_price} "
                         f"to {rt.settings.gateway.value} ({rt.execution_label.value})", actor=actor, causal_id=causal_id)
            async with rt.submit_lock:
                await rt.prepare_gate(snapshot)
                order = await rt.execution.submit(intent)
            refreshed = rt.ledger.get_order(intent.client_order_id)
            state = getattr(getattr(refreshed, "state", None), "value", None)
            summary["order"] = _jsonable(refreshed if refreshed is not None else order)
            ok = state not in (None, "REJECTED")
            act.stage(stage, "passed" if ok else "rejected", item=intent.client_order_id, detail=f"order state {state}")
            self._card(candidate, preds, proposal, "SUBMITTED" if ok else "REJECTED",
                       proposal.rationale or "submitted", [] if ok else [f"broker: {state}"])
            return await self._finish(summary | {"outcome": "SUBMITTED" if ok else "BLOCKED"})
        except Exception as exc:
            log_event(log, "decision cycle failed", logging.ERROR, causal_id=causal_id, error=repr(exc))
            err_kind, err_detail = type(exc).__name__, repr(exc)
            rt.db.write(lambda c: record_failure(c, component="pipeline", kind=err_kind, detail=err_detail,
                                                 causal_id=causal_id))
            act.stage(stage, "error", detail=f"{err_kind}: {exc}"[:500])
            act.activity("error", f"Cycle failed at {stage}: {err_kind}: {exc}"[:400], actor=actor, causal_id=causal_id)
            self._card(None, None, None, "ERROR", f"{err_kind} at {stage}", [])
            return await self._finish(summary | {"outcome": "ERROR", "reason": f"{err_kind}: {exc}"}, reconcile=False)

    # ------------------------------------------------------------------ helpers
    def _quant_proposal(self, causal_id: str, candidate: QuantSignal | None, snapshot: Any, evidence: list[Any],
                        note: str) -> TradeProposal:
        rt = self.rt
        now = rt.now()
        proposer = Proposer.QUANT_MODEL if rt.model_suite_active() else Proposer.QUANT_BASELINE
        base: dict[str, Any] = dict(
            proposal_id=new_id("prop"), causal_id=causal_id, created_at=now, data_asof=snapshot.data_asof,
            proposer=proposer, dataset_version=rt.dataset_version, policy_version_seen=rt.policy.version,
            strategy_id="quant-model-suite" if proposer == Proposer.QUANT_MODEL else "quant-baseline-v1")
        if candidate is None:
            return TradeProposal(**base, action=DecisionAction.NO_TRADE,
                                 no_trade_reason="no candidate cleared the net-of-cost edge threshold")
        quote = snapshot.quote
        if quote is None:
            return TradeProposal(**base, action=DecisionAction.NO_TRADE, no_trade_reason="no fresh quote")
        limit = round_price(quote.ask, side_buy=True)
        qty = floor_shares(rt.policy.max_order_notional * Decimal("0.5") / limit)
        if qty <= 0:
            return TradeProposal(**base, action=DecisionAction.NO_TRADE, no_trade_reason="budget below one share")
        stop = round_price(limit * (1 - rt.policy.default_stop_loss_bps / BPS), side_buy=False)
        return TradeProposal(
            **base, action=DecisionAction.TRADE, symbol=candidate.symbol, side=Side.BUY, quantity=qty,
            limit_price=limit, stop_price=stop, max_planned_loss=qty * (limit - stop),
            evidence_ids=tuple(e.evidence_id for e in evidence) or (candidate.signal_id,),
            signal_id=candidate.signal_id,
            rationale=(f"{candidate.strategy_id}: expected {candidate.expected_return_bps} bps, cost "
                       f"{candidate.est_total_cost_bps} bps, net edge {candidate.net_edge_bps} bps ({note})"),
        )

    def _card(self, candidate: QuantSignal | None, preds: Any, proposal: TradeProposal | None, status: str,
              reason: str, failures: list[str]) -> None:
        rt = self.rt
        action = "NO_TRADE"
        if proposal is not None and proposal.action == DecisionAction.TRADE and proposal.side is not None:
            action = proposal.side.value
        conf, calibrated = None, False
        if preds is not None and preds.success_probability is not None:
            conf, calibrated = _f(preds.success_probability), preds.probability_calibrated
        elif candidate is not None and candidate.probability_up is not None:
            conf, calibrated = _f(candidate.probability_up), candidate.probability_calibrated
        symbol = (proposal.symbol if proposal else None) or (candidate.symbol if candidate else None)
        rt.activity.decision({
            "ts": rt.now_iso(), "symbol": symbol, "side": action if action != "NO_TRADE" else None,
            "action": action, "status": status,
            "expected_return_bps": _f(candidate.expected_return_bps) if candidate else None,
            "expected_cost_bps": _f(candidate.est_total_cost_bps) if candidate else None,
            "net_edge_bps": _f(candidate.net_edge_bps) if candidate else None,
            "confidence": conf, "confidence_calibrated": calibrated,
            "volatility_bps": _f(preds.vol_forecast_bps) if preds is not None else None,
            "regime": preds.regime.value if preds is not None else "UNKNOWN",
            "reason": reason[:600], "risk_failures": failures[:20],
            "proposer": proposal.proposer.value if proposal else None,
            "is_simulated": rt.execution_label.value == "SIMULATED" or rt.sim_clock is not None,
        })

    async def _no_trade(self, summary: dict[str, Any], reason: str, *, stage_error: bool = False) -> dict[str, Any]:
        for s in ("FEATURES", "MODELS", "GPT_REVIEW", "RISK_ENGINE", "BROKER"):
            self.rt.activity.stage(s, "skipped", detail=reason)
        self.rt.activity.stage("ORDER_DECISION", "passed", detail=f"NO_TRADE: {reason}")
        self._card(None, None, None, "ERROR" if stage_error else "NO_TRADE", reason, [])
        return await self._finish(summary | {"outcome": "NO_TRADE", "reason": reason})

    def _persist_signals(self, causal_id: str, signals: list[QuantSignal]) -> None:
        def _w(conn: Any) -> None:
            for s in signals:
                conn.execute(
                    "INSERT OR IGNORE INTO quant_signals(signal_id, causal_id, strategy_id, symbol, generated_at,"
                    " is_candidate, net_edge_bps, dataset_version, payload_json) VALUES (?,?,?,?,?,?,?,?,?)",
                    (s.signal_id, causal_id, s.strategy_id, s.symbol,
                     utc_iso(s.generated_at), int(s.is_candidate), format(s.net_edge_bps, "f"), s.dataset_version,
                     canonical_json(s)),
                )

        if signals:
            self.rt.db.write(_w)

    async def _finish(self, summary: dict[str, Any], *, reconcile: bool = True) -> dict[str, Any]:
        rt, act = self.rt, self.rt.activity
        if reconcile:  # 8. RECONCILIATION: broker-authoritative check after every cycle
            act.stage("RECONCILIATION", "running")
            try:
                report = await rt.reconcile()
                act.stage("RECONCILIATION", "passed" if report.status.value == "CLEAN" else "rejected",
                          detail=f"{report.status.value}, {len(report.mismatches)} mismatches")
                act.activity("reconcile", f"Reconciled positions and orders with {rt.settings.gateway.value}: "
                             f"{report.status.value}", causal_id=summary["causal_id"])
            except Exception as exc:
                act.stage("RECONCILIATION", "error", detail=f"{type(exc).__name__}: {exc}"[:500])
        brief = {k: summary.get(k) for k in ("causal_id", "ran_at", "candidate_symbol", "outcome", "reason",
                                             "client_order_id", "execution_label", "session_id")}

        def _w(conn: Any) -> None:
            set_state(conn, "last_cycle", brief)
            append_audit(conn, event_type="DECISION_CYCLE", source="pipeline", mode=rt.settings.mode.value,
                         payload=brief, causal_id=summary["causal_id"], dataset_version=rt.dataset_version,
                         policy_version=rt.policy.version)

        rt.db.write(_w)
        act.end_cycle()
        return summary


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    dump = getattr(obj, "model_dump", None)
    if dump is not None:
        return dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return str(obj)
