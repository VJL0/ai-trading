"""Offline end-to-end risk -> signed intent -> real SDK HTTP -> durable fills/restart.

Only broker HTTP responses and market quotes are fixtures; authorization, signature
verification, lifecycle, SQLite, reconciliation and audit chain are production code.
This is not evidence of an order executed at Alpaca's hosted paper endpoint.
"""
from __future__ import annotations

from pathlib import Path

from packages.contracts.enums import DataFeed, GatewayKind, OrderState, ReconciliationStatus, TradingMode
from packages.core.audit import verify_chain
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.signing import IntentSigner
from services.order_ledger.alpaca_gateway import AlpacaPaperGateway
from services.order_ledger.entries import EntriesLock
from services.order_ledger.execution import ExecutionService
from services.order_ledger.ledger import OrderLedger
from services.order_ledger.reconcile import Reconciler
from services.risk.engine import RiskEngine
from services.risk.policy_store import PolicyStore
from tests.unit.order_ledger.ol_support import paper_secret_store
from tests.unit.order_ledger.test_alpaca_gateway import BrokerHTTP
from tests.unit.risk.risk_factories import NOW, make_ctx, make_market, make_proposal


async def test_risk_sdk_fill_and_durable_restart(tmp_path: Path) -> None:
    clock = ManualClock(NOW)
    signer = IntentSigner.load_or_create(tmp_path / 'keys')
    http = BrokerHTTP()
    db = Database(tmp_path / 'ledger.sqlite')
    try:
        ledger = OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier)
        entries = EntriesLock(db, TradingMode.PAPER, clock=clock)
        gateway = AlpacaPaperGateway(paper_secret_store(), ledger=ledger, verifier=signer.verifier,
                                    clock=clock, http_session=http)
        reconciler = Reconciler(db, ledger, gateway, entries, clock)
        assert (await reconciler.startup_recovery()).status == ReconciliationStatus.CLEAN
        policies = PolicyStore(db, TradingMode.PAPER)
        policies.ensure_default()
        policy = policies.active()
        risk = RiskEngine(policy, signer, clock)
        market = make_market(feed=DataFeed.IEX)
        ctx = make_ctx(gateway=GatewayKind.ALPACA_PAPER, market=market, account=await gateway.get_account())
        proposal = make_proposal()
        decision = risk.evaluate(proposal, ctx)
        assert decision.approved, decision.reasons
        intent = risk.authorize(proposal, decision, ctx)
        ledger.record_market_snapshot(market)
        ledger.record_proposal(proposal)
        ledger.record_decision(decision)
        ledger.record_intent(intent)
        gate_ctx = ctx.model_copy(update={'open_orders': tuple(ledger.open_orders())})
        execution = ExecutionService(ledger, gateway, lambda i: risk.final_gate(
            i, gate_ctx, signer.verifier, policy.version), clock, entries_lock=entries)
        order = await execution.submit(intent)
        assert order is not None, ledger.get_record(intent.client_order_id)
        assert intent.client_order_id.startswith('lat-paper-')
        http.orders[intent.client_order_id].update(status='filled', filled_qty=str(intent.quantity),
                                                  filled_avg_price='100')
        # Close SQLite before processing the broker fill, reproducing a process crash.
    finally:
        db.close()
    reopened = Database(tmp_path / 'ledger.sqlite')
    try:
        ledger = OrderLedger(reopened, clock, TradingMode.PAPER, verifier=signer.verifier)
        entries = EntriesLock(reopened, TradingMode.PAPER, clock=clock)
        assert entries.is_locked()[0]
        gateway = AlpacaPaperGateway(paper_secret_store(), ledger=ledger, verifier=signer.verifier,
                                    clock=clock, http_session=http)
        reconciler = Reconciler(reopened, ledger, gateway, entries, clock)
        report = await reconciler.startup_recovery()
        assert report.status == ReconciliationStatus.CLEAN, report
        assert ledger.get_order(intent.client_order_id).state == OrderState.FILLED
        assert ledger.position_quantities() == {'SPY': intent.quantity}
        assert len(http.posts) == 1
        assert (await reconciler.reconcile()).status == ReconciliationStatus.CLEAN
        assert ledger.position_quantities() == {'SPY': intent.quantity}
        assert verify_chain(reopened)[0]
        assert not entries.is_locked()[0]
    finally:
        reopened.close()
