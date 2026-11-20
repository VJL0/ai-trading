/*
 * GatedBrokerageModel: the LEAN brokerage-model hook (IBrokerageModel.CanSubmitOrder) that refuses ANY
 * order which does not carry a valid ECDSA P-256 signed LAT intent token matching it exactly.
 *
 * LEAN calls CanSubmitOrder from BrokerageTransactionHandler.HandleSubmitOrderRequest before
 * IBrokerage.PlaceOrder. Because a strategy could replace its brokerage model (QCAlgorithm.SetBrokerageModel
 * is public and not locked), the same IntentGate is enforced again at the final engine boundary by
 * GatedPaperBrokerage / GatedAlpacaBrokerage.PlaceOrder.
 *
 * Updates are refused unconditionally: a signed intent is immutable (LEAN skips CanUpdateOrder in live
 * mode, so the gated brokerages refuse UpdateOrder too).
 */

using System;
using Lat.Lean.Costs;
using Lat.Lean.Gate;
using QuantConnect.Brokerages;
using QuantConnect.Orders;
using QuantConnect.Securities;

namespace Lat.Lean.Brokerages
{
    public sealed class GatedBrokerageModel : LatBrokerageModel
    {
        public const string Layer = nameof(GatedBrokerageModel);

        private readonly IntentGate _gate;
        private readonly Func<IOrderProvider> _orderProvider;
        private readonly IGateRefusalSink _sink;

        public IntentGate Gate => _gate;

        public GatedBrokerageModel(IBrokerageModel inner, IntentGate gate, Func<IOrderProvider> orderProvider,
            IGateRefusalSink sink = null, LatCostConfig costs = null)
            : base(inner, costs)
        {
            _gate = gate ?? throw new ArgumentNullException(nameof(gate));
            _orderProvider = orderProvider;
            _sink = sink;
        }

        public override bool CanSubmitOrder(Security security, Order order, out BrokerageMessageEvent message)
        {
            GateDecision decision;
            try
            {
                var facts = LeanOrderFacts.From(order, security, _orderProvider?.Invoke());
                decision = _gate.Evaluate(facts, consume: true);
            }
            catch (Exception e)
            {
                decision = GateDecision.Refuse("gate error (fail closed): " + e.GetType().Name + ": " + e.Message);
            }

            if (!decision.Allowed)
            {
                message = new BrokerageMessageEvent(BrokerageMessageType.Warning, "LatGateRefused", "LAT gate refused order: " + decision.Reason);
                _sink?.Refused(Layer, order?.Id ?? -1, decision, LeanOrderFacts.Describe(order));
                return false;
            }

            // the inner LEAN model (Alpaca/Default) must accept it too; its refusal is also reported
            if (!base.CanSubmitOrder(security, order, out message))
            {
                var reason = "inner brokerage model refused: " + (message?.Message ?? "no reason");
                _sink?.Refused(Layer, order.Id, GateDecision.Refuse(reason, decision.Intent.IntentId, decision.Intent.ClientOrderId), LeanOrderFacts.Describe(order));
                return false;
            }
            message = null;
            return true;
        }

        public override bool CanUpdateOrder(Security security, Order order, UpdateOrderRequest request, out BrokerageMessageEvent message)
        {
            var reason = "order updates are not supported: signed intents are immutable (cancel and submit a new signed intent)";
            message = new BrokerageMessageEvent(BrokerageMessageType.Warning, "LatGateRefused", "LAT gate refused update: " + reason);
            _sink?.Refused(Layer, order?.Id ?? -1, GateDecision.Refuse(reason), LeanOrderFacts.Describe(order));
            return false;
        }
    }
}
