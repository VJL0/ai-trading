/*
 * GatedPaperBrokerage: LEAN's PaperBrokerage (local simulated fills, no external broker) with the LAT
 * IntentGate enforced at IBrokerage.PlaceOrder - the final engine boundary every LEAN order passes,
 * regardless of which brokerage model the algorithm installed. Execution label: SIMULATED.
 *
 * Selected with config "live-mode-brokerage": "GatedPaperBrokerage" (see GatedPaperBrokerageFactory).
 */

using System;
using Lat.Lean.Bridge;
using Lat.Lean.Gate;
using QuantConnect;
using QuantConnect.Brokerages;
using QuantConnect.Brokerages.Paper;
using QuantConnect.Interfaces;
using QuantConnect.Logging;
using QuantConnect.Orders;
using QuantConnect.Packets;
using QuantConnect.Securities;

namespace Lat.Lean.Brokerages
{
    [BrokerageFactory(typeof(GatedPaperBrokerageFactory))]
    public class GatedPaperBrokerage : PaperBrokerage
    {
        public const string Layer = nameof(GatedPaperBrokerage);

        private readonly IntentGate _gate;
        private readonly IGateRefusalSink _sink;

        public GatedPaperBrokerage(IAlgorithm algorithm, LiveNodePacket job, IntentGate gate, IGateRefusalSink sink)
            : base(algorithm, job)
        {
            _gate = gate ?? throw new ArgumentNullException(nameof(gate));
            _sink = sink;
            Log.Trace($"GatedPaperBrokerage(): SIMULATED paper brokerage, intent key {_gate.KeyId}, gateway {_gate.Gateway}");
        }

        public override bool PlaceOrder(Order order)
        {
            var decision = GateEvaluation.Evaluate(_gate, order, Algorithm);
            if (!decision.Allowed)
            {
                _sink?.Refused(Layer, order?.Id ?? -1, decision, LeanOrderFacts.Describe(order));
                OnMessage(new BrokerageMessageEvent(BrokerageMessageType.Warning, "LatGateRefused", $"{Layer} refused order {order?.Id}: {decision.Reason}"));
                return false;
            }
            return base.PlaceOrder(order);
        }

        /// <summary>Drains pending refusal reports (bounded) before LEAN disposes the brokerage.</summary>
        public override void Dispose()
        {
            (_sink as BridgeClient)?.Stop(TimeSpan.FromSeconds(3));
            base.Dispose();
        }

        public override bool UpdateOrder(Order order)
        {
            var decision = GateDecision.Refuse("order updates are not supported: signed intents are immutable");
            _sink?.Refused(Layer, order?.Id ?? -1, decision, LeanOrderFacts.Describe(order));
            OnMessage(new BrokerageMessageEvent(BrokerageMessageType.Warning, "LatGateRefused", $"{Layer} refused update of order {order?.Id}"));
            return false;
        }
    }

    /// <summary>Shared evaluation used by the gated brokerages (never throws: errors refuse).</summary>
    public static class GateEvaluation
    {
        public static GateDecision Evaluate(IntentGate gate, Order order, IAlgorithm algorithm)
        {
            try
            {
                if (order == null) return GateDecision.Refuse("no order");
                Security security = null;
                if (algorithm?.Securities != null && !algorithm.Securities.TryGetValue(order.Symbol, out security))
                {
                    return GateDecision.Refuse($"security {order.Symbol} is not subscribed");
                }
                var facts = LeanOrderFacts.From(order, security, algorithm?.Transactions);
                return gate.Evaluate(facts, consume: true);
            }
            catch (Exception e)
            {
                return GateDecision.Refuse("gate error (fail closed): " + e.GetType().Name + ": " + e.Message);
            }
        }
    }

    public class GatedPaperBrokerageFactory : BrokerageFactory
    {
        public GatedPaperBrokerageFactory() : base(typeof(GatedPaperBrokerage))
        {
        }

        public override System.Collections.Generic.Dictionary<string, string> BrokerageData => new System.Collections.Generic.Dictionary<string, string>
        {
            { "live-cash-balance", QuantConnect.Configuration.Config.Get("live-cash-balance") },
            { "live-holdings", QuantConnect.Configuration.Config.Get("live-holdings") },
        };

        /// <summary>Even an algorithm that never sets a model gets the gated one (with the PaperBrokerage default inside).</summary>
        public override IBrokerageModel GetBrokerageModel(IOrderProvider orderProvider)
        {
            var gate = LatGateFactory.Create(LatGateway.LEAN_PAPER_SIM, k => QuantConnect.Configuration.Config.Get(k));
            return new GatedBrokerageModel(new DefaultBrokerageModel(AccountType.Cash), gate, () => orderProvider, null,
                Costs.LatCostConfig.FromLookup(k => QuantConnect.Configuration.Config.Get(k)));
        }

        public override IBrokerage CreateBrokerage(LiveNodePacket job, IAlgorithm algorithm)
        {
            var get = new Func<string, string>(k => QuantConnect.Configuration.Config.Get(k));
            var gate = LatGateFactory.Create(LatGateway.LEAN_PAPER_SIM, get);
            var settings = BridgeSettings.FromConfig(get);
            if (settings.Gateway != LatGateway.LEAN_PAPER_SIM)
            {
                throw new InvalidOperationException("GatedPaperBrokerage requires lat-gateway=LEAN_PAPER_SIM");
            }
            var sink = new BridgeClient(settings, m => Log.Trace(m)) { AlgorithmName = "GatedPaperBrokerage" };
            sink.Start(withPoller: false);
            return new GatedPaperBrokerage(algorithm, job, gate, sink);
        }

        public override void Dispose()
        {
        }
    }
}
