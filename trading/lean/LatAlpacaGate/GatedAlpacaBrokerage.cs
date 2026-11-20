/*
 * GatedAlpacaBrokerage: the official QuantConnect Alpaca brokerage (LEAN plugin, pinned source) with
 *   * paper trading HARD-CODED (isPaperTrading: true) - the 'alpaca-paper-trading' setting is ignored,
 *     and any configuration claiming otherwise is refused;
 *   * the LAT IntentGate enforced at PlaceOrder (final engine boundary) and UpdateOrder refused.
 * Selected with "live-mode-brokerage": "GatedAlpacaBrokerage" and "data-queue-handler": ["GatedAlpacaBrokerage"]
 * (the factory registers the same instance as the data queue handler, so there is one Alpaca connection).
 *
 * Client order ids: the plugin's CreateAlpacaOrder() never sets Alpaca's client_order_id, so Alpaca assigns
 * its own. Reconciliation therefore maps our client_order_id -> LEAN order id -> Alpaca order id
 * (order.BrokerId, reported in every /engine/events payload as broker_order_id) + intent_id.
 *
 * STATUS: compiles against the pinned plugin source; NOT exercised (no Alpaca paper keys, and the plugin's
 * QuantConnect module-licence check needs a QuantConnect account - see trading/README.md "Blocked").
 */

using System;
using System.Collections.Generic;
using Lat.Lean.Bridge;
using Lat.Lean.Brokerages;
using Lat.Lean.Costs;
using Lat.Lean.Gate;
using QuantConnect;
using QuantConnect.Brokerages;
using QuantConnect.Brokerages.Alpaca;
using QuantConnect.Configuration;
using QuantConnect.Interfaces;
using QuantConnect.Logging;
using QuantConnect.Orders;
using QuantConnect.Packets;
using QuantConnect.Securities;
using QuantConnect.Util;

namespace Lat.Lean.Alpaca
{
    [BrokerageFactory(typeof(GatedAlpacaBrokerageFactory))]
    public class GatedAlpacaBrokerage : AlpacaBrokerage
    {
        public const string Layer = nameof(GatedAlpacaBrokerage);
        public const bool PaperTradingHardCoded = true;

        private readonly IntentGate _gate;
        private readonly IGateRefusalSink _sink;
        private readonly IAlgorithm _algorithm;

        /// <summary>Required by MEF for IDataQueueHandler discovery; an instance built this way is uninitialised and
        /// cannot trade (fail closed). The factory always uses the full constructor.</summary>
        public GatedAlpacaBrokerage()
        {
        }

        public GatedAlpacaBrokerage(string apiKey, string apiSecret, IAlgorithm algorithm, IntentGate gate, IGateRefusalSink sink)
            : base(apiKey, apiSecret, accessToken: null, isPaperTrading: PaperTradingHardCoded, algorithm)
        {
            _gate = gate ?? throw new ArgumentNullException(nameof(gate));
            _sink = sink;
            _algorithm = algorithm;
            Log.Trace($"GatedAlpacaBrokerage(): Alpaca PAPER endpoint (hard-coded), intent key {_gate.KeyId}");
        }

        public override bool PlaceOrder(Order order)
        {
            if (_gate == null)
            {
                return false; // uninitialised (MEF) instance: never trades
            }
            var decision = GateEvaluation.Evaluate(_gate, order, _algorithm);
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
            _sink?.Refused(Layer, order?.Id ?? -1, GateDecision.Refuse("order updates are not supported: signed intents are immutable"), LeanOrderFacts.Describe(order));
            return false;
        }
    }

    public class GatedAlpacaBrokerageFactory : BrokerageFactory
    {
        public GatedAlpacaBrokerageFactory() : base(typeof(GatedAlpacaBrokerage))
        {
        }

        public override Dictionary<string, string> BrokerageData => new Dictionary<string, string>
        {
            { "alpaca-api-key", Config.Get("alpaca-api-key") },
            { "alpaca-api-secret", Config.Get("alpaca-api-secret") },
            { "alpaca-paper-trading", "true" },
        };

        public override IBrokerageModel GetBrokerageModel(IOrderProvider orderProvider)
        {
            var gate = LatGateFactory.Create(LatGateway.LEAN_ALPACA_PAPER, k => Config.Get(k));
            return new GatedBrokerageModel(new AlpacaBrokerageModel(), gate, () => orderProvider, null, LatCostConfig.FromLookup(k => Config.Get(k)));
        }

        public override IBrokerage CreateBrokerage(LiveNodePacket job, IAlgorithm algorithm)
        {
            if (!string.Equals(Config.Get("alpaca-paper-trading"), "true", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("GatedAlpacaBrokerage: alpaca-paper-trading must be true; live trading is not part of this build");
            }
            job.BrokerageData.TryGetValue("alpaca-api-key", out var key);
            job.BrokerageData.TryGetValue("alpaca-api-secret", out var secret);
            if (string.IsNullOrEmpty(key) || string.IsNullOrEmpty(secret))
            {
                throw new InvalidOperationException("GatedAlpacaBrokerage: Alpaca PAPER key id/secret missing (BLOCKED)");
            }
            if (key.StartsWith("AK", StringComparison.Ordinal))
            {
                throw new InvalidOperationException("GatedAlpacaBrokerage: key id looks like a LIVE Alpaca key (AK...); refused");
            }
            Func<string, string> get = k => Config.Get(k);
            var gate = LatGateFactory.Create(LatGateway.LEAN_ALPACA_PAPER, get);
            var settings = BridgeSettings.FromConfig(get);
            if (settings.Gateway != LatGateway.LEAN_ALPACA_PAPER)
            {
                throw new InvalidOperationException("GatedAlpacaBrokerage requires lat-gateway=LEAN_ALPACA_PAPER");
            }
            var sink = new BridgeClient(settings, m => Log.Trace(m)) { AlgorithmName = "GatedAlpacaBrokerage" };
            sink.Start(withPoller: false);
            var brokerage = new GatedAlpacaBrokerage(key, secret, algorithm, gate, sink);
            Composer.Instance.AddPart<IDataQueueHandler>(brokerage);
            return brokerage;
        }

        public override void Dispose()
        {
        }
    }
}
