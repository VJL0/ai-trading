/*
 * TEST ONLY. A hostile strategy built on the bridge: once the bridge is ready it
 *   1. places an UNSIGNED limit order          -> must be refused by GatedBrokerageModel (algorithm hook),
 *   2. replaces the brokerage model with LEAN's permissive DefaultBrokerageModel (SetBrokerageModel is public),
 *   3. places an UNSIGNED market order          -> must be refused by GatedPaperBrokerage.PlaceOrder (final boundary),
 *   4. places a limit order with a FORGED token -> must be refused by the final boundary.
 * The bridge integrity check then halts intake. No order may ever fill.
 *
 * LEAN signals a synchronous order ticket (OrderSet) BEFORE BrokerageTransactionHandler runs the buying-power and
 * CanSubmitOrder checks on its own thread, so step 1 waits for the first order to leave status New before the
 * model swap; otherwise the swap could race the model hook (the final boundary would still refuse the order).
 */

using System;
using System.Linq;
using System.Threading;
using Lat.Lean.Algorithms;
using Lat.Lean.Gate;
using QuantConnect;
using QuantConnect.Brokerages;
using QuantConnect.Orders;

namespace Lat.Lean.TestAlgorithms
{
    public class LatRogueStrategyAlgorithm : LatIntentBridgeAlgorithm
    {
        protected override void OnBridgeReady()
        {
            var spy = Universe.Values.First();
            var price = Securities[spy].Price;
            Log("ROGUE: attempting unsigned limit order through the gated model");
            var first = LimitOrder(spy, 1, price, tag: "rogue unsigned limit",
                orderProperties: new OrderProperties { TimeInForce = TimeInForce.Day });
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (first.Status == OrderStatus.New && DateTime.UtcNow < deadline)
            {
                Thread.Sleep(20);
            }
            Log($"ROGUE: unsigned limit order {first.OrderId} status {first.Status}");

            Log("ROGUE: replacing the brokerage model with DefaultBrokerageModel");
            SetBrokerageModel(new DefaultBrokerageModel(AccountType.Cash));

            Log("ROGUE: attempting unsigned market order after swapping the model");
            MarketOrder(spy, 1, tag: "rogue unsigned market");

            Log("ROGUE: attempting limit order with a forged token after swapping the model");
            var forged = IntentToken.Encode("lat-intent-v1|intent_id=forged", new string('a', 128));
            LimitOrder(spy, 1, price, tag: forged, orderProperties: new OrderProperties { TimeInForce = TimeInForce.Day });
        }
    }
}
