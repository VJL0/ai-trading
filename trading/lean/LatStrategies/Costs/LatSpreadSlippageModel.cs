/*
 * Slippage = half the bid/ask spread + adverse impact in bps, in price units.
 * With no valid quote, half of an assumed spread (bps of price) is used. For MARKET orders that already
 * cross a known quote in the fill model only the impact part is returned (no double counting); the LAT
 * paths only use LIMIT orders, whose fills LatFillModel adjusts with this model (capped at the limit).
 */

using QuantConnect.Orders;
using QuantConnect.Orders.Slippage;
using QuantConnect.Securities;

namespace Lat.Lean.Costs
{
    public sealed class LatSpreadSlippageModel : ISlippageModel
    {
        public LatCostConfig Config { get; }

        public LatSpreadSlippageModel(LatCostConfig config = null)
        {
            Config = config ?? LatCostConfig.Default;
        }

        /// <summary>Pure arithmetic (unit-tested).</summary>
        public decimal Compute(decimal price, decimal bid, decimal ask, bool marketOrderCrossesQuote)
        {
            if (price <= 0m) return 0m;
            var hasQuote = bid > 0m && ask > bid;
            var impact = price * Config.SlippageBps / 10_000m;
            if (hasQuote && marketOrderCrossesQuote) return impact;
            var halfSpread = hasQuote ? (ask - bid) / 2m : price * Config.AssumedSpreadBps / 2m / 10_000m;
            return halfSpread + impact;
        }

        public decimal GetSlippageApproximation(Security asset, Order order)
        {
            var price = asset.Price;
            return Compute(price, asset.BidPrice, asset.AskPrice, order?.Type == OrderType.Market);
        }
    }
}
