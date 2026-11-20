/*
 * Equity fill model whose LIMIT fills pay the modelled spread/slippage (LEAN's EquityFillModel fills a limit
 * at the limit or at a favourable open, i.e. at trade prices, ignoring the spread). The adjusted price is
 * rounded adversely to the tick and never worse than the limit price. Used identically in the backtest and
 * by LEAN's PaperBrokerage in the simulated live path.
 */

using System;
using QuantConnect.Orders;
using QuantConnect.Orders.Fills;
using QuantConnect.Securities;

namespace Lat.Lean.Costs
{
    public sealed class LatFillModel : EquityFillModel
    {
        public override OrderEvent LimitFill(Security asset, LimitOrder order)
        {
            var fill = base.LimitFill(asset, order);
            if (fill.FillQuantity == 0m || (fill.Status != OrderStatus.Filled && fill.Status != OrderStatus.PartiallyFilled))
            {
                return fill;
            }
            var slip = asset.SlippageModel?.GetSlippageApproximation(asset, order) ?? 0m;
            var tick = asset.SymbolProperties?.MinimumPriceVariation ?? 0.01m;
            fill.FillPrice = AdjustLimitFill(order.Direction == OrderDirection.Buy, fill.FillPrice, slip, order.LimitPrice, tick);
            return fill;
        }

        /// <summary>Pure arithmetic (unit-tested).</summary>
        public static decimal AdjustLimitFill(bool isBuy, decimal basePrice, decimal slippage, decimal limitPrice, decimal tick)
        {
            if (slippage < 0m) slippage = 0m;
            if (tick <= 0m) tick = 0.01m;
            if (isBuy)
            {
                var px = Math.Ceiling((basePrice + slippage) / tick) * tick;
                return Math.Min(px, limitPrice);
            }
            var sp = Math.Floor((basePrice - slippage) / tick) * tick;
            return Math.Max(sp, limitPrice);
        }
    }
}
