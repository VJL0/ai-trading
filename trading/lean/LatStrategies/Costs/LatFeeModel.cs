/*
 * Alpaca-style US equity fees: $0 commission (configurable) + SEC Section 31 fee on sale proceeds
 * + FINRA TAF per share sold (capped). Buys pay commission only.
 */

using System;
using QuantConnect;
using QuantConnect.Orders;
using QuantConnect.Orders.Fees;
using QuantConnect.Securities;

namespace Lat.Lean.Costs
{
    public sealed class FeeBreakdown
    {
        public decimal Commission { get; init; }
        public decimal SecFee { get; init; }
        public decimal FinraTaf { get; init; }
        public decimal Total => Commission + SecFee + FinraTaf;
    }

    public sealed class LatFeeModel : FeeModel
    {
        public LatCostConfig Config { get; }

        public LatFeeModel(LatCostConfig config = null)
        {
            Config = config ?? LatCostConfig.Default;
        }

        /// <summary>Pure fee arithmetic (unit-tested). <paramref name="shares"/> is absolute, price is the execution price.</summary>
        public FeeBreakdown Compute(bool isSell, decimal shares, decimal price)
        {
            if (shares < 0m) throw new ArgumentOutOfRangeException(nameof(shares));
            if (price < 0m) throw new ArgumentOutOfRangeException(nameof(price));
            if (!isSell || shares == 0m)
            {
                return new FeeBreakdown { Commission = Config.CommissionPerOrder };
            }
            var sec = shares * price * Config.SecFeeRatePerMillion / 1_000_000m;
            var taf = Math.Min(shares * Config.TafPerShare, Config.TafMaxPerTrade);
            if (Config.RoundRegulatoryFeesUpToCent)
            {
                sec = CeilCent(sec);
                taf = Math.Min(CeilCent(taf), Config.TafMaxPerTrade);
            }
            return new FeeBreakdown { Commission = Config.CommissionPerOrder, SecFee = sec, FinraTaf = taf };
        }

        public static decimal CeilCent(decimal value) => Math.Ceiling(value * 100m) / 100m;

        public override OrderFee GetOrderFee(OrderFeeParameters parameters)
        {
            var order = parameters.Order;
            var security = parameters.Security;
            // execution price proxy: the security's current price (LEAN computes fees right after the fill),
            // falling back to the limit price
            var price = security.Price > 0m ? security.Price : (order as LimitOrder)?.LimitPrice ?? 0m;
            var fee = Compute(order.Direction == OrderDirection.Sell, order.AbsoluteQuantity, price);
            return new OrderFee(new CashAmount(fee.Total, Currencies.USD));
        }
    }
}
