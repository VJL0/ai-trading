/*
 * Extracts the facts the IntentGate decides on from live LEAN objects (Order + Security + open order tickets).
 * Anything that cannot be inspected is reported as unsupported so the gate refuses it.
 */

using System;
using System.Linq;
using QuantConnect;
using QuantConnect.Orders;
using QuantConnect.Orders.TimeInForces;
using QuantConnect.Interfaces;
using QuantConnect.Securities;

namespace Lat.Lean.Gate
{
    public static class LeanOrderFacts
    {
        public static OrderFacts From(Order order, Security security, IOrderProvider orderProvider)
        {
            if (order == null) throw new ArgumentNullException(nameof(order));
            var facts = new OrderFacts
            {
                LeanOrderId = order.Id,
                Tag = order.Tag,
                Symbol = order.Symbol?.Value,
                SecurityType = order.SecurityType.ToString(),
                Market = order.Symbol?.ID.Market,
                OrderType = order.Type.ToString(),
                Quantity = order.Quantity,
                LimitPrice = (order as LimitOrder)?.LimitPrice,
                TimeInForce = DescribeTimeInForce(order.TimeInForce),
                HasContingencyOrGroup = order.Contingency != null || order.GroupOrderManager != null,
                ExchangeOpen = IsExchangeOpen(security),
                HeldQuantity = security?.Holdings?.Quantity ?? 0m,
            };

            // only the plain LEAN OrderProperties (or Alpaca's, with extended hours OFF) are understood
            var props = order.Properties;
            if (props != null)
            {
                var typeName = props.GetType().Name;
                if (typeName == "AlpacaOrderProperties")
                {
                    facts.OutsideRegularTradingHours = ReadOutsideRegularTradingHours(props, out var readable);
                    if (!readable) facts.UnsupportedPropertiesType = typeName + " (unreadable OutsideRegularTradingHours)";
                }
                else if (props.GetType() != typeof(OrderProperties))
                {
                    facts.UnsupportedPropertiesType = props.GetType().FullName;
                }
            }

            if (security == null)
            {
                facts.SecurityType = "Unknown";
            }

            if (orderProvider != null && order.Symbol != null)
            {
                decimal pending = 0m;
                foreach (var t in orderProvider.GetOpenOrderTickets(t => t.Symbol == order.Symbol && t.OrderId != order.Id))
                {
                    var remaining = t.Quantity - t.QuantityFilled;
                    if (remaining < 0m) pending += -remaining;
                }
                facts.PendingSellQuantity = pending;
            }
            else
            {
                // cannot see other open orders: assume the worst for sells
                facts.PendingSellQuantity = decimal.MaxValue / 4;
            }
            return facts;
        }

        /// <summary>Regular-session state; any error (e.g. no clock attached) reads as closed.</summary>
        public static bool IsExchangeOpen(Security security)
        {
            try
            {
                return security?.Exchange != null && security.Exchange.ExchangeOpen;
            }
            catch (Exception)
            {
                return false;
            }
        }

        public static string DescribeTimeInForce(TimeInForce tif)
        {
            switch (tif)
            {
                case DayTimeInForce _: return "DAY";
                case GoodTilCanceledTimeInForce _: return "GTC";
                case GoodTilDateTimeInForce _: return "GTD";
                case null: return "NONE";
                default: return tif.GetType().Name;
            }
        }

        private static bool ReadOutsideRegularTradingHours(IOrderProperties props, out bool readable)
        {
            var p = props.GetType().GetProperty("OutsideRegularTradingHours");
            if (p == null || p.PropertyType != typeof(bool))
            {
                readable = false;
                return true;
            }
            readable = true;
            return (bool)p.GetValue(props);
        }

        /// <summary>Best effort summary for logs and refusal reports (never trusted for decisions).</summary>
        public static string Describe(Order order) =>
            order == null ? "<null order>" :
            $"LEAN order {order.Id} {order.Type} {order.Symbol?.Value} qty={order.Quantity} " +
            $"limit={(order as LimitOrder)?.LimitPrice.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "-"} " +
            $"tif={DescribeTimeInForce(order.TimeInForce)} tagged={(order.Tag ?? string.Empty).StartsWith(IntentToken.TagPrefix, StringComparison.Ordinal)}";
    }
}
