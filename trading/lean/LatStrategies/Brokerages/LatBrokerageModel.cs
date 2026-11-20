/*
 * Decorator over a LEAN brokerage model (AlpacaBrokerageModel for Alpaca, DefaultBrokerageModel for the
 * PaperBrokerage simulation) that applies the LAT account and cost policy identically in backtest and live:
 *   * cash account, leverage 1, no shorting (long-only, no margin borrowing),
 *   * LatFeeModel / LatSpreadSlippageModel / LatFillModel.
 * Order acceptance is delegated to the inner model (GatedBrokerageModel adds the signed-intent gate).
 */

using System;
using System.Collections.Generic;
using Lat.Lean.Costs;
using QuantConnect;
using QuantConnect.Benchmarks;
using QuantConnect.Brokerages;
using QuantConnect.Data.Market;
using QuantConnect.Interfaces;
using QuantConnect.Orders;
using QuantConnect.Orders.Fees;
using QuantConnect.Orders.Fills;
using QuantConnect.Orders.Slippage;
using QuantConnect.Securities;
using QuantConnect.Securities.Equity;

namespace Lat.Lean.Brokerages
{
    /// <summary>Nothing is shortable: LEAN's default NullShortableProvider would allow unlimited shorting.</summary>
    public sealed class NoShortingProvider : IShortableProvider
    {
        public static NoShortingProvider Instance { get; } = new NoShortingProvider();
        public decimal FeeRate(Symbol symbol, DateTime localTime) => 0m;
        public decimal RebateRate(Symbol symbol, DateTime localTime) => 0m;
        public long? ShortableQuantity(Symbol symbol, DateTime localTime) => 0;
    }

    /// <remarks>Derives from DefaultBrokerageModel only so LEAN's BrokerageModel.GetBrokerageName (used when writing
    /// results) recognises it; every member is overridden and delegates to <see cref="Inner"/> or applies LAT policy.</remarks>
    public class LatBrokerageModel : DefaultBrokerageModel
    {
        protected IBrokerageModel Inner { get; }
        public LatCostConfig Costs { get; }

        public LatBrokerageModel(IBrokerageModel inner, LatCostConfig costs = null)
            : base(AccountType.Cash)
        {
            Inner = inner ?? throw new ArgumentNullException(nameof(inner));
            Costs = costs ?? LatCostConfig.Default;
        }

        public string InnerModelName => Inner.GetType().Name;

        public override AccountType AccountType => AccountType.Cash;
        public override decimal RequiredFreeBuyingPowerPercent => Inner.RequiredFreeBuyingPowerPercent;
        public override IReadOnlyDictionary<SecurityType, string> DefaultMarkets => Inner.DefaultMarkets;

        public override bool CanSubmitOrder(Security security, Order order, out BrokerageMessageEvent message)
        {
            if (security.Type != SecurityType.Equity)
            {
                message = new BrokerageMessageEvent(BrokerageMessageType.Warning, "LatUnsupported", $"LAT: security type {security.Type} not supported (US equities only)");
                return false;
            }
            return Inner.CanSubmitOrder(security, order, out message);
        }

        public override bool CanUpdateOrder(Security security, Order order, UpdateOrderRequest request, out BrokerageMessageEvent message) =>
            Inner.CanUpdateOrder(security, order, request, out message);

        public override bool CanExecuteOrder(Security security, Order order) => Inner.CanExecuteOrder(security, order);
        public override void ApplySplit(List<OrderTicket> tickets, Split split) => Inner.ApplySplit(tickets, split);
        public override decimal GetLeverage(Security security) => 1m;
        public override IBenchmark GetBenchmark(SecurityManager securities) => Inner.GetBenchmark(securities);
        public override IFillModel GetFillModel(Security security) => new LatFillModel();
        public override IFeeModel GetFeeModel(Security security) => new LatFeeModel(Costs);
        public override ISlippageModel GetSlippageModel(Security security) => new LatSpreadSlippageModel(Costs);

        public override ISettlementModel GetSettlementModel(Security security) =>
            security.Type == SecurityType.Equity
                ? new DelayedSettlementModel(Equity.DefaultSettlementDays, Equity.DefaultSettlementTime)
                : new ImmediateSettlementModel();

        public override IMarginInterestRateModel GetMarginInterestRateModel(Security security) => MarginInterestRateModel.Null;

        // leverage 1 margin model == cash-only buying power for equities (no margin debit possible)
        public override IBuyingPowerModel GetBuyingPowerModel(Security security) => new SecurityMarginModel(1m, RequiredFreeBuyingPowerPercent);

        public override IShortableProvider GetShortableProvider(Security security) => NoShortingProvider.Instance;
    }
}
