using System;
using Lat.Lean.Costs;
using NUnit.Framework;
using QuantConnect;
using QuantConnect.Data.Market;
using QuantConnect.Orders;
using QuantConnect.Orders.Fees;
using QuantConnect.Securities;
using QuantConnect.Securities.Equity;

namespace Lat.Lean.Tests
{
    [TestFixture]
    public class CostModelTests
    {
        private readonly LatFeeModel _fees = new LatFeeModel();

        [Test]
        public void BuysPayOnlyTheCommissionWhichIsZeroAtAlpaca()
        {
            var f = _fees.Compute(isSell: false, shares: 1000m, price: 450m);
            Assert.That(f.Total, Is.EqualTo(0m));
            Assert.That(f.SecFee, Is.EqualTo(0m));
            Assert.That(f.FinraTaf, Is.EqualTo(0m));
        }

        [Test]
        public void SellPaysSecSection31AndFinraTafRoundedUpToTheCent()
        {
            // 100 x 450 = 45,000 proceeds; SEC 45,000 * 27.80 / 1e6 = 1.251 -> 1.26 ; TAF 100 * 0.000166 = 0.0166 -> 0.02
            var f = _fees.Compute(isSell: true, shares: 100m, price: 450m);
            Assert.That(f.SecFee, Is.EqualTo(1.26m));
            Assert.That(f.FinraTaf, Is.EqualTo(0.02m));
            Assert.That(f.Total, Is.EqualTo(1.28m));
        }

        [Test]
        public void TafIsCappedPerTrade()
        {
            // 100,000 shares * 0.000166 = 16.60 -> capped at 8.30 ; SEC 1,000,000 * 27.80/1e6 = 27.80
            var f = _fees.Compute(isSell: true, shares: 100_000m, price: 10m);
            Assert.That(f.FinraTaf, Is.EqualTo(8.30m));
            Assert.That(f.SecFee, Is.EqualTo(27.80m));
        }

        [Test]
        public void ExactArithmeticWithoutRounding()
        {
            var m = new LatFeeModel(new LatCostConfig { RoundRegulatoryFeesUpToCent = false, CommissionPerOrder = 1m });
            var f = m.Compute(isSell: true, shares: 100m, price: 450m);
            Assert.That(f.SecFee, Is.EqualTo(1.251m));
            Assert.That(f.FinraTaf, Is.EqualTo(0.0166m));
            Assert.That(f.Total, Is.EqualTo(2.2676m));
        }

        [Test]
        public void ConfigOverridesAreParsedAsDecimalsAndValidated()
        {
            var cfg = LatCostConfig.FromLookup(k => k == "lat-fee-sec-rate-per-million" ? "0" : k == "lat-slippage-bps" ? "2.5" : null);
            Assert.That(cfg.SecFeeRatePerMillion, Is.EqualTo(0m));
            Assert.That(cfg.SlippageBps, Is.EqualTo(2.5m));
            Assert.Throws<ArgumentException>(() => LatCostConfig.FromLookup(k => k == "lat-fee-taf-per-share" ? "-1" : null));
            Assert.Throws<ArgumentException>(() => LatCostConfig.FromLookup(k => k == "lat-slippage-bps" ? "1e3" : null));
        }

        [Test]
        public void LeanFeeModelInterfaceUsesTheSameArithmetic()
        {
            var spy = new Symbol(SecurityIdentifier.GenerateEquity("SPY", Market.USA, mapSymbol: false), "SPY");
            var sec = new Equity(spy, SecurityExchangeHours.AlwaysOpen(TimeZones.NewYork), new Cash(Currencies.USD, 0, 1),
                SymbolProperties.GetDefault(Currencies.USD), ErrorCurrencyConverter.Instance, RegisteredSecurityDataTypesProvider.Null, new SecurityCache());
            sec.SetMarketPrice(new TradeBar(DateTime.UtcNow, spy, 450m, 450m, 450m, 450m, 1));
            var sell = _fees.GetOrderFee(new OrderFeeParameters(sec, new LimitOrder(spy, -100m, 449m, DateTime.UtcNow)));
            var buy = _fees.GetOrderFee(new OrderFeeParameters(sec, new LimitOrder(spy, 100m, 451m, DateTime.UtcNow)));
            Assert.That(sell.Value.Amount, Is.EqualTo(1.28m));
            Assert.That(sell.Value.Currency, Is.EqualTo(Currencies.USD));
            Assert.That(buy.Value.Amount, Is.EqualTo(0m));
        }

        [Test]
        public void SlippageIsHalfSpreadPlusImpact()
        {
            var s = new LatSpreadSlippageModel();
            Assert.That(s.Compute(450m, 449.95m, 450.05m, marketOrderCrossesQuote: false), Is.EqualTo(0.05m + 0.045m));
            Assert.That(s.Compute(450m, 0m, 0m, false), Is.EqualTo(0.045m + 0.045m), "assumed 2 bps spread when no quote");
            Assert.That(s.Compute(450m, 449.95m, 450.05m, marketOrderCrossesQuote: true), Is.EqualTo(0.045m), "market fill already crossed the quote");
            Assert.That(s.Compute(0m, 1m, 2m, false), Is.EqualTo(0m));
        }

        [TestCase(true, 450.00, 0.095, 450.10, 450.10)]
        [TestCase(true, 449.90, 0.095, 450.10, 450.00)]
        [TestCase(true, 450.08, 0.095, 450.10, 450.10)] // capped at the limit
        [TestCase(false, 450.00, 0.095, 449.90, 449.90)]
        [TestCase(false, 450.20, 0.095, 449.90, 450.10)]
        public void LimitFillPaysSlippageButNeverWorseThanLimit(bool buy, decimal basePx, decimal slip, decimal limit, decimal expected)
        {
            Assert.That(LatFillModel.AdjustLimitFill(buy, basePx, slip, limit, 0.01m), Is.EqualTo(expected));
        }
    }
}
