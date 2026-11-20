using System;
using System.Collections.Generic;
using Lat.Lean.Brokerages;
using Lat.Lean.Gate;
using NUnit.Framework;
using QuantConnect;
using QuantConnect.Brokerages;
using QuantConnect.Data.Market;
using QuantConnect.Interfaces;
using QuantConnect.Orders;
using QuantConnect.Securities;
using QuantConnect.Securities.Equity;

namespace Lat.Lean.Tests
{
    /// <summary>The gate wired into LEAN's real IBrokerageModel hook, with real LEAN Security/Order objects.</summary>
    [TestFixture]
    public class GatedBrokerageModelTests
    {
        private sealed class RecordingSink : IGateRefusalSink
        {
            public readonly List<(string Layer, int OrderId, GateDecision Decision)> Refusals = new();
            public void Refused(string layer, int leanOrderId, GateDecision decision, string orderSummary) => Refusals.Add((layer, leanOrderId, decision));
        }

        private static readonly Symbol Spy = new Symbol(SecurityIdentifier.GenerateEquity("SPY", Market.USA, mapSymbol: false), "SPY");
        private Equity _spy;
        private RecordingSink _sink;
        private GatedBrokerageModel _model;

        [SetUp]
        public void SetUp()
        {
            _spy = new Equity(Spy, SecurityExchangeHours.AlwaysOpen(TimeZones.NewYork), new Cash(Currencies.USD, 0, 1),
                SymbolProperties.GetDefault(Currencies.USD), ErrorCurrencyConverter.Instance, RegisteredSecurityDataTypesProvider.Null, new SecurityCache());
            _spy.SetLocalTimeKeeper(new TimeKeeper(new DateTime(2026, 11, 5, 14, 30, 30, DateTimeKind.Utc), TimeZones.NewYork).GetLocalTimeKeeper(TimeZones.NewYork));
            _spy.SetMarketPrice(new TradeBar(new DateTime(2026, 11, 5, 9, 31, 0), Spy, 450m, 450m, 450m, 450m, 1000));
            _sink = new RecordingSink();
            _model = new GatedBrokerageModel(new DefaultBrokerageModel(AccountType.Cash), Vectors.Gate(), () => new EmptyOrderProvider(), _sink);
        }

        private static LimitOrder Limit(VectorCase c, decimal? qty = null, decimal? px = null, string tag = null, int id = 1, IOrderProperties props = null)
        {
            var q = qty ?? (c.Side == "BUY" ? c.Quantity : -c.Quantity);
            var order = new LimitOrder(Spy, q, px ?? c.LimitPrice, new DateTime(2026, 11, 5, 14, 30, 30, DateTimeKind.Utc), tag ?? Vectors.Token(c),
                props ?? new OrderProperties { TimeInForce = TimeInForce.Day });
            SetId(order, id);
            return order;
        }

        // Order.Id has an internal setter (LEAN assigns ids in the transaction handler)
        private static void SetId(Order o, int id) => typeof(Order).GetProperty(nameof(Order.Id)).GetSetMethod(true).Invoke(o, new object[] { id });

        private bool Submit(Order o, out BrokerageMessageEvent msg) => _model.CanSubmitOrder(_spy, o, out msg);

        [Test]
        public void SignedMatchingLimitOrderPasses()
        {
            Assert.That(Submit(Limit(Vectors.Case("valid_buy")), out var msg), Is.True, msg?.Message);
            Assert.That(_sink.Refusals, Is.Empty);
        }

        [Test]
        public void UnsignedStrategyOrderIsRefusedAndReported()
        {
            Assert.That(Submit(Limit(Vectors.Case("valid_buy"), tag: "my strategy wants SPY"), out var msg), Is.False);
            Assert.That(msg.Message, Does.Contain("unsigned"));
            Assert.That(_sink.Refusals, Has.Count.EqualTo(1));
            Assert.That(_sink.Refusals[0].Layer, Is.EqualTo(GatedBrokerageModel.Layer));
        }

        [Test]
        public void TamperedIntentIsRefused()
        {
            var c = Vectors.Case("tampered_quantity");
            Assert.That(Submit(Limit(c, qty: 1000m), out _), Is.False);
            Assert.That(_sink.Refusals[0].Decision.ClaimedClientOrderId, Is.EqualTo("lat-sim-vec-001"), "claimed ids reported for audit");
        }

        [Test]
        public void IntentSignedByAnotherKeyIsRefused() => Assert.That(Submit(Limit(Vectors.Case("signed_by_other_key")), out _), Is.False);

        [Test]
        public void ExpiredIntentIsRefused() => Assert.That(Submit(Limit(Vectors.Case("expired")), out _), Is.False);

        [Test]
        public void MarketOrderCarryingAValidTokenIsRefused()
        {
            var c = Vectors.Case("valid_buy");
            var mkt = new MarketOrder(Spy, 10m, new DateTime(2026, 11, 5, 14, 30, 30, DateTimeKind.Utc), Vectors.Token(c));
            SetId(mkt, 1);
            Assert.That(Submit(mkt, out var msg), Is.False);
            Assert.That(msg.Message, Does.Contain("LIMIT only"));
        }

        [Test]
        public void QuantityMismatchIsRefused() => Assert.That(Submit(Limit(Vectors.Case("valid_buy"), qty: 11m), out _), Is.False);

        [Test]
        public void ShortSaleIsRefusedAndHeldSellAllowed()
        {
            var sell = Vectors.Case("valid_sell");
            Assert.That(Submit(Limit(sell), out var m1), Is.False, "no holdings -> short sale");
            Assert.That(m1.Message, Does.Contain("short sale"));
            _spy.Holdings.SetHoldings(440m, 5m);
            Assert.That(Submit(Limit(sell, id: 2), out var m2), Is.True, m2?.Message + " | " + string.Join(";", _sink.Refusals.ConvertAll(r => r.Decision.Reason)));
        }

        [Test]
        public void SellIsRefusedWhenOpenOrdersCannotBeInspected()
        {
            _spy.Holdings.SetHoldings(440m, 5m);
            var blind = new GatedBrokerageModel(new DefaultBrokerageModel(AccountType.Cash), Vectors.Gate(), () => null, _sink);
            Assert.That(blind.CanSubmitOrder(_spy, Limit(Vectors.Case("valid_sell")), out var msg), Is.False);
            Assert.That(msg.Message, Does.Contain("short sale"));
        }

        [Test]
        public void ExtendedHoursPropertiesAreRefused()
        {
            var props = new QuantConnect.Orders.AlpacaOrderProperties { TimeInForce = TimeInForce.Day, OutsideRegularTradingHours = true };
            Assert.That(Submit(Limit(Vectors.Case("valid_buy"), props: props), out _), Is.False);
            var regular = new QuantConnect.Orders.AlpacaOrderProperties { TimeInForce = TimeInForce.Day, OutsideRegularTradingHours = false };
            Assert.That(Submit(Limit(Vectors.Case("valid_buy"), props: regular), out var msg), Is.True, msg?.Message);
        }

        [Test]
        public void GoodTilCanceledIsRefused()
        {
            var props = new OrderProperties { TimeInForce = TimeInForce.GoodTilCanceled };
            Assert.That(Submit(Limit(Vectors.Case("valid_buy"), props: props), out _), Is.False);
        }

        [Test]
        public void UpdatesAreAlwaysRefused()
        {
            var order = Limit(Vectors.Case("valid_buy"));
            var req = new UpdateOrderRequest(DateTime.UtcNow, 1, new UpdateOrderFields { Quantity = 10m });
            Assert.That(_model.CanUpdateOrder(_spy, order, req, out var msg), Is.False);
            Assert.That(msg.Message, Does.Contain("immutable"));
        }

        [Test]
        public void ModelPolicyIsCashNoLeverageNoShortingWithLatCosts()
        {
            Assert.That(_model.AccountType, Is.EqualTo(AccountType.Cash));
            Assert.That(_model.GetLeverage(_spy), Is.EqualTo(1m));
            Assert.That(_model.GetShortableProvider(_spy).ShortableQuantity(Spy, DateTime.UtcNow), Is.EqualTo(0));
            Assert.That(_model.GetFeeModel(_spy), Is.InstanceOf<Costs.LatFeeModel>());
            Assert.That(_model.GetSlippageModel(_spy), Is.InstanceOf<Costs.LatSpreadSlippageModel>());
            Assert.That(_model.GetFillModel(_spy), Is.InstanceOf<Costs.LatFillModel>());
        }

        [Test]
        public void InnerAlpacaModelStillApplies()
        {
            var model = new GatedBrokerageModel(new AlpacaBrokerageModel(), Vectors.Gate(), () => new EmptyOrderProvider(), _sink);
            Assert.That(model.CanSubmitOrder(_spy, Limit(Vectors.Case("valid_buy")), out var msg), Is.True, msg?.Message);
            Assert.That(model.AccountType, Is.EqualTo(AccountType.Cash), "cash account even though AlpacaBrokerageModel defaults to margin");
        }
    }
}
