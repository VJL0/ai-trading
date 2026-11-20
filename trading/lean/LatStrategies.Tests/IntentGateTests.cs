using System;
using System.IO;
using Lat.Lean.Gate;
using NUnit.Framework;

namespace Lat.Lean.Tests
{
    [TestFixture]
    public class IntentGateTests
    {
        private static GateDecision Eval(OrderFacts f, IntentGate gate = null) => (gate ?? Vectors.Gate()).Evaluate(f, consume: true);

        [Test]
        public void ValidSignedLimitBuyIsAllowed()
        {
            var d = Eval(Vectors.Facts(Vectors.Case("valid_buy")));
            Assert.That(d.Allowed, Is.True, d.Reason);
            Assert.That(d.Intent.ClientOrderId, Is.EqualTo("lat-sim-vec-001"));
        }

        [Test]
        public void UnsignedOrderIsRefused()
        {
            var f = Vectors.Facts(Vectors.Case("valid_buy"));
            f.Tag = "strategy decided to buy";
            var d = Eval(f);
            Assert.That(d.Allowed, Is.False);
            Assert.That(d.Reason, Does.Contain("unsigned"));
            f.Tag = null;
            Assert.That(Eval(f).Allowed, Is.False);
        }

        [TestCase("tampered_quantity")]
        [TestCase("signed_by_other_key")]
        [TestCase("bitflip_signature")]
        public void ForgedOrTamperedIntentsAreRefused(string name)
        {
            var c = Vectors.Case(name);
            var d = Eval(Vectors.Facts(c));
            Assert.That(d.Allowed, Is.False);
            Assert.That(d.Reason, Does.Contain("signature verification failed"));
        }

        [Test]
        public void TamperedQuantityCannotBeExecutedEvenIfOrderMatchesTamperedString()
        {
            var c = Vectors.Case("tampered_quantity");
            var f = Vectors.Facts(c);
            f.Quantity = 1000m; // the order matches the tampered string exactly: signature still fails
            Assert.That(Eval(f).Allowed, Is.False);
        }

        [Test]
        public void ExpiredIntentIsRefused()
        {
            var d = Eval(Vectors.Facts(Vectors.Case("expired")));
            Assert.That(d.Allowed, Is.False);
            Assert.That(d.Reason, Does.Contain("expired"));
        }

        [Test]
        public void IntentFromTheFutureIsRefused()
        {
            var gate = Vectors.Gate(now: new DateTime(2026, 11, 5, 14, 29, 0, DateTimeKind.Utc));
            Assert.That(Eval(Vectors.Facts(Vectors.Case("valid_buy")), gate).Allowed, Is.False);
        }

        [Test]
        public void MarketOrdersAreRefused()
        {
            var f = Vectors.Facts(Vectors.Case("valid_buy"));
            f.OrderType = "Market";
            f.LimitPrice = null;
            Assert.That(Eval(f).Allowed, Is.False, "LEAN market order with a valid limit intent");
            Assert.That(Eval(Vectors.Facts(Vectors.Case("market_order_intent"))).Reason, Does.Contain("LIMIT only"));
        }

        [TestCase(9, 450.10, "SPY", "quantity mismatch")]
        [TestCase(11, 450.10, "SPY", "quantity mismatch")]
        [TestCase(10, 450.11, "SPY", "limit price mismatch")]
        [TestCase(10, 450.10, "QQQ", "symbol mismatch")]
        public void AnyDeviationFromTheSignedOrderIsRefused(decimal qty, decimal limit, string symbol, string reason)
        {
            var f = Vectors.Facts(Vectors.Case("valid_buy"));
            f.Quantity = qty;
            f.LimitPrice = limit;
            f.Symbol = symbol;
            var d = Eval(f);
            Assert.That(d.Allowed, Is.False);
            Assert.That(d.Reason, Does.Contain(reason));
        }

        [Test]
        public void SideMismatchIsRefused()
        {
            var f = Vectors.Facts(Vectors.Case("valid_buy"), held: 100m);
            f.Quantity = -10m;
            Assert.That(Eval(f).Reason, Does.Contain("side mismatch"));
        }

        [Test]
        public void SellIsAllowedOnlyAgainstHeldShares()
        {
            Assert.That(Eval(Vectors.Facts(Vectors.Case("valid_sell"), held: 5m)).Allowed, Is.True);
            var shortSale = Eval(Vectors.Facts(Vectors.Case("valid_sell"), held: 0m));
            Assert.That(shortSale.Allowed, Is.False);
            Assert.That(shortSale.Reason, Does.Contain("short sale refused"));
            Assert.That(Eval(Vectors.Facts(Vectors.Case("sell_more_than_held"), held: 5m)).Allowed, Is.False);
        }

        [Test]
        public void PendingSellsCountAgainstHoldings()
        {
            var f = Vectors.Facts(Vectors.Case("valid_sell"), held: 5m);
            f.PendingSellQuantity = 1m;
            Assert.That(Eval(f).Allowed, Is.False);
        }

        [Test]
        public void ExistingShortPositionBlocksEverything()
        {
            Assert.That(Eval(Vectors.Facts(Vectors.Case("valid_buy"), held: -1m)).Allowed, Is.False);
        }

        [Test]
        public void ExtendedHoursAndClosedExchangeAreRefused()
        {
            var f = Vectors.Facts(Vectors.Case("valid_buy"));
            f.OutsideRegularTradingHours = true;
            Assert.That(Eval(f).Allowed, Is.False);
            var g = Vectors.Facts(Vectors.Case("valid_buy"));
            g.ExchangeOpen = false;
            Assert.That(Eval(g).Reason, Does.Contain("regular session"));
            var unset = Vectors.Facts(Vectors.Case("valid_buy"));
            unset = new OrderFacts { LeanOrderId = 1, Tag = unset.Tag, Symbol = "SPY", SecurityType = "Equity", Market = "usa",
                OrderType = "Limit", Quantity = 10m, LimitPrice = 450.10m, TimeInForce = "DAY" };
            Assert.That(Eval(unset).Allowed, Is.False, "ExchangeOpen defaults to closed (fail closed)");
        }

        [TestCase("GTC")]
        [TestCase("GTD")]
        public void NonDayTimeInForceIsRefused(string tif)
        {
            var f = Vectors.Facts(Vectors.Case("valid_buy"));
            f.TimeInForce = tif;
            Assert.That(Eval(f).Allowed, Is.False);
        }

        [Test]
        public void UnsupportedFeaturesAreRejectedNotDegraded()
        {
            var a = Vectors.Facts(Vectors.Case("valid_buy"));
            a.HasContingencyOrGroup = true;
            Assert.That(Eval(a).Allowed, Is.False);
            var b = Vectors.Facts(Vectors.Case("valid_buy"));
            b.UnsupportedPropertiesType = "InteractiveBrokersOrderProperties";
            Assert.That(Eval(b).Allowed, Is.False);
            var c = Vectors.Facts(Vectors.Case("valid_buy"));
            c.SecurityType = "Option";
            Assert.That(Eval(c).Allowed, Is.False);
            var d = Vectors.Facts(Vectors.Case("valid_buy"));
            d.Market = "india";
            Assert.That(Eval(d).Allowed, Is.False);
        }

        [Test]
        public void IntentForAnotherGatewayIsRefused()
        {
            Assert.That(Eval(Vectors.Facts(Vectors.Case("wrong_gateway"))).Reason, Does.Contain("gateway"));
            var alpacaGate = Vectors.Gate(LatGateway.LEAN_ALPACA_PAPER);
            Assert.That(Eval(Vectors.Facts(Vectors.Case("valid_buy")), alpacaGate).Allowed, Is.False);
            Assert.That(Eval(Vectors.Facts(Vectors.Case("wrong_gateway")), alpacaGate).Allowed, Is.True);
        }

        [Test]
        public void WrongConfiguredKeyRefusesEverything()
        {
            var gate = new IntentGate(IntentSignatureVerifier.FromPem(Vectors.OtherPublicKeyPem), LatGateway.LEAN_PAPER_SIM, () => Vectors.NowUtc);
            Assert.That(gate.Evaluate(Vectors.Facts(Vectors.Case("valid_buy")), true).Allowed, Is.False);
        }

        [Test]
        public void OneIntentBindsToOneLeanOrder()
        {
            var gate = Vectors.Gate();
            var c = Vectors.Case("valid_buy");
            Assert.That(gate.Evaluate(Vectors.Facts(c, leanOrderId: 7), true).Allowed, Is.True);
            Assert.That(gate.Evaluate(Vectors.Facts(c, leanOrderId: 7), true).Allowed, Is.True, "same order re-checked (model + brokerage)");
            var replay = gate.Evaluate(Vectors.Facts(c, leanOrderId: 8), true);
            Assert.That(replay.Allowed, Is.False);
            Assert.That(replay.Reason, Does.Contain("duplicate intent"));
            // the original order may be live: a duplicate refusal must not be attributable to its ids
            Assert.That(replay.ClaimedClientOrderId, Is.Null);
            Assert.That(replay.ClaimedIntentId, Is.Null);
            Assert.That(replay.Reason, Does.Contain(c.ClientOrderId));
        }

        [Test]
        public void JournalRefusesReplayAcrossEngineRestart()
        {
            var path = Path.Combine(Path.GetTempPath(), "lat-journal-" + Guid.NewGuid().ToString("N") + ".tsv");
            try
            {
                var c = Vectors.Case("valid_buy");
                var first = Vectors.Gate(journal: new UsedIntentJournal(path, "run1"));
                Assert.That(first.Evaluate(Vectors.Facts(c, leanOrderId: 1), true).Allowed, Is.True);
                // "restart": new process state, same journal file; LEAN order ids restart at 1
                var second = Vectors.Gate(journal: new UsedIntentJournal(path, "run2"));
                var d = second.Evaluate(Vectors.Facts(c, leanOrderId: 1), true);
                Assert.That(d.Allowed, Is.False);
                Assert.That(d.Reason, Does.Contain("earlier engine run"));
                Assert.That(d.ClaimedClientOrderId, Is.Null, "replay refusals never target the original order");
                Assert.That(second.Evaluate(Vectors.Facts(Vectors.Case("valid_sell"), leanOrderId: 2, held: 5m), true).Allowed, Is.True);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void FactoryFailsClosedWithoutKeyOrJournal()
        {
            var dir = Path.Combine(Path.GetTempPath(), "lat-factory-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            var pem = Path.Combine(dir, "intent_signing.pub.pem");
            File.WriteAllText(pem, Vectors.PublicKeyPem);
            var saved = Environment.GetEnvironmentVariable(IntentSignatureVerifier.PubKeyPathEnv);
            Environment.SetEnvironmentVariable(IntentSignatureVerifier.PubKeyPathEnv, null);
            try
            {
                Func<string, string> Cfg(string key, string journal) => k =>
                    k == IntentSignatureVerifier.PubKeyPathConfig ? key : k == LatGateFactory.JournalPathConfig ? journal : null;
                var journalPath = Path.Combine(dir, "used.tsv");
                Assert.Throws<InvalidOperationException>(() => LatGateFactory.Create(LatGateway.LEAN_PAPER_SIM, Cfg(null, journalPath)), "no key");
                Assert.Throws<FileNotFoundException>(() => LatGateFactory.Create(LatGateway.LEAN_PAPER_SIM, Cfg(pem + ".missing", journalPath)));
                var noJournal = Assert.Throws<InvalidOperationException>(() => LatGateFactory.Create(LatGateway.LEAN_PAPER_SIM, Cfg(pem, null)));
                Assert.That(noJournal.Message, Does.Contain("replay protection"));
                var gate = LatGateFactory.Create(LatGateway.LEAN_PAPER_SIM, Cfg(pem, journalPath), () => Vectors.NowUtc);
                Assert.That(gate.KeyId, Is.EqualTo(Vectors.KeyId));
                Assert.That(gate.Evaluate(Vectors.Facts(Vectors.Case("valid_buy")), true).Allowed, Is.True);
                Assert.That(File.ReadAllText(journalPath), Does.StartWith("int_vec_001\tlat-sim-vec-001\t1\t"));
            }
            finally
            {
                Environment.SetEnvironmentVariable(IntentSignatureVerifier.PubKeyPathEnv, saved);
                Directory.Delete(dir, true);
            }
        }

        [Test]
        public void CorruptJournalFailsClosedAtStartup()
        {
            var path = Path.Combine(Path.GetTempPath(), "lat-journal-" + Guid.NewGuid().ToString("N") + ".tsv");
            File.WriteAllText(path, "garbage-without-tabs\n");
            try
            {
                Assert.Throws<InvalidDataException>(() => new UsedIntentJournal(path, "r"));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void ParserIsStrict()
        {
            var good = Vectors.Case("valid_buy").SigningString;
            Assert.That(SignedIntent.TryParse(good, out _, out _), Is.True);
            Assert.That(SignedIntent.TryParse(good.Replace("lat-intent-v1", "lat-intent-v2"), out _, out _), Is.False);
            Assert.That(SignedIntent.TryParse(good + "|extra=1", out _, out _), Is.False);
            Assert.That(SignedIntent.TryParse(good.Replace("|quantity=10|", "|quantity=1e1|"), out _, out _), Is.False);
            Assert.That(SignedIntent.TryParse(good.Replace("|quantity=10|", "|quantity=-10|"), out _, out _), Is.False);
            Assert.That(SignedIntent.TryParse(good.Replace("intent_id=", "intentid="), out _, out _), Is.False);
            Assert.That(SignedIntent.TryParse(good.Replace("Z|expires_at", "+00:00|expires_at"), out _, out _), Is.False, "UTC 'Z' required");
        }
    }
}
