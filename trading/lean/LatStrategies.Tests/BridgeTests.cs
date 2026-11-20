using System;
using System.Collections.Generic;
using System.Text.Json;
using Lat.Lean.Bridge;
using Lat.Lean.Gate;
using NUnit.Framework;

namespace Lat.Lean.Tests
{
    [TestFixture]
    public class BridgeTests
    {
        private static Func<string, string> Cfg(Dictionary<string, string> d) => k => d.TryGetValue(k, out var v) ? v : null;
        private static Func<string, string> Env(string token) => k => k == BridgeSettings.EngineTokenEnv ? token : null;

        [TestCase("http://127.0.0.1:8765")]
        [TestCase("http://localhost:9000/ignored/path")]
        [TestCase("http://[::1]:8765")]
        public void LoopbackUrlsAreAccepted(string url) => Assert.That(BridgeSettings.ParseLoopbackUrl(url).IsLoopback, Is.True);

        [TestCase("http://192.168.1.10:8765")]
        [TestCase("https://api.example.com")]
        [TestCase("http://user:pw@127.0.0.1:8765")]
        [TestCase("ftp://127.0.0.1")]
        [TestCase("not a url")]
        public void NonLoopbackOrOddUrlsAreRefused(string url) =>
            Assert.Throws<InvalidOperationException>(() => BridgeSettings.ParseLoopbackUrl(url));

        [Test]
        public void SettingsFailClosed()
        {
            var ok = new Dictionary<string, string> { ["lat-gateway"] = "LEAN_PAPER_SIM", ["lat-engine-id"] = "lean-test" };
            Assert.That(BridgeSettings.FromConfig(Cfg(ok), Env("0123456789abcdef0123")).Gateway, Is.EqualTo(LatGateway.LEAN_PAPER_SIM));
            Assert.Throws<InvalidOperationException>(() => BridgeSettings.FromConfig(Cfg(ok), Env(null)), "token required");
            Assert.Throws<InvalidOperationException>(() => BridgeSettings.FromConfig(Cfg(ok), Env("short")), "token too short");
            var live = new Dictionary<string, string>(ok) { ["lat-gateway"] = "LEAN_ALPACA_LIVE" };
            Assert.Throws<InvalidOperationException>(() => BridgeSettings.FromConfig(Cfg(live), Env("0123456789abcdef0123")));
            var liveMode = new Dictionary<string, string>(ok) { ["lat-mode"] = "LIVE_APPROVAL" };
            Assert.Throws<InvalidOperationException>(() => BridgeSettings.FromConfig(Cfg(liveMode), Env("0123456789abcdef0123")));
            var remote = new Dictionary<string, string>(ok) { ["lat-bridge-url"] = "http://10.0.0.5:8765" };
            Assert.Throws<InvalidOperationException>(() => BridgeSettings.FromConfig(Cfg(remote), Env("0123456789abcdef0123")));
        }

        [Test]
        public void EventPayloadUsesDecimalStringsAndUtcZTimestamps()
        {
            var json = EnginePayloads.ToJson("lean-test", new EnginePayloads.OrderEventPayload
            {
                ClientOrderId = "lat-sim-1", IntentId = "int_1", LeanOrderId = 3, LeanEventId = 2, BrokerOrderId = "3",
                Status = "Filled", FillQuantity = 10m, FillPrice = 450.10m, CumFilledQuantity = 10m,
                EventUtc = new DateTime(2026, 11, 5, 14, 31, 2, 123, DateTimeKind.Utc), Message = "ok",
            });
            using var doc = JsonDocument.Parse(json);
            var r = doc.RootElement;
            Assert.That(r.GetProperty("fill_quantity").ValueKind, Is.EqualTo(JsonValueKind.String));
            Assert.That(r.GetProperty("fill_price").GetString(), Is.EqualTo("450.10"));
            Assert.That(r.GetProperty("cum_filled_quantity").GetString(), Is.EqualTo("10"));
            Assert.That(r.GetProperty("lean_order_id").GetInt32(), Is.EqualTo(3));
            Assert.That(r.GetProperty("lean_event_id").GetInt32(), Is.EqualTo(2));
            Assert.That(r.GetProperty("event_ts").GetString(), Is.EqualTo("2026-11-05T14:31:02.123000Z"));
            Assert.That(json, Does.Not.Contain("450.1,"), "no JSON numbers for money");
        }

        [Test]
        public void IntentsResponseParsing()
        {
            var list = EnginePayloads.ParseIntents("{\"intents\":[{\"intent_id\":\"i\",\"client_order_id\":\"c\",\"signing_string\":\"s\",\"signature\":\"x\",\"key_id\":\"k\",\"signature_alg\":\"ECDSA_P256_SHA256\"}]}");
            Assert.That(list, Has.Count.EqualTo(1));
            Assert.That(list[0].ClientOrderId, Is.EqualTo("c"));
            Assert.That(EnginePayloads.ParseIntents("{\"intents\":[]}"), Is.Empty);
            Assert.Throws<FormatException>(() => EnginePayloads.ParseIntents("[]"));
        }
    }
}
