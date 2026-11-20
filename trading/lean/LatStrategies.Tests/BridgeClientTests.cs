using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Lat.Lean.Bridge;
using Lat.Lean.Gate;
using NUnit.Framework;

namespace Lat.Lean.Tests
{
    /// <summary>In-memory HTTP handler: records every request and answers from a scripted status sequence.</summary>
    public sealed class RecordingHandler : HttpMessageHandler
    {
        public readonly ConcurrentQueue<(string Method, string Path, string Token, string Body)> Requests = new();
        private readonly ConcurrentQueue<HttpStatusCode> _script;

        public RecordingHandler(params HttpStatusCode[] script) => _script = new ConcurrentQueue<HttpStatusCode>(script);

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var body = request.Content == null ? "" : await request.Content.ReadAsStringAsync(cancellationToken);
            request.Headers.TryGetValues("X-LAT-Engine-Token", out var tokens);
            Requests.Enqueue((request.Method.Method, request.RequestUri!.AbsolutePath, tokens?.SingleOrDefault(), body));
            var code = _script.TryDequeue(out var c) ? c : HttpStatusCode.OK;
            return new HttpResponseMessage(code) { Content = new StringContent(request.Method == HttpMethod.Get ? "{\"intents\":[]}" : "{}") };
        }
    }

    [TestFixture]
    public class BridgeClientTests
    {
        private const string Token = "test-engine-token-0123456789";

        private static BridgeSettings Settings(string outbox) => BridgeSettings.FromConfig(
            k => k switch
            {
                "lat-gateway" => "LEAN_PAPER_SIM",
                "lat-engine-id" => "lean-unit",
                "lat-bridge-url" => "http://127.0.0.1:1",
                "lat-bridge-outbox-path" => outbox,
                _ => null,
            },
            k => k == BridgeSettings.EngineTokenEnv ? Token : null);

        private static bool WaitFor(Func<bool> condition, int ms = 10000)
        {
            var deadline = DateTime.UtcNow.AddMilliseconds(ms);
            while (DateTime.UtcNow < deadline)
            {
                if (condition()) return true;
                Thread.Sleep(20);
            }
            return condition();
        }

        [Test]
        public void TransientApiFailuresAreRetriedInOrderAndTheTokenIsNeverLogged()
        {
            var outbox = Path.Combine(Path.GetTempPath(), "lat-outbox-" + Guid.NewGuid().ToString("N") + ".jsonl");
            var logs = new ConcurrentQueue<string>();
            var handler = new RecordingHandler(HttpStatusCode.ServiceUnavailable, HttpStatusCode.ServiceUnavailable);
            try
            {
                using var client = new BridgeClient(Settings(outbox), logs.Enqueue, handler);
                client.Start(withPoller: false);
                client.ReportRejection("coid-1", "int-1", "first");
                client.ReportRejection("coid-2", "int-2", "second");
                Assert.That(WaitFor(() => client.Sent == 2), Is.True, "both records delivered after the outage");
                client.Stop(TimeSpan.FromSeconds(2));

                var posts = handler.Requests.Where(r => r.Method == "POST" && r.Path == BridgeClient.RejectionsPath).ToList();
                Assert.That(posts.Count, Is.EqualTo(4), "two 503 answers + two deliveries");
                // a sink-only client (the gated brokerage's refusal reporter) shares the engine id with the algorithm's
                // client: it must never post heartbeats, or it could report STOPPED while the algorithm still runs
                Assert.That(handler.Requests.Any(r => r.Path == BridgeClient.HeartbeatPath), Is.False,
                    "no heartbeat from a client started without the poller");
                Assert.That(posts.Select(p => JsonDocument.Parse(p.Body).RootElement.GetProperty("reason").GetString()),
                    Is.EqualTo(new[] { "first", "first", "first", "second" }), "strictly in order, nothing dropped or reordered");
                Assert.That(handler.Requests.All(p => p.Token == Token), Is.True, "every request authenticates");
                Assert.That(client.SendFailures, Is.EqualTo(2));
                Assert.That(File.ReadAllLines(outbox), Has.Length.EqualTo(2), "local outbox keeps one copy per record");
                Assert.That(File.ReadAllBytes(outbox)[0], Is.EqualTo((byte)'{'), "JSONL outbox has no UTF-8 BOM");
                foreach (var line in File.ReadAllLines(outbox)) JsonDocument.Parse(line).Dispose();
                Assert.That(logs.Any(l => l.Contains(Token)), Is.False, "engine token never logged");
                Assert.That(File.ReadAllText(outbox), Does.Not.Contain(Token));
            }
            finally
            {
                File.Delete(outbox);
            }
        }

        [Test]
        public void PollerClientHeartbeatsAndSendsFinalStoppedHeartbeatWithoutClaimingWhileIntakeIsOff()
        {
            var handler = new RecordingHandler();
            using var client = new BridgeClient(Settings(null), null, handler) { AlgorithmName = "UnitBridge" };
            client.SetStatus("RUNNING");
            client.Start(withPoller: true);
            Assert.That(WaitFor(() => handler.Requests.Any(r => r.Path == BridgeClient.HeartbeatPath)), Is.True, "periodic heartbeat");
            client.Stop(TimeSpan.FromSeconds(2));

            var beats = handler.Requests.Where(r => r.Path == BridgeClient.HeartbeatPath)
                .Select(r => JsonDocument.Parse(r.Body).RootElement).ToList();
            Assert.That(beats.First().GetProperty("status").GetString(), Is.EqualTo("RUNNING"));
            Assert.That(beats.Last().GetProperty("status").GetString(), Is.EqualTo("STOPPED"), "final STOPPED heartbeat on stop");
            Assert.That(beats.All(b => b.GetProperty("gateway").GetString() == "LEAN_PAPER_SIM" && b.GetProperty("mode").GetString() == "PAPER"), Is.True);
            Assert.That(handler.Requests.Any(r => r.Path == BridgeClient.IntentsPath), Is.False, "intake never enabled: nothing claimed");
            Assert.That(handler.Requests.All(r => r.Token == Token), Is.True);
        }

        [Test]
        public void PermanentClientErrorsAreNotRetriedForever()
        {
            var handler = new RecordingHandler(HttpStatusCode.UnprocessableEntity);
            using var client = new BridgeClient(Settings(null), null, handler);
            client.Start(withPoller: false);
            client.ReportRejection("coid-1", "int-1", "bad");
            client.ReportRejection("coid-2", "int-2", "good");
            Assert.That(WaitFor(() => client.Sent == 1), Is.True);
            client.Stop(TimeSpan.FromSeconds(2));
            Assert.That(handler.Requests.Count(r => r.Path == BridgeClient.RejectionsPath), Is.EqualTo(2), "422 is not retried");
        }

        [Test]
        public void DuplicateRefusalsAreReportedWithoutIdsAndFlaggedUnkeyed()
        {
            var handler = new RecordingHandler();
            using var client = new BridgeClient(Settings(null), null, handler);
            client.Start(withPoller: false);
            var gate = Vectors.Gate();
            var c = Vectors.Case("valid_buy");
            Assert.That(gate.Evaluate(Vectors.Facts(c, leanOrderId: 1), true).Allowed, Is.True);
            var dup = gate.Evaluate(Vectors.Facts(c, leanOrderId: 2), true);
            var tampered = gate.Evaluate(Vectors.Facts(Vectors.Case("tampered_quantity"), leanOrderId: 3), true);
            client.Refused("GatedBrokerageModel", 2, dup, "dup");
            client.Refused("GatedBrokerageModel", 3, tampered, "tampered");
            Assert.That(WaitFor(() => client.Sent == 2), Is.True);
            client.Stop(TimeSpan.FromSeconds(2));

            Assert.That(client.UnkeyedRefusal(2), Is.True, "duplicate: events must not be posted under the original ids");
            Assert.That(client.UnkeyedRefusal(3), Is.False, "first refusal of an intent is keyed (ledger may reject it)");
            Assert.That(client.RefusalReported(3), Is.True);
            var bodies = handler.Requests.Where(r => r.Path == BridgeClient.RejectionsPath).Select(r => JsonDocument.Parse(r.Body).RootElement).ToList();
            Assert.That(bodies[0].GetProperty("client_order_id").ValueKind, Is.EqualTo(JsonValueKind.Null));
            Assert.That(bodies[0].GetProperty("intent_id").ValueKind, Is.EqualTo(JsonValueKind.Null));
            Assert.That(bodies[0].GetProperty("reason").GetString(), Does.Contain("duplicate intent").And.Contain(c.ClientOrderId));
            Assert.That(bodies[1].GetProperty("client_order_id").GetString(), Is.EqualTo(Vectors.Case("tampered_quantity").ClientOrderId));
        }
    }
}
