/*
 * HTTP client of the local engine bridge (127.0.0.1 only). All network I/O happens on two background threads
 * so the LEAN algorithm thread never blocks on HTTP:
 *   * poller : GET  /api/v1/engine/intents/next (only while intake is enabled) + POST /api/v1/engine/heartbeat
 *   * sender : POST /api/v1/engine/events and /api/v1/engine/rejections, strictly in enqueue order, retried
 *              with capped backoff (never dropped while the process lives; each record is also appended to a
 *              local JSONL outbox file first, so the audit trail survives an API outage).
 * Money/quantity values are sent as decimal strings, timestamps as ISO-8601 UTC with a trailing 'Z'.
 * The engine token is sent only in the X-LAT-Engine-Token header and never logged.
 */

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using Lat.Lean.Brokerages;
using Lat.Lean.Gate;

namespace Lat.Lean.Bridge
{
    public sealed class BridgeClient : IGateRefusalSink, IDisposable
    {
        public const string IntentsPath = "/api/v1/engine/intents/next";
        public const string EventsPath = "/api/v1/engine/events";
        public const string HeartbeatPath = "/api/v1/engine/heartbeat";
        public const string RejectionsPath = "/api/v1/engine/rejections";

        // JSONL outbox must not start with a UTF-8 BOM (File.AppendAllText with Encoding.UTF8 writes one on creation)
        private static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(false);

        private readonly BridgeSettings _settings;
        private readonly HttpClient _http;
        private readonly BlockingCollection<(string Path, string Json)> _outbox = new BlockingCollection<(string, string)>();
        private readonly ConcurrentQueue<BridgeIntent> _intents = new ConcurrentQueue<BridgeIntent>();
        // LEAN order id -> whether the refusal report was keyed by intent/client order ids
        private readonly ConcurrentDictionary<int, bool> _refusedOrders = new ConcurrentDictionary<int, bool>();
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private readonly object _fileLock = new object();
        private readonly Action<string> _log;
        private Thread _poller;
        private Thread _sender;
        private volatile bool _intakeEnabled;
        private volatile string _status = "STARTING";
        private long _sent;
        private long _sendFailures;
        private long _pollFailures;
        private long _claimed;

        public string AlgorithmName { get; set; } = "";
        public BridgeSettings Settings => _settings;
        public long Sent => Interlocked.Read(ref _sent);
        public long SendFailures => Interlocked.Read(ref _sendFailures);
        public long PollFailures => Interlocked.Read(ref _pollFailures);
        public long Claimed => Interlocked.Read(ref _claimed);
        public int PendingOutbound => _outbox.Count;

        public BridgeClient(BridgeSettings settings, Action<string> log = null, HttpMessageHandler handler = null)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _log = log ?? (_ => { });
            _http = handler == null ? new HttpClient() : new HttpClient(handler);
            _http.BaseAddress = settings.BaseUrl;
            _http.Timeout = settings.HttpTimeout;
            _http.DefaultRequestHeaders.Add("X-LAT-Engine-Token", settings.EngineToken);
            _http.DefaultRequestHeaders.Add("X-LAT-Request", "1");
        }

        /// <summary>Starts the sender (always) and, when <paramref name="withPoller"/>, the intent poller + heartbeat.</summary>
        public void Start(bool withPoller)
        {
            _sender = new Thread(SendLoop) { IsBackground = true, Name = "LAT bridge sender" };
            _sender.Start();
            if (withPoller)
            {
                _poller = new Thread(PollLoop) { IsBackground = true, Name = "LAT bridge poller" };
                _poller.Start();
            }
        }

        /// <summary>Intents are only claimed from the API while intake is enabled (engine ready, not halted).</summary>
        public void SetIntake(bool enabled) => _intakeEnabled = enabled;
        public bool IntakeEnabled => _intakeEnabled;
        public void SetStatus(string status) => _status = status;

        public bool TryDequeueIntent(out BridgeIntent intent) => _intents.TryDequeue(out intent);

        // ------------------------------------------------------------------ outbound records

        public void ReportEvent(EnginePayloads.OrderEventPayload e) => Enqueue(EventsPath, EnginePayloads.ToJson(_settings.EngineId, e));

        public void ReportRejection(string clientOrderId, string intentId, string reason) =>
            Enqueue(RejectionsPath, EnginePayloads.RejectionJson(_settings.EngineId, clientOrderId, intentId, reason));

        public void Refused(string layer, int leanOrderId, GateDecision decision, string orderSummary)
        {
            var reason = $"[{layer}] {decision?.Reason ?? "refused"} ({orderSummary}; lean_order_id={leanOrderId.ToString(CultureInfo.InvariantCulture)})";
            _log($"LAT-GATE-REFUSED {reason}");
            var keyed = decision?.ClaimedClientOrderId != null || decision?.ClaimedIntentId != null;
            if (leanOrderId > 0) _refusedOrders[leanOrderId] = keyed;
            ReportRejection(decision?.ClaimedClientOrderId, decision?.ClaimedIntentId, reason);
        }

        /// <summary>True when this client already reported a gate refusal for the LEAN order.</summary>
        public bool RefusalReported(int leanOrderId) => _refusedOrders.ContainsKey(leanOrderId);

        /// <summary>True when the gate refused the LEAN order with an UNKEYED report (duplicate / replayed intent): the
        /// order's events must not be posted under the intent's ids, whose original order may be live.</summary>
        public bool UnkeyedRefusal(int leanOrderId) => _refusedOrders.TryGetValue(leanOrderId, out var keyed) && !keyed;

        private void Enqueue(string path, string json)
        {
            AppendOutboxFile(path, json);
            if (!_outbox.IsAddingCompleted)
            {
                try
                {
                    _outbox.Add((path, json));
                }
                catch (InvalidOperationException)
                {
                    // shutting down: the record is preserved in the local outbox file
                }
            }
        }

        private void AppendOutboxFile(string path, string json)
        {
            if (string.IsNullOrEmpty(_settings.OutboxPath)) return;
            try
            {
                lock (_fileLock)
                {
                    var dir = Path.GetDirectoryName(Path.GetFullPath(_settings.OutboxPath));
                    if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                    File.AppendAllText(_settings.OutboxPath,
                        "{\"path\":" + JsonSerializer.Serialize(path) + ",\"body\":" + json + "}\n", Utf8NoBom);
                }
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
            {
                _log("LAT bridge: cannot append local outbox: " + e.Message);
            }
        }

        private void SendLoop()
        {
            try
            {
                foreach (var item in _outbox.GetConsumingEnumerable(_cts.Token))
                {
                    var delay = 200;
                    while (true)
                    {
                        if (TryPost(item.Path, item.Json, out var status))
                        {
                            Interlocked.Increment(ref _sent);
                            break;
                        }
                        Interlocked.Increment(ref _sendFailures);
                        if (status >= 400 && status < 500 && status != 408 && status != 429)
                        {
                            // permanent client error (malformed/unauthorised): keep the local outbox copy, do not spin
                            _log($"LAT bridge: POST {item.Path} refused with HTTP {status}; kept in local outbox only");
                            break;
                        }
                        if (_cts.IsCancellationRequested) return;
                        Thread.Sleep(delay);
                        delay = Math.Min(delay * 2, 5000);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
        }

        private bool TryPost(string path, string json, out int status)
        {
            status = 0;
            try
            {
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var resp = _http.PostAsync(path, content, _cts.Token).GetAwaiter().GetResult();
                status = (int)resp.StatusCode;
                return resp.IsSuccessStatusCode;
            }
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is IOException)
            {
                return false;
            }
        }

        // ------------------------------------------------------------------ poller + heartbeat

        private void PollLoop()
        {
            var lastBeat = DateTime.MinValue;
            while (!_cts.IsCancellationRequested)
            {
                var now = DateTime.UtcNow;
                if (now - lastBeat >= _settings.HeartbeatInterval)
                {
                    lastBeat = now;
                    TryPost(HeartbeatPath, EnginePayloads.HeartbeatJson(_settings, AlgorithmName, _status, now), out _);
                }
                if (_intakeEnabled)
                {
                    foreach (var intent in FetchIntents(1))
                    {
                        Interlocked.Increment(ref _claimed);
                        _intents.Enqueue(intent);
                    }
                }
                _cts.Token.WaitHandle.WaitOne(_settings.PollInterval);
            }
        }

        /// <summary>One GET of claimed intents (each is handed out by the API at most once). Errors yield none.</summary>
        public List<BridgeIntent> FetchIntents(int max)
        {
            var result = new List<BridgeIntent>();
            try
            {
                var url = $"{IntentsPath}?engine_id={Uri.EscapeDataString(_settings.EngineId)}&max={max.ToString(CultureInfo.InvariantCulture)}";
                using var resp = _http.GetAsync(url, _cts.Token).GetAwaiter().GetResult();
                if (!resp.IsSuccessStatusCode)
                {
                    Interlocked.Increment(ref _pollFailures);
                    return result;
                }
                var body = resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
                return EnginePayloads.ParseIntents(body);
            }
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is IOException || e is JsonException || e is FormatException)
            {
                Interlocked.Increment(ref _pollFailures);
                return result;
            }
        }

        /// <summary>Sends the final heartbeat and drains the outbox (bounded wait).</summary>
        public void Stop(TimeSpan drain)
        {
            _intakeEnabled = false;
            _status = "STOPPED";
            if (_poller != null)
            {
                TryPost(HeartbeatPath, EnginePayloads.HeartbeatJson(_settings, AlgorithmName, "STOPPED", DateTime.UtcNow), out _);
            }
            _outbox.CompleteAdding();
            var deadline = DateTime.UtcNow + drain;
            while (_outbox.Count > 0 && DateTime.UtcNow < deadline)
            {
                Thread.Sleep(50);
            }
            _cts.Cancel();
        }

        public void Dispose()
        {
            if (!_cts.IsCancellationRequested) _cts.Cancel();
            _http.Dispose();
        }
    }
}
