/*
 * Settings of the LEAN <-> local API engine bridge (docs/architecture/api-contract.md, "Engine bridge").
 * Read from the per-run LEAN config (written by scripts/lean_live_sim.sh / lean_paper_alpaca.sh); the engine
 * token only from the environment (LAT_ENGINE_TOKEN), never from a file in the repo. Fail closed on anything
 * missing or not loopback.
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Text.RegularExpressions;
using Lat.Lean.Gate;

namespace Lat.Lean.Bridge
{
    public sealed class BridgeSettings
    {
        public const string EngineTokenEnv = "LAT_ENGINE_TOKEN";
        public const string DefaultBaseUrl = "http://127.0.0.1:8765";
        private static readonly Regex EngineIdPattern = new Regex("^[A-Za-z0-9_.:\\-]{1,100}$", RegexOptions.Compiled);

        public Uri BaseUrl { get; init; }
        public string EngineToken { get; init; }
        public string EngineId { get; init; }
        public LatGateway Gateway { get; init; }
        public string Mode { get; init; } = LatBuild.AcceptedMode;
        public string LeanCommit { get; init; } = "";
        public IReadOnlyList<string> Symbols { get; init; } = new[] { "SPY" };
        public TimeSpan PollInterval { get; init; } = TimeSpan.FromMilliseconds(500);
        public TimeSpan HeartbeatInterval { get; init; } = TimeSpan.FromSeconds(2);
        public TimeSpan HttpTimeout { get; init; } = TimeSpan.FromSeconds(5);
        /// <summary>Local JSONL copy of everything sent to the API (audit trail even if the API is down).</summary>
        public string OutboxPath { get; init; }
        /// <summary>0 = run until stopped. Used by the simulated live demo/tests to end the engine cleanly.</summary>
        public int ExitAfterSeconds { get; init; }
        /// <summary>Claimed intents are not submitted when the symbol's last market data is older than this.</summary>
        public int MaxDataAgeSeconds { get; init; } = 120;

        public string ExpectedBrokerageName => Gateway == LatGateway.LEAN_PAPER_SIM ? "GatedPaperBrokerage" : "GatedAlpacaBrokerage";

        public static BridgeSettings FromConfig(Func<string, string> get, Func<string, string> env = null)
        {
            if (get == null) throw new ArgumentNullException(nameof(get));
            env ??= Environment.GetEnvironmentVariable;

            var url = get("lat-bridge-url");
            var baseUrl = ParseLoopbackUrl(string.IsNullOrWhiteSpace(url) ? DefaultBaseUrl : url.Trim());

            var token = env(EngineTokenEnv);
            if (string.IsNullOrWhiteSpace(token) || token.Length < 16)
            {
                throw new InvalidOperationException($"LAT bridge: {EngineTokenEnv} is not set (or too short); refusing to start");
            }

            var gwText = get("lat-gateway");
            if (!Enum.TryParse<LatGateway>(gwText ?? "", false, out var gateway) || !Enum.IsDefined(typeof(LatGateway), gateway))
            {
                throw new InvalidOperationException($"LAT bridge: 'lat-gateway' must be LEAN_PAPER_SIM or LEAN_ALPACA_PAPER, got '{gwText}'");
            }

            var mode = get("lat-mode");
            if (!string.IsNullOrEmpty(mode) && mode != LatBuild.AcceptedMode)
            {
                throw new InvalidOperationException($"LAT bridge: mode '{mode}' refused; this build executes PAPER only");
            }

            var engineId = get("lat-engine-id");
            if (string.IsNullOrWhiteSpace(engineId))
            {
                engineId = $"lean-{gateway.ToString().ToLowerInvariant().Replace('_', '-')}-{Environment.ProcessId}";
            }
            if (!EngineIdPattern.IsMatch(engineId)) throw new InvalidOperationException("LAT bridge: illegal engine id");

            var symbols = (get("lat-symbols") ?? "SPY").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => s.ToUpperInvariant()).Distinct().ToList();
            if (symbols.Count == 0 || symbols.Any(s => !Regex.IsMatch(s, "^[A-Z][A-Z0-9.]{0,9}$")))
            {
                throw new InvalidOperationException("LAT bridge: 'lat-symbols' must list US equity tickers");
            }

            return new BridgeSettings
            {
                BaseUrl = baseUrl,
                EngineToken = token,
                EngineId = engineId,
                Gateway = gateway,
                LeanCommit = get("lat-lean-commit") ?? "",
                Symbols = symbols,
                PollInterval = TimeSpan.FromMilliseconds(ReadInt(get, "lat-poll-ms", 500, 50, 60000)),
                HeartbeatInterval = TimeSpan.FromMilliseconds(ReadInt(get, "lat-heartbeat-ms", 2000, 200, 60000)),
                OutboxPath = get("lat-bridge-outbox-path"),
                ExitAfterSeconds = ReadInt(get, "lat-exit-after-seconds", 0, 0, 7 * 24 * 3600),
                MaxDataAgeSeconds = ReadInt(get, "lat-max-data-age-seconds", 120, 1, 3600),
            };
        }

        /// <summary>Only http(s) URLs whose host is a loopback address are accepted: intents never leave this Mac.</summary>
        public static Uri ParseLoopbackUrl(string text)
        {
            if (!Uri.TryCreate(text, UriKind.Absolute, out var uri) || (uri.Scheme != "http" && uri.Scheme != "https"))
            {
                throw new InvalidOperationException($"LAT bridge: '{text}' is not an http(s) URL");
            }
            var host = uri.Host.Trim('[', ']');
            var loopback = host == "localhost" || (IPAddress.TryParse(host, out var ip) && IPAddress.IsLoopback(ip));
            if (!loopback)
            {
                throw new InvalidOperationException($"LAT bridge: base URL host '{uri.Host}' is not loopback; refusing (local-only product boundary)");
            }
            if (!string.IsNullOrEmpty(uri.UserInfo)) throw new InvalidOperationException("LAT bridge: credentials in URL refused");
            return new Uri(uri.GetLeftPart(UriPartial.Authority));
        }

        private static int ReadInt(Func<string, string> get, string key, int fallback, int min, int max)
        {
            var t = get(key);
            if (string.IsNullOrWhiteSpace(t)) return fallback;
            if (!int.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) || v < min || v > max)
            {
                throw new InvalidOperationException($"LAT bridge: '{key}' must be an integer in [{min}, {max}]");
            }
            return v;
        }
    }
}
