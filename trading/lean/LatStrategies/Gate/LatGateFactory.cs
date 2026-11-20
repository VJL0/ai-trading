/*
 * Builds an IntentGate from the LEAN run configuration. Fails closed: a missing/invalid public key, a missing or
 * unreadable journal throws, which makes LEAN abort brokerage creation / algorithm initialisation.
 */

using System;

namespace Lat.Lean.Gate
{
    public static class LatGateFactory
    {
        public const string JournalPathConfig = "lat-gate-journal-path";

        public static IntentGate Create(LatGateway gateway, Func<string, string> configGet, Func<DateTime> utcNow = null)
        {
            if (LatBuild.LiveExecutionCompiled)
            {
                throw new InvalidOperationException("LAT: live execution is not part of this build");
            }
            var verifier = IntentSignatureVerifier.FromEnvironmentOrConfig(configGet);
            var journalPath = configGet?.Invoke(JournalPathConfig);
            if (string.IsNullOrWhiteSpace(journalPath))
            {
                // without the durable journal an engine restart could re-execute an already consumed intent
                throw new InvalidOperationException($"LAT gate: config '{JournalPathConfig}' is required (replay protection across restarts); refusing to start");
            }
            var runId = configGet?.Invoke("lat-run-id");
            return new IntentGate(verifier, gateway, utcNow ?? (() => DateTime.UtcNow), new UsedIntentJournal(journalPath, runId));
        }
    }
}
