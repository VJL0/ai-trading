/*
 * Durable, append-only record of consumed intents so a signed intent token can never be executed twice,
 * even across LEAN restarts (replay protection at the final order boundary, spec invariant 3).
 *
 * Line format (tab separated, UTF-8): intent_id \t client_order_id \t lean_order_id \t run_id \t utc_iso
 * Entries present when the journal is opened belong to earlier engine runs and are refused outright;
 * entries appended during this run are tracked by the in-memory maps of IntentGate.
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace Lat.Lean.Gate
{
    public sealed class UsedIntentJournal
    {
        private readonly string _path;
        private readonly string _runId;
        private readonly object _lock = new object();
        private readonly HashSet<string> _priorIntentIds = new HashSet<string>(StringComparer.Ordinal);
        private readonly HashSet<string> _priorClientOrderIds = new HashSet<string>(StringComparer.Ordinal);

        public string Path => _path;
        public int PriorCount => _priorIntentIds.Count;

        /// <summary>Opens (creating if needed) the journal. Throws when it cannot be read: fail closed at startup.</summary>
        public UsedIntentJournal(string path, string runId)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("journal path required", nameof(path));
            _path = System.IO.Path.GetFullPath(path);
            _runId = string.IsNullOrWhiteSpace(runId) ? "run" : runId.Replace('\t', '_');
            var dir = System.IO.Path.GetDirectoryName(_path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            if (File.Exists(_path))
            {
                foreach (var line in File.ReadAllLines(_path, Encoding.UTF8))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = line.Split('\t');
                    if (parts.Length < 2 || parts[0].Length == 0 || parts[1].Length == 0)
                    {
                        throw new InvalidDataException($"corrupt intent journal line in {_path}; refusing to start");
                    }
                    _priorIntentIds.Add(parts[0]);
                    _priorClientOrderIds.Add(parts[1]);
                }
            }
        }

        public bool ConsumedByEarlierRun(string intentId, string clientOrderId)
        {
            lock (_lock)
            {
                return _priorIntentIds.Contains(intentId ?? string.Empty) || _priorClientOrderIds.Contains(clientOrderId ?? string.Empty);
            }
        }

        /// <summary>Appends and fsyncs one consumption record. Returns false (never throws) on I/O failure.</summary>
        public bool TryAppend(string intentId, string clientOrderId, int leanOrderId, out string error)
        {
            error = null;
            if (string.IsNullOrEmpty(intentId) || string.IsNullOrEmpty(clientOrderId) ||
                intentId.IndexOfAny(new[] { '\t', '\n', '\r' }) >= 0 || clientOrderId.IndexOfAny(new[] { '\t', '\n', '\r' }) >= 0)
            {
                error = "illegal intent identifiers";
                return false;
            }
            var line = string.Join("\t", intentId, clientOrderId, leanOrderId.ToString(CultureInfo.InvariantCulture), _runId,
                DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'", CultureInfo.InvariantCulture)) + "\n";
            lock (_lock)
            {
                try
                {
                    using var fs = new FileStream(_path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
                    var bytes = Encoding.UTF8.GetBytes(line);
                    fs.Write(bytes, 0, bytes.Length);
                    fs.Flush(true);
                    return true;
                }
                catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
                {
                    error = e.Message;
                    return false;
                }
            }
        }
    }
}
