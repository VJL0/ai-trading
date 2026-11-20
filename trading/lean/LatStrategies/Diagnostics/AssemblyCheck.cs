/*
 * Runtime proof of which LEAN / plugin assemblies are loaded (requirement: exactly one QuantConnect.Common,
 * the one built from the pinned LEAN source). Logged once per run as "LAT-ASSEMBLY-CHECK ..." lines.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Lat.Lean.Diagnostics
{
    public static class AssemblyCheck
    {
        private static readonly string[] Watched =
        {
            "QuantConnect.Common", "QuantConnect.Brokerages", "QuantConnect.Lean.Engine", "QuantConnect.Algorithm",
            "QuantConnect.Brokerages.Alpaca", "Alpaca.Markets", "LatStrategies", "LatAlpacaGate",
        };

        public static IEnumerable<string> Report()
        {
            var loaded = AppDomain.CurrentDomain.GetAssemblies();
            foreach (var name in Watched)
            {
                var copies = loaded.Where(a => string.Equals(a.GetName().Name, name, StringComparison.Ordinal)).ToList();
                if (copies.Count == 0)
                {
                    yield return $"LAT-ASSEMBLY-CHECK {name} copies=0";
                    continue;
                }
                var details = string.Join(" | ", copies.Select(a =>
                    $"{a.GetName().Version} {(string.IsNullOrEmpty(a.Location) ? "<loaded-from-bytes>" : a.Location)} mvid={a.ManifestModule.ModuleVersionId}"));
                yield return $"LAT-ASSEMBLY-CHECK {name} copies={copies.Count} {details}";
            }
            var common = typeof(QuantConnect.Symbol).Assembly;
            yield return $"LAT-ASSEMBLY-CHECK resolved typeof(Symbol) from {common.GetName().Name} {common.Location} mvid={common.ManifestModule.ModuleVersionId}";
        }

        /// <summary>Number of distinct QuantConnect.Common assemblies (must be 1).</summary>
        public static int QuantConnectCommonCopies() =>
            AppDomain.CurrentDomain.GetAssemblies().Count(a => a.GetName().Name == "QuantConnect.Common");
    }
}
