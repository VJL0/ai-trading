/*
 * Cost assumptions shared by the backtest and the live/paper paths (one source of truth).
 *
 * Defaults (verify against current regulator notices before relying on them):
 *  * Alpaca US equities commission: $0 per order.
 *  * SEC Section 31 fee on SELL proceeds: $27.80 per $1,000,000 (rate effective 2024-05-22). The SEC
 *    resets this rate by fee-rate advisory; override with config 'lat-fee-sec-rate-per-million'.
 *  * FINRA Trading Activity Fee on shares SOLD: $0.000166/share, capped at $8.30 per trade.
 *  * Regulatory fees rounded UP to the next cent (how Alpaca passes them through).
 *  * Slippage: half the quoted spread (or half of 'lat-assumed-spread-bps' when no quote is known)
 *    plus 'lat-slippage-bps' of adverse impact.
 * All values are decimals; floats are never used for money.
 */

using System;
using System.Globalization;

namespace Lat.Lean.Costs
{
    public sealed class LatCostConfig
    {
        public decimal CommissionPerOrder { get; init; } = 0m;
        public decimal SecFeeRatePerMillion { get; init; } = 27.80m;
        public decimal TafPerShare { get; init; } = 0.000166m;
        public decimal TafMaxPerTrade { get; init; } = 8.30m;
        public bool RoundRegulatoryFeesUpToCent { get; init; } = true;
        public decimal SlippageBps { get; init; } = 1m;
        public decimal AssumedSpreadBps { get; init; } = 2m;

        public static LatCostConfig Default { get; } = new LatCostConfig();

        /// <summary>Reads overrides from a key lookup (LEAN Config.Get / algorithm parameters). Invalid values throw.</summary>
        public static LatCostConfig FromLookup(Func<string, string> get)
        {
            if (get == null) return Default;
            var d = Default;
            return new LatCostConfig
            {
                CommissionPerOrder = Read(get, "lat-fee-commission-per-order", d.CommissionPerOrder),
                SecFeeRatePerMillion = Read(get, "lat-fee-sec-rate-per-million", d.SecFeeRatePerMillion),
                TafPerShare = Read(get, "lat-fee-taf-per-share", d.TafPerShare),
                TafMaxPerTrade = Read(get, "lat-fee-taf-max-per-trade", d.TafMaxPerTrade),
                SlippageBps = Read(get, "lat-slippage-bps", d.SlippageBps),
                AssumedSpreadBps = Read(get, "lat-assumed-spread-bps", d.AssumedSpreadBps),
                RoundRegulatoryFeesUpToCent = !string.Equals(get("lat-fee-round-up-to-cent"), "false", StringComparison.OrdinalIgnoreCase),
            };
        }

        private static decimal Read(Func<string, string> get, string key, decimal fallback)
        {
            var text = get(key);
            if (string.IsNullOrWhiteSpace(text)) return fallback;
            if (!decimal.TryParse(text.Trim(), NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var value) || value < 0m)
            {
                throw new ArgumentException($"cost setting '{key}' must be a non-negative decimal, got '{text}'");
            }
            return value;
        }

        public override string ToString() => string.Create(CultureInfo.InvariantCulture,
            $"commission/order={CommissionPerOrder} sec/$1M={SecFeeRatePerMillion} taf/share={TafPerShare} taf-max={TafMaxPerTrade} " +
            $"round-up-cent={RoundRegulatoryFeesUpToCent} slippage-bps={SlippageBps} assumed-spread-bps={AssumedSpreadBps}");
    }
}
