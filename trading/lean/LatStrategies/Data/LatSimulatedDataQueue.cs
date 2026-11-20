/*
 * LatSimulatedDataQueue: a tiny deterministic IDataQueueHandler for the LOCAL SIMULATED live path.
 * Every 'lat-sim-tick-ms' it emits, for each subscribed equity listed in 'lat-sim-quotes', one trade tick at
 * 'last' and one quote tick at 'bid'/'ask'. Prices are fixed decimals from the run config, e.g.
 *   "lat-sim-quotes": "{\"SPY\":{\"bid\":\"449.95\",\"ask\":\"450.05\",\"last\":\"450.00\"}}"
 * Symbols without configured prices receive NO data (so LEAN cannot trade them: fail closed).
 * Everything produced here is SIMULATED and labelled as such in logs; it is never a market feed.
 */

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using System.Threading;
using QuantConnect;
using QuantConnect.Configuration;
using QuantConnect.Data;
using QuantConnect.Data.Market;
using QuantConnect.Interfaces;
using QuantConnect.Lean.Engine.DataFeeds;
using QuantConnect.Logging;
using QuantConnect.Packets;
using QuantConnect.Securities;
using QuantConnect.Util;

namespace Lat.Lean.Data
{
    public sealed class SimQuote
    {
        public decimal Bid { get; init; }
        public decimal Ask { get; init; }
        public decimal Last { get; init; }
    }

    public class LatSimulatedDataQueue : IDataQueueHandler
    {
        public const string Label = "SIMULATED";

        private readonly IDataAggregator _aggregator;
        private readonly EventBasedDataQueueHandlerSubscriptionManager _subscriptions;
        private readonly IReadOnlyDictionary<string, SimQuote> _quotes;
        private readonly ConcurrentDictionary<Symbol, NodaTime.DateTimeZone> _zones = new ConcurrentDictionary<Symbol, NodaTime.DateTimeZone>();
        private readonly MarketHoursDatabase _mhdb = MarketHoursDatabase.FromDataFolder();
        private readonly Timer _timer;
        private long _emitted;

        public LatSimulatedDataQueue()
            : this(Composer.Instance.GetExportedValueByTypeName<IDataAggregator>(Config.Get("data-aggregator", "QuantConnect.Lean.Engine.DataFeeds.AggregationManager"), forceTypeNameOnExisting: false),
                   ParseQuotes(Config.Get("lat-sim-quotes")),
                   Config.GetInt("lat-sim-tick-ms", 1000))
        {
        }

        public LatSimulatedDataQueue(IDataAggregator aggregator, IReadOnlyDictionary<string, SimQuote> quotes, int tickMs)
        {
            _aggregator = aggregator ?? throw new ArgumentNullException(nameof(aggregator));
            _quotes = quotes ?? new Dictionary<string, SimQuote>();
            _subscriptions = new EventBasedDataQueueHandlerSubscriptionManager();
            _subscriptions.SubscribeImpl += (s, t) => true;
            _subscriptions.UnsubscribeImpl += (s, t) => true;
            var period = Math.Clamp(tickMs, 100, 60000);
            _timer = new Timer(_ => Emit(), null, period, period);
            Log.Trace($"LatSimulatedDataQueue(): {Label} data for [{string.Join(",", _quotes.Keys)}] every {period} ms");
        }

        public long Emitted => Interlocked.Read(ref _emitted);

        public static IReadOnlyDictionary<string, SimQuote> ParseQuotes(string json)
        {
            var result = new Dictionary<string, SimQuote>(StringComparer.OrdinalIgnoreCase);
            if (string.IsNullOrWhiteSpace(json)) return result;
            using var doc = JsonDocument.Parse(json);
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                decimal Read(string name)
                {
                    if (!p.Value.TryGetProperty(name, out var v) || v.ValueKind != JsonValueKind.String ||
                        !decimal.TryParse(v.GetString(), NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var d) || d <= 0m)
                    {
                        throw new ArgumentException($"lat-sim-quotes.{p.Name}.{name} must be a positive decimal string");
                    }
                    return d;
                }
                var q = new SimQuote { Bid = Read("bid"), Ask = Read("ask"), Last = Read("last") };
                if (q.Ask <= q.Bid) throw new ArgumentException($"lat-sim-quotes.{p.Name}: ask must exceed bid");
                result[p.Name.ToUpperInvariant()] = q;
            }
            return result;
        }

        public IEnumerator<BaseData> Subscribe(SubscriptionDataConfig dataConfig, EventHandler newDataAvailableHandler)
        {
            var enumerator = _aggregator.Add(dataConfig, newDataAvailableHandler);
            _subscriptions.Subscribe(dataConfig);
            return enumerator;
        }

        public void Unsubscribe(SubscriptionDataConfig dataConfig)
        {
            _subscriptions.Unsubscribe(dataConfig);
            _aggregator.Remove(dataConfig);
        }

        public void SetJob(LiveNodePacket job)
        {
        }

        public bool IsConnected => true;

        private void Emit()
        {
            try
            {
                var utcNow = DateTime.UtcNow;
                foreach (var symbol in _subscriptions.GetSubscribedSymbols())
                {
                    if (symbol.SecurityType != SecurityType.Equity || symbol.IsCanonical()) continue;
                    if (!_quotes.TryGetValue(symbol.Value, out var q)) continue;
                    var zone = _zones.GetOrAdd(symbol, s => _mhdb.GetExchangeHours(s.ID.Market, s, s.SecurityType).TimeZone);
                    var local = utcNow.ConvertFromUtc(zone);
                    _aggregator.Update(new Tick { Time = local, Symbol = symbol, Value = q.Last, Quantity = 100, TickType = TickType.Trade });
                    _aggregator.Update(new Tick(local, symbol, "", "", bidSize: 100, bidPrice: q.Bid, askPrice: q.Ask, askSize: 100));
                    Interlocked.Add(ref _emitted, 2);
                }
            }
            catch (Exception e)
            {
                Log.Error(e, "LatSimulatedDataQueue.Emit()");
            }
        }

        public void Dispose()
        {
            _timer.DisposeSafely();
        }
    }
}
