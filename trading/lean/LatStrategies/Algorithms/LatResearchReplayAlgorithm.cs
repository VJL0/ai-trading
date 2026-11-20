// Backtest-only replay of hash-bound OOS scores. No foundation-model inference in the order path.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Lat.Lean.Brokerages;
using Lat.Lean.Costs;
using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Brokerages;
using QuantConnect.Data;
using QuantConnect.Orders;

namespace Lat.Lean.Algorithms
{
    public class LatResearchReplayAlgorithm : QCAlgorithm
    {
        private record Score(DateTime Decision, DateTime Exit, string Ticker, decimal Value);
        private Queue<Score> _scores;
        private readonly Dictionary<string, Symbol> _symbols = new();
        private OrderTicket _working;
        private Symbol _held;
        private DateTime _exit;
        private DateTime _submitted;

        public override void Initialize()
        {
            if (LiveMode) throw new InvalidOperationException("Research replay is backtest-only");
            var path = GetParameter("scores-file");
            var expected = GetParameter("scores-sha256");
            var actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            if (actual != expected) throw new ArgumentException("OOS score hash mismatch");
            DateTime FromNs(string value) => DateTime.UnixEpoch.AddTicks(long.Parse(value, CultureInfo.InvariantCulture) / 100);
            var records = File.ReadLines(path).Skip(1).Select(line => line.Split(',')).Select(c =>
                new Score(FromNs(c[0]), FromNs(c[1]), c[2], decimal.Parse(c[3], CultureInfo.InvariantCulture)))
                .OrderBy(s => s.Decision).ThenByDescending(s => s.Value).ThenBy(s => s.Ticker).ToList();
            if (records.Count == 0 || records.Any(s => s.Exit <= s.Decision)) throw new ArgumentException("Invalid scores");
            SetTimeZone(TimeZones.Utc);
            SetStartDate(records.First().Decision.Date);
            SetEndDate(records.Last().Exit.Date);
            SetCash(100000);
            SetBrokerageModel(new LatBrokerageModel(new AlpacaBrokerageModel(), LatCostConfig.FromLookup(k => GetParameter(k))));
            foreach (var ticker in records.Select(s => s.Ticker).Distinct())
                _symbols[ticker] = AddEquity(ticker, Resolution.Minute, Market.USA, fillForward: false,
                                           leverage: 1m, extendedMarketHours: false).Symbol;
            _scores = new Queue<Score>(records);
            Log("OOS research replay: simulated limit fills with LAT fee/slippage models; no promotion authority");
        }

        public override void OnData(Slice slice)
        {
            // Scores only become usable at their recorded decision time. Never consume future scores.
            if (_working != null && _working.Status.IsOpen() && UtcTime > _submitted.AddMinutes(1))
                _working.Cancel("one-bar limit expiry");
            if (_held != null && Portfolio[_held].Quantity > 0 && UtcTime >= _exit
                && (_working == null || !_working.Status.IsOpen()) && slice.Bars.TryGetValue(_held, out var exitBar))
            {
                _submitted = UtcTime;
                _working = LimitOrder(_held, -Portfolio[_held].Quantity, Math.Floor(exitBar.Close * .999m * 100m) / 100m,
                                      tag: "OOS exit", orderProperties: new OrderProperties { TimeInForce = TimeInForce.Day });
            }
            while (_scores.Count > 0 && _scores.Peek().Decision <= UtcTime)
            {
                var score = _scores.Dequeue();
                if (score.Value <= 0 || score.Exit <= UtcTime || UtcTime - score.Decision >= TimeSpan.FromMinutes(1)
                    || Portfolio.Invested || (_working != null && _working.Status.IsOpen())) continue;
                var symbol = _symbols[score.Ticker];
                if (!slice.Bars.TryGetValue(symbol, out var bar)) continue;
                var limit = Math.Ceiling(bar.Close * 1.001m * 100m) / 100m;
                var quantity = Math.Floor(Portfolio.Cash * .95m / limit);
                if (quantity < 1) continue;
                _held = symbol;
                _exit = score.Exit;
                _submitted = UtcTime;
                _working = LimitOrder(symbol, quantity, limit, tag: "OOS entry",
                                      orderProperties: new OrderProperties { TimeInForce = TimeInForce.Day });
            }
        }
    }
}
