/*
 * LatBaselineAlgorithm: deterministic, non-ML, long-only SMA trend baseline (one position at a time,
 * regular hours only, LIMIT orders only) with the LAT cost models (LatFeeModel, LatSpreadSlippageModel,
 * LatFillModel) applied through LatBrokerageModel over LEAN's AlpacaBrokerageModel - the same cost/account
 * policy objects the live bridge path installs. It makes NO claim of an edge; it is the benchmark every AI
 * pipeline must beat after costs.
 *
 * Parameters (LEAN "parameters" config, all optional):
 *   symbol=SPY  resolution=Daily|Minute  start=2010-01-04  end=2020-12-31  cash=100000
 *   fast=20 slow=100 (bars)  entry-offset-bps=10  allocation=0.95
 *
 * Time-in-force: Minute runs use DAY limit orders (as the live path). With Daily bars LEAN expires DAY orders
 * at the close before the next daily bar can be evaluated, so Daily runs submit GTC limits and cancel any
 * order still open after the one following session bar (DAY semantics emulated on daily bars).
 */

using System;
using System.Globalization;
using Lat.Lean.Brokerages;
using Lat.Lean.Costs;
using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Brokerages;
using QuantConnect.Data;
using QuantConnect.Indicators;
using QuantConnect.Orders;

namespace Lat.Lean.Algorithms
{
    public class LatBaselineAlgorithm : QCAlgorithm
    {
        private Symbol _symbol;
        private SimpleMovingAverage _fast;
        private SimpleMovingAverage _slow;
        private Resolution _resolution;
        private decimal _offsetBps;
        private decimal _allocation;
        private LatCostConfig _costs;
        private int _barsSinceOrder;
        private OrderTicket _working;

        public override void Initialize()
        {
            if (LiveMode)
            {
                // the baseline originates orders itself, which the live gate forbids by design
                throw new InvalidOperationException("LatBaselineAlgorithm is backtest-only; live/paper execution uses LatIntentBridgeAlgorithm");
            }

            var ticker = GetParameter("symbol", "SPY").ToUpperInvariant();
            _resolution = (Resolution)Enum.Parse(typeof(Resolution), GetParameter("resolution", "Daily"), true);
            if (_resolution != Resolution.Daily && _resolution != Resolution.Minute)
            {
                throw new ArgumentException("resolution must be Daily or Minute");
            }
            SetStartDate(DateTime.ParseExact(GetParameter("start", _resolution == Resolution.Daily ? "2010-01-04" : "2013-10-07"), "yyyy-MM-dd", CultureInfo.InvariantCulture));
            SetEndDate(DateTime.ParseExact(GetParameter("end", _resolution == Resolution.Daily ? "2020-12-31" : "2013-10-11"), "yyyy-MM-dd", CultureInfo.InvariantCulture));
            SetCash(decimal.Parse(GetParameter("cash", "100000"), CultureInfo.InvariantCulture));
            _offsetBps = decimal.Parse(GetParameter("entry-offset-bps", "10"), CultureInfo.InvariantCulture);
            _allocation = decimal.Parse(GetParameter("allocation", "0.95"), CultureInfo.InvariantCulture);
            if (_allocation <= 0m || _allocation > 1m) throw new ArgumentException("allocation must be in (0, 1] (no leverage)");

            _costs = LatCostConfig.FromLookup(k => GetParameter(k));
            SetBrokerageModel(new LatBrokerageModel(new AlpacaBrokerageModel(), _costs));
            Log($"LAT baseline: costs {_costs}; brokerage model LatBrokerageModel(AlpacaBrokerageModel), cash account, leverage 1");

            _symbol = AddEquity(ticker, _resolution, Market.USA, fillForward: true, leverage: 1m, extendedMarketHours: false).Symbol;
            SetBenchmark(_symbol);

            var fast = int.Parse(GetParameter("fast", _resolution == Resolution.Daily ? "20" : "15"), CultureInfo.InvariantCulture);
            var slow = int.Parse(GetParameter("slow", _resolution == Resolution.Daily ? "100" : "60"), CultureInfo.InvariantCulture);
            if (fast <= 0 || slow <= fast) throw new ArgumentException("require 0 < fast < slow");
            _fast = SMA(_symbol, fast, _resolution);
            _slow = SMA(_symbol, slow, _resolution);
            SetWarmUp(slow, _resolution);
        }

        public override void OnData(Slice slice)
        {
            if (IsWarmingUp || !slice.Bars.TryGetValue(_symbol, out var bar)) return;
            if (!_fast.IsReady || !_slow.IsReady) return;

            if (_working != null)
            {
                _barsSinceOrder++;
                var open = _working.Status.IsOpen();
                if (open && _resolution == Resolution.Daily && _barsSinceOrder >= 1)
                {
                    _working.Cancel("DAY emulation on daily bars: not filled in the next session");
                }
                if (open) return;
                _working = null;
            }

            // regular session only (minute bars carry exchange time), with a quiet window around open/close
            if (_resolution == Resolution.Minute)
            {
                var t = Time.TimeOfDay;
                var holding = Portfolio[_symbol].Quantity;
                if (holding > 0 && t >= new TimeSpan(15, 50, 0))
                {
                    Submit(-holding, bar.Close, "flat into the close");
                    return;
                }
                if (t < new TimeSpan(9, 45, 0) || t >= new TimeSpan(15, 45, 0)) return;
            }

            var held = Portfolio[_symbol].Quantity;
            if (held == 0m && _fast.Current.Value > _slow.Current.Value)
            {
                var limit = RoundToTick(bar.Close * (1m + _offsetBps / 10_000m), up: true);
                var qty = Math.Floor(Portfolio.Cash * _allocation / limit);
                if (qty >= 1m) Submit(qty, limit, "fast SMA above slow SMA");
            }
            else if (held > 0m && _fast.Current.Value < _slow.Current.Value)
            {
                Submit(-held, bar.Close, "fast SMA below slow SMA");
            }
        }

        private void Submit(decimal quantity, decimal referencePrice, string why)
        {
            var limit = quantity > 0m ? RoundToTick(referencePrice, up: true)
                : RoundToTick(referencePrice * (1m - _offsetBps / 10_000m), up: false);
            var props = new OrderProperties
            {
                TimeInForce = _resolution == Resolution.Daily ? TimeInForce.GoodTilCanceled : TimeInForce.Day,
            };
            _barsSinceOrder = 0;
            _working = LimitOrder(_symbol, quantity, limit, tag: "baseline: " + why, orderProperties: props);
        }

        private decimal RoundToTick(decimal price, bool up)
        {
            var tick = Securities[_symbol].SymbolProperties.MinimumPriceVariation;
            if (tick <= 0m) tick = 0.01m;
            return (up ? Math.Ceiling(price / tick) : Math.Floor(price / tick)) * tick;
        }

        public override void OnOrderEvent(OrderEvent orderEvent)
        {
            if (orderEvent.Status == OrderStatus.Filled || orderEvent.Status == OrderStatus.PartiallyFilled)
            {
                Debug($"LAT fill {orderEvent.Symbol} qty={orderEvent.FillQuantity} px={orderEvent.FillPrice} fee={orderEvent.OrderFee.Value.Amount}");
            }
        }

        public override void OnEndOfAlgorithm()
        {
            Log($"LAT baseline done: equity={Portfolio.TotalPortfolioValue} fees={Portfolio.TotalFees} orders={Transactions.OrdersCount}");
        }
    }
}
