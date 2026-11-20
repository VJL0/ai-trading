/*
 * LatIntentBridgeAlgorithm: the ONLY LEAN algorithm allowed in live/paper mode. It never decides to trade.
 *
 *   local API (risk engine signs intents) --GET /engine/intents/next--> this algorithm
 *     -> LimitOrder(DAY) built from the SIGNED string, tag = LAT intent token
 *     -> GatedBrokerageModel.CanSubmitOrder (ECDSA P-256 verify + exact match)          [algorithm hook]
 *     -> GatedPaperBrokerage / GatedAlpacaBrokerage.PlaceOrder (same gate again)          [final boundary]
 *   every OnOrderEvent -> POST /engine/events ; every refusal -> POST /engine/rejections ; heartbeat every 2 s
 *
 * Safety properties:
 *   * orders are created only inside SubmitIntent(), which is only fed by intents claimed from the API;
 *   * order parameters come from the signed string (never from the unauthenticated envelope);
 *   * intake starts only when every universe security has data and stops (halt) on any integrity breach:
 *     brokerage model replaced, unexpected non-intent order activity, gateway/brokerage mismatch;
 *   * no retries: an intent is submitted at most once; failures are reported, the ledger reconciles.
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Lat.Lean.Bridge;
using Lat.Lean.Brokerages;
using Lat.Lean.Costs;
using Lat.Lean.Diagnostics;
using Lat.Lean.Gate;
using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Brokerages;
using QuantConnect.Configuration;
using QuantConnect.Data;
using QuantConnect.Orders;

namespace Lat.Lean.Algorithms
{
    public class LatIntentBridgeAlgorithm : QCAlgorithm
    {
        private readonly object _sync = new object();
        private readonly object _eventSync = new object();
        private (string IntentId, string ClientOrderId)? _pendingMapping;
        private readonly Dictionary<int, (string IntentId, string ClientOrderId)> _intentByOrder = new Dictionary<int, (string, string)>();
        private readonly Dictionary<int, decimal> _cumFilled = new Dictionary<int, decimal>();
        private readonly Dictionary<string, Symbol> _universe = new Dictionary<string, Symbol>(StringComparer.Ordinal);
        // every intent / client order id this run has already turned into a LEAN order (duplicate deliveries are refused)
        private readonly Dictionary<string, int> _submittedIntents = new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _submittedClientOrderIds = new Dictionary<string, int>(StringComparer.Ordinal);
        private BridgeSettings _settings;
        private BridgeClient _client;
        private GatedBrokerageModel _model;
        private DateTime _startedUtc;
        private bool _ready;
        private bool _readyHookCalled;
        private string _haltReason;

        protected BridgeClient Bridge => _client;
        protected bool Halted => _haltReason != null;
        protected IReadOnlyDictionary<string, Symbol> Universe => _universe;

        public override void Initialize()
        {
            if (!LiveMode)
            {
                throw new InvalidOperationException("LatIntentBridgeAlgorithm runs in live mode only (backtests use LatBaselineAlgorithm)");
            }
            Func<string, string> get = k => Config.Get(k);
            _settings = BridgeSettings.FromConfig(get);

            var brokerage = Config.Get("live-mode-brokerage");
            if (brokerage != _settings.ExpectedBrokerageName)
            {
                throw new InvalidOperationException(
                    $"LAT: live-mode-brokerage '{brokerage}' refused; gateway {_settings.Gateway} requires '{_settings.ExpectedBrokerageName}' (gated final boundary)");
            }
            if (_settings.Gateway == LatGateway.LEAN_ALPACA_PAPER &&
                !string.Equals(Config.Get("alpaca-paper-trading"), "true", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("LAT: alpaca-paper-trading must be true (live trading is not part of this build)");
            }

            foreach (var line in AssemblyCheck.Report()) Log(line);
            if (AssemblyCheck.QuantConnectCommonCopies() != 1)
            {
                throw new InvalidOperationException("LAT: more than one QuantConnect.Common assembly loaded; refusing to run");
            }

            var gate = LatGateFactory.Create(_settings.Gateway, get);
            _client = new BridgeClient(_settings, m => Log(m)) { AlgorithmName = GetType().Name };
            var costs = LatCostConfig.FromLookup(get);
            IBrokerageModel inner = _settings.Gateway == LatGateway.LEAN_PAPER_SIM
                ? new DefaultBrokerageModel(AccountType.Cash)
                : new AlpacaBrokerageModel();
            _model = new GatedBrokerageModel(inner, gate, () => Transactions, _client, costs);
            SetBrokerageModel(_model);

            var resolution = (Resolution)Enum.Parse(typeof(Resolution), Config.Get("lat-resolution", "Second"), true);
            foreach (var ticker in _settings.Symbols)
            {
                // no fill-forward: a dead feed must look stale, never be papered over with synthetic bars
                _universe[ticker] = AddEquity(ticker, resolution, Market.USA, fillForward: false, leverage: 1m, extendedMarketHours: false).Symbol;
            }

            _startedUtc = DateTime.UtcNow;
            _client.SetStatus("STARTING");
            _client.Start(withPoller: true);
            Schedule.On(DateRules.EveryDay(), TimeRules.Every(TimeSpan.FromSeconds(1)), Tick);
            Log($"LAT bridge: engine {_settings.EngineId} gateway {_settings.Gateway} ({(_settings.Gateway == LatGateway.LEAN_PAPER_SIM ? "SIMULATED" : "PAPER")}) " +
                $"api {_settings.BaseUrl} key {gate.KeyId} costs [{costs}] symbols [{string.Join(",", _settings.Symbols)}]");
        }

        /// <summary>Hook for test subclasses (LatTestAlgorithms). The production bridge does nothing here.</summary>
        protected virtual void OnBridgeReady()
        {
        }

        private void Tick()
        {
            lock (_sync)
            {
                CheckIntegrity();
                if (!_ready && !Halted && _universe.Values.All(s => Securities[s].HasData))
                {
                    _ready = true;
                    _client.SetStatus("RUNNING");
                    _client.SetIntake(true);
                    Log("LAT bridge: all universe securities have data; intent intake ENABLED");
                }
                if (_ready && !_readyHookCalled)
                {
                    _readyHookCalled = true;
                    OnBridgeReady();
                    CheckIntegrity();
                }
                while (_client.TryDequeueIntent(out var intent))
                {
                    SubmitIntent(intent);
                }
                if (_settings.ExitAfterSeconds > 0 && (DateTime.UtcNow - _startedUtc).TotalSeconds >= _settings.ExitAfterSeconds)
                {
                    _client.SetIntake(false);
                    Quit($"LAT bridge: configured run time of {_settings.ExitAfterSeconds}s reached");
                }
            }
        }

        private void CheckIntegrity()
        {
            if (Halted) return;
            if (!ReferenceEquals(BrokerageModel, _model))
            {
                Halt($"brokerage model replaced at runtime ({BrokerageModel?.GetType().Name}); the gated model is mandatory");
            }
        }

        protected void Halt(string reason)
        {
            if (Halted) return;
            _haltReason = reason;
            _client.SetIntake(false);
            _client.SetStatus("HALTED");
            Error("LAT bridge HALTED: " + reason);
            _client.ReportRejection(null, null, "[LatIntentBridgeAlgorithm] engine halted, no further intents accepted: " + reason);
        }

        private void SubmitIntent(BridgeIntent bi)
        {
            if (IsDuplicateDelivery(bi.IntentId, bi.ClientOrderId))
            {
                return;
            }
            if (Halted)
            {
                _client.ReportRejection(bi.ClientOrderId, bi.IntentId, "[LatIntentBridgeAlgorithm] engine halted: " + _haltReason);
                return;
            }
            // parse the (still unverified) signing string to learn WHAT to submit; the gate verifies it
            if (!SignedIntent.TryParse(bi.SigningString, out var parsed, out var parseError))
            {
                _client.ReportRejection(bi.ClientOrderId, bi.IntentId, "[LatIntentBridgeAlgorithm] malformed intent, not submitted: " + parseError);
                return;
            }
            if (parsed.IntentId != bi.IntentId || parsed.ClientOrderId != bi.ClientOrderId)
            {
                _client.ReportRejection(bi.ClientOrderId, bi.IntentId, "[LatIntentBridgeAlgorithm] envelope ids do not match the signed string, not submitted");
                return;
            }
            if (!_universe.TryGetValue(parsed.Symbol, out var symbol))
            {
                _client.ReportRejection(parsed.ClientOrderId, parsed.IntentId, $"[LatIntentBridgeAlgorithm] symbol {parsed.Symbol} is not in this engine's universe, not submitted");
                return;
            }
            if (!IsDataFresh(symbol, out var staleness))
            {
                _client.ReportRejection(parsed.ClientOrderId, parsed.IntentId, "[LatIntentBridgeAlgorithm] market data stale or missing, not submitted: " + staleness);
                return;
            }
            string tag;
            try
            {
                tag = EnginePayloads.TagFor(bi);
            }
            catch (ArgumentException e)
            {
                _client.ReportRejection(parsed.ClientOrderId, parsed.IntentId, "[LatIntentBridgeAlgorithm] malformed signature, not submitted: " + e.Message);
                return;
            }
            if (parsed.Side != "BUY" && parsed.Side != "SELL")
            {
                _client.ReportRejection(parsed.ClientOrderId, parsed.IntentId, "[LatIntentBridgeAlgorithm] unsupported side");
                return;
            }

            var quantity = parsed.IsBuy ? parsed.Quantity : -parsed.Quantity;
            var props = new OrderProperties { TimeInForce = TimeInForce.Day };
            Log($"LAT bridge: submitting claimed intent {parsed.IntentId} coid={parsed.ClientOrderId} {parsed.Side} {parsed.Quantity} {parsed.Symbol} @ {parsed.LimitPrice}");
            // the LEAN order id is assigned synchronously inside LimitOrder; register it from the ticket and
            // also pre-register by the next id so events raised during submission are attributed correctly
            var ticket = LimitOrderWithMapping(symbol, quantity, parsed.LimitPrice, tag, props, parsed);
            if (ticket == null)
            {
                _client.ReportRejection(parsed.ClientOrderId, parsed.IntentId, "[LatIntentBridgeAlgorithm] LEAN returned no order ticket");
            }
        }

        /// <summary>Fail closed on price freshness: the last (non fill-forward) data point must be younger than
        /// lat-max-data-age-seconds.</summary>
        private bool IsDataFresh(Symbol symbol, out string detail)
        {
            var security = Securities[symbol];
            var last = security.GetLastData();
            if (last == null || last.IsFillForward)
            {
                detail = last == null ? "no data received" : "last data point is fill-forward";
                return false;
            }
            var ageSeconds = (DateTime.UtcNow - last.EndTime.ConvertToUtc(security.Exchange.TimeZone)).TotalSeconds;
            detail = string.Format(CultureInfo.InvariantCulture, "last {0} data at {1:O} ({2:F1}s old, max {3}s)",
                last.GetType().Name, last.EndTime, ageSeconds, _settings.MaxDataAgeSeconds);
            return ageSeconds <= _settings.MaxDataAgeSeconds;
        }

        /// <summary>A second delivery of an intent already submitted in this run is never resubmitted. The report carries
        /// no ids, so it cannot be applied to the (possibly live) original order; the API records an anomaly instead.</summary>
        private bool IsDuplicateDelivery(string intentId, string clientOrderId)
        {
            if ((intentId != null && _submittedIntents.TryGetValue(intentId, out var prior)) ||
                (clientOrderId != null && _submittedClientOrderIds.TryGetValue(clientOrderId, out prior)))
            {
                _client.ReportRejection(null, null, $"[LatIntentBridgeAlgorithm] duplicate delivery of intent {intentId} " +
                    $"(client_order_id {clientOrderId}) ignored: already submitted as LEAN order {prior}; not resubmitted");
                return true;
            }
            return false;
        }

        private OrderTicket LimitOrderWithMapping(Symbol symbol, decimal quantity, decimal limit, string tag, OrderProperties props, SignedIntent parsed)
        {
            lock (_eventSync)
            {
                _pendingMapping = (parsed.IntentId, parsed.ClientOrderId);
            }
            try
            {
                var ticket = LimitOrder(symbol, quantity, limit, asynchronous: false, tag: tag, orderProperties: props);
                var leanOrderId = ticket != null && ticket.OrderId > 0 ? ticket.OrderId : -1;
                _submittedIntents[parsed.IntentId] = leanOrderId;
                _submittedClientOrderIds[parsed.ClientOrderId] = leanOrderId;
                if (leanOrderId > 0)
                {
                    lock (_eventSync)
                    {
                        _intentByOrder[leanOrderId] = (parsed.IntentId, parsed.ClientOrderId);
                    }
                }
                return ticket;
            }
            finally
            {
                lock (_eventSync)
                {
                    _pendingMapping = null;
                }
            }
        }

        public override void OnOrderEvent(OrderEvent orderEvent)
        {
            (string IntentId, string ClientOrderId) ids;
            decimal cum;
            lock (_eventSync)
            {
                if (!_intentByOrder.TryGetValue(orderEvent.OrderId, out ids))
                {
                    if (_pendingMapping.HasValue && IsOurTag(orderEvent.OrderId))
                    {
                        ids = _pendingMapping.Value;
                        _intentByOrder[orderEvent.OrderId] = ids;
                    }
                    else
                    {
                        OnNonIntentOrderEvent(orderEvent);
                        return;
                    }
                }
                _cumFilled.TryGetValue(orderEvent.OrderId, out cum);
                cum += Math.Abs(orderEvent.FillQuantity);
                _cumFilled[orderEvent.OrderId] = cum;
            }

            if (orderEvent.Status == OrderStatus.Invalid && _client.UnkeyedRefusal(orderEvent.OrderId))
            {
                // duplicate/replayed intent refused by the gate: already reported without ids; posting this event under
                // the intent's ids could make the ledger change the (possibly live) original order
                Log($"LAT bridge: order {orderEvent.OrderId} refused as a duplicate/replay of {ids.ClientOrderId}; event not posted");
                return;
            }
            var order = Transactions.GetOrderById(orderEvent.OrderId);
            var isFill = orderEvent.Status == OrderStatus.Filled || orderEvent.Status == OrderStatus.PartiallyFilled;
            _client.ReportEvent(new EnginePayloads.OrderEventPayload
            {
                ClientOrderId = ids.ClientOrderId,
                IntentId = ids.IntentId,
                LeanOrderId = orderEvent.OrderId,
                LeanEventId = orderEvent.Id,
                BrokerOrderId = order?.BrokerId?.LastOrDefault(),
                Status = orderEvent.Status.ToString(),
                FillQuantity = isFill ? Math.Abs(orderEvent.FillQuantity) : (decimal?)null,
                FillPrice = isFill ? orderEvent.FillPrice : (decimal?)null,
                CumFilledQuantity = cum,
                EventUtc = orderEvent.UtcTime,
                Message = orderEvent.Message,
            });
            // every claimed intent that LEAN refused before the gate saw it (e.g. LEAN's own buying-power or
            // price-data checks) is reported as a rejection too; gate refusals were already reported by the sink,
            // brokerage-boundary refusals by the gated brokerage itself
            if (orderEvent.Status == OrderStatus.Invalid && !_client.RefusalReported(orderEvent.OrderId) &&
                (orderEvent.Message ?? string.Empty).IndexOf("Brokerage failed to place orders", StringComparison.Ordinal) < 0)
            {
                _client.ReportRejection(ids.ClientOrderId, ids.IntentId, "[LEAN pre-submission checks] order invalid, not sent to the brokerage: " + orderEvent.Message);
            }
            Log(string.Format(CultureInfo.InvariantCulture, "LAT bridge: event order={0} event={1} coid={2} status={3} fill={4}@{5} cum={6} broker_id={7}",
                orderEvent.OrderId, orderEvent.Id, ids.ClientOrderId, orderEvent.Status, orderEvent.FillQuantity, orderEvent.FillPrice, cum,
                order?.BrokerId?.LastOrDefault() ?? "-"));
        }

        private bool IsOurTag(int orderId)
        {
            var tag = Transactions.GetOrderById(orderId)?.Tag;
            return tag != null && tag.StartsWith(IntentToken.TagPrefix, StringComparison.Ordinal);
        }

        /// <summary>An order that did not come from a claimed intent produced an event: never acceptable once it
        /// goes beyond a refusal. Reported; activity other than Invalid halts the bridge.</summary>
        private void OnNonIntentOrderEvent(OrderEvent e)
        {
            var summary = LeanOrderFacts.Describe(Transactions.GetOrderById(e.OrderId));
            if (e.Status == OrderStatus.Invalid)
            {
                Log($"LAT bridge: non-intent order {e.OrderId} refused ({e.Status}): {summary}");
                return;
            }
            Halt($"non-intent order activity detected: order {e.OrderId} status {e.Status} ({summary})");
        }

        public override void OnBrokerageMessage(BrokerageMessageEvent messageEvent)
        {
            Log($"LAT bridge: brokerage message {messageEvent.Type} {messageEvent.Code}: {messageEvent.Message}");
        }

        public override void OnEndOfAlgorithm()
        {
            if (_client != null)
            {
                _client.SetIntake(false);
                // claimed but never submitted (engine stopping): report so the ledger can reject them; never kept for later
                while (_client.TryDequeueIntent(out var unsent))
                {
                    if (IsDuplicateDelivery(unsent.IntentId, unsent.ClientOrderId)) continue;
                    _client.ReportRejection(unsent.ClientOrderId, unsent.IntentId,
                        "[LatIntentBridgeAlgorithm] engine stopping: claimed intent was not submitted (never sent to the brokerage)");
                }
            }
            Log($"LAT bridge: stopping; claimed={_client?.Claimed} sent={_client?.Sent} send_failures={_client?.SendFailures} poll_failures={_client?.PollFailures}");
            _client?.Stop(TimeSpan.FromSeconds(5));
        }
    }
}
