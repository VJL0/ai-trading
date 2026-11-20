/*
 * LAT intent gate: the pure, LEAN-type-free decision logic shared by
 *   * GatedBrokerageModel.CanSubmitOrder         (algorithm-level LEAN hook), and
 *   * GatedPaperBrokerage / GatedAlpacaBrokerage  (engine-level final boundary at PlaceOrder).
 *
 * Every rule fails closed: anything not explicitly supported is refused, never degraded.
 */

using System;
using System.Collections.Generic;

namespace Lat.Lean.Gate
{
    /// <summary>Compile-time execution boundary, mirrors packages/core/modes.py. Not configurable.</summary>
    public static class LatBuild
    {
        public const bool LiveExecutionCompiled = false;
        public const string AcceptedMode = "PAPER";
    }

    /// <summary>The execution gateways this LEAN build can act as (paper/simulated only).</summary>
    public enum LatGateway
    {
        /// <summary>LEAN PaperBrokerage on a simulated data feed (SIMULATED).</summary>
        LEAN_PAPER_SIM,
        /// <summary>LEAN + official Alpaca plugin against the Alpaca PAPER endpoint (PAPER).</summary>
        LEAN_ALPACA_PAPER,
    }

    /// <summary>The subset of a LEAN order + security state the gate decides on.</summary>
    public sealed class OrderFacts
    {
        public int LeanOrderId { get; set; }
        public string Tag { get; set; }
        public string Symbol { get; set; }
        public string SecurityType { get; set; }
        public string Market { get; set; }
        /// <summary>LEAN order type name, e.g. "Limit", "Market".</summary>
        public string OrderType { get; set; }
        /// <summary>Signed LEAN quantity: positive buy, negative sell.</summary>
        public decimal Quantity { get; set; }
        public decimal? LimitPrice { get; set; }
        /// <summary>"DAY", "GTC", "GTD" or the LEAN type name for anything else.</summary>
        public string TimeInForce { get; set; }
        public bool OutsideRegularTradingHours { get; set; }
        /// <summary>Set when the order properties are of an unknown type (refused: cannot inspect them).</summary>
        public string UnsupportedPropertiesType { get; set; }
        public bool HasContingencyOrGroup { get; set; }
        /// <summary>Regular-session exchange state at submission time. Defaults to false (closed) so an unset fact fails closed.</summary>
        public bool ExchangeOpen { get; set; }
        public decimal HeldQuantity { get; set; }
        /// <summary>Remaining quantity of OTHER open sell orders for this symbol (absolute value).</summary>
        public decimal PendingSellQuantity { get; set; }
    }

    public sealed class GateDecision
    {
        public bool Allowed { get; private set; }
        public string Reason { get; private set; }
        public SignedIntent Intent { get; private set; }
        /// <summary>Best-effort ids for refusal reporting even when the signature is invalid (never trusted).</summary>
        public string ClaimedIntentId { get; private set; }
        public string ClaimedClientOrderId { get; private set; }

        public static GateDecision Allow(SignedIntent intent) =>
            new GateDecision { Allowed = true, Reason = "ok", Intent = intent, ClaimedIntentId = intent.IntentId, ClaimedClientOrderId = intent.ClientOrderId };

        public static GateDecision Refuse(string reason, string intentId = null, string clientOrderId = null) =>
            new GateDecision { Allowed = false, Reason = reason, ClaimedIntentId = intentId, ClaimedClientOrderId = clientOrderId };
    }

    public sealed class IntentGate
    {
        public const string SupportedSignatureAlg = IntentSignatureVerifier.SignatureAlg;

        private readonly IntentSignatureVerifier _verifier;
        private readonly LatGateway _gateway;
        private readonly Func<DateTime> _utcNow;
        private readonly object _lock = new object();
        private readonly Dictionary<string, int> _usedIntents = new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _usedClientOrderIds = new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly UsedIntentJournal _journal;

        /// <summary>Allowed clock skew for intents created "in the future" relative to the engine clock.</summary>
        public TimeSpan MaxClockSkew { get; set; } = TimeSpan.FromSeconds(5);

        public LatGateway Gateway => _gateway;
        public string KeyId => _verifier.KeyId;

        /// <param name="journal">Optional durable record of consumed intents (replay protection across engine restarts).
        /// Intents found in the journal are refused for any LEAN order (order ids are not stable across restarts).</param>
        public IntentGate(IntentSignatureVerifier verifier, LatGateway gateway, Func<DateTime> utcNow, UsedIntentJournal journal = null)
        {
            _verifier = verifier ?? throw new ArgumentNullException(nameof(verifier));
            _gateway = gateway;
            _utcNow = utcNow ?? (() => DateTime.UtcNow);
            _journal = journal;
        }

        /// <summary>
        /// Decide whether the order may proceed. When <paramref name="consume"/> is true and the order is
        /// allowed, the intent id and client order id are bound to the LEAN order id so a replayed token
        /// on a different order is refused.
        /// </summary>
        public GateDecision Evaluate(OrderFacts o, bool consume)
        {
            if (o == null) return GateDecision.Refuse("no order");
            if (LatBuild.LiveExecutionCompiled)
            {
                // unreachable in this build; kept so flipping the constant alone does not open a path
                return GateDecision.Refuse("live execution requires the human release gate (not present in this build)");
            }

            if (!IntentToken.TryDecode(o.Tag, out var signingString, out var signatureHex, out var tokenError))
            {
                return GateDecision.Refuse(tokenError);
            }

            // best-effort ids for the refusal report (not trusted until verified)
            SignedIntent.TryParse(signingString, out var claimed, out _);
            var cid = claimed?.IntentId;
            var ccoid = claimed?.ClientOrderId;

            if (!_verifier.Verify(signingString, signatureHex))
            {
                return GateDecision.Refuse("intent signature verification failed (forged, tampered or wrong key)", cid, ccoid);
            }
            if (!SignedIntent.TryParse(signingString, out var intent, out var parseError))
            {
                return GateDecision.Refuse("malformed signed intent: " + parseError, cid, ccoid);
            }
            GateDecision Refuse(string reason) => GateDecision.Refuse(reason, intent.IntentId, intent.ClientOrderId);

            if (intent.SignatureAlg != SupportedSignatureAlg) return Refuse($"unsupported signature_alg '{intent.SignatureAlg}'");
            if (intent.KeyId != _verifier.KeyId) return Refuse($"key_id '{intent.KeyId}' does not match the configured verification key '{_verifier.KeyId}'");
            if (intent.Mode != LatBuild.AcceptedMode) return Refuse($"mode '{intent.Mode}' is not executable by this build (PAPER only)");
            if (intent.Gateway != _gateway.ToString()) return Refuse($"intent is for gateway '{intent.Gateway}', this engine is '{_gateway}'");
            if (intent.RequiresHumanApproval && string.IsNullOrEmpty(intent.HumanApprovalId)) return Refuse("intent requires human approval but carries no human_approval_id");

            var now = _utcNow();
            if (now >= intent.ExpiresAtUtc) return Refuse($"intent expired at {intent.ExpiresAtUtc:O} (now {now:O})");
            if (intent.CreatedAtUtc > now + MaxClockSkew) return Refuse($"intent created_at {intent.CreatedAtUtc:O} is in the future (now {now:O})");

            // ---- unsupported features: reject, never degrade
            if (intent.OrderType != "LIMIT") return Refuse($"signed order_type '{intent.OrderType}' not supported (LIMIT only)");
            if (o.OrderType != "Limit") return Refuse($"LEAN order type '{o.OrderType}' not allowed (LIMIT only)");
            if (intent.TimeInForce != "DAY") return Refuse($"signed time_in_force '{intent.TimeInForce}' not supported (DAY only)");
            if (o.TimeInForce != "DAY") return Refuse($"LEAN time in force '{o.TimeInForce}' does not match signed DAY");
            if (intent.ExtendedHours) return Refuse("extended-hours intents are not permitted");
            if (o.OutsideRegularTradingHours) return Refuse("order requests outside-regular-hours execution");
            if (!string.IsNullOrEmpty(o.UnsupportedPropertiesType)) return Refuse($"unsupported order properties type '{o.UnsupportedPropertiesType}'");
            if (o.HasContingencyOrGroup) return Refuse("contingent/combo/bracket orders are not supported");
            if (o.SecurityType != "Equity") return Refuse($"security type '{o.SecurityType}' not supported (US equities only)");
            if (!o.ExchangeOpen) return Refuse("exchange is not in its regular session (regular-hours orders only)");
            if (!string.Equals(o.Market, "usa", StringComparison.OrdinalIgnoreCase)) return Refuse($"market '{o.Market}' not supported (usa only)");

            // ---- exact match between the LEAN order and the signed parameters
            if (o.Symbol != intent.Symbol) return Refuse($"symbol mismatch: order '{o.Symbol}' vs signed '{intent.Symbol}'");
            if (intent.Quantity <= 0m || intent.LimitPrice <= 0m) return Refuse("signed quantity and limit_price must be positive");
            if (intent.Quantity != decimal.Truncate(intent.Quantity)) return Refuse("fractional share intents are not supported by the LEAN path");
            if (intent.IsBuy && o.Quantity <= 0m) return Refuse("side mismatch: signed BUY but order quantity is not positive");
            if (intent.IsSell && o.Quantity >= 0m) return Refuse("side mismatch: signed SELL but order quantity is not negative");
            if (Math.Abs(o.Quantity) != intent.Quantity) return Refuse($"quantity mismatch: order {Math.Abs(o.Quantity)} vs signed {intent.Quantity}");
            if (!o.LimitPrice.HasValue || o.LimitPrice.Value != intent.LimitPrice) return Refuse($"limit price mismatch: order {o.LimitPrice} vs signed {intent.LimitPrice}");

            // ---- long-only: no shorting, never sell more than held (net of other pending sells)
            if (o.HeldQuantity < 0m) return Refuse("a short position exists; long-only engine refuses all orders for this symbol");
            if (intent.IsSell && intent.Quantity + o.PendingSellQuantity > o.HeldQuantity)
            {
                return Refuse($"sell of {intent.Quantity} exceeds held {o.HeldQuantity} minus pending sells {o.PendingSellQuantity} (short sale refused)");
            }

            // ---- replay protection (one intent -> one LEAN order)
            // These refusals deliberately carry NO intent/client order ids in their structured fields: the original
            // order for that intent may be live, and a rejection report keyed by its ids could make the ledger
            // reject it. The ids are only named in the reason text (audit/anomaly), never used to change state.
            lock (_lock)
            {
                if (_journal != null && _journal.ConsumedByEarlierRun(intent.IntentId, intent.ClientOrderId))
                {
                    return GateDecision.Refuse($"replayed intent: {intent.IntentId} / {intent.ClientOrderId} was already consumed by an earlier engine run");
                }
                if (_usedIntents.TryGetValue(intent.IntentId, out var boundOrder) && boundOrder != o.LeanOrderId)
                {
                    return GateDecision.Refuse($"duplicate intent: {intent.IntentId} / {intent.ClientOrderId} already bound to LEAN order {boundOrder}");
                }
                if (_usedClientOrderIds.TryGetValue(intent.ClientOrderId, out var boundOrder2) && boundOrder2 != o.LeanOrderId)
                {
                    return GateDecision.Refuse($"duplicate client_order_id: {intent.ClientOrderId} (intent {intent.IntentId}) already bound to LEAN order {boundOrder2}");
                }
                if (consume && !_usedIntents.ContainsKey(intent.IntentId))
                {
                    // journal first: if the durable write fails the order is refused (fail closed)
                    if (_journal != null && !_journal.TryAppend(intent.IntentId, intent.ClientOrderId, o.LeanOrderId, out var journalError))
                    {
                        return Refuse("cannot record intent consumption durably: " + journalError);
                    }
                    _usedIntents[intent.IntentId] = o.LeanOrderId;
                    _usedClientOrderIds[intent.ClientOrderId] = o.LeanOrderId;
                }
            }
            return GateDecision.Allow(intent);
        }
    }
}
