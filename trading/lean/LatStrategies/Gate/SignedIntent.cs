/*
 * LAT (local-ai-trader) LEAN extensions.
 * Parser for the exact string the Python risk engine signs (packages/contracts/models.py::intent_signing_string).
 *
 *   lat-intent-v1|intent_id=..|client_order_id=..|proposal_id=..|decision_id=..|causal_id=..|mode=..|gateway=..
 *   |symbol=..|side=..|quantity=..|order_type=..|limit_price=..|time_in_force=..|extended_hours=..
 *   |policy_version=..|created_at=..|expires_at=..|requires_human_approval=..|human_approval_id=..|key_id=..
 *   |signature_alg=..
 *
 * The parser is strict: exact prefix, exact field order, exact field count, no duplicates, no extras.
 * Order parameters executed by LEAN are taken FROM this signed string, so what executes is what was signed.
 */

using System;
using System.Globalization;

namespace Lat.Lean.Gate
{
    /// <summary>An order intent parsed from a verified (or to-be-verified) signing string.</summary>
    public sealed class SignedIntent
    {
        public const string Prefix = "lat-intent-v1";

        /// <summary>Must equal packages.contracts.models.INTENT_SIGNING_FIELDS (order matters).</summary>
        public static readonly string[] Fields =
        {
            "intent_id", "client_order_id", "proposal_id", "decision_id", "causal_id", "mode", "gateway",
            "symbol", "side", "quantity", "order_type", "limit_price", "time_in_force", "extended_hours",
            "policy_version", "created_at", "expires_at", "requires_human_approval", "human_approval_id",
            "key_id", "signature_alg",
        };

        public string SigningString { get; private set; }
        public string IntentId { get; private set; }
        public string ClientOrderId { get; private set; }
        public string ProposalId { get; private set; }
        public string DecisionId { get; private set; }
        public string CausalId { get; private set; }
        public string Mode { get; private set; }
        public string Gateway { get; private set; }
        public string Symbol { get; private set; }
        public string Side { get; private set; }
        public decimal Quantity { get; private set; }
        public string OrderType { get; private set; }
        public decimal LimitPrice { get; private set; }
        public string TimeInForce { get; private set; }
        public bool ExtendedHours { get; private set; }
        public string PolicyVersion { get; private set; }
        public DateTime CreatedAtUtc { get; private set; }
        public DateTime ExpiresAtUtc { get; private set; }
        public bool RequiresHumanApproval { get; private set; }
        public string HumanApprovalId { get; private set; }
        public string KeyId { get; private set; }
        public string SignatureAlg { get; private set; }

        public bool IsBuy => Side == "BUY";
        public bool IsSell => Side == "SELL";

        private SignedIntent() { }

        /// <summary>
        /// Strictly parses a signing string. Never throws; returns false with a reason on any deviation.
        /// Parsing does NOT imply authenticity: callers must verify the signature over the same string.
        /// </summary>
        public static bool TryParse(string signingString, out SignedIntent intent, out string error)
        {
            intent = null;
            error = null;
            if (string.IsNullOrEmpty(signingString))
            {
                error = "empty signing string";
                return false;
            }
            if (signingString.IndexOf('\n') >= 0 || signingString.IndexOf('\r') >= 0)
            {
                error = "signing string contains a newline";
                return false;
            }
            var parts = signingString.Split('|');
            if (parts.Length != Fields.Length + 1)
            {
                error = $"expected {Fields.Length + 1} '|' separated parts, got {parts.Length}";
                return false;
            }
            if (parts[0] != Prefix)
            {
                error = $"unsupported intent version '{parts[0]}'";
                return false;
            }

            var values = new string[Fields.Length];
            for (var i = 0; i < Fields.Length; i++)
            {
                var part = parts[i + 1];
                var eq = part.IndexOf('=');
                if (eq <= 0)
                {
                    error = $"malformed field at position {i + 1}";
                    return false;
                }
                var name = part.Substring(0, eq);
                if (name != Fields[i])
                {
                    error = $"field {i + 1} must be '{Fields[i]}' but is '{name}'";
                    return false;
                }
                values[i] = part.Substring(eq + 1);
            }

            var r = new SignedIntent { SigningString = signingString };
            r.IntentId = values[0];
            r.ClientOrderId = values[1];
            r.ProposalId = values[2];
            r.DecisionId = values[3];
            r.CausalId = values[4];
            r.Mode = values[5];
            r.Gateway = values[6];
            r.Symbol = values[7];
            r.Side = values[8];
            r.OrderType = values[10];
            r.TimeInForce = values[12];
            r.PolicyVersion = values[14];
            r.HumanApprovalId = values[18];
            r.KeyId = values[19];
            r.SignatureAlg = values[20];

            if (string.IsNullOrEmpty(r.IntentId) || string.IsNullOrEmpty(r.ClientOrderId))
            {
                error = "intent_id and client_order_id are required";
                return false;
            }
            if (r.ClientOrderId.Length > 128)
            {
                error = "client_order_id longer than 128 characters";
                return false;
            }
            if (string.IsNullOrEmpty(r.Symbol) || string.IsNullOrEmpty(r.PolicyVersion) || string.IsNullOrEmpty(r.KeyId))
            {
                error = "symbol, policy_version and key_id are required";
                return false;
            }
            if (r.Side != "BUY" && r.Side != "SELL")
            {
                error = $"unsupported side '{r.Side}'";
                return false;
            }
            if (!TryParseDecimal(values[9], out var qty))
            {
                error = $"quantity '{values[9]}' is not a decimal";
                return false;
            }
            if (!TryParseDecimal(values[11], out var px))
            {
                error = $"limit_price '{values[11]}' is not a decimal";
                return false;
            }
            r.Quantity = qty;
            r.LimitPrice = px;
            if (!TryParseBool(values[13], out var ext) || !TryParseBool(values[17], out var human))
            {
                error = "extended_hours and requires_human_approval must be 'true' or 'false'";
                return false;
            }
            r.ExtendedHours = ext;
            r.RequiresHumanApproval = human;
            if (!TryParseUtc(values[15], out var created) || !TryParseUtc(values[16], out var expires))
            {
                error = "created_at/expires_at must be ISO-8601 UTC timestamps ending in 'Z'";
                return false;
            }
            r.CreatedAtUtc = created;
            r.ExpiresAtUtc = expires;
            intent = r;
            return true;
        }

        /// <summary>Plain fixed-point decimal ("10", "450.25"); no exponent, sign, thousands separators or whitespace.</summary>
        public static bool TryParseDecimal(string text, out decimal value)
        {
            value = 0m;
            if (string.IsNullOrEmpty(text) || text.Length > 40)
            {
                return false;
            }
            foreach (var c in text)
            {
                if (!(c == '.' || (c >= '0' && c <= '9')))
                {
                    return false;
                }
            }
            return decimal.TryParse(text, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryParseBool(string text, out bool value)
        {
            value = text == "true";
            return text == "true" || text == "false";
        }

        /// <summary>Accepts the Python contract serialisation: ISO-8601 with a trailing 'Z' (UTC only).</summary>
        public static bool TryParseUtc(string text, out DateTime value)
        {
            value = default;
            if (string.IsNullOrEmpty(text) || !text.EndsWith("Z", StringComparison.Ordinal))
            {
                return false;
            }
            if (!DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dto))
            {
                return false;
            }
            value = dto.UtcDateTime;
            return true;
        }
    }
}
