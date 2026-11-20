using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Lat.Lean.Gate;

namespace Lat.Lean.Tests
{
    public sealed class VectorCase
    {
        public string Name;
        public string SigningString;
        public string Signature;
        public string IntentId;
        public string ClientOrderId;
        public string Symbol;
        public string Side;
        public decimal Quantity;
        public decimal LimitPrice;
        public bool ExpectSignatureValid;
        public string ExpectGate;
    }

    /// <summary>Loads vectors/intent_vectors.json produced by trading/lean/tools/gen_intent_vectors.py (Python signer).</summary>
    public static class Vectors
    {
        private static readonly Lazy<JsonDocument> Doc = new Lazy<JsonDocument>(() =>
            JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "vectors", "intent_vectors.json"))));

        public static string PublicKeyPem => Doc.Value.RootElement.GetProperty("public_key_pem").GetString();
        public static string KeyId => Doc.Value.RootElement.GetProperty("key_id").GetString();
        public static string OtherPublicKeyPem => Doc.Value.RootElement.GetProperty("other_public_key_pem").GetString();
        public static string OtherKeyId => Doc.Value.RootElement.GetProperty("other_key_id").GetString();
        public static DateTime NowUtc => DateTimeOffset.Parse(Doc.Value.RootElement.GetProperty("now_utc").GetString()).UtcDateTime;

        public static IReadOnlyList<VectorCase> Cases => Doc.Value.RootElement.GetProperty("cases").EnumerateArray().Select(c => new VectorCase
        {
            Name = c.GetProperty("name").GetString(),
            SigningString = c.GetProperty("signing_string").GetString(),
            Signature = c.GetProperty("signature").GetString(),
            IntentId = c.GetProperty("intent_id").GetString(),
            ClientOrderId = c.GetProperty("client_order_id").GetString(),
            Symbol = c.GetProperty("symbol").GetString(),
            Side = c.GetProperty("side").GetString(),
            Quantity = decimal.Parse(c.GetProperty("quantity").GetString(), System.Globalization.CultureInfo.InvariantCulture),
            LimitPrice = decimal.Parse(c.GetProperty("limit_price").GetString(), System.Globalization.CultureInfo.InvariantCulture),
            ExpectSignatureValid = c.GetProperty("expect_signature_valid").GetBoolean(),
            ExpectGate = c.GetProperty("expect_gate").GetString(),
        }).ToList();

        public static VectorCase Case(string name) => Cases.Single(c => c.Name == name);

        public static string Token(VectorCase c) => IntentToken.Encode(c.SigningString, c.Signature);

        public static IntentGate Gate(LatGateway gw = LatGateway.LEAN_PAPER_SIM, DateTime? now = null, UsedIntentJournal journal = null) =>
            new IntentGate(IntentSignatureVerifier.FromPem(PublicKeyPem), gw, () => now ?? NowUtc, journal);

        /// <summary>Facts of a LEAN limit order exactly matching the vector case.</summary>
        public static OrderFacts Facts(VectorCase c, int leanOrderId = 1, decimal held = 0m) => new OrderFacts
        {
            LeanOrderId = leanOrderId,
            Tag = Token(c),
            Symbol = c.Symbol,
            SecurityType = "Equity",
            Market = "usa",
            OrderType = "Limit",
            Quantity = c.Side == "BUY" ? c.Quantity : -c.Quantity,
            LimitPrice = c.LimitPrice,
            TimeInForce = "DAY",
            ExchangeOpen = true,
            HeldQuantity = held,
        };
    }
}

namespace Lat.Lean.Tests
{
    /// <summary>Order provider with no open orders (the gate treats a missing provider as "unknown" and refuses sells).</summary>
    public sealed class EmptyOrderProvider : QuantConnect.Securities.IOrderProvider
    {
        public int OrdersCount => 0;
        public QuantConnect.Orders.Order GetOrderById(int orderId) => null;
        public System.Collections.Generic.List<QuantConnect.Orders.Order> GetOrdersByBrokerageId(string brokerageId) => new();
        public System.Collections.Generic.IEnumerable<QuantConnect.Orders.OrderTicket> GetOrderTickets(System.Func<QuantConnect.Orders.OrderTicket, bool> filter = null) => System.Array.Empty<QuantConnect.Orders.OrderTicket>();
        public System.Collections.Generic.IEnumerable<QuantConnect.Orders.OrderTicket> GetOpenOrderTickets(System.Func<QuantConnect.Orders.OrderTicket, bool> filter = null) => System.Array.Empty<QuantConnect.Orders.OrderTicket>();
        public QuantConnect.Orders.OrderTicket GetOrderTicket(int orderId) => null;
        public System.Collections.Generic.IEnumerable<QuantConnect.Orders.Order> GetOrders(System.Func<QuantConnect.Orders.Order, bool> filter = null) => System.Array.Empty<QuantConnect.Orders.Order>();
        public System.Collections.Generic.List<QuantConnect.Orders.Order> GetOpenOrders(System.Func<QuantConnect.Orders.Order, bool> filter = null) => new();
        public QuantConnect.Securities.ProjectedHoldings GetProjectedHoldings(QuantConnect.Securities.Security security) => new(security.Holdings.Quantity, 0m);
    }
}
