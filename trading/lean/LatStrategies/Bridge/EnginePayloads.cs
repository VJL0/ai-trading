/*
 * JSON payloads of the engine bridge contract. Decimals are written as strings (the Python side rejects
 * binary floats), integers as JSON integers, timestamps as ISO-8601 UTC "yyyy-MM-ddTHH:mm:ss.ffffffZ".
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using Lat.Lean.Gate;

namespace Lat.Lean.Bridge
{
    public static class EnginePayloads
    {
        public sealed class OrderEventPayload
        {
            public string ClientOrderId { get; init; }
            public string IntentId { get; init; }
            public int LeanOrderId { get; init; }
            public int LeanEventId { get; init; }
            public string BrokerOrderId { get; init; }
            public string Status { get; init; }
            public decimal? FillQuantity { get; init; }
            public decimal? FillPrice { get; init; }
            public decimal? CumFilledQuantity { get; init; }
            public DateTime EventUtc { get; init; }
            public string Message { get; init; }
        }

        public static string Iso(DateTime utc)
        {
            var u = utc.Kind == DateTimeKind.Local ? utc.ToUniversalTime() : DateTime.SpecifyKind(utc, DateTimeKind.Utc);
            return u.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'", CultureInfo.InvariantCulture);
        }

        public static string Dec(decimal d) => d.ToString(CultureInfo.InvariantCulture);

        private static string Write(Action<Utf8JsonWriter> body)
        {
            using var ms = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms))
            {
                w.WriteStartObject();
                body(w);
                w.WriteEndObject();
            }
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        private static void StrOrNull(Utf8JsonWriter w, string name, string value)
        {
            if (value == null) w.WriteNull(name);
            else w.WriteString(name, value);
        }

        private static void DecOrNull(Utf8JsonWriter w, string name, decimal? value)
        {
            if (value.HasValue) w.WriteString(name, Dec(value.Value));
            else w.WriteNull(name);
        }

        public static string ToJson(string engineId, OrderEventPayload e) => Write(w =>
        {
            w.WriteString("engine_id", engineId);
            w.WriteString("client_order_id", e.ClientOrderId);
            w.WriteString("intent_id", e.IntentId);
            w.WriteNumber("lean_order_id", e.LeanOrderId);
            StrOrNull(w, "broker_order_id", e.BrokerOrderId);
            w.WriteString("status", e.Status);
            DecOrNull(w, "fill_quantity", e.FillQuantity);
            DecOrNull(w, "fill_price", e.FillPrice);
            DecOrNull(w, "cum_filled_quantity", e.CumFilledQuantity);
            w.WriteString("event_ts", Iso(e.EventUtc));
            w.WriteNumber("lean_event_id", e.LeanEventId);
            w.WriteString("message", Truncate(e.Message ?? "", 500));
        });

        public static string RejectionJson(string engineId, string clientOrderId, string intentId, string reason) => Write(w =>
        {
            w.WriteString("engine_id", engineId);
            StrOrNull(w, "client_order_id", Truncate(clientOrderId, 128));
            StrOrNull(w, "intent_id", Truncate(intentId, 128));
            w.WriteString("reason", Truncate(string.IsNullOrWhiteSpace(reason) ? "refused" : reason, 2000));
        });

        public static string HeartbeatJson(BridgeSettings s, string algorithm, string status, DateTime utc) => Write(w =>
        {
            w.WriteString("engine_id", s.EngineId);
            w.WriteString("mode", s.Mode);
            w.WriteString("gateway", s.Gateway.ToString());
            w.WriteString("algorithm", algorithm ?? "");
            w.WriteString("lean_commit", s.LeanCommit ?? "");
            w.WriteString("ts", Iso(utc));
            w.WriteString("status", status ?? "UNKNOWN");
        });

        /// <summary>Parses {"intents":[{intent_id, client_order_id, signing_string, signature, key_id, signature_alg}]}.</summary>
        public static List<BridgeIntent> ParseIntents(string json)
        {
            var list = new List<BridgeIntent>();
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Object ||
                !doc.RootElement.TryGetProperty("intents", out var arr) || arr.ValueKind != JsonValueKind.Array)
            {
                throw new FormatException("intents response must be an object with an 'intents' array");
            }
            foreach (var el in arr.EnumerateArray())
            {
                if (el.ValueKind != JsonValueKind.Object) throw new FormatException("intent entry must be an object");
                list.Add(new BridgeIntent
                {
                    IntentId = Str(el, "intent_id"),
                    ClientOrderId = Str(el, "client_order_id"),
                    SigningString = Str(el, "signing_string"),
                    Signature = Str(el, "signature"),
                    KeyId = Str(el, "key_id"),
                    SignatureAlg = Str(el, "signature_alg"),
                });
            }
            return list;
        }

        private static string Str(JsonElement el, string name) =>
            el.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

        private static string Truncate(string s, int max) => s == null ? null : (s.Length <= max ? s : s.Substring(0, max));

        /// <summary>Token for an order tag built from a delivered intent (throws on a malformed signature).</summary>
        public static string TagFor(BridgeIntent intent) => IntentToken.Encode(intent.SigningString, intent.Signature);
    }
}
