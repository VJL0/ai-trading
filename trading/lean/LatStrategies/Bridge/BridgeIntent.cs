namespace Lat.Lean.Bridge
{
    /// <summary>One claimed intent as delivered by GET /api/v1/engine/intents/next. Untrusted until the gate verifies it.</summary>
    public sealed class BridgeIntent
    {
        public string IntentId { get; init; }
        public string ClientOrderId { get; init; }
        public string SigningString { get; init; }
        public string Signature { get; init; }
        public string KeyId { get; init; }
        public string SignatureAlg { get; init; }
    }
}
