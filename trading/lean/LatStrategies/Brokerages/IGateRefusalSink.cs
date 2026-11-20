using Lat.Lean.Gate;

namespace Lat.Lean.Brokerages
{
    /// <summary>Receives every order refusal of a LAT gate (reported to the local API /engine/rejections).</summary>
    public interface IGateRefusalSink
    {
        /// <param name="layer">"GatedBrokerageModel" (algorithm hook) or the brokerage class (final engine boundary).</param>
        void Refused(string layer, int leanOrderId, GateDecision decision, string orderSummary);
    }
}
