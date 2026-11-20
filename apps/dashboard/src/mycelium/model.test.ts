import { describe, expect, it } from "vitest";
import { initialLiveOpsState, liveOpsReducer } from "../ops/reducer";
import { colonyPosition, MAX_MEMORIES, memoriesFromOps, mergeMemories, packMemories, type Memory } from "./model";
const memory = (id: string, at: number): Memory => ({ id, at, kind: "market", colony: "model-a", causal: null, label: id, detail: {}, confidence: null, calibrated: false, rejected: false });
describe("Mycelium evidence boundary", () => {
  it("never creates activity from an empty system", () => expect(memoriesFromOps(initialLiveOpsState())).toEqual([]));
  it("deduplicates corrections, sorts out-of-order events, and bounds storage", () => {
    const old = Array.from({length: MAX_MEMORIES + 20}, (_, i) => memory(String(i), i));
    const result = mergeMemories(old, [{...memory("520", 520), label:"corrected"}, memory("future", 1000)]);
    expect(result).toHaveLength(MAX_MEMORIES);
    expect(result.at(-1)?.id).toBe("future");
    expect(result.find(e => e.id === "520")?.label).toBe("corrected");
    expect(result[0]?.at).toBe(21);
  });
  it("preserves unknown and uncalibrated confidence without synthesizing predictions", () => {
    let ops = initialLiveOpsState();
    for (const [seq, confidence] of [[1, null], [2, "0.72"], [3, "NaN"], [4, "1.1"]] as const) ops = liveOpsReducer(ops, {type:"stream", name:"decision", at:1000, data:{seq, ts:`2026-11-18T12:00:0${seq}Z`, causal_id:"cycle", action:"BUY", status:"REJECTED", confidence, confidence_calibrated:false, proposer:"strategy-a", risk_failures:["spread"]}});
    const events = memoriesFromOps(ops);
    expect(events.map(e => e.confidence)).toEqual([null, null, 0.72, null]);
    expect(events.every(e => e.rejected && !e.calibrated && e.colony === "strategy-a")).toBe(true);
    const packed = packMemories(events, events[0]!.at);
    expect(packed[5]).toBe(-1);
    expect(packed[6]).toBe(1);
    expect(packed[12 * 2 + 5]).toBeCloseTo(0.72);
  });
  it("drops invalid timestamps and retains order application truth", () => {
    const ops = initialLiveOpsState();
    ops.orderEvents = [{eventId:"a", seq:1, clientOrderId:"o", eventType:"FILL", state:"FILLED", source:"broker", ts:"2026-11-18T12:00:00Z", applied:false, note:"duplicate"}, {eventId:"b", seq:2, clientOrderId:"o", eventType:"FILL", state:"FILLED", source:"broker", ts:null, applied:true, note:null}];
    const result = memoriesFromOps(ops);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({kind:"outcome", rejected:true});
    expect(result[0]?.label).toContain("not applied");
  });
  it("keeps replay snapshots independent and positions stable through eviction", () => {
    const replay = [memory("a", 1000)];
    const live = mergeMemories(replay, [memory("b", 2000)]);
    expect(replay).toHaveLength(1);
    expect(live).toHaveLength(2);
    expect(colonyPosition("model-a")).toEqual(colonyPosition("model-a"));
    expect(packMemories(replay, 1000)[2]).toBe(0);
  });
});
