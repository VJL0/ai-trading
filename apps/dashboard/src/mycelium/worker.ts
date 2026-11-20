import { MyceliumRenderer } from "./renderer";
import type { WorkerInput, WorkerOutput, SceneSettings } from "./model";
const send = (message: WorkerOutput) => postMessage(message);
let renderer: MyceliumRenderer | null = null;
let ready = false;
let disposed = false;
let pendingData: Extract<WorkerInput, { type: "data" }> | null = null;
let settings: SceneSettings | null = null;
let receivedAt = 0;
let frameId = 0;
function tick(now: number) {
  if (disposed) return;
  if (settings) renderer?.setSettings({ ...settings, time: settings.time + (now - receivedAt) / 1000 * settings.rate });
  void renderer?.frame(now);
  frameId = requestAnimationFrame(tick);
}
onmessage = async (message: MessageEvent<WorkerInput>) => {
  const m = message.data;
  try {
    if (m.type === "init") {
      renderer = new MyceliumRenderer(m.canvas, metrics => send({ type: "metrics", metrics }), message => send({ type: "error", message }));
      await renderer.init(m.width, m.height);
      if (disposed) return;
      ready = true;
      if (pendingData) renderer.setData(pendingData.data, pendingData.count);
      pendingData = null;
      send({ type: "ready" });
      frameId = requestAnimationFrame(tick);
    } else if (m.type === "data") {
      if (ready) renderer?.setData(m.data, m.count); else pendingData = m;
    } else if (m.type === "settings") {
      settings = m.settings;
      receivedAt = performance.now();
    } else if (m.type === "resize") renderer?.resize(m.width, m.height);
    else if (m.type === "dispose") { disposed = true; cancelAnimationFrame(frameId); renderer?.destroy(); close(); }
  } catch (e) { renderer?.destroy(); send({ type: "error", message: e instanceof Error ? e.message : "Unable to start WebGPU" }); }
};
