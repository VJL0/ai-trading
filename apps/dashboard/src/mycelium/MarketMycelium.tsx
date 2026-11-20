import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../state/AppContext";
import { bannerModel } from "../lib/mode";
import { serverNow } from "../ops/reducer";
import { colonyPosition, KINDS, memoriesFromOps, mergeMemories, packMemories, type Memory, type Metrics, type WorkerInput, type WorkerOutput } from "./model";
import "./mycelium.css";

export function MarketMycelium() {
  const { ops, stream, status, statusReceivedAt } = useApp();
  const incoming = useMemo(() => memoriesFromOps(ops), [ops.rows, ops.orderEvents, ops.orderCausal, ops.audit]);
  const [memory, setMemory] = useState<Memory[]>([]);
  const [replay, setReplay] = useState<Memory[] | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  const [colony, setColony] = useState<string | null>(null);
  const [camera, setCamera] = useState({ zoom: 1, panX: 0, panY: 0 });
  const [reduced, setReduced] = useState(() => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [animate, setAnimate] = useState(true);
  const [active, setActive] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [gpu, setGpu] = useState("Starting GPU");
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [now, setNow] = useState(Date.now());
  const surface = useRef<HTMLDivElement>(null);
  const worker = useRef<Worker | null>(null);
  const drag = useRef<{ x: number; y: number; panX: number; panY: number; moved: boolean } | null>(null);
  useEffect(() => { setMemory(previous => mergeMemories(previous, incoming)); }, [incoming]);
  useEffect(() => {
    const query = matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => setReduced(query.matches);
    query.addEventListener("change", change);
    return () => query.removeEventListener("change", change);
  }, []);
  useEffect(() => {
    let previous = performance.now();
    const timer = setInterval(() => {
      const t = performance.now();
      setNow(Date.now());
      if (playing && replay) setCursor(c => Math.min(replay.at(-1)?.at ?? c, c + (t - previous) * speed));
      previous = t;
    }, 100);
    return () => clearInterval(timer);
  }, [playing, replay, speed]);
  useEffect(() => { if (replay && cursor >= (replay.at(-1)?.at ?? 0)) setPlaying(false); }, [cursor, replay]);
  useEffect(() => {
    const host = surface.current;
    if (!host) return;
    let visible = true;
    const visibility = () => setActive(visible && !document.hidden);
    const observer = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting ?? false; visibility(); });
    observer.observe(host);
    document.addEventListener("visibilitychange", visibility);
    return () => { observer.disconnect(); document.removeEventListener("visibilitychange", visibility); };
  }, []);
  useEffect(() => {
    const host = surface.current;
    if (!host) return;
    setError(null); setGpu("Starting GPU"); setMetrics(null);
    // A fresh canvas is required after transfer, including React StrictMode and device-loss retry.
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Market Mycelium: recorded trading activity. Use the adjacent event inspector for accessible details.");
    host.prepend(canvas);
    let w: Worker | null = null;
    let resize: ResizeObserver | null = null;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const fail = (message: string) => { setError(message); setGpu("Inspection mode"); w?.terminate(); };
    try {
      if (!canvas.transferControlToOffscreen || !navigator.gpu) throw new Error("WebGPU or offscreen rendering is unavailable. Recorded events and replay remain inspectable below.");
      w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      worker.current = w;
      w.onmessage = (event: MessageEvent<WorkerOutput>) => {
        if (event.data.type === "ready") { clearTimeout(timeout); setGpu("WebGPU online"); }
        else if (event.data.type === "error") { clearTimeout(timeout); fail(event.data.message); }
        else setMetrics(event.data.metrics);
      };
      w.onerror = () => { clearTimeout(timeout); fail("The rendering worker stopped. Retry to rebuild the scene."); };
      const offscreen = canvas.transferControlToOffscreen();
      const size = () => ({ width: Math.round(host.clientWidth * Math.min(devicePixelRatio, 2)), height: Math.round(host.clientHeight * Math.min(devicePixelRatio, 2)) });
      w.postMessage({ type: "init", canvas: offscreen, ...size() } satisfies WorkerInput, [offscreen]);
      resize = new ResizeObserver(() => w?.postMessage({ type: "resize", ...size() } satisfies WorkerInput));
      resize.observe(host);
      timeout = setTimeout(() => fail("GPU initialization timed out. Retry to rebuild the scene."), 15000);
    } catch (e) { fail(e instanceof Error ? e.message : "Unable to initialize WebGPU"); }
    return () => { clearTimeout(timeout); resize?.disconnect(); w?.postMessage({ type: "dispose" } satisfies WorkerInput); const retiring = w; setTimeout(() => retiring?.terminate(), 100); worker.current = null; canvas.remove(); };
  }, [attempt]);

  const scene = replay ?? memory;
  const epoch = scene[0]?.at ?? 0;
  // A newly observed event may lead the last heartbeat (or a paused simulated clock).
  const liveTime = Math.max(serverNow(ops.clock, now), memory.at(-1)?.at ?? 0);
  const time = replay ? cursor : liveTime;
  const visibleEvents = scene.filter(e => e.at <= time && (!colony || e.colony === colony));
  const colonies = [...new Set(scene.filter(e => e.at <= time).map(e => e.colony))];
  // Place readable labels near their anchors; never let adjacent colonies hide each other's controls.
  const labelWidth = 170;
  const labelHeight = 30;
  const width = surface.current?.clientWidth ?? 800;
  const height = surface.current?.clientHeight ?? 680;
  const labels: {name: string; x: number; y: number}[] = [];
  for (const name of colonies.slice(0, 24)) {
    const [x, y] = colonyPosition(name);
    const left = width / 2 + (x * camera.zoom + camera.panX) * height / 2 - labelWidth / 2;
    let top = height / 2 - (y * camera.zoom + camera.panY) * height / 2 + 24;
    for (let step = 0; step < 24 && labels.some(l => Math.abs(l.x - left) < labelWidth && Math.abs(l.y - top) < labelHeight); step++) top += labelHeight;
    labels.push({name, x:left, y:top});
  }
  const event = scene.find(e => e.id === selected && e.at <= time) ?? null;
  const selectedIndex = scene.findIndex(e => e.id === event?.id);
  useEffect(() => {
    const data = packMemories(scene, epoch);
    worker.current?.postMessage({ type: "data", data, count: scene.length } satisfies WorkerInput, [data.buffer]);
  }, [scene, epoch, attempt]);
  useEffect(() => {
    worker.current?.postMessage({ type: "settings", settings: { time: (time - epoch) / 1000, ...camera, selected: selectedIndex, motion: animate && !reduced, active, rate: replay ? playing ? speed : 0 : ops.clock.sim ? ops.clock.sim.rate ?? 0 : 1 } } satisfies WorkerInput);
  }, [time, epoch, camera, selectedIndex, animate, reduced, active, replay, playing, speed, ops.clock.sim, attempt]);
  function beginReplay() {
    setReplay([...memory]); setCursor(memory[0]?.at ?? 0); setPlaying(false); setSelected(null);
  }
  function inspect(e: Memory) { setSelected(e.id); }
  return <section className="mycelium" aria-labelledby="mycelium-title" data-gpu={gpu}>
    <div className="myc-heading">
      <div><div className="myc-eyebrow">Intelligence observatory / 01</div><h2 id="mycelium-title">Market <em>Mycelium</em></h2><p>A living memory of decisions, constraints, and consequence.</p></div>
      <div className="myc-status"><span className={stream.state === "open" ? "myc-dot" : "myc-dot offline"} />{replay ? "HISTORICAL REPLAY" : stream.state === "open" ? "STREAM CONNECTED" : "STREAM OFFLINE"}<small>Current {bannerModel(status, statusReceivedAt, now).label} · {gpu}</small></div>
    </div>
    <div className="myc-body">
      <div className="myc-world">
        <div className="myc-surface" ref={surface}
          onPointerDown={e => { if ((e.target as HTMLElement).tagName !== "CANVAS") return; e.currentTarget.setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY, panX: camera.panX, panY: camera.panY, moved: false }; }}
          onPointerMove={e => { const d = drag.current; if (!d) return; if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) d.moved = true; const h = e.currentTarget.clientHeight; setCamera(c => ({ ...c, panX: d.panX + 2 * (e.clientX - d.x) / h, panY: d.panY - 2 * (e.clientY - d.y) / h })); }}
          onPointerCancel={() => { drag.current = null; }}
          onPointerUp={e => { const d = drag.current; drag.current = null; if (!d || d.moved) return; const box = e.currentTarget.getBoundingClientRect(); let nearest: Memory | null = null; let distance = 42; for (const item of visibleEvents) { const [x, y] = colonyPosition(item.colony); const delta = Math.hypot(e.clientX - box.left - box.width / 2 - (x * camera.zoom + camera.panX) * box.height / 2, e.clientY - box.top - box.height / 2 + (y * camera.zoom + camera.panY) * box.height / 2); if (delta <= distance) { distance = delta; nearest = item; } } if (nearest) inspect(nearest); }}>
          <div className="myc-coordinate">{replay ? "MEMORY / FROZEN WINDOW" : "OBSERVED ACTIVITY / READ ONLY"}<span>{visibleEvents.length} visible records</span></div>
          {!error && labels.map(({name, x, y}) => <button key={name} className={`myc-colony ${colony === name ? "chosen" : ""}`} style={{ left: x, top: y }} onClick={() => setColony(colony === name ? null : name)} title={`Inspect ${name}`}>{name}</button>)}
          {error ? <div className="myc-empty" role="status"><h3>Memory remains available.</h3><p>{error}</p><button onClick={() => setAttempt(a => a + 1)}>Retry WebGPU</button></div> : !visibleEvents.length ? <div className="myc-empty"><h3>Listening for a first signal.</h3><p>The organism grows only from recorded activity.</p></div> : null}
          <div className="myc-view-controls"><button aria-label="Zoom out" onClick={() => setCamera(c => ({ ...c, zoom: Math.max(0.4, c.zoom / 1.2) }))}>−</button><button aria-label="Zoom in" onClick={() => setCamera(c => ({ ...c, zoom: Math.min(3, c.zoom * 1.2) }))}>+</button><button onClick={() => setCamera({ zoom: 1, panX: 0, panY: 0 })}>Recenter</button><button aria-pressed={!animate || reduced} disabled={reduced} onClick={() => setAnimate(a => !a)}>{!animate || reduced ? "Motion paused" : "Pause motion"}</button></div>
          <div className="myc-legend">{KINDS.map(k => <span key={k} className={`myc-kind-${k}`}>{k}</span>)}</div>
        </div>
        <div className="myc-transport">
          <button onClick={() => { if (replay) { setReplay(null); setPlaying(false); } else beginReplay(); }} disabled={!memory.length}>{replay ? "Return to live" : "Explore memory"}</button>
          {replay ? <><button onClick={() => { if (cursor >= (scene.at(-1)?.at ?? 0)) setCursor(epoch); setPlaying(p => !p); }} disabled={scene.length < 2}>{playing ? "Pause replay" : "Play replay"}</button><select aria-label="Replay speed" value={speed} onChange={e => setSpeed(+e.target.value)}><option value={1}>1×</option><option value={10}>10×</option><option value={60}>60×</option></select><input aria-label="Replay time" type="range" min={epoch} max={Math.max(epoch + 1, scene.at(-1)?.at ?? epoch)} value={cursor} onChange={e => { setCursor(+e.target.value); setPlaying(false); }} /></> : <span className="myc-transport-note">Drag to explore · select a colony · inspect a signal</span>}
          <time>{Number.isFinite(time) ? new Date(time).toISOString().replace("T", " ").slice(0, 23) + " UTC" : "—"}</time>
        </div>
      </div>
      <aside className="myc-inspector" aria-label="Mycelium event inspector">
        <div className="myc-inspector-title"><span className="myc-eyebrow">{event ? "Selected signal" : "Memory archive"}</span>{colony && <button onClick={() => setColony(null)}>All colonies</button>}</div>
        {event ? <div className="myc-detail"><button className="myc-back" onClick={() => setSelected(null)}>← Event archive</button><span className={`myc-kind-${event.kind}`}>{event.kind}</span><h3>{event.label}</h3><p>{event.colony}</p><time>{new Date(event.at).toISOString()}</time>{event.confidence !== null ? <p>Reported confidence: {Math.round(event.confidence * 100)}% · {event.calibrated ? "calibrated" : "uncalibrated"}. Fan width = 1 − confidence; this is not a price distribution.</p> : <p>No model confidence attached to this event.</p>}<p>Causal link: <code>{event.causal ?? "not supplied"}</code></p><details open><summary>Source record</summary><pre>{JSON.stringify(event.detail, null, 2)}</pre></details></div> : <><p className="myc-inspector-note">{colony ?? "All observed colonies"} · newest first</p><div className="myc-events">{[...visibleEvents].reverse().map(e => <button key={e.id} onClick={() => inspect(e)} className={`myc-event myc-kind-${e.kind}`}><time>{new Date(e.at).toISOString().slice(11, 23)}</time><strong>{e.label}</strong><small>{e.colony}</small></button>)}{!visibleEvents.length && <p>No events in this window.</p>}</div></>}
      </aside>
    </div>
    <div className="myc-foot"><span>{metrics ? `${metrics.fps} FPS · p95 ${metrics.p95} ms · ${metrics.particles.toLocaleString()} samples · ${metrics.gpuMs === null ? "GPU timing unavailable" : `GPU ${Math.round(metrics.gpuMs * 100) / 100} ms`}` : "Waiting for render measurements"}</span><span title={metrics?.adapter}>{metrics?.resolution ?? "WebGPU compute + light"} · bounded to 512 memories</span></div>
    <details className="myc-key"><summary>How to read this organism</summary><p>Colonies are reported proposers, actors, or service sources—not inferred competing strategies. Shared causal IDs share a root. Teal vessels record market processing, violet fans show reported confidence, amber membranes record risk checks, blue signals record orders, and pale green records outcomes. Coral marks rejection or failure. Pulses decay after observed event timestamps; older vessels remain as memory. No market ticks, profit estimates, or predictive distributions are synthesized. Replay freezes the retained session window while live ingestion continues. Its time scale follows source timestamps, including simulated time. Reloading restores only the history retained by the API. Motion pause freezes flow effects; live data continues to arrive.</p></details>
  </section>;
}
