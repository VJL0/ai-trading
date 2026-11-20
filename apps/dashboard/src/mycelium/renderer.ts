/// <reference types="@webgpu/types" />
import { MAX_MEMORIES, SAMPLES, type Metrics, type SceneSettings } from "./model";
import { shader, renderShader, toneShader } from "./shaders";

/** Owns every GPU resource. No networking, trading clients, or mutable application state. */
export class MyceliumRenderer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private uniform!: GPUBuffer;
  private events!: GPUBuffer;
  private particles!: GPUBuffer;
  private compute!: GPUComputePipeline;
  private render!: GPURenderPipeline;
  private vessel!: GPURenderPipeline;
  private background!: GPURenderPipeline;
  private computeGroup!: GPUBindGroup;
  private renderGroup!: GPUBindGroup;
  private vesselGroup!: GPUBindGroup;
  private backgroundGroup!: GPUBindGroup;
  private light: GPUTexture | null = null;
  private tone!: GPURenderPipeline;
  private toneGroup!: GPUBindGroup;
  private query: GPUQuerySet | null = null;
  private resolve: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private values = new Float32Array(16);
  private count = 0;
  private destroyed = false;
  private failed = false;
  private settings: SceneSettings = { time: 0, zoom: 1, panX: 0, panY: 0, selected: -1, motion: true, active: true, rate: 1 };
  private intervals: number[] = [];
  private lastFrame = 0;
  private lastReport = 0;
  private gpuMs: number | null = null;
  private adapterName = "WebGPU";
  private framePending = false;

  constructor(private canvas: OffscreenCanvas, private report: (metrics: Metrics) => void, private error: (message: string) => void) {}

  async init(width: number, height: number) {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser. Event inspection remains available.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available. Event inspection remains available.");
    const timestamp = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({ requiredFeatures: timestamp ? ["timestamp-query"] : [] });
    this.device = device;
    if (this.destroyed) { device.destroy(); return; }
    this.adapterName = [adapter.info.vendor, adapter.info.architecture, adapter.info.device, adapter.info.description].filter(Boolean).join(" / ") || "WebGPU adapter";
    device.lost.then(info => { if (!this.destroyed) this.fail(`GPU device lost: ${info.message || info.reason}. Retry to rebuild the scene.`); });
    device.addEventListener("uncapturederror", event => this.fail(`GPU rendering stopped: ${event.error.message}`));
    const context = this.canvas.getContext("webgpu");
    if (!context) throw new Error("Unable to initialize the WebGPU canvas.");
    this.context = context;
    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "opaque" });
    this.resize(width, height);
    this.uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.events = device.createBuffer({ size: MAX_MEMORIES * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.particles = device.createBuffer({ size: MAX_MEMORIES * SAMPLES * 32, usage: GPUBufferUsage.STORAGE });
    const cs = device.createShaderModule({ code: shader, label: "Mycelium growth" });
    const rs = device.createShaderModule({ code: renderShader, label: "Mycelium light" });
    const diagnostics = await Promise.all([cs.getCompilationInfo(), rs.getCompilationInfo()]);
    const errors = diagnostics.flatMap(d => d.messages).filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("; "));
    if (this.destroyed) return;
    const format: GPUTextureFormat = "rgba16float";
    [this.compute, this.render, this.background, this.vessel] = await Promise.all([
      device.createComputePipelineAsync({ layout: "auto", compute: { module: cs, entryPoint: "grow" } }),
      device.createRenderPipelineAsync({ layout: "auto", vertex: { module: rs, entryPoint: "vs" }, fragment: { module: rs, entryPoint: "fs", targets: [{ format, blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" } } }] }, primitive: { topology: "triangle-list" } }),
      device.createRenderPipelineAsync({ layout: "auto", vertex: { module: rs, entryPoint: "backgroundVs" }, fragment: { module: rs, entryPoint: "backgroundFs", targets: [{ format }] } }),
      device.createRenderPipelineAsync({ layout: "auto", vertex: { module: rs, entryPoint: "vesselVs" }, fragment: { module: rs, entryPoint: "vesselFs", targets: [{ format, blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" } } }] } }),
    ]);
    if (this.destroyed) return;
    this.tone = await device.createRenderPipelineAsync({ layout: "auto", vertex: { module: device.createShaderModule({ code: toneShader }), entryPoint: "vs" }, fragment: { module: device.createShaderModule({ code: toneShader }), entryPoint: "fs", targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }] } });
    if (this.destroyed) return;
    this.resize(width, height);
    this.computeGroup = device.createBindGroup({ layout: this.compute.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: this.events } }, { binding: 2, resource: { buffer: this.particles } }] });
    this.renderGroup = device.createBindGroup({ layout: this.render.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: this.particles } }] });
    this.vesselGroup = device.createBindGroup({ layout: this.vessel.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: this.particles } }] });
    this.backgroundGroup = device.createBindGroup({ layout: this.background.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
    if (timestamp) {
      this.query = device.createQuerySet({ type: "timestamp", count: 2 });
      this.resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
  }
  private fail(message: string) { if (!this.failed && !this.destroyed) { this.failed = true; this.error(message); } }
  resize(width: number, height: number) {
    // Cap raster cost at ~3 MP; keep aspect ratio and never allocate a zero-size texture.
    const scale = Math.min(1, Math.sqrt(3_000_000 / Math.max(1, width * height)), (this.device?.limits.maxTextureDimension2D ?? 4096) / Math.max(width, height, 1));
    this.canvas.width = Math.max(1, Math.round(width * scale));
    this.canvas.height = Math.max(1, Math.round(height * scale));
    if (this.tone) {
      this.light?.destroy();
      this.light = this.device.createTexture({ size: [this.canvas.width, this.canvas.height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.toneGroup = this.device.createBindGroup({ layout: this.tone.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.light.createView() }] });
    }
  }
  setData(data: Float32Array<ArrayBuffer>, count: number) {
    if (this.destroyed || this.failed) return;
    this.count = Math.min(MAX_MEMORIES, count, Math.floor(data.length / 12));
    if (this.count) this.device.queue.writeBuffer(this.events, 0, data.buffer, data.byteOffset, this.count * 48);
  }
  setSettings(settings: SceneSettings) {
    if (settings.active !== this.settings.active) { this.lastFrame = 0; this.intervals = []; this.lastReport = performance.now(); }
    this.settings = settings;
  }
  async frame(now: number) {
    if (this.destroyed || this.failed || this.framePending || !this.settings.active) return;
    this.framePending = true;
    try {
      if (this.lastFrame) this.intervals.push(now - this.lastFrame);
      this.lastFrame = now;
      const s = this.settings;
      this.values.set([this.canvas.width, this.canvas.height, 0, 0, s.time, 0, this.count, s.selected, s.zoom, s.panX, s.panY, 0, s.motion ? 1 : 0, 0, 0, 0]);
      this.device.queue.writeBuffer(this.uniform, 0, this.values);
      const encoder = this.device.createCommandEncoder();
      const sample = now - this.lastReport >= 1000;
      const compute = encoder.beginComputePass(this.query && sample ? { timestampWrites: { querySet: this.query, beginningOfPassWriteIndex: 0 } } : {});
      compute.setPipeline(this.compute);
      compute.setBindGroup(0, this.computeGroup);
      if (this.count) compute.dispatchWorkgroups(Math.ceil(this.count * SAMPLES / 64));
      compute.end();
      const render = encoder.beginRenderPass({ colorAttachments: [{ view: this.light!.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      render.setPipeline(this.background);
      render.setBindGroup(0, this.backgroundGroup);
      render.draw(3);
      render.setPipeline(this.vessel);
      render.setBindGroup(0, this.vesselGroup);
      render.draw(6, this.count * SAMPLES);
      render.setPipeline(this.render);
      render.setBindGroup(0, this.renderGroup);
      render.draw(6, this.count * SAMPLES);
      render.end();
      const tone = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }], ...(this.query && sample ? { timestampWrites: { querySet: this.query, endOfPassWriteIndex: 1 } } : {}) });
      tone.setPipeline(this.tone);
      tone.setBindGroup(0, this.toneGroup);
      tone.draw(3);
      tone.end();
      if (sample && this.query && this.resolve && this.readback) {
        encoder.resolveQuerySet(this.query, 0, 2, this.resolve, 0);
        encoder.copyBufferToBuffer(this.resolve, 0, this.readback, 0, 16);
      }
      this.device.queue.submit([encoder.finish()]);
      // Bound outstanding submissions to one. A slow GPU cannot accumulate a command backlog.
      await this.device.queue.onSubmittedWorkDone();
      if (sample && this.readback && !this.destroyed) {
        await this.readback.mapAsync(GPUMapMode.READ);
        const stamps = new BigUint64Array(this.readback.getMappedRange());
        this.gpuMs = Number(stamps[1]! - stamps[0]!) / 1e6;
        this.readback.unmap();
      }
      if (sample && !this.destroyed) {
        const sorted = [...this.intervals].sort((a, b) => a - b);
        const total = this.intervals.reduce((a, b) => a + b, 0);
        this.report({ fps: total ? Math.round(this.intervals.length * 1000 / total) : 0, p95: Math.round((sorted[Math.floor(sorted.length * 0.95)] ?? 0) * 10) / 10, gpuMs: this.gpuMs, particles: this.count * SAMPLES, resolution: `${this.canvas.width} × ${this.canvas.height}`, adapter: this.adapterName });
        this.intervals = [];
        this.lastReport = now;
      }
    } catch (e) { this.fail(e instanceof Error ? e.message : "WebGPU rendering failed"); }
    finally { this.framePending = false; }
  }
  destroy() {
    this.destroyed = true;
    this.context?.unconfigure();
    this.light?.destroy();
    this.uniform?.destroy(); this.events?.destroy(); this.particles?.destroy();
    this.query?.destroy(); this.resolve?.destroy(); this.readback?.destroy(); this.device?.destroy();
  }
}
