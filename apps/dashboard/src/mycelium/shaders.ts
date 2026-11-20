/** One 48-byte record per observed event; 256 procedural samples per record. No CPU particle loop. */
export const shader = /* wgsl */ `
struct Params { viewport: vec4f, clock: vec4f, camera: vec4f, flags: vec4f }
struct Event { site: vec4f, evidence: vec4f, origin: vec4f }
struct Particle { position: vec4f, color: vec4f }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> events: array<Event>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
fn palette(kind: f32) -> vec3f {
  if kind < 0.5 { return vec3f(0.16, 0.88, 0.72); }
  if kind < 1.5 { return vec3f(0.66, 0.43, 1.0); }
  if kind < 2.5 { return vec3f(1.0, 0.66, 0.22); }
  if kind < 3.5 { return vec3f(0.22, 0.62, 1.0); }
  if kind < 4.5 { return vec3f(0.72, 1.0, 0.85); }
  return vec3f(1.0, 0.23, 0.27);
}
@compute @workgroup_size(64)
fn grow(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if id >= u32(p.clock.z) * 256u { return; }
  let eventIndex = id / 256u;
  let e = events[eventIndex];
  let sampleIndex = id % 256u;
  let branch = f32(sampleIndex / 64u);
  let t = f32(sampleIndex % 64u) / 63.0;
  let age = p.clock.x - e.site.z;
  let seed = e.evidence.x * 6.283185;
  let direction = e.site.xy - e.origin.xy;
  let normal = normalize(vec2f(-direction.y, direction.x) + vec2f(0.0001));
  let excitation = exp(-max(age, 0.0) * 0.23) * p.flags.x;
  let flow = p.clock.x * 1.7 * p.flags.x;
  // Curl-like displacement follows an observed event's vascular path; it is not a price forecast.
  let curl = sin(t * 13.0 + seed + flow * excitation * 0.18) * sin(t * 3.14159);
  let leaf = vec2f(cos(seed + branch * 1.7), sin(seed + branch * 1.7)) * (0.05 + e.evidence.x * 0.12);
  let vein = (branch - 1.5) * 0.008 + curl * (0.035 + 0.03 * sin(seed));
  let crown = smoothstep(0.35, 1.0, t);
  var pos = mix(e.origin.xy, e.site.xy, t) + normal * vein + leaf * crown;
  // Each record grows a distinct dendrite; repeated activity thickens the living colony.
  pos += normal * sin(t * 24.0 + seed) * sin(t * 3.14159) * crown * 0.018;
  // Prediction fans express reported 1-confidence, never a fabricated probability distribution.
  if e.site.w == 1.0 && e.evidence.y >= 0.0 {
    let spread = (1.0 - e.evidence.y) * 0.32;
    pos += normal * (branch - 1.5) * spread * pow(t, 2.5);
    pos += normalize(direction) * t * t * 0.12;
  }
  // Risk observations weave a membrane around their colony; rejection turns the membrane coral.
  if e.site.w == 2.0 {
    let a = t * 6.283185;
    let radius = 0.11 + branch * 0.009 + e.evidence.x * 0.025 + sin(a * 7.0 + seed) * 0.005;
    pos = e.site.xy + vec2f(cos(a), sin(a)) * radius;
  }
  let pulse = exp(-pow((t - fract(max(age, 0.0) * 0.30)) * 16.0, 2.0)) * excitation;
  let selected = select(0.0, 1.0, i32(eventIndex) == i32(p.clock.w));
  let visible = select(0.0, 1.0, age >= 0.0);
  var color = palette(e.site.w);
  if e.evidence.z > 0.5 { color = vec3f(1.0, 0.27, 0.25); }
  let strength = (0.20 + 0.10 * exp(-max(age, 0.0) / 300.0) + pulse * 1.8 + selected * 0.7) * visible * min(1.0, sqrt(80.0 / max(p.clock.z, 1.0)));
  particles[id].position = vec4f(pos, 1.3 + pulse * 2.5 + selected, 0.0);
  particles[id].color = vec4f(color, strength);
}
`;
export const renderShader = /* wgsl */ `
struct Params { viewport: vec4f, clock: vec4f, camera: vec4f, flags: vec4f }
struct Particle { position: vec4f, color: vec4f }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f, @location(1) color: vec4f }
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) instance: u32) -> Vertex {
  let corners = array<vec2f, 6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
  let point = particles[instance];
  let uv = corners[v];
  let aspect = p.viewport.x / p.viewport.y;
  let center = (point.position.xy * p.camera.x + p.camera.yz) * vec2f(1.0 / aspect, 1.0);
  let size = point.position.z * 3.5;
  var o: Vertex;
  o.position = vec4f(center + uv * size / p.viewport.xy * 2.0, 0, 1);
  o.uv = uv;
  o.color = point.color;
  return o;
}
@vertex fn vesselVs(@builtin(vertex_index) v: u32, @builtin(instance_index) instance: u32) -> Vertex {
  let point = particles[instance];
  let next = particles[select(instance + 1u, instance, instance % 64u == 63u)];
  let corners = array<vec2f, 6>(vec2f(0,-1),vec2f(1,-1),vec2f(0,1),vec2f(0,1),vec2f(1,-1),vec2f(1,1));
  let corner = corners[v];
  let tangent = normalize(next.position.xy - point.position.xy + vec2f(0.000001));
  let normal = vec2f(-tangent.y, tangent.x);
  let center = mix(point.position.xy, next.position.xy, corner.x) * p.camera.x + p.camera.yz;
  let width = (0.8 + point.color.a * 0.3) / p.viewport.y * 2.0;
  var o: Vertex;
  o.position = vec4f((center + normal * corner.y * width) * vec2f(p.viewport.y / p.viewport.x, 1), 0, 1);
  o.uv = vec2f(0, corner.y);
  o.color = point.color;
  return o;
}
@fragment fn vesselFs(v: Vertex) -> @location(0) vec4f {
  return vec4f(v.color.rgb * v.color.a * exp(-v.uv.y * v.uv.y * 3.0) * 0.85, 1);
}
@fragment fn fs(v: Vertex) -> @location(0) vec4f {
  let d = dot(v.uv, v.uv);
  let glow = exp(-d * 5.0) * 0.23 + exp(-d * 35.0) * 0.8;
  return vec4f(v.color.rgb * v.color.a * glow, 1.0);
}
@vertex fn backgroundVs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
  return vec4f(f32((id << 1u) & 2u) * 2.0 - 1.0, f32(id & 2u) * 2.0 - 1.0, 0, 1);
}
@fragment fn backgroundFs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let uv = (position.xy / p.viewport.xy - 0.5) * vec2f(p.viewport.x / p.viewport.y, 1.0);
  let halo = exp(-dot(uv, uv) * 6.0);
  let grain = fract(sin(dot(position.xy, vec2f(12.9898, 78.233))) * 43758.5453) * 0.008;
  return vec4f(vec3f(0.008, 0.018, 0.027) + halo * vec3f(0.009, 0.029, 0.035) + grain, 1);
}
`;

export const toneShader = /* wgsl */ `
@group(0) @binding(0) var light: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
  return vec4f(f32((id << 1u) & 2u) * 2.0 - 1.0, f32(id & 2u) * 2.0 - 1.0, 0, 1);
}
@fragment fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let radiance = textureLoad(light, vec2i(position.xy), 0).rgb;
  // Preserve chromatic identity when many vessels converge instead of clipping to white.
  let peak = max(radiance.r, max(radiance.g, radiance.b));
  return vec4f(radiance / (1.0 + peak), 1);
}
`;
