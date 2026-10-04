/**
 * Editor overlay rendering (LDR, drawn on the swapchain after post-processing).
 *
 * Three kinds of primitives, all depth-aware by SAMPLING the scene depth (no depth
 * attachment, so it works whatever the target size / sample count):
 *  - highlight: screen-space outline + soft tint of the REAL rendered obstacle. The pixels of
 *    an inflated proxy mesh reconstruct the world position from the opaque depth and test it
 *    against the obstacle proxy (with a normal-agreement heuristic so the surrounding ground
 *    is excluded); the outline is the anti-aliased edge of that mask. Parts hidden behind
 *    water / particles are dimmed.
 *  - ghost: a translucent procedural proxy mesh (noisy ellipsoid / slab / capsule) with a
 *    fresnel rim, a bright contact line where it meets the scene, hatched where buried.
 *  - lines: thin anti-aliased screen-space lines with soft glow (footprint rings draped on
 *    the terrain, gizmo hints); hidden parts are drawn faint and dotted.
 */
import type { FrameContext } from '../app/modules';
import { shaderModule } from '../gpu/util';
import { frameBindingsWGSL } from '../render/frame';
import type { RendererShared } from '../render/Renderer';
import { invPoseMatrix, poseMatrix, proxyParams, safeScale, type PoseLike } from './math';

export type RGBA = [number, number, number, number];

export interface OverlayShape {
  pose: PoseLike;
  mode: 'ghost' | 'highlight';
  /** Display-space colour and overall opacity. */
  color: RGBA;
  /** Fill / body opacity. */
  fill: number;
  /** Outline (highlight) or fresnel-rim (ghost) opacity. */
  edge: number;
  /** Opacity factor where hidden (ghost: buried; highlight: behind water/particles). */
  occluded: number;
  /** 0..1 pulsing. */
  pulse?: number;
  seed?: number;
  /** Ghost surface-noise amplitude override. */
  noise?: number;
}

export interface OverlaySegment {
  a: readonly number[];
  b: readonly number[];
  /** Half width (CSS px; multiplied by devicePixelRatio). */
  width: number;
  color: RGBA;
  glow?: number;
  glowAlpha?: number;
  /** Opacity factor where hidden. */
  occluded?: number;
  /** Dash period in metres (0 = solid) and phase start. */
  dash?: number;
  dashStart?: number;
}

const LAT = 20; // even
const LON = 28;
const SHAPE_VERTS = (LAT + 1) * LON * 6;
const SHAPE_FLOATS = 48;
const SEG_FLOATS = 16;
export const MAX_OVERLAY_SHAPES = 16;
export const MAX_OVERLAY_SEGMENTS = 4096;

const overlayWGSL = /* wgsl */ `
${frameBindingsWGSL}

struct OverlayUniforms {
  viewport: vec4f, // target width, height, 1/width, 1/height (px)
  depthMap: vec4f, // depth texture width, height, depth px per target px (x, y)
  misc: vec4f,     // real time (s), device pixel ratio, -, -
}
struct Shape {
  model: mat4x4f,    // local (rotated, unscaled) -> world
  invModel: mat4x4f, // world -> local
  scale: vec4f,      // half extents, w = 1 for capsules (logs)
  shape: vec4f,      // super-ellipsoid exponent, noise amplitude, seed, mesh inflate (> 1: highlight hull)
  color: vec4f,
  style: vec4f,      // fill, edge, occluded factor, pulse
}
struct Segment {
  a: vec4f,     // xyz, half width (px)
  b: vec4f,     // xyz, dash period (m; 0 = solid)
  color: vec4f,
  style: vec4f, // glow width (px), glow alpha, occluded alpha, dash start (m)
}
@group(2) @binding(0) var sceneDepth: texture_depth_2d;
@group(2) @binding(1) var opaqueDepthTex: texture_2d<f32>;
@group(2) @binding(2) var normalTex: texture_2d<f32>;
@group(2) @binding(3) var<uniform> ov: OverlayUniforms;
@group(2) @binding(4) var<storage, read> shapes: array<Shape>;
@group(2) @binding(5) var<storage, read> segs: array<Segment>;

const LAT: u32 = ${LAT}u;
const LON: u32 = ${LON}u;

fn texel(px: vec2f) -> vec2i {
  let dims = vec2i(ov.depthMap.xy);
  return clamp(vec2i(px * ov.depthMap.zw), vec2i(0), dims - 1);
}
fn sceneDepthAt(px: vec2f) -> f32 { return textureLoad(sceneDepth, texel(px), 0); }
fn opaqueDepthAt(px: vec2f) -> f32 { return textureLoad(opaqueDepthTex, texel(px), 0).r; }
fn normalAt(px: vec2f) -> vec3f { return textureLoad(normalTex, texel(px), 0).xyz; }
fn worldAtPx(px: vec2f, d: f32) -> vec3f { return worldFromDepth(px * ov.viewport.zw, d); }
fn linDepth(d: f32) -> f32 { return select(1e9, linearDepth(max(d, 1e-9)), d > 0.0); }

fn hash31(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}
fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(hash31(i), hash31(i + vec3f(1.0, 0.0, 0.0)), u.x);
  let b = mix(hash31(i + vec3f(0.0, 1.0, 0.0)), hash31(i + vec3f(1.0, 1.0, 0.0)), u.x);
  let c = mix(hash31(i + vec3f(0.0, 0.0, 1.0)), hash31(i + vec3f(1.0, 0.0, 1.0)), u.x);
  let d = mix(hash31(i + vec3f(0.0, 1.0, 1.0)), hash31(i + vec3f(1.0, 1.0, 1.0)), u.x);
  return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}
fn fbm3(p: vec3f) -> f32 { return 0.55 * vnoise(p) + 0.3 * vnoise(p * 2.03 + 17.1) + 0.15 * vnoise(p * 4.07 + 3.7); }

/** Proxy surface point (local, unscaled frame) for lat-long coordinates; theta from +X. */
fn proxyLocal(s: Shape, theta: f32, side: f32, phi: f32) -> vec3f {
  let st = sin(theta);
  let u = vec3f(cos(theta), st * cos(phi), st * sin(phi));
  var n = 1.0;
  if (s.shape.y > 0.0) {
    let off = vec3f(fract(s.shape.z * 0.1371) * 37.0, fract(s.shape.z * 0.0713) * 53.0, fract(s.shape.z * 0.1137) * 41.0);
    n = 1.0 + 2.0 * s.shape.y * (fbm3(u * 1.9 + off) - 0.5);
  }
  if (s.scale.w > 0.5) {
    let r = 0.5 * (s.scale.y + s.scale.z);
    let hl = max(s.scale.x - r, 0.0);
    return vec3f(u.x * r * n + side * hl, u.y * s.scale.y * n, u.z * s.scale.z * n);
  }
  let e = max(s.shape.x, 1.0);
  return sign(u) * pow(max(abs(u), vec3f(1e-6)), vec3f(2.0 / e)) * s.scale.xyz * n;
}

/** Ring k → (theta, side): the equator ring is duplicated so capsules get a cylinder. */
fn ringOf(k: u32) -> vec2f {
  let half = LAT / 2u;
  if (k <= half) { return vec2f(f32(k) / f32(LAT) * PI, 1.0); }
  return vec2f(f32(k - 1u) / f32(LAT) * PI, -1.0);
}

struct ShapeOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) inst: u32,
}

@vertex fn vsShape(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ShapeOut {
  let s = shapes[ii];
  let quad = vi / 6u;
  let c = vi % 6u;
  var dk = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
  var dj = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
  let k = quad / LON + dk[c];
  let j = quad % LON + dj[c];
  let rs = ringOf(k);
  let phi = f32(j % LON) / f32(LON) * 2.0 * PI;
  var p = proxyLocal(s, rs.x, rs.y, phi);
  let e = 0.02;
  let th = clamp(rs.x, e, PI - 2.0 * e);
  let p0 = proxyLocal(s, th, rs.y, phi);
  var n = cross(proxyLocal(s, th + e, rs.y, phi) - p0, proxyLocal(s, th, rs.y, phi + e) - p0);
  n = select(p, n, dot(n, n) > 1e-20);
  n = normalize(n + vec3f(0.0, 1e-9, 0.0));
  if (s.shape.w > 1.0) { p = p * s.shape.w + n * 0.008; }
  let wp = (s.model * vec4f(p, 1.0)).xyz;
  var o: ShapeOut;
  o.pos = frame.viewProj * vec4f(wp, 1.0);
  o.world = wp;
  o.normal = normalize((s.model * vec4f(n, 0.0)).xyz);
  o.inst = ii;
  return o;
}

@fragment fn fsGhost(in: ShapeOut) -> @location(0) vec4f {
  let s = shapes[in.inst];
  let px = in.pos.xy;
  let lf = linDepth(in.pos.z);
  let ls = linDepth(sceneDepthAt(px));
  let n = normalize(in.normal);
  let v = normalize(frame.cameraPos.xyz - in.world);
  let ndv = clamp(dot(n, v), 0.0, 1.0);
  let fres = pow(1.0 - ndv, 3.0);
  let lightK = 0.5 + 0.5 * max(dot(n, frame.sunDir.xyz), 0.0);
  var col = s.color.rgb * (0.55 + 0.45 * lightK) + vec3f(0.3) * fres;
  var a = s.style.x * (0.55 + 0.45 * lightK) + s.style.y * fres;
  let gap = abs(lf - ls);
  let contact = (1.0 - smoothstep(0.0, 0.0025 + 0.0035 * lf, gap)) * step(ls, 1e8);
  if (lf > ls + 0.0015 + 0.002 * lf) {
    let hatch = 0.3 + 0.7 * step(0.5, fract((px.x - px.y) / (7.0 * ov.misc.y)));
    a *= s.style.z * hatch;
  }
  a = clamp(a + contact * 0.85, 0.0, 1.0) * s.color.a;
  col = mix(col, vec3f(1.0), contact * 0.5);
  if (a < 0.002) { discard; }
  return vec4f(col * a, a);
}

/** xyz: local gradient of the proxy function, w: normalised radius q (1 on the surface). */
fn shapeQ(s: Shape, lp: vec3f) -> vec4f {
  if (s.scale.w > 0.5) {
    let r = 0.5 * (s.scale.y + s.scale.z);
    let hl = max(s.scale.x - r, 0.0);
    let ex = sign(lp.x) * max(abs(lp.x) - hl, 0.0) / r;
    let v = vec3f(ex, lp.y / s.scale.y, lp.z / s.scale.z);
    return vec4f(v.x / r, v.y / s.scale.y, v.z / s.scale.z, length(v));
  }
  let e = max(s.shape.x, 1.0);
  let v = lp / s.scale.xyz;
  let av = max(abs(v), vec3f(1e-6));
  let q = pow(dot(pow(av, vec3f(e)), vec3f(1.0)), 1.0 / e);
  let g = sign(v) * pow(av, vec3f(e - 1.0)) / s.scale.xyz;
  return vec4f(g, q);
}

/** 1 where the visible opaque surface at px belongs to the obstacle. */
fn maskAt(s: Shape, px: vec2f) -> f32 {
  let d = opaqueDepthAt(px);
  if (d <= 0.0) { return 0.0; }
  let w = worldAtPx(px, d);
  let lp = (s.invModel * vec4f(w, 1.0)).xyz;
  let gq = shapeQ(s, lp);
  let np = normalize((s.model * vec4f(gq.xyz, 0.0)).xyz + vec3f(0.0, 1e-7, 0.0));
  let nr = normalAt(px);
  let agree = select(0.0, dot(normalize(nr), np), dot(nr, nr) > 0.01);
  // Generous where the rendered normal agrees with the proxy (noisy rock surface),
  // tight where it does not (the ground around an embedded rock).
  let tol = 0.07 + 0.38 * smoothstep(0.15, 0.6, agree);
  return 1.0 - smoothstep(1.0 + tol - 0.05, 1.0 + tol + 0.05, gq.w);
}

@fragment fn fsHighlight(in: ShapeOut) -> @location(0) vec4f {
  let s = shapes[in.inst];
  let px = floor(in.pos.xy) + 0.5;
  let dpr = ov.misc.y;
  let m0 = maskAt(s, px);
  var near = 0.0;
  var far = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let a = f32(i) * (PI / 4.0) + 0.3927;
    let dir = vec2f(cos(a), sin(a));
    near = max(near, maskAt(s, px + dir * 1.5 * dpr));
    far = max(far, maskAt(s, px + dir * 3.5 * dpr));
  }
  let outline = clamp(near - m0, 0.0, 1.0);
  let glow = clamp(far - m0, 0.0, 1.0) * 0.35;
  let covered = sceneDepthAt(px) > opaqueDepthAt(px) + 1e-6;
  let vis = select(1.0, s.style.z, covered);
  let pulse = 1.0 + s.style.w * 0.35 * sin(ov.misc.x * 5.0);
  var a = (max(outline, glow) * s.style.y + m0 * s.style.x) * pulse;
  a = clamp(a, 0.0, 1.0) * vis * s.color.a;
  if (a < 0.002) { discard; }
  return vec4f(s.color.rgb * a, a);
}

struct LineOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f, // across (px), along (px)
  @location(1) @interpolate(flat) inst: u32,
  @location(2) @interpolate(flat) lenPx: f32,
  @location(3) @interpolate(flat) lenWorld: f32,
}

@vertex fn vsLine(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LineOut {
  let sg = segs[ii];
  var ca = frame.viewProj * vec4f(sg.a.xyz, 1.0);
  var cb = frame.viewProj * vec4f(sg.b.xyz, 1.0);
  var o: LineOut;
  o.inst = ii;
  o.lenWorld = distance(sg.a.xyz, sg.b.xyz);
  o.uv = vec2f(0.0);
  o.lenPx = 0.0;
  let eps = 1e-5;
  if (ca.w < eps && cb.w < eps) {
    o.pos = vec4f(4.0, 4.0, 0.5, 1.0);
    return o;
  }
  if (ca.w < eps) { ca = mix(ca, cb, (eps - ca.w) / (cb.w - ca.w)); }
  if (cb.w < eps) { cb = mix(cb, ca, (eps - cb.w) / (ca.w - cb.w)); }
  let half = 0.5 * ov.viewport.xy;
  let sa = ca.xy / ca.w * half;
  let sb = cb.xy / cb.w * half;
  let dv = sb - sa;
  let len = length(dv);
  let dir = select(vec2f(1.0, 0.0), dv / max(len, 1e-6), len > 1e-4);
  let nrm = vec2f(-dir.y, dir.x);
  let ext = (sg.a.w + sg.style.x + 1.5) * ov.misc.y;
  var ts = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
  var sd = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
  let t = ts[vi % 6u];
  let side = sd[vi % 6u];
  let p = mix(sa, sb, t) + nrm * side * ext + dir * (t * 2.0 - 1.0) * ext;
  let z = clamp(mix(ca.z / ca.w, cb.z / cb.w, t), 0.0, 1.0);
  o.pos = vec4f(p / half, z, 1.0);
  o.uv = vec2f(side * ext, t * len + (t * 2.0 - 1.0) * ext);
  o.lenPx = len;
  return o;
}

@fragment fn fsLine(in: LineOut) -> @location(0) vec4f {
  let sg = segs[in.inst];
  let dpr = ov.misc.y;
  let along = in.uv.y;
  let dAlong = max(max(-along, along - in.lenPx), 0.0);
  let d = length(vec2f(in.uv.x, dAlong));
  let hw = sg.a.w * dpr;
  let core = clamp(hw + 0.5 - d, 0.0, 1.0);
  let gw = max(sg.style.x * dpr, 0.001);
  let glow = sg.style.y * exp(-2.0 * pow(max(d - hw, 0.0) / gw, 2.0)) * step(0.0001, sg.style.x);
  var a = max(core, glow);
  if (sg.b.w > 0.0) {
    let tAlong = clamp(along / max(in.lenPx, 1e-3), 0.0, 1.0);
    let ph = fract((sg.style.w + tAlong * in.lenWorld) / sg.b.w);
    a *= smoothstep(0.0, 0.08, ph) * (1.0 - smoothstep(0.5, 0.58, ph));
  }
  let ls = linDepth(sceneDepthAt(in.pos.xy));
  let lf = linDepth(in.pos.z);
  if (lf > ls * 1.003 + 0.004) {
    let dots = step(0.45, fract(along / (5.0 * dpr)));
    a *= sg.style.z * dots;
  }
  a *= sg.color.a;
  if (a < 0.002) { discard; }
  return vec4f(sg.color.rgb * a, a);
}
`;

export class OverlayRenderer {
  private layout: GPUBindGroupLayout;
  private ghostPipe: GPURenderPipeline;
  private highlightPipe: GPURenderPipeline;
  private linePipe: GPURenderPipeline;
  private uniform: GPUBuffer;
  private shapeBuf: GPUBuffer;
  private segBuf: GPUBuffer;
  private shapeData = new Float32Array(MAX_OVERLAY_SHAPES * SHAPE_FLOATS);
  private segData = new Float32Array(MAX_OVERLAY_SEGMENTS * SEG_FLOATS);
  private uniformData = new Float32Array(12);
  private bindGroup?: GPUBindGroup;
  private bindKey: unknown[] = [];
  private tmpA = new Float32Array(16);
  private tmpB = new Float32Array(16);

  constructor(private device: GPUDevice, shared: RendererShared) {
    const F = GPUShaderStage.FRAGMENT;
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.layout = device.createBindGroupLayout({
      label: 'editor.overlayLayout',
      entries: [
        { binding: 0, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: F, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: F, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 4, visibility: VF, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: VF, buffer: { type: 'read-only-storage' } },
      ],
    });
    const layout = device.createPipelineLayout({
      label: 'editor.overlay',
      bindGroupLayouts: [shared.frameLayout, shared.lightingLayout, this.layout],
    });
    const module = shaderModule(device, 'editor.overlay', overlayWGSL);
    const blend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const targets: GPUColorTargetState[] = [{ format: shared.presentationFormat, blend }];
    const pipe = (label: string, vs: string, fs: string, cullMode: GPUCullMode) =>
      device.createRenderPipeline({
        label,
        layout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets },
        primitive: { topology: 'triangle-list', cullMode, frontFace: 'ccw' },
      });
    this.highlightPipe = pipe('editor.overlay.highlight', 'vsShape', 'fsHighlight', 'front');
    this.ghostPipe = pipe('editor.overlay.ghost', 'vsShape', 'fsGhost', 'back');
    this.linePipe = pipe('editor.overlay.lines', 'vsLine', 'fsLine', 'none');
    this.uniform = device.createBuffer({ label: 'editor.overlay.uniforms', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shapeBuf = device.createBuffer({ label: 'editor.overlay.shapes', size: this.shapeData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.segBuf = device.createBuffer({ label: 'editor.overlay.segments', size: this.segData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  /** 1×1 stand-ins (zero = "sky / nothing") used if the scene targets cannot be sampled as expected. */
  private dummies?: { depth: GPUTexture; r32: GPUTexture; rgba: GPUTexture };

  private usableTargets(frame: FrameContext) {
    const t = frame.targets;
    const ok = (tex: GPUTexture) => tex.sampleCount === 1 && (tex.usage & GPUTextureUsage.TEXTURE_BINDING) !== 0;
    return ok(t.depth) && ok(t.opaqueDepth) && ok(t.normal) && t.opaqueDepth.format === 'r32float';
  }

  private ensureBindGroup(frame: FrameContext): [number, number] {
    const t = frame.targets;
    const usable = this.usableTargets(frame);
    const key = [t.depth, t.opaqueDepth, t.normal, usable];
    const size: [number, number] = usable ? [t.width, t.height] : [1, 1];
    if (this.bindGroup && key.every((k, i) => k === this.bindKey[i])) return size;
    this.bindKey = key;
    let views = { depth: t.depthView, opaque: t.opaqueDepthView, normal: t.normalView };
    if (!usable) {
      const tex = (format: GPUTextureFormat) => this.device.createTexture({ size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
      this.dummies ??= { depth: tex('depth32float'), r32: tex('r32float'), rgba: tex('rgba16float') };
      views = { depth: this.dummies.depth.createView(), opaque: this.dummies.r32.createView(), normal: this.dummies.rgba.createView() };
    }
    this.bindGroup = this.device.createBindGroup({
      label: 'editor.overlay',
      layout: this.layout,
      entries: [
        { binding: 0, resource: views.depth },
        { binding: 1, resource: views.opaque },
        { binding: 2, resource: views.normal },
        { binding: 3, resource: { buffer: this.uniform } },
        { binding: 4, resource: { buffer: this.shapeBuf } },
        { binding: 5, resource: { buffer: this.segBuf } },
      ],
    });
    return size;
  }

  private writeShape(i: number, s: OverlayShape) {
    const o = i * SHAPE_FLOATS;
    const d = this.shapeData;
    const pp = proxyParams(s.pose.kind);
    d.set(poseMatrix(s.pose, this.tmpA), o);
    d.set(invPoseMatrix(s.pose, this.tmpB), o + 16);
    const sc = safeScale(s.pose.scale);
    d.set([sc[0], sc[1], sc[2], pp.capsule ? 1 : 0], o + 32);
    const ghost = s.mode === 'ghost';
    d.set([pp.exponent, ghost ? (s.noise ?? pp.noise) : 0, (s.seed ?? 0) % 9973, ghost ? 1 : 1.6], o + 36);
    d.set(s.color, o + 40);
    d.set([s.fill, s.edge, s.occluded, s.pulse ?? 0], o + 44);
  }

  private writeSegment(i: number, s: OverlaySegment) {
    const o = i * SEG_FLOATS;
    this.segData.set(
      [
        s.a[0], s.a[1], s.a[2], s.width,
        s.b[0], s.b[1], s.b[2], s.dash ?? 0,
        s.color[0], s.color[1], s.color[2], s.color[3],
        s.glow ?? 0, s.glowAlpha ?? 0, s.occluded ?? 0.3, s.dashStart ?? 0,
      ],
      o,
    );
  }

  /** Records the overlay pass. Shapes / segments are drawn in the order highlight → ghost → lines. */
  draw(
    encoder: GPUCommandEncoder,
    frame: FrameContext,
    target: GPUTextureView,
    viewport: [number, number],
    dpr: number,
    shapes: readonly OverlayShape[],
    segments: readonly OverlaySegment[],
  ) {
    const ok = (s: OverlayShape) => [...s.pose.position, ...s.pose.rotation, ...s.pose.scale].every(Number.isFinite);
    const hl = shapes.filter((s) => s.mode === 'highlight' && ok(s));
    const gh = shapes.filter((s) => s.mode === 'ghost' && ok(s));
    const nShapes = Math.min(MAX_OVERLAY_SHAPES, hl.length + gh.length);
    const nh = Math.min(hl.length, nShapes);
    const ng = nShapes - nh;
    const segs = segments.filter((s) => [...s.a, ...s.b].every(Number.isFinite));
    const nSeg = Math.min(MAX_OVERLAY_SEGMENTS, segs.length);
    if (!nShapes && !nSeg) return;
    const w = Math.max(1, viewport[0]);
    const h = Math.max(1, viewport[1]);
    const [dw, dh] = this.ensureBindGroup(frame);
    this.uniformData.set([w, h, 1 / w, 1 / h, dw, dh, dw / w, dh / h, frame.realTime % 3600, Math.max(0.5, Math.min(4, dpr || 1)), 0, 0]);
    this.device.queue.writeBuffer(this.uniform, 0, this.uniformData);
    [...hl.slice(0, nh), ...gh.slice(0, ng)].forEach((s, i) => this.writeShape(i, s));
    if (nShapes) this.device.queue.writeBuffer(this.shapeBuf, 0, this.shapeData, 0, nShapes * SHAPE_FLOATS);
    for (let i = 0; i < nSeg; i++) this.writeSegment(i, segs[i]);
    if (nSeg) this.device.queue.writeBuffer(this.segBuf, 0, this.segData, 0, nSeg * SEG_FLOATS);

    const pass = encoder.beginRenderPass({
      label: 'editor.overlay',
      colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store' }],
    });
    pass.setBindGroup(0, frame.frameBindGroup);
    pass.setBindGroup(1, frame.lightingBindGroup);
    pass.setBindGroup(2, this.bindGroup!);
    if (nh) {
      pass.setPipeline(this.highlightPipe);
      pass.draw(SHAPE_VERTS, nh, 0, 0);
    }
    if (ng) {
      pass.setPipeline(this.ghostPipe);
      pass.draw(SHAPE_VERTS, ng, 0, nh);
    }
    if (nSeg) {
      pass.setPipeline(this.linePipe);
      pass.draw(6, nSeg, 0, 0);
    }
    pass.end();
  }

  destroy() {
    this.uniform.destroy();
    this.shapeBuf.destroy();
    this.segBuf.destroy();
    if (this.dummies) Object.values(this.dummies).forEach((t) => t.destroy());
  }
}
