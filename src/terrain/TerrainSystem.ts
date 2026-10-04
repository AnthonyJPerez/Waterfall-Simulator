/**
 * Terrain + obstacles: generates the base creek bed, bakes the shared bed
 * heightfield and 3-D SDF, answers CPU ray casts and renders terrain/rocks.
 *
 * STUB IMPLEMENTATION — to be replaced by the terrain agent (keep the TerrainSystem API).
 */
import type { FrameContext, ModuleContext, RayHit, TerrainSystem } from '../app/modules';
import { computePipeline, shaderModule } from '../gpu/util';
import { DEPTH_FORMAT, frameBindingsWGSL, HDR_FORMAT, lightingBindingsWGSL, NORMAL_FORMAT } from '../render/frame';
import { lightingFunctionsWGSL } from '../render/env/lighting';
import { worldUniformsWGSL } from '../world/wgsl';
import type { Obstacle } from '../world/scene';
import type { ScenePreset } from './presets';

function smoothstep(a: number, b: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Stub base terrain: a channel with a ledge drop into a pool. */
export function stubBaseHeight(preset: ScenePreset, x: number, z: number): number {
  const t = preset.terrain as any;
  const ledgeX = t.ledgeX ?? 1.0;
  const up = (t.upstreamY ?? 0.3) - 0.03 * x;
  const poolDepth = t.poolY ?? -0.15;
  const tail = t.tailY ?? 0.0;
  const lip = ledgeX + 0.02 * Math.sin(z * 23.0) + 0.015 * Math.sin(z * 57.0 + 1.0);
  let y: number;
  if (x < lip) y = up;
  else {
    const u = (x - lip) / (preset.extent.sizeX - lip);
    y = poolDepth + (tail - poolDepth) * smoothstep(0.25, 1.0, u);
    y = Math.min(y, up);
  }
  const zc = preset.extent.sizeZ / 2;
  const bank = smoothstep(0.38, 0.6, Math.abs(z - zc));
  y = y + bank * 0.35;
  y += 0.006 * Math.sin(x * 41 + z * 13) * Math.cos(z * 37 - x * 7);
  return y;
}

function ellipsoidTop(o: Obstacle, x: number, z: number): number {
  // Ignores rotation (stub). Returns -Infinity outside the footprint.
  const dx = (x - o.position[0]) / o.scale[0];
  const dz = (z - o.position[2]) / o.scale[2];
  const r2 = dx * dx + dz * dz;
  if (r2 >= 1) return -Infinity;
  return o.position[1] + o.scale[1] * Math.sqrt(1 - r2);
}

const MAX_OBSTACLES = 64;

export class StubTerrainSystem implements TerrainSystem {
  bedVersion = 0;
  private base: Float32Array<ArrayBuffer>;
  private bed: Float32Array<ArrayBuffer>;
  private dirty = true;
  private sdfPipeline: GPUComputePipeline;
  private obstacleBuf: GPUBuffer;
  private sdfBindGroup: GPUBindGroup;
  private drawPipeline: GPURenderPipeline;
  private shadowPipeline: GPURenderPipeline;
  private indexBuf: GPUBuffer;
  private indexCount: number;
  private unsub: () => void;

  constructor(private ctx: ModuleContext) {
    const { world, preset, gpu } = ctx;
    const device = gpu.device;
    const d = world.domain;
    this.base = new Float32Array(d.nx * d.nz);
    for (let j = 0; j < d.nz; j++)
      for (let i = 0; i < d.nx; i++) this.base[j * d.nx + i] = stubBaseHeight(preset, (i + 0.5) * d.cellSize, (j + 0.5) * d.cellSize);
    this.bed = new Float32Array(this.base.length); this.bed.set(this.base);
    this.unsub = ctx.scene.onChange(() => (this.dirty = true));

    this.obstacleBuf = device.createBuffer({ size: 16 + MAX_OBSTACLES * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.sdfPipeline = computePipeline(
      device,
      'terrainStub.sdf',
      /* wgsl */ `
${worldUniformsWGSL}
struct Ell { center: vec4f, radii: vec4f }
struct Obs { count: u32, _p0: u32, _p1: u32, _p2: u32, e: array<Ell, ${MAX_OBSTACLES}> }
@group(0) @binding(0) var<uniform> world: WorldUniforms;
@group(0) @binding(1) var bedTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> obs: Obs;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
fn bedAt(c: vec2i) -> f32 {
  let n = vec2i(i32(world.grid.x), i32(world.grid.y));
  return textureLoad(bedTex, clamp(c, vec2i(0), n - 1), 0).r;
}
fn bedH(xz: vec2f) -> f32 {
  let g = xz * world.grid.w - 0.5; let f = floor(g); let t = g - f; let c = vec2i(f);
  return mix(mix(bedAt(c), bedAt(c + vec2i(1, 0)), t.x), mix(bedAt(c + vec2i(0, 1)), bedAt(c + vec2i(1, 1)), t.x), t.y);
}
fn sdf(p: vec3f) -> f32 {
  let h = bedH(p.xz);
  let e = world.grid.z * 2.0;
  let gx = (bedH(p.xz + vec2f(e, 0.0)) - bedH(p.xz - vec2f(e, 0.0))) / (2.0 * e);
  let gz = (bedH(p.xz + vec2f(0.0, e)) - bedH(p.xz - vec2f(0.0, e))) / (2.0 * e);
  var d = (p.y - h) / sqrt(1.0 + gx * gx + gz * gz);
  for (var i = 0u; i < obs.count; i++) {
    let q = (p - obs.e[i].center.xyz) / obs.e[i].radii.xyz;
    let k = length(q);
    d = min(d, (k - 1.0) * min(obs.e[i].radii.x, min(obs.e[i].radii.y, obs.e[i].radii.z)));
  }
  return d;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let dims = vec3u(world.sdfDims.xyz);
  if (any(g >= dims)) { return; }
  let v = world.sdfDims.w;
  let p = vec3f(0.0, world.domainSize.z, 0.0) + (vec3f(g) + 0.5) * v;
  let d = sdf(p);
  let e = v * 0.5;
  let n = vec3f(sdf(p + vec3f(e, 0, 0)) - sdf(p - vec3f(e, 0, 0)), sdf(p + vec3f(0, e, 0)) - sdf(p - vec3f(0, e, 0)), sdf(p + vec3f(0, 0, e)) - sdf(p - vec3f(0, 0, e)));
  textureStore(dst, g, vec4f(d, normalize(n + vec3f(0.0, 1e-6, 0.0))));
}
`,
    );
    this.sdfBindGroup = device.createBindGroup({
      layout: this.sdfPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: world.uniforms } },
        { binding: 1, resource: world.views.bed },
        { binding: 2, resource: { buffer: this.obstacleBuf } },
        { binding: 3, resource: world.textures.sdf.createView({ dimension: '3d' }) },
      ],
    });

    // Grid mesh over the bed texture (one vertex per cell centre).
    const nx = d.nx, nz = d.nz;
    const idx = new Uint32Array((nx - 1) * (nz - 1) * 6);
    let k = 0;
    for (let j = 0; j < nz - 1; j++)
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i, b = a + 1, c = a + nx, e = c + 1;
        idx.set([a, c, b, b, c, e], k);
        k += 6;
      }
    this.indexCount = idx.length;
    this.indexBuf = device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.indexBuf, 0, idx);

    const code = /* wgsl */ `
${frameBindingsWGSL}
${lightingBindingsWGSL}
${lightingFunctionsWGSL}
struct VOut { @builtin(position) pos: vec4f, @location(0) world: vec3f }
fn gridPos(vi: u32) -> vec3f {
  let nx = u32(world.grid.x);
  let c = vec2i(i32(vi % nx), i32(vi / nx));
  let xz = (vec2f(c) + 0.5) * world.grid.z;
  return vec3f(xz.x, bedAt(c), xz.y);
}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let p = gridPos(vi);
  var o: VOut; o.pos = frame.viewProj * vec4f(p, 1.0); o.world = p; return o;
}
@vertex fn vsShadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return frame.shadowViewProj * vec4f(gridPos(vi), 1.0);
}
struct FOut { @location(0) color: vec4f, @location(1) normal: vec4f }
@fragment fn fs(in: VOut) -> FOut {
  let e = world.grid.z;
  let hx = bedHeight(in.world.xz + vec2f(e, 0.0)) - bedHeight(in.world.xz - vec2f(e, 0.0));
  let hz = bedHeight(in.world.xz + vec2f(0.0, e)) - bedHeight(in.world.xz - vec2f(0.0, e));
  let n = normalize(vec3f(-hx, 2.0 * e, -hz));
  let slope = 1.0 - n.y;
  var albedo = mix(vec3f(0.36, 0.33, 0.27), vec3f(0.42, 0.40, 0.37), smoothstep(0.1, 0.5, slope));
  let wet = textureSampleLevel(wetnessTex, linearClamp, gridUv(in.world.xz), 0.0).r;
  let w = waterSample(in.world.xz);
  let under = w.x > 0.0005;
  albedo *= select(1.0 - 0.45 * wet, 0.55, under);
  let ndl = max(dot(n, frame.sunDir.xyz), 0.0);
  var light = sunIrradiance() * ndl * sunVisibility(in.world, n) + ambientIrradiance(n);
  if (under) { light = sunIrradiance() * ndl * sunVisibility(in.world, n) * causticsAt(in.world) + ambientIrradiance(n); }
  var o: FOut;
  o.color = vec4f(albedo * light / PI, 1.0);
  o.normal = vec4f(n, 0.8);
  return o;
}
`;
    const module = shaderModule(device, 'terrainStub.draw', code);
    const layout = device.createPipelineLayout({ bindGroupLayouts: [ctx.shared.frameLayout, ctx.shared.lightingLayout] });
    this.drawPipeline = device.createRenderPipeline({
      label: 'terrainStub.draw',
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }, { format: NORMAL_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'greater', depthWriteEnabled: true },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
    this.shadowPipeline = device.createRenderPipeline({
      label: 'terrainStub.shadow',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ctx.shared.frameLayout] }),
      vertex: { module, entryPoint: 'vsShadow' },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'less', depthWriteEnabled: true, depthBias: 2, depthBiasSlopeScale: 2 },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  private rebake(encoder: GPUCommandEncoder) {
    const { world, scene, gpu } = this.ctx;
    const d = world.domain;
    this.bed.set(this.base);
    for (const o of scene.obstacles) {
      const r = Math.max(o.scale[0], o.scale[2]);
      const i0 = Math.max(0, Math.floor((o.position[0] - r) / d.cellSize)), i1 = Math.min(d.nx - 1, Math.ceil((o.position[0] + r) / d.cellSize));
      const j0 = Math.max(0, Math.floor((o.position[2] - r) / d.cellSize)), j1 = Math.min(d.nz - 1, Math.ceil((o.position[2] + r) / d.cellSize));
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const top = ellipsoidTop(o, (i + 0.5) * d.cellSize, (j + 0.5) * d.cellSize);
          const k = j * d.nx + i;
          if (top > this.bed[k]) this.bed[k] = top;
        }
    }
    gpu.device.queue.writeTexture({ texture: world.textures.bed }, this.bed, { bytesPerRow: d.nx * 4 }, [d.nx, d.nz]);
    const obs = new Float32Array(4 + MAX_OBSTACLES * 8);
    const n = Math.min(MAX_OBSTACLES, scene.obstacles.length);
    new Uint32Array(obs.buffer)[0] = n;
    scene.obstacles.slice(0, n).forEach((o, i) => {
      obs.set([o.position[0], o.position[1], o.position[2], 0, o.scale[0], o.scale[1], o.scale[2], 0], 4 + i * 8);
    });
    gpu.device.queue.writeBuffer(this.obstacleBuf, 0, obs);
    const pass = encoder.beginComputePass({ label: 'terrainStub.sdf' });
    pass.setPipeline(this.sdfPipeline);
    pass.setBindGroup(0, this.sdfBindGroup);
    pass.dispatchWorkgroups(Math.ceil(d.sdfNx / 4), Math.ceil(d.sdfNy / 4), Math.ceil(d.sdfNz / 4));
    pass.end();
    this.bedVersion++;
  }

  update(encoder: GPUCommandEncoder, _frame: FrameContext) {
    if (this.dirty) {
      this.dirty = false;
      this.rebake(encoder);
    }
  }

  heightAt(x: number, z: number): number {
    const d = this.ctx.world.domain;
    const i = Math.min(d.nx - 1, Math.max(0, Math.floor(x / d.cellSize)));
    const j = Math.min(d.nz - 1, Math.max(0, Math.floor(z / d.cellSize)));
    return this.bed[j * d.nx + i];
  }

  raycast(origin: readonly number[], dir: readonly number[], maxDistance = 20): RayHit | null {
    const step = this.ctx.world.domain.cellSize * 0.5;
    let prev = origin[1] - this.heightAt(origin[0], origin[2]);
    for (let t = step; t < maxDistance; t += step) {
      const x = origin[0] + dir[0] * t, y = origin[1] + dir[1] * t, z = origin[2] + dir[2] * t;
      const h = y - this.heightAt(x, z);
      if (h < 0 && prev >= 0) {
        const obstacle = this.ctx.scene.obstacles.find((o) => ellipsoidTop(o, x, z) > -Infinity && ellipsoidTop(o, x, z) >= y - 0.01);
        return { point: [x, y, z], normal: [0, 1, 0], distance: t, obstacleId: obstacle?.id };
      }
      prev = h;
    }
    return null;
  }

  drawShadow(pass: GPURenderPassEncoder, frame: FrameContext) {
    pass.setPipeline(this.shadowPipeline);
    pass.setBindGroup(0, frame.frameBindGroup);
    pass.setIndexBuffer(this.indexBuf, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  drawOpaque(pass: GPURenderPassEncoder, frame: FrameContext) {
    pass.setPipeline(this.drawPipeline);
    pass.setBindGroup(0, frame.frameBindGroup);
    pass.setBindGroup(1, frame.lightingBindGroup);
    pass.setIndexBuffer(this.indexBuf, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  destroy() {
    this.unsub();
    this.obstacleBuf.destroy();
    this.indexBuf.destroy();
  }
}
