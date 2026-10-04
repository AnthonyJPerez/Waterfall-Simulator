/**
 * Falling-water particle simulation (over-edge flow, jets, trickles, splashes).
 *
 * STUB IMPLEMENTATION — ballistic particles emitted from world.overflow, absorbed into
 * world.deposit when they hit water/bed. To be replaced by the particles agent
 * (keep the ParticleSim API and the ParticleView buffer layouts).
 */
import type { FrameContext, ModuleContext, ParticleSim, ParticleView } from '../../app/modules';
import { computePipeline } from '../../gpu/util';
import { couplingWGSL, worldUniformsWGSL } from '../../world/wgsl';

export class StubParticleSim implements ParticleSim {
  readonly view: ParticleView;
  stats: Record<string, number> = {};
  private emit: GPUComputePipeline;
  private update: GPUComputePipeline;
  private clearEvents: GPUComputePipeline;
  private bg: GPUBindGroup;
  private cursor: GPUBuffer;

  constructor(private ctx: ModuleContext) {
    const { gpu, world } = ctx;
    const device = gpu.device;
    const d = world.domain;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const positions = device.createBuffer({ label: 'particles.pos', size: d.maxParticles * 16, usage: S | GPUBufferUsage.VERTEX });
    const velocities = device.createBuffer({ label: 'particles.vel', size: d.maxParticles * 16, usage: S | GPUBufferUsage.VERTEX });
    const drawArgs = device.createBuffer({ label: 'particles.args', size: 16, usage: S | GPUBufferUsage.INDIRECT });
    device.queue.writeBuffer(drawArgs, 0, new Uint32Array([6, d.maxParticles, 0, 0]));
    const diffuse = device.createBuffer({ label: 'particles.diffuse', size: d.maxDiffuseParticles * 32, usage: S });
    const diffuseDrawArgs = device.createBuffer({ label: 'particles.diffuseArgs', size: 16, usage: S | GPUBufferUsage.INDIRECT });
    device.queue.writeBuffer(diffuseDrawArgs, 0, new Uint32Array([6, 0, 0, 0]));
    this.cursor = device.createBuffer({ size: 16, usage: S });
    this.view = { maxParticles: d.maxParticles, radius: d.particleRadius, positions, velocities, drawArgs, maxDiffuse: d.maxDiffuseParticles, diffuse, diffuseDrawArgs };

    const common = /* wgsl */ `
${worldUniformsWGSL}
${couplingWGSL}
@group(0) @binding(0) var<uniform> world: WorldUniforms;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> overflow: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> deposit: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> events: SimEvents;
@group(0) @binding(6) var<storage, read_write> cursor: array<atomic<u32>, 4>;
@group(0) @binding(7) var bedTex: texture_2d<f32>;
@group(0) @binding(8) var waterTex: texture_2d<f32>;
fn hash(n: u32) -> f32 { var x = n * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return f32((x >> 22u) ^ x) / 4294967295.0; }
fn cellOf(xz: vec2f) -> vec2i { return clamp(vec2i(floor(xz * world.grid.w)), vec2i(0), vec2i(world.grid.xy) - 1); }
`;
    const C = GPUShaderStage.COMPUTE;
    const bgl = device.createBindGroupLayout({
      label: 'particlesStub.bgl',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5, 6].map((binding) => ({ binding, visibility: C, buffer: { type: 'storage' as const } })),
        { binding: 7, visibility: C, texture: { sampleType: 'unfilterable-float' as const } },
        { binding: 8, visibility: C, texture: { sampleType: 'unfilterable-float' as const } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    this.emit = computePipeline(
      device,
      'particlesStub.emit',
      `${common}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = u32(world.grid.x) * u32(world.grid.y);
  if (g.x >= n) { return; }
  var o = overflow[g.x];
  let pv = world.particle.y;
  if (o.x < pv) { return; }
  let count = min(u32(o.x / pv), 8u);
  let u = o.y / max(o.x, 1e-12);
  let i = g.x % u32(world.grid.x);
  let j = g.x / u32(world.grid.x);
  for (var k = 0u; k < count; k++) {
    let slot = atomicAdd(&cursor[0], 1u) % u32(world.particle.z);
    let r = vec2f(hash(slot * 3u + u32(world.time.z)), hash(slot * 7u + 11u));
    let xz = (vec2f(f32(i), f32(j)) + vec2f(1.0, r.y)) * world.grid.z;
    pos[slot] = vec4f(xz.x, o.w - world.particle.x * (0.5 + 2.0 * r.x), xz.y, 1.0);
    vel[slot] = vec4f(u, 0.0, (r.x - 0.5) * 0.05, 0.0);
  }
  o.x -= f32(count) * pv;
  o.y -= f32(count) * pv * u;
  overflow[g.x] = o;
}`,
      'main',
      layout,
    );
    this.update = computePipeline(
      device,
      'particlesStub.update',
      `${common}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= u32(world.particle.z)) { return; }
  var p = pos[g.x];
  if (p.w <= 0.0) { return; }
  var v = vel[g.x];
  let dt = world.time.y;
  v.y -= world.physics.x * dt;
  p = vec4f(p.xyz + v.xyz * dt, p.w);
  let c = cellOf(p.xz);
  let b = textureLoad(bedTex, c, 0).r;
  let w = textureLoad(waterTex, c, 0);
  let out = p.x < 0.0 || p.z < 0.0 || p.x > world.domainSize.x || p.z > world.domainSize.y || p.y < world.domainSize.z;
  if (p.y < b + w.x || out) {
    if (!out) {
      let idx = u32(c.y) * u32(world.grid.x) + u32(c.x);
      let vol = world.particle.y * p.w;
      atomicAdd(&deposit[idx * 4u + 0u], i32(vol * DEPOSIT_VOLUME_SCALE));
      atomicAdd(&deposit[idx * 4u + 1u], i32(vol * v.x * DEPOSIT_MOMENTUM_SCALE));
      atomicAdd(&deposit[idx * 4u + 2u], i32(vol * v.z * DEPOSIT_MOMENTUM_SCALE));
      let e = atomicAdd(&events.count, 1u);
      if (e < MAX_IMPACT_EVENTS) {
        events.events[e].posKind = vec4f(p.xyz, f32(select(IMPACT_SOLID, IMPACT_POOL, w.x > 0.001)));
        events.events[e].velVol = vec4f(v.xyz, vol);
      }
    }
    p.w = 0.0;
  }
  pos[g.x] = p;
  vel[g.x] = v;
}`,
      'main',
      layout,
    );
    this.clearEvents = computePipeline(
      device,
      'particlesStub.clearEvents',
      `${common}
@compute @workgroup_size(1)
fn main() { atomicStore(&events.count, 0u); atomicStore(&events.droppedCount, 0u); atomicStore(&events.droppedVolume, 0u); }`,
      'main',
      layout,
    );
    this.bg = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: world.uniforms } },
        { binding: 1, resource: { buffer: positions } },
        { binding: 2, resource: { buffer: velocities } },
        { binding: 3, resource: { buffer: world.overflow } },
        { binding: 4, resource: { buffer: world.deposit } },
        { binding: 5, resource: { buffer: world.simEvents } },
        { binding: 6, resource: { buffer: this.cursor } },
        { binding: 7, resource: world.views.bed },
        { binding: 8, resource: world.views.water },
      ],
    });
  }

  step(encoder: GPUCommandEncoder, frame: FrameContext) {
    const d = this.ctx.world.domain;
    const run = (p: GPUComputePipeline, n: number) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, this.bg);
      pass.dispatchWorkgroups(n);
      pass.end();
    };
    run(this.clearEvents, 1);
    if (frame.dt <= 0) return;
    run(this.emit, Math.ceil((d.nx * d.nz) / 64));
    run(this.update, Math.ceil(d.maxParticles / 64));
  }

  reset() {}

  destroy() {
    this.view.positions.destroy();
    this.view.velocities.destroy();
    this.view.drawArgs.destroy();
    this.view.diffuse.destroy();
    this.view.diffuseDrawArgs.destroy();
    this.cursor.destroy();
  }
}
