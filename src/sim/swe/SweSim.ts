/**
 * Shallow-water heightfield simulation of the stream.
 *
 * STUB IMPLEMENTATION — produces a plausible static water state so other modules can
 * be developed against the contract. To be replaced by the SWE agent (keep the WaterSim API).
 *
 * Contract (see docs/ARCHITECTURE.md):
 *   writes world.water   (h, ux, uz, foam)       every frame
 *   writes world.waterAux(turbulence, aeration, divergence, 0)
 *   consumes + clears world.deposit (particles → SWE)
 *   accumulates world.overflow (SWE → particles: water leaving over drops)
 *   writes world.sweStats
 */
import type { FrameContext, ModuleContext, WaterSim } from '../../app/modules';
import { computePipeline } from '../../gpu/util';
import { couplingWGSL, worldUniformsWGSL } from '../../world/wgsl';

export class StubSweSim implements WaterSim {
  stats: Record<string, number> = {};
  private pipeline: GPUComputePipeline;
  private bindGroup: GPUBindGroup;
  private params: GPUBuffer;

  constructor(private ctx: ModuleContext) {
    const { gpu, world, preset } = ctx;
    const device = gpu.device;
    // struct P { count: vec4f, boxes: array<vec4f, 8>, levels: array<vec4f, 2> } = 44 floats
    this.params = device.createBuffer({ size: 44 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const iw = preset.initialWater.slice(0, 8);
    const data = new Float32Array(44);
    data[0] = iw.length;
    iw.forEach((w, i) => {
      data.set([w.x0, w.z0, w.x1, w.z1], 4 + i * 4);
      data[36 + i] = w.level;
    });
    device.queue.writeBuffer(this.params, 0, data);

    this.pipeline = computePipeline(
      device,
      'sweStub',
      /* wgsl */ `
${worldUniformsWGSL}
${couplingWGSL}
struct P { count: vec4f, boxes: array<vec4f, 8>, levels: array<vec4f, 2> }
@group(0) @binding(0) var<uniform> world: WorldUniforms;
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var bedTex: texture_2d<f32>;
@group(0) @binding(3) var waterOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> deposit: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> overflow: array<vec4f>;
fn bedAt(c: vec2i) -> f32 {
  let n = vec2i(i32(world.grid.x), i32(world.grid.y));
  return textureLoad(bedTex, clamp(c, vec2i(0), n - 1), 0).r;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = vec2u(u32(world.grid.x), u32(world.grid.y));
  if (any(g.xy >= n)) { return; }
  let c = vec2i(g.xy);
  let idx = g.y * n.x + g.x;
  let xz = (vec2f(c) + 0.5) * world.grid.z;
  let b = bedAt(c);
  var level = -1e9;
  for (var i = 0u; i < u32(p.count.x); i++) {
    let bx = p.boxes[i];
    if (xz.x >= bx.x && xz.y >= bx.y && xz.x <= bx.z && xz.y <= bx.w) { level = max(level, p.levels[i / 4u][i % 4u]); }
  }
  var h = max(level - b, 0.0);
  // Thin upstream sheet so the ledge has water.
  let bNext = bedAt(c + vec2i(4, 0));
  let up = b > 0.2 && abs(xz.y - world.domainSize.y * 0.5) < 0.33;
  if (up) { h = max(h, 0.02); }
  let u = select(0.0, world.flow.y, h > 0.0);
  textureStore(waterOut, c, vec4f(h, u, 0.0, 0.0));
  // Clear deposits (consumed).
  atomicStore(&deposit[idx * 4u + 0u], 0); atomicStore(&deposit[idx * 4u + 1u], 0);
  atomicStore(&deposit[idx * 4u + 2u], 0); atomicStore(&deposit[idx * 4u + 3u], 0);
  // Overflow at the lip: this cell is wet and the bed drops sharply downstream.
  if (up && b - bNext > 0.05) {
    let dt = world.time.y;
    let q = h * u * world.grid.z * dt; // m^3 crossing this cell's downstream face this frame
    var o = overflow[idx];
    o = vec4f(o.x + q, o.y + q * u, o.z, b + h);
    overflow[idx] = o;
  }
}
`,
    );
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: world.uniforms } },
        { binding: 1, resource: { buffer: this.params } },
        { binding: 2, resource: world.views.bed },
        { binding: 3, resource: world.views.water },
        { binding: 4, resource: { buffer: world.deposit } },
        { binding: 5, resource: { buffer: world.overflow } },
      ],
    });
  }

  step(encoder: GPUCommandEncoder, _frame: FrameContext) {
    const d = this.ctx.world.domain;
    const pass = encoder.beginComputePass({ label: 'sweStub' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(d.nx / 8), Math.ceil(d.nz / 8));
    pass.end();
  }

  reset() {}

  destroy() {
    this.params.destroy();
  }
}
