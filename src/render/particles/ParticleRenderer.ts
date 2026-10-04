/**
 * Falling-water particle rendering (screen-space fluid surface) + spray/foam/bubbles.
 *
 * STUB IMPLEMENTATION — draws each particle as a lit sphere impostor. To be replaced by the
 * particle-rendering agent (keep the ParticleRendererModule API).
 */
import type { FrameContext, ModuleContext, ParticleRendererModule, ParticleSim } from '../../app/modules';
import { shaderModule } from '../../gpu/util';
import { DEPTH_FORMAT, frameBindingsWGSL, HDR_FORMAT, lightingBindingsWGSL } from '../frame';
import { lightingFunctionsWGSL } from '../env/lighting';

export class StubParticleRenderer implements ParticleRendererModule {
  private pipeline: GPURenderPipeline;
  private bg: GPUBindGroup;

  constructor(private ctx: ModuleContext, private sim: ParticleSim) {
    const device = ctx.gpu.device;
    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.bg = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: sim.view.positions } },
        { binding: 1, resource: { buffer: sim.view.velocities } },
      ],
    });
    const module = shaderModule(
      device,
      'particlesStub.draw',
      /* wgsl */ `
${frameBindingsWGSL}
${lightingBindingsWGSL}
${lightingFunctionsWGSL}
@group(2) @binding(0) var<storage, read> pos: array<vec4f>;
@group(2) @binding(1) var<storage, read> vel: array<vec4f>;
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) center: vec3f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = pos[ii];
  var o: VOut;
  if (p.w <= 0.0) { o.pos = vec4f(0.0, 0.0, -1.0, 1.0); return o; }
  let corners = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let c = corners[vi];
  let r = world.particle.x * 1.3;
  let right = vec3f(frame.view[0][0], frame.view[1][0], frame.view[2][0]);
  let up = vec3f(frame.view[0][1], frame.view[1][1], frame.view[2][1]);
  let wp = p.xyz + (right * c.x + up * c.y) * r;
  o.pos = frame.viewProj * vec4f(wp, 1.0);
  o.uv = c;
  o.center = p.xyz;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let right = vec3f(frame.view[0][0], frame.view[1][0], frame.view[2][0]);
  let up = vec3f(frame.view[0][1], frame.view[1][1], frame.view[2][1]);
  let fwd = -vec3f(frame.view[0][2], frame.view[1][2], frame.view[2][2]);
  let n = normalize(right * in.uv.x + up * in.uv.y - fwd * sqrt(1.0 - r2));
  let v = -fwd;
  let r = reflect(-v, n);
  let fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let col = mix(vec3f(0.5, 0.6, 0.62) * ambientIrradiance(n) * 0.4, envRadiance(r, 0.05), fres)
          + sunIrradiance() * pow(max(dot(r, frame.sunDir.xyz), 0.0), 200.0) * 4.0;
  return vec4f(col, 1.0);
}
`,
    );
    this.pipeline = device.createRenderPipeline({
      label: 'particlesStub.draw',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ctx.shared.frameLayout, ctx.shared.lightingLayout, bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'greater', depthWriteEnabled: true },
      primitive: { topology: 'triangle-list' },
    });
  }

  draw(encoder: GPUCommandEncoder, frame: FrameContext) {
    const t = frame.targets;
    const pass = encoder.beginRenderPass({
      label: 'particlesStub',
      colorAttachments: [{ view: t.hdrView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: t.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frame.frameBindGroup);
    pass.setBindGroup(1, frame.lightingBindGroup);
    pass.setBindGroup(2, this.bg);
    pass.drawIndirect(this.sim.view.drawArgs, 0);
    pass.end();
  }

  resize() {}
  destroy() {}
}
