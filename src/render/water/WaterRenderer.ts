/**
 * Heightfield water surface rendering + caustics.
 *
 * STUB IMPLEMENTATION — a basic refractive/reflective surface; caustics are left at 1.
 * To be replaced by the water-rendering agent (keep the WaterRendererModule API).
 */
import type { FrameContext, ModuleContext, WaterRendererModule } from '../../app/modules';
import { shaderModule } from '../../gpu/util';
import { DEPTH_FORMAT, frameBindingsWGSL, HDR_FORMAT, lightingBindingsWGSL } from '../frame';
import { lightingFunctionsWGSL } from '../env/lighting';

export class StubWaterRenderer implements WaterRendererModule {
  private pipeline: GPURenderPipeline;
  private indexBuf: GPUBuffer;
  private indexCount: number;
  private bgl: GPUBindGroupLayout;
  private bg?: GPUBindGroup;
  private bgFor?: GPUTexture;

  constructor(private ctx: ModuleContext) {
    const device = ctx.gpu.device;
    const d = ctx.world.domain;
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

    this.bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const module = shaderModule(
      device,
      'waterStub',
      /* wgsl */ `
${frameBindingsWGSL}
${lightingBindingsWGSL}
${lightingFunctionsWGSL}
@group(2) @binding(0) var sceneColor: texture_2d<f32>;
@group(2) @binding(1) var opaqueDepth: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4f, @location(0) world: vec3f, @location(1) depth: f32 }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let nx = u32(world.grid.x);
  let c = vec2i(i32(vi % nx), i32(vi / nx));
  let xz = (vec2f(c) + 0.5) * world.grid.z;
  let w = waterLoad(c);
  let b = bedAt(c);
  let y = select(b - 0.01, b + w.x, w.x > 0.0005);
  var o: VOut;
  o.world = vec3f(xz.x, y, xz.y);
  o.pos = frame.viewProj * vec4f(o.world, 1.0);
  o.depth = w.x;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  if (in.depth < 0.0005) { discard; }
  let e = world.grid.z;
  let hx = (bedHeight(in.world.xz + vec2f(e, 0.0)) + waterSample(in.world.xz + vec2f(e, 0.0)).x) - (bedHeight(in.world.xz - vec2f(e, 0.0)) + waterSample(in.world.xz - vec2f(e, 0.0)).x);
  let hz = (bedHeight(in.world.xz + vec2f(0.0, e)) + waterSample(in.world.xz + vec2f(0.0, e)).x) - (bedHeight(in.world.xz - vec2f(0.0, e)) + waterSample(in.world.xz - vec2f(0.0, e)).x);
  let n = normalize(vec3f(-hx, 2.0 * e, -hz));
  let v = normalize(frame.cameraPos.xyz - in.world);
  let f0 = 0.02;
  let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let uv = in.pos.xy * frame.screen.zw;
  let od = textureLoad(opaqueDepth, vec2i(in.pos.xy), 0).r;
  let thick = max(linearDepth(od) - linearDepth(in.pos.z), 0.0);
  let refr = textureSampleLevel(sceneColor, linearClamp, clamp(uv + n.xz * 0.03, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
  let trans = exp(-frame.waterAbsorb.rgb * thick);
  let r = reflect(-v, n);
  let refl = envRadiance(r, 0.0);
  let spec = sunIrradiance() * pow(max(dot(r, frame.sunDir.xyz), 0.0), 800.0) * 20.0;
  let col = mix(refr * trans + vec3f(0.02, 0.04, 0.04) * (1.0 - trans), refl, fres) + spec;
  return vec4f(col, 1.0);
}
`,
    );
    this.pipeline = device.createRenderPipeline({
      label: 'waterStub',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ctx.shared.frameLayout, ctx.shared.lightingLayout, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'greater', depthWriteEnabled: true },
      primitive: { topology: 'triangle-list' },
    });
  }

  computeCaustics(_encoder: GPUCommandEncoder, _frame: FrameContext) {}

  draw(encoder: GPUCommandEncoder, frame: FrameContext) {
    const t = frame.targets;
    if (this.bgFor !== t.sceneColor) {
      this.bgFor = t.sceneColor;
      this.bg = this.ctx.gpu.device.createBindGroup({
        layout: this.bgl,
        entries: [
          { binding: 0, resource: t.sceneColorView },
          { binding: 1, resource: t.opaqueDepthView },
        ],
      });
    }
    const pass = encoder.beginRenderPass({
      label: 'waterStub',
      colorAttachments: [{ view: t.hdrView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: t.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, frame.frameBindGroup);
    pass.setBindGroup(1, frame.lightingBindGroup);
    pass.setBindGroup(2, this.bg!);
    pass.setIndexBuffer(this.indexBuf, 'uint32');
    pass.drawIndexed(this.indexCount);
    pass.end();
  }

  resize() {}

  destroy() {
    this.indexBuf.destroy();
  }
}
