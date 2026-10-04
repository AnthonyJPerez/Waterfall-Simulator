/**
 * Environment module: sky, sun, environment cube for reflections, sun shadow map
 * and the final post-processing chain (tonemap → swapchain).
 *
 * STUB IMPLEMENTATION — to be replaced by the env agent (keep the public API).
 */
import { mat4, vec3 } from 'wgpu-matrix';
import type { FrameContext } from '../../app/modules';
import { computePipeline, shaderModule } from '../../gpu/util';
import { DEPTH_FORMAT, frameBindingsWGSL, HDR_FORMAT, lightingBindingsWGSL, NORMAL_FORMAT } from '../frame';
import { lightingFunctionsWGSL } from './lighting';
import type { RendererShared } from '../Renderer';

const ENV_SIZE = 64;
const ENV_MIPS = 6;
const SHADOW_SIZE = 2048;

export class EnvRenderer {
  readonly envCube: GPUTexture;
  readonly shadowMap: GPUTexture;
  readonly shadowMapView: GPUTextureView;
  readonly shadowSampler: GPUSampler;
  readonly envSampler: GPUSampler;
  /** Sun-space view-projection (standard Z, orthographic) — written into FrameUniforms.shadowViewProj. */
  readonly shadowViewProj = mat4.identity();
  private envPipeline: GPUComputePipeline;
  private envDirty = true;
  private lastLightKey = '';
  private skyPipeline: GPURenderPipeline;
  private postPipeline: GPURenderPipeline;
  private postBindGroup?: GPUBindGroup;
  private postBindGroupFor?: GPUTexture;
  private envUniform: GPUBuffer;

  constructor(private device: GPUDevice, private shared: RendererShared) {
    this.envCube = device.createTexture({
      label: 'env.cube',
      size: [ENV_SIZE, ENV_SIZE, 6],
      mipLevelCount: ENV_MIPS,
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    this.shadowMap = device.createTexture({
      label: 'env.shadowMap',
      size: [SHADOW_SIZE, SHADOW_SIZE],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowMapView = this.shadowMap.createView();
    this.shadowSampler = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
    this.envSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    this.envUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.envPipeline = computePipeline(
      device,
      'env.fillCube',
      /* wgsl */ `
${frameBindingsWGSL.replace(/@group\(0\) @binding\((\d+)\)/g, (_m, b) => `@group(0) @binding(${b})`)}
@group(1) @binding(0) var dst: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(1) var<uniform> mip: vec4u;
// minimal sky copy for the stub (the real env module prefilters properly)
fn sky(dir: vec3f) -> vec3f {
  let d = normalize(dir);
  let h = d.y;
  var c = mix(vec3f(0.62, 0.70, 0.78), vec3f(0.18, 0.32, 0.62), pow(max(h, 0.0), 0.45));
  c = mix(c, vec3f(0.10, 0.12, 0.07), smoothstep(0.02, -0.2, h));
  let mu = max(dot(d, frame.sunDir.xyz), 0.0);
  c += vec3f(1.0, 0.85, 0.6) * (0.25 * pow(mu, 8.0) + 2.0 * pow(mu, 256.0));
  return c * 3.0;
}
fn cubeDir(face: u32, uv: vec2f) -> vec3f {
  let a = uv * 2.0 - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -a.y, -a.x); }
    case 1u: { return vec3f(-1.0, -a.y, a.x); }
    case 2u: { return vec3f(a.x, 1.0, a.y); }
    case 3u: { return vec3f(a.x, -1.0, -a.y); }
    case 4u: { return vec3f(a.x, -a.y, 1.0); }
    default: { return vec3f(-a.x, -a.y, -1.0); }
  }
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let size = textureDimensions(dst).x;
  if (g.x >= size || g.y >= size) { return; }
  let uv = (vec2f(g.xy) + 0.5) / f32(size);
  textureStore(dst, g.xy, g.z, vec4f(sky(cubeDir(g.z, uv)), 1.0));
}
`,
    );

    const skyModule = shaderModule(
      device,
      'env.sky',
      /* wgsl */ `
${frameBindingsWGSL}
${lightingBindingsWGSL}
${lightingFunctionsWGSL}
struct VOut { @builtin(position) pos: vec4f, @location(0) ndc: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VOut; o.pos = vec4f(p, 0.0, 1.0); o.ndc = p; return o;
}
struct FOut { @location(0) color: vec4f, @location(1) normal: vec4f }
@fragment fn fs(in: VOut) -> FOut {
  let w = frame.invViewProj * vec4f(in.ndc, 0.5, 1.0);
  let dir = normalize(w.xyz / w.w - frame.cameraPos.xyz);
  var o: FOut;
  o.color = vec4f(skyRadiance(dir), 1.0);
  o.normal = vec4f(0.0);
  return o;
}
`,
    );
    this.skyPipeline = device.createRenderPipeline({
      label: 'env.sky',
      layout: device.createPipelineLayout({ bindGroupLayouts: [shared.frameLayout, shared.lightingLayout] }),
      vertex: { module: skyModule, entryPoint: 'vs' },
      fragment: { module: skyModule, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }, { format: NORMAL_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'greater-equal', depthWriteEnabled: false },
      primitive: { topology: 'triangle-list' },
    });

    const postModule = shaderModule(
      device,
      'env.post',
      /* wgsl */ `
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> post: vec4f; // exposure
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VOut; o.pos = vec4f(p, 0.0, 1.0); return o;
}
fn aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2i(in.pos.xy), 0).rgb * post.x;
  let m = aces(c);
  return vec4f(pow(m, vec3f(1.0 / 2.2)), 1.0);
}
`,
    );
    this.postPipeline = device.createRenderPipeline({
      label: 'env.post',
      layout: 'auto',
      vertex: { module: postModule, entryPoint: 'vs' },
      fragment: { module: postModule, entryPoint: 'fs', targets: [{ format: shared.presentationFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** Updates sun shadow matrix and regenerates the environment cube when lighting changes. */
  update(encoder: GPUCommandEncoder, frame: FrameContext) {
    const d = frame.world.domain;
    const sun = this.shared.sunDir;
    const center = [d.sizeX / 2, (d.minY + d.maxY) / 2, d.sizeZ / 2];
    const radius = Math.hypot(d.sizeX, d.sizeZ, d.maxY - d.minY) * 0.5 + 0.2;
    const eye = vec3.add(center, vec3.scale(sun, radius * 2));
    const up = Math.abs(sun[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0];
    const view = mat4.lookAt(eye, center, up);
    const proj = mat4.ortho(-radius, radius, -radius, radius, 0.01, radius * 4);
    mat4.multiply(proj, view, this.shadowViewProj);

    const key = `${frame.params.light.timeOfDay}|${frame.params.light.sunAzimuth}|${frame.params.light.cloudiness}`;
    if (key !== this.lastLightKey) {
      this.lastLightKey = key;
      this.envDirty = true;
    }
    if (this.envDirty) {
      this.envDirty = false;
      for (let m = 0; m < ENV_MIPS; m++) {
        const size = Math.max(1, ENV_SIZE >> m);
        const bg = this.device.createBindGroup({
          layout: this.envPipeline.getBindGroupLayout(1),
          entries: [{ binding: 0, resource: this.envCube.createView({ dimension: '2d-array', baseMipLevel: m, mipLevelCount: 1 }) }],
        });
        const frameBg = this.device.createBindGroup({
          layout: this.envPipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: frame.frameUniforms } }],
        });
        const pass = encoder.beginComputePass({ label: `env.cube.mip${m}` });
        pass.setPipeline(this.envPipeline);
        pass.setBindGroup(0, frameBg);
        pass.setBindGroup(1, bg);
        pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), 6);
        pass.end();
      }
    }
  }

  /** Begins the sun shadow-map depth pass. Casters draw with FrameUniforms.shadowViewProj (standard Z, 'less'). */
  beginShadowPass(encoder: GPUCommandEncoder): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      label: 'env.shadow',
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowMapView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
  }

  /** Draws the sky into pixels left at depth 0 (call inside the opaque pass, after opaque geometry). */
  drawSky(pass: GPURenderPassEncoder, frame: FrameContext) {
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, frame.frameBindGroup);
    pass.setBindGroup(1, frame.lightingBindGroup);
    pass.draw(3);
  }

  /** HDR → display. Writes the final image into `target` (swapchain view). */
  post(encoder: GPUCommandEncoder, frame: FrameContext, target: GPUTextureView) {
    if (this.postBindGroupFor !== frame.targets.hdr) {
      this.postBindGroupFor = frame.targets.hdr;
      this.postBindGroup = this.device.createBindGroup({
        layout: this.postPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: frame.targets.hdrView },
          { binding: 1, resource: { buffer: this.envUniform } },
        ],
      });
    }
    this.device.queue.writeBuffer(this.envUniform, 0, new Float32Array([Math.pow(2, frame.params.light.exposure), 0, 0, 0]));
    const pass = encoder.beginRenderPass({
      label: 'env.post',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.postPipeline);
    pass.setBindGroup(0, this.postBindGroup!);
    pass.draw(3);
    pass.end();
  }

  resize(_w: number, _h: number) {}

  destroy() {
    this.envCube.destroy();
    this.shadowMap.destroy();
  }
}
