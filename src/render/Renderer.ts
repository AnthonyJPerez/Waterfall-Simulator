/**
 * Frame graph. Owns the render targets, the shared group(0)/group(1) layouts and
 * the FrameUniforms buffer, and calls the render modules in a fixed order:
 *
 *   1. env.update                 (sky/env cube regeneration, shadow matrix)
 *   2. shadow pass                terrain.drawShadow
 *   3. water.computeCaustics      → world.caustics
 *   4. opaque pass (hdr+normal)   terrain.drawOpaque, env.drawSky
 *   5. copy hdr→sceneColor, depth→opaqueDepth
 *   6. water.draw                 (own passes; loads hdr + depth)
 *   7. copy hdr→sceneColor
 *   8. particles.draw             (own passes; loads hdr + depth)
 *   9. env.post                   → swapchain
 *  10. editor.drawOverlay         → swapchain (LDR, depth-tested against main depth)
 */
import { vec3 } from 'wgpu-matrix';
import type { FrameContext, Modules } from '../app/modules';
import { debugSkip } from '../app/App';
import type { Params } from '../app/params';
import type { Camera } from '../core/camera';
import { createUniformBuffer, shaderModule, StructWriter } from '../gpu/util';
import type { GpuContext } from '../gpu/device';
import type { World } from '../world/World';
import {
  createRenderTargets,
  DEBUG_VIEW_IDS,
  DEPTH_FORMAT,
  destroyRenderTargets,
  FRAME_UNIFORMS_BYTES,
  FU,
  type RenderTargets,
} from './frame';

/** Resources every render module may need at construction time. */
export interface RendererShared {
  device: GPUDevice;
  presentationFormat: GPUTextureFormat;
  frameLayout: GPUBindGroupLayout;
  lightingLayout: GPUBindGroupLayout;
  /** Unit vector towards the sun (updated every frame). */
  sunDir: Float32Array;
}

export function sunDirection(params: Params): Float32Array {
  // Simple solar model: elevation from hour of day (peak ~62° at 13:00), azimuth from params.
  const t = (params.light.timeOfDay - 6.5) / 13; // 0 at sunrise, 1 at sunset
  const elev = Math.max(0.02, Math.sin(Math.PI * Math.min(Math.max(t, 0), 1)) * (62 * Math.PI) / 180);
  const az = (params.light.sunAzimuth * Math.PI) / 180 + (t - 0.5) * 1.2;
  return new Float32Array(vec3.normalize([Math.cos(elev) * Math.cos(az), Math.sin(elev), Math.cos(elev) * Math.sin(az)]));
}

export class Renderer {
  readonly shared: RendererShared;
  targets: RenderTargets;
  readonly frameUniforms: GPUBuffer;
  private writer = new StructWriter(FRAME_UNIFORMS_BYTES);
  private samplers: { linearClamp: GPUSampler; linearRepeat: GPUSampler; nearestClamp: GPUSampler };
  private frameBindGroup?: GPUBindGroup;
  private lightingBindGroup?: GPUBindGroup;
  private bindKey: unknown[] = [];
  private depthCopyPipeline: GPURenderPipeline;
  private depthCopyBindGroup?: GPUBindGroup;

  constructor(private gpu: GpuContext) {
    const device = gpu.device;
    const v = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    const frameLayout = device.createBindGroupLayout({
      label: 'frameLayout',
      entries: [
        { binding: 0, visibility: v, buffer: { type: 'uniform' } },
        { binding: 1, visibility: v, buffer: { type: 'uniform' } },
        { binding: 2, visibility: v, sampler: { type: 'filtering' } },
        { binding: 3, visibility: v, sampler: { type: 'filtering' } },
        { binding: 4, visibility: v, sampler: { type: 'non-filtering' } },
        { binding: 5, visibility: v, texture: { sampleType: 'unfilterable-float' } },
        { binding: 6, visibility: v, texture: { sampleType: 'unfilterable-float' } },
        { binding: 7, visibility: v, texture: { sampleType: 'float' } },
        { binding: 8, visibility: v, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 9, visibility: v, texture: { sampleType: 'float' } },
      ],
    });
    const lightingLayout = device.createBindGroupLayout({
      label: 'lightingLayout',
      entries: [
        { binding: 0, visibility: v, texture: { sampleType: 'float', viewDimension: 'cube' } },
        { binding: 1, visibility: v, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: v, sampler: { type: 'comparison' } },
        { binding: 3, visibility: v, texture: { sampleType: 'float' } },
        { binding: 4, visibility: v, sampler: { type: 'filtering' } },
      ],
    });
    this.shared = {
      device,
      presentationFormat: gpu.presentationFormat,
      frameLayout,
      lightingLayout,
      sunDir: new Float32Array([0.3, 0.8, 0.2]),
    };
    this.samplers = {
      linearClamp: device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' }),
      linearRepeat: device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        addressModeW: 'repeat',
      }),
      nearestClamp: device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' }),
    };
    this.frameUniforms = createUniformBuffer(device, 'frameUniforms', FRAME_UNIFORMS_BYTES);
    this.targets = createRenderTargets(device, Math.max(1, gpu.canvas.width), Math.max(1, gpu.canvas.height));

    const dc = shaderModule(
      device,
      'depthCopy',
      /* wgsl */ `
@group(0) @binding(0) var depthTex: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return vec4f(p, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(depthTex, vec2i(pos.xy), 0), 0.0, 0.0, 1.0);
}
`,
    );
    this.depthCopyPipeline = device.createRenderPipeline({
      label: 'depthCopy',
      layout: 'auto',
      vertex: { module: dc, entryPoint: 'vs' },
      fragment: { module: dc, entryPoint: 'fs', targets: [{ format: 'r32float' }] },
    });
  }

  resize(width: number, height: number) {
    if (width === this.targets.width && height === this.targets.height) return false;
    destroyRenderTargets(this.targets);
    this.targets = createRenderTargets(this.gpu.device, width, height);
    this.depthCopyBindGroup = undefined;
    return true;
  }

  /** Fills FrameUniforms for this frame. */
  updateFrameUniforms(camera: Camera, params: Params, modules: Modules, time: { sim: number; dt: number; real: number; frame: number }) {
    const w = this.writer;
    const sun = sunDirection(params);
    this.shared.sunDir.set(sun);
    w.setMat4(FU.viewProj, camera.viewProj);
    w.setMat4(FU.invViewProj, camera.invViewProj);
    w.setMat4(FU.view, camera.view);
    w.setMat4(FU.proj, camera.proj);
    w.setMat4(FU.invProj, camera.invProj);
    w.setMat4(FU.prevViewProj, camera.prevViewProj);
    w.setMat4(FU.shadowViewProj, modules.env.shadowViewProj);
    w.setMat4(FU.invView, camera.invView);
    w.setVec4(FU.cameraPos, camera.position[0], camera.position[1], camera.position[2], camera.near);
    const f = camera.forward();
    w.setVec4(FU.cameraFwd, f[0], f[1], f[2], camera.far);
    w.setVec4(FU.sunDir, sun[0], sun[1], sun[2], 0.00465);
    // Sun irradiance: reddens and dims near the horizon; clouds attenuate.
    const elev = Math.asin(sun[1]);
    // Kasten–Young air mass approximation.
    const elevDeg = (elev * 180) / Math.PI;
    const air = 1 / (Math.sin(elev) + 0.50572 * Math.pow(elevDeg + 6.07995, -1.6364));
    const ext = (k: number) => Math.exp(-k * Math.min(air, 20));
    const cloud = params.light.cloudiness;
    const sunI = 6.0 * (1 - 0.85 * cloud);
    w.setVec4(FU.sunColor, sunI * ext(0.06), sunI * ext(0.11), sunI * ext(0.22), params.light.canopy);
    const amb = 0.9 + 0.6 * cloud;
    w.setVec4(FU.ambient, 0.42 * amb, 0.55 * amb, 0.75 * amb, cloud);
    w.setVec4(FU.screen, this.targets.width, this.targets.height, 1 / this.targets.width, 1 / this.targets.height);
    w.setVec4(FU.time, time.sim, time.dt, time.real, time.frame);
    w.setVec4(FU.water, params.water.clarity, params.water.turbidity, params.water.foam, params.water.ripples);
    // Optical properties: pure water absorption + tannin/sediment; clarity scales the impurities.
    const impurity = (1 - params.water.clarity) * 1.5 + params.water.turbidity * 2.0;
    w.setVec4(FU.waterAbsorb, 0.45 + 0.4 * impurity, 0.065 + 1.0 * impurity, 0.03 + 2.2 * impurity, 0.05 + 3.0 * impurity);
    w.setVec4(FU.waterScatter, 0.55 + 0.25 * params.water.turbidity, 0.6 + 0.1 * params.water.turbidity, 0.5 - 0.2 * params.water.turbidity, 1.333);
    w.setVec4(FU.post, Math.pow(2, params.light.exposure), DEBUG_VIEW_IDS[params.debug.view] ?? 0, params.camera.dof ? 1 : 0, params.camera.aperture);
    w.setVec4(FU.jitter, 0, 0, 0, 0);
    w.upload(this.gpu.device, this.frameUniforms);
  }

  /** (Re)creates the group(0) and group(1) bind groups when their resources change. */
  private ensureBindGroups(world: World, modules: Modules) {
    const key = [world, modules.env, this.targets];
    if (this.frameBindGroup && key.every((k, i) => k === this.bindKey[i])) return;
    this.bindKey = key;
    const d = this.gpu.device;
    this.frameBindGroup = d.createBindGroup({
      label: 'frameBindGroup',
      layout: this.shared.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameUniforms } },
        { binding: 1, resource: { buffer: world.uniforms } },
        { binding: 2, resource: this.samplers.linearClamp },
        { binding: 3, resource: this.samplers.linearRepeat },
        { binding: 4, resource: this.samplers.nearestClamp },
        { binding: 5, resource: world.views.bed },
        { binding: 6, resource: world.views.water },
        { binding: 7, resource: world.views.waterAux },
        { binding: 8, resource: world.textures.sdf.createView({ dimension: '3d' }) },
        { binding: 9, resource: world.views.wetness },
      ],
    });
    this.lightingBindGroup = d.createBindGroup({
      label: 'lightingBindGroup',
      layout: this.shared.lightingLayout,
      entries: [
        { binding: 0, resource: modules.env.envCube.createView({ dimension: 'cube' }) },
        { binding: 1, resource: modules.env.shadowMapView },
        { binding: 2, resource: modules.env.shadowSampler },
        { binding: 3, resource: world.views.caustics },
        { binding: 4, resource: modules.env.envSampler },
      ],
    });
  }

  /** Builds the per-frame context passed to every module. */
  makeFrameContext(base: Omit<FrameContext, 'targets' | 'frameBindGroup' | 'lightingBindGroup' | 'frameUniforms'>, modules: Modules): FrameContext {
    this.ensureBindGroups(base.world, modules);
    return {
      ...base,
      targets: this.targets,
      frameBindGroup: this.frameBindGroup!,
      lightingBindGroup: this.lightingBindGroup!,
      frameUniforms: this.frameUniforms,
    };
  }

  render(encoder: GPUCommandEncoder, frame: FrameContext, modules: Modules, swapchainView: GPUTextureView) {
    const t = this.targets;
    const skip = debugSkip();
    if (!skip.has('env')) modules.env.update(encoder, frame);

    const shadow = modules.env.beginShadowPass(encoder);
    if (!skip.has('shadow')) modules.terrain.drawShadow(shadow, frame);
    shadow.end();

    if (!skip.has('caustics')) modules.water.computeCaustics(encoder, frame);

    const opaque = encoder.beginRenderPass({
      label: 'opaque',
      colorAttachments: [
        { view: t.hdrView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        { view: t.normalView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: t.depthView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    if (!skip.has('terrainDraw')) modules.terrain.drawOpaque(opaque, frame);
    if (!skip.has('sky')) modules.env.drawSky(opaque, frame);
    opaque.end();

    encoder.copyTextureToTexture({ texture: t.hdr }, { texture: t.sceneColor }, [t.width, t.height]);
    this.copyDepth(encoder);

    if (!skip.has('water')) modules.water.draw(encoder, frame);

    encoder.copyTextureToTexture({ texture: t.hdr }, { texture: t.sceneColor }, [t.width, t.height]);

    if (!skip.has('particlesDraw')) modules.particlesRenderer.draw(encoder, frame);

    modules.env.post(encoder, frame, swapchainView);
    if (!skip.has('editor')) modules.editor.drawOverlay(encoder, frame, swapchainView);
  }

  private copyDepth(encoder: GPUCommandEncoder) {
    const t = this.targets;
    if (!this.depthCopyBindGroup) {
      this.depthCopyBindGroup = this.gpu.device.createBindGroup({
        layout: this.depthCopyPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: t.depthView }],
      });
    }
    const pass = encoder.beginRenderPass({
      label: 'depthCopy',
      colorAttachments: [{ view: t.opaqueDepthView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(this.depthCopyPipeline);
    pass.setBindGroup(0, this.depthCopyBindGroup);
    pass.draw(3);
    pass.end();
  }
}

export { DEPTH_FORMAT };
