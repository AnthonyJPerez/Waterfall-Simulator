/**
 * World = the shared GPU resource registry for one simulation domain.
 *
 * Producers / consumers (see docs/ARCHITECTURE.md for the full contract):
 *   bed        r32float [nx, nz]            produced by terrain (base terrain ∪ obstacle tops), read by everyone
 *   sdf        rgba16float 3D [sdfN*]       produced by terrain (distance, outward normal), read by particles / renderers
 *   water      rgba32float [nx, nz]         produced by SWE: (depth h, velocity ux, velocity uz, foam 0..1)
 *   waterAux   rgba16float [nx, nz]         produced by SWE: (turbulence 0..1, aeration 0..1, surface divergence, unused)
 *   wetness    rgba8unorm [nx, nz]          produced by terrain: (wetness 0..1, splash wetness 0..1, unused, unused)
 *   caustics   rgba16float [cnx, cnz]       produced by water renderer: caustic irradiance multiplier on the bed (1 = none)
 *   deposit    atomic<i32> x4 per cell      particles → SWE: (volume, momentum x, momentum z, energy) fixed-point
 *   overflow   vec4f per cell               SWE → particles: (pending volume m³, momentum x, momentum z, lip surface y)
 *   simEvents  SimEvents                    particles → audio: impact list
 *   sweStats   SweStats                     SWE → audio: per-tile flow statistics
 *   uniforms   WorldUniforms                App → everyone, updated every frame
 */
import type { Params } from '../app/params';
import type { Domain } from './domain';
import { createStorageBuffer, createUniformBuffer, StructWriter } from '../gpu/util';
import { SIM_EVENTS_BYTES, SWE_STATS_BYTES, WORLD_UNIFORMS_BYTES } from './wgsl';

export interface WorldTextures {
  bed: GPUTexture;
  sdf: GPUTexture;
  water: GPUTexture;
  waterAux: GPUTexture;
  wetness: GPUTexture;
  caustics: GPUTexture;
}

export class World {
  readonly textures: WorldTextures;
  readonly views: Record<keyof WorldTextures, GPUTextureView>;
  readonly deposit: GPUBuffer;
  readonly overflow: GPUBuffer;
  readonly simEvents: GPUBuffer;
  readonly sweStats: GPUBuffer;
  readonly uniforms: GPUBuffer;
  readonly causticsSize: [number, number];
  private writer = new StructWriter(WORLD_UNIFORMS_BYTES);
  simTime = 0;
  frameIndex = 0;

  constructor(readonly device: GPUDevice, readonly domain: Domain) {
    const { nx, nz } = domain;
    const tex2d = (label: string, format: GPUTextureFormat, w = nx, h = nz, extra: GPUTextureUsageFlags = 0) =>
      device.createTexture({
        label,
        size: [w, h],
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | extra,
      });
    const cScale = Math.min(2, Math.floor(2048 / Math.max(nx, nz)) || 1);
    this.causticsSize = [nx * cScale, nz * cScale];
    this.textures = {
      bed: tex2d('world.bed', 'r32float'),
      sdf: device.createTexture({
        label: 'world.sdf',
        size: [domain.sdfNx, domain.sdfNy, domain.sdfNz],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      }),
      water: tex2d('world.water', 'rgba32float'),
      waterAux: tex2d('world.waterAux', 'rgba16float'),
      wetness: tex2d('world.wetness', 'rgba8unorm'),
      caustics: tex2d('world.caustics', 'rgba16float', this.causticsSize[0], this.causticsSize[1], GPUTextureUsage.RENDER_ATTACHMENT),
    };
    this.views = Object.fromEntries(Object.entries(this.textures).map(([k, t]) => [k, t.createView({ label: `world.${k}.view` })])) as any;

    const cells = nx * nz;
    this.deposit = createStorageBuffer(device, 'world.deposit', cells * 16);
    this.overflow = createStorageBuffer(device, 'world.overflow', cells * 16);
    this.simEvents = createStorageBuffer(device, 'world.simEvents', SIM_EVENTS_BYTES);
    this.sweStats = createStorageBuffer(device, 'world.sweStats', SWE_STATS_BYTES);
    this.uniforms = createUniformBuffer(device, 'world.uniforms', WORLD_UNIFORMS_BYTES);

    // Caustics default to 1 (no modulation) until the water renderer writes them.
    const ones = new Uint16Array(this.causticsSize[0] * this.causticsSize[1] * 4).fill(0x3c00);
    device.queue.writeTexture({ texture: this.textures.caustics }, ones, { bytesPerRow: this.causticsSize[0] * 8 }, this.causticsSize);
  }

  /** Volume of one falling-water particle (m³): a cube of side 2r (particles are packed at spacing 2r). */
  get particleVolume() {
    const d = 2 * this.domain.particleRadius;
    return d * d * d;
  }

  updateUniforms(params: Params, frameDt: number) {
    const d = this.domain;
    const w = this.writer;
    w.setVec4(0, d.sizeX, d.sizeZ, d.minY, d.maxY);
    w.setVec4(4, d.nx, d.nz, d.cellSize, 1 / d.cellSize);
    w.setVec4(8, d.sdfNx, d.sdfNy, d.sdfNz, d.sdfVoxel);
    w.setVec4(12, params.flow.rate / 1000, params.flow.velocity, params.flow.turbulence, params.flow.roughness);
    w.setVec4(16, 9.81, 1000, 1.0e-6, 0.072);
    w.setVec4(20, this.simTime, frameDt, this.frameIndex, params.sim.timeScale);
    w.setVec4(24, params.sim.cohesion, params.sim.adhesion, params.water.foam, params.water.ripples);
    w.setVec4(28, d.particleRadius, this.particleVolume, d.maxParticles, d.maxDiffuseParticles);
    w.upload(this.device, this.uniforms);
  }

  destroy() {
    Object.values(this.textures).forEach((t) => t.destroy());
    [this.deposit, this.overflow, this.simEvents, this.sweStats, this.uniforms].forEach((b) => b.destroy());
  }
}
