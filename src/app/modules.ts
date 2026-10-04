/**
 * Module contracts. Every subsystem implements one of these interfaces; App wires
 * them together and calls them in a fixed order each frame (see App.frame()).
 */
import type { Camera } from '../core/camera';
import type { GpuContext } from '../gpu/device';
import type { GpuProfiler } from '../gpu/util';
import type { RenderTargets } from '../render/frame';
import type { RendererShared } from '../render/Renderer';
import type { SceneModel } from '../world/scene';
import type { World } from '../world/World';
import type { Params, ParamStore } from './params';
import type { ScenePreset } from '../terrain/presets';
import type { EnvRenderer } from '../render/env/EnvRenderer';

export interface FrameContext {
  /** Simulation time step for this frame (s) — 0 when paused. Already multiplied by timeScale. */
  dt: number;
  /** Wall-clock frame time (s). */
  realDt: number;
  /** Accumulated simulation time (s). */
  time: number;
  realTime: number;
  frameIndex: number;
  camera: Camera;
  params: Params;
  world: World;
  targets: RenderTargets;
  /** @group(0) bind group (frameBindingsWGSL). */
  frameBindGroup: GPUBindGroup;
  /** @group(1) bind group (lightingBindingsWGSL). */
  lightingBindGroup: GPUBindGroup;
  frameUniforms: GPUBuffer;
  profiler: GpuProfiler;
}

/** Everything a module may need at construction. */
export interface ModuleContext {
  gpu: GpuContext;
  params: ParamStore;
  scene: SceneModel;
  world: World;
  shared: RendererShared;
  preset: ScenePreset;
}

export interface RayHit {
  point: [number, number, number];
  normal: [number, number, number];
  distance: number;
  /** Obstacle id if an obstacle was hit, else undefined (terrain). */
  obstacleId?: number;
}

/** Terrain, obstacles and their GPU representations (bed heightfield + SDF) + their rendering. */
export interface TerrainSystem {
  /** Re-bakes dirty regions of world.bed / world.sdf after scene edits; updates wetness. */
  update(encoder: GPUCommandEncoder, frame: FrameContext): void;
  /** CPU ray cast against terrain + obstacles (for editing / camera focus). */
  raycast(origin: readonly number[], dir: readonly number[], maxDistance?: number): RayHit | null;
  /** CPU bed height including obstacles (m). */
  heightAt(x: number, z: number): number;
  /** Depth-only draw into the sun shadow map (FrameUniforms.shadowViewProj, standard Z 'less'). */
  drawShadow(pass: GPURenderPassEncoder, frame: FrameContext): void;
  /** Draws terrain + obstacles into the opaque pass (targets: hdr rgba16float, normal rgba16float; reversed-Z depth). */
  drawOpaque(pass: GPURenderPassEncoder, frame: FrameContext): void;
  /** True while bed/SDF are being re-baked (sims may want to know). */
  readonly bedVersion: number;
  destroy(): void;
}

/** Shallow-water (heightfield) simulation of the stream. */
export interface WaterSim {
  /** Advances the simulation by frame.dt seconds (internally sub-stepped). */
  step(encoder: GPUCommandEncoder, frame: FrameContext): void;
  /** Re-initialise water from the preset (initial pools) and clear dynamic state. */
  reset(): void;
  /** Latest async-readback statistics (may lag a few frames). */
  readonly stats: Readonly<Record<string, number>>;
  destroy(): void;
}

export interface ParticleView {
  maxParticles: number;
  /** Rendering radius of a falling-water particle (m). */
  radius: number;
  /** array<vec4f>: xyz world position, w > 0 alive (w = particle volume scale, 1 = nominal), w <= 0 dead. */
  positions: GPUBuffer;
  /** array<vec4f>: xyz velocity (m/s), w = aeration / whiteness 0..1. */
  velocities: GPUBuffer;
  /** Indirect draw args for instanced quads: [vertexCount=6, instanceCount, 0, 0]. instanceCount = high-water mark; skip dead particles. */
  drawArgs: GPUBuffer;
  maxDiffuse: number;
  /** array<DiffuseParticle>: { posLife: vec4f (xyz, remaining life s; <= 0 dead), velType: vec4f (xyz vel, type 0 spray / 1 foam / 2 bubble / 3 mist) }. */
  diffuse: GPUBuffer;
  /** Indirect draw args for diffuse particles [6, instanceCount, 0, 0]. */
  diffuseDrawArgs: GPUBuffer;
}

/** Falling / free-surface water particles (over-edge flow, jets, trickles, splashes, spray). */
export interface ParticleSim {
  step(encoder: GPUCommandEncoder, frame: FrameContext): void;
  reset(): void;
  readonly view: ParticleView;
  readonly stats: Readonly<Record<string, number>>;
  destroy(): void;
}

export interface EnvModule {
  readonly envCube: GPUTexture;
  readonly shadowMapView: GPUTextureView;
  readonly shadowSampler: GPUSampler;
  readonly envSampler: GPUSampler;
  readonly shadowViewProj: Float32Array;
  update(encoder: GPUCommandEncoder, frame: FrameContext): void;
  beginShadowPass(encoder: GPUCommandEncoder): GPURenderPassEncoder;
  drawSky(pass: GPURenderPassEncoder, frame: FrameContext): void;
  post(encoder: GPUCommandEncoder, frame: FrameContext, target: GPUTextureView): void;
  resize(w: number, h: number): void;
  destroy(): void;
}

export interface WaterRendererModule {
  /** Writes world.caustics (rgba16float irradiance multiplier on the bed, 1 = none). */
  computeCaustics(encoder: GPUCommandEncoder, frame: FrameContext): void;
  /** Renders the heightfield water surface into targets.hdr (loads hdr/depth; sceneColor + opaqueDepth hold the opaque scene). */
  draw(encoder: GPUCommandEncoder, frame: FrameContext): void;
  resize(w: number, h: number): void;
  destroy(): void;
}

export interface ParticleRendererModule {
  /** Renders falling water + diffuse spray/foam/bubbles into targets.hdr (sceneColor holds the scene incl. water surface). */
  draw(encoder: GPUCommandEncoder, frame: FrameContext): void;
  resize(w: number, h: number): void;
  destroy(): void;
}

export interface AudioModule {
  /** Must be called from a user gesture (browser autoplay policy). Idempotent. */
  start(): Promise<void>;
  readonly running: boolean;
  /** Called every frame after the sims were encoded: encode GPU→CPU copies of world.simEvents / world.sweStats. */
  encodeReadback(encoder: GPUCommandEncoder, frame: FrameContext): void;
  /** Called after queue.submit of the frame's command buffer. */
  afterSubmit(frame: FrameContext): void;
  destroy(): void;
}

export interface EditorModule {
  /** Per-frame update (hover, drag). */
  update(frame: FrameContext): void;
  /** LDR overlay (selection outline, gizmos, placement preview) drawn on the swapchain after post. */
  drawOverlay(encoder: GPUCommandEncoder, frame: FrameContext, target: GPUTextureView): void;
  destroy(): void;
}

export interface Modules {
  terrain: TerrainSystem;
  water: WaterRendererModule;
  swe: WaterSim;
  particles: ParticleSim;
  particlesRenderer: ParticleRendererModule;
  env: EnvRenderer;
  audio: AudioModule;
  editor: EditorModule;
}
