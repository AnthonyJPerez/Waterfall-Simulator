/**
 * Module factory: the single place that decides which implementation backs each
 * contract. Module agents swap their stub for the real implementation here.
 */
import type { Camera, OrbitController } from '../core/camera';
import type { ModuleContext, Modules } from './modules';
import { StubTerrainSystem } from '../terrain/TerrainSystem';
import { StubSweSim } from '../sim/swe/SweSim';
import { StubParticleSim } from '../sim/particles/ParticleSim';
import { EnvRenderer } from '../render/env/EnvRenderer';
import { StubWaterRenderer } from '../render/water/WaterRenderer';
import { StubParticleRenderer } from '../render/particles/ParticleRenderer';
import { StubAudioEngine } from '../audio/AudioEngine';
import { ObstacleEditor } from '../editor/Editor';

export interface HostContext {
  canvas: HTMLCanvasElement;
  camera: Camera;
  orbit: OrbitController;
}

export function createModules(ctx: ModuleContext, host: HostContext): Modules {
  const terrain = new StubTerrainSystem(ctx);
  const swe = new StubSweSim(ctx);
  const particles = new StubParticleSim(ctx);
  const env = new EnvRenderer(ctx.gpu.device, ctx.shared);
  const water = new StubWaterRenderer(ctx);
  const particlesRenderer = new StubParticleRenderer(ctx, particles);
  const audio = new StubAudioEngine(ctx);
  const editor = new ObstacleEditor(ctx, host.canvas, host.camera, host.orbit, terrain);
  return { terrain, swe, particles, env, water, particlesRenderer, audio, editor };
}
