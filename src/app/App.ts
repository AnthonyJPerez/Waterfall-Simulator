/**
 * Application shell: owns the GPU context, parameters, scene, world and all
 * modules; runs the frame loop in a fixed order:
 *
 *   input/camera → uniforms → terrain.update → swe.step → particles.step
 *   → audio.encodeReadback → renderer.render (→ swapchain) → submit → audio.afterSubmit
 *
 * Test automation hooks are exposed on window.__wf (see scripts/shot.mjs).
 */
import { Camera, OrbitController } from '../core/camera';
import { initGpu, readTexturePng, type GpuContext } from '../gpu/device';
import { GpuProfiler } from '../gpu/util';
import { Renderer } from '../render/Renderer';
import { makeDomain } from '../world/domain';
import { SceneModel } from '../world/scene';
import { World } from '../world/World';
import { getPreset, type ScenePreset } from '../terrain/presets';
import type { FrameContext, ModuleContext, Modules } from './modules';
import { ParamStore, type Params, type Quality } from './params';
import { createModules } from './registry';
import { Panel } from '../ui/Panel';

/** Debug: comma separated module/pass names to skip, from the `skip` URL parameter (e.g. ?skip=particles,water). */
export function debugSkip(): Set<string> {
  const w = window as any;
  if (!w.__wfSkip) w.__wfSkip = new Set((new URLSearchParams(location.search).get('skip') ?? '').split(',').filter(Boolean));
  return w.__wfSkip;
}

export interface AppOptions {
  /** Fixed simulation dt per frame (s) for deterministic captures; undefined = real time. */
  fixedDt?: number;
  /** Skip building the UI panel. */
  noUi?: boolean;
  /** Render into an offscreen texture instead of the canvas (automation; see readOffscreenPng). */
  offscreen?: boolean;
}

export class App {
  gpu!: GpuContext;
  readonly params = new ParamStore();
  readonly scene = new SceneModel();
  readonly camera = new Camera();
  orbit!: OrbitController;
  renderer!: Renderer;
  world!: World;
  modules!: Modules;
  preset!: ScenePreset;
  panel?: Panel;
  profiler!: GpuProfiler;
  frameIndex = 0;
  simTime = 0;
  private lastTime = 0;
  private realTime = 0;
  private running = false;
  private rebuildPending: 'world' | null = null;
  private fpsAvg = 60;
  private frameWaiters: { n: number; resolve: () => void }[] = [];
  private lastError = '';
  private offscreenTarget?: GPUTexture;
  /** Frames submitted but not yet completed on the GPU (frame pacing, essential in offscreen mode). */
  private inFlight = 0;

  constructor(private canvas: HTMLCanvasElement, private uiContainer: HTMLElement | null, private options: AppOptions = {}) {}

  async init(overrides: Partial<{ preset: string; quality: Quality }> = {}) {
    this.gpu = await initGpu(this.canvas, !!this.options.offscreen);
    if (overrides.preset) this.params.values.scene.preset = overrides.preset;
    this.params.values.sim.quality = overrides.quality ?? (this.gpu.isSoftware ? 'low' : this.params.values.sim.quality);
    this.renderer = new Renderer(this.gpu);
    this.profiler = new GpuProfiler(this.gpu.device, this.gpu.features.timestampQuery);
    this.orbit = new OrbitController(this.camera, this.canvas);
    this.buildWorld(true);

    this.params.on('sim.quality', () => (this.rebuildPending = 'world'));
    this.params.on('camera.fov', () => (this.camera.fovY = (this.params.values.camera.fov * Math.PI) / 180));
    this.camera.fovY = (this.params.values.camera.fov * Math.PI) / 180;

    if (!this.options.noUi && this.uiContainer) {
      this.panel = new Panel(this.uiContainer, this.params, {
        resetWater: () => {
          this.modules.swe.reset();
          this.modules.particles.reset();
        },
        loadPreset: (id) => this.loadPreset(id),
        enableAudio: () => this.enableAudio(),
        clearObstacles: () => [...this.scene.obstacles].forEach((o) => this.scene.remove(o.id)),
      }, this);
    }
    window.addEventListener('resize', () => this.handleResize());
    this.handleResize();
    // Browsers require a user gesture before audio can start.
    const gesture = () => {
      if (this.params.values.audio.enabled) this.enableAudio();
    };
    window.addEventListener('pointerdown', gesture, { once: true });
    window.addEventListener('keydown', gesture, { once: true });
  }

  async enableAudio() {
    try {
      await this.modules.audio.start();
    } catch (e) {
      console.warn('[audio] failed to start', e);
    }
  }

  loadPreset(id: string) {
    this.params.values.scene.preset = id;
    this.rebuildPending = 'world';
  }

  /** (Re)creates the World and all modules for the current preset + quality. */
  private buildWorld(applyPresetDefaults: boolean) {
    const audioWasRunning = this.modules?.audio.running;
    if (this.modules) Object.values(this.modules).forEach((m) => m.destroy());
    this.world?.destroy();

    this.preset = getPreset(this.params.values.scene.preset);
    const domain = makeDomain(this.preset.extent, this.params.values.sim.quality);
    this.world = new World(this.gpu.device, domain);
    this.scene.reset(this.preset.obstacles, this.preset.inflows);
    if (applyPresetDefaults) {
      Object.assign(this.params.values.flow, this.preset.flow);
      this.orbit.setPose(this.preset.camera.position, this.preset.camera.target);
    }
    const ctx: ModuleContext = {
      gpu: this.gpu,
      params: this.params,
      scene: this.scene,
      world: this.world,
      shared: this.renderer.shared,
      preset: this.preset,
    };
    this.modules = createModules(ctx, { canvas: this.canvas, camera: this.camera, orbit: this.orbit });
    this.modules.swe.reset();
    this.simTime = 0;
    if (audioWasRunning) this.enableAudio();
    this.panel?.refresh();
  }

  private handleResize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    if (this.renderer.resize(w, h)) {
      this.modules.env.resize(w, h);
      this.modules.water.resize(w, h);
      this.modules.particlesRenderer.resize(w, h);
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now();
    const loop = () => {
      if (!this.running) return;
      try {
        // Keep at most 2 frames in flight so the GPU queue never builds an unbounded backlog.
        if (this.inFlight < 2) this.frame();
      } catch (e) {
        const msg = String((e as Error)?.stack ?? e);
        if (msg !== this.lastError) console.error('[frame]', e);
        this.lastError = msg;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
  }

  private swapchainView(): GPUTextureView {
    if (!this.gpu.offscreen) return this.gpu.context.getCurrentTexture().createView();
    const { width, height } = this.canvas;
    if (!this.offscreenTarget || this.offscreenTarget.width !== width || this.offscreenTarget.height !== height) {
      this.offscreenTarget?.destroy();
      this.offscreenTarget = this.gpu.device.createTexture({
        label: 'offscreenTarget',
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
      });
    }
    return this.offscreenTarget.createView();
  }

  /** PNG data URL of the last offscreen frame (offscreen mode only). */
  async readOffscreenPng(): Promise<string | null> {
    if (!this.offscreenTarget) return null;
    return readTexturePng(this.gpu.device, this.offscreenTarget);
  }

  /** Resolves after n more frames have been submitted and completed on the GPU. */
  waitFrames(n: number): Promise<void> {
    return new Promise((resolve) => this.frameWaiters.push({ n, resolve }));
  }

  frame() {
    if (this.rebuildPending) {
      const applyDefaults = this.preset.id !== this.params.values.scene.preset;
      this.rebuildPending = null;
      this.buildWorld(applyDefaults);
      this.handleResize();
    }
    const now = performance.now();
    const realDt = Math.min(0.1, (now - this.lastTime) / 1000);
    this.lastTime = now;
    this.realTime += realDt;
    this.fpsAvg = this.fpsAvg * 0.95 + (1 / Math.max(realDt, 1e-3)) * 0.05;
    const p: Params = this.params.values;
    const baseDt = this.options.fixedDt ?? Math.min(realDt, 1 / 30);
    const dt = p.sim.paused ? 0 : baseDt * p.sim.timeScale;
    this.simTime += dt;

    this.handleResize();
    this.orbit.update(realDt);
    this.camera.update(this.canvas.width / this.canvas.height);

    this.world.simTime = this.simTime;
    this.world.frameIndex = this.frameIndex;
    this.world.updateUniforms(p, dt);
    this.renderer.updateFrameUniforms(this.camera, p, this.modules, { sim: this.simTime, dt, real: this.realTime, frame: this.frameIndex });

    const frame: FrameContext = this.renderer.makeFrameContext(
      {
        dt,
        realDt,
        time: this.simTime,
        realTime: this.realTime,
        frameIndex: this.frameIndex,
        camera: this.camera,
        params: p,
        world: this.world,
        profiler: this.profiler,
      },
      this.modules,
    );

    const device = this.gpu.device;
    const encoder = device.createCommandEncoder({ label: `frame ${this.frameIndex}` });
    this.profiler.beginFrame();
    const skip = debugSkip();
    this.modules.editor.update(frame);
    if (!skip.has('terrain')) this.modules.terrain.update(encoder, frame);
    if (!skip.has('swe')) this.modules.swe.step(encoder, frame);
    if (!skip.has('particles')) this.modules.particles.step(encoder, frame);
    if (!skip.has('audio')) this.modules.audio.encodeReadback(encoder, frame);
    const swap = this.swapchainView();
    this.renderer.render(encoder, frame, this.modules, swap);
    const profilerDone = this.profiler.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    this.inFlight++;
    device.queue.onSubmittedWorkDone().then(() => this.inFlight--);
    profilerDone?.();
    this.modules.audio.afterSubmit(frame);
    this.frameIndex++;

    if (this.frameWaiters.length) {
      for (const w of this.frameWaiters) w.n--;
      const done = this.frameWaiters.filter((w) => w.n <= 0);
      this.frameWaiters = this.frameWaiters.filter((w) => w.n > 0);
      if (done.length) device.queue.onSubmittedWorkDone().then(() => done.forEach((w) => w.resolve()));
    }

    if (this.panel && this.frameIndex % 15 === 0) {
      this.panel.stats.fps = this.fpsAvg;
      const s = { ...this.modules.swe.stats, ...this.modules.particles.stats };
      const prof = [...this.profiler.results].map(([k, v]) => `${k}: ${v.toFixed(2)}ms`);
      this.panel.stats.info = [
        `grid ${this.world.domain.nx}×${this.world.domain.nz} @ ${(this.world.domain.cellSize * 1000).toFixed(1)}mm`,
        ...Object.entries(s).map(([k, v]) => `${k}: ${typeof v === 'number' ? +v.toPrecision(4) : v}`),
        ...prof,
      ].join('\n');
      this.panel.refresh();
    }
  }
}
