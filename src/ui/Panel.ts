/**
 * Control panel (Tweakpane v4) bound to the ParamStore, plus the on-canvas chrome
 * (tool bar, hint line, help overlay, sound call-to-action; see chrome.ts).
 *
 * With a `host` (the App) the panel also offers camera views, screenshots, scene reset,
 * live flow / particle readouts and the audio state; without one it degrades to the
 * parameter bindings and the actions passed in.
 */
import { Pane } from 'tweakpane';
import type { ButtonApi, FolderApi } from 'tweakpane';
import type { ParamStore } from '../app/params';
import { PRESETS, type ScenePreset } from '../terrain/presets';
import type { Camera, OrbitController } from '../core/camera';
import type { SceneModel } from '../world/scene';
import { ADD_SIZE_RANGE, EDITOR_STATUS_EVENT, modKey, sendEditorCommand, type EditorCommand, type EditorStatus } from '../editor/protocol';
import { kindTemplate } from '../editor/placement';
import { computeCameraViews, type CameraView } from './cameraViews';
import { EditorChrome } from './chrome';
import { downloadScreenshot, type ScreenshotHost } from './screenshot';
import { flowReadouts, statLines } from './statsFormat';

export interface PanelActions {
  resetWater(): void;
  loadPreset(id: string): void;
  enableAudio(): void;
  clearObstacles(): void;
}

/** What the panel needs from the application (App satisfies this structurally). */
export interface PanelHost extends ScreenshotHost {
  readonly modules: {
    swe: { readonly stats: Readonly<Record<string, number>> };
    particles: { readonly stats: Readonly<Record<string, number>> };
    audio: { readonly running: boolean };
    terrain: { heightAt(x: number, z: number): number };
  };
  readonly scene: SceneModel;
  readonly preset: ScenePreset;
  readonly orbit: OrbitController;
  readonly camera: Camera;
  readonly world: { readonly domain: { sizeX: number; sizeZ: number; minY: number; maxY: number; nx: number; nz: number; cellSize: number } };
  readonly profiler?: { readonly results: Map<string, number> };
}

type Bindable = Record<string, any>;

export class Panel {
  readonly pane: Pane;
  readonly stats: Record<string, string | number> = { fps: 0, particles: 0, info: '' };
  private readouts = { inflow: '—', speed: '—', volume: '—', falling: '—', particles: '—', obstacles: '0', selection: 'none', live: '' };
  private chrome?: EditorChrome;
  private timer: ReturnType<typeof setInterval> | null = null;
  private soundBtn: ButtonApi;
  private editorButtons: Partial<Record<EditorCommand, ButtonApi>> = {};
  private descEl: HTMLElement | null = null;
  private refreshQueued = false;
  private refreshing = false;
  private status: EditorStatus | null = null;
  private canvas: HTMLCanvasElement | null;
  private unsubs: (() => unknown)[] = [];

  constructor(
    container: HTMLElement,
    private store: ParamStore,
    private actions: PanelActions,
    private host?: PanelHost,
  ) {
    const p = store.values;
    const pane = new Pane({ container, title: 'Creekside' });
    this.pane = pane;
    this.canvas = host?.gpu.canvas ?? (document.getElementById('view') as HTMLCanvasElement | null);
    const bind = (folder: FolderApi | Pane, obj: Bindable, key: string, path: string, opts: Record<string, unknown> = {}) =>
      folder.addBinding(obj, key, opts).on('change', () => {
        if (!this.refreshing) store.notify(path);
      });
    const monitor = (folder: FolderApi | Pane, key: keyof Panel['readouts'], label: string, extra: Record<string, unknown> = {}) =>
      folder.addBinding(this.readouts, key, { readonly: true, label, interval: 250, ...extra });

    // Sound call-to-action (hidden once audio runs).
    this.soundBtn = pane.addButton({ title: '♪  Enable sound', label: 'audio is off' });
    this.soundBtn.on('click', () => {
      store.set('audio.enabled', true);
      actions.enableAudio();
    });

    // --- Scene
    const scene = pane.addFolder({ title: 'Scene' });
    const presetOptions = Object.fromEntries(PRESETS.map((x) => [x.name, x.id]));
    scene.addBinding(p.scene, 'preset', { options: presetOptions, label: 'preset' }).on('change', (ev: any) => {
      this.updateDescription();
      if (!this.refreshing) actions.loadPreset(ev.value);
    });
    this.descEl = document.createElement('div');
    this.descEl.className = 'wf-preset-desc';
    scene.element.querySelector('.tp-fldv_c')?.appendChild(this.descEl);
    this.updateDescription();
    scene.addButton({ title: 'Reset water' }).on('click', () => actions.resetWater());
    if (host) scene.addButton({ title: 'Reset scene (restore rocks)' }).on('click', () => this.resetScene());
    if (host) {
      const shot = scene.addButton({ title: 'Screenshot (PNG)' });
      shot.on('click', async () => {
        shot.title = 'Saving…';
        shot.disabled = true;
        try {
          await downloadScreenshot(host);
        } finally {
          shot.title = 'Screenshot (PNG)';
          shot.disabled = false;
        }
      });
    }

    // --- Flow (inputs + live readouts)
    const flow = pane.addFolder({ title: 'Flow' });
    bind(flow, p.flow, 'rate', 'flow.rate', { min: 0.02, max: 25, step: 0.01, label: 'flow (L/s)' });
    bind(flow, p.flow, 'velocity', 'flow.velocity', { min: 0.05, max: 2.5, step: 0.01, label: 'velocity (m/s)' });
    bind(flow, p.flow, 'widthScale', 'flow.widthScale', { min: 0.1, max: 1.5, step: 0.01, label: 'inflow width' });
    bind(flow, p.flow, 'turbulence', 'flow.turbulence', { min: 0, max: 1, step: 0.01, label: 'turbulence' });
    bind(flow, p.flow, 'roughness', 'flow.roughness', { min: 0.01, max: 0.08, step: 0.001, label: 'bed roughness n' });
    if (host) {
      flow.addBlade({ view: 'separator' });
      monitor(flow, 'inflow', 'actual inflow');
      monitor(flow, 'falling', 'falling water');
      monitor(flow, 'speed', 'max speed');
      monitor(flow, 'volume', 'water volume');
      monitor(flow, 'particles', 'particles');
    }

    // --- Rocks & obstacles
    const tools = pane.addFolder({ title: 'Rocks & Obstacles' });
    bind(tools, p.editor, 'tool', 'editor.tool', { options: { 'Orbit  (1)': 'orbit', 'Add  (2)': 'add', 'Move  (3)': 'move', 'Delete  (4)': 'delete' }, label: 'tool' });
    bind(tools, p.editor, 'addKind', 'editor.addKind', { options: { Boulder: 'boulder', Cobble: 'cobble', Slab: 'slab', Log: 'log' }, label: 'new object' });
    bind(tools, p.editor, 'addSize', 'editor.addSize', { min: ADD_SIZE_RANGE[0], max: ADD_SIZE_RANGE[1], step: 0.005, label: 'size (m)' });
    monitor(tools, 'selection', 'selected');
    const M = modKey();
    const edBtn = (cmd: EditorCommand, title: string, label?: string) => {
      const b = tools.addButton({ title, label });
      b.on('click', () => this.canvas && sendEditorCommand(this.canvas, cmd));
      this.editorButtons[cmd] = b;
      return b;
    };
    edBtn('undo', `Undo  (${M}Z)`);
    edBtn('redo', `Redo  (${M}Shift+Z)`);
    edBtn('duplicate', `Duplicate selected  (${M}D)`);
    edBtn('delete', 'Delete selected  (Del)');
    tools.addButton({ title: 'Remove all obstacles' }).on('click', () => actions.clearObstacles());
    monitor(tools, 'obstacles', 'in scene');

    // --- Camera
    const cam = pane.addFolder({ title: 'Camera' });
    if (host) {
      const views: { id: CameraView['id']; title: string }[] = [
        { id: 'preset', title: 'Preset view' },
        { id: 'lip', title: 'Close-up on the lip' },
        { id: 'pool', title: 'Plunge pool' },
        { id: 'eye', title: 'Eye level' },
        { id: 'overview', title: 'Overview' },
      ];
      for (const v of views) cam.addButton({ title: v.title }).on('click', () => this.flyToView(v.id));
    }
    bind(cam, p.camera, 'fov', 'camera.fov', { min: 15, max: 90, step: 1, label: 'field of view' });
    bind(cam, p.camera, 'dof', 'camera.dof', { label: 'depth of field' });
    bind(cam, p.camera, 'aperture', 'camera.aperture', { min: 1.4, max: 22, step: 0.1, label: 'f-stop' });

    // --- Water
    const water = pane.addFolder({ title: 'Water', expanded: false });
    bind(water, p.water, 'clarity', 'water.clarity', { min: 0, max: 1, step: 0.01 });
    bind(water, p.water, 'turbidity', 'water.turbidity', { min: 0, max: 1, step: 0.01, label: 'tannin / silt' });
    bind(water, p.water, 'foam', 'water.foam', { min: 0, max: 2, step: 0.01 });
    bind(water, p.water, 'ripples', 'water.ripples', { min: 0, max: 2, step: 0.01 });

    // --- Simulation
    const sim = pane.addFolder({ title: 'Simulation', expanded: false });
    bind(sim, p.sim, 'quality', 'sim.quality', { options: { Low: 'low', Medium: 'medium', High: 'high', Ultra: 'ultra' } });
    bind(sim, p.sim, 'timeScale', 'sim.timeScale', { min: 0.05, max: 1, step: 0.01, label: 'time scale' });
    bind(sim, p.sim, 'paused', 'sim.paused', { label: 'paused  (P)' });
    bind(sim, p.sim, 'cohesion', 'sim.cohesion', { min: 0, max: 3, step: 0.01, label: 'surface tension' });
    bind(sim, p.sim, 'adhesion', 'sim.adhesion', { min: 0, max: 3, step: 0.01, label: 'rock adhesion' });

    // --- Lighting
    const light = pane.addFolder({ title: 'Lighting', expanded: false });
    bind(light, p.light, 'timeOfDay', 'light.timeOfDay', { min: 5.5, max: 19.5, step: 0.05, label: 'time of day' });
    bind(light, p.light, 'sunAzimuth', 'light.sunAzimuth', { min: -180, max: 180, step: 1, label: 'sun azimuth' });
    bind(light, p.light, 'cloudiness', 'light.cloudiness', { min: 0, max: 1, step: 0.01 });
    bind(light, p.light, 'canopy', 'light.canopy', { min: 0, max: 1, step: 0.01, label: 'canopy shade' });
    bind(light, p.light, 'exposure', 'light.exposure', { min: -3, max: 3, step: 0.05 });

    // --- Audio
    const audio = pane.addFolder({ title: 'Audio', expanded: false });
    audio.addButton({ title: 'Enable sound' }).on('click', () => {
      store.set('audio.enabled', true);
      actions.enableAudio();
    });
    bind(audio, p.audio, 'enabled', 'audio.enabled');
    bind(audio, p.audio, 'volume', 'audio.volume', { min: 0, max: 1, step: 0.01 });
    bind(audio, p.audio, 'bubbles', 'audio.bubbles', { min: 0, max: 2, step: 0.01, label: 'babble / drops' });
    bind(audio, p.audio, 'roar', 'audio.roar', { min: 0, max: 2, step: 0.01, label: 'rush / roar' });
    bind(audio, p.audio, 'ambience', 'audio.ambience', { min: 0, max: 1, step: 0.01, label: 'forest' });
    bind(audio, p.audio, 'reverb', 'audio.reverb', { min: 0, max: 1, step: 0.01 });

    // --- Debug / stats
    const debug = pane.addFolder({ title: 'Stats & Debug', expanded: false });
    bind(debug, p.debug, 'view', 'debug.view', {
      options: { None: 'none', Depth: 'depth', Velocity: 'velocity', Foam: 'foam', Particles: 'particles', SDF: 'sdf', Caustics: 'caustics', Wetness: 'wetness', Normals: 'normals' },
    });
    bind(debug, p.debug, 'showStats', 'debug.showStats', { label: 'stats overlay' });
    debug.addBinding(this.stats, 'fps', { readonly: true, view: 'graph', min: 0, max: 90, label: 'fps' });
    debug.addBinding(this.stats, 'fps', { readonly: true, format: (v: number) => v.toFixed(1), label: '' });
    debug.addBinding(this.stats, 'info', { readonly: true, multiline: true, rows: 8, label: 'info' });
    if (host) debug.addBinding(this.readouts, 'live', { readonly: true, multiline: true, rows: 8, label: 'live', interval: 500 });

    // Small screens: start collapsed.
    if (typeof window !== 'undefined' && (window.innerWidth < 720 || window.innerHeight < 520)) pane.expanded = false;

    // Keep the panel in sync with programmatic changes (keyboard shortcuts, presets, scripts).
    this.unsubs.push(store.onAny(() => this.queueRefresh()));

    if (this.canvas) {
      this.chrome = new EditorChrome({
        canvas: this.canvas,
        store,
        enableAudio: () => {
          store.set('audio.enabled', true);
          actions.enableAudio();
        },
        audioRunning: () => (host ? !!host.modules?.audio?.running : undefined),
      });
      const onStatus = (e: Event) => {
        this.status = (e as CustomEvent<EditorStatus>).detail;
        this.applyStatus();
      };
      this.canvas.addEventListener(EDITOR_STATUS_EVENT, onStatus);
      this.unsubs.push(() => this.canvas?.removeEventListener(EDITOR_STATUS_EVENT, onStatus));
      sendEditorCommand(this.canvas, 'status');
    }
    this.applyStatus();
    this.poll();
    this.timer = setInterval(() => this.poll(), 250);
  }

  private updateDescription() {
    if (!this.descEl) return;
    const preset = PRESETS.find((x) => x.id === this.store.values.scene.preset);
    this.descEl.textContent = preset?.description ?? '';
  }

  private queueRefresh() {
    if (this.refreshQueued) return;
    this.refreshQueued = true;
    requestAnimationFrame(() => {
      this.refreshQueued = false;
      this.refresh();
    });
  }

  private applyStatus() {
    const st = this.status;
    const b = this.editorButtons;
    if (b.undo) {
      b.undo.disabled = !st?.canUndo;
      b.undo.title = `Undo${st?.undoLabel ? ' ' + st.undoLabel : ''}  (${modKey()}Z)`;
    }
    if (b.redo) {
      b.redo.disabled = !st?.canRedo;
      b.redo.title = `Redo${st?.redoLabel ? ' ' + st.redoLabel : ''}  (${modKey()}Shift+Z)`;
    }
    const hasSel = !!st?.selected;
    if (b.delete) b.delete.disabled = !hasSel;
    if (b.duplicate) b.duplicate.disabled = !hasSel;
    this.readouts.selection = st?.selected ? `${kindTemplate(st.selected.kind).label} #${st.selected.id} · ${Math.round(st.selected.size * 100)} cm` : 'none';
    if (st) this.readouts.obstacles = String(st.obstacleCount);
  }

  /** Periodic UI state: audio CTA, live readouts, stats HUD. */
  private poll() {
    const host = this.host;
    let running: boolean | undefined;
    try {
      running = host ? !!host.modules?.audio?.running : undefined;
    } catch {
      running = undefined;
    }
    this.soundBtn.hidden = running === true;
    this.chrome?.update();
    if (!host) return;
    try {
      const swe = host.modules.swe.stats ?? {};
      const parts = host.modules.particles.stats ?? {};
      const r = flowReadouts(swe, parts);
      Object.assign(this.readouts, r, { obstacles: String(host.scene.obstacles.length) });
      const d = host.world.domain;
      const lines = [
        `grid ${d.nx}×${d.nz} @ ${(d.cellSize * 1000).toFixed(1)} mm`,
        ...statLines('swe.', swe),
        ...statLines('particles.', parts),
        ...[...(host.profiler?.results ?? new Map())].map(([k, v]) => `gpu ${k}: ${v.toFixed(2)} ms`),
      ];
      this.readouts.live = lines.join('\n');
      if (this.store.values.debug.showStats) {
        const fps = typeof this.stats.fps === 'number' ? this.stats.fps : 0;
        this.chrome?.setStats(
          [`${fps.toFixed(1)} fps`, `inflow ${r.inflow} · falling ${r.falling}`, `max speed ${r.speed} · volume ${r.volume}`, `particles ${r.particles}`, ...lines.slice(0, 1)].join('\n'),
        );
      }
    } catch {
      /* modules may be mid-rebuild */
    }
  }

  private flyToView(id: CameraView['id']) {
    const host = this.host;
    if (!host) return;
    try {
      const views = computeCameraViews({
        extent: host.world.domain,
        presetCamera: host.preset.camera,
        heightAt: (x, z) => host.modules.terrain.heightAt(x, z),
        inflowZ: host.scene.inflows[0]?.z,
        initialWater: host.preset.initialWater,
        fovY: host.camera.fovY,
        aspect: host.camera.aspect,
      });
      const v = views.find((x) => x.id === id);
      if (v) host.orbit.flyTo(v.position, v.target);
    } catch (e) {
      console.warn('[ui] camera view failed', e);
    }
  }

  /** Restores the preset's obstacles (one undoable step: removes + adds in one burst). */
  resetScene() {
    const host = this.host;
    if (!host) return;
    const scene = host.scene;
    [...scene.obstacles].forEach((o) => scene.remove(o.id));
    for (const o of host.preset.obstacles) {
      scene.add({ ...o, position: [...o.position], rotation: [...o.rotation], scale: [...o.scale] });
    }
  }

  refresh() {
    this.refreshing = true;
    try {
      this.pane.refresh();
    } finally {
      this.refreshing = false;
    }
    this.updateDescription();
  }

  dispose() {
    if (this.timer) clearInterval(this.timer);
    this.unsubs.forEach((u) => u());
    this.chrome?.destroy();
    this.pane.dispose();
  }
}
