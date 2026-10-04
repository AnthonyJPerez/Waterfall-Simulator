/**
 * Control panel (Tweakpane) bound to the ParamStore.
 */
import { Pane } from 'tweakpane';
import type { ParamStore } from '../app/params';
import { PRESETS } from '../terrain/presets';

export interface PanelActions {
  resetWater(): void;
  loadPreset(id: string): void;
  enableAudio(): void;
  clearObstacles(): void;
}

export class Panel {
  readonly pane: Pane;
  readonly stats: Record<string, string | number> = { fps: 0, particles: 0, info: '' };

  constructor(container: HTMLElement, private store: ParamStore, actions: PanelActions) {
    const p = store.values;
    const pane = new Pane({ container, title: 'Waterfall Simulator' });
    this.pane = pane;
    const bind = (folder: any, obj: any, key: string, path: string, opts: Record<string, unknown> = {}) =>
      folder.addBinding(obj, key, opts).on('change', () => store.notify(path));

    const scene = pane.addFolder({ title: 'Scene' });
    const presetOptions = Object.fromEntries(PRESETS.map((x) => [x.name, x.id]));
    scene.addBinding(p.scene, 'preset', { options: presetOptions, label: 'preset' }).on('change', (ev: any) => actions.loadPreset(ev.value));
    scene.addButton({ title: 'Reset water' }).on('click', () => actions.resetWater());

    const flow = pane.addFolder({ title: 'Flow' });
    bind(flow, p.flow, 'rate', 'flow.rate', { min: 0.02, max: 25, step: 0.01, label: 'flow (L/s)' });
    bind(flow, p.flow, 'velocity', 'flow.velocity', { min: 0.05, max: 2.5, step: 0.01, label: 'velocity (m/s)' });
    bind(flow, p.flow, 'widthScale', 'flow.widthScale', { min: 0.1, max: 1.5, step: 0.01, label: 'inflow width' });
    bind(flow, p.flow, 'turbulence', 'flow.turbulence', { min: 0, max: 1, step: 0.01, label: 'turbulence' });
    bind(flow, p.flow, 'roughness', 'flow.roughness', { min: 0.01, max: 0.08, step: 0.001, label: 'bed roughness n' });

    const tools = pane.addFolder({ title: 'Rocks & Obstacles' });
    bind(tools, p.editor, 'tool', 'editor.tool', { options: { Orbit: 'orbit', Add: 'add', Move: 'move', Delete: 'delete' }, label: 'tool' });
    bind(tools, p.editor, 'addKind', 'editor.addKind', { options: { Boulder: 'boulder', Cobble: 'cobble', Slab: 'slab', Log: 'log' }, label: 'new object' });
    bind(tools, p.editor, 'addSize', 'editor.addSize', { min: 0.03, max: 0.4, step: 0.005, label: 'size (m)' });
    tools.addButton({ title: 'Remove all obstacles' }).on('click', () => actions.clearObstacles());

    const water = pane.addFolder({ title: 'Water', expanded: false });
    bind(water, p.water, 'clarity', 'water.clarity', { min: 0, max: 1, step: 0.01 });
    bind(water, p.water, 'turbidity', 'water.turbidity', { min: 0, max: 1, step: 0.01, label: 'tannin / silt' });
    bind(water, p.water, 'foam', 'water.foam', { min: 0, max: 2, step: 0.01 });
    bind(water, p.water, 'ripples', 'water.ripples', { min: 0, max: 2, step: 0.01 });

    const sim = pane.addFolder({ title: 'Simulation', expanded: false });
    bind(sim, p.sim, 'quality', 'sim.quality', { options: { Low: 'low', Medium: 'medium', High: 'high', Ultra: 'ultra' } });
    bind(sim, p.sim, 'timeScale', 'sim.timeScale', { min: 0.05, max: 1, step: 0.01, label: 'time scale' });
    bind(sim, p.sim, 'paused', 'sim.paused');
    bind(sim, p.sim, 'cohesion', 'sim.cohesion', { min: 0, max: 3, step: 0.01, label: 'surface tension' });
    bind(sim, p.sim, 'adhesion', 'sim.adhesion', { min: 0, max: 3, step: 0.01, label: 'rock adhesion' });

    const light = pane.addFolder({ title: 'Lighting & Camera', expanded: false });
    bind(light, p.light, 'timeOfDay', 'light.timeOfDay', { min: 5.5, max: 19.5, step: 0.05, label: 'time of day' });
    bind(light, p.light, 'sunAzimuth', 'light.sunAzimuth', { min: -180, max: 180, step: 1, label: 'sun azimuth' });
    bind(light, p.light, 'cloudiness', 'light.cloudiness', { min: 0, max: 1, step: 0.01 });
    bind(light, p.light, 'canopy', 'light.canopy', { min: 0, max: 1, step: 0.01, label: 'canopy shade' });
    bind(light, p.light, 'exposure', 'light.exposure', { min: -3, max: 3, step: 0.05 });
    bind(light, p.camera, 'fov', 'camera.fov', { min: 15, max: 90, step: 1 });
    bind(light, p.camera, 'dof', 'camera.dof', { label: 'depth of field' });
    bind(light, p.camera, 'aperture', 'camera.aperture', { min: 1.4, max: 22, step: 0.1, label: 'f-stop' });

    const audio = pane.addFolder({ title: 'Audio', expanded: false });
    audio.addButton({ title: 'Enable sound' }).on('click', () => actions.enableAudio());
    bind(audio, p.audio, 'enabled', 'audio.enabled');
    bind(audio, p.audio, 'volume', 'audio.volume', { min: 0, max: 1, step: 0.01 });
    bind(audio, p.audio, 'bubbles', 'audio.bubbles', { min: 0, max: 2, step: 0.01, label: 'babble / drops' });
    bind(audio, p.audio, 'roar', 'audio.roar', { min: 0, max: 2, step: 0.01, label: 'rush / roar' });
    bind(audio, p.audio, 'ambience', 'audio.ambience', { min: 0, max: 1, step: 0.01, label: 'forest' });
    bind(audio, p.audio, 'reverb', 'audio.reverb', { min: 0, max: 1, step: 0.01 });

    const debug = pane.addFolder({ title: 'Debug', expanded: false });
    bind(debug, p.debug, 'view', 'debug.view', {
      options: { None: 'none', Depth: 'depth', Velocity: 'velocity', Foam: 'foam', Particles: 'particles', SDF: 'sdf', Caustics: 'caustics', Wetness: 'wetness', Normals: 'normals' },
    });
    bind(debug, p.debug, 'showStats', 'debug.showStats', { label: 'stats' });
    debug.addBinding(this.stats, 'fps', { readonly: true, format: (v: number) => v.toFixed(1) });
    debug.addBinding(this.stats, 'info', { readonly: true, multiline: true, rows: 6 });
  }

  refresh() {
    this.pane.refresh();
  }
}
