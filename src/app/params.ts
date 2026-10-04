/**
 * Global, user-tweakable parameters. A single mutable object tree with change
 * notification. Modules read values every frame (cheap) and may subscribe to
 * structural changes (quality, preset) that require resource reallocation.
 */

export type Quality = 'low' | 'medium' | 'high' | 'ultra';
export type EditorTool = 'orbit' | 'add' | 'move' | 'delete';
export type ObstacleKind = 'boulder' | 'cobble' | 'slab' | 'log';
export type DebugView = 'none' | 'depth' | 'velocity' | 'foam' | 'particles' | 'sdf' | 'caustics' | 'wetness' | 'normals';

export interface Params {
  scene: {
    /** Preset id (see src/terrain/presets.ts). */
    preset: string;
  };
  flow: {
    /** Volumetric flow rate of the main inflow, litres per second. */
    rate: number;
    /** Mean inflow velocity (m/s). Together with rate this sets the inflow depth. */
    velocity: number;
    /** Multiplier on the preset's inflow width (0.2..1.5). */
    widthScale: number;
    /** Inflow unsteadiness / incoming turbulence (0..1). */
    turbulence: number;
    /** Bed roughness — Manning's n (s/m^(1/3)). */
    roughness: number;
  };
  water: {
    /** 0 = murky, 1 = crystal clear. Scales absorption/scattering. */
    clarity: number;
    /** Suspended sediment / tannin amount (0..1): browner, more scattering. */
    turbidity: number;
    /** Whitewater / foam intensity multiplier. */
    foam: number;
    /** Small-scale capillary ripple intensity multiplier. */
    ripples: number;
  };
  sim: {
    quality: Quality;
    /** Simulation speed multiplier (0.05..1). < 1 gives slow motion. */
    timeScale: number;
    paused: boolean;
    /** Falling-water particle cohesion (surface tension) multiplier. */
    cohesion: number;
    /** Falling-water adhesion to rock (Coanda / clinging) multiplier. */
    adhesion: number;
  };
  light: {
    /** Hour of day 5..20 (drives sun elevation). */
    timeOfDay: number;
    /** Sun compass azimuth in degrees (0 = +X / downstream). */
    sunAzimuth: number;
    /** 0 clear .. 1 overcast. */
    cloudiness: number;
    /** Forest canopy dappled shading 0..1. */
    canopy: number;
    /** Exposure compensation in EV. */
    exposure: number;
  };
  audio: {
    enabled: boolean;
    /** Master volume 0..1. */
    volume: number;
    /** Bubble/droplet (babble, plinks) layer gain. */
    bubbles: number;
    /** Broadband roar / rush layer gain. */
    roar: number;
    /** Forest ambience gain. */
    ambience: number;
    /** Reverb send 0..1. */
    reverb: number;
  };
  camera: {
    fov: number;
    /** Depth of field (macro look). */
    dof: boolean;
    /** f-stop for DoF. */
    aperture: number;
  };
  editor: {
    tool: EditorTool;
    addKind: ObstacleKind;
    /** Size (m) for new obstacles. */
    addSize: number;
  };
  debug: {
    view: DebugView;
    showStats: boolean;
  };
}

export function defaultParams(): Params {
  return {
    scene: { preset: 'ledge' },
    flow: { rate: 3.0, velocity: 0.6, widthScale: 1, turbulence: 0.3, roughness: 0.03 },
    water: { clarity: 0.8, turbidity: 0.15, foam: 1, ripples: 1 },
    sim: { quality: 'medium', timeScale: 1, paused: false, cohesion: 1, adhesion: 1 },
    light: { timeOfDay: 14, sunAzimuth: 35, cloudiness: 0.2, canopy: 0.35, exposure: 0 },
    audio: { enabled: true, volume: 0.8, bubbles: 1, roar: 1, ambience: 0.35, reverb: 0.3 },
    camera: { fov: 50, dof: false, aperture: 4 },
    editor: { tool: 'orbit', addKind: 'boulder', addSize: 0.18 },
    debug: { view: 'none', showStats: false },
  };
}

type Listener = (path: string, value: unknown) => void;

/** Holds the live Params object and dispatches change notifications by dotted path. */
export class ParamStore {
  readonly values: Params;
  private listeners = new Map<string, Set<Listener>>();
  private anyListeners = new Set<Listener>();

  constructor(initial?: Params) {
    this.values = initial ?? defaultParams();
  }

  get<T = unknown>(path: string): T {
    return path.split('.').reduce<any>((o, k) => (o == null ? o : o[k]), this.values) as T;
  }

  /** Sets a value by dotted path (e.g. "flow.rate") and notifies listeners. */
  set(path: string, value: unknown) {
    const keys = path.split('.');
    let o: any = this.values;
    for (let i = 0; i < keys.length - 1; i++) o = o[keys[i]];
    const last = keys[keys.length - 1];
    if (o[last] === value) return;
    o[last] = value;
    this.notify(path, value);
  }

  /** Notifies listeners after an in-place mutation (e.g. by the UI binding). */
  notify(path: string, value: unknown = this.get(path)) {
    for (const [p, set] of this.listeners) {
      if (path === p || path.startsWith(p + '.') || p.startsWith(path + '.')) set.forEach((l) => l(path, value));
    }
    this.anyListeners.forEach((l) => l(path, value));
  }

  /** Subscribe to a path prefix ("flow" fires for "flow.rate"). Returns an unsubscribe fn. */
  on(pathPrefix: string, listener: Listener): () => void {
    let set = this.listeners.get(pathPrefix);
    if (!set) this.listeners.set(pathPrefix, (set = new Set()));
    set.add(listener);
    return () => set!.delete(listener);
  }

  onAny(listener: Listener): () => void {
    this.anyListeners.add(listener);
    return () => this.anyListeners.delete(listener);
  }
}
