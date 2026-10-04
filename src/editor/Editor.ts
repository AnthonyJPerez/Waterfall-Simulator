/**
 * Obstacle editor: add / select / move / raise / rotate / tilt / scale / duplicate / delete
 * rocks and logs, with undo/redo, a placement ghost, selection outlines and gizmo hints.
 *
 * Tools (params.editor.tool; keys 1-4):
 *   orbit  — camera only
 *   add    — a ghost of params.editor.addKind follows the cursor; click places it (partly
 *            embedded, slabs/logs follow the slope); Alt+wheel or [ ] size, Q/E or Shift+wheel
 *            rotate, R re-rolls the shape, K cycles the kind
 *   move   — hover highlight, click selects, drag slides over the bed (the dragged obstacle's
 *            own baked top is masked out of the ground), Shift/Alt+drag raises/lowers,
 *            Q/E yaw, T/G tilt, [ ]/Alt+wheel scale, arrows nudge, Del deletes, Ctrl/⌘+D
 *            duplicates, Esc deselects (or cancels a drag)
 *   delete — hover highlight, click removes
 * Everywhere: drag empty space to orbit, right-drag to pan, wheel to zoom (towards the
 * cursor), double-click to focus, Ctrl/⌘+Z undo, Ctrl/⌘+Shift+Z or Ctrl+Y redo, F focus.
 *
 * Scene edits while dragging / scrolling are rate-limited (≤ 20 Hz) because the terrain
 * re-bakes on each SceneModel change; the final pose is always committed on release.
 * The editor only uses the TerrainSystem / Camera / OrbitController contracts.
 */
import type { EditorModule, FrameContext, ModuleContext, RayHit, TerrainSystem } from '../app/modules';
import type { EditorTool, ObstacleKind } from '../app/params';
import type { Camera, OrbitController } from '../core/camera';
import type { Obstacle, SceneChange } from '../world/scene';
import { cloneData, SceneHistory, type ObstacleData } from './history';
import {
  add,
  clamp,
  cross,
  footprintOf,
  footprintOutline,
  footprintRadius,
  maxHalfExtent,
  normalize,
  pickProxies,
  proxyTopY,
  qRotate,
  qRotateYaw,
  qTilt,
  resolvePick,
  scale as vscale,
  sub,
  type Quat,
  type V3,
} from './math';
import { OverlayRenderer, type OverlaySegment, type OverlayShape, type RGBA } from './overlay';
import {
  bilinearGround,
  clampPositionToDomain,
  computePlacement,
  groundUnder,
  kindTemplate,
  marchGround,
  maskedGround,
  MAX_HALF_EXTENT,
  MIN_HALF_EXTENT,
  OBSTACLE_KINDS,
  rayPlaneY,
  rollVariation,
  type GroundFn,
  type PlacedObstacle,
  type Variation,
} from './placement';
import { EDITOR_COMMAND_EVENT, EDITOR_STATUS_EVENT, isMac, modKey, overlayControl, type EditorCommand, type EditorStatus } from './protocol';
import { TrailingThrottle } from './throttle';

export const EDITOR_TOOLS: EditorTool[] = ['orbit', 'add', 'move', 'delete'];
/** Minimum time between SceneModel updates while dragging / scrolling (≤ 20 Hz). */
export const COMMIT_INTERVAL_MS = 50;
const CLICK_SLOP_PX = 4;
const DOUBLE_CLICK_MS = 350;
export const ADD_SIZE_RANGE: [number, number] = [0.02, 0.5];
const YAW_STEP = Math.PI / 12; // 15°
const TILT_STEP = Math.PI / 24; // 7.5°
const SCALE_STEP = 1.08;

const COLORS = {
  ghost: [0.62, 0.9, 1.0, 1] as RGBA,
  ghostBad: [1.0, 0.45, 0.38, 1] as RGBA,
  hover: [1, 1, 1, 1] as RGBA,
  selected: [1.0, 0.8, 0.38, 1] as RGBA,
  danger: [1.0, 0.36, 0.3, 1] as RGBA,
  gizmo: [0.9, 0.95, 1.0, 1] as RGBA,
};

interface Pose {
  position: V3;
  rotation: Quat;
  scale: V3;
}

interface Press {
  kind: 'add' | 'move' | 'delete' | 'empty';
  id?: number;
  x: number;
  y: number;
  pointerId: number;
  moved: boolean;
}

interface Drag {
  id: number;
  kind: ObstacleKind;
  start: Pose;
  live: Pose;
  mode: 'slide' | 'lift';
  /** centre.xz - cursor ground point.xz at the last (re)base. */
  offset: [number, number];
  /** centre.y - ground under the footprint. */
  embed: number;
  liftRef: { clientY: number; y: number };
  /** Recently committed poses (their footprints are masked out of the ground). */
  committed: { pose: Pose; t: number }[];
  pointerId: number;
}

interface PointerState {
  x: number;
  y: number;
  inside: boolean;
  shift: boolean;
  alt: boolean;
  type: string;
}

const clonePose = (p: Pose): Pose => ({ position: [...p.position], rotation: [...p.rotation], scale: [...p.scale] });
const poseOf = (o: Pick<Obstacle, 'position' | 'rotation' | 'scale'>): Pose => ({
  position: [o.position[0], o.position[1], o.position[2]],
  rotation: [o.rotation[0], o.rotation[1], o.rotation[2], o.rotation[3]],
  scale: [o.scale[0], o.scale[1], o.scale[2]],
});
const samePose = (a: Pose, b: Pose) =>
  [...a.position, ...a.rotation, ...a.scale].every((v, i) => Math.abs(v - [...b.position, ...b.rotation, ...b.scale][i]) < 1e-7);

function isTypingTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || !!el.isContentEditable;
}

/** Obstacles of the current preset survive a world rebuild (quality switch) — see restoreAfterRebuild. */
let rebuildStash: { presetId: string; obstacles: ObstacleData[] } | null = null;

export class ObstacleEditor implements EditorModule {
  private overlay: OverlayRenderer;
  readonly history: SceneHistory;
  private ground: GroundFn;
  private variation: Variation;
  private ghostYaw = 0;
  private ghost: { placed: PlacedObstacle; valid: boolean } | null = null;
  private ghostHiddenAt: [number, number] | null = null;
  private hoverId: number | null = null;
  private selected: number | null = null;
  private pointer: PointerState | null = null;
  private press: Press | null = null;
  private drag: Drag | null = null;
  private lastClick = { t: -1e9, x: -1e9, y: -1e9 };
  private dirty = true;
  private lastCamKey = '';
  private lastSceneVersion = -1;
  private commitThrottle: TrailingThrottle<Pose>;
  private xformThrottle: TrailingThrottle<{ id: number; pose: Pose; label: string }>;
  private xformLive: { id: number; pose: Pose } | null = null;
  private lastRotateAt = -1e9;
  private lastTiltAt = -1e9;
  private time = 0;
  private shapes: OverlayShape[] = [];
  private segs: OverlaySegment[] = [];
  private label: HTMLDivElement | null = null;
  private lastStatus = '';
  private unsubs: (() => unknown)[] = [];
  private destroyed = false;

  constructor(
    private ctx: ModuleContext,
    private canvas: HTMLCanvasElement,
    private camera: Camera,
    private orbit: OrbitController,
    private terrain: TerrainSystem,
  ) {
    this.restoreAfterRebuild();
    this.history = new SceneHistory(ctx.scene);
    this.overlay = new OverlayRenderer(ctx.gpu.device, ctx.shared);
    const d = ctx.world.domain;
    this.ground = bilinearGround((x, z) => this.heightAtSafe(x, z), d.cellSize);
    this.variation = rollVariation();
    this.commitThrottle = new TrailingThrottle(COMMIT_INTERVAL_MS, (p) => this.commitDragPose(p));
    this.xformThrottle = new TrailingThrottle(COMMIT_INTERVAL_MS, (x) => this.commitXform(x));

    canvas.addEventListener('pointerdown', this.onPointerDown, { capture: true });
    canvas.addEventListener('wheel', this.onWheel, { capture: true, passive: false });
    canvas.addEventListener('dblclick', this.onDblClick);
    canvas.addEventListener('pointerleave', this.onPointerLeave);
    canvas.addEventListener(EDITOR_COMMAND_EVENT, this.onCommand as EventListener);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('pointercancel', this.onPointerCancel);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('visibilitychange', this.onVisibility);

    this.unsubs.push(ctx.scene.onChange((c) => this.onSceneChange(c)));
    this.unsubs.push(ctx.params.on('editor', (path) => this.onEditorParam(path)));
    this.unsubs.push(this.history.onChange(() => (this.dirty = true)));

    // Camera helpers through the OrbitController contract.
    orbit.ground = this.orbitGround;
    orbit.pick = this.orbitPick;
    orbit.bounds = {
      min: [-0.25 * d.sizeX, d.minY - 0.2, -0.25 * d.sizeZ],
      max: [1.25 * d.sizeX, d.maxY + 0.6, 1.25 * d.sizeZ],
    };

    this.createLabel();
    this.emitStatus();
  }

  // ------------------------------------------------------------------------------------------
  // World rebuild (quality switch) keeps the user's edits of the same preset.

  private restoreAfterRebuild() {
    const stash = rebuildStash;
    rebuildStash = null;
    const { scene, preset } = this.ctx;
    if (!stash || stash.presetId !== preset.id) return;
    const same =
      stash.obstacles.length === scene.obstacles.length &&
      stash.obstacles.every((o, i) => JSON.stringify(o) === JSON.stringify(cloneData(scene.obstacles[i])));
    if (same) return;
    [...scene.obstacles].forEach((o) => scene.remove(o.id));
    stash.obstacles.forEach((o) => scene.add(cloneData(o)));
  }

  // ------------------------------------------------------------------------------------------
  // Queries

  private heightAtSafe(x: number, z: number): number {
    const d = this.ctx.world.domain;
    const cx = clamp(x, d.cellSize * 0.5, d.sizeX - d.cellSize * 0.5);
    const cz = clamp(z, d.cellSize * 0.5, d.sizeZ - d.cellSize * 0.5);
    try {
      const h = this.terrain.heightAt(cx, cz);
      return Number.isFinite(h) ? h : NaN;
    } catch {
      return NaN;
    }
  }

  private orbitGround = (x: number, z: number) => this.heightAtSafe(x, z);

  private orbitPick = (nx: number, ny: number): V3 | null => {
    const r = this.camera.rayFromNdc(nx, ny);
    return this.raycast(r.origin, r.dir)?.point ?? null;
  };

  private raycast(origin: V3, dir: V3): RayHit | null {
    if (![...origin, ...dir].every(Number.isFinite)) return null;
    try {
      const h = this.terrain.raycast(origin, dir, 60);
      if (!h || !h.point.every(Number.isFinite)) return null;
      return h;
    } catch {
      return null;
    }
  }

  private rayAt(clientX: number, clientY: number): { origin: V3; dir: V3 } | null {
    const r = this.canvas.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) return null;
    const nx = ((clientX - r.left) / r.width) * 2 - 1;
    const ny = 1 - ((clientY - r.top) / r.height) * 2;
    const ray = this.camera.rayFromNdc(nx, ny);
    return [...ray.origin, ...ray.dir].every(Number.isFinite) ? ray : null;
  }

  /** Obstacle (if any) and surface point under a screen position. */
  pickAt(clientX: number, clientY: number): { id?: number; point: V3 | null } {
    const ray = this.rayAt(clientX, clientY);
    if (!ray) return { point: null };
    const hit = this.raycast(ray.origin, ray.dir);
    const obs = this.ctx.scene.obstacles;
    const proxy = pickProxies(obs, ray.origin, ray.dir);
    const id = resolvePick(hit, proxy, obs);
    let point: V3 | null = hit?.point ? [...hit.point] : null;
    if (id !== undefined && proxy && proxy.id === id && (!hit || hit.obstacleId !== id)) {
      point = add(ray.origin, vscale(ray.dir, proxy.t));
    }
    return { id, point };
  }

  private get tool(): EditorTool {
    const t = this.ctx.params.values.editor.tool;
    return EDITOR_TOOLS.includes(t) ? t : 'orbit';
  }

  private setTool(t: EditorTool) {
    this.ctx.params.set('editor.tool', t);
  }

  private variationFor(kind: ObstacleKind): Variation {
    if (kind !== 'log') return this.variation;
    // Logs default to lying across the stream, with a little randomness.
    return { ...this.variation, yaw: Math.PI / 2 + (this.variation.yaw / (2 * Math.PI) - 0.5) * 0.5 };
  }

  private cameraDistanceTo(p: readonly number[]) {
    return Math.max(0.01, Math.hypot(p[0] - this.camera.position[0], p[1] - this.camera.position[1], p[2] - this.camera.position[2]));
  }

  private worldPerPixel(p: readonly number[]) {
    const h = Math.max(1, this.canvas.clientHeight || this.canvas.height || 600);
    return (2 * this.cameraDistanceTo(p) * Math.tan(this.camera.fovY / 2)) / h;
  }

  private cameraRightXZ(): V3 {
    const f = this.camera.forward();
    const r = normalize(cross(f, [0, 1, 0]), [1, 0, 0]);
    return normalize([r[0], 0, r[2]], [1, 0, 0]);
  }

  private cameraForwardXZ(): V3 {
    const f = this.camera.forward();
    return normalize([f[0], 0, f[2]], [0, 0, -1]);
  }

  // ------------------------------------------------------------------------------------------
  // Pointer input

  private consume(e: Event) {
    (e as any).__consumed = true;
  }

  private updatePointer(e: PointerEvent | WheelEvent | MouseEvent, inside?: boolean) {
    const ins = inside ?? e.target === this.canvas;
    this.pointer = { x: e.clientX, y: e.clientY, inside: ins, shift: e.shiftKey, alt: e.altKey, type: (e as PointerEvent).pointerType ?? 'mouse' };
    this.dirty = true;
  }

  private onPointerDown = (e: PointerEvent) => {
    if (this.destroyed) return;
    this.updatePointer(e, true);
    if (this.press || this.drag) {
      // A second finger / button while editing: keep it away from the orbit controller.
      if (e.pointerType === 'touch') this.consume(e);
      return;
    }
    if (e.button !== 0) return;
    const tool = this.tool;
    if (tool === 'orbit') return;
    const base = { x: e.clientX, y: e.clientY, pointerId: e.pointerId, moved: false };
    if (tool === 'add') {
      // Not consumed: dragging still orbits; a click (no movement) places on release.
      this.ghostHiddenAt = null;
      this.press = { kind: 'add', ...base };
      if (e.pointerType !== 'mouse') this.updateGhost();
      return;
    }
    const pk = this.pickAt(e.clientX, e.clientY);
    if (pk.id === undefined) {
      this.press = { kind: 'empty', ...base };
      return;
    }
    this.consume(e);
    e.preventDefault();
    if (tool === 'move') {
      this.select(pk.id);
      this.press = { kind: 'move', id: pk.id, ...base };
    } else {
      this.hoverId = pk.id;
      this.press = { kind: 'delete', id: pk.id, ...base };
    }
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic / inactive pointer */
    }
  };

  private onPointerMove = (e: PointerEvent) => {
    if (this.destroyed) return;
    const overCanvas = e.target === this.canvas || !!this.drag;
    this.updatePointer(e, overCanvas);
    const p = this.press;
    if (p && p.pointerId === e.pointerId && !p.moved && Math.hypot(e.clientX - p.x, e.clientY - p.y) > CLICK_SLOP_PX) {
      p.moved = true;
      if (p.kind === 'move' && p.id !== undefined && !this.drag) this.startDrag(p);
    }
    if (this.ghostHiddenAt && Math.hypot(e.clientX - this.ghostHiddenAt[0], e.clientY - this.ghostHiddenAt[1]) > 6) this.ghostHiddenAt = null;
  };

  private onPointerUp = (e: PointerEvent) => {
    if (this.destroyed) return;
    if (this.pointer) this.updatePointer(e, e.target === this.canvas);
    const p = this.press;
    if (this.drag && this.drag.pointerId === e.pointerId) {
      this.updateDrag(); // final position under the release point
      this.endDrag(true);
    } else if (p && p.pointerId === e.pointerId && !p.moved) {
      const dbl = performance.now() - this.lastClick.t < DOUBLE_CLICK_MS && Math.hypot(e.clientX - this.lastClick.x, e.clientY - this.lastClick.y) < 8;
      this.lastClick = { t: performance.now(), x: e.clientX, y: e.clientY };
      if (!dbl) this.click(p, e);
    }
    this.press = null;
    if (e.pointerType === 'touch' && this.pointer) this.pointer.inside = false;
    try {
      if (this.canvas.hasPointerCapture?.(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };

  private onPointerCancel = (e: PointerEvent) => {
    if (this.drag && this.drag.pointerId === e.pointerId) this.endDrag(true);
    if (this.press?.pointerId === e.pointerId) this.press = null;
  };

  private onPointerLeave = () => {
    if (this.pointer && !this.drag) this.pointer.inside = false;
    this.dirty = true;
  };

  private click(p: Press, e: PointerEvent) {
    const tool = this.tool;
    if (p.kind === 'add' && tool === 'add') {
      this.updatePointer(e, true);
      this.updateGhost();
      this.placeGhost();
    } else if (p.kind === 'empty' && tool === 'move') {
      this.deselect();
    } else if (p.kind === 'delete' && tool === 'delete' && p.id !== undefined) {
      const pk = this.pickAt(e.clientX, e.clientY);
      if (pk.id === p.id) this.deleteObstacle(p.id);
    }
  }

  private onDblClick = (e: MouseEvent) => {
    if (this.destroyed) return;
    const pk = this.pickAt(e.clientX, e.clientY);
    if (!pk.point) return;
    if (this.tool === 'move' && pk.id !== undefined) this.select(pk.id);
    const dist = this.orbit.goalDistance;
    this.orbit.focusOn(pk.point, Math.min(dist, Math.max(0.15, dist * 0.7)));
  };

  private onWheel = (e: WheelEvent) => {
    if (this.destroyed || !(e.altKey || e.shiftKey)) return;
    const tool = this.tool;
    const hasSel = tool === 'move' && this.selected !== null && !!this.ctx.scene.get(this.selected);
    if (tool !== 'add' && !hasSel) return;
    this.consume(e);
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const d = clamp((e.deltaY || e.deltaX || 0) * unit, -400, 400);
    if (!d) return;
    if (e.altKey) {
      const f = Math.exp(-d * 0.0015);
      if (tool === 'add') this.resizeGhost(f);
      else this.scaleSelected(f);
    } else {
      const a = -d * 0.003;
      if (tool === 'add') this.rotateGhost(a);
      else this.rotateSelected(a);
    }
  };

  // ------------------------------------------------------------------------------------------
  // Keyboard & commands

  private onKeyDown = (e: KeyboardEvent) => {
    if (this.destroyed || isTypingTarget(e.target)) return;
    if (this.pointer) {
      this.pointer.shift = e.shiftKey;
      this.pointer.alt = e.altKey;
    }
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key;
    const lower = k.length === 1 ? k.toLowerCase() : k;
    if (mod) {
      if (lower === 'z') {
        e.preventDefault();
        if (e.shiftKey) this.redo();
        else this.undo();
      } else if (lower === 'y') {
        e.preventDefault();
        this.redo();
      } else if (lower === 'd') {
        e.preventDefault();
        this.duplicateSelected();
      }
      return;
    }
    if (e.altKey && lower !== 'Alt') return;
    const tool = this.tool;
    const sel = tool === 'move' && this.selected !== null;
    const fine = e.shiftKey ? 1 / 3 : 1;
    let handled = true;
    switch (lower) {
      case '1':
      case '2':
      case '3':
      case '4':
        this.setTool(EDITOR_TOOLS[Number(lower) - 1]);
        break;
      case 'Escape':
        this.escape();
        break;
      case 'Delete':
      case 'Backspace':
        if (sel) this.deleteObstacle(this.selected!);
        else handled = false;
        break;
      case 'q':
      case 'e': {
        const a = (lower === 'q' ? 1 : -1) * YAW_STEP * fine;
        if (tool === 'add') this.rotateGhost(a);
        else if (sel) this.rotateSelected(a);
        else handled = false;
        break;
      }
      case 't':
      case 'g':
        if (sel) this.tiltSelected((lower === 't' ? 1 : -1) * TILT_STEP * fine);
        else handled = false;
        break;
      case '[':
      case ']':
      case '{':
      case '}': {
        const f = lower === '[' || lower === '{' ? 1 / SCALE_STEP : SCALE_STEP;
        if (tool === 'add') this.resizeGhost(f);
        else if (sel) this.scaleSelected(f);
        else handled = false;
        break;
      }
      case 'r':
        if (tool === 'add') this.reroll();
        else handled = false;
        break;
      case 'k': {
        const kinds = OBSTACLE_KINDS;
        const i = kinds.indexOf(this.ctx.params.values.editor.addKind);
        this.ctx.params.set('editor.addKind', kinds[(i + (e.shiftKey ? kinds.length - 1 : 1)) % kinds.length]);
        if (tool !== 'add') this.setTool('add');
        break;
      }
      case 'f':
        this.focus();
        break;
      case 'p':
        this.ctx.params.set('sim.paused', !this.ctx.params.values.sim.paused);
        break;
      case 'ArrowLeft':
      case 'ArrowRight':
      case 'ArrowUp':
      case 'ArrowDown': {
        if (!sel) {
          handled = false;
          break;
        }
        const step = e.shiftKey ? 0.002 : 0.01;
        const r = this.cameraRightXZ();
        const f = this.cameraForwardXZ();
        const dir = lower === 'ArrowLeft' ? vscale(r, -1) : lower === 'ArrowRight' ? r : lower === 'ArrowUp' ? f : vscale(f, -1);
        this.nudgeSelected(vscale(dir, step), 0);
        break;
      }
      case 'PageUp':
      case 'PageDown':
        if (sel) this.nudgeSelected([0, 0, 0], (lower === 'PageUp' ? 1 : -1) * (e.shiftKey ? 0.001 : 0.005));
        else handled = false;
        break;
      default:
        handled = false;
    }
    if (handled) {
      e.preventDefault();
      this.dirty = true;
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    if (this.pointer) {
      this.pointer.shift = e.shiftKey;
      this.pointer.alt = e.altKey;
      this.dirty = true;
    }
  };

  private onBlur = () => {
    if (this.drag) this.endDrag(true);
    this.press = null;
    if (this.pointer) {
      this.pointer.shift = false;
      this.pointer.alt = false;
    }
    this.xformThrottle.flush();
  };

  private onVisibility = () => {
    if (document.visibilityState === 'hidden') this.onBlur();
  };

  private onCommand = (e: CustomEvent<{ cmd: EditorCommand }>) => {
    switch (e.detail?.cmd) {
      case 'undo':
        this.undo();
        break;
      case 'redo':
        this.redo();
        break;
      case 'delete':
        if (this.selected !== null) this.deleteObstacle(this.selected);
        break;
      case 'duplicate':
        this.duplicateSelected();
        break;
      case 'deselect':
        this.deselect();
        break;
      case 'focus':
        this.focus();
        break;
      case 'reroll':
        this.reroll();
        break;
      case 'status':
        this.lastStatus = '';
        this.emitStatus();
        break;
    }
  };

  private escape() {
    if (this.drag) this.endDrag(false);
    else if (this.selected !== null) this.deselect();
    else if (this.tool !== 'orbit') this.setTool('orbit');
  }

  private onEditorParam(path: string) {
    if (path === 'editor.tool' || path === 'editor') {
      if (this.drag) this.endDrag(true);
      this.press = null;
      this.xformThrottle.flush();
      if (this.tool !== 'move') this.selected = null;
      this.hoverId = null;
      this.ghost = null;
    }
    this.dirty = true;
  }

  private onSceneChange(c: SceneChange) {
    this.dirty = true;
    if (c.type === 'reset') {
      if (this.drag) {
        this.commitThrottle.cancel();
        this.history.end();
        this.drag = null;
      }
      this.xformThrottle.cancel();
      this.xformLive = null;
      this.selected = null;
      this.hoverId = null;
      return;
    }
    if (c.type === 'obstacle-removed') {
      const id = c.obstacle.id;
      if (this.drag?.id === id) {
        this.commitThrottle.cancel();
        this.drag = null;
        this.history.end();
      }
      if (this.xformLive?.id === id) {
        this.xformThrottle.cancel();
        this.xformLive = null;
      }
      if (this.selected === id) this.selected = null;
      if (this.hoverId === id) this.hoverId = null;
    }
  }

  // ------------------------------------------------------------------------------------------
  // Actions

  private select(id: number) {
    if (this.selected !== id) this.xformThrottle.flush();
    this.selected = this.ctx.scene.get(id) ? id : null;
    this.dirty = true;
  }

  private deselect() {
    this.xformThrottle.flush();
    this.selected = null;
    this.dirty = true;
  }

  private newVariation() {
    this.variation = rollVariation();
  }

  reroll() {
    this.newVariation();
    this.dirty = true;
  }

  private rotateGhost(a: number) {
    this.ghostYaw += a;
    this.lastRotateAt = this.time;
    this.dirty = true;
  }

  private resizeGhost(f: number) {
    const ed = this.ctx.params.values.editor;
    const s = clamp((Number.isFinite(ed.addSize) ? ed.addSize : 0.15) * f, ADD_SIZE_RANGE[0], ADD_SIZE_RANGE[1]);
    this.ctx.params.set('editor.addSize', Math.round(s * 1000) / 1000);
  }

  /** Places the current ghost. Returns the new obstacle (or null). */
  placeGhost(): Obstacle | null {
    const g = this.ghost;
    if (!g || !g.valid) return null;
    const p = g.placed;
    const o = this.history.transact(`add ${p.kind}`, () =>
      this.ctx.scene.add({ kind: p.kind, position: [...p.position], rotation: [...p.rotation], scale: [...p.scale], seed: p.seed, roughness: p.roughness, moss: p.moss }),
    );
    this.newVariation();
    this.ghost = null;
    if (this.pointer) this.ghostHiddenAt = [this.pointer.x, this.pointer.y];
    this.dirty = true;
    return o;
  }

  private deleteObstacle(id: number) {
    if (this.drag?.id === id) this.endDrag(false);
    this.xformThrottle.flush();
    if (!this.ctx.scene.get(id)) return;
    const kind = this.ctx.scene.get(id)!.kind;
    this.history.transact(`delete ${kind}`, () => this.ctx.scene.remove(id));
    if (this.selected === id) this.selected = null;
    if (this.hoverId === id) this.hoverId = null;
    this.dirty = true;
  }

  undo() {
    if (this.drag) return;
    this.xformThrottle.flush();
    this.afterHistory(this.history.undo());
  }

  redo() {
    if (this.drag) return;
    this.xformThrottle.flush();
    this.afterHistory(this.history.redo());
  }

  private afterHistory(ids: number[] | null) {
    if (ids === null) return;
    this.hoverId = null;
    const alive = ids.filter((id) => this.ctx.scene.get(id));
    if (this.tool === 'move' && alive.length) this.selected = alive[alive.length - 1];
    else if (this.selected !== null && !this.ctx.scene.get(this.selected)) this.selected = null;
    this.dirty = true;
  }

  private focus() {
    const o = this.selected !== null ? this.ctx.scene.get(this.selected) : undefined;
    if (o) {
      this.orbit.focusOn([...o.position], clamp(6 * maxHalfExtent(o.scale), 0.1, 3));
      return;
    }
    const p = this.ghost?.placed.position ?? (this.pointer ? this.pickAt(this.pointer.x, this.pointer.y).point : null);
    if (p) this.orbit.focusOn([...p], Math.min(this.orbit.goalDistance, 0.6));
  }

  // --- transforms of the selection (keyboard / wheel); rate limited, coalesced in the history

  private transformSelected(label: string, fn: (p: Pose, kind: ObstacleKind) => Pose) {
    const id = this.selected;
    if (id === null) return;
    const o = this.ctx.scene.get(id);
    if (!o) return;
    if (this.drag && this.drag.id === id) {
      const d = this.drag;
      d.live = fn(clonePose(d.live), o.kind);
      d.liftRef = { clientY: this.pointer?.y ?? 0, y: d.live.position[1] };
      d.embed = d.live.position[1] - groundUnder(this.dragGround(), { kind: o.kind, ...d.live }, d.live.position[1]);
      this.commitThrottle.push(clonePose(d.live));
      return;
    }
    const base = this.xformLive && this.xformLive.id === id ? this.xformLive.pose : poseOf(o);
    const next = fn(clonePose(base), o.kind);
    if (![...next.position, ...next.rotation, ...next.scale].every(Number.isFinite)) return;
    clampPositionToDomain(next.position, this.ctx.world.domain);
    this.xformLive = { id, pose: next };
    this.xformThrottle.push({ id, pose: next, label });
    this.dirty = true;
  }

  private commitXform(x: { id: number; pose: Pose; label: string }) {
    if (!this.ctx.scene.get(x.id)) return;
    this.history.transact(x.label, () => this.ctx.scene.update(x.id, clonePose(x.pose)), `xf:${x.id}`);
    if (this.xformLive && this.xformLive.pose === x.pose) this.xformLive = null;
  }

  /** Ground with the given obstacle's own (baked) footprint replaced by the surrounding ground. */
  private groundWithout(kind: ObstacleKind, p: Pose): GroundFn {
    return maskedGround(this.ground, [footprintOf({ kind, ...p })]);
  }

  rotateSelected(a: number) {
    this.lastRotateAt = this.time;
    this.transformSelected('rotate', (p) => ({ ...p, rotation: qRotateYaw(p.rotation, a) }));
  }

  tiltSelected(a: number) {
    this.lastTiltAt = this.time;
    const camRight = this.cameraRightXZ();
    this.transformSelected('tilt', (p) => ({ ...p, rotation: qTilt(p.rotation, a, camRight) }));
  }

  scaleSelected(f: number) {
    this.transformSelected('scale', (p, kind) => {
      const mx = Math.max(...p.scale);
      const mn = Math.min(...p.scale);
      const k = clamp(f, MIN_HALF_EXTENT / Math.max(mn, 1e-6), MAX_HALF_EXTENT / Math.max(mx, 1e-6));
      const g = groundUnder(this.groundWithout(kind, p), { kind, ...p }, p.position[1]);
      const y = g + (p.position[1] - g) * k;
      return { ...p, position: [p.position[0], y, p.position[2]], scale: p.scale.map((s) => s * k) as V3 };
    });
  }

  nudgeSelected(dxz: readonly number[], dy: number) {
    this.transformSelected(dy ? 'raise' : 'nudge', (p, kind) => {
      const ground = this.groundWithout(kind, p);
      const g0 = groundUnder(ground, { kind, ...p }, p.position[1]);
      const np: V3 = [p.position[0] + dxz[0], p.position[1], p.position[2] + dxz[2]];
      clampPositionToDomain(np, this.ctx.world.domain);
      if (dxz[0] || dxz[2]) np[1] = groundUnder(ground, { kind, ...p, position: np }, g0) + (p.position[1] - g0);
      np[1] += dy;
      return { ...p, position: np };
    });
  }

  duplicateSelected() {
    const id = this.selected;
    if (id === null || this.drag) return;
    this.xformThrottle.flush();
    const o = this.ctx.scene.get(id);
    if (!o) return;
    const p = poseOf(o);
    const own = this.groundWithout(o.kind, p);
    const embed = p.position[1] - groundUnder(own, o, p.position[1]);
    const r = footprintRadius(footprintOf(o));
    const off = vscale(this.cameraRightXZ(), 2.1 * r + 0.01);
    const np: V3 = [p.position[0] + off[0], p.position[1], p.position[2] + off[2]];
    clampPositionToDomain(np, this.ctx.world.domain);
    np[1] = groundUnder(this.ground, { ...o, position: np }, p.position[1]) + embed;
    clampPositionToDomain(np, this.ctx.world.domain);
    const n = this.history.transact(`duplicate ${o.kind}`, () => this.ctx.scene.add({ ...cloneData(o), position: np }));
    this.selected = n.id;
    this.dirty = true;
  }

  // --- dragging

  private dragGround(): GroundFn {
    const d = this.drag;
    if (!d) return this.ground;
    return maskedGround(
      this.ground,
      d.committed.map((c) => footprintOf({ kind: d.kind, ...c.pose })),
    );
  }

  private cursorGround(ground: GroundFn, ray: { origin: V3; dir: V3 }, fallbackY: number): V3 | null {
    const hit = marchGround(ground, ray.origin, ray.dir, this.ctx.world.domain);
    if (hit) return hit.point;
    const t = rayPlaneY(ray.origin, ray.dir, fallbackY);
    if (t === null) return null;
    const d = this.ctx.world.domain;
    const maxT = 4 * Math.hypot(d.sizeX, d.sizeZ) + this.cameraDistanceTo(ray.origin);
    return add(ray.origin, vscale(ray.dir, Math.min(t, maxT)));
  }

  private startDrag(p: Press) {
    const o = p.id !== undefined ? this.ctx.scene.get(p.id) : undefined;
    if (!o) return;
    this.xformThrottle.flush();
    const start = poseOf(o);
    this.drag = {
      id: o.id,
      kind: o.kind,
      start,
      live: clonePose(start),
      mode: 'slide',
      offset: [0, 0],
      embed: 0,
      liftRef: { clientY: p.y, y: start.position[1] },
      committed: [{ pose: clonePose(start), t: performance.now() }],
      pointerId: p.pointerId,
    };
    const ray = this.rayAt(p.x, p.y);
    const ground = this.dragGround();
    const g0 = ray ? this.cursorGround(ground, ray, start.position[1]) : null;
    this.drag.offset = g0 ? [start.position[0] - g0[0], start.position[2] - g0[2]] : [0, 0];
    this.drag.embed = start.position[1] - groundUnder(ground, o, start.position[1]);
    this.drag.mode = this.pointer && (this.pointer.shift || this.pointer.alt) ? 'lift' : 'slide';
    this.commitThrottle.reset();
    this.history.begin(`move ${o.kind}`);
    this.dirty = true;
  }

  private updateDrag() {
    const d = this.drag;
    const ptr = this.pointer;
    if (!d || !ptr) return;
    const mode: Drag['mode'] = ptr.shift || ptr.alt ? 'lift' : 'slide';
    const ground = this.dragGround();
    const ray = this.rayAt(ptr.x, ptr.y);
    if (mode !== d.mode) {
      d.mode = mode;
      if (mode === 'lift') d.liftRef = { clientY: ptr.y, y: d.live.position[1] };
      else if (ray) {
        const g = this.cursorGround(ground, ray, d.live.position[1]);
        if (g) d.offset = [d.live.position[0] - g[0], d.live.position[2] - g[2]];
        d.embed = d.live.position[1] - groundUnder(ground, { kind: d.kind, ...d.live }, d.live.position[1]);
      }
    }
    const dom = this.ctx.world.domain;
    if (d.mode === 'slide') {
      if (!ray) return;
      const g = this.cursorGround(ground, ray, d.live.position[1] - d.embed);
      if (!g) return;
      const np: V3 = [g[0] + d.offset[0], d.live.position[1], g[2] + d.offset[1]];
      clampPositionToDomain(np, dom);
      np[1] = groundUnder(ground, { kind: d.kind, ...d.live, position: np }, d.live.position[1] - d.embed) + d.embed;
      clampPositionToDomain(np, dom);
      d.live.position = np;
    } else {
      const s = this.worldPerPixel(d.live.position);
      d.live.position[1] = clamp(d.liftRef.y + (d.liftRef.clientY - ptr.y) * s, dom.minY, dom.maxY);
    }
    if (!samePose(d.live, d.committed[d.committed.length - 1].pose)) this.commitThrottle.push(clonePose(d.live));
  }

  private commitDragPose(p: Pose) {
    const d = this.drag;
    if (!d || !this.ctx.scene.get(d.id)) return;
    this.ctx.scene.update(d.id, clonePose(p));
    const t = performance.now();
    d.committed.push({ pose: clonePose(p), t });
    // Keep footprints the terrain may still have baked (re-bakes lag a frame or two).
    while (d.committed.length > 2 && t - d.committed[0].t > 500) d.committed.shift();
    if (d.committed.length > 12) d.committed.splice(0, d.committed.length - 12);
  }

  private endDrag(commit: boolean) {
    const d = this.drag;
    if (!d) return;
    if (commit) {
      this.commitThrottle.flush();
      if (!samePose(d.live, d.committed[d.committed.length - 1].pose)) this.commitDragPose(d.live);
    } else {
      this.commitThrottle.cancel();
      if (this.ctx.scene.get(d.id)) this.ctx.scene.update(d.id, clonePose(d.start));
    }
    this.drag = null;
    this.history.end();
    this.dirty = true;
  }

  // ------------------------------------------------------------------------------------------
  // Per-frame update

  private updateGhost() {
    const ptr = this.pointer;
    if (!ptr || !ptr.inside) {
      this.ghost = null;
      return;
    }
    const ray = this.rayAt(ptr.x, ptr.y);
    const hit = ray ? this.raycast(ray.origin, ray.dir) : null;
    if (!hit) {
      this.ghost = null;
      return;
    }
    const d = this.ctx.world.domain;
    const ed = this.ctx.params.values.editor;
    const kind: ObstacleKind = OBSTACLE_KINDS.includes(ed.addKind) ? ed.addKind : 'boulder';
    const size = clamp(Number.isFinite(ed.addSize) ? ed.addSize : 0.15, ADD_SIZE_RANGE[0], ADD_SIZE_RANGE[1]);
    const [x, , z] = hit.point;
    const valid = x >= 0 && x <= d.sizeX && z >= 0 && z <= d.sizeZ;
    const placed = computePlacement({
      kind,
      size,
      variation: this.variationFor(kind),
      x,
      z,
      ground: this.ground,
      fallbackY: hit.point[1],
      yawOffset: this.ghostYaw,
      domain: d,
    });
    this.ghost = { placed, valid };
  }

  private updateHover() {
    const ptr = this.pointer;
    if (!ptr || !ptr.inside || this.press?.kind === 'move') {
      if (!this.press) this.hoverId = null;
      return;
    }
    this.hoverId = this.pickAt(ptr.x, ptr.y).id ?? null;
  }

  update(frame: FrameContext) {
    if (this.destroyed) return;
    this.time = frame.realTime;
    const vp = this.camera.viewProj;
    const camKey = `${vp[0].toFixed(5)},${vp[5].toFixed(5)},${vp[10].toFixed(6)},${vp[12].toFixed(5)},${vp[13].toFixed(5)},${vp[14].toFixed(5)},${vp[2].toFixed(5)},${vp[8].toFixed(5)}`;
    if (camKey !== this.lastCamKey) {
      this.lastCamKey = camKey;
      this.dirty = true;
    }
    if (this.ctx.scene.version !== this.lastSceneVersion) {
      this.lastSceneVersion = this.ctx.scene.version;
      this.dirty = true;
    }
    if (this.selected !== null && !this.ctx.scene.get(this.selected)) this.selected = null;
    if (this.hoverId !== null && !this.ctx.scene.get(this.hoverId)) this.hoverId = null;
    const tool = this.tool;
    if (this.drag) this.updateDrag();
    else if (this.dirty) {
      if (tool === 'add') this.updateGhost();
      else if (tool === 'move' || tool === 'delete') this.updateHover();
      else this.hoverId = null;
    }
    this.dirty = false;
    this.commitThrottle.tick();
    this.xformThrottle.tick();
    this.buildOverlay();
    this.updateLabel();
    this.emitStatus();
  }

  drawOverlay(encoder: GPUCommandEncoder, frame: FrameContext, target: GPUTextureView) {
    if (this.destroyed || overlayControl.suppressed > 0) return;
    if (!this.shapes.length && !this.segs.length) return;
    const dpr = this.canvas.clientWidth > 0 ? this.canvas.width / this.canvas.clientWidth : 1;
    this.overlay.draw(encoder, frame, target, [this.canvas.width, this.canvas.height], dpr, this.shapes, this.segs);
  }

  // ------------------------------------------------------------------------------------------
  // Overlay content

  private ring(out: OverlaySegment[], kind: ObstacleKind, p: Pose, ground: GroundFn, color: RGBA, alpha: number, width = 0.8, dash = 0) {
    const pts = footprintOutline(footprintOf({ kind, ...p }), 72, 1.08, 0.008);
    const lift = 0.0025 + 0.0025 * this.cameraDistanceTo(p.position);
    const fallback = p.position[1] - p.scale[1] * 0.5;
    const world = pts.map(([x, z]): V3 => {
      const y = ground(x, z);
      return [x, (Number.isFinite(y) ? y : fallback) + lift, z];
    });
    this.polyline(out, world, true, { color: [color[0], color[1], color[2], alpha], width, glow: 3, glowAlpha: 0.22, occluded: 0.35, dash });
  }

  private polyline(
    out: OverlaySegment[],
    pts: V3[],
    closed: boolean,
    style: Omit<OverlaySegment, 'a' | 'b' | 'dashStart'>,
  ) {
    let acc = 0;
    const n = closed ? pts.length : pts.length - 1;
    for (let i = 0; i < n; i++) {
      const a = pts[i];
      const b = pts[(i + 1) % pts.length];
      out.push({ ...style, a, b, dashStart: acc });
      acc += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
  }

  private gizmo(out: OverlaySegment[], kind: ObstacleKind, p: Pose, ground: GroundFn) {
    const f = footprintOf({ kind, ...p });
    const R = footprintRadius(f);
    const cx = p.position[0];
    const cz = p.position[2];
    const gy0 = ground(cx, cz);
    const gy = Number.isFinite(gy0) ? gy0 : p.position[1] - p.scale[1];
    const top = proxyTopY({ kind, ...p });
    const lifting = this.drag?.mode === 'lift' || (this.pointer?.shift && !this.drag && this.hoverId === this.selected);
    const stemTop = top + 0.25 * R + 0.012;
    const c = lifting ? COLORS.selected : COLORS.gizmo;
    const a = lifting ? 0.95 : 0.5;
    out.push({ a: [cx, Math.min(gy, p.position[1]), cz], b: [cx, stemTop, cz], width: lifting ? 1 : 0.6, color: [c[0], c[1], c[2], a], glow: 3, glowAlpha: 0.2, occluded: 0.35 });
    // Up/down chevrons (raise/lower hint).
    const r = this.cameraRightXZ();
    const ch = Math.max(0.006, 0.12 * R);
    for (const dir of [1, -1]) {
      const tip: V3 = [cx, stemTop + (dir > 0 ? ch * 0.9 : -ch * 0.2), cz];
      const wing = (s: number): V3 => [tip[0] + r[0] * ch * 0.6 * s, tip[1] - dir * ch * 0.7, tip[2] + r[2] * ch * 0.6 * s];
      if (dir < 0 && !lifting) continue;
      out.push({ a: wing(-1), b: tip, width: 0.7, color: [c[0], c[1], c[2], a], glow: 2, glowAlpha: 0.2, occluded: 0.3 });
      out.push({ a: tip, b: wing(1), width: 0.7, color: [c[0], c[1], c[2], a], glow: 2, glowAlpha: 0.2, occluded: 0.3 });
    }
    // Yaw arcs with arrow heads facing the camera sides (brighter right after rotating).
    const recent = clamp(1 - (this.time - this.lastRotateAt) / 0.8, 0, 1);
    const ya = 0.28 + 0.6 * recent;
    const yc: RGBA = recent > 0 ? [COLORS.selected[0], COLORS.selected[1], COLORS.selected[2], ya] : [COLORS.gizmo[0], COLORS.gizmo[1], COLORS.gizmo[2], ya];
    const rr = 1.28 * R + 0.012;
    const lift = 0.003 + 0.0025 * this.cameraDistanceTo(p.position);
    const f2 = this.cameraForwardXZ();
    const base = Math.atan2(-f2[2], -f2[0]); // towards the camera
    for (const side of [-1, 1]) {
      const center = base + side * 0.95;
      const span = 0.55;
      const pts: V3[] = [];
      for (let i = 0; i <= 10; i++) {
        const ang = center - span / 2 + (span * i) / 10;
        const x = cx + Math.cos(ang) * rr;
        const z = cz + Math.sin(ang) * rr;
        const y = ground(x, z);
        pts.push([x, (Number.isFinite(y) ? y : gy) + lift, z]);
      }
      this.polyline(out, pts, false, { color: yc, width: 0.7, glow: 2.5, glowAlpha: 0.2, occluded: 0.3 });
      // Arrow head at the end pointing along the arc (counter-clockwise for side > 0).
      const end = side > 0 ? pts[pts.length - 1] : pts[0];
      const prev = side > 0 ? pts[pts.length - 2] : pts[1];
      const t = normalize(sub(end, prev));
      const n: V3 = [-t[2], 0, t[0]];
      const h = Math.max(0.005, 0.08 * rr);
      out.push({ a: add(add(end, vscale(t, -h)), vscale(n, h * 0.6)), b: end, width: 0.7, color: yc, glow: 2, glowAlpha: 0.2, occluded: 0.3 });
      out.push({ a: add(add(end, vscale(t, -h)), vscale(n, -h * 0.6)), b: end, width: 0.7, color: yc, glow: 2, glowAlpha: 0.2, occluded: 0.3 });
    }
    // Tilt hint: the obstacle's local up axis.
    const tr = clamp(1 - (this.time - this.lastTiltAt) / 0.9, 0, 1);
    if (tr > 0) {
      const up = qRotate(p.rotation, [0, 1, 0]);
      const len = Math.max(p.scale[1] * 1.8, 0.03);
      out.push({ a: p.position, b: add(p.position, vscale(up, len)), width: 0.9, color: [COLORS.selected[0], COLORS.selected[1], COLORS.selected[2], 0.9 * tr], glow: 3, glowAlpha: 0.25, occluded: 0.45 });
    }
  }

  private buildOverlay() {
    const shapes: OverlayShape[] = [];
    const segs: OverlaySegment[] = [];
    const tool = this.tool;
    const scene = this.ctx.scene;
    if (tool === 'add' && this.ghost && !this.ghostHiddenAt && this.pointer?.inside) {
      const g = this.ghost;
      const c = g.valid ? COLORS.ghost : COLORS.ghostBad;
      shapes.push({ pose: g.placed, mode: 'ghost', color: c, fill: 0.16, edge: 0.6, occluded: 0.5, seed: g.placed.seed });
      this.ring(segs, g.placed.kind, g.placed, this.ground, c, 0.8, 0.8);
    }
    if ((tool === 'move' || tool === 'delete') && this.hoverId !== null && this.hoverId !== this.selected) {
      const o = scene.get(this.hoverId);
      if (o) {
        const del = tool === 'delete';
        const c = del ? COLORS.danger : COLORS.hover;
        shapes.push({ pose: o, mode: 'highlight', color: c, fill: del ? 0.18 : 0.05, edge: del ? 0.95 : 0.7, occluded: 0.5, pulse: del ? 1 : 0 });
        this.ring(segs, o.kind, poseOf(o), this.ground, c, del ? 0.7 : 0.45, 0.7, del ? 0 : 0.012);
      }
    }
    if (tool === 'move' && this.selected !== null) {
      const o = scene.get(this.selected);
      if (o) {
        const live = this.drag?.id === o.id ? this.drag.live : this.xformLive?.id === o.id ? this.xformLive.pose : poseOf(o);
        const ground = this.drag ? this.dragGround() : this.ground;
        shapes.push({ pose: o, mode: 'highlight', color: COLORS.selected, fill: 0.08, edge: 0.95, occluded: 0.55 });
        if (!samePose(live, poseOf(o))) {
          shapes.push({ pose: { kind: o.kind, ...live }, mode: 'ghost', color: COLORS.selected, fill: 0.06, edge: 0.5, occluded: 0.35, seed: o.seed, noise: 0.04 });
        }
        this.ring(segs, o.kind, live, ground, COLORS.selected, 0.85, 0.85);
        this.gizmo(segs, o.kind, live, ground);
      }
    }
    this.shapes = shapes;
    this.segs = segs;
  }

  // ------------------------------------------------------------------------------------------
  // DOM: floating label + status for the UI

  private createLabel() {
    if (typeof document === 'undefined') return;
    const el = document.createElement('div');
    el.className = 'wf-editor-label';
    Object.assign(el.style, {
      position: 'fixed',
      left: '0px',
      top: '0px',
      pointerEvents: 'none',
      zIndex: '2',
      font: '500 11px/1.2 system-ui, sans-serif',
      color: '#eef5f2',
      background: 'rgba(12, 18, 17, 0.72)',
      border: '1px solid rgba(255,255,255,0.12)',
      padding: '3px 8px',
      borderRadius: '10px',
      whiteSpace: 'nowrap',
      transform: 'translate(-50%, -140%)',
      opacity: '0',
      transition: 'opacity 0.15s',
      backdropFilter: 'blur(4px)',
    } as Partial<CSSStyleDeclaration>);
    document.body.appendChild(el);
    this.label = el;
  }

  private sizeText(kind: ObstacleKind, s: readonly number[]) {
    const cm = (m: number) => `${Math.round(m * 100)}`;
    if (kind === 'log') return `${cm(2 * s[0])} cm long, ⌀ ${cm(s[1] + s[2])} cm`;
    if (kind === 'slab') return `${cm(2 * s[0])} × ${cm(2 * s[2])} × ${cm(2 * s[1])} cm`;
    return `⌀ ${cm(s[0] + s[2])} cm`;
  }

  private updateLabel() {
    const el = this.label;
    if (!el) return;
    let text = '';
    let anchor: V3 | null = null;
    const tool = this.tool;
    if (tool === 'add' && this.ghost && !this.ghostHiddenAt && this.pointer?.inside) {
      const g = this.ghost.placed;
      text = this.ghost.valid ? `${kindTemplate(g.kind).label} · ${this.sizeText(g.kind, g.scale)}` : 'Outside the stream';
      anchor = [g.position[0], proxyTopY(g), g.position[2]];
    } else if (tool === 'move' && this.selected !== null) {
      const o = this.ctx.scene.get(this.selected);
      if (o) {
        const live = this.drag?.live ?? this.xformLive?.pose ?? poseOf(o);
        text = `${kindTemplate(o.kind).label} · ${this.sizeText(o.kind, live.scale)}`;
        if (this.drag?.mode === 'lift') {
          const dh = (this.drag.live.position[1] - this.drag.start.position[1]) * 100;
          text += ` · ${dh >= 0 ? '+' : '−'}${Math.abs(dh).toFixed(1)} cm`;
        }
        anchor = [live.position[0], proxyTopY({ kind: o.kind, ...live }), live.position[2]];
      }
    }
    const hidden = document.body.classList.contains('noui');
    if (!text || !anchor || hidden) {
      el.style.opacity = '0';
      return;
    }
    const m = this.camera.viewProj;
    const p = [anchor[0], anchor[1] + 0.01, anchor[2], 1];
    const clip = [0, 1, 2, 3].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] * p[3]);
    if (!(clip[3] > 1e-4)) {
      el.style.opacity = '0';
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const sx = rect.left + ((clip[0] / clip[3] + 1) / 2) * rect.width;
    const sy = rect.top + ((1 - clip[1] / clip[3]) / 2) * rect.height;
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) return;
    if (el.textContent !== text) el.textContent = text;
    el.style.left = `${Math.round(sx)}px`;
    el.style.top = `${Math.round(sy)}px`;
    el.style.opacity = '1';
  }

  private hint(): string {
    const M = modKey();
    const alt = isMac() ? '⌥' : 'Alt';
    const ed = this.ctx.params.values.editor;
    switch (this.tool) {
      case 'add':
        return `Click to place a ${kindTemplate(ed.addKind).label.toLowerCase()} · ${alt}+wheel or [ ] size · Q/E or Shift+wheel rotate · R new shape · K kind · drag to orbit · Esc done`;
      case 'move':
        if (this.drag?.mode === 'lift') return 'Raising / lowering · release Shift to slide · Esc cancels';
        if (this.drag) return 'Sliding over the bed · hold Shift to raise / lower · Esc cancels';
        if (this.selected !== null)
          return `Drag to slide · Shift+drag raise/lower · Q/E rotate · T/G tilt · [ ] or ${alt}+wheel scale · arrows nudge · Del delete · ${M}D duplicate · Esc deselect`;
        return 'Click a rock or log to select it · drag empty space to orbit · double-click to focus';
      case 'delete':
        return `Click a rock or log to remove it · ${M}Z undo · drag to orbit`;
      default:
        return 'Drag to orbit · right-drag to pan · wheel to zoom · double-click to focus · 2 add rocks · 3 move · H help';
    }
  }

  private emitStatus() {
    const ed = this.ctx.params.values.editor;
    const sel = this.selected !== null ? this.ctx.scene.get(this.selected) : undefined;
    const status: EditorStatus = {
      tool: this.tool,
      addKind: ed.addKind,
      addSize: ed.addSize,
      selected: sel ? { id: sel.id, kind: sel.kind, size: 2 * maxHalfExtent(sel.scale) } : null,
      hoveredId: this.hoverId,
      dragging: this.drag ? this.drag.mode : 'none',
      canUndo: this.history.canUndo,
      canRedo: this.history.canRedo,
      undoLabel: this.history.undoLabel,
      redoLabel: this.history.redoLabel,
      obstacleCount: this.ctx.scene.obstacles.length,
      hint: this.hint(),
    };
    const key = JSON.stringify(status);
    if (key === this.lastStatus) return;
    this.lastStatus = key;
    this.canvas.dispatchEvent(new CustomEvent(EDITOR_STATUS_EVENT, { detail: status }));
  }

  // ------------------------------------------------------------------------------------------

  /** Test / automation hook: current editor state. */
  debugState() {
    return {
      tool: this.tool,
      selected: this.selected,
      hovered: this.hoverId,
      ghost: this.ghost ? { position: [...this.ghost.placed.position], valid: this.ghost.valid } : null,
      dragging: this.drag?.mode ?? 'none',
      undo: this.history.undoDepth,
      redo: this.history.redoDepth,
      overlay: { shapes: this.shapes.length, segments: this.segs.length },
    };
  }

  destroy() {
    if (this.destroyed) return;
    if (this.drag) this.endDrag(true);
    this.xformThrottle.flush();
    this.history.flush();
    rebuildStash = { presetId: this.ctx.preset.id, obstacles: this.ctx.scene.obstacles.map(cloneData) };
    this.destroyed = true;
    const c = this.canvas;
    c.removeEventListener('pointerdown', this.onPointerDown, { capture: true });
    c.removeEventListener('wheel', this.onWheel, { capture: true });
    c.removeEventListener('dblclick', this.onDblClick);
    c.removeEventListener('pointerleave', this.onPointerLeave);
    c.removeEventListener(EDITOR_COMMAND_EVENT, this.onCommand as EventListener);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('pointercancel', this.onPointerCancel);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.unsubs.forEach((u) => u());
    this.history.dispose();
    if (this.orbit.ground === this.orbitGround) this.orbit.ground = null;
    if (this.orbit.pick === this.orbitPick) this.orbit.pick = null;
    this.label?.remove();
    this.label = null;
    this.overlay.destroy();
  }
}
