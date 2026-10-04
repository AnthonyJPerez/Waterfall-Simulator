/**
 * Perspective camera (reversed-Z: near → depth 1, far → depth 0; use depthCompare 'greater'
 * and clear depth to 0) plus an orbit controller tuned for close-up (macro) inspection:
 * smoothed motion with inertia, zoom towards the cursor, pan limits, ground clearance,
 * touch (one-finger orbit, two-finger pinch/pan) and animated fly-to / focus.
 */
import { mat4, vec3 } from 'wgpu-matrix';

export type V3 = [number, number, number];

export class Camera {
  position: V3 = [1.6, 0.6, 1.4];
  target: V3 = [1.0, 0.2, 0.6];
  up: V3 = [0, 1, 0];
  fovY = (50 * Math.PI) / 180;
  near = 0.005;
  far = 400;
  aspect = 1;

  view = mat4.identity();
  proj = mat4.identity();
  viewProj = mat4.identity();
  invViewProj = mat4.identity();
  invProj = mat4.identity();
  invView = mat4.identity();
  prevViewProj = mat4.identity();

  update(aspect: number) {
    this.aspect = aspect;
    mat4.copy(this.viewProj, this.prevViewProj);
    mat4.lookAt(this.position, this.target, this.up, this.view);
    mat4.perspectiveReverseZ(this.fovY, aspect, this.near, this.far, this.proj);
    mat4.multiply(this.proj, this.view, this.viewProj);
    mat4.inverse(this.viewProj, this.invViewProj);
    mat4.inverse(this.proj, this.invProj);
    mat4.inverse(this.view, this.invView);
  }

  forward(): V3 {
    const f = vec3.normalize(vec3.subtract(this.target, this.position));
    return [f[0], f[1], f[2]];
  }

  /** World-space ray through a pixel (ndc in [-1, 1], y up). */
  rayFromNdc(nx: number, ny: number): { origin: V3; dir: V3 } {
    const pNear = vec3.transformMat4([nx, ny, 1], this.invViewProj); // reversed-Z: depth 1 = near
    const pFar = vec3.transformMat4([nx, ny, 0.0001], this.invViewProj);
    const d = vec3.normalize(vec3.subtract(pFar, pNear));
    return { origin: [pNear[0], pNear[1], pNear[2]], dir: [d[0], d[1], d[2]] };
  }
}

export interface OrbitBounds {
  min: V3;
  max: V3;
}

/** Events marked with `__consumed = true` (by the editor, in the capture phase) are ignored. */
const consumed = (e: Event) => !!(e as any).__consumed;

/** The angle equivalent to `a` that is closest to `ref`. */
export const wrapAngleNear = (a: number, ref: number) => ref + Math.atan2(Math.sin(a - ref), Math.cos(a - ref));

const clampN = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

interface OrbitState {
  yaw: number;
  pitch: number;
  distance: number;
  target: V3;
}

/**
 * Orbit / pan / dolly controller.
 *  - Left drag: orbit (unless the editor consumed the pointerdown); 3 px drag threshold so
 *    clicks never nudge the view
 *  - Right / middle drag, or Shift + left drag: pan in the view plane (1:1 with the cursor)
 *  - Wheel / pinch: dolly, anchored at the point under the cursor (when `pick` is set)
 *  - Touch: one finger orbits, two fingers pinch-zoom and pan
 *  - focusOn / flyTo: smooth animated re-targeting (double-click focus is done by the editor)
 * Input moves a goal state; update() eases the camera towards it (with orbit inertia).
 */
export class OrbitController {
  yaw = 0;
  pitch = 0.4;
  distance = 1.2;
  /** Macro close-ups: down to 2 cm from the target. */
  minDistance = 0.02;
  maxDistance = 12;
  minPitch = -0.45;
  maxPitch = 1.55;
  enabled = true;
  rotateSpeed = 0.0055;
  zoomSpeed = 0.0012;
  /** Easing rate towards the goal (1/s). */
  smoothing = 22;
  /** Orbit inertia decay (1/s). */
  inertiaDecay = 6;
  /** Ground height query (m) used to keep the camera above the bed. */
  ground: ((x: number, z: number) => number) | null = null;
  groundClearance = 0.008;
  /** Target (pan) limits. */
  bounds: OrbitBounds | null = null;
  /** Cursor pick (NDC → world point) for zoom-to-cursor. */
  pick: ((nx: number, ny: number) => V3 | null) | null = null;

  private goal: OrbitState = { yaw: 0, pitch: 0.4, distance: 1.2, target: [0, 0, 0] };
  private vel = { yaw: 0, pitch: 0 };
  private mode: 'orbit' | 'pan' | 'pinch' | null = null;
  private pending: 'orbit' | 'pan' | null = null;
  private downAt = [0, 0];
  private last = [0, 0];
  private recent: { t: number; dyaw: number; dpitch: number }[] = [];
  private touches = new Map<number, [number, number]>();
  private pinchRef: { spread: number; mid: [number, number] } | null = null;
  private anim: { rate: number; t: number; duration: number } | null = null;
  private win: (Window & typeof globalThis) | null = typeof window !== 'undefined' ? window : null;

  constructor(private camera: Camera, private element: HTMLElement) {
    this.syncFromCamera();
    element.addEventListener('pointerdown', this.onDown);
    this.win?.addEventListener('pointermove', this.onMove);
    this.win?.addEventListener('pointerup', this.onUp);
    this.win?.addEventListener('pointercancel', this.onUp);
    this.win?.addEventListener('blur', this.onBlur);
    element.addEventListener('wheel', this.onWheel, { passive: false });
    element.addEventListener('contextmenu', this.onContextMenu);
  }

  /** True while the user drags / pinches, an animation runs or inertia is coasting. */
  get isInteracting() {
    return this.mode !== null || this.anim !== null || Math.abs(this.vel.yaw) + Math.abs(this.vel.pitch) > 1e-3;
  }

  /** Where the camera is heading (after easing). */
  get goalTarget(): V3 {
    return [...this.goal.target];
  }

  get goalDistance() {
    return this.goal.distance;
  }

  syncFromCamera() {
    const c = this.camera;
    const d = vec3.subtract(c.position, c.target);
    const len = vec3.length(d);
    this.distance = Number.isFinite(len) && len > 1e-6 ? len : 1;
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / this.distance)));
    if (!Number.isFinite(this.yaw)) this.yaw = 0;
    if (!Number.isFinite(this.pitch)) this.pitch = 0.4;
    this.goal = { yaw: this.yaw, pitch: this.pitch, distance: this.distance, target: [...c.target] };
  }

  /** Immediately places the camera (no easing). */
  setPose(position: V3, target: V3) {
    if (![...position, ...target].every(Number.isFinite)) return;
    this.camera.position = [...position];
    this.camera.target = [...target];
    this.syncFromCamera();
    this.anim = null;
    this.vel = { yaw: 0, pitch: 0 };
  }

  /** Smoothly moves the orbit target (and optionally the distance). */
  focusOn(target: V3, distance?: number) {
    if (!target.every(Number.isFinite)) return;
    this.goal.target = [...target];
    if (distance !== undefined && Number.isFinite(distance)) this.goal.distance = distance;
    this.anim = { rate: 7, t: 0, duration: 1.4 };
    this.vel = { yaw: 0, pitch: 0 };
  }

  /** Smoothly flies to a full camera pose. */
  flyTo(position: V3, target: V3, rate = 4.5) {
    if (![...position, ...target].every(Number.isFinite)) return;
    const d = vec3.subtract(position, target);
    const dist = Math.max(vec3.length(d), 1e-4);
    this.goal = {
      yaw: wrapAngleNear(Math.atan2(d[0], d[2]), this.yaw),
      pitch: Math.asin(Math.max(-1, Math.min(1, d[1] / dist))),
      distance: dist,
      target: [...target],
    };
    this.anim = { rate, t: 0, duration: 3 };
    this.vel = { yaw: 0, pitch: 0 };
  }

  /** Dolly by `factor` (< 1 = closer), anchored at the world point under `ndc` (if pickable). */
  zoom(factor: number, ndc?: [number, number]) {
    if (!Number.isFinite(factor) || factor <= 0) return;
    const g = this.goal;
    const nd = clampN(g.distance * factor, this.minDistance, this.maxDistance);
    const f = nd / g.distance;
    if (ndc && this.pick && Math.abs(f - 1) > 1e-6) {
      const h = this.pick(ndc[0], ndc[1]);
      if (h && h.every(Number.isFinite)) {
        // Scaling the whole camera rig about h keeps h fixed under the cursor.
        for (let i = 0; i < 3; i++) g.target[i] = h[i] + (g.target[i] - h[i]) * f;
      }
    }
    g.distance = nd;
    this.anim = null;
  }

  /** Pans the goal target by screen pixels (CSS px). */
  panPixels(dx: number, dy: number) {
    const c = this.camera;
    const fwd = vec3.normalize(vec3.subtract(c.target, c.position));
    let right = vec3.cross(fwd, [0, 1, 0]);
    if (vec3.length(right) < 1e-6) right = vec3.create(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    right = vec3.normalize(right);
    const up = vec3.cross(right, fwd);
    const h = Math.max(1, this.element.clientHeight || 600);
    const s = (2 * this.goal.distance * Math.tan(c.fovY / 2)) / h;
    for (let i = 0; i < 3; i++) this.goal.target[i] += (-dx * right[i] + dy * up[i]) * s;
    this.anim = null;
  }

  /** Orbits the goal by screen pixels. */
  orbitPixels(dx: number, dy: number) {
    const dyaw = -dx * this.rotateSpeed;
    const dpitch = dy * this.rotateSpeed;
    this.goal.yaw += dyaw;
    this.goal.pitch = clampN(this.goal.pitch + dpitch, this.minPitch, this.maxPitch);
    const t = now();
    this.recent.push({ t, dyaw, dpitch });
    while (this.recent.length && t - this.recent[0].t > 90) this.recent.shift();
    this.anim = null;
  }

  private clampGoal() {
    const g = this.goal;
    g.pitch = clampN(Number.isFinite(g.pitch) ? g.pitch : 0.4, this.minPitch, this.maxPitch);
    g.distance = clampN(Number.isFinite(g.distance) ? g.distance : 1, this.minDistance, this.maxDistance);
    if (!Number.isFinite(g.yaw)) g.yaw = this.yaw;
    for (let i = 0; i < 3; i++) if (!Number.isFinite(g.target[i])) g.target[i] = this.camera.target[i] ?? 0;
    if (this.bounds) for (let i = 0; i < 3; i++) g.target[i] = clampN(g.target[i], this.bounds.min[i], this.bounds.max[i]);
  }

  /** Call once per frame with the real (wall-clock) frame time. */
  update(dt: number) {
    dt = Number.isFinite(dt) ? clampN(dt, 0, 0.1) : 0;
    if (this.anim) {
      this.anim.t += dt;
      if (this.anim.t > this.anim.duration) this.anim = null;
    }
    if (!this.mode && (this.vel.yaw || this.vel.pitch)) {
      this.goal.yaw += this.vel.yaw * dt;
      this.goal.pitch += this.vel.pitch * dt;
      const k = Math.exp(-this.inertiaDecay * dt);
      this.vel.yaw *= k;
      this.vel.pitch *= k;
      if (Math.abs(this.vel.yaw) + Math.abs(this.vel.pitch) < 1e-3) this.vel = { yaw: 0, pitch: 0 };
      if (this.goal.pitch <= this.minPitch || this.goal.pitch >= this.maxPitch) this.vel.pitch = 0;
    }
    this.clampGoal();
    const g = this.goal;
    const rate = this.anim ? this.anim.rate : this.smoothing;
    const k = 1 - Math.exp(-rate * dt);
    this.yaw += (g.yaw - this.yaw) * k;
    this.pitch += (g.pitch - this.pitch) * k;
    const ld = Math.log(Math.max(this.distance, 1e-6));
    this.distance = Math.exp(ld + (Math.log(g.distance) - ld) * k);
    const t = this.camera.target;
    for (let i = 0; i < 3; i++) t[i] += (g.target[i] - t[i]) * k;
    if (this.anim) {
      const err = Math.abs(g.yaw - this.yaw) + Math.abs(g.pitch - this.pitch) + Math.abs(Math.log(g.distance / this.distance)) + vec3.distance(g.target, t);
      if (err < 1e-4) this.anim = null;
    }
    this.pitch = clampN(this.pitch, this.minPitch, this.maxPitch);
    this.distance = clampN(this.distance, this.minDistance, this.maxDistance);
    this.applyPosition();
  }

  private applyPosition() {
    const t = this.camera.target;
    const place = (pitch: number): V3 => {
      const cp = Math.cos(pitch);
      return [t[0] + this.distance * cp * Math.sin(this.yaw), t[1] + this.distance * Math.sin(pitch), t[2] + this.distance * cp * Math.cos(this.yaw)];
    };
    let pos = place(this.pitch);
    if (this.ground) {
      const gy = this.ground(pos[0], pos[2]);
      if (Number.isFinite(gy) && pos[1] < gy + this.groundClearance) {
        // Pitch up to clear the ground (keeps the target framed); lift the eye as a last resort.
        const need = Math.asin(clampN((gy + this.groundClearance - t[1]) / this.distance, -1, 1));
        const p = clampN(Math.max(this.pitch, need), this.minPitch, this.maxPitch);
        if (p > this.pitch) {
          this.pitch = p;
          this.goal.pitch = Math.max(this.goal.pitch, p);
          pos = place(p);
        }
        const gy2 = this.ground(pos[0], pos[2]);
        if (Number.isFinite(gy2) && pos[1] < gy2 + this.groundClearance) pos[1] = gy2 + this.groundClearance;
      }
    }
    if (pos.every(Number.isFinite)) this.camera.position = pos;
  }

  private ndcOf(e: { clientX: number; clientY: number }): [number, number] {
    const r = this.element.getBoundingClientRect();
    return [((e.clientX - r.left) / Math.max(1, r.width)) * 2 - 1, 1 - ((e.clientY - r.top) / Math.max(1, r.height)) * 2];
  }

  private onDown = (e: PointerEvent) => {
    if (!this.enabled || consumed(e)) return;
    this.anim = null;
    this.vel = { yaw: 0, pitch: 0 };
    if (e.pointerType === 'touch') {
      this.touches.set(e.pointerId, [e.clientX, e.clientY]);
      if (this.touches.size === 1) {
        this.pending = 'orbit';
        this.mode = null;
        this.downAt = [e.clientX, e.clientY];
        this.last = [e.clientX, e.clientY];
      } else if (this.touches.size === 2) {
        this.pending = null;
        this.mode = 'pinch';
        this.pinchRef = this.pinchState();
      }
      return;
    }
    if (e.button === 0 && !e.shiftKey) this.pending = 'orbit';
    else if (e.button === 2 || e.button === 1 || (e.button === 0 && e.shiftKey)) this.pending = 'pan';
    else return;
    this.mode = null;
    this.downAt = [e.clientX, e.clientY];
    this.last = [e.clientX, e.clientY];
    this.recent = [];
  };

  private pinchState() {
    const pts = [...this.touches.values()].slice(0, 2);
    const mid: [number, number] = [(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2];
    return { spread: Math.max(1, Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])), mid };
  }

  private onMove = (e: PointerEvent) => {
    if (e.pointerType === 'touch' && this.touches.has(e.pointerId)) {
      this.touches.set(e.pointerId, [e.clientX, e.clientY]);
      if (this.mode === 'pinch') {
        if (this.touches.size >= 2 && this.pinchRef) {
          const s = this.pinchState();
          this.zoom(this.pinchRef.spread / s.spread, this.ndcOf({ clientX: s.mid[0], clientY: s.mid[1] }));
          this.panPixels(s.mid[0] - this.pinchRef.mid[0], s.mid[1] - this.pinchRef.mid[1]);
          this.pinchRef = s;
        }
        return;
      }
    }
    if (!this.pending && !this.mode) return;
    if (this.mode === 'pinch') return;
    if (!this.mode) {
      if (Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]) < 3) return;
      this.mode = this.pending;
    }
    const dx = e.clientX - this.last[0];
    const dy = e.clientY - this.last[1];
    this.last = [e.clientX, e.clientY];
    if (this.mode === 'orbit') this.orbitPixels(dx, dy);
    else if (this.mode === 'pan') this.panPixels(dx, dy);
  };

  private onUp = (e: PointerEvent) => {
    if (e.pointerType === 'touch') {
      this.touches.delete(e.pointerId);
      if (this.touches.size === 1 && this.mode === 'pinch') {
        const [p] = [...this.touches.values()];
        this.mode = 'orbit';
        this.pending = 'orbit';
        this.last = [p[0], p[1]];
        this.pinchRef = null;
        return;
      }
      if (this.touches.size > 0) return;
    }
    if (this.mode === 'orbit' && this.recent.length) {
      const t = now();
      const rec = this.recent.filter((r) => t - r.t < 90);
      const span = rec.length ? Math.max(16, t - rec[0].t) / 1000 : 1;
      const sy = rec.reduce((a, r) => a + r.dyaw, 0);
      const sp = rec.reduce((a, r) => a + r.dpitch, 0);
      this.vel = { yaw: clampN(sy / span, -6, 6) * 0.6, pitch: clampN(sp / span, -6, 6) * 0.6 };
    }
    this.mode = null;
    this.pending = null;
    this.pinchRef = null;
    this.recent = [];
  };

  private onBlur = () => {
    this.mode = null;
    this.pending = null;
    this.touches.clear();
    this.pinchRef = null;
  };

  private onWheel = (e: WheelEvent) => {
    if (!this.enabled || consumed(e)) return;
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const delta = clampN((e.deltaY || 0) * unit, -600, 600);
    if (!delta) return;
    this.zoom(Math.exp(delta * this.zoomSpeed), this.ndcOf(e));
  };

  private onContextMenu = (e: Event) => e.preventDefault();

  dispose() {
    this.element.removeEventListener('pointerdown', this.onDown);
    this.win?.removeEventListener('pointermove', this.onMove);
    this.win?.removeEventListener('pointerup', this.onUp);
    this.win?.removeEventListener('pointercancel', this.onUp);
    this.win?.removeEventListener('blur', this.onBlur);
    this.element.removeEventListener('wheel', this.onWheel);
    this.element.removeEventListener('contextmenu', this.onContextMenu);
  }
}
