/**
 * Perspective camera (reversed-Z: near → depth 1, far → depth 0; use depthCompare 'greater'
 * and clear depth to 0) plus an orbit controller tuned for close-up inspection.
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

/**
 * Orbit / pan / dolly controller.
 *  - Left drag: orbit (when the editor does not consume the event)
 *  - Right drag or Shift+left drag: pan
 *  - Wheel: dolly towards the cursor-independent target (exponential)
 *  - Double click: refocus (handled by App via focusOn)
 */
export class OrbitController {
  yaw = 0;
  pitch = 0.4;
  distance = 1.2;
  minDistance = 0.04;
  maxDistance = 12;
  enabled = true;
  private dragging: 'orbit' | 'pan' | null = null;
  private last = [0, 0];
  private animTarget: { target: V3; distance?: number; t: number } | null = null;

  constructor(private camera: Camera, private element: HTMLElement) {
    this.syncFromCamera();
    element.addEventListener('pointerdown', this.onDown);
    window.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    element.addEventListener('wheel', this.onWheel, { passive: false });
    element.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  syncFromCamera() {
    const c = this.camera;
    const d = vec3.subtract(c.position, c.target);
    this.distance = vec3.length(d);
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / this.distance)));
  }

  setPose(position: V3, target: V3) {
    this.camera.position = [...position];
    this.camera.target = [...target];
    this.syncFromCamera();
  }

  focusOn(target: V3, distance?: number) {
    this.animTarget = { target: [...target], distance, t: 0 };
  }

  /** Call once per frame. */
  update(dt: number) {
    if (this.animTarget) {
      const a = this.animTarget;
      const k = 1 - Math.exp(-dt * 8);
      const t = this.camera.target;
      for (let i = 0; i < 3; i++) t[i] += (a.target[i] - t[i]) * k;
      if (a.distance) this.distance += (a.distance - this.distance) * k;
      a.t += dt;
      if (a.t > 1.5) this.animTarget = null;
    }
    this.pitch = Math.max(-0.2, Math.min(1.5, this.pitch));
    this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
    const cp = Math.cos(this.pitch);
    const t = this.camera.target;
    this.camera.position = [
      t[0] + this.distance * cp * Math.sin(this.yaw),
      t[1] + this.distance * Math.sin(this.pitch),
      t[2] + this.distance * cp * Math.cos(this.yaw),
    ];
  }

  private onDown = (e: PointerEvent) => {
    if (!this.enabled || (e as any).__consumed) return;
    if (e.button === 0 && !e.shiftKey) this.dragging = 'orbit';
    else if (e.button === 2 || e.button === 1 || (e.button === 0 && e.shiftKey)) this.dragging = 'pan';
    else return;
    this.last = [e.clientX, e.clientY];
    this.animTarget = null;
  };

  private onMove = (e: PointerEvent) => {
    if (!this.dragging) return;
    const dx = e.clientX - this.last[0];
    const dy = e.clientY - this.last[1];
    this.last = [e.clientX, e.clientY];
    if (this.dragging === 'orbit') {
      this.yaw -= dx * 0.005;
      this.pitch += dy * 0.005;
    } else {
      const c = this.camera;
      const fwd = vec3.normalize(vec3.subtract(c.target, c.position));
      const right = vec3.normalize(vec3.cross(fwd, [0, 1, 0]));
      const up = vec3.cross(right, fwd);
      const s = this.distance * 0.0015;
      for (let i = 0; i < 3; i++) c.target[i] += (-dx * right[i] + dy * up[i]) * s;
    }
  };

  private onUp = () => {
    this.dragging = null;
  };

  private onWheel = (e: WheelEvent) => {
    if (!this.enabled) return;
    e.preventDefault();
    this.distance *= Math.exp(e.deltaY * 0.0012);
  };

  dispose() {
    this.element.removeEventListener('pointerdown', this.onDown);
    window.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    this.element.removeEventListener('wheel', this.onWheel);
  }
}
