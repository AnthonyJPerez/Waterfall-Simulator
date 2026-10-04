/**
 * Pure math for the obstacle editor: vectors, quaternions, obstacle proxy shapes
 * (ellipsoid / super-ellipsoid / capsule), ray picking and footprints.
 *
 * The real rock & log meshes belong to the terrain module; the editor only needs a
 * close proxy of each obstacle (derived from SceneModel position / rotation / scale)
 * for picking, outlines, footprints and the placement ghost.
 *
 * Conventions: quaternions are [x, y, z, w]; matrices for WGSL are column-major
 * Float32Array(16); obstacle `scale` are half-extents along local axes (logs: x = half
 * length, y/z = radii).
 */
import type { ObstacleKind } from '../app/params';

export type V3 = [number, number, number];
export type Quat = [number, number, number, number];

// ---------------------------------------------------------------------------------------------
// Vectors

export const add = (a: readonly number[], b: readonly number[]): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: readonly number[], s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: readonly number[], b: readonly number[]): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const length = (a: readonly number[]) => Math.hypot(a[0], a[1], a[2]);
export function normalize(a: readonly number[], fallback: V3 = [0, 1, 0]): V3 {
  const l = length(a);
  return l > 1e-12 && Number.isFinite(l) ? [a[0] / l, a[1] / l, a[2] / l] : [...fallback];
}
export const lerp = (a: readonly number[], b: readonly number[], t: number): V3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];
export const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);
export const finite3 = (a: readonly number[]) => Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]);

// ---------------------------------------------------------------------------------------------
// Quaternions

export const qIdentity = (): Quat => [0, 0, 0, 1];

export function qNormalize(q: readonly number[]): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(l > 1e-12) || !Number.isFinite(l)) return qIdentity();
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

/** Hamilton product a ⊗ b (applies b first, then a). */
export function qMul(a: readonly number[], b: readonly number[]): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export const qConj = (q: readonly number[]): Quat => [-q[0], -q[1], -q[2], q[3]];

export function qAxisAngle(axis: readonly number[], angle: number): Quat {
  const n = normalize(axis, [0, 1, 0]);
  const s = Math.sin(angle / 2);
  return [n[0] * s, n[1] * s, n[2] * s, Math.cos(angle / 2)];
}

export function qRotate(q: readonly number[], v: readonly number[]): V3 {
  // v' = v + 2 w (u × v) + 2 u × (u × v)
  const u: V3 = [q[0], q[1], q[2]];
  const w = q[3];
  const t = scale(cross(u, v), 2);
  return add(add(v, scale(t, w)), cross(u, t));
}

/** Shortest-arc rotation taking unit vector a to unit vector b. */
export function qFromTo(a: readonly number[], b: readonly number[]): Quat {
  const na = normalize(a);
  const nb = normalize(b);
  const d = dot(na, nb);
  if (d > 1 - 1e-9) return qIdentity();
  if (d < -1 + 1e-9) {
    // 180°: any axis perpendicular to a.
    let axis = cross([1, 0, 0], na);
    if (length(axis) < 1e-6) axis = cross([0, 0, 1], na);
    return qAxisAngle(axis, Math.PI);
  }
  const c = cross(na, nb);
  return qNormalize([c[0], c[1], c[2], 1 + d]);
}

/** Heading of the rotated local +X axis in the XZ plane (rad); rotation about +Y by a gives a. */
export function qYawAngle(q: readonly number[]): number {
  const x = qRotate(q, [1, 0, 0]);
  if (Math.hypot(x[0], x[2]) < 1e-6) {
    const z = qRotate(q, [0, 0, 1]);
    return Math.atan2(z[0], z[2]);
  }
  return Math.atan2(-x[2], x[0]);
}

/** Angle (rad) between the rotated local +Y axis and world up. */
export function qTiltAngle(q: readonly number[]): number {
  const y = qRotate(q, [0, 1, 0]);
  return Math.acos(clamp(y[1], -1, 1));
}

/** Rotation about world +Y through the obstacle centre (pre-multiplied). */
export const qRotateYaw = (q: readonly number[], angle: number): Quat => qNormalize(qMul(qAxisAngle([0, 1, 0], angle), q));

/**
 * Tilts an orientation about a horizontal world axis: the horizontal projection of the
 * obstacle's local Z axis (so logs/slabs lift their +X end), falling back to `fallbackAxis`.
 */
export function qTilt(q: readonly number[], angle: number, fallbackAxis: readonly number[] = [1, 0, 0]): Quat {
  const lz = qRotate(q, [0, 0, 1]);
  let axis: V3 = [lz[0], 0, lz[2]];
  if (Math.hypot(axis[0], axis[2]) < 1e-3) axis = [fallbackAxis[0], 0, fallbackAxis[2]];
  if (Math.hypot(axis[0], axis[2]) < 1e-6) axis = [1, 0, 0];
  return qNormalize(qMul(qAxisAngle(axis, angle), q));
}

/** Column-major 3×3 rotation matrix as rows r[i][j] = R_ij. */
export function qToMat3(q: readonly number[]): number[][] {
  const [x, y, z, w] = qNormalize(q);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

// ---------------------------------------------------------------------------------------------
// Obstacle proxies

export interface PoseLike {
  kind: ObstacleKind;
  position: readonly number[];
  rotation: readonly number[];
  scale: readonly number[];
}

export interface ProxyParams {
  /** Super-ellipsoid exponent (2 = ellipsoid). */
  exponent: number;
  /** Logs are capsules along local X. */
  capsule: boolean;
  /** Relative surface-noise amplitude used by the ghost mesh. */
  noise: number;
}

export function proxyParams(kind: ObstacleKind | string): ProxyParams {
  switch (kind) {
    case 'slab':
      return { exponent: 3, capsule: false, noise: 0.05 };
    case 'cobble':
      return { exponent: 2, capsule: false, noise: 0.06 };
    case 'log':
      return { exponent: 2, capsule: true, noise: 0.02 };
    default:
      return { exponent: 2, capsule: false, noise: 0.12 };
  }
}

/** Sanitised half extents (never zero / NaN). */
export function safeScale(s: readonly number[]): V3 {
  const f = (v: number) => (Number.isFinite(v) ? Math.max(Math.abs(v), 1e-4) : 0.05);
  return [f(s[0]), f(s[1]), f(s[2])];
}

/** Capsule geometry of a log proxy: half-length of the axis segment and mean radius. */
export function capsuleDims(s: readonly number[]) {
  const sc = safeScale(s);
  const r = 0.5 * (sc[1] + sc[2]);
  return { r, hl: Math.max(0, sc[0] - r), ry: sc[1], rz: sc[2] };
}

/** World point → obstacle local frame (rotated, NOT scaled). */
export function worldToLocal(o: PoseLike, p: readonly number[]): V3 {
  return qRotate(qConj(qNormalize(o.rotation)), sub(p, o.position));
}

export function localToWorld(o: PoseLike, p: readonly number[]): V3 {
  return add(qRotate(qNormalize(o.rotation), p), o.position);
}

/** Normalised proxy "radius" of a world point: < 1 inside, 1 on the proxy surface. */
export function proxyQ(o: PoseLike, p: readonly number[]): number {
  const l = worldToLocal(o, p);
  const pp = proxyParams(o.kind);
  if (pp.capsule) {
    const { r, hl, ry, rz } = capsuleDims(o.scale);
    const ex = Math.max(Math.abs(l[0]) - hl, 0) / r;
    return Math.hypot(ex, l[1] / ry, l[2] / rz);
  }
  const s = safeScale(o.scale);
  const e = pp.exponent;
  const v = [Math.abs(l[0] / s[0]), Math.abs(l[1] / s[1]), Math.abs(l[2] / s[2])];
  if (e === 2) return Math.hypot(v[0], v[1], v[2]);
  return Math.pow(Math.pow(v[0], e) + Math.pow(v[1], e) + Math.pow(v[2], e), 1 / e);
}

/** Smallest root t >= tMin of a t² + b t + c = 0 (null if none). */
function smallestRoot(a: number, b: number, c: number, tMin = 0): number | null {
  if (Math.abs(a) < 1e-14) {
    if (Math.abs(b) < 1e-14) return null;
    const t = -c / b;
    return t >= tMin ? t : null;
  }
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t0 = (-b - sq) / (2 * a);
  const t1 = (-b + sq) / (2 * a);
  const lo = Math.min(t0, t1);
  const hi = Math.max(t0, t1);
  if (lo >= tMin) return lo;
  if (hi >= tMin) return hi;
  return null;
}

function raySphereLocal(o: V3, d: V3, c: V3, r: number): number | null {
  const oc = sub(o, c);
  return smallestRoot(dot(d, d), 2 * dot(oc, d), dot(oc, oc) - r * r);
}

/**
 * Ray / proxy intersection. `dir` need not be normalised; the returned t is in units of
 * `dir` (so world distance when dir is unit). `inflate` scales the proxy about its centre.
 * Returns the entry distance, or the exit distance when the origin is inside, or null.
 */
export function rayProxy(o: PoseLike, origin: readonly number[], dir: readonly number[], inflate = 1): number | null {
  if (!finite3(origin) || !finite3(dir) || !finite3(o.position)) return null;
  const qi = qConj(qNormalize(o.rotation));
  const lo = qRotate(qi, sub(origin, o.position));
  const ld = qRotate(qi, dir);
  const pp = proxyParams(o.kind);
  if (pp.capsule) {
    const { r: r0, hl: hl0, ry, rz } = capsuleDims(o.scale);
    const r = r0 * inflate;
    const hl = hl0 * inflate;
    // Map the elliptical cross-section to a circle of radius r.
    const sy = r / (ry * inflate);
    const sz = r / (rz * inflate);
    const O: V3 = [lo[0], lo[1] * sy, lo[2] * sz];
    const D: V3 = [ld[0], ld[1] * sy, ld[2] * sz];
    let best: number | null = null;
    const consider = (t: number | null) => {
      if (t !== null && (best === null || t < best)) best = t;
    };
    // Infinite cylinder around X, then clip to the segment.
    const a = D[1] * D[1] + D[2] * D[2];
    const b = 2 * (O[1] * D[1] + O[2] * D[2]);
    const c = O[1] * O[1] + O[2] * O[2] - r * r;
    if (a > 1e-14) {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        for (const t of [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]) {
          if (t >= 0 && Math.abs(O[0] + D[0] * t) <= hl) consider(t);
        }
      }
    }
    consider(raySphereLocal(O, D, [hl, 0, 0], r));
    consider(raySphereLocal(O, D, [-hl, 0, 0], r));
    return best;
  }
  // (Super-)ellipsoids are picked with their bounding ellipsoid (slabs get a little extra).
  const s = safeScale(o.scale);
  const k = inflate * (pp.exponent > 2 ? Math.pow(3, 0.5 - 1 / pp.exponent) * 0.92 : 1);
  const O: V3 = [lo[0] / (s[0] * k), lo[1] / (s[1] * k), lo[2] / (s[2] * k)];
  const D: V3 = [ld[0] / (s[0] * k), ld[1] / (s[1] * k), ld[2] / (s[2] * k)];
  return smallestRoot(dot(D, D), 2 * dot(O, D), dot(O, O) - 1);
}

export interface ObstacleLike extends PoseLike {
  id: number;
}

/** Nearest proxy hit along a ray. */
export function pickProxies(obstacles: readonly ObstacleLike[], origin: readonly number[], dir: readonly number[], inflate = 1.04) {
  let best: { id: number; t: number } | null = null;
  for (const o of obstacles) {
    const t = rayProxy(o, origin, dir, inflate);
    if (t !== null && (best === null || t < best.t)) best = { id: o.id, t };
  }
  return best;
}

export const maxHalfExtent = (s: readonly number[]) => Math.max(...safeScale(s));

/**
 * Combines the terrain ray cast (authoritative surface) with the proxy picks:
 *  - an obstacle id reported by the terrain wins;
 *  - otherwise the nearest proxy hit is accepted when it is not clearly behind the
 *    terrain surface (real rocks are noisy, so allow a size-relative tolerance).
 */
export function resolvePick(
  terrainHit: { distance: number; obstacleId?: number } | null,
  proxyHit: { id: number; t: number } | null,
  obstacles: readonly ObstacleLike[],
): number | undefined {
  if (terrainHit && terrainHit.obstacleId !== undefined && obstacles.some((o) => o.id === terrainHit.obstacleId)) return terrainHit.obstacleId;
  if (!proxyHit) return undefined;
  if (!terrainHit) return proxyHit.id;
  const o = obstacles.find((x) => x.id === proxyHit.id);
  if (!o) return undefined;
  const tol = 0.22 * maxHalfExtent(o.scale) + 0.006;
  return proxyHit.t <= terrainHit.distance + tol ? proxyHit.id : undefined;
}

// ---------------------------------------------------------------------------------------------
// Footprints (projection of the proxy onto the XZ plane)

export type Footprint =
  | { type: 'ellipse'; cx: number; cz: number; /** L Lᵀ = covariance of the projected ellipse (2×2 lower). */ l11: number; l21: number; l22: number }
  | { type: 'stadium'; cx: number; cz: number; ax: number; az: number; bx: number; bz: number; r: number };

export function footprintOf(o: PoseLike): Footprint {
  const R = qToMat3(o.rotation);
  const cx = o.position[0];
  const cz = o.position[2];
  if (proxyParams(o.kind).capsule) {
    const { r, hl, ry, rz } = capsuleDims(o.scale);
    // Axis endpoints projected; radius = horizontal extent of the cross-section.
    const axis: V3 = [R[0][0] * hl, R[1][0] * hl, R[2][0] * hl];
    const ey: V3 = [R[0][1] * ry, R[1][1] * ry, R[2][1] * ry];
    const ez: V3 = [R[0][2] * rz, R[1][2] * rz, R[2][2] * rz];
    const rh = Math.max(r * 0.5, Math.sqrt(Math.max(ey[0] * ey[0] + ez[0] * ez[0], ey[2] * ey[2] + ez[2] * ez[2])));
    return { type: 'stadium', cx, cz, ax: cx + axis[0], az: cz + axis[2], bx: cx - axis[0], bz: cz - axis[2], r: rh };
  }
  const s = safeScale(o.scale);
  const pp = proxyParams(o.kind);
  const k = pp.exponent > 2 ? 1.12 : 1; // super-ellipsoid corners stick out a little
  // K = R S² Rᵀ restricted to (x, z).
  const s2 = [s[0] * s[0] * k * k, s[1] * s[1] * k * k, s[2] * s[2] * k * k];
  let kxx = 0;
  let kxz = 0;
  let kzz = 0;
  for (let j = 0; j < 3; j++) {
    kxx += R[0][j] * R[0][j] * s2[j];
    kxz += R[0][j] * R[2][j] * s2[j];
    kzz += R[2][j] * R[2][j] * s2[j];
  }
  const l11 = Math.sqrt(Math.max(kxx, 1e-10));
  const l21 = kxz / l11;
  const l22 = Math.sqrt(Math.max(kzz - l21 * l21, 1e-10));
  return { type: 'ellipse', cx, cz, l11, l21, l22 };
}

/** True when (x, z) is inside the footprint scaled by `inflate` and grown by `margin` (m). */
export function footprintContains(f: Footprint, x: number, z: number, inflate = 1, margin = 0): boolean {
  if (f.type === 'stadium') {
    const abx = f.bx - f.ax;
    const abz = f.bz - f.az;
    const ll = abx * abx + abz * abz;
    const t = ll > 1e-12 ? clamp(((x - f.ax) * abx + (z - f.az) * abz) / ll, 0, 1) : 0;
    const px = f.ax + abx * t;
    const pz = f.az + abz * t;
    // Inflate about the centre: scale the segment and radius.
    const cx = f.cx + (px - f.cx) * inflate;
    const cz = f.cz + (pz - f.cz) * inflate;
    return Math.hypot(x - cx, z - cz) <= f.r * inflate + margin;
  }
  // Solve L u = d.
  const dx = x - f.cx;
  const dz = z - f.cz;
  const u1 = dx / f.l11;
  const u2 = (dz - f.l21 * u1) / f.l22;
  const rr = Math.hypot(u1, u2);
  if (margin === 0) return rr <= inflate;
  // Approximate margin: scale the normalised radius by the local half-extent along the direction.
  const dist = Math.hypot(dx, dz);
  if (dist < 1e-9) return true;
  const extent = dist / Math.max(rr, 1e-9); // footprint radius along this direction
  return dist <= extent * inflate + margin;
}

/** Closed outline polyline (XZ) of the footprint, `n` points. */
export function footprintOutline(f: Footprint, n: number, inflate = 1, margin = 0): [number, number][] {
  const pts: [number, number][] = [];
  n = Math.max(8, Math.floor(n));
  if (f.type === 'stadium') {
    const ax = f.cx + (f.ax - f.cx) * inflate;
    const az = f.cz + (f.az - f.cz) * inflate;
    const bx = f.cx + (f.bx - f.cx) * inflate;
    const bz = f.cz + (f.bz - f.cz) * inflate;
    const r = f.r * inflate + margin;
    const ang = Math.atan2(az - bz, ax - bx);
    const half = Math.floor(n / 2);
    for (let i = 0; i < half; i++) {
      const a = ang - Math.PI / 2 + (Math.PI * i) / (half - 1);
      pts.push([ax + Math.cos(a) * r, az + Math.sin(a) * r]);
    }
    for (let i = 0; i < n - half; i++) {
      const a = ang + Math.PI / 2 + (Math.PI * i) / (n - half - 1);
      pts.push([bx + Math.cos(a) * r, bz + Math.sin(a) * r]);
    }
    return pts;
  }
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const c = Math.cos(a);
    const s = Math.sin(a);
    let dx = f.l11 * c;
    let dz = f.l21 * c + f.l22 * s;
    if (margin) {
      const d = Math.hypot(dx, dz) || 1;
      dx += (dx / d) * (margin / inflate);
      dz += (dz / d) * (margin / inflate);
    }
    pts.push([f.cx + dx * inflate, f.cz + dz * inflate]);
  }
  return pts;
}

/** Largest horizontal radius of the footprint about its centre. */
export function footprintRadius(f: Footprint): number {
  if (f.type === 'stadium') return Math.hypot(f.ax - f.cx, f.az - f.cz) + f.r;
  // Largest singular value of L.
  const a = f.l11 * f.l11 + f.l21 * f.l21;
  const b = f.l21 * f.l22;
  const c = f.l22 * f.l22;
  const tr = a + c;
  const det = a * c - b * b;
  return Math.sqrt(Math.max(0, tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det))));
}

// ---------------------------------------------------------------------------------------------
// Matrices for the GPU overlay

/** Column-major mat4: obstacle local (rotated, unscaled) → world. */
export function poseMatrix(o: PoseLike, out = new Float32Array(16)): Float32Array {
  const R = qToMat3(o.rotation);
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < 3; r++) out[c * 4 + r] = R[r][c];
    out[c * 4 + 3] = 0;
  }
  out[12] = o.position[0];
  out[13] = o.position[1];
  out[14] = o.position[2];
  out[15] = 1;
  return out;
}

/** Column-major mat4: world → obstacle local (rotated, unscaled). */
export function invPoseMatrix(o: PoseLike, out = new Float32Array(16)): Float32Array {
  const R = qToMat3(o.rotation);
  // Rᵀ, translation -Rᵀ p
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < 3; r++) out[c * 4 + r] = R[c][r];
    out[c * 4 + 3] = 0;
  }
  const p = o.position;
  for (let r = 0; r < 3; r++) out[12 + r] = -(R[0][r] * p[0] + R[1][r] * p[1] + R[2][r] * p[2]);
  out[15] = 1;
  return out;
}

/** Top-most world point of the proxy (approx.: rotated local axes' vertical extents). */
export function proxyTopY(o: PoseLike): number {
  const R = qToMat3(o.rotation);
  const s = safeScale(o.scale);
  if (proxyParams(o.kind).capsule) {
    const { r, hl } = capsuleDims(o.scale);
    return o.position[1] + Math.abs(R[1][0]) * hl + r;
  }
  const ext = Math.sqrt(R[1][0] ** 2 * s[0] ** 2 + R[1][1] ** 2 * s[1] ** 2 + R[1][2] ** 2 * s[2] ** 2);
  return o.position[1] + ext;
}

/** Lowest world y of the proxy along the vertical line through (x, z), or null if outside. */
export function proxyBottomAt(o: PoseLike, x: number, z: number): number | null {
  const far = 100;
  const t = rayProxy(o, [x, o.position[1] - far, z], [0, 1, 0]);
  if (t === null) return null;
  return o.position[1] - far + t;
}
