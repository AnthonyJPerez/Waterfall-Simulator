/**
 * Placement logic: how new obstacles are sized, oriented and seated on the ground so they
 * look naturally bedded (partly embedded, slabs/logs following the surface), plus the
 * ground queries used while dragging (a heightfield ray-march that can ignore the dragged
 * obstacle's own footprint, which the terrain has baked into its height field).
 */
import type { ObstacleKind } from '../app/params';
import {
  add,
  clamp,
  type Footprint,
  footprintContains,
  footprintOf,
  footprintOutline,
  normalize,
  type PoseLike,
  proxyBottomAt,
  qAxisAngle,
  qFromTo,
  qMul,
  qNormalize,
  type Quat,
  scale as vscale,
  type V3,
} from './math';

export type GroundFn = (x: number, z: number) => number;

export interface DomainLike {
  sizeX: number;
  sizeZ: number;
  minY: number;
  maxY: number;
  cellSize: number;
}

export interface KindTemplate {
  /** Half extents relative to the requested size (logs: x = half length, y/z = radius). */
  ratio: V3;
  /** Centre height above the ground in units of the vertical half extent (min, max). */
  embed: [number, number];
  /** 0..1: how much the local up axis follows the ground normal. */
  follow: number;
  /** Maximum tilt from vertical (rad). */
  maxTilt: number;
  roughness: number;
  moss: number;
  /** Random per-axis size variation (relative amplitude). */
  stretch: number;
  label: string;
}

export const KIND_TEMPLATES: Record<ObstacleKind, KindTemplate> = {
  boulder: { ratio: [1, 0.68, 0.85], embed: [0.05, 0.35], follow: 0.35, maxTilt: 0.35, roughness: 0.18, moss: 0.35, stretch: 0.14, label: 'Boulder' },
  cobble: { ratio: [1, 0.55, 0.78], embed: [0.2, 0.42], follow: 0.6, maxTilt: 0.45, roughness: 0.08, moss: 0.1, stretch: 0.12, label: 'Cobble' },
  slab: { ratio: [1, 0.24, 0.72], embed: [0.0, 0.45], follow: 1, maxTilt: 0.6, roughness: 0.07, moss: 0.25, stretch: 0.15, label: 'Slab' },
  log: { ratio: [2.6, 0.3, 0.3], embed: [0.35, 0.7], follow: 1, maxTilt: 0.5, roughness: 0.04, moss: 0.45, stretch: 0.1, label: 'Log' },
};

export const OBSTACLE_KINDS: ObstacleKind[] = ['boulder', 'cobble', 'slab', 'log'];

export function kindTemplate(kind: ObstacleKind | string): KindTemplate {
  return (KIND_TEMPLATES as Record<string, KindTemplate>)[kind] ?? KIND_TEMPLATES.boulder;
}

/** Small, fast deterministic PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Per-placement randomness (re-rolled after every placement or with R). */
export interface Variation {
  seed: number;
  yaw: number;
  /** Per-axis multipliers in [-1, 1] (scaled by the kind's stretch). */
  stretch: V3;
  /** 0..1 position inside the kind's embed range. */
  embed: number;
}

export function rollVariation(rand: () => number = Math.random): Variation {
  return {
    seed: Math.floor(rand() * 1e6),
    yaw: rand() * Math.PI * 2,
    stretch: [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1],
    embed: rand(),
  };
}

export const MIN_HALF_EXTENT = 0.006;
export const MAX_HALF_EXTENT = 1.5;

export function scaleFor(kind: ObstacleKind, size: number, v: Pick<Variation, 'stretch'>): V3 {
  const t = kindTemplate(kind);
  const s = Number.isFinite(size) ? clamp(size, 0.005, 1) : 0.15;
  const st = v.stretch;
  const k = (i: number) => 1 + t.stretch * clamp(Number.isFinite(st[i]) ? st[i] : 0, -1, 1);
  const out: V3 = [t.ratio[0] * s * k(0), t.ratio[1] * s * k(1), t.ratio[2] * s * k(2)];
  if (kind === 'log') out[2] = out[1]; // round logs
  return out.map((x) => clamp(x, MIN_HALF_EXTENT, MAX_HALF_EXTENT)) as V3;
}

/** Bilinear interpolation of a (possibly nearest-cell) height query over the grid's cell centres. */
export function bilinearGround(heightAt: GroundFn, cellSize: number): GroundFn {
  const cs = cellSize > 0 && Number.isFinite(cellSize) ? cellSize : 0.01;
  return (x, z) => {
    const gx = x / cs - 0.5;
    const gz = z / cs - 0.5;
    const i = Math.floor(gx);
    const j = Math.floor(gz);
    const tx = gx - i;
    const tz = gz - j;
    const h = (a: number, b: number) => {
      const v = heightAt((a + 0.5) * cs, (b + 0.5) * cs);
      return Number.isFinite(v) ? v : NaN;
    };
    const h00 = h(i, j);
    const h10 = h(i + 1, j);
    const h01 = h(i, j + 1);
    const h11 = h(i + 1, j + 1);
    const v = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
    if (Number.isFinite(v)) return v;
    const c = heightAt(x, z);
    return Number.isFinite(c) ? c : NaN;
  };
}

export interface PlaneFit {
  /** Height of the fitted plane at the centre. */
  height: number;
  normal: V3;
  min: number;
  max: number;
  /** Samples used (world points). */
  points: V3[];
}

/** Least-squares plane y = a + b dx + c dz through points around (cx, cz). */
export function fitPlane(points: readonly V3[], cx: number, cz: number): { a: number; b: number; c: number } | null {
  const n = points.length;
  if (n === 0) return null;
  let sx = 0, sz = 0, sy = 0, sxx = 0, szz = 0, sxz = 0, sxy = 0, szy = 0;
  for (const p of points) {
    const dx = p[0] - cx;
    const dz = p[2] - cz;
    sx += dx; sz += dz; sy += p[1];
    sxx += dx * dx; szz += dz * dz; sxz += dx * dz;
    sxy += dx * p[1]; szy += dz * p[1];
  }
  if (n < 3) return { a: sy / n, b: 0, c: 0 };
  // Normal equations for [a b c].
  const M = [
    [n, sx, sz],
    [sx, sxx, sxz],
    [sz, sxz, szz],
  ];
  const r = [sy, sxy, szy];
  const det3 = (m: number[][]) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det3(M);
  if (Math.abs(D) < 1e-18) return { a: sy / n, b: 0, c: 0 };
  const col = (k: number) => M.map((row, i) => row.map((v, j) => (j === k ? r[i] : v)));
  const a = det3(col(0)) / D;
  const b = det3(col(1)) / D;
  const c = det3(col(2)) / D;
  if (![a, b, c].every(Number.isFinite)) return { a: sy / n, b: 0, c: 0 };
  return { a, b, c };
}

/** Samples the ground under a footprint (centre + two rings) and fits a plane. */
export function groundFit(ground: GroundFn, f: Footprint, fallbackY = 0): PlaneFit {
  const pts: V3[] = [];
  const push = (x: number, z: number) => {
    const y = ground(x, z);
    if (Number.isFinite(y)) pts.push([x, y, z]);
  };
  push(f.cx, f.cz);
  for (const [x, z] of footprintOutline(f, 8, 0.45)) push(x, z);
  for (const [x, z] of footprintOutline(f, 12, 0.85)) push(x, z);
  if (!pts.length) return { height: fallbackY, normal: [0, 1, 0], min: fallbackY, max: fallbackY, points: [] };
  const plane = fitPlane(pts, f.cx, f.cz)!;
  let min = Infinity;
  let max = -Infinity;
  for (const p of pts) {
    min = Math.min(min, p[1]);
    max = Math.max(max, p[1]);
  }
  // Keep the centre height within the sampled range (a poor fit on a step must not float/bury).
  const height = clamp(plane.a, min, max);
  return { height, normal: normalize([-plane.b, 1, -plane.c]), min, max, points: pts };
}

/** Limits the angle between `n` and +Y. */
export function limitTilt(n: readonly number[], maxTilt: number): V3 {
  const u = normalize(n);
  const ang = Math.acos(clamp(u[1], -1, 1));
  if (ang <= maxTilt) return u;
  const h = Math.hypot(u[0], u[2]);
  if (h < 1e-9) return [0, 1, 0];
  const s = Math.sin(maxTilt);
  return [(u[0] / h) * s, Math.cos(maxTilt), (u[2] / h) * s];
}

export interface PlacedObstacle {
  kind: ObstacleKind;
  position: V3;
  rotation: Quat;
  scale: V3;
  seed: number;
  roughness: number;
  moss: number;
}

export interface PlacementInput {
  kind: ObstacleKind;
  size: number;
  variation: Variation;
  /** Placement point (cursor hit) xz. */
  x: number;
  z: number;
  ground: GroundFn;
  /** Used when the ground cannot be sampled. */
  fallbackY?: number;
  /** Extra yaw on top of the variation (user rotation of the ghost). */
  yawOffset?: number;
  domain?: DomainLike;
}

/**
 * Seats an obstacle on the ground: plane-fit under the footprint, orient (yaw, then tilt to
 * follow the slope per kind), embed partially, and lower it so no part of its core
 * footprint floats above the ground.
 */
export function computePlacement(inp: PlacementInput): PlacedObstacle {
  const t = kindTemplate(inp.kind);
  const sc = scaleFor(inp.kind, inp.size, inp.variation);
  const yaw = (inp.variation.yaw || 0) + (inp.yawOffset || 0);
  const qYaw = qAxisAngle([0, 1, 0], yaw);
  let x = Number.isFinite(inp.x) ? inp.x : 0;
  let z = Number.isFinite(inp.z) ? inp.z : 0;
  if (inp.domain) {
    x = clamp(x, 0, inp.domain.sizeX);
    z = clamp(z, 0, inp.domain.sizeZ);
  }
  const fallbackY = Number.isFinite(inp.fallbackY) ? (inp.fallbackY as number) : 0;
  const flat: PoseLike = { kind: inp.kind, position: [x, 0, z], rotation: qYaw, scale: sc };
  const fit = groundFit(inp.ground, footprintOf(flat), fallbackY);
  const up = limitTilt(normalize(add(vscale([0, 1, 0], 1 - t.follow), vscale(fit.normal, t.follow))), t.maxTilt);
  const rotation = qNormalize(qMul(qFromTo([0, 1, 0], up), qYaw));
  const embedK = t.embed[0] + (t.embed[1] - t.embed[0]) * clamp(inp.variation.embed ?? 0.5, 0, 1);
  const vertical = sc[1];
  const position: V3 = add([x, fit.height, z], vscale(up, embedK * vertical));
  // Anti-floating: the underside must reach the ground across the sampled core footprint.
  const pose: PoseLike = { kind: inp.kind, position, rotation, scale: sc };
  let lower = 0;
  for (const p of fit.points) {
    const yb = proxyBottomAt(pose, p[0], p[2]);
    if (yb !== null && Number.isFinite(yb)) lower = Math.max(lower, yb - (p[1] - 0.001));
  }
  position[1] -= Math.min(lower, 0.6 * vertical);
  if (inp.domain) clampPositionToDomain(position, inp.domain);
  return {
    kind: inp.kind,
    position,
    rotation,
    scale: sc,
    seed: inp.variation.seed >>> 0,
    roughness: t.roughness,
    moss: t.moss,
  };
}

export function clampPositionToDomain(p: V3, d: Pick<DomainLike, 'sizeX' | 'sizeZ' | 'minY' | 'maxY'>): V3 {
  p[0] = clamp(Number.isFinite(p[0]) ? p[0] : d.sizeX / 2, 0, d.sizeX);
  p[2] = clamp(Number.isFinite(p[2]) ? p[2] : d.sizeZ / 2, 0, d.sizeZ);
  p[1] = clamp(Number.isFinite(p[1]) ? p[1] : (d.minY + d.maxY) / 2, d.minY, d.maxY);
  return p;
}

/** Height of the ground under an obstacle's footprint (plane fit at the centre). */
export function groundUnder(ground: GroundFn, pose: PoseLike, fallbackY = 0): number {
  return groundFit(ground, footprintOf(pose), fallbackY).height;
}

/**
 * Ground query that ignores some footprints (e.g. the obstacle being dragged, whose top the
 * terrain has baked into the bed height). Inside a mask the ground is replaced by a plane
 * fitted to samples just outside it.
 */
export function maskedGround(base: GroundFn, masks: readonly Footprint[], inflate = 1.25, margin = 0.012): GroundFn {
  if (!masks.length) return base;
  const inside = (x: number, z: number) => masks.findIndex((m) => footprintContains(m, x, z, inflate, margin));
  const planes = masks.map((m) => {
    const pts: V3[] = [];
    for (const ring of [1.12, 1.4]) {
      for (const [x, z] of footprintOutline(m, 16, inflate * ring, margin * 1.5)) {
        if (inside(x, z) >= 0) continue;
        const y = base(x, z);
        if (Number.isFinite(y)) pts.push([x, y, z]);
      }
    }
    const fit = fitPlane(pts, m.cx, m.cz);
    if (!fit) return null;
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of pts) {
      lo = Math.min(lo, p[1]);
      hi = Math.max(hi, p[1]);
    }
    return { ...fit, lo, hi, cx: m.cx, cz: m.cz };
  });
  return (x, z) => {
    const k = inside(x, z);
    if (k < 0) return base(x, z);
    const p = planes[k];
    if (!p) return base(x, z);
    return clamp(p.a + p.b * (x - p.cx) + p.c * (z - p.cz), p.lo, p.hi);
  };
}

/** Ray / horizontal plane y = h. Returns t > 0 or null. */
export function rayPlaneY(origin: readonly number[], dir: readonly number[], h: number): number | null {
  if (Math.abs(dir[1]) < 1e-9) return null;
  const t = (h - origin[1]) / dir[1];
  return t > 0 && Number.isFinite(t) ? t : null;
}

/** Ray vs axis-aligned box; returns [tEnter, tExit] clipped to t >= 0, or null. */
export function rayBox(origin: readonly number[], dir: readonly number[], lo: readonly number[], hi: readonly number[]): [number, number] | null {
  let t0 = 0;
  let t1 = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(dir[i]) < 1e-12) {
      if (origin[i] < lo[i] || origin[i] > hi[i]) return null;
      continue;
    }
    let a = (lo[i] - origin[i]) / dir[i];
    let b = (hi[i] - origin[i]) / dir[i];
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
    if (t0 > t1) return null;
  }
  return [t0, t1];
}

/**
 * CPU ray-march against a height field (first downward crossing), refined by bisection.
 * Used for dragging with the dragged obstacle masked out of the ground.
 */
export function marchGround(
  ground: GroundFn,
  origin: readonly number[],
  dir: readonly number[],
  d: DomainLike,
  maxDist = 50,
): { t: number; point: V3 } | null {
  const box = rayBox(origin, dir, [0, d.minY - 0.05, 0], [d.sizeX, d.maxY + 0.5, d.sizeZ]);
  if (!box) return null;
  const tEnd = Math.min(box[1], maxDist);
  const step = Math.max(d.cellSize * 0.75, 1e-4);
  const f = (t: number) => {
    const x = origin[0] + dir[0] * t;
    const z = origin[2] + dir[2] * t;
    let g = ground(x, z);
    if (!Number.isFinite(g)) g = d.minY;
    return origin[1] + dir[1] * t - g;
  };
  let tPrev = box[0];
  let fPrev = f(tPrev);
  const maxSteps = 20000;
  let n = 0;
  for (let t = tPrev + step; t <= tEnd + step && n < maxSteps; t += step, n++) {
    const tc = Math.min(t, tEnd);
    const fc = f(tc);
    if (fPrev >= 0 && fc < 0) {
      let a = tPrev;
      let b = tc;
      for (let i = 0; i < 24; i++) {
        const m = (a + b) / 2;
        if (f(m) >= 0) a = m;
        else b = m;
      }
      const tt = (a + b) / 2;
      return { t: tt, point: [origin[0] + dir[0] * tt, origin[1] + dir[1] * tt, origin[2] + dir[2] * tt] };
    }
    tPrev = tc;
    fPrev = fc;
    if (tc >= tEnd) break;
  }
  return null;
}
