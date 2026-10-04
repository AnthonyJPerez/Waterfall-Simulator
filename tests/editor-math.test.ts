import { describe, expect, it } from 'vitest';
import {
  footprintContains,
  footprintOf,
  footprintOutline,
  footprintRadius,
  invPoseMatrix,
  pickProxies,
  poseMatrix,
  proxyBottomAt,
  proxyQ,
  proxyTopY,
  qAxisAngle,
  qFromTo,
  qMul,
  qRotate,
  qRotateYaw,
  qTilt,
  qTiltAngle,
  qYawAngle,
  rayProxy,
  resolvePick,
  type ObstacleLike,
  type V3,
} from '../src/editor/math';

const close3 = (a: readonly number[], b: readonly number[], d = 1e-6) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], -Math.log10(d)));

const boulder = (id: number, p: V3, s: V3 = [0.1, 0.07, 0.085], yaw = 0): ObstacleLike => ({
  id,
  kind: 'boulder',
  position: p,
  rotation: qAxisAngle([0, 1, 0], yaw),
  scale: s,
});

describe('quaternions', () => {
  it('rotates vectors and composes like matrices', () => {
    const q = qAxisAngle([0, 1, 0], Math.PI / 2);
    close3(qRotate(q, [1, 0, 0]), [0, 0, -1]);
    const q2 = qMul(qAxisAngle([1, 0, 0], Math.PI / 2), q); // yaw first, then roll about X
    close3(qRotate(q2, [1, 0, 0]), qRotate(qAxisAngle([1, 0, 0], Math.PI / 2), [0, 0, -1]));
  });
  it('extracts yaw and tilt', () => {
    for (const a of [-2.5, -0.3, 0, 0.7, 3]) expect(qYawAngle(qAxisAngle([0, 1, 0], a))).toBeCloseTo(a, 6);
    const tilted = qTilt(qAxisAngle([0, 1, 0], 0.4), 0.3);
    expect(qTiltAngle(tilted)).toBeCloseTo(0.3, 6);
    expect(qYawAngle(qRotateYaw(qAxisAngle([0, 1, 0], 0.2), 0.5))).toBeCloseTo(0.7, 6);
  });
  it('shortest arc maps a to b, including opposite vectors', () => {
    close3(qRotate(qFromTo([0, 1, 0], [1, 0, 0]), [0, 1, 0]), [1, 0, 0]);
    close3(qRotate(qFromTo([0, 1, 0], [0, -1, 0]), [0, 1, 0]), [0, -1, 0]);
    close3(qRotate(qFromTo([0, 1, 0], [0, 1, 0]), [0, 1, 0]), [0, 1, 0]);
  });
});

describe('proxy shapes & picking', () => {
  it('proxyQ is 1 on the ellipsoid surface', () => {
    const o = boulder(1, [1, 0.2, 0.5], [0.1, 0.07, 0.085], 0.6);
    const lp = qRotate(o.rotation, [0, 0.07, 0]);
    expect(proxyQ(o, [1 + lp[0], 0.2 + lp[1], 0.5 + lp[2]])).toBeCloseTo(1, 6);
    expect(proxyQ(o, [1, 0.2, 0.5])).toBe(0);
  });
  it('ray hits ellipsoid at the expected distance (rotated)', () => {
    const o = boulder(1, [0, 0, 0], [0.2, 0.1, 0.1], Math.PI / 2); // long axis now along Z
    const t = rayProxy(o, [0, 0, 5], [0, 0, -1]);
    expect(t).toBeCloseTo(4.8, 6);
    expect(rayProxy(o, [0, 0, 5], [0, 0, 1])).toBeNull();
    expect(rayProxy(o, [0, 5, 0], [0, -1, 0])).toBeCloseTo(4.9, 6);
  });
  it('ray hits capsule logs on the cylinder and the caps', () => {
    const log: ObstacleLike = { id: 3, kind: 'log', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [0.5, 0.05, 0.05] };
    expect(rayProxy(log, [0.2, 1, 0], [0, -1, 0])).toBeCloseTo(0.95, 6); // cylinder side
    expect(rayProxy(log, [2, 0, 0], [-1, 0, 0])).toBeCloseTo(1.5, 6); // cap tip at x = 0.5
    expect(rayProxy(log, [0.2, 1, 0.2], [0, -1, 0])).toBeNull();
  });
  it('picks the nearest proxy and combines with the terrain hit', () => {
    const obs = [boulder(1, [0, 0, 0]), boulder(2, [0, 0, -1])];
    const hit = pickProxies(obs, [0, 0, 3], [0, 0, -1]);
    expect(hit?.id).toBe(1);
    // terrain id wins
    expect(resolvePick({ distance: 1, obstacleId: 2 }, hit, obs)).toBe(2);
    // proxy just behind the terrain hit (noisy rock) still selects
    expect(resolvePick({ distance: hit!.t - 0.01 }, hit, obs)).toBe(1);
    // proxy far behind the terrain → terrain occludes
    expect(resolvePick({ distance: hit!.t - 0.5 }, hit, obs)).toBeUndefined();
    expect(resolvePick(null, hit, obs)).toBe(1);
    expect(resolvePick({ distance: 1, obstacleId: 99 }, null, obs)).toBeUndefined();
  });
  it('handles degenerate inputs without throwing or NaN', () => {
    const o: ObstacleLike = { id: 1, kind: 'boulder', position: [0, 0, 0], rotation: [0, 0, 0, 0], scale: [0, NaN, -1] };
    expect(() => rayProxy(o, [0, 1, 0], [0, -1, 0])).not.toThrow();
    expect(rayProxy(o, [NaN, 1, 0], [0, -1, 0])).toBeNull();
    expect(Number.isFinite(proxyQ(o, [0.01, 0, 0]))).toBe(true);
  });
  it('bottom / top of the proxy', () => {
    const o = boulder(1, [0.5, 0.1, 0.5], [0.1, 0.05, 0.1]);
    expect(proxyBottomAt(o, 0.5, 0.5)).toBeCloseTo(0.05, 6);
    expect(proxyBottomAt(o, 0.8, 0.5)).toBeNull();
    expect(proxyTopY(o)).toBeCloseTo(0.15, 6);
  });
  it('pose matrices are inverse of each other', () => {
    const o = { kind: 'slab' as const, position: [0.3, 0.1, -0.2], rotation: qMul(qAxisAngle([1, 0, 0], 0.4), qAxisAngle([0, 1, 0], 1.1)), scale: [1, 1, 1] };
    const m = poseMatrix(o);
    const mi = invPoseMatrix(o);
    const p = [0.1, 0.2, 0.3, 1];
    const mul = (M: Float32Array, v: number[]) => [0, 1, 2, 3].map((r) => M[r] * v[0] + M[4 + r] * v[1] + M[8 + r] * v[2] + M[12 + r] * v[3]);
    close3(mul(mi, mul(m, p)), p, 1e-5);
  });
});

describe('footprints', () => {
  it('projects a yawed ellipsoid to the expected ellipse', () => {
    const o = boulder(1, [1, 0, 1], [0.2, 0.1, 0.1], Math.PI / 2);
    const f = footprintOf(o);
    expect(footprintContains(f, 1, 1.19)).toBe(true); // long axis along Z after yaw
    expect(footprintContains(f, 1.19, 1)).toBe(false);
    expect(footprintRadius(f)).toBeCloseTo(0.2, 6);
    for (const [x, z] of footprintOutline(f, 32)) {
      const r = Math.hypot((x - 1) / 0.1, (z - 1) / 0.2);
      expect(r).toBeCloseTo(1, 5);
    }
  });
  it('tilted ellipsoid footprint grows', () => {
    const o = { kind: 'boulder' as const, position: [0, 0, 0], rotation: qAxisAngle([0, 0, 1], Math.PI / 2), scale: [0.05, 0.3, 0.05] };
    const f = footprintOf(o);
    expect(footprintRadius(f)).toBeCloseTo(0.3, 6);
  });
  it('log footprint is a stadium along its axis', () => {
    const log: ObstacleLike = { id: 3, kind: 'log', position: [1, 0, 1], rotation: [0, 0, 0, 1], scale: [0.5, 0.05, 0.05] };
    const f = footprintOf(log);
    expect(footprintContains(f, 1.48, 1)).toBe(true);
    expect(footprintContains(f, 1, 1.07)).toBe(false);
    expect(footprintContains(f, 1, 1.07, 1, 0.03)).toBe(true);
    expect(footprintOutline(f, 24).length).toBe(24);
  });
});
