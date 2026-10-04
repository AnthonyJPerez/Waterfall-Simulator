import { describe, expect, it } from 'vitest';
import { Camera, OrbitController, wrapAngleNear, type V3 } from '../src/core/camera';

function fakeElement() {
  return {
    addEventListener() {},
    removeEventListener() {},
    clientHeight: 600,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
  } as unknown as HTMLElement;
}

function rig(pos: V3 = [1, 1, 2], target: V3 = [1, 0, 0]) {
  const cam = new Camera();
  const orbit = new OrbitController(cam, fakeElement());
  orbit.setPose(pos, target);
  orbit.update(0.016);
  cam.update(4 / 3);
  return { cam, orbit };
}

const settle = (orbit: OrbitController, cam: Camera, seconds = 3) => {
  for (let t = 0; t < seconds; t += 1 / 60) {
    orbit.update(1 / 60);
    cam.update(4 / 3);
  }
};

const dist = (a: readonly number[], b: readonly number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe('OrbitController', () => {
  it('setPose is exact and stable', () => {
    const { cam, orbit } = rig([1.9, 0.65, 1.55], [1.1, 0.12, 0.6]);
    settle(orbit, cam, 0.5);
    expect(dist(cam.position, [1.9, 0.65, 1.55])).toBeLessThan(1e-6);
    expect(dist(cam.target, [1.1, 0.12, 0.6])).toBeLessThan(1e-9);
  });

  it('eases towards the goal (smoothing) instead of jumping', () => {
    const { cam, orbit } = rig();
    orbit.orbitPixels(100, 0);
    orbit.update(1 / 60);
    const partial = orbit.yaw;
    settle(orbit, cam, 2);
    expect(Math.abs(partial)).toBeGreaterThan(0);
    expect(Math.abs(orbit.yaw)).toBeGreaterThan(Math.abs(partial));
    expect(orbit.yaw).toBeCloseTo(-100 * orbit.rotateSpeed, 4);
  });

  it('zooms towards the picked point and keeps it fixed on screen', () => {
    const { cam, orbit } = rig([1, 1, 2], [1, 0, 0]);
    const h: V3 = [1.3, 0, 0.2];
    orbit.pick = () => h;
    const before = project(cam, h);
    orbit.zoom(0.5, [0.2, -0.1]);
    settle(orbit, cam);
    expect(orbit.distance).toBeCloseTo(0.5 * Math.hypot(1, 2), 3);
    const after = project(cam, h);
    expect(Math.hypot(after[0] - before[0], after[1] - before[1])).toBeLessThan(1e-3);
  });

  it('respects min distance for macro shots and target bounds', () => {
    const { cam, orbit } = rig();
    orbit.zoom(1e-6);
    settle(orbit, cam);
    expect(orbit.distance).toBeCloseTo(orbit.minDistance, 6);
    orbit.bounds = { min: [0, -1, 0], max: [2, 1, 1] };
    orbit.panPixels(-1e6, 0);
    settle(orbit, cam);
    for (let i = 0; i < 3; i++) {
      expect(cam.target[i]).toBeGreaterThanOrEqual(orbit.bounds.min[i] - 1e-9);
      expect(cam.target[i]).toBeLessThanOrEqual(orbit.bounds.max[i] + 1e-9);
    }
  });

  it('keeps the camera above the ground', () => {
    const { cam, orbit } = rig([1, 0.05, 2], [1, 0, 0]);
    orbit.ground = () => 0.4;
    settle(orbit, cam, 0.2);
    expect(cam.position[1]).toBeGreaterThanOrEqual(0.4 + orbit.groundClearance - 1e-9);
  });

  it('flies to a pose and focuses smoothly', () => {
    const { cam, orbit } = rig();
    orbit.flyTo([2, 0.5, 0.5], [1.5, 0.1, 0.6]);
    settle(orbit, cam, 4);
    expect(dist(cam.position, [2, 0.5, 0.5])).toBeLessThan(1e-3);
    expect(dist(cam.target, [1.5, 0.1, 0.6])).toBeLessThan(1e-3);
    orbit.focusOn([1, 0, 1], 0.3);
    settle(orbit, cam, 4);
    expect(dist(cam.target, [1, 0, 1])).toBeLessThan(1e-3);
    expect(orbit.distance).toBeCloseTo(0.3, 3);
  });

  it('never produces NaN for bad inputs', () => {
    const { cam, orbit } = rig();
    orbit.update(NaN);
    orbit.zoom(NaN);
    orbit.focusOn([NaN, 0, 0]);
    orbit.setPose([NaN, 1, 1], [0, 0, 0]);
    orbit.ground = () => NaN;
    orbit.update(1e9);
    expect(cam.position.every(Number.isFinite)).toBe(true);
    expect(wrapAngleNear(7, 0)).toBeCloseTo(7 - 2 * Math.PI, 9);
  });
});

function project(cam: Camera, p: readonly number[]) {
  const m = cam.viewProj;
  const c = [0, 1, 2, 3].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
  return [c[0] / c[3], c[1] / c[3]];
}
