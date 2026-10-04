import { describe, expect, it } from 'vitest';
import { footprintOf, proxyBottomAt, qRotate, qTiltAngle } from '../src/editor/math';
import {
  bilinearGround,
  computePlacement,
  fitPlane,
  groundUnder,
  marchGround,
  maskedGround,
  mulberry32,
  rollVariation,
  scaleFor,
  type DomainLike,
} from '../src/editor/placement';

const domain: DomainLike = { sizeX: 2.4, sizeZ: 1.2, minY: -0.35, maxY: 0.75, cellSize: 0.008 };

describe('placement', () => {
  const flat = () => 0.1;
  const slope = (x: number) => 0.3 - 0.5 * x; // 26.6° downhill along +X

  it('sizes kinds from the requested size with bounded variation', () => {
    const v = rollVariation(mulberry32(1));
    const b = scaleFor('boulder', 0.2, v);
    expect(b[0]).toBeGreaterThan(0.2 * 0.85);
    expect(b[0]).toBeLessThan(0.2 * 1.15);
    const log = scaleFor('log', 0.2, v);
    expect(log[1]).toBe(log[2]);
    expect(log[0]).toBeGreaterThan(log[1] * 5);
    const bad = scaleFor('boulder', NaN, { stretch: [NaN, 5, -5] });
    expect(bad.every(Number.isFinite)).toBe(true);
  });

  it('seats a boulder partly embedded on flat ground', () => {
    const p = computePlacement({ kind: 'boulder', size: 0.15, variation: { seed: 3, yaw: 0.3, stretch: [0, 0, 0], embed: 0.5 }, x: 1, z: 0.6, ground: flat, domain });
    const bottom = p.position[1] - p.scale[1];
    const top = p.position[1] + p.scale[1];
    expect(bottom).toBeLessThan(0.1); // embedded
    expect(top).toBeGreaterThan(0.1 + p.scale[1]); // mostly exposed
    expect(qTiltAngle(p.rotation)).toBeCloseTo(0, 6);
    expect(p.seed).toBe(3);
  });

  it('aligns slabs to the slope and never floats', () => {
    const p = computePlacement({ kind: 'slab', size: 0.2, variation: { seed: 1, yaw: 0, stretch: [0, 0, 0], embed: 0 }, x: 0.5, z: 0.6, ground: slope, domain });
    const up = qRotate(p.rotation, [0, 1, 0]);
    expect(Math.acos(up[1])).toBeCloseTo(Math.atan(0.5), 2);
    const f = footprintOf(p);
    for (const t of [-0.7, -0.3, 0, 0.3, 0.7]) {
      const x = f.cx + t * 0.2;
      const yb = proxyBottomAt(p, x, 0.6);
      if (yb !== null) expect(yb).toBeLessThanOrEqual(slope(x) + 0.002);
    }
  });

  it('boulders only partially follow the slope', () => {
    const p = computePlacement({ kind: 'boulder', size: 0.2, variation: { seed: 1, yaw: 0, stretch: [0, 0, 0], embed: 0.5 }, x: 0.5, z: 0.6, ground: slope, domain });
    const tilt = qTiltAngle(p.rotation);
    expect(tilt).toBeGreaterThan(0.05);
    expect(tilt).toBeLessThan(Math.atan(0.5) * 0.6);
  });

  it('is robust to non-finite ground and clamps to the domain', () => {
    const p = computePlacement({ kind: 'cobble', size: 0.1, variation: rollVariation(mulberry32(9)), x: 99, z: -5, ground: () => NaN, fallbackY: 0.2, domain });
    expect(p.position.every(Number.isFinite)).toBe(true);
    expect(p.position[0]).toBeLessThanOrEqual(domain.sizeX);
    expect(p.position[2]).toBeGreaterThanOrEqual(0);
  });
});

describe('ground queries', () => {
  it('fits planes', () => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) pts.push([i, 1 + 0.2 * i - 0.1 * j, j]);
    const f = fitPlane(pts, 0, 0)!;
    expect(f.a).toBeCloseTo(1, 9);
    expect(f.b).toBeCloseTo(0.2, 9);
    expect(f.c).toBeCloseTo(-0.1, 9);
  });

  it('bilinear ground smooths a nearest-cell height function', () => {
    const cs = 0.01;
    const nearest = (x: number, z: number) => Math.floor(x / cs) * cs * 0.5 + 0 * z;
    const g = bilinearGround(nearest, cs);
    expect(g(0.105, 0)).toBeCloseTo(0.105 * 0.5 - cs * 0.25, 9);
  });

  it('masked ground ignores the masked obstacle top', () => {
    const rock = { kind: 'boulder' as const, position: [1, 0.1, 0.6], rotation: [0, 0, 0, 1], scale: [0.1, 0.08, 0.1] };
    const terrainWithRock = (x: number, z: number) => {
      const r2 = ((x - 1) / 0.1) ** 2 + ((z - 0.6) / 0.1) ** 2;
      return r2 < 1 ? Math.max(0.1, 0.1 + 0.08 * Math.sqrt(1 - r2)) : 0.1;
    };
    const masked = maskedGround(terrainWithRock, [footprintOf(rock)]);
    expect(masked(1, 0.6)).toBeCloseTo(0.1, 6);
    expect(groundUnder(masked, rock)).toBeCloseTo(0.1, 6);
    expect(groundUnder(terrainWithRock, rock)).toBeGreaterThan(0.13);
  });

  it('ray-marches a heightfield and finds the first crossing', () => {
    const ground = (x: number) => (x < 1 ? 0.3 : -0.1);
    const o = [0.2, 1, 0.6];
    const d = [0.6, -0.8, 0];
    const hit = marchGround(ground, o, d, domain)!;
    expect(hit).not.toBeNull();
    expect(hit.point[1]).toBeCloseTo(0.3, 3);
    // a ray that never meets the ground inside the domain
    expect(marchGround(ground, [0.2, 1, 0.6], [0, 1, 0], domain)).toBeNull();
  });
});
