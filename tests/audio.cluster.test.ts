import { describe, expect, it } from 'vitest';
import { EmitterClusterer, type ClusterPoint } from '../src/audio/cluster';
import { makeRng } from '../src/audio/physics';

const pt = (x: number, y: number, z: number, w = 1, ext = 0.03): ClusterPoint => ({ x, y, z, w, ext });

describe('EmitterClusterer', () => {
  it('keeps emitters stable under jittered sources (temporal coherence)', () => {
    const c = new EmitterClusterer(8);
    const rand = makeRng(1);
    const assign = new Int32Array(64);
    const listener = [0, 0.5, 1.5];
    const srcA = [1.0, 0, 0.5];
    const srcB = [1.6, 0, 0.9];
    let idA = -1;
    let idB = -1;
    let maxStep = 0;
    let prev: number[] | null = null;
    for (let f = 0; f < 200; f++) {
      const pts: ClusterPoint[] = [];
      for (let k = 0; k < 10; k++) pts.push(pt(srcA[0] + (rand() - 0.5) * 0.04, srcA[1], srcA[2] + (rand() - 0.5) * 0.04, 1 + rand()));
      for (let k = 0; k < 10; k++) pts.push(pt(srcB[0] + (rand() - 0.5) * 0.04, srcB[1], srcB[2] + (rand() - 0.5) * 0.04, 0.5 + rand()));
      c.update(pts, pts.length, listener, 1 / 60, assign);
      const a = assign[0];
      const b = assign[10];
      if (f === 0) {
        idA = a;
        idB = b;
      }
      // Every point of a source always goes to the same emitter, and the two sources stay apart.
      for (let k = 0; k < 10; k++) expect(assign[k]).toBe(idA);
      for (let k = 10; k < 20; k++) expect(assign[k]).toBe(idB);
      expect(idA).not.toBe(idB);
      const e = c.emitters[idA];
      if (prev) maxStep = Math.max(maxStep, Math.hypot(e.x - prev[0], e.z - prev[2]));
      prev = [e.x, e.y, e.z];
    }
    expect(c.emitters.filter((e) => e.active).length).toBe(2);
    // Smoothed: per-update motion is far below the 4 cm jitter.
    expect(maxStep).toBeLessThan(0.01);
    const e = c.emitters[idA];
    expect(Math.hypot(e.x - srcA[0], e.z - srcA[2])).toBeLessThan(0.02);
  });

  it('separates nearby sources close to the listener, merges them far away (angular resolution)', () => {
    const assign = new Int32Array(8);
    const near = new EmitterClusterer(8);
    // Two drips 10 cm apart, 30 cm from the camera.
    near.update([pt(0, 0, -0.3), pt(0.1, 0, -0.3)], 2, [0, 0, 0], 1 / 60, assign);
    expect(assign[0]).not.toBe(assign[1]);
    const far = new EmitterClusterer(8);
    far.update([pt(0, 0, -5), pt(0.1, 0, -5)], 2, [0, 0, 0], 1 / 60, assign);
    expect(assign[0]).toBe(assign[1]);
    expect(far.mergeRadius(5)).toBeGreaterThan(near.mergeRadius(0.3));
  });

  it('assigns every point even when there are more clusters than emitters', () => {
    const c = new EmitterClusterer(8);
    const pts: ClusterPoint[] = [];
    for (let i = 0; i < 20; i++) pts.push(pt(i * 0.5, 0, 0, 1 + i));
    const assign = new Int32Array(20);
    c.update(pts, pts.length, [5, 1, 3], 1 / 60, assign);
    for (let i = 0; i < 20; i++) {
      expect(assign[i]).toBeGreaterThanOrEqual(0);
      expect(assign[i]).toBeLessThan(8);
    }
    expect(c.emitters.filter((e) => e.active).length).toBe(8);
  });

  it('retires emitters after the hold time and lets a much stronger new source take over', () => {
    const c = new EmitterClusterer(2, { hold: 0.5 });
    const assign = new Int32Array(4);
    c.update([pt(0, 0, 0, 1), pt(3, 0, 0, 1)], 2, [1.5, 1, 2], 1 / 60, assign);
    expect(c.emitters.filter((e) => e.active).length).toBe(2);
    // A third, far-away, 100× stronger source steals the weakest emitter.
    c.update([pt(0, 0, 0, 1), pt(3, 0, 0, 0.5), pt(1.5, 0, -3, 100)], 3, [1.5, 1, 2], 1 / 60, assign);
    const strong = assign[2];
    expect(strong).toBeGreaterThanOrEqual(0);
    expect(Math.hypot(c.emitters[strong].x - 1.5, c.emitters[strong].z + 3)).toBeLessThan(0.005);
    expect(c.emitters[strong].spawned).toBe(true);
    // Everything goes silent: emitters free up after the hold time.
    for (let i = 0; i < 40; i++) c.update([], 0, [1.5, 1, 2], 1 / 60, assign);
    expect(c.emitters.filter((e) => e.active).length).toBe(0);
  });

  it('ignores invalid points', () => {
    const c = new EmitterClusterer(4);
    const assign = new Int32Array(3);
    c.update([pt(NaN, 0, 0), pt(0, 0, 0, -1), pt(0, 0, 0, Infinity)], 3, [0, 0, 1], 1 / 60, assign);
    expect([...assign]).toEqual([-1, -1, -1]);
    for (const e of c.emitters) expect(e.active).toBe(false);
  });
});
