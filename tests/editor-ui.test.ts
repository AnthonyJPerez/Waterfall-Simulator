import { describe, expect, it } from 'vitest';
import { analyzeStream, computeCameraViews } from '../src/ui/cameraViews';
import { flowReadouts, formatCount, pickStat, statLines } from '../src/ui/statsFormat';
import { screenshotFileName } from '../src/ui/screenshot';
import { helpSections } from '../src/ui/chrome';

const extent = { sizeX: 2.4, sizeZ: 1.2, minY: -0.35, maxY: 0.75 };
// Ledge at x = 1.0: 0.3 m upstream, pool at -0.15 m, banks rising at the sides.
const ledge = (x: number, z: number) => {
  const bank = Math.max(0, Math.abs(z - 0.6) - 0.4) * 1.5;
  if (x < 1.0) return 0.3 - 0.03 * x + bank;
  return -0.15 + Math.max(0, x - 1.8) * 0.2 + bank;
};

describe('camera views', () => {
  it('finds the lip and the pool of a ledge', () => {
    const f = analyzeStream({ extent, presetCamera: { position: [0, 0, 0], target: [1, 0, 0] }, heightAt: ledge });
    expect(f.lipX).toBeGreaterThan(0.9);
    expect(f.lipX).toBeLessThan(1.02);
    expect(f.poolX).toBeGreaterThan(1.0);
    expect(f.drop).toBeGreaterThan(0.4);
    expect(f.z).toBeCloseTo(0.6);
  });

  it('uses the preset water level for the pool', () => {
    const f = analyzeStream({
      extent,
      presetCamera: { position: [0, 0, 0], target: [1, 0, 0] },
      heightAt: ledge,
      initialWater: [{ x0: 1, z0: 0, x1: 2.4, z1: 1.2, level: 0.08 }],
    });
    expect(f.waterY).toBeCloseTo(0.08);
  });

  it('computes five finite views that stay above the ground', () => {
    const views = computeCameraViews({ extent, presetCamera: { position: [1.9, 0.65, 1.55], target: [1.1, 0.12, 0.6] }, heightAt: ledge, fovY: 0.87, aspect: 16 / 9 });
    expect(views.map((v) => v.id)).toEqual(['preset', 'lip', 'pool', 'eye', 'overview']);
    for (const v of views) {
      expect([...v.position, ...v.target].every(Number.isFinite)).toBe(true);
      expect(v.position[1]).toBeGreaterThan(ledge(Math.min(2.4, Math.max(0, v.position[0])), Math.min(1.2, Math.max(0, v.position[2]))));
    }
    const lip = views[1];
    expect(Math.hypot(lip.position[0] - lip.target[0], lip.position[1] - lip.target[1], lip.position[2] - lip.target[2])).toBeLessThan(0.8);
    const ov = views[4];
    expect(Math.hypot(ov.position[0] - ov.target[0], ov.position[1] - ov.target[1], ov.position[2] - ov.target[2])).toBeGreaterThan(1.3);
    expect(views[3].position[1]).toBeLessThan(0.2); // eye level stays low
  });

  it('survives a flat or broken height function', () => {
    for (const h of [() => 0, () => NaN]) {
      const views = computeCameraViews({ extent, presetCamera: { position: [1, 1, 1], target: [0, 0, 0] }, heightAt: h });
      for (const v of views) expect([...v.position, ...v.target].every(Number.isFinite)).toBe(true);
    }
  });
});

describe('stats formatting', () => {
  it('formats documented SWE stats with units and particles generically', () => {
    const r = flowReadouts({ inflowRate: 0.003, overflowRate: 0.0012, maxSpeed: 1.234, totalVolume: 0.0456 }, { alive: 12345, diffuse: 999 });
    expect(r.inflow).toBe('3.00 L/s');
    expect(r.falling).toBe('1.20 L/s');
    expect(r.speed).toBe('1.23 m/s');
    expect(r.volume).toBe('45.6 L');
    expect(r.particles).toBe('12.3k water · 999 spray');
    const empty = flowReadouts({}, {});
    expect(empty.inflow).toBe('—');
    expect(empty.particles).toBe('—');
  });
  it('lists any stats generically and skips non-finite garbage gracefully', () => {
    expect(statLines('swe.', { inflowRate: 0.002, substeps: 3, weird: NaN, obj: {} as any })).toEqual(['swe.inflowRate: 2.00 L/s', 'swe.substeps: 3', 'swe.weird: NaN']);
    expect(pickStat({ a: NaN, b: 2 }, ['a', 'b'])).toBe(2);
    expect(formatCount(2_500_000)).toBe('2.50M');
  });
  it('screenshot names are sortable and filesystem safe', () => {
    expect(screenshotFileName(new Date(2026, 9, 4, 7, 5, 3))).toBe('creekside-20261004-070503.png');
  });
  it('help lists every tool shortcut', () => {
    const all = helpSections().flatMap((s) => s.rows.map((r) => r.join(' ')));
    for (const k of ['1  2  3  4', 'Esc', 'Q / E', 'T / G', 'R', 'Del · Backspace']) expect(all.some((r) => r.includes(k))).toBe(true);
  });
});
