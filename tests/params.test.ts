import { describe, expect, it } from 'vitest';
import { ParamStore } from '../src/app/params';
import { makeDomain } from '../src/world/domain';
import { SceneModel } from '../src/world/scene';

describe('ParamStore', () => {
  it('gets/sets by path and notifies prefix listeners', () => {
    const s = new ParamStore();
    const seen: string[] = [];
    s.on('flow', (p) => seen.push(p));
    s.set('flow.rate', 7);
    expect(s.get('flow.rate')).toBe(7);
    expect(seen).toEqual(['flow.rate']);
    s.set('water.clarity', 0.1);
    expect(seen).toEqual(['flow.rate']);
  });
});

describe('makeDomain', () => {
  it('produces square cells and consistent grid sizes', () => {
    const d = makeDomain({ sizeX: 2.4, sizeZ: 1.2, minY: -0.3, maxY: 0.8 }, 'medium');
    expect(d.nx).toBe(400);
    expect(d.nz).toBe(200);
    expect(d.cellSize * d.nz).toBeCloseTo(d.sizeZ, 9);
    expect(d.sdfNy).toBeGreaterThan(0);
  });
});

describe('SceneModel', () => {
  it('adds, updates and removes obstacles with change events', () => {
    const m = new SceneModel();
    const events: string[] = [];
    m.onChange((c) => events.push(c.type));
    const o = m.add({ kind: 'boulder', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], seed: 1, roughness: 0.1 });
    m.update(o.id, { position: [1, 0, 0] });
    m.remove(o.id);
    expect(events).toEqual(['obstacle-added', 'obstacle-moved', 'obstacle-removed']);
    expect(m.obstacles.length).toBe(0);
  });
});
