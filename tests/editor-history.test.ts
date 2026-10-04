import { describe, expect, it } from 'vitest';
import { SceneModel } from '../src/world/scene';
import { compressOps, SceneHistory } from '../src/editor/history';
import { TrailingThrottle, RateGate } from '../src/editor/throttle';

const rock = (x: number) => ({ kind: 'boulder' as const, position: [x, 0, 0] as [number, number, number], rotation: [0, 0, 0, 1] as [number, number, number, number], scale: [0.1, 0.07, 0.08] as [number, number, number], seed: 1, roughness: 0.1 });

function setup() {
  const scene = new SceneModel();
  let t = 0;
  const queue: (() => void)[] = [];
  const h = new SceneHistory(scene, { now: () => t, schedule: (fn) => queue.push(fn) });
  const tick = (dt = 0) => {
    t += dt;
    queue.splice(0).forEach((f) => f());
  };
  return { scene, h, tick, advance: (dt: number) => (t += dt) };
}

describe('SceneHistory', () => {
  it('undoes and redoes add / move / delete with stable lineage ids', () => {
    const { scene, h, tick } = setup();
    const a = scene.add(rock(1));
    tick();
    h.transact('move', () => {
      scene.update(a.id, { position: [1.1, 0, 0] });
      scene.update(a.id, { position: [1.2, 0, 0] });
    });
    scene.remove(a.id);
    tick();
    expect(h.undoDepth).toBe(3);
    expect(scene.obstacles.length).toBe(0);

    h.undo(); // un-delete → new id
    expect(scene.obstacles.length).toBe(1);
    const reborn = scene.obstacles[0];
    expect(reborn.id).not.toBe(a.id);
    expect(h.resolveId(a.id)).toBe(reborn.id);
    expect(reborn.position[0]).toBeCloseTo(1.2);

    h.undo(); // un-move (one entry for the whole drag)
    expect(scene.get(h.resolveId(a.id))!.position[0]).toBeCloseTo(1);
    h.undo(); // un-add
    expect(scene.obstacles.length).toBe(0);
    expect(h.canUndo).toBe(false);

    h.redo();
    h.redo();
    expect(scene.get(h.resolveId(a.id))!.position[0]).toBeCloseTo(1.2);
    h.redo();
    expect(scene.obstacles.length).toBe(0);
    expect(h.canRedo).toBe(false);
  });

  it('keeps ids resolvable across edits made on a re-added obstacle', () => {
    const { scene, h, tick } = setup();
    const a = scene.add(rock(1));
    tick();
    scene.remove(a.id);
    tick();
    h.undo(); // a re-added as id2
    const id2 = h.resolveId(a.id);
    scene.update(id2, { position: [2, 0, 0] }); // new entry referencing id2 (clears redo)
    tick();
    h.undo(); // move back
    h.undo(); // remove (undo of add)
    expect(scene.obstacles.length).toBe(0);
    h.redo(); // add again → id3
    h.redo(); // move must find id3 via lineage of id2
    expect(scene.obstacles.length).toBe(1);
    expect(scene.obstacles[0].position[0]).toBeCloseTo(2);
  });

  it('groups a synchronous burst (remove all) into one entry', () => {
    const { scene, h, tick } = setup();
    [1, 2, 3].forEach((x) => scene.add(rock(x)));
    tick();
    expect(h.undoDepth).toBe(1);
    [...scene.obstacles].forEach((o) => scene.remove(o.id));
    tick();
    expect(h.undoDepth).toBe(2);
    expect(h.undoLabel).toBe('remove 3 obstacles');
    h.undo();
    expect(h.redoLabel).toBe('remove 3 obstacles');
    expect(scene.obstacles.map((o) => o.position[0])).toEqual([3, 2, 1]);
  });

  it('coalesces repeated keyed edits within the window only', () => {
    const { scene, h, tick, advance } = setup();
    const a = scene.add(rock(1));
    tick();
    for (let i = 0; i < 5; i++) {
      h.transact('rotate', () => scene.update(a.id, { position: [1 + i * 0.01, 0, 0] }), `xf:${a.id}`);
      advance(100);
    }
    expect(h.undoDepth).toBe(2);
    advance(5000);
    h.transact('rotate', () => scene.update(a.id, { position: [3, 0, 0] }), `xf:${a.id}`);
    expect(h.undoDepth).toBe(3);
  });

  it('ignores no-op transactions and clears on reset', () => {
    const { scene, h, tick } = setup();
    const a = scene.add(rock(1));
    tick();
    h.transact('drag', () => {
      scene.update(a.id, { position: [2, 0, 0] });
      scene.update(a.id, { position: [1, 0, 0] });
    });
    expect(h.undoDepth).toBe(1);
    scene.reset([], []);
    expect(h.undoDepth).toBe(0);
    expect(h.canRedo).toBe(false);
  });

  it('caps history length and stops after dispose', () => {
    const scene = new SceneModel();
    const h = new SceneHistory(scene, { limit: 3, schedule: (fn) => fn() });
    for (let i = 0; i < 6; i++) scene.add(rock(i));
    expect(h.undoDepth).toBe(3);
    h.dispose();
    scene.add(rock(9));
    expect(h.undoDepth).toBe(3);
  });

  it('compressOps folds add+update and cancels add+remove', () => {
    const d = rock(1);
    const ops = compressOps([
      { kind: 'add', id: 1, data: d },
      { kind: 'update', id: 1, before: d, after: { ...d, position: [2, 0, 0] } },
      { kind: 'add', id: 2, data: d },
      { kind: 'remove', id: 2, data: d },
    ]);
    expect(ops).toHaveLength(1);
    expect(ops[0].kind).toBe('add');
    expect((ops[0] as any).data.position[0]).toBe(2);
  });
});

describe('throttling', () => {
  it('commits at most once per interval and always flushes the final value', () => {
    let t = 0;
    const got: number[] = [];
    const th = new TrailingThrottle<number>(50, (v) => got.push(v), () => t);
    for (let i = 0; i < 20; i++) {
      th.push(i);
      t += 10; // 100 Hz input
    }
    // 200 ms of input at 20 Hz → ≤ 5 commits
    expect(got.length).toBeLessThanOrEqual(5);
    expect(got[0]).toBe(0);
    th.flush();
    expect(got[got.length - 1]).toBe(19);
    expect(th.flush()).toBe(false);
  });
  it('tick commits a pending value after the interval', () => {
    let t = 0;
    const got: number[] = [];
    const th = new TrailingThrottle<number>(50, (v) => got.push(v), () => t);
    th.push(1);
    th.push(2);
    expect(got).toEqual([1]);
    t = 30;
    th.tick();
    expect(got).toEqual([1]);
    t = 60;
    th.tick();
    expect(got).toEqual([1, 2]);
    th.push(3);
    th.cancel();
    t = 500;
    th.tick();
    expect(got).toEqual([1, 2]);
  });
  it('RateGate limits frequency', () => {
    let t = 0;
    const g = new RateGate(100, () => t);
    expect(g.ready()).toBe(true);
    expect(g.ready()).toBe(false);
    t = 150;
    expect(g.ready()).toBe(true);
  });
});
