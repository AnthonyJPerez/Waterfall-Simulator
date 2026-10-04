/**
 * Undo / redo for obstacle edits.
 *
 * The history OBSERVES SceneModel change events, so every edit is undoable no matter who
 * made it (editor tools, the panel's "Remove all", scripts). Events are grouped into
 * entries:
 *  - explicitly with begin()/end() (e.g. a whole drag → one entry),
 *  - otherwise automatically per synchronous burst (closed at the next microtask), so
 *    "remove all" is a single entry,
 *  - and consecutive entries with the same coalesce key within `coalesceMs` merge (e.g.
 *    repeated Q/E rotations or wheel scaling).
 *
 * SceneModel assigns a fresh id whenever an obstacle is re-added (undo of a delete, redo of
 * an add), so ids are tracked by lineage: every op stores the id it saw, and
 * resolveId(anyHistoricalId) returns the obstacle's current id.
 *
 * A scene 'reset' (preset load / world rebuild) clears the history.
 */
import type { Obstacle, SceneChange } from '../world/scene';

export type ObstacleData = Omit<Obstacle, 'id'>;

export interface SceneLike {
  readonly obstacles: readonly Obstacle[];
  add(o: ObstacleData): Obstacle;
  remove(id: number): void;
  update(id: number, patch: Partial<ObstacleData>): void;
  get(id: number): Obstacle | undefined;
  onChange(fn: (c: SceneChange) => void): unknown;
}

export type HistoryOp =
  | { kind: 'add'; id: number; data: ObstacleData }
  | { kind: 'remove'; id: number; data: ObstacleData }
  | { kind: 'update'; id: number; before: ObstacleData; after: ObstacleData };

export interface HistoryEntry {
  label: string;
  ops: HistoryOp[];
  coalesceKey?: string;
  time: number;
}

export interface HistoryOptions {
  limit?: number;
  coalesceMs?: number;
  now?: () => number;
  /** Schedules the closing of an automatic group (default: queueMicrotask). */
  schedule?: (fn: () => void) => void;
}

export function cloneData(o: Obstacle | ObstacleData): ObstacleData {
  const { id: _id, ...rest } = o as Obstacle;
  return {
    ...rest,
    position: [o.position[0], o.position[1], o.position[2]],
    rotation: [o.rotation[0], o.rotation[1], o.rotation[2], o.rotation[3]],
    scale: [o.scale[0], o.scale[1], o.scale[2]],
  };
}

function sameData(a: ObstacleData, b: ObstacleData): boolean {
  const eq = (x: readonly number[], y: readonly number[]) => x.length === y.length && x.every((v, i) => Math.abs(v - y[i]) < 1e-9);
  return (
    a.kind === b.kind &&
    a.seed === b.seed &&
    a.roughness === b.roughness &&
    a.moss === b.moss &&
    eq(a.position, b.position) &&
    eq(a.rotation, b.rotation) &&
    eq(a.scale, b.scale)
  );
}

/** Merges consecutive updates of the same id and drops no-op updates / add+remove pairs. */
export function compressOps(ops: readonly HistoryOp[]): HistoryOp[] {
  const out: HistoryOp[] = [];
  for (const op of ops) {
    const prev = out[out.length - 1];
    if (op.kind === 'update' && prev && prev.id === op.id) {
      if (prev.kind === 'update') {
        out[out.length - 1] = { kind: 'update', id: op.id, before: prev.before, after: op.after };
        continue;
      }
      if (prev.kind === 'add') {
        out[out.length - 1] = { kind: 'add', id: op.id, data: op.after };
        continue;
      }
    }
    if (op.kind === 'remove' && prev && prev.id === op.id && prev.kind === 'add') {
      out.pop();
      continue;
    }
    out.push(op);
  }
  return out.filter((op) => !(op.kind === 'update' && sameData(op.before, op.after)));
}

export class SceneHistory {
  private undoStack: HistoryEntry[] = [];
  private redoStack: HistoryEntry[] = [];
  private open: { label: string; coalesceKey?: string; ops: HistoryOp[]; explicit: number } | null = null;
  private autoScheduled = false;
  private applying = false;
  /** lineage: any id ever seen → canonical id; canonical → current id */
  private canonOf = new Map<number, number>();
  private current = new Map<number, number>();
  private listeners = new Set<() => void>();
  private unsub: unknown;
  private disposed = false;
  readonly limit: number;
  readonly coalesceMs: number;
  private now: () => number;
  private schedule: (fn: () => void) => void;

  constructor(
    private scene: SceneLike,
    opts: HistoryOptions = {},
  ) {
    this.limit = opts.limit ?? 200;
    this.coalesceMs = opts.coalesceMs ?? 900;
    this.now = opts.now ?? (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
    this.schedule = opts.schedule ?? ((fn) => queueMicrotask(fn));
    this.unsub = scene.onChange((c) => this.onSceneChange(c));
  }

  get canUndo() {
    return this.undoStack.length > 0 || (this.open !== null && this.open.ops.length > 0 && this.open.explicit === 0);
  }
  get canRedo() {
    return this.redoStack.length > 0;
  }
  get undoLabel() {
    return this.undoStack[this.undoStack.length - 1]?.label;
  }
  get redoLabel() {
    return this.redoStack[this.redoStack.length - 1]?.label;
  }
  get undoDepth() {
    return this.undoStack.length;
  }
  get redoDepth() {
    return this.redoStack.length;
  }
  get inTransaction() {
    return (this.open?.explicit ?? 0) > 0;
  }

  onChange(fn: () => void) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    this.listeners.forEach((l) => l());
  }

  /** Current id of an obstacle given any id it has had (or the id itself). */
  resolveId(id: number): number {
    const c = this.canonOf.get(id) ?? id;
    return this.current.get(c) ?? id;
  }

  private linkId(oldId: number, newId: number) {
    const c = this.canonOf.get(oldId) ?? oldId;
    this.canonOf.set(oldId, c);
    this.canonOf.set(newId, c);
    this.current.set(c, newId);
  }

  /** Opens an explicit group; nested begin/end pairs are flattened into the outer group. */
  begin(label: string, coalesceKey?: string) {
    if (this.disposed) return;
    if (this.open && this.open.explicit === 0) this.close(); // finish a pending automatic group first
    if (!this.open) this.open = { label, coalesceKey, ops: [], explicit: 0 };
    this.open.explicit++;
  }

  end() {
    if (!this.open || this.open.explicit === 0) return;
    this.open.explicit--;
    if (this.open.explicit === 0) this.close();
  }

  /** Runs fn inside a group. */
  transact<T>(label: string, fn: () => T, coalesceKey?: string): T {
    this.begin(label, coalesceKey);
    try {
      return fn();
    } finally {
      this.end();
    }
  }

  /** Closes a pending automatic group now. */
  flush() {
    if (this.open && this.open.explicit === 0) this.close();
  }

  private onSceneChange(c: SceneChange) {
    if (this.disposed || this.applying) return;
    if (c.type === 'reset') {
      this.clear();
      return;
    }
    let op: HistoryOp;
    if (c.type === 'obstacle-added') op = { kind: 'add', id: c.obstacle.id, data: cloneData(c.obstacle) };
    else if (c.type === 'obstacle-removed') op = { kind: 'remove', id: c.obstacle.id, data: cloneData(c.obstacle) };
    else op = { kind: 'update', id: c.obstacle.id, before: cloneData(c.previous), after: cloneData(c.obstacle) };
    if (!this.open) this.open = { label: autoLabel(op), ops: [], explicit: 0 };
    this.open.ops.push(op);
    if (this.open.explicit === 0 && !this.autoScheduled) {
      this.autoScheduled = true;
      this.schedule(() => {
        this.autoScheduled = false;
        this.flush();
      });
    }
  }

  private close() {
    const g = this.open;
    this.open = null;
    if (!g) return;
    const ops = compressOps(g.ops);
    if (!ops.length) return;
    const t = this.now();
    const last = this.undoStack[this.undoStack.length - 1];
    if (g.coalesceKey && last && last.coalesceKey === g.coalesceKey && t - last.time < this.coalesceMs && this.redoStack.length === 0) {
      last.ops = compressOps([...last.ops, ...ops]);
      last.time = t;
      if (!last.ops.length) this.undoStack.pop();
    } else {
      this.undoStack.push({ label: g.label === 'auto' ? autoLabel(ops[0]) : g.label, ops, coalesceKey: g.coalesceKey, time: t });
      if (this.undoStack.length > this.limit) this.undoStack.splice(0, this.undoStack.length - this.limit);
    }
    this.redoStack = [];
    this.emit();
  }

  /** Undoes the last entry. Returns the ids (current) affected, or null if nothing to undo. */
  undo(): number[] | null {
    if (this.inTransaction) return null;
    this.flush();
    const e = this.undoStack.pop();
    if (!e) return null;
    const affected = this.apply(e.ops, true);
    this.redoStack.push({ ...e, time: this.now() });
    this.emit();
    return affected;
  }

  redo(): number[] | null {
    if (this.inTransaction) return null;
    this.flush();
    const e = this.redoStack.pop();
    if (!e) return null;
    const affected = this.apply(e.ops, false);
    // Re-pushed entries must never coalesce with new edits.
    this.undoStack.push({ ...e, coalesceKey: undefined, time: this.now() });
    this.emit();
    return affected;
  }

  private apply(ops: readonly HistoryOp[], inverse: boolean): number[] {
    const affected: number[] = [];
    this.applying = true;
    try {
      const seq = inverse ? [...ops].reverse() : ops;
      for (const op of seq) {
        const id = this.resolveId(op.id);
        const doAdd = (op.kind === 'add' && !inverse) || (op.kind === 'remove' && inverse);
        const doRemove = (op.kind === 'add' && inverse) || (op.kind === 'remove' && !inverse);
        if (doAdd) {
          const data = (op as { data: ObstacleData }).data;
          const o = this.scene.add(cloneData(data));
          this.linkId(op.id, o.id);
          affected.push(o.id);
        } else if (doRemove) {
          if (this.scene.get(id)) this.scene.remove(id);
        } else if (op.kind === 'update') {
          if (this.scene.get(id)) {
            this.scene.update(id, cloneData(inverse ? op.before : op.after));
            affected.push(id);
          }
        }
      }
    } finally {
      this.applying = false;
    }
    return affected;
  }

  clear() {
    this.undoStack = [];
    this.redoStack = [];
    this.open = this.open && this.open.explicit > 0 ? { ...this.open, ops: [] } : null;
    this.canonOf.clear();
    this.current.clear();
    this.emit();
  }

  dispose() {
    if (this.disposed) return;
    this.flush();
    this.disposed = true;
    const u = this.unsub;
    if (typeof u === 'function') u();
    this.listeners.clear();
  }
}

function autoLabel(op: HistoryOp): string {
  const what = op.kind === 'update' ? op.after.kind : op.data.kind;
  return op.kind === 'add' ? `add ${what}` : op.kind === 'remove' ? `delete ${what}` : `move ${what}`;
}
