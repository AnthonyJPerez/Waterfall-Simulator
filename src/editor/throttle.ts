/**
 * Rate limiting for scene edits: while dragging / scrolling, the live pose changes every
 * frame but the terrain re-bakes on every SceneModel change, so commits are limited to a
 * maximum rate with a guaranteed trailing commit (flush) of the final value.
 */

export type Clock = () => number;

const defaultClock: Clock = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export class TrailingThrottle<T> {
  private pending: { value: T } | null = null;
  private last = -Infinity;
  /** Number of times the sink was invoked (diagnostics / tests). */
  commits = 0;

  /**
   * @param intervalMs minimum time between commits (e.g. 50 ms = 20 Hz)
   * @param sink receives committed values
   */
  constructor(
    public intervalMs: number,
    private sink: (value: T) => void,
    private now: Clock = defaultClock,
  ) {}

  /** Offers a new value: committed immediately when the interval has elapsed, else kept pending. */
  push(value: T) {
    this.pending = { value };
    this.tick();
  }

  /** Commits the pending value if the interval elapsed. Call every frame. */
  tick() {
    if (!this.pending) return;
    const t = this.now();
    if (t - this.last >= this.intervalMs) this.commit(t);
  }

  /** Commits the pending value now (e.g. on release). Returns true if something was committed. */
  flush(): boolean {
    if (!this.pending) return false;
    this.commit(this.now());
    return true;
  }

  /** Drops the pending value. */
  cancel() {
    this.pending = null;
  }

  get hasPending() {
    return this.pending !== null;
  }

  /** Allows the next push to commit immediately. */
  reset() {
    this.last = -Infinity;
  }

  private commit(t: number) {
    const p = this.pending!;
    this.pending = null;
    this.last = t;
    this.commits++;
    this.sink(p.value);
  }
}

/** Simple "at most every N ms" gate. */
export class RateGate {
  private last = -Infinity;
  constructor(
    public intervalMs: number,
    private now: Clock = defaultClock,
  ) {}
  ready(): boolean {
    const t = this.now();
    if (t - this.last < this.intervalMs) return false;
    this.last = t;
    return true;
  }
}
