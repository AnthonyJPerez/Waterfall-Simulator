/**
 * Groups sound sources (impact bins, SWE tiles) into a handful of spatial emitters (one PannerNode each)
 * with temporal coherence:
 *  - existing emitters keep their identity; points join the emitter that is perceptually closest,
 *  - the merge radius grows with the distance to the listener (≈ constant angular resolution), so sources
 *    close to the camera are kept separate (near-field detail) while distant ones are lumped,
 *  - positions are smoothed, emitters only retire after a hold time (so their tails can ring out),
 *  - when all emitters are busy, a much stronger new source may take over the weakest emitter.
 */

export interface ClusterPoint {
  x: number;
  y: number;
  z: number;
  /** Perceptual weight (estimated acoustic power × distance gain²). */
  w: number;
  /** Spatial half-extent of the source (m). */
  ext: number;
}

export interface EmitterState {
  active: boolean;
  x: number;
  y: number;
  z: number;
  /** Smoothed spatial extent (m, rms spread of the assigned sources). */
  ext: number;
  /** Smoothed perceptual weight. */
  weight: number;
  /** Seconds without any assigned source. */
  idle: number;
  /** True for the update in which the emitter was (re)spawned at a new location. */
  spawned: boolean;
  /** Monotonic id of the current occupant (changes on respawn). */
  generation: number;
  /** Merged into a neighbour: accepts no new sources, frees after the hold time. */
  retiring: boolean;
}

export interface ClusterOptions {
  /** Merge radius = clamp(angular · distance, minRadius, maxRadius). */
  angular: number;
  minRadius: number;
  maxRadius: number;
  /** Seconds an emitter stays allocated after its sources vanish. */
  hold: number;
  /** Position smoothing time constant (s). */
  tau: number;
  /** A new source may evict the weakest emitter if it is this many times stronger. */
  stealRatio: number;
}

export const DEFAULT_CLUSTER_OPTIONS: ClusterOptions = {
  angular: 0.22,
  minRadius: 0.035,
  maxRadius: 0.8,
  hold: 0.8,
  tau: 0.12,
  stealRatio: 4,
};

export class EmitterClusterer {
  readonly emitters: EmitterState[] = [];
  readonly opts: ClusterOptions;
  private order: number[] = [];
  private acc: Float64Array;
  private generationCounter = 0;

  constructor(readonly count: number, opts: Partial<ClusterOptions> = {}) {
    this.opts = { ...DEFAULT_CLUSTER_OPTIONS, ...opts };
    for (let i = 0; i < count; i++) {
      this.emitters.push({ active: false, x: 0, y: 0, z: 0, ext: 0.05, weight: 0, idle: 0, spawned: false, generation: 0, retiring: false });
    }
    this.acc = new Float64Array(count * 6);
  }

  mergeRadius(distToListener: number): number {
    const o = this.opts;
    return Math.min(o.maxRadius, Math.max(o.minRadius, o.angular * distToListener));
  }

  private dist(e: EmitterState, x: number, y: number, z: number) {
    return Math.hypot(e.x - x, e.y - y, e.z - z);
  }

  private spawn(i: number, p: ClusterPoint) {
    const e = this.emitters[i];
    e.active = true;
    e.x = p.x;
    e.y = p.y;
    e.z = p.z;
    e.ext = Math.max(0.01, p.ext);
    e.weight = p.w;
    e.idle = 0;
    e.spawned = true;
    e.retiring = false;
    e.generation = ++this.generationCounter;
    this.acc.fill(0, i * 6, i * 6 + 6);
  }

  /**
   * Assigns every point to an emitter (written to `assign[i]`) and updates emitter positions.
   * @param listener listener world position
   * @param dt real time since the previous update (s)
   */
  update(points: readonly ClusterPoint[], nPoints: number, listener: readonly number[], dt: number, assign: Int32Array): void {
    const E = this.emitters;
    const n = Math.min(nPoints, points.length, assign.length);
    const acc = this.acc;
    acc.fill(0);
    for (const e of E) e.spawned = false;

    const order = this.order;
    order.length = 0;
    for (let i = 0; i < n; i++) {
      const p = points[i];
      if (Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z) && p.w >= 0 && Number.isFinite(p.w)) order.push(i);
      else assign[i] = -1;
    }
    order.sort((a, b) => points[b].w - points[a].w);

    const lx = listener[0];
    const ly = listener[1];
    const lz = listener[2];
    for (const i of order) {
      const p = points[i];
      let best = -1;
      let bestRatio = Infinity;
      for (let k = 0; k < E.length; k++) {
        const e = E[k];
        if (!e.active || e.retiring) continue;
        const R = this.mergeRadius(Math.hypot(e.x - lx, e.y - ly, e.z - lz)) + 0.5 * p.ext;
        const ratio = this.dist(e, p.x, p.y, p.z) / R;
        if (ratio < bestRatio) {
          bestRatio = ratio;
          best = k;
        }
      }
      if (bestRatio > 1) {
        // Prefer a free slot, then an idle (retiring) one, then steal the weakest if much quieter.
        let slot = E.findIndex((e) => !e.active);
        if (slot < 0) {
          let idleMax = this.opts.hold * 0.5;
          for (let k = 0; k < E.length; k++) {
            if (acc[k * 6 + 3] === 0 && (E[k].idle > idleMax || (E[k].retiring && E[k].idle > idleMax * 0.5))) {
              idleMax = E[k].idle;
              slot = k;
            }
          }
        }
        if (slot < 0 && bestRatio > 2) {
          let weakest = -1;
          let wMin = Infinity;
          for (let k = 0; k < E.length; k++) {
            if (!E[k].active) continue;
            const wk = Math.max(E[k].weight, acc[k * 6 + 3]);
            if (wk < wMin) {
              wMin = wk;
              weakest = k;
            }
          }
          if (weakest >= 0 && p.w > this.opts.stealRatio * wMin && acc[weakest * 6 + 3] < p.w / this.opts.stealRatio) slot = weakest;
        }
        if (slot >= 0) {
          this.spawn(slot, p);
          best = slot;
          bestRatio = 0;
        }
      }
      if (best < 0) {
        assign[i] = -1;
        continue;
      }
      assign[i] = best;
      // Points force-assigned beyond the merge radius (all emitters busy) are routed to the emitter but
      // barely move its position.
      const forced = Math.max(1, bestRatio);
      const w = Math.max(p.w, 1e-12) / (forced * forced * forced * forced);
      const o = best * 6;
      acc[o] += w * p.x;
      acc[o + 1] += w * p.y;
      acc[o + 2] += w * p.z;
      acc[o + 3] += w;
      acc[o + 4] += w * (p.x * p.x + p.y * p.y + p.z * p.z);
      acc[o + 5] += w * p.ext * p.ext;
    }

    const k = 1 - Math.exp(-Math.max(0, dt) / Math.max(1e-3, this.opts.tau));
    for (let i = 0; i < E.length; i++) {
      const e = E[i];
      if (!e.active) continue;
      const o = i * 6;
      const W = acc[o + 3];
      if (W > 0) {
        const tx = acc[o] / W;
        const ty = acc[o + 1] / W;
        const tz = acc[o + 2] / W;
        const spread = Math.sqrt(Math.max(0, acc[o + 4] / W - (tx * tx + ty * ty + tz * tz)) + acc[o + 5] / W);
        if (e.spawned) {
          e.x = tx;
          e.y = ty;
          e.z = tz;
          e.ext = Math.max(0.01, spread);
        } else {
          e.x += (tx - e.x) * k;
          e.y += (ty - e.y) * k;
          e.z += (tz - e.z) * k;
          e.ext += (Math.max(0.01, spread) - e.ext) * k;
        }
        e.weight += (W - e.weight) * Math.max(k, 0.3);
        e.idle = 0;
      } else {
        e.idle += Math.max(0, dt);
        e.weight *= 1 - k;
        if (e.idle > this.opts.hold) {
          e.active = false;
          e.retiring = false;
        }
      }
    }

    // Merge emitters that converged onto each other (the weaker retires; it is idle from now on).
    for (let a = 0; a < E.length; a++) {
      const ea = E[a];
      if (!ea.active || ea.retiring || acc[a * 6 + 3] === 0) continue;
      for (let b = a + 1; b < E.length; b++) {
        const eb = E[b];
        if (!eb.active || eb.retiring || acc[b * 6 + 3] === 0) continue;
        const R = 0.35 * this.mergeRadius(Math.hypot(ea.x - lx, ea.y - ly, ea.z - lz));
        if (this.dist(ea, eb.x, eb.y, eb.z) < R) {
          const weak = acc[a * 6 + 3] < acc[b * 6 + 3] ? a : b;
          E[weak].retiring = true;
          if (weak === a) break;
        }
      }
    }
  }

  reset() {
    for (const e of this.emitters) {
      e.active = false;
      e.weight = 0;
      e.idle = 0;
      e.spawned = false;
      e.retiring = false;
    }
  }
}
