/**
 * CPU-side scene description: the immovable obstacles (rocks, logs) and water
 * sources. The editor mutates it; the terrain module listens and re-bakes the
 * GPU collision/bed representations; the simulations react automatically.
 *
 * Coordinate system (shared by every module):
 *   - Units: metres, seconds, kilograms.
 *   - Right-handed, +Y up. The stream flows mainly towards +X.
 *   - The simulation domain spans x ∈ [0, domain.sizeX], z ∈ [0, domain.sizeZ],
 *     y ∈ [domain.minY, domain.maxY].
 */
import type { ObstacleKind } from '../app/params';

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // x, y, z, w

export interface Obstacle {
  id: number;
  kind: ObstacleKind;
  /** World position of the obstacle's local origin (shape centre). */
  position: Vec3;
  /** Orientation quaternion (x, y, z, w). */
  rotation: Quat;
  /** Half-extents / radii along local axes (m). For logs: x = half length, y = z = radius. */
  scale: Vec3;
  /** Shape seed: drives the procedural surface detail (deterministic). */
  seed: number;
  /** Surface noise amplitude relative to size (0..0.5). */
  roughness: number;
  /** Optional: moss coverage hint 0..1 for rendering. */
  moss?: number;
}

export interface InflowSource {
  id: number;
  /**
   * 'edge': water enters through the domain's x = 0 boundary, centred at `z`
   *         across `width` (m), flowing +X.
   * 'point': a spring / overflow pipe: water appears in a disc of radius width/2
   *          at (x, z) with horizontal direction `dir`.
   */
  kind: 'edge' | 'point';
  x: number;
  z: number;
  width: number;
  /** Unit horizontal direction for 'point' sources (ignored for 'edge'). */
  dir: [number, number];
  /** Fraction of the global flow rate (params.flow.rate) delivered by this source. */
  share: number;
}

export type SceneChange =
  | { type: 'obstacle-added'; obstacle: Obstacle }
  | { type: 'obstacle-removed'; obstacle: Obstacle }
  | { type: 'obstacle-moved'; obstacle: Obstacle; previous: Obstacle }
  | { type: 'reset' };

export class SceneModel {
  obstacles: Obstacle[] = [];
  inflows: InflowSource[] = [];
  private nextId = 1;
  private listeners = new Set<(c: SceneChange) => void>();
  /** Bumped on every change; consumers can compare to detect staleness. */
  version = 0;

  onChange(fn: (c: SceneChange) => void) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(c: SceneChange) {
    this.version++;
    this.listeners.forEach((l) => l(c));
  }

  reset(obstacles: Omit<Obstacle, 'id'>[], inflows: Omit<InflowSource, 'id'>[]) {
    this.obstacles = obstacles.map((o) => ({ ...o, id: this.nextId++ }));
    this.inflows = inflows.map((s) => ({ ...s, id: this.nextId++ }));
    this.emit({ type: 'reset' });
  }

  add(o: Omit<Obstacle, 'id'>): Obstacle {
    const obstacle = { ...o, id: this.nextId++ };
    this.obstacles.push(obstacle);
    this.emit({ type: 'obstacle-added', obstacle });
    return obstacle;
  }

  remove(id: number) {
    const i = this.obstacles.findIndex((o) => o.id === id);
    if (i < 0) return;
    const [obstacle] = this.obstacles.splice(i, 1);
    this.emit({ type: 'obstacle-removed', obstacle });
  }

  update(id: number, patch: Partial<Omit<Obstacle, 'id'>>) {
    const o = this.obstacles.find((x) => x.id === id);
    if (!o) return;
    const previous = { ...o, position: [...o.position] as Vec3, rotation: [...o.rotation] as Quat, scale: [...o.scale] as Vec3 };
    Object.assign(o, patch);
    this.emit({ type: 'obstacle-moved', obstacle: o, previous });
  }

  get(id: number) {
    return this.obstacles.find((o) => o.id === id);
  }
}
