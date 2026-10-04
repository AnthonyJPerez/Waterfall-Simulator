/**
 * Scene presets: terrain description, obstacles, water sources, initial water
 * and a camera pose. The terrain module interprets `terrain`.
 *
 * STUB CONTENT — the terrain agent replaces/extends these with hand-crafted scenes.
 */
import type { Params } from '../app/params';
import type { DomainExtent } from '../world/domain';
import type { InflowSource, Obstacle } from '../world/scene';

export interface TerrainSpec {
  seed: number;
  /** Generator id understood by src/terrain/generate.ts. */
  kind: string;
  /** Generator-specific parameters. */
  [key: string]: unknown;
}

export interface InitialWater {
  /** Axis-aligned region (world x/z) filled up to `level` (world y) at reset. */
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  level: number;
}

export interface ScenePreset {
  id: string;
  name: string;
  description: string;
  extent: DomainExtent;
  terrain: TerrainSpec;
  obstacles: Omit<Obstacle, 'id'>[];
  inflows: Omit<InflowSource, 'id'>[];
  initialWater: InitialWater[];
  /** Suggested flow parameters applied when the preset is loaded. */
  flow: Partial<Params['flow']>;
  camera: { position: [number, number, number]; target: [number, number, number] };
}

const rock = (x: number, y: number, z: number, r: number, seed: number, squash = 0.7): Omit<Obstacle, 'id'> => ({
  kind: 'boulder',
  position: [x, y, z],
  rotation: [0, Math.sin(seed) * 0.5, 0, Math.cos(seed) * 0.5].map((v, i, a) => v / Math.hypot(...a)) as any,
  scale: [r, r * squash, r * 0.85],
  seed,
  roughness: 0.18,
});

export const PRESETS: ScenePreset[] = [
  {
    id: 'ledge',
    name: 'Mossy Ledge Falls',
    description: 'A creek spilling over an uneven bedrock ledge into a clear plunge pool.',
    extent: { sizeX: 2.4, sizeZ: 1.2, minY: -0.35, maxY: 0.75 },
    terrain: { seed: 7, kind: 'ledge', ledgeX: 1.0, upstreamY: 0.3, poolY: -0.15, tailY: 0.0 },
    obstacles: [rock(0.55, 0.3, 0.45, 0.09, 3), rock(0.7, 0.3, 0.8, 0.07, 5), rock(1.9, 0.02, 0.3, 0.12, 9)],
    inflows: [{ kind: 'edge', x: 0, z: 0.6, width: 0.7, dir: [1, 0], share: 1 }],
    initialWater: [{ x0: 1.0, z0: 0.0, x1: 2.4, z1: 1.2, level: 0.08 }],
    flow: { rate: 3, velocity: 0.6 },
    camera: { position: [1.9, 0.65, 1.55], target: [1.1, 0.12, 0.6] },
  },
];

export function getPreset(id: string): ScenePreset {
  return PRESETS.find((p) => p.id === id) ?? PRESETS[0];
}
