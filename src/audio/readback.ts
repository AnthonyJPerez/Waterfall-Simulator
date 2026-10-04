/**
 * CPU-side views of the GPU coupling structs read back for audio (layouts: src/world/wgsl.ts).
 */
import {
  DEPOSIT_VOLUME_SCALE,
  IMPACT_EVENT_BYTES,
  MAX_IMPACT_EVENTS,
  SIM_EVENTS_HEADER_BYTES,
  STAT_TILES_X,
  STAT_TILES_Z,
  SWE_STATS_HEADER_BYTES,
  TILE_STAT_BYTES,
} from '../world/wgsl';

export const EVENT_FLOATS = IMPACT_EVENT_BYTES / 4; // posKind (4) + velVol (4)
export const TILE_FLOATS = TILE_STAT_BYTES / 4; // centroid (4) + flow (4)
export const NUM_TILES = STAT_TILES_X * STAT_TILES_Z;

export interface ParsedEvents {
  /** Number of events present in `data`. */
  n: number;
  /** n × 8 floats: px, py, pz, kind, vx, vy, vz, volume. */
  data: Float32Array;
  /** Raw append counter reported by the particle sim. */
  count: number;
  /** Events recorded in the GPU list but not copied (adaptive copy size too small). */
  unread: number;
  /** Impacts that did not fit in the list at all. */
  notRecorded: number;
  /** Volume (m³) of the impacts that did not fit (0 if the producer does not report it). */
  droppedVolume: number;
}

/** Parses a (possibly truncated) copy of SimEvents: header + the first k events. */
export function parseSimEvents(buf: ArrayBuffer): ParsedEvents {
  if (buf.byteLength < SIM_EVENTS_HEADER_BYTES) return { n: 0, data: new Float32Array(0), count: 0, unread: 0, notRecorded: 0, droppedVolume: 0 };
  const h = new Uint32Array(buf, 0, 4);
  const count = h[0];
  const droppedCount = h[1];
  const droppedVolume = h[2] / DEPOSIT_VOLUME_SCALE;
  const capacity = Math.floor((buf.byteLength - SIM_EVENTS_HEADER_BYTES) / IMPACT_EVENT_BYTES);
  const recorded = Math.min(count, MAX_IMPACT_EVENTS);
  const n = Math.min(recorded, capacity);
  const data = new Float32Array(buf, SIM_EVENTS_HEADER_BYTES, n * EVENT_FLOATS);
  return {
    n,
    data,
    count,
    unread: recorded - n,
    notRecorded: Math.max(droppedCount, count - MAX_IMPACT_EVENTS, 0),
    droppedVolume: Number.isFinite(droppedVolume) ? droppedVolume : 0,
  };
}

export interface ParsedStats {
  totalVolume: number;
  inflowRate: number;
  outflowRate: number;
  overflowRate: number;
  depositRate: number;
  maxSpeed: number;
  substeps: number;
  /** NUM_TILES × 8 floats: centroid x, y, z, wetted area (m²), mean speed, turbulence, jump intensity, mean depth. */
  tiles: Float32Array;
}

export function parseSweStats(buf: ArrayBuffer): ParsedStats | null {
  if (buf.byteLength < SWE_STATS_HEADER_BYTES + NUM_TILES * TILE_STAT_BYTES) return null;
  const f = new Float32Array(buf, 0, SWE_STATS_HEADER_BYTES / 4);
  const fin = (x: number) => (Number.isFinite(x) ? x : 0);
  return {
    totalVolume: fin(f[0]),
    inflowRate: fin(f[1]),
    outflowRate: fin(f[2]),
    overflowRate: fin(f[3]),
    depositRate: fin(f[4]),
    maxSpeed: fin(f[5]),
    substeps: fin(f[6]),
    tiles: new Float32Array(buf.slice(SWE_STATS_HEADER_BYTES, SWE_STATS_HEADER_BYTES + NUM_TILES * TILE_STAT_BYTES)),
  };
}

/** Builds a SimEvents buffer (tests / synthetic sources). */
export function buildSimEventsBuffer(events: { pos: number[]; kind: number; vel: number[]; vol: number }[], extra: { droppedCount?: number; droppedVolume?: number } = {}): ArrayBuffer {
  const n = Math.min(events.length, MAX_IMPACT_EVENTS);
  const buf = new ArrayBuffer(SIM_EVENTS_HEADER_BYTES + n * IMPACT_EVENT_BYTES);
  const h = new Uint32Array(buf, 0, 4);
  h[0] = events.length;
  h[1] = extra.droppedCount ?? 0;
  h[2] = Math.round((extra.droppedVolume ?? 0) * DEPOSIT_VOLUME_SCALE);
  const f = new Float32Array(buf, SIM_EVENTS_HEADER_BYTES, n * EVENT_FLOATS);
  for (let i = 0; i < n; i++) {
    const e = events[i];
    f.set([e.pos[0], e.pos[1], e.pos[2], e.kind, e.vel[0], e.vel[1], e.vel[2], e.vol], i * EVENT_FLOATS);
  }
  return buf;
}

/** Builds a SweStats buffer (tests / synthetic sources). */
export function buildSweStatsBuffer(
  tiles: { pos: number[]; area: number; speed: number; turbulence: number; jump: number; depth: number }[],
  totals: Partial<Omit<ParsedStats, 'tiles'>> = {},
): ArrayBuffer {
  const buf = new ArrayBuffer(SWE_STATS_HEADER_BYTES + NUM_TILES * TILE_STAT_BYTES);
  const f = new Float32Array(buf);
  f[0] = totals.totalVolume ?? 0;
  f[1] = totals.inflowRate ?? 0;
  f[2] = totals.outflowRate ?? 0;
  f[3] = totals.overflowRate ?? 0;
  f[4] = totals.depositRate ?? 0;
  f[5] = totals.maxSpeed ?? 0;
  f[6] = totals.substeps ?? 0;
  const base = SWE_STATS_HEADER_BYTES / 4;
  tiles.slice(0, NUM_TILES).forEach((t, i) => {
    f.set([t.pos[0], t.pos[1], t.pos[2], t.area, t.speed, t.turbulence, t.jump, t.depth], base + i * TILE_FLOATS);
  });
  return buf;
}
