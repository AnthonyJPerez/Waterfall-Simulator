/**
 * Formatting of the simulation statistics shown in the panel. The SWE / particle modules
 * expose free-form Record<string, number> stats; documented SweStats names get units, any
 * other key is shown generically.
 */

export type Stats = Readonly<Record<string, unknown>>;

const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

/** First finite numeric value among candidate keys. */
export function pickStat(stats: Stats, keys: readonly string[]): number | undefined {
  for (const k of keys) {
    const v = num(stats[k]);
    if (v !== undefined) return v;
  }
  return undefined;
}

/** Converts a value to litres/second given its key naming convention. */
function toLps(key: string, v: number) {
  return /lps|litre|liter/i.test(key) ? v : v * 1000;
}

export interface FlowReadouts {
  inflow: string;
  outflow: string;
  falling: string;
  speed: string;
  volume: string;
  particles: string;
}

const DASH = '—';

function findKey(stats: Stats, keys: readonly string[]) {
  return keys.find((k) => num(stats[k]) !== undefined);
}

export function flowReadouts(swe: Stats, particles: Stats): FlowReadouts {
  const fmt = (keys: string[], conv: (k: string, v: number) => string, s: Stats) => {
    const k = findKey(s, keys);
    return k ? conv(k, num(s[k])!) : DASH;
  };
  const lps = (k: string, v: number) => `${toLps(k, v).toFixed(2)} L/s`;
  const inflow = fmt(['inflowRate', 'inflow', 'inflowLps'], lps, swe);
  const outflow = fmt(['outflowRate', 'outflow', 'outflowLps'], lps, swe);
  const falling = fmt(['overflowRate', 'overflow', 'overflowLps'], lps, swe);
  const speed = fmt(['maxSpeed', 'maxVelocity', 'speedMax'], (_k, v) => `${v.toFixed(2)} m/s`, swe);
  const volume = fmt(['totalVolume', 'volume', 'waterVolume'], (k, v) => `${(/litre|liter|L$/.test(k) ? v : v * 1000).toFixed(1)} L`, swe);
  const alive = pickStat(particles, ['alive', 'particles', 'liveParticles', 'aliveParticles', 'count', 'active']);
  const diffuse = pickStat(particles, ['diffuse', 'diffuseAlive', 'aliveDiffuse', 'spray', 'diffuseParticles']);
  const pParts: string[] = [];
  if (alive !== undefined) pParts.push(`${formatCount(alive)} water`);
  if (diffuse !== undefined) pParts.push(`${formatCount(diffuse)} spray`);
  return { inflow, outflow, falling, speed, volume, particles: pParts.length ? pParts.join(' · ') : DASH };
}

export function formatCount(n: number) {
  if (!Number.isFinite(n)) return DASH;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return `${Math.round(n)}`;
}

const UNITS: Record<string, (v: number) => string> = {
  totalVolume: (v) => `${(v * 1000).toFixed(2)} L`,
  inflowRate: (v) => `${(v * 1000).toFixed(2)} L/s`,
  outflowRate: (v) => `${(v * 1000).toFixed(2)} L/s`,
  overflowRate: (v) => `${(v * 1000).toFixed(2)} L/s`,
  depositRate: (v) => `${(v * 1000).toFixed(2)} L/s`,
  maxSpeed: (v) => `${v.toFixed(2)} m/s`,
};

/** "key: value" lines for every stat (numbers to 4 significant digits). */
export function statLines(prefix: string, stats: Stats): string[] {
  const out: string[] = [];
  for (const [k, v] of Object.entries(stats ?? {})) {
    let s: string;
    if (typeof v === 'number') s = !Number.isFinite(v) ? String(v) : UNITS[k] ? UNITS[k](v) : `${+v.toPrecision(4)}`;
    else if (typeof v === 'string' || typeof v === 'boolean') s = String(v);
    else continue;
    out.push(`${prefix}${k}: ${s}`);
  }
  return out;
}
