/**
 * Offline rendering harness for the audio module: synthetic simulation streams (trickle, small fall,
 * cascade, rapid) → AudioMapper → the same DSP core the AudioWorklet runs → mono / per-emitter signals.
 * Shared by the audio tests (not a test file itself).
 */
import { AudioMapper, IMPACT_POOL, IMPACT_SOLID, type ListenerPose } from '../src/audio/mapping';
import { makeRng, poisson, freeFallSpeed } from '../src/audio/physics';
import { LAYOUT, SYNTH } from '../src/audio/protocol';
import { buildSimEventsBuffer, buildSweStatsBuffer, parseSimEvents, parseSweStats, type ParsedEvents, type ParsedStats } from '../src/audio/readback';

export const FS = 48000;
export const PARTICLE_VOLUME = 2.16e-7; // medium quality: (2 × 3 mm)³

export interface Scenario {
  name: string;
  /** Called once per frame: returns the impact events (or null) and SWE stats (or null). */
  frame(i: number, dt: number, rand: () => number): { events: ParsedEvents | null; stats: ParsedStats | null };
  listener: ListenerPose;
}

export function listenerAt(pos: [number, number, number], target: [number, number, number]): ListenerPose {
  const f = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]];
  const l = Math.hypot(f[0], f[1], f[2]);
  const fwd = [f[0] / l, f[1] / l, f[2] / l];
  const r = [fwd[1] * 0 - fwd[2] * 1, fwd[2] * 0 - fwd[0] * 0, fwd[0] * 1 - fwd[1] * 0];
  const rl = Math.hypot(r[0], r[1], r[2]) || 1;
  const right = [r[0] / rl, r[1] / rl, r[2] / rl];
  const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
  return { pos, fwd, right, up };
}

/**
 * A falling sheet/stream of flow Q (m³/s) of width w (m, along z) dropping height h onto a pool at
 * (x0, y0, z0); `solidShare` of the impacts hit rock instead.
 */
export function fallScenario(name: string, Q: number, width: number, height: number, solidShare = 0.1, listenerDist = 1.0): Scenario {
  const x0 = 1.2;
  const y0 = -0.05;
  const z0 = 0.6;
  const v = freeFallSpeed(height);
  return {
    name,
    listener: listenerAt([x0 + listenerDist * 0.8, y0 + listenerDist * 0.6, z0], [x0, y0, z0]),
    frame(_i, dt, rand) {
      const n = poisson((Q * dt) / PARTICLE_VOLUME, rand);
      const evs = [];
      for (let k = 0; k < n; k++) {
        const z = z0 + (rand() - 0.5) * width;
        const x = x0 + (rand() - 0.5) * 0.02;
        const solid = rand() < solidShare;
        const vv = v * (0.9 + 0.2 * rand());
        evs.push({ pos: [x, y0, z], kind: solid ? IMPACT_SOLID : IMPACT_POOL, vel: [0.3, -vv, 0], vol: PARTICLE_VOLUME });
      }
      return { events: parseSimEvents(buildSimEventsBuffer(evs)), stats: null };
    },
  };
}

/** Riffle / small rapid: no falling water, only SWE tiles with turbulence. */
export function rapidScenario(name = 'rapid', turbulence = 0.5, speed = 1.2, nTiles = 30, listenerDist = 1.0): Scenario {
  const tiles = [];
  for (let i = 0; i < nTiles; i++) {
    const tx = i % 10;
    const tz = Math.floor(i / 10);
    tiles.push({ pos: [0.8 + tx * 0.15, 0.0, 0.45 + tz * 0.15], area: 0.0225, speed: speed, turbulence, jump: i % 7 === 0 ? 0.3 : 0, depth: 0.05 });
  }
  const stats = parseSweStats(buildSweStatsBuffer(tiles))!;
  return {
    name,
    listener: listenerAt([1.55 + listenerDist * 0.8, listenerDist * 0.6, 0.6], [1.55, 0, 0.6]),
    frame: () => ({ events: null, stats }),
  };
}

export interface RenderResult {
  mono: Float32Array;
  emitters: Float32Array[];
  ambL: Float32Array;
  ambR: Float32Array;
  rev: Float32Array;
  mapper: AudioMapper;
  bubblesPerSec: number;
  burstsPerSec: number;
  diag: { individualPower: number; lumpedPower: number; jetRate: number; babbleRate: number; bubbles: number[] };
}

export interface RenderOptions {
  seconds?: number;
  fps?: number;
  seed?: number;
  gains?: { bubbles: number; roar: number; ambience: number };
  timeScale?: number;
  paused?: (t: number) => boolean;
}

/** Runs mapper + synth exactly like the engine/worklet do (one batch per frame, 128-sample blocks). */
export function renderScenario(sc: Scenario, opts: RenderOptions = {}): RenderResult {
  const seconds = opts.seconds ?? 2;
  const fps = opts.fps ?? 60;
  const dt = 1 / fps;
  const E = 8;
  const mapper = new AudioMapper(E, opts.seed ?? 7);
  mapper.sampleRate = FS;
  const synth = new SYNTH.Synth(FS, E, opts.seed ?? 3);
  const rand = makeRng((opts.seed ?? 7) * 31 + 1);
  const total = Math.round(seconds * FS);
  const mono = new Float32Array(total);
  const emitters = Array.from({ length: E }, () => new Float32Array(total));
  const ambL = new Float32Array(total);
  const ambR = new Float32Array(total);
  const rev = new Float32Array(total);
  const outs = Array.from({ length: E }, () => new Float32Array(128));
  const bl = new Float32Array(128);
  const br = new Float32Array(128);
  const rv = new Float32Array(128);
  let rendered = 0;
  let nextFrameAt = 0;
  let frame = 0;
  let bubbles = 0;
  let bursts = 0;
  const diag = { individualPower: 0, lumpedPower: 0, jetRate: 0, babbleRate: 0, bubbles: [0, 0, 0, 0, 0] };
  const gains = opts.gains ?? { bubbles: 1, roar: 1, ambience: 0 };
  while (rendered < total) {
    while (rendered >= nextFrameAt) {
      const t = frame * dt;
      const paused = opts.paused?.(t) ?? false;
      const ts = opts.timeScale ?? 1;
      const { events, stats } = sc.frame(frame, dt * ts, rand);
      mapper.process({
        simDt: paused ? 0 : dt * ts,
        realDt: dt,
        timeScale: ts,
        paused,
        listener: sc.listener,
        events: paused ? null : events,
        stats,
        gains,
      });
      synth.setEmitters(mapper.emitterParams);
      synth.setGlobals(mapper.globals);
      synth.schedule(mapper.bubbles.data, mapper.bubbles.n, mapper.bursts.data, mapper.bursts.n, dt);
      bubbles += mapper.bubbles.n;
      bursts += mapper.bursts.n;
      diag.individualPower += mapper.diag.individualPower;
      diag.lumpedPower += mapper.diag.lumpedPower;
      diag.jetRate += mapper.diag.jetRate;
      diag.babbleRate += mapper.diag.babbleRate;
      mapper.diag.bubbles.forEach((c, k) => (diag.bubbles[k] += c));
      frame++;
      nextFrameAt = Math.round(frame * dt * FS);
    }
    const n = Math.min(128, total - rendered);
    synth.render(outs, bl, br, rv, n);
    for (let k = 0; k < n; k++) {
      let s = 0;
      for (let e = 0; e < E; e++) {
        emitters[e][rendered + k] = outs[e][k];
        s += outs[e][k];
      }
      mono[rendered + k] = s;
      ambL[rendered + k] = bl[k];
      ambR[rendered + k] = br[k];
      rev[rendered + k] = rv[k];
    }
    rendered += n;
  }
  diag.individualPower /= frame;
  diag.lumpedPower /= frame;
  diag.jetRate /= frame;
  diag.babbleRate /= frame;
  return { mono, emitters, ambL, ambR, rev, mapper, bubblesPerSec: bubbles / seconds, burstsPerSec: bursts / seconds, diag };
}

// ---------------------------------------------------------------------------------------------------
// Analysis helpers

export function rms(x: Float32Array, from = 0, to = x.length): number {
  let s = 0;
  for (let i = from; i < to; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, to - from));
}

export function peak(x: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  return m;
}

export const db = (x: number) => 20 * Math.log10(Math.max(x, 1e-12));

/** In-place radix-2 FFT (re, im arrays of length 2^k). */
export function fft(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k + len / 2] = re[i + k] - ar;
        im[i + k + len / 2] = im[i + k] - ai;
        re[i + k] += ar;
        im[i + k] += ai;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/** Averaged (Welch, Hann) power spectrum; returns power per bin (bin width FS / n). */
export function powerSpectrum(x: Float32Array, n = 4096, from = 0): Float64Array {
  const ps = new Float64Array(n / 2);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let count = 0;
  for (let s = from; s + n <= x.length; s += n / 2) {
    for (let i = 0; i < n; i++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
      re[i] = x[s + i] * w;
      im[i] = 0;
    }
    fft(re, im);
    for (let i = 0; i < n / 2; i++) ps[i] += re[i] * re[i] + im[i] * im[i];
    count++;
  }
  if (count) for (let i = 0; i < n / 2; i++) ps[i] /= count;
  return ps;
}

export function spectralCentroid(x: Float32Array, from = 0, fmin = 30): number {
  const n = 4096;
  const ps = powerSpectrum(x, n, from);
  let a = 0;
  let b = 0;
  for (let i = 1; i < ps.length; i++) {
    const f = (i * FS) / n;
    if (f < fmin) continue;
    a += f * ps[i];
    b += ps[i];
  }
  return b > 0 ? a / b : 0;
}

/** Energy per octave band centred at 63, 125, …, 16000 Hz (dB, relative to the strongest band). */
export function octaveBands(x: Float32Array, from = 0): number[] {
  const n = 4096;
  const ps = powerSpectrum(x, n, from);
  const centres = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
  const e = centres.map((c) => {
    let s = 0;
    for (let i = 1; i < ps.length; i++) {
      const f = (i * FS) / n;
      if (f >= c / Math.SQRT2 && f < c * Math.SQRT2) s += ps[i];
    }
    return s;
  });
  const m = Math.max(...e);
  return e.map((v) => 10 * Math.log10(Math.max(v, 1e-30) / m));
}

/** 16-bit PCM WAV encoder (for listening to the offline renders). */
export function wav(channels: Float32Array[], fs = FS): Uint8Array {
  const n = channels[0].length;
  const ch = channels.length;
  const buf = new ArrayBuffer(44 + n * ch * 2);
  const v = new DataView(buf);
  const s = (o: number, t: string) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  s(0, 'RIFF');
  v.setUint32(4, 36 + n * ch * 2, true);
  s(8, 'WAVE');
  s(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, ch, true);
  v.setUint32(24, fs, true);
  v.setUint32(28, fs * ch * 2, true);
  v.setUint16(32, ch * 2, true);
  v.setUint16(34, 16, true);
  s(36, 'data');
  v.setUint32(40, n * ch * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < ch; c++) {
      v.setInt16(o, Math.max(-32767, Math.min(32767, Math.round(channels[c][i] * 32767))), true);
      o += 2;
    }
  return new Uint8Array(buf);
}

export { LAYOUT };
