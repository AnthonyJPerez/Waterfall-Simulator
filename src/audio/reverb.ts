/**
 * Procedural outdoor impulse response for a small rocky creek bed: a handful of discrete early reflections
 * (nearby rock faces / banks, 2.5–55 ms), then a short diffuse tail (foliage + rock scattering) whose
 * decay is frequency dependent (RT60 ≈ 1.2 s lows, 0.8 s mids, 0.35 s highs). Stereo-decorrelated,
 * normalised to unit energy so the reverb send gain directly sets the wet level.
 */
import { makeRng } from './physics';

export interface ImpulseOptions {
  duration: number;
  rt60Low: number;
  rt60Mid: number;
  rt60High: number;
  earlyTaps: number;
  /** Energy ratio tail / early. */
  tailRatio: number;
}

export const DEFAULT_IMPULSE: ImpulseOptions = {
  duration: 1.5,
  rt60Low: 1.2,
  rt60Mid: 0.8,
  rt60High: 0.35,
  earlyTaps: 12,
  tailRatio: 1.6,
};

export function generateImpulseResponse(sampleRate: number, seed = 11, opts: Partial<ImpulseOptions> = {}): [Float32Array, Float32Array] {
  const o = { ...DEFAULT_IMPULSE, ...opts };
  const fs = sampleRate > 0 ? sampleRate : 48000;
  const n = Math.max(64, Math.round(o.duration * fs));
  const rand = makeRng(seed);
  const out: [Float32Array, Float32Array] = [new Float32Array(n), new Float32Array(n)];
  for (let ch = 0; ch < 2; ch++) {
    const y = out[ch];
    // Early reflections (slightly low-passed by the rough rock: 3-tap kernel).
    const early = new Float32Array(n);
    for (let i = 0; i < o.earlyTaps; i++) {
      const t = 0.0025 + 0.0525 * Math.pow(rand(), 1.3);
      const k = Math.round(t * fs);
      const a = (rand() < 0.5 ? -1 : 1) * (0.5 + 0.5 * rand()) * Math.max(0.1, 1 - t / 0.08);
      if (k + 2 < n) {
        early[k] += 0.25 * a;
        early[k + 1] += 0.5 * a;
        early[k + 2] += 0.25 * a;
      }
    }
    // Diffuse tail: three bands of noise with their own decay.
    const tail = new Float32Array(n);
    const cLow = 1 - Math.exp((-2 * Math.PI * 500) / fs);
    const cHigh = 1 - Math.exp((-2 * Math.PI * 3500) / fs);
    let lpLow = 0;
    let lpHigh = 0;
    const dl = 6.907755 / o.rt60Low;
    const dm = 6.907755 / o.rt60Mid;
    const dh = 6.907755 / o.rt60High;
    for (let k = 0; k < n; k++) {
      const t = k / fs;
      const x = rand() * 2 - 1;
      lpLow += (x - lpLow) * cLow;
      lpHigh += (x - lpHigh) * cHigh;
      const low = lpLow;
      const mid = lpHigh - lpLow;
      const high = x - lpHigh;
      const onset = t < 0.006 ? 0 : t < 0.04 ? Math.pow((t - 0.006) / 0.034, 2) : 1;
      tail[k] = onset * (low * 2.2 * Math.exp(-dl * t) + mid * Math.exp(-dm * t) + high * 0.8 * Math.exp(-dh * t));
    }
    let eE = 0;
    let eT = 0;
    for (let k = 0; k < n; k++) {
      eE += early[k] * early[k];
      eT += tail[k] * tail[k];
    }
    const tailGain = eT > 0 && eE > 0 ? Math.sqrt((o.tailRatio * eE) / eT) : 1;
    // Fade the last 50 ms to avoid a truncation click.
    const fadeN = Math.min(n, Math.round(0.05 * fs));
    for (let k = 0; k < n; k++) {
      const fade = k > n - fadeN ? (n - k) / fadeN : 1;
      y[k] = (early[k] + tail[k] * tailGain) * fade;
    }
  }
  let e = 0;
  for (const y of out) for (let k = 0; k < n; k++) e += y[k] * y[k];
  const norm = e > 0 ? 1 / Math.sqrt(e / 2) : 1;
  for (const y of out) for (let k = 0; k < n; k++) y[k] *= norm;
  return out;
}

/** Soft-clip transfer curve: linear up to `knee`, tanh-shaped above, bounded by `ceiling` (< 1). */
export function softClipCurve(size = 4096, knee = 0.8, ceiling = 0.97): Float32Array {
  const c = new Float32Array(size);
  const room = ceiling - knee;
  for (let i = 0; i < size; i++) {
    const x = (i / (size - 1)) * 2 - 1;
    const ax = Math.abs(x);
    const y = ax <= knee ? ax : knee + room * Math.tanh((ax - knee) / room);
    c[i] = Math.sign(x) * y;
  }
  return c;
}
