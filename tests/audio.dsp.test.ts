import { describe, expect, it } from 'vitest';
import { bubbleDamping } from '../src/audio/physics';
import { LAYOUT, PROCESSOR_NAME, SYNTH } from '../src/audio/protocol';
import { generateImpulseResponse, softClipCurve } from '../src/audio/reverb';
import { workletSource } from '../src/audio/worklet';
import { db, fallScenario, FS, octaveBands, peak, powerSpectrum, rapidScenario, renderScenario, rms, spectralCentroid, wav } from './audio.harness';

const E = 4;

function makeSynth(seed = 1) {
  const s = new SYNTH.Synth(FS, E, seed);
  const em = new Float32Array(E * LAYOUT.EMITTER_STRIDE);
  for (let e = 0; e < E; e++) {
    const o = e * LAYOUT.EMITTER_STRIDE;
    em[o + LAYOUT.E_GAIN] = 1;
    em[o + LAYOUT.E_AIR] = 22000;
    em[o + LAYOUT.E_BRIGHT] = 12000;
    em[o + LAYOUT.E_RUMBLE_F] = 120;
    em[o + LAYOUT.E_MID_F] = 800;
    em[o + LAYOUT.E_HISS_F] = 3000;
    em[o + LAYOUT.E_ACTIVE] = 1;
  }
  s.setEmitters(em);
  const g = new Float32Array(LAYOUT.GLOBAL_STRIDE);
  g[LAYOUT.G_BUBBLES] = 1;
  g[LAYOUT.G_ROAR] = 1;
  g[LAYOUT.G_AMBIENCE] = 0;
  g[LAYOUT.G_WATER_FADE] = 1;
  g[LAYOUT.G_AMB_FADE] = 1;
  g[LAYOUT.G_WIND] = 0.5;
  g[LAYOUT.G_BIRDS] = 0;
  g[LAYOUT.G_OUT_GAIN] = 1;
  s.setGlobals(g);
  return { s, em, g };
}

/** Renders n samples; returns emitter 0 output and the sum of all emitters. */
function render(s: InstanceType<typeof SYNTH.Synth>, n: number) {
  const outs = Array.from({ length: E }, () => new Float32Array(128));
  const l = new Float32Array(128);
  const r = new Float32Array(128);
  const rv = new Float32Array(128);
  const e0 = new Float32Array(n);
  const mix = new Float32Array(n);
  const ambL = new Float32Array(n);
  const ambR = new Float32Array(n);
  for (let i = 0; i < n; i += 128) {
    const k = Math.min(128, n - i);
    s.render(outs, l, r, rv, k);
    for (let j = 0; j < k; j++) {
      e0[i + j] = outs[0][j];
      let m = 0;
      for (let e = 0; e < E; e++) m += outs[e][j];
      mix[i + j] = m;
      ambL[i + j] = l[j];
      ambR[i + j] = r[j];
    }
  }
  return { e0, mix, ambL, ambR };
}

function bubbleRecord(t: number, em: number, f0: number, d: number, chirp: number, amp: number) {
  const b = new Float32Array(LAYOUT.BUBBLE_STRIDE);
  b[LAYOUT.B_T] = t;
  b[LAYOUT.B_EM] = em;
  b[LAYOUT.B_F0] = f0;
  b[LAYOUT.B_D] = d;
  b[LAYOUT.B_CHIRP] = chirp;
  b[LAYOUT.B_AMP] = amp;
  return b;
}

describe('DSP core: bubble oscillator', () => {
  it('rings at f0 with the Minnaert damping, starts at zero and stops below -60 dB', () => {
    const { s } = makeSynth();
    const f0 = 2000;
    const d = bubbleDamping(f0);
    s.schedule(bubbleRecord(0, 0, f0, d, 0, 0.5), 1, null, 0, 0.01);
    const { e0 } = render(s, FS / 2);
    // Find onset (first non-zero sample) — must start at (almost) zero amplitude: no click.
    let on = 0;
    while (on < e0.length && e0[on] === 0) on++;
    expect(on).toBeGreaterThan(0);
    expect(Math.abs(e0[on])).toBeLessThan(0.5 * Math.sin((2 * Math.PI * f0) / FS) * 1.01 + 1e-6);
    // Frequency from zero crossings over the first 10 ms.
    let zc = 0;
    for (let i = on + 1; i < on + FS / 100; i++) if (e0[i - 1] <= 0 && e0[i] > 0) zc++;
    expect(Math.abs(zc / 0.01 - f0) / f0).toBeLessThan(0.06);
    // Decay rate: envelope ratio between two windows 5 ms apart ≈ e^(-d·0.005).
    const env = (a: number) => peak(e0.subarray(on + a, on + a + Math.round(FS / f0) + 2));
    const ratio = env(Math.round(0.01 * FS)) / env(Math.round(0.005 * FS));
    expect(Math.abs(Math.log(ratio) / -0.005 - d) / d).toBeLessThan(0.1);
    expect(s.activeVoices).toBe(0);
  });

  it('chirps upwards', () => {
    const { s } = makeSynth();
    const f0 = 1000;
    const d = 40;
    s.schedule(bubbleRecord(0, 0, f0, d, 4, 0.5), 1, null, 0, 0.01); // f(t) = f0 (1 + 4t): +40 % at 100 ms
    const { e0 } = render(s, FS / 4);
    const freqAt = (t0: number) => {
      let zc = 0;
      const a = Math.round(t0 * FS);
      const n = Math.round(0.02 * FS);
      for (let i = a + 1; i < a + n; i++) if (e0[i - 1] <= 0 && e0[i] > 0) zc++;
      return zc / 0.02;
    };
    const f1 = freqAt(0.015);
    const f2 = freqAt(0.1);
    expect(f2 / f1).toBeGreaterThan(1.25);
    expect(f2 / f1).toBeLessThan(1.55);
  });

  it('bounds voices, steals the quietest, ignores invalid records and never emits NaN', () => {
    const s = new SYNTH.Synth(FS, E, 1, { maxVoices: 64 });
    const n = 5000;
    const recs = new Float32Array(n * LAYOUT.BUBBLE_STRIDE);
    for (let i = 0; i < n; i++) {
      const o = i * LAYOUT.BUBBLE_STRIDE;
      recs[o + LAYOUT.B_T] = (i % 100) / 1000;
      recs[o + LAYOUT.B_EM] = i % 7; // out of range emitters are clamped
      recs[o + LAYOUT.B_F0] = i % 13 === 0 ? NaN : 200 + (i % 50) * 400;
      recs[o + LAYOUT.B_D] = i % 17 === 0 ? -5 : 100;
      recs[o + LAYOUT.B_CHIRP] = i % 19 === 0 ? 1e9 : 10;
      recs[o + LAYOUT.B_AMP] = i % 23 === 0 ? Infinity : 0.05;
    }
    s.schedule(recs, n, null, 0, 0.1);
    const outs = Array.from({ length: E }, () => new Float32Array(128));
    for (let b = 0; b < 200; b++) {
      s.render(outs, null, null, null, 128);
      expect(s.activeVoices).toBeLessThanOrEqual(64);
      for (const o of outs) for (const v of o) expect(Number.isFinite(v)).toBe(true);
    }
    expect(s.st.stolen).toBeGreaterThan(0);
  });
});

describe('DSP core: continuous layers', () => {
  it('noise layers are calibrated to rms level and ramp smoothly (no zipper / clicks)', () => {
    const { s, em } = makeSynth(2);
    em[LAYOUT.E_MID] = 0.1;
    em[LAYOUT.E_MID_F] = 800;
    s.setEmitters(em);
    // The water layers fade in over ~0.6 s when a synth is created: measure once settled.
    const a = render(s, 3 * FS);
    const r = rms(a.e0, 2.5 * FS);
    expect(r).toBeGreaterThan(0.1 * 0.7);
    expect(r).toBeLessThan(0.1 * 1.4);
    // Level ramps: the first 5 ms are far quieter than the steady state (80 ms smoothing).
    expect(rms(a.e0, 0, 240)).toBeLessThan(0.3 * r);
    for (const [band, f] of [
      [LAYOUT.E_RUMBLE, 120],
      [LAYOUT.E_HISS, 3000],
    ]) {
      const t = makeSynth(3);
      t.em[band] = 0.1;
      t.s.setEmitters(t.em);
      const x = render(t.s, 3 * FS);
      const rr = rms(x.e0, 2.5 * FS);
      expect(rr).toBeGreaterThan(0.1 * 0.6);
      expect(rr).toBeLessThan(0.1 * 1.6);
      // Spectrum lies on the right side of the cut-off.
      const c = spectralCentroid(x.e0, 2 * FS, 10);
      if (band === LAYOUT.E_RUMBLE) expect(c).toBeLessThan(f * 2.5);
      else expect(c).toBeGreaterThan(f);
    }
  });

  it('fades continuous layers on pause while bubble tails ring out', () => {
    const { s, em, g } = makeSynth(4);
    em[LAYOUT.E_HISS] = 0.2;
    s.setEmitters(em);
    render(s, FS);
    s.schedule(bubbleRecord(0, 1, 800, bubbleDamping(800), 0, 0.4), 1, null, 0, 0.01);
    g[LAYOUT.G_WATER_FADE] = 0;
    s.setGlobals(g);
    const x = render(s, 3 * FS);
    const early = rms(x.e0, 0, FS / 10);
    const late = rms(x.e0, 2.5 * FS, 3 * FS);
    expect(late).toBeLessThan(early * 0.05);
    // The bubble on emitter 1 rang out naturally (not cut by the fade).
    const b = rms(x.mix, 0, FS / 10);
    expect(b).toBeGreaterThan(early);
    // Smooth fade: no sample-to-sample jump larger than the steady-state noise allows.
    let maxJump = 0;
    for (let i = 1; i < x.e0.length; i++) maxJump = Math.max(maxJump, Math.abs(x.e0[i] - x.e0[i - 1]));
    expect(maxJump).toBeLessThan(1.5);
  });

  it('forest ambience: stereo, decorrelated, subtle; birds are synthesised', () => {
    const { s, g } = makeSynth(5);
    g[LAYOUT.G_AMBIENCE] = 1;
    g[LAYOUT.G_BIRDS] = 0;
    s.setGlobals(g);
    const x = render(s, 6 * FS);
    const from = FS;
    const l = rms(x.ambL, from);
    const r = rms(x.ambR, from);
    expect(db(l)).toBeGreaterThan(-55);
    expect(db(l)).toBeLessThan(-18);
    expect(db(r)).toBeGreaterThan(-55);
    let c = 0;
    for (let i = from; i < x.ambL.length; i++) c += x.ambL[i] * x.ambR[i];
    const corr = c / (x.ambL.length - from) / (l * r);
    expect(Math.abs(corr)).toBeLessThan(0.5);
    expect(s.st.birds).toBe(0);
    // Birds: distant FM chirps, energy mostly 2–7 kHz.
    const b = makeSynth(6);
    b.g[LAYOUT.G_AMBIENCE] = 1;
    b.g[LAYOUT.G_BIRDS] = 2;
    b.g[LAYOUT.G_WIND] = 0;
    b.s.setGlobals(b.g);
    const y = render(b.s, 6 * FS);
    expect(b.s.st.birds).toBeGreaterThan(3);
    expect(peak(y.ambL)).toBeLessThan(0.5);
    const birdBands = octaveBands(y.ambL, FS);
    expect(Math.max(birdBands[5], birdBands[6], birdBands[7])).toBe(0); // 2k / 4k / 8k octave holds the maximum
    for (const v of y.ambL) expect(Number.isFinite(v)).toBe(true);
  });
});

describe('offline renders of synthetic simulation streams', () => {
  const secs = 3;
  const from = 0.25 * FS;
  const trickle = renderScenario(fallScenario('trickle', 0.05e-3, 0.01, 0.3, 0.1), { seconds: secs });
  const fall = renderScenario(fallScenario('fall', 3e-3, 0.4, 0.45, 0.1), { seconds: secs });
  const cascade = renderScenario(fallScenario('cascade', 20e-3, 0.6, 0.6, 0.1), { seconds: secs });
  const rapid = renderScenario(rapidScenario('rapid', 0.5, 1.2, 30), { seconds: secs });

  // AUDIO_WAV=1 npx vitest run tests/audio.dsp.test.ts → writes the renders to shots/audio/*.wav for listening.
  it('optionally writes WAV files', async () => {
    const env = (globalThis as any).process?.env ?? {};
    if (!env.AUDIO_WAV) return;
    const fsName = 'node:fs';
    const fs = await import(/* @vite-ignore */ fsName);
    fs.mkdirSync('shots/audio', { recursive: true });
    for (const [n, r] of Object.entries({ trickle, fall, cascade, rapid })) fs.writeFileSync(`shots/audio/${n}.wav`, wav([r.mono]));
  });

  it('no NaN, no gross overs', () => {
    for (const r of [trickle, fall, cascade, rapid]) {
      for (const v of r.mono) expect(Number.isFinite(v)).toBe(true);
      // Pre-master headroom: the master limiter + soft clip follow, but the synth itself must stay sane.
      expect(peak(r.mono)).toBeLessThan(2.5);
    }
  });

  it('loudness ordering: trickle < rapid, small fall < cascade', () => {
    const L = (r: typeof trickle) => db(rms(r.mono, from));
    expect(L(trickle)).toBeLessThan(L(fall) - 6);
    expect(L(fall)).toBeLessThan(L(cascade) - 3);
    expect(L(trickle)).toBeLessThan(L(rapid));
    expect(L(cascade)).toBeLessThan(-3);
    expect(L(trickle)).toBeGreaterThan(-50);
  });

  it('trickle: distinct plinks (impulsive, 2–15 kHz centroid)', () => {
    const c = spectralCentroid(trickle.mono, from);
    expect(c).toBeGreaterThan(2000);
    expect(c).toBeLessThan(15000);
    // Crest factor of separate events is high (noise would be ~12 dB).
    expect(db(peak(trickle.mono)) - db(rms(trickle.mono, from))).toBeGreaterThan(14);
  });

  it('babble: centroid 300–3000 Hz', () => {
    const c = spectralCentroid(rapid.mono, from);
    expect(c).toBeGreaterThan(300);
    expect(c).toBeLessThan(3000);
  });

  it('cascade roar is broadband', () => {
    const bands = octaveBands(cascade.mono, from);
    expect(bands.filter((b) => b > -20).length).toBeGreaterThanOrEqual(7);
    // Low rumble present for big flows, much less for the trickle.
    const low = (r: typeof trickle) => {
      const ps = powerSpectrum(r.mono, 4096, from);
      let lo = 0;
      let all = 0;
      ps.forEach((p, i) => {
        const f = (i * FS) / 4096;
        if (f > 30 && f < 200) lo += p;
        all += p;
      });
      return lo / all;
    };
    expect(low(cascade)).toBeGreaterThan(low(trickle) * 3);
  });
});

describe('worklet module', () => {
  it('the generated source registers a working processor in an AudioWorkletGlobalScope', () => {
    const registered: Record<string, any> = {};
    class FakePort {
      onmessage: ((ev: { data: unknown }) => void) | null = null;
      sent: unknown[] = [];
      postMessage(m: unknown) {
        this.sent.push(m);
      }
    }
    class AudioWorkletProcessor {
      port = new FakePort();
    }
    const scope = {
      AudioWorkletProcessor,
      registerProcessor: (name: string, cls: unknown) => (registered[name] = cls),
      sampleRate: 44100,
      currentFrame: 0,
    };
    const src = workletSource();
    expect(src).toContain('registerProcessor');
    // Evaluate with only the worklet globals in scope (no access to this module's bindings).
    new Function('globalThis', src)(scope);
    const Proc = registered[PROCESSOR_NAME];
    expect(Proc).toBeTypeOf('function');
    const p = new Proc({ processorOptions: { numEmitters: 8, seed: 3 } });
    const outputs = [...Array.from({ length: 8 }, () => [new Float32Array(128)]), [new Float32Array(128), new Float32Array(128)], [new Float32Array(128)]];
    const em = new Float32Array(8 * LAYOUT.EMITTER_STRIDE);
    for (let e = 0; e < 8; e++) {
      em[e * LAYOUT.EMITTER_STRIDE + LAYOUT.E_GAIN] = 1;
      em[e * LAYOUT.EMITTER_STRIDE + LAYOUT.E_AIR] = 20000;
      em[e * LAYOUT.EMITTER_STRIDE + LAYOUT.E_ACTIVE] = 1;
    }
    const g = new Float32Array(LAYOUT.GLOBAL_STRIDE).fill(1);
    p.port.onmessage({ data: { type: 'frame', duration: 0.016, bubbles: bubbleRecord(0, 2, 1500, 200, 10, 0.3), nBubbles: 1, bursts: new Float32Array(0), nBursts: 0, emitters: em, globals: g } });
    p.port.onmessage({ data: null });
    p.port.onmessage({ data: { type: 'frame', bubbles: 'garbage' } });
    let energy = 0;
    for (let b = 0; b < 400; b++) {
      expect(p.process([], outputs)).toBe(true);
      for (const o of outputs) for (const ch of o) for (const v of ch) {
        expect(Number.isFinite(v)).toBe(true);
        energy += v * v;
      }
    }
    expect(energy).toBeGreaterThan(0);
    expect(p.port.sent.some((m: any) => m?.type === 'stats')).toBe(true);
    p.port.onmessage({ data: { type: 'dispose' } });
    expect(p.process([], outputs)).toBe(false);
  });
});

describe('reverb impulse response and output safety', () => {
  it('procedural IR: finite, unit energy, decays, decorrelated, early reflections', () => {
    const [l, r] = generateImpulseResponse(FS);
    let e = 0;
    let lr = 0;
    let el = 0;
    let er = 0;
    let tail = 0;
    let early = 0;
    for (let i = 0; i < l.length; i++) {
      expect(Number.isFinite(l[i]) && Number.isFinite(r[i])).toBe(true);
      e += l[i] * l[i] + r[i] * r[i];
      lr += l[i] * r[i];
      el += l[i] * l[i];
      er += r[i] * r[i];
      if (i > 0.8 * l.length) tail += l[i] * l[i];
      if (i < 0.06 * FS) early += l[i] * l[i];
    }
    expect(e / 2).toBeCloseTo(1, 3);
    expect(Math.abs(lr) / Math.sqrt(el * er)).toBeLessThan(0.3);
    expect(tail / el).toBeLessThan(0.01);
    expect(early / el).toBeGreaterThan(0.2);
  });

  it('soft clip curve is monotonic, linear below the knee and bounded', () => {
    const c = softClipCurve(4096, 0.8, 0.97);
    for (let i = 1; i < c.length; i++) expect(c[i]).toBeGreaterThanOrEqual(c[i - 1]);
    expect(Math.max(...c)).toBeLessThanOrEqual(0.97);
    const mid = Math.round(0.25 * 4095); // x = -0.5
    expect(c[mid]).toBeCloseTo(-0.5, 2);
  });
});
