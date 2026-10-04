import { describe, expect, it } from 'vitest';
import { AudioMapper, BUBBLE_KIND, IMPACT_DROP, IMPACT_POOL, IMPACT_SOLID, TUNING, type MapperInput } from '../src/audio/mapping';
import { makeRng, minnaertFrequency } from '../src/audio/physics';
import { LAYOUT } from '../src/audio/protocol';
import { buildSimEventsBuffer, buildSweStatsBuffer, parseSimEvents, parseSweStats } from '../src/audio/readback';
import { MAX_IMPACT_EVENTS } from '../src/world/wgsl';
import { fallScenario, listenerAt, PARTICLE_VOLUME, rapidScenario, type Scenario } from './audio.harness';

const gains = { bubbles: 1, roar: 1, ambience: 0.35 };

/** Runs the mapper alone for `seconds` and accumulates per-second statistics. */
function runMapper(sc: Scenario, seconds = 2, opts: Partial<MapperInput> = {}) {
  const m = new AudioMapper(8, 11);
  const rand = makeRng(5);
  const dt = 1 / 60;
  const frames = Math.round(seconds / dt);
  const acc = { bubbles: [0, 0, 0, 0, 0], bursts: 0, individual: 0, lumped: 0, noise: 0, total: 0, freqs: [] as number[], kinds: [] as number[], nanFree: true };
  for (let f = 0; f < frames; f++) {
    const ts = opts.timeScale ?? 1;
    const { events, stats } = sc.frame(f, dt * ts, rand);
    m.process({ simDt: dt * ts, realDt: dt, timeScale: ts, paused: false, listener: sc.listener, events, stats, gains, ...opts });
    const d = m.diag;
    d.bubbles.forEach((c, k) => (acc.bubbles[k] += c));
    acc.bursts += d.bursts;
    acc.individual += d.individualPower;
    acc.lumped += d.lumpedPower;
    acc.noise += d.rumblePower + d.midPower + d.hissPower;
    for (let i = 0; i < m.bubbles.n; i++) {
      acc.freqs.push(m.bubbles.data[i * LAYOUT.BUBBLE_STRIDE + LAYOUT.B_F0]);
      acc.kinds.push(m.bubbles.data[i * LAYOUT.BUBBLE_STRIDE + LAYOUT.B_KIND]);
    }
    for (const v of [...m.bubbles.data.subarray(0, m.bubbles.n * 8), ...m.bursts.data.subarray(0, m.bursts.n * 8), ...m.emitterParams, ...m.globals]) {
      if (!Number.isFinite(v)) acc.nanFree = false;
    }
  }
  const per = (x: number) => x / seconds;
  return {
    mapper: m,
    bubblesPerSec: per(acc.bubbles.reduce((a, b) => a + b)),
    kinds: acc.bubbles.map(per),
    burstsPerSec: per(acc.bursts),
    individual: acc.individual / frames,
    lumped: acc.lumped / frames,
    // Expected mean-square output at the reference distance (individual + lumped + continuous noise; lumped ⊂ noise).
    power: acc.individual / frames + acc.noise / frames,
    freqs: acc.freqs,
    kindsList: acc.kinds,
    nanFree: acc.nanFree,
  };
}

const dB = (p: number) => 10 * Math.log10(Math.max(p, 1e-30));

describe('AudioMapper: flow-rate scaling', () => {
  const trickle = runMapper(fallScenario('trickle', 0.05e-3, 0.01, 0.3, 0.1));
  const fall = runMapper(fallScenario('fall', 3e-3, 0.4, 0.45, 0.1));
  const cascade = runMapper(fallScenario('cascade', 20e-3, 0.6, 0.6, 0.1));

  it('0.05 L/s: a delicate trickle of individual drips and plinks', () => {
    // Every impact is handled individually: no lumped noise, a few dozen plinks per second.
    expect(trickle.lumped).toBe(0);
    expect(trickle.kinds[BUBBLE_KIND.drop]).toBeGreaterThan(10);
    expect(trickle.kinds[BUBBLE_KIND.drop]).toBeLessThan(200);
    expect(trickle.kinds[BUBBLE_KIND.jet]).toBe(0);
    // Plinks: mostly 1.5–6 kHz from crater-collapse bubbles of the drop train; sparkles above.
    const f = trickle.freqs.slice().sort((a, b) => a - b);
    const med = f[Math.floor(f.length / 2)];
    expect(med).toBeGreaterThan(1500);
    expect(med).toBeLessThan(6000);
    expect(trickle.nanFree).toBe(true);
  });

  it('3 L/s: a lively small fall dominated by the plunging bubble cloud', () => {
    expect(fall.kinds[BUBBLE_KIND.jet]).toBeGreaterThan(500);
    expect(fall.bubblesPerSec).toBeLessThanOrEqual(TUNING.bubbleBudget * 1.05);
    expect(fall.burstsPerSec).toBeLessThanOrEqual(TUNING.burstBudget * 1.05);
    expect(fall.nanFree).toBe(true);
  });

  it('20 L/s: a roaring cascade (dense population rendered as noise of equal power)', () => {
    expect(cascade.lumped).toBeGreaterThan(cascade.individual * 0.3);
    expect(cascade.bubblesPerSec).toBeLessThanOrEqual(TUNING.bubbleBudget * 1.05);
    expect(cascade.nanFree).toBe(true);
  });

  it('loudness grows monotonically and sensibly with flow', () => {
    const t = dB(trickle.power);
    const f = dB(fall.power);
    const c = dB(cascade.power);
    expect(f - t).toBeGreaterThan(6);
    expect(c - f).toBeGreaterThan(3);
    expect(c - f).toBeLessThan(16);
  });
});

describe('AudioMapper: stream babble from SWE tiles', () => {
  it('rate ∝ turbulence × area × speed, radii 1–8 mm (≈ 400–3300 Hz)', () => {
    const lo = runMapper(rapidScenario('lo', 0.1, 0.8, 4));
    const hi = runMapper(rapidScenario('hi', 0.2, 0.8, 4));
    const r = hi.kinds[BUBBLE_KIND.babble] / lo.kinds[BUBBLE_KIND.babble];
    expect(r).toBeGreaterThan(1.6);
    expect(r).toBeLessThan(2.5);
    const expected = TUNING.babbleRate * 0.1 * 0.0225 * 0.8 * 4;
    expect(Math.abs(lo.kinds[BUBBLE_KIND.babble] - expected) / expected).toBeLessThan(0.2);
    for (let i = 0; i < hi.freqs.length; i++) {
      if (hi.kindsList[i] !== BUBBLE_KIND.babble) continue;
      expect(hi.freqs[i]).toBeGreaterThan(minnaertFrequency(8.1e-3));
      expect(hi.freqs[i]).toBeLessThan(minnaertFrequency(0.5e-3));
    }
  });

  it('hydraulic jumps add low gurgles; dense babble is budgeted', () => {
    const r = runMapper(rapidScenario('rapid', 0.6, 1.5, 40));
    expect(r.kinds[BUBBLE_KIND.jump]).toBeGreaterThan(0);
    const jumpF = r.freqs.filter((_, i) => r.kindsList[i] === BUBBLE_KIND.jump);
    for (const f of jumpF) expect(f).toBeLessThan(1200);
    expect(r.bubblesPerSec).toBeLessThanOrEqual(TUNING.bubbleBudget * 1.05);
    expect(r.lumped).toBeGreaterThan(0);
  });
});

describe('AudioMapper: robustness and context', () => {
  it('survives garbage input without NaN', () => {
    const m = new AudioMapper(8, 1);
    const bad = parseSimEvents(
      buildSimEventsBuffer(
        [
          { pos: [NaN, 0, 0], kind: 0, vel: [0, -2, 0], vol: 1e-7 },
          { pos: [1, 0, 0], kind: 7, vel: [0, -2, 0], vol: 1e-7 },
          { pos: [1, 0, 0], kind: 0, vel: [Infinity, -2, 0], vol: -5 },
          { pos: [1, 0, 0], kind: 1, vel: [1e9, 1e9, 0], vol: 1e9 },
          { pos: [1e30, -1e30, 1e30], kind: 2, vel: [0, -1, 0], vol: 0 },
        ],
        { droppedCount: 100000, droppedVolume: 1 },
      ),
    );
    const tiles = parseSweStats(
      buildSweStatsBuffer([
        { pos: [NaN, 0, 0], area: 1, speed: 1, turbulence: 1, jump: 1, depth: 1 },
        { pos: [1, 0, 1], area: 1e9, speed: 1e9, turbulence: 1e9, jump: 1e9, depth: 1e9 },
        { pos: [1, 0, 1], area: -1, speed: -1, turbulence: -1, jump: -1, depth: -1 },
      ]),
    );
    for (const [simDt, realDt] of [
      [1 / 60, 1 / 60],
      [0, 1 / 60],
      [1e-7, 1e-7],
      [5, 5],
      [NaN, NaN],
    ]) {
      m.process({ simDt, realDt, timeScale: NaN, paused: false, listener: listenerAt([1, 1, 1], [1, 0, 0]), events: bad, stats: tiles, gains: { bubbles: NaN, roar: Infinity, ambience: -1 } });
      for (const v of [...m.bubbles.data.subarray(0, m.bubbles.n * 8), ...m.bursts.data.subarray(0, m.bursts.n * 8), ...m.emitterParams, ...m.globals]) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('accounts for impacts that did not fit in the event list', () => {
    const sc = fallScenario('fall', 3e-3, 0.4, 0.45, 0);
    const base = runMapper(sc, 0.5);
    const withDropped: Scenario = {
      ...sc,
      frame(i, dt, rand) {
        const r = sc.frame(i, dt, rand);
        return { ...r, events: r.events ? { ...r.events, notRecorded: 500, droppedVolume: 500 * PARTICLE_VOLUME } : null };
      },
    };
    const more = runMapper(withDropped, 0.5);
    expect(more.power).toBeGreaterThan(base.power * 1.5);
  });

  it('near-field drips keep their own emitter and survive the voice budget next to a loud fall', () => {
    const fall = fallScenario('fall', 20e-3, 0.6, 0.6, 0.1);
    const listener = listenerAt([1.2, 0.0, 2.6], [1.2, 0, 0.6]); // fall 2 m away
    const drip = [1.25, -0.05, 2.45]; // 15 cm from the camera
    const m = new AudioMapper(8, 3);
    const rand = makeRng(9);
    let dripBubbles = 0;
    let dripEmitter = -1;
    for (let f = 0; f < 120; f++) {
      const r = fall.frame(f, 1 / 60, rand);
      const evs = [];
      for (let i = 0; i < r.events!.n; i++) {
        const o = i * 8;
        evs.push({ pos: [r.events!.data[o], r.events!.data[o + 1], r.events!.data[o + 2]], kind: r.events!.data[o + 3], vel: [0, r.events!.data[o + 5], 0], vol: r.events!.data[o + 7] });
      }
      if (f % 6 === 0) evs.push({ pos: drip, kind: IMPACT_POOL, vel: [0, -2.4, 0], vol: 1.1e-7 });
      const events = parseSimEvents(buildSimEventsBuffer(evs.slice(-MAX_IMPACT_EVENTS)));
      m.process({ simDt: 1 / 60, realDt: 1 / 60, timeScale: 1, paused: false, listener, events, stats: null, gains });
      const em = m.clusterer.emitters;
      for (let e = 0; e < 8; e++) if (em[e].active && Math.hypot(em[e].x - drip[0], em[e].z - drip[2]) < 0.03) dripEmitter = e;
      for (let i = 0; i < m.bubbles.n; i++) {
        const o = i * LAYOUT.BUBBLE_STRIDE;
        if (m.bubbles.data[o + LAYOUT.B_EM] === dripEmitter && m.bubbles.data[o + LAYOUT.B_KIND] === BUBBLE_KIND.drop) dripBubbles++;
      }
    }
    expect(dripEmitter).toBeGreaterThanOrEqual(0);
    // 20 drips × entrainment probability (~0.3) — some plinks must have been synthesised individually.
    expect(dripBubbles).toBeGreaterThan(1);
    const g = m.emitterParams[dripEmitter * LAYOUT.EMITTER_STRIDE + LAYOUT.E_GAIN];
    expect(g).toBeGreaterThan(2);
    // Far emitters are softer and duller.
    const em = m.clusterer.emitters;
    const far = em.findIndex((e, i) => e.active && i !== dripEmitter && Math.hypot(e.x - 1.2, e.z - 0.6) < 0.4);
    expect(far).toBeGreaterThanOrEqual(0);
    expect(m.emitterParams[far * LAYOUT.EMITTER_STRIDE + LAYOUT.E_GAIN]).toBeLessThan(0.35);
    expect(m.emitterParams[far * LAYOUT.EMITTER_STRIDE + LAYOUT.E_AIR]).toBeLessThan(m.emitterParams[dripEmitter * LAYOUT.EMITTER_STRIDE + LAYOUT.E_AIR]);
  });

  it('slow motion lowers pitch and event rate coherently', () => {
    const sc = fallScenario('trickle', 0.05e-3, 0.01, 0.3, 0);
    const normal = runMapper(sc, 2);
    const slow = runMapper(sc, 2, { timeScale: 0.25 });
    const med = (a: number[]) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
    expect(med(slow.freqs) / med(normal.freqs)).toBeGreaterThan(0.35);
    expect(med(slow.freqs) / med(normal.freqs)).toBeLessThan(0.7);
    expect(slow.bubblesPerSec).toBeLessThan(normal.bubblesPerSec * 0.5);
  });

  it('pausing stops new sounds and fades the continuous layers (tails ring out in the synth)', () => {
    const sc = fallScenario('fall', 3e-3, 0.4, 0.45, 0.1);
    const m = new AudioMapper(8, 2);
    const rand = makeRng(1);
    for (let f = 0; f < 30; f++) {
      const r = sc.frame(f, 1 / 60, rand);
      m.process({ simDt: 1 / 60, realDt: 1 / 60, timeScale: 1, paused: false, listener: sc.listener, ...r, gains });
    }
    expect(m.bubbles.n).toBeGreaterThan(0);
    const r = sc.frame(31, 1 / 60, rand);
    m.process({ simDt: 0, realDt: 1 / 60, timeScale: 1, paused: true, listener: sc.listener, ...r, gains });
    expect(m.bubbles.n).toBe(0);
    expect(m.bursts.n).toBe(0);
    expect(m.globals[LAYOUT.G_WATER_FADE]).toBe(0);
    for (let e = 0; e < 8; e++) {
      const o = e * LAYOUT.EMITTER_STRIDE;
      expect(m.emitterParams[o + LAYOUT.E_RUMBLE] + m.emitterParams[o + LAYOUT.E_MID] + m.emitterParams[o + LAYOUT.E_HISS]).toBe(0);
    }
  });

  it('solid impacts make splatter bursts, not bubbles; droplets make high plinks', () => {
    const m = new AudioMapper(8, 4);
    let bubbles = 0;
    let bursts = 0;
    const listener = listenerAt([1, 0.5, 1], [1, 0, 0.5]);
    for (let f = 0; f < 60; f++) {
      const evs = Array.from({ length: 5 }, () => ({ pos: [1, 0.1, 0.5], kind: IMPACT_SOLID, vel: [0, -3, 0], vol: 2e-7 }));
      m.process({ simDt: 1 / 60, realDt: 1 / 60, timeScale: 1, paused: false, listener, events: parseSimEvents(buildSimEventsBuffer(evs)), stats: null, gains });
      bubbles += m.bubbles.n;
      bursts += m.bursts.n;
    }
    expect(bubbles).toBe(0);
    expect(bursts).toBeGreaterThan(300);
    // Small droplets (1 mm) at 3 m/s: regular entrainment → 10–25 kHz plinks.
    const d = new AudioMapper(8, 5);
    const freqs: number[] = [];
    for (let f = 0; f < 60; f++) {
      const evs = Array.from({ length: 3 }, () => ({ pos: [1, 0, 0.5], kind: IMPACT_DROP, vel: [0, -3, 0], vol: (Math.PI / 6) * 1e-9 }));
      d.process({ simDt: 1 / 60, realDt: 1 / 60, timeScale: 1, paused: false, listener, events: parseSimEvents(buildSimEventsBuffer(evs)), stats: null, gains });
      for (let i = 0; i < d.bubbles.n; i++) freqs.push(d.bubbles.data[i * LAYOUT.BUBBLE_STRIDE + LAYOUT.B_F0]);
    }
    expect(freqs.length).toBeGreaterThan(40);
    for (const f of freqs) expect(f).toBeGreaterThan(9000);
  });
});
