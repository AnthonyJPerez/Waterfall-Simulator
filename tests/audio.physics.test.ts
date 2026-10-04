import { describe, expect, it } from 'vitest';
import {
  bubbleDamping,
  bubbleEnergy,
  bubbleFrequencyAt,
  bubbleLifetime,
  bubbleQ,
  chirpRate,
  CumulativeTable,
  dropBubbleRadiusRange,
  dropEntrainmentProbability,
  equivalentDiameter,
  freeFallSpeed,
  makeRng,
  meanBubbleVolume,
  minnaertFrequency,
  minnaertRadius,
  PiecewisePowerLaw,
  plungingAirRatio,
  plungingBubbleDistribution,
  poisson,
} from '../src/audio/physics';

describe('Minnaert resonance and bubble damping', () => {
  it('f0 ≈ 3.26 / r', () => {
    expect(minnaertFrequency(1e-3)).toBeCloseTo(3260, 6);
    expect(minnaertFrequency(3e-3)).toBeCloseTo(1086.7, 1);
    // Full Minnaert formula f = sqrt(3γp0/ρ) / (2πr) with γ = 1.4, p0 = 101325 Pa.
    const exact = Math.sqrt((3 * 1.4 * 101325) / 1000) / (2 * Math.PI * 1e-3);
    expect(Math.abs(minnaertFrequency(1e-3) - exact) / exact).toBeLessThan(0.02);
    expect(minnaertRadius(minnaertFrequency(2.5e-3))).toBeCloseTo(2.5e-3, 9);
  });

  it('damping matches van den Doel (2005) d = 0.13/r + 0.0072 r^-1.5 within 15 %', () => {
    for (const f of [300, 600, 1000, 2000, 4000, 8000, 15000]) {
      const r = minnaertRadius(f);
      const ref = 0.13 / r + 0.0072 * Math.pow(r, -1.5);
      expect(Math.abs(bubbleDamping(f) - ref) / ref).toBeLessThan(0.15);
    }
    expect(bubbleDamping(1000)).toBeCloseTo(43 + 0.0014 * Math.pow(1000, 1.5), 9);
  });

  it('gives plausible resonance quality and lifetimes', () => {
    for (const f of [300, 1000, 3000, 10000, 20000]) {
      const q = bubbleQ(f);
      expect(q).toBeGreaterThan(3);
      expect(q).toBeLessThan(80);
    }
    // A 1 mm bubble (3.26 kHz) rings for a few tens of ms; a 5 mm one for ~0.1-0.3 s.
    expect(bubbleLifetime(bubbleDamping(3260))).toBeGreaterThan(0.005);
    expect(bubbleLifetime(bubbleDamping(3260))).toBeLessThan(0.05);
    expect(bubbleLifetime(bubbleDamping(652))).toBeGreaterThan(0.08);
    expect(bubbleLifetime(bubbleDamping(652))).toBeLessThan(0.4);
  });

  it('rising chirp f(t) = f0 (1 + ξ d t)', () => {
    const f0 = 2000;
    const d = bubbleDamping(f0);
    const s = chirpRate(d, 0.1);
    expect(bubbleFrequencyAt(f0, s, 0)).toBe(f0);
    // Over the -60 dB lifetime the pitch rises by ξ·6.9 ≈ 69 %.
    expect(bubbleFrequencyAt(f0, s, bubbleLifetime(d)) / f0).toBeCloseTo(1 + 0.1 * 6.907755, 3);
    expect(chirpRate(d, -1)).toBe(0);
  });

  it('energy of an exponentially decaying sinusoid', () => {
    const A = 0.3;
    const f = 1500;
    const d = bubbleDamping(f);
    const fs = 192000;
    let e = 0;
    for (let i = 0; i < fs; i++) {
      const t = i / fs;
      const y = A * Math.exp(-d * t) * Math.sin(2 * Math.PI * f * t);
      e += (y * y) / fs;
    }
    expect(Math.abs(e - bubbleEnergy(A, d)) / e).toBeLessThan(0.02);
  });
});

describe('bubble size distributions', () => {
  const dist = plungingBubbleDistribution(0.1e-3, 1e-3, 6e-3);

  it('is a normalised, continuous piecewise power law', () => {
    expect(dist.cdf(dist.min)).toBe(0);
    expect(dist.cdf(dist.max)).toBe(1);
    // Continuity at the Hinze scale.
    expect(dist.pdf(1e-3 * (1 - 1e-9)) / dist.pdf(1e-3 * (1 + 1e-9))).toBeCloseTo(1, 4);
    // Numerical normalisation.
    let s = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
      const a = Math.exp(Math.log(dist.min) + ((Math.log(dist.max) - Math.log(dist.min)) * i) / n);
      const b = Math.exp(Math.log(dist.min) + ((Math.log(dist.max) - Math.log(dist.min)) * (i + 1)) / n);
      s += dist.pdf(Math.sqrt(a * b)) * (b - a);
    }
    expect(s).toBeCloseTo(1, 3);
    for (const u of [0.01, 0.2, 0.5, 0.9, 0.999]) expect(dist.cdf(dist.quantile(u))).toBeCloseTo(u, 9);
  });

  it('sampling reproduces the r^-3/2 (below Hinze) and r^-10/3 (above) slopes', () => {
    const rand = makeRng(42);
    const N = 400000;
    const edges = Array.from({ length: 25 }, (_, i) => 0.1e-3 * Math.pow(60, i / 24));
    const hist = new Float64Array(24);
    for (let i = 0; i < N; i++) {
      const r = dist.sample(rand());
      let k = 0;
      while (k < 23 && r >= edges[k + 1]) k++;
      hist[k]++;
    }
    const slope = (k0: number, k1: number) => {
      // Density ∝ count / bin width; least-squares slope in log-log.
      const xs: number[] = [];
      const ys: number[] = [];
      for (let k = k0; k <= k1; k++) {
        xs.push(Math.log(Math.sqrt(edges[k] * edges[k + 1])));
        ys.push(Math.log(hist[k] / (edges[k + 1] - edges[k])));
      }
      const mx = xs.reduce((a, b) => a + b) / xs.length;
      const my = ys.reduce((a, b) => a + b) / ys.length;
      let num = 0;
      let den = 0;
      xs.forEach((x, i) => {
        num += (x - mx) * (ys[i] - my);
        den += (x - mx) ** 2;
      });
      return num / den;
    };
    // Bins entirely below 1 mm: edges < 1e-3 → k ≤ 12; above: k ≥ 14.
    expect(slope(1, 11)).toBeCloseTo(-1.5, 0);
    expect(Math.abs(slope(1, 11) + 1.5)).toBeLessThan(0.15);
    expect(Math.abs(slope(14, 21) + 10 / 3)).toBeLessThan(0.3);
  });

  it('truncated sampling stays in range and moments are consistent', () => {
    const rand = makeRng(3);
    for (let i = 0; i < 1000; i++) {
      const r = dist.sample(rand(), 2e-3, 4e-3);
      expect(r).toBeGreaterThanOrEqual(2e-3 * (1 - 1e-9));
      expect(r).toBeLessThanOrEqual(4e-3 * (1 + 1e-9));
    }
    // Monte-Carlo mean volume vs analytic moment.
    let s = 0;
    const N = 200000;
    for (let i = 0; i < N; i++) s += dist.sample(rand()) ** 3;
    expect(Math.abs(s / N - dist.moment(3)) / dist.moment(3)).toBeLessThan(0.05);
    const v = meanBubbleVolume(dist);
    expect(v).toBeGreaterThan(1e-9);
    expect(v).toBeLessThan(1e-8);
    // Log-uniform single segment (exponent -1).
    const lu = new PiecewisePowerLaw([1e-3, 8e-3], [-1]);
    expect(lu.quantile(0.5)).toBeCloseTo(Math.sqrt(8) * 1e-3, 9);
  });

  it('cumulative table integrates p(r)·g(r)', () => {
    const t = new CumulativeTable(dist, (r) => r ** 3);
    expect(Math.abs(t.total - dist.moment(3)) / dist.moment(3)).toBeLessThan(0.01);
    expect(Math.abs(t.at(1e-3) - dist.partialMoment(3, dist.min, 1e-3)) / dist.partialMoment(3, dist.min, 1e-3)).toBeLessThan(0.02);
    expect(t.at(0)).toBe(0);
  });
});

describe('drop impacts and plunging jets', () => {
  it('regular entrainment window (Pumphrey & Elmore) gives high-pitched plinks', () => {
    expect(dropEntrainmentProbability(1.0e-3, 3)).toBeGreaterThan(0.5);
    expect(dropEntrainmentProbability(0.4e-3, 3)).toBeLessThan(0.1);
    expect(dropEntrainmentProbability(1.0e-3, 0.8)).toBeLessThan(0.1);
    const [lo, hi] = dropBubbleRadiusRange(1.0e-3);
    expect(minnaertFrequency(hi)).toBeGreaterThan(10000);
    expect(minnaertFrequency(lo)).toBeLessThan(30000);
  });

  it('large drops entrain bigger, lower bubbles with moderate probability', () => {
    const p = dropEntrainmentProbability(5e-3, 2.5);
    expect(p).toBeGreaterThan(0.15);
    expect(p).toBeLessThan(0.6);
    expect(dropEntrainmentProbability(5e-3, 0.3)).toBeLessThan(0.02);
    const [lo, hi] = dropBubbleRadiusRange(5e-3);
    expect(minnaertFrequency(hi)).toBeGreaterThan(1000);
    expect(minnaertFrequency(lo)).toBeLessThan(6000);
    for (let D = 0.2e-3; D < 15e-3; D *= 1.3) for (let v = 0; v < 12; v += 0.7) {
      const q = dropEntrainmentProbability(D, v);
      expect(q).toBeGreaterThanOrEqual(0);
      expect(q).toBeLessThanOrEqual(1);
    }
  });

  it('plunging air entrainment starts near 1 m/s and grows with impact speed', () => {
    expect(plungingAirRatio(0.5)).toBe(0);
    expect(plungingAirRatio(0.9)).toBe(0);
    let prev = 0;
    for (let v = 1; v < 8; v += 0.5) {
      const b = plungingAirRatio(v);
      expect(b).toBeGreaterThanOrEqual(prev);
      prev = b;
    }
    const b3 = plungingAirRatio(freeFallSpeed(0.45));
    expect(b3).toBeGreaterThan(0.01);
    expect(b3).toBeLessThan(0.2);
  });

  it('geometry helpers', () => {
    expect(equivalentDiameter((Math.PI / 6) * 1e-9)).toBeCloseTo(1e-3, 9);
    expect(freeFallSpeed(0.3)).toBeCloseTo(2.426, 2);
  });
});

describe('random helpers', () => {
  it('poisson has the right mean and variance', () => {
    const rand = makeRng(7);
    for (const lam of [0.3, 3, 25, 200]) {
      const N = 20000;
      let s = 0;
      let s2 = 0;
      for (let i = 0; i < N; i++) {
        const k = poisson(lam, rand);
        s += k;
        s2 += k * k;
      }
      const mean = s / N;
      const v = s2 / N - mean * mean;
      expect(Math.abs(mean - lam) / lam).toBeLessThan(0.05);
      expect(Math.abs(v - lam) / lam).toBeLessThan(0.1);
    }
    expect(poisson(0, rand)).toBe(0);
    expect(poisson(NaN, rand)).toBe(0);
  });

  it('rng is deterministic and uniform', () => {
    const a = makeRng(5);
    const b = makeRng(5);
    let s = 0;
    for (let i = 0; i < 10000; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      s += x;
    }
    expect(s / 10000).toBeCloseTo(0.5, 1);
  });
});
