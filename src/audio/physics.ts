/**
 * Physical acoustics of small water features — pure, dependency-free helpers.
 *
 * References
 *  - Minnaert (1933): resonance of a gas bubble in liquid, f0 = (1 / 2πr)·sqrt(3γp0/ρ) ≈ 3.26 / r  [Hz, r in m].
 *  - van den Doel (2005) "Physically based models for liquid sounds": bubble = A·sin(2π∫f dt)·e^(−d t),
 *    d = 0.13/r + 0.0072·r^(−3/2) ≈ 0.043·f0 + 0.0014·f0^1.5, rising chirp f(t) = f0·(1 + σt), σ = ξ·d.
 *  - Pumphrey & Elmore (1990), Oguz & Prosperetti (1990): regular entrainment window for drop impacts
 *    (D ≈ 0.8–1.2 mm at ≈ 2–4.5 m/s → bubbles r ≈ 0.15–0.35 mm, 10–20 kHz "plink"); large drops (D ≳ 2.5 mm)
 *    entrain bigger bubbles irregularly when the crater collapses ("plunk", ≈ 1–3 kHz).
 *  - Garrett, Li & Farmer (2000), Deane & Stokes (2002): plunging bubble-size spectrum ∝ r^(−10/3) above the
 *    Hinze scale (≈ 1 mm), flatter (∝ r^(−3/2)) below.
 *  - Ervine (1997), Bin (1993): plunging-jet air entrainment starts at ≈ 0.8–1 m/s impact speed and grows
 *    super-linearly with speed (Q_air / Q_water ≈ 0.01–0.15 for the falls simulated here).
 *
 * Everything is in SI units (m, s, Hz, m³).
 */

export const MINNAERT_CONSTANT = 3.26; // Hz·m (air bubble in water at 1 atm, γ = 1.4)
export const WATER_DENSITY = 1000;
export const GRAVITY = 9.81;

/** Minnaert resonance frequency (Hz) of a bubble of radius r (m). */
export function minnaertFrequency(r: number): number {
  return MINNAERT_CONSTANT / Math.max(r, 1e-6);
}

/** Bubble radius (m) resonating at frequency f (Hz). */
export function minnaertRadius(f: number): number {
  return MINNAERT_CONSTANT / Math.max(f, 1e-3);
}

/** Total damping constant d (1/s) of a bubble ringing at f0 (thermal + viscous + radiation; van den Doel 2005). */
export function bubbleDamping(f0: number): number {
  const f = Math.max(f0, 0);
  return 0.043 * f + 0.0014 * Math.pow(f, 1.5);
}

/** Quality factor of the bubble resonance (π f0 / d). */
export function bubbleQ(f0: number): number {
  return (Math.PI * f0) / Math.max(bubbleDamping(f0), 1e-9);
}

/** Chirp rate σ (1/s) for f(t) = f0·(1 + σ t); ξ ≈ 0.1 near the surface, → 0 for deep bubbles. */
export function chirpRate(d: number, xi: number): number {
  return Math.max(0, xi) * d;
}

/** Instantaneous frequency of a chirping bubble at time t. */
export function bubbleFrequencyAt(f0: number, sigma: number, t: number): number {
  return f0 * (1 + sigma * t);
}

/** Radiated energy proxy ∫ y² dt of y = A·e^(−dt)·sin(ωt) (ω ≫ d): A² / (4d). */
export function bubbleEnergy(amplitude: number, d: number): number {
  return (amplitude * amplitude) / (4 * Math.max(d, 1e-6));
}

/** Time (s) for a bubble to decay by `db` decibels. */
export function bubbleLifetime(d: number, db = 60): number {
  return (db / 20) * Math.LN10 / Math.max(d, 1e-6);
}

/** Equivalent spherical diameter (m) of a water parcel of volume V (m³). */
export function equivalentDiameter(volume: number): number {
  return Math.cbrt((6 * Math.max(volume, 0)) / Math.PI);
}

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** Smooth window: 1 inside [lo, hi], falling to 0 over `soft` (relative) outside, on a log axis. */
function logWindow(x: number, lo: number, hi: number, soft: number): number {
  if (x <= 0) return 0;
  const lx = Math.log(x);
  const a = smoothstep(Math.log(lo) - soft, Math.log(lo), lx);
  const b = 1 - smoothstep(Math.log(hi), Math.log(hi) + soft, lx);
  return a * b;
}

/**
 * Probability that a drop of diameter D (m) hitting water at speed v (m/s) entrains a ringing bubble.
 * Regular entrainment (Pumphrey & Elmore): narrow window around D ≈ 0.8–1.2 mm, v ≈ 2–4.5 m/s, high
 * probability. Large drops (crater collapse / irregular entrainment) entrain with moderate probability
 * once the impact is energetic enough.
 */
export function dropEntrainmentProbability(D: number, v: number): number {
  const reg = 0.75 * logWindow(D, 0.8e-3, 1.2e-3, 0.35) * logWindow(v, 2.0, 4.5, 0.25);
  const large = 0.3 * smoothstep(1.6e-3, 3.2e-3, D) * smoothstep(0.7, 1.8, v);
  return Math.min(0.95, Math.max(reg, large));
}

/** Range [lo, hi] of the entrained bubble radius (m) for a drop of diameter D (m). */
export function dropBubbleRadiusRange(D: number): [number, number] {
  return [0.15 * D, 0.36 * D];
}

/** Delay (s) between first contact and bubble pinch-off (crater collapse), grows with drop size. */
export function dropBubbleDelay(D: number): number {
  return 0.002 + 3.5 * D; // 2 ms + 3.5 ms per mm
}

/** Air-to-water volume ratio entrained by a plunging jet/sheet impacting at speed v (m/s). */
export function plungingAirRatio(v: number): number {
  const x = Math.min(3, Math.max(0, (v - 0.9) / 2.0));
  return 0.06 * Math.pow(x, 1.3);
}

/** Impact speed (m/s) after a free fall of height h (m) from rest. */
export function freeFallSpeed(h: number): number {
  return Math.sqrt(2 * GRAVITY * Math.max(0, h));
}

/**
 * Continuous piecewise power-law probability density on [breaks[0], breaks[n]] with
 * density ∝ r^exps[i] on [breaks[i], breaks[i+1]]. Analytic CDF, inverse CDF and moments.
 */
export class PiecewisePowerLaw {
  readonly breaks: number[];
  readonly exps: number[];
  private coef: number[] = [];
  private mass: number[] = [];
  private cum: number[] = [];
  readonly total: number;

  constructor(breaks: number[], exps: number[]) {
    if (breaks.length !== exps.length + 1) throw new Error('PiecewisePowerLaw: breaks/exps mismatch');
    this.breaks = breaks.slice();
    this.exps = exps.slice();
    let c = 1;
    let acc = 0;
    for (let i = 0; i < exps.length; i++) {
      if (i > 0) c = c * Math.pow(breaks[i], exps[i - 1] - exps[i]);
      this.coef.push(c);
      const m = PiecewisePowerLaw.segIntegral(c, exps[i], breaks[i], breaks[i + 1]);
      this.cum.push(acc);
      this.mass.push(m);
      acc += m;
    }
    this.total = acc;
  }

  get min() {
    return this.breaks[0];
  }
  get max() {
    return this.breaks[this.breaks.length - 1];
  }

  private static segIntegral(c: number, a: number, x: number, y: number): number {
    if (y <= x) return 0;
    if (Math.abs(a + 1) < 1e-9) return c * Math.log(y / x);
    return (c * (Math.pow(y, a + 1) - Math.pow(x, a + 1))) / (a + 1);
  }

  private seg(r: number): number {
    for (let i = 0; i < this.exps.length; i++) if (r <= this.breaks[i + 1]) return i;
    return this.exps.length - 1;
  }

  /** Normalised density. */
  pdf(r: number): number {
    if (r < this.min || r > this.max) return 0;
    const i = this.seg(r);
    return (this.coef[i] * Math.pow(r, this.exps[i])) / this.total;
  }

  /** Normalised CDF. */
  cdf(r: number): number {
    if (r <= this.min) return 0;
    if (r >= this.max) return 1;
    const i = this.seg(r);
    return (this.cum[i] + PiecewisePowerLaw.segIntegral(this.coef[i], this.exps[i], this.breaks[i], r)) / this.total;
  }

  /** Inverse CDF. */
  quantile(u: number): number {
    const target = Math.min(1, Math.max(0, u)) * this.total;
    let i = 0;
    while (i < this.exps.length - 1 && this.cum[i] + this.mass[i] < target) i++;
    const m = target - this.cum[i];
    const c = this.coef[i];
    const a = this.exps[i];
    const x = this.breaks[i];
    let r: number;
    if (Math.abs(a + 1) < 1e-9) r = x * Math.exp(m / c);
    else r = Math.pow(Math.pow(x, a + 1) + (m * (a + 1)) / c, 1 / (a + 1));
    return Math.min(this.max, Math.max(this.min, r));
  }

  /** Sample from the distribution truncated to [lo, hi] with a uniform variate u ∈ [0, 1). */
  sample(u: number, lo = this.min, hi = this.max): number {
    const a = this.cdf(Math.max(lo, this.min));
    const b = this.cdf(Math.min(hi, this.max));
    return this.quantile(a + (b - a) * u);
  }

  /** E[r^k] over the (normalised) distribution restricted to [lo, hi] (not renormalised: ∫_lo^hi r^k p(r) dr). */
  partialMoment(k: number, lo = this.min, hi = this.max): number {
    let s = 0;
    for (let i = 0; i < this.exps.length; i++) {
      const x = Math.max(lo, this.breaks[i]);
      const y = Math.min(hi, this.breaks[i + 1]);
      if (y > x) s += PiecewisePowerLaw.segIntegral(this.coef[i], this.exps[i] + k, x, y);
    }
    return s / this.total;
  }

  /** Mean of r^k over the whole distribution. */
  moment(k: number): number {
    return this.partialMoment(k);
  }
}

/** Plunging-jet bubble size spectrum (Garrett–Li–Farmer above the Hinze scale, Deane–Stokes below). */
export function plungingBubbleDistribution(rMin = 0.1e-3, rHinze = 1.0e-3, rMax = 6e-3): PiecewisePowerLaw {
  return new PiecewisePowerLaw([rMin, rHinze, rMax], [-1.5, -10 / 3]);
}

/** Mean bubble volume (m³) of a radius distribution. */
export function meanBubbleVolume(dist: PiecewisePowerLaw): number {
  return (4 / 3) * Math.PI * dist.moment(3);
}

/**
 * Tabulated cumulative integral C(r) = ∫_{rmin}^{r} p(r') g(r') dr' on a log grid, for an arbitrary
 * weight g (used for "energy below radius r" of a bubble population, which is not a pure power law
 * because of the damping law).
 */
export class CumulativeTable {
  readonly logR: Float64Array;
  readonly cum: Float64Array;
  constructor(dist: PiecewisePowerLaw, g: (r: number) => number, n = 96) {
    this.logR = new Float64Array(n);
    this.cum = new Float64Array(n);
    const l0 = Math.log(dist.min);
    const l1 = Math.log(dist.max);
    let prevR = dist.min;
    let prevV = dist.pdf(prevR) * g(prevR);
    this.logR[0] = l0;
    for (let i = 1; i < n; i++) {
      const lr = l0 + ((l1 - l0) * i) / (n - 1);
      const r = Math.exp(lr);
      // Sub-sample each interval (trapezoid on 8 sub-steps in log space) for accuracy.
      let acc = 0;
      let pr = prevR;
      let pv = prevV;
      for (let s = 1; s <= 8; s++) {
        const rr = Math.exp(Math.log(prevR) + ((lr - Math.log(prevR)) * s) / 8);
        const vv = dist.pdf(Math.min(rr, dist.max * (1 - 1e-12))) * g(rr);
        acc += 0.5 * (pv + vv) * (rr - pr);
        pr = rr;
        pv = vv;
      }
      this.logR[i] = lr;
      this.cum[i] = this.cum[i - 1] + acc;
      prevR = r;
      prevV = pv;
    }
  }
  /** C(r), linearly interpolated in log r. */
  at(r: number): number {
    const n = this.logR.length;
    const lr = Math.log(Math.max(r, 1e-12));
    if (lr <= this.logR[0]) return 0;
    if (lr >= this.logR[n - 1]) return this.cum[n - 1];
    const x = ((lr - this.logR[0]) / (this.logR[n - 1] - this.logR[0])) * (n - 1);
    const i = Math.floor(x);
    const t = x - i;
    return this.cum[i] * (1 - t) + this.cum[i + 1] * t;
  }
  get total() {
    return this.cum[this.cum.length - 1];
  }
}

/** Poisson-distributed integer with mean λ (inversion for small λ, normal approximation for large λ). */
export function poisson(lambda: number, rand: () => number): number {
  if (!(lambda > 0)) return 0;
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= rand();
    } while (p > L && k < 1000);
    return k - 1;
  }
  // Box–Muller normal approximation.
  const u1 = Math.max(1e-12, rand());
  const u2 = rand();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * z));
}

/** Small, fast, seedable PRNG (mulberry32). Returns a function producing floats in [0, 1). */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0 || 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
