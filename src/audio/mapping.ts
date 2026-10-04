/**
 * Simulation → sound mapping (main thread, pure TS, deterministic for a given seed).
 *
 * Inputs per frame: the particle impact list (SimEvents), the SWE tile statistics (SweStats), the frame
 * timing and the listener pose. Output: a batch of individual bubble resonances and noise bursts for the
 * worklet, plus per-emitter continuous-layer parameters (rumble / rush / hiss levels and spectra,
 * distance gain, air absorption, reverb send).
 *
 * Mapping summary (see docs/AUDIO.md for the full description and the assumptions made about the producers):
 *  - Impacts are binned on a 6 cm grid; the volume flux of each bin decides how much of it behaves as
 *    discrete drops (sparse) vs. a plunging jet / sheet (dense). Sparse IMPACT_POOL / IMPACT_DROP → impact
 *    transient + bubble with the drop-entrainment probability (Pumphrey & Elmore) and a size-dependent
 *    radius; sparse IMPACT_SOLID → splatter noise bursts; dense IMPACT_POOL → plunging-jet bubble cloud
 *    (air entrainment ratio β(v), Garrett–Li–Farmer size spectrum), low rumble and splash hiss.
 *  - SWE tiles → stream babble (Poisson bubbles, rate ∝ turbulence × area × speed, radii 1–8 mm shaped by
 *    depth and speed), hydraulic-jump gurgles, and a subtle broadband "rush".
 *  - Populations too dense to synthesise individually are split by bubble size: the largest (most
 *    distinct) bubbles are synthesised, the rest is rendered as band-limited noise of the same power
 *    (incoherent Poisson sum: power = λ·A²/4d), so loudness scales continuously with flow.
 *  - All sources are clustered into spatial emitters (cluster.ts).
 */
import { EmitterClusterer, type ClusterPoint } from './cluster';
import {
  CumulativeTable,
  bubbleDamping,
  dropBubbleDelay,
  dropBubbleRadiusRange,
  dropEntrainmentProbability,
  equivalentDiameter,
  makeRng,
  MAX_DROP_DIAMETER,
  meanBubbleVolume,
  minnaertFrequency,
  plungingAirRatio,
  plungingBubbleDistribution,
  poisson,
  smoothstep,
  WATER_DENSITY,
  type PiecewisePowerLaw,
} from './physics';
import { LAYOUT } from './protocol';
import { EVENT_FLOATS, NUM_TILES, TILE_FLOATS, type ParsedEvents, type ParsedStats } from './readback';

export const IMPACT_POOL = 0;
export const IMPACT_SOLID = 1;
export const IMPACT_DROP = 2;

export const BUBBLE_KIND = { drop: 0, secondary: 1, jet: 2, babble: 3, jump: 4 } as const;

/** Calibration constants (output units: 1.0 = full scale for a source at `refDistance`). */
export const TUNING = {
  // --- distance model
  refDistance: 0.5,
  minDistance: 0.07,
  maxGain: 2.5,
  airRefDistance: 0.6,
  airMinCutoff: 2500,
  reverbSend: 0.55,
  // --- bubble amplitude A = k·ε·(r / 1 mm)^alpha
  alpha: 1.2,
  kDrop: 0.15,
  kSecondary: 0.07,
  kJet: 0.02,
  kBabble: 0.015,
  kJump: 0.05,
  kTransient: 0.035,
  kSplat: 0.05,
  // --- populations
  /** Babble bubbles per second per (turbulence × wetted m² × m/s). */
  babbleRate: 15000,
  /** Jump gurgles per second per (jump intensity × m²). */
  jumpRate: 400,
  /** Rush rms per sqrt(m² · (m/s)³). */
  rushK: 0.05,
  /** Rumble rms per sqrt(W) of plunging kinetic power. */
  rumbleK: 0.015,
  /** Splash hiss rms per sqrt(W) of plunging kinetic power (× (v/3 m/s)). */
  plungeHissK: 0.012,
  // --- dense / sparse classification of impact bins (volume flux, m³/s per bin)
  binSize: 0.06,
  densePool: [0.02e-3, 0.15e-3],
  denseSolid: [0.02e-3, 0.15e-3],
  denseDrop: [0.005e-3, 0.06e-3],
  fluxTau: 0.15,
  // --- budgets (individual voices started per real second)
  bubbleBudget: 2400,
  burstBudget: 1400,
  discreteShare: 0.6,
  // --- spectra
  hissBrightness: 9000,
};

export interface ListenerPose {
  pos: readonly number[];
  fwd: readonly number[];
  right: readonly number[];
  up: readonly number[];
}

export interface MapperInput {
  /** Simulation seconds covered by this batch (0 when paused). */
  simDt: number;
  /** Real (wall-clock) seconds covered by this batch. */
  realDt: number;
  timeScale: number;
  paused: boolean;
  listener: ListenerPose;
  events: ParsedEvents | null;
  stats: ParsedStats | null;
  gains: { bubbles: number; roar: number; ambience: number };
}

export interface MapperDiagnostics {
  events: number;
  sparseEvents: number;
  denseVolume: number;
  bubbles: number[]; // per BUBBLE_KIND
  bursts: number;
  /** Mean-square power (per real second) of everything that was synthesised individually. */
  individualPower: number;
  /** Mean-square power rendered as noise in place of individual bubbles/bursts. */
  lumpedPower: number;
  rumblePower: number;
  midPower: number;
  hissPower: number;
  activeEmitters: number;
  jetRate: number;
  babbleRate: number;
}

const fin = (x: number, d = 0) => (Number.isFinite(x) ? x : d);
const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);

interface Bin {
  key: number;
  sx: number;
  sy: number;
  sz: number;
  sv: number;
  n: Float64Array; // per kind
  vol: Float64Array;
  volV: Float64Array;
  volV2: Float64Array;
  head: number;
  wDense: Float64Array;
  denseVol: Float64Array;
  denseVolV: Float64Array;
  denseVolV2: Float64Array;
  denseN: Float64Array;
  power: number;
}

interface FluxMem {
  q: Float64Array;
  seen: number;
}

function newBin(): Bin {
  return {
    key: 0,
    sx: 0,
    sy: 0,
    sz: 0,
    sv: 0,
    n: new Float64Array(3),
    vol: new Float64Array(3),
    volV: new Float64Array(3),
    volV2: new Float64Array(3),
    head: -1,
    wDense: new Float64Array(3),
    denseVol: new Float64Array(3),
    denseVolV: new Float64Array(3),
    denseVolV2: new Float64Array(3),
    denseN: new Float64Array(3),
    power: 0,
  };
}

/** Growable record list (Float32 records + priority / energy / frequency per record). */
class RecordList {
  data: Float32Array;
  pri: Float64Array;
  energy: Float64Array;
  freq: Float64Array;
  n = 0;
  constructor(readonly stride: number, cap = 256) {
    this.data = new Float32Array(cap * stride);
    this.pri = new Float64Array(cap);
    this.energy = new Float64Array(cap);
    this.freq = new Float64Array(cap);
  }
  clear() {
    this.n = 0;
  }
  add(): number {
    if (this.n >= this.pri.length) {
      const cap = this.pri.length * 2;
      const d = new Float32Array(cap * this.stride);
      d.set(this.data);
      this.data = d;
      const grow = (a: Float64Array) => {
        const b = new Float64Array(cap);
        b.set(a);
        return b;
      };
      this.pri = grow(this.pri);
      this.energy = grow(this.energy);
      this.freq = grow(this.freq);
    }
    return this.n++;
  }
}

/** Per emitter accumulators of the current batch. */
interface EmitterAcc {
  jetQ: number;
  jetQv: number;
  jetQv2: number;
  rumbleP: number;
  rumbleF: number; // Σ P·ln f
  midP: number;
  midF: number;
  hissP: number;
  hissF: number;
  tiles: number[];
  power: number;
  gain: number;
  dist: number;
}

export class AudioMapper {
  readonly clusterer: EmitterClusterer;
  readonly E: number;
  readonly rand: () => number;
  /** Output records for the worklet. */
  readonly bubbles: RecordList;
  readonly bursts: RecordList;
  readonly emitterParams: Float32Array;
  readonly globals: Float32Array;
  diag: MapperDiagnostics = AudioMapper.emptyDiag();
  /** Default represented volume of an impact whose volume is missing/invalid (m³). */
  defaultEventVolume = 2.16e-7;
  /** Output sample rate (Nyquist guard). */
  sampleRate = 48000;

  private jetDist: PiecewisePowerLaw;
  private jetMeanVol: number;
  private jetEnergyTable: CumulativeTable;
  private bins: Bin[] = [];
  private nBins = 0;
  private binIndex = new Map<number, number>();
  private flux = new Map<number, FluxMem>();
  private nextEvent = new Int32Array(0);
  private points: ClusterPoint[] = [];
  private assign = new Int32Array(0);
  private acc: EmitterAcc[] = [];
  private cand: RecordList;
  private candBursts: RecordList;
  private dropEventRate = 0;
  private rateScale = 1;
  private tilePower = new Float64Array(NUM_TILES);
  private tileLambda = new Float64Array(NUM_TILES);
  private tileEnergy = new Float64Array(NUM_TILES);
  private tileJump = new Float64Array(NUM_TILES);
  private tileRush = new Float64Array(NUM_TILES);
  private tileFreq = new Float64Array(NUM_TILES);
  private tileRange = new Float64Array(NUM_TILES * 2);
  private order: number[] = [];

  constructor(numEmitters: number, seed = 1) {
    this.E = numEmitters;
    this.clusterer = new EmitterClusterer(numEmitters);
    this.rand = makeRng(seed);
    this.bubbles = new RecordList(LAYOUT.BUBBLE_STRIDE);
    this.bursts = new RecordList(LAYOUT.BURST_STRIDE);
    this.cand = new RecordList(LAYOUT.BUBBLE_STRIDE);
    this.candBursts = new RecordList(LAYOUT.BURST_STRIDE);
    this.emitterParams = new Float32Array(numEmitters * LAYOUT.EMITTER_STRIDE);
    this.globals = new Float32Array(LAYOUT.GLOBAL_STRIDE);
    for (let e = 0; e < numEmitters; e++) {
      this.acc.push({ jetQ: 0, jetQv: 0, jetQv2: 0, rumbleP: 0, rumbleF: 0, midP: 0, midF: 0, hissP: 0, hissF: 0, tiles: [], power: 0, gain: 0, dist: 1 });
    }
    this.jetDist = plungingBubbleDistribution();
    this.jetMeanVol = meanBubbleVolume(this.jetDist);
    const a2 = 2 * TUNING.alpha;
    this.jetEnergyTable = new CumulativeTable(this.jetDist, (r) => Math.pow(r / 1e-3, a2) / (4 * bubbleDamping(minnaertFrequency(r))));
  }

  static emptyDiag(): MapperDiagnostics {
    return {
      events: 0,
      sparseEvents: 0,
      denseVolume: 0,
      bubbles: [0, 0, 0, 0, 0],
      bursts: 0,
      individualPower: 0,
      lumpedPower: 0,
      rumblePower: 0,
      midPower: 0,
      hissPower: 0,
      activeEmitters: 0,
      jetRate: 0,
      babbleRate: 0,
    };
  }

  /** Mean bubble volume of the plunging spectrum (m³). */
  get plungeMeanBubbleVolume() {
    return this.jetMeanVol;
  }

  reset() {
    this.flux.clear();
    this.clusterer.reset();
    this.dropEventRate = 0;
  }

  /** Pitch / time-stretch factor for slow motion (bubble frequencies and dampings × p). */
  static pitchFactor(timeScale: number): number {
    return Math.sqrt(clamp(fin(timeScale, 1), 0.05, 1));
  }

  /** Distance gain of a source at distance d with spatial extent ext. */
  static distanceGain(d: number, ext = 0): number {
    const T = TUNING;
    const eff = Math.sqrt(d * d + 0.25 * ext * ext + T.minDistance * T.minDistance);
    return clamp(T.refDistance / eff, 0, T.maxGain);
  }

  static airCutoff(d: number): number {
    return clamp(20000 * Math.sqrt(TUNING.airRefDistance / Math.max(d, 0.05)), TUNING.airMinCutoff, 20000);
  }

  /** Expected radiated energy of one sparse drop impact (transient + probabilistic bubble), unit pitch. */
  static dropEventEnergy(D: number, v: number, kind: number): number {
    const T = TUNING;
    const [lo, hi] = dropBubbleRadiusRange(D);
    const r = 0.5 * (lo + hi);
    const f = minnaertFrequency(r);
    const P = dropEntrainmentProbability(D, v);
    const eps2 = 1.0833 * clamp(v / 2.5, 0.25, 2.56);
    const k = kind === IMPACT_DROP ? T.kDrop * 0.6 : T.kDrop;
    const A = k * Math.pow(r / 1e-3, T.alpha);
    const eb = (P * A * A * eps2) / (4 * bubbleDamping(f));
    const at = T.kTransient * Math.pow(D / 5e-3, 0.8) * Math.pow(v / 2.5, 1.5);
    const decay = clamp(1200 * Math.sqrt(5e-3 / D), 500, 4000);
    return eb + (at * at * 0.85) / (2 * decay);
  }

  static splatEnergy(D: number, v: number): number {
    const T = TUNING;
    const as = T.kSplat * Math.pow(D / 5e-3, 0.8) * Math.pow(v / 2.5, 1.5);
    return (as * as * 0.85 * 0.58) / (2 * 1000);
  }

  private sampleUniform(a: number, b: number) {
    return a + (b - a) * this.rand();
  }

  private resetBatch() {
    this.bubbles.clear();
    this.bursts.clear();
    this.cand.clear();
    this.candBursts.clear();
    this.diag = AudioMapper.emptyDiag();
    for (const a of this.acc) {
      a.jetQ = a.jetQv = a.jetQv2 = 0;
      a.rumbleP = a.rumbleF = a.midP = a.midF = a.hissP = a.hissF = 0;
      a.tiles.length = 0;
      a.power = 0;
    }
  }

  private addNoise(e: number, band: 'rumble' | 'mid' | 'hiss', P: number, f: number) {
    if (!(P > 0) || !Number.isFinite(P) || e < 0) return;
    const a = this.acc[e];
    const lf = Math.log(clamp(fin(f, 1000), 20, 22000));
    if (band === 'rumble') {
      a.rumbleP += P;
      a.rumbleF += P * lf;
    } else if (band === 'mid') {
      a.midP += P;
      a.midF += P * lf;
    } else {
      a.hissP += P;
      a.hissF += P * lf;
    }
  }

  /** Processes one frame/batch; results in bubbles / bursts / emitterParams / globals / diag. */
  process(inp: MapperInput): void {
    this.resetBatch();
    const T = TUNING;
    const simDt = clamp(fin(inp.simDt), 0, 0.25);
    const realDt = clamp(fin(inp.realDt, 1 / 60), 1e-4, 0.25);
    const live = simDt > 0 && !inp.paused;
    const p = AudioMapper.pitchFactor(inp.timeScale);
    const nyqF = 0.42 * this.sampleRate;
    // Sim-to-real rate (slow motion, slow frames): converts per-sim-second powers to per-real-second.
    if (live) this.rateScale += (clamp(simDt / realDt, 0, 1.5) - this.rateScale) * 0.2;
    const powerScale = this.rateScale / p;
    const L = inp.listener.pos;

    // ---- 1. Impact events → spatial bins (+ smoothed per-bin volume flux).
    this.nBins = 0;
    this.binIndex.clear();
    const ev = live ? inp.events : null;
    if (ev && ev.n > 0) this.binEvents(ev, simDt);
    this.updateFlux(simDt);

    // ---- 2. SWE tiles.
    const st = live ? inp.stats : null;
    this.lastTiles = st ? st.tiles : null;
    let nTiles = 0;
    if (st) nTiles = this.analyseTiles(st);

    // ---- 3. Clustering (bins + tiles) with perceptual weights.
    const nPts = this.nBins + nTiles;
    while (this.points.length < nPts) this.points.push({ x: 0, y: 0, z: 0, w: 0, ext: 0 });
    if (this.assign.length < nPts) this.assign = new Int32Array(Math.max(nPts, 64) * 2);
    for (let b = 0; b < this.nBins; b++) {
      const bin = this.bins[b];
      const pt = this.points[b];
      pt.x = bin.sx / bin.sv;
      pt.y = bin.sy / bin.sv;
      pt.z = bin.sz / bin.sv;
      pt.ext = 0.5 * T.binSize;
      const d = Math.hypot(pt.x - L[0], pt.y - L[1], pt.z - L[2]);
      const g = AudioMapper.distanceGain(d, pt.ext);
      pt.w = bin.power * g * g + 1e-12;
    }
    const tileIdx: number[] = this.order;
    tileIdx.length = 0;
    if (st) {
      for (let t = 0; t < NUM_TILES; t++) if (this.tilePower[t] > 0) tileIdx.push(t);
      tileIdx.forEach((t, k) => {
        const o = t * TILE_FLOATS;
        const pt = this.points[this.nBins + k];
        pt.x = st.tiles[o];
        pt.y = st.tiles[o + 1];
        pt.z = st.tiles[o + 2];
        pt.ext = 0.5 * Math.sqrt(clamp(st.tiles[o + 3], 0, 10));
        const d = Math.hypot(pt.x - L[0], pt.y - L[1], pt.z - L[2]);
        const g = AudioMapper.distanceGain(d, pt.ext);
        pt.w = this.tilePower[t] * g * g + 1e-12;
      });
    }
    this.clusterer.update(this.points, nPts, L, realDt, this.assign);
    const em = this.clusterer.emitters;
    for (let e = 0; e < this.E; e++) {
      const a = this.acc[e];
      const s = em[e];
      a.dist = Math.hypot(s.x - L[0], s.y - L[1], s.z - L[2]);
      a.gain = s.active ? AudioMapper.distanceGain(a.dist, s.ext) : 0;
    }

    // ---- 4. Discrete events → candidate bubbles / bursts; dense parts → emitter accumulators.
    if (ev && ev.n > 0) this.discreteEvents(ev, simDt, realDt, p, nyqF);
    // Dense / unrecorded volumes.
    for (let b = 0; b < this.nBins; b++) {
      const bin = this.bins[b];
      const e = this.assign[b];
      if (e < 0) continue;
      const a = this.acc[e];
      const dv = bin.denseVol[IMPACT_POOL];
      if (dv > 0) {
        a.jetQ += dv / simDt;
        a.jetQv += bin.denseVolV[IMPACT_POOL] / simDt;
        a.jetQv2 += bin.denseVolV2[IMPACT_POOL] / simDt;
      }
      const ns = bin.denseN[IMPACT_SOLID];
      if (ns > 0) {
        const vs = bin.denseVolV[IMPACT_SOLID] / Math.max(bin.denseVol[IMPACT_SOLID], 1e-15);
        const Ds = equivalentDiameter(bin.denseVol[IMPACT_SOLID] / ns);
        this.addNoise(e, 'hiss', (ns / simDt) * AudioMapper.splatEnergy(Ds, vs) * powerScale, 2600 * p);
        this.diag.lumpedPower += (ns / simDt) * AudioMapper.splatEnergy(Ds, vs) * powerScale;
      }
      const nd = bin.denseN[IMPACT_DROP];
      if (nd > 0) {
        const vd = bin.denseVolV[IMPACT_DROP] / Math.max(bin.denseVol[IMPACT_DROP], 1e-15);
        const Dd = clamp(equivalentDiameter(bin.denseVol[IMPACT_DROP] / nd), 0.3e-3, 4e-3);
        const P = (nd / simDt) * AudioMapper.dropEventEnergy(Dd, vd, IMPACT_DROP) * powerScale;
        this.addNoise(e, 'hiss', P, Math.min(9000, minnaertFrequency(0.25 * Dd)) * p);
        this.diag.lumpedPower += P;
      }
      this.diag.denseVolume += bin.denseVol[0] + bin.denseVol[1] + bin.denseVol[2];
    }

    // ---- 5. Budget: discrete candidates first (they carry the near-field detail), by perceived amplitude.
    const capBub = Math.max(2, Math.round(T.bubbleBudget * realDt));
    const capBurst = Math.max(2, Math.round(T.burstBudget * realDt));
    this.selectCandidates(this.cand, this.bubbles, Math.round(capBub * T.discreteShare), simDt, 'bubble');
    this.selectCandidates(this.candBursts, this.bursts, capBurst, simDt, 'burst');

    // ---- 6. Continuous populations per emitter: plunging jets, babble, jumps, rush, rumble.
    if (st) {
      tileIdx.forEach((t, k) => {
        const e = this.assign[this.nBins + k];
        if (e >= 0) this.acc[e].tiles.push(t);
      });
    }
    const remaining = Math.max(0, capBub - this.bubbles.n);
    let qSum = 0;
    for (let e = 0; e < this.E; e++) {
      const a = this.acc[e];
      let P = 0;
      if (a.jetQ > 0) {
        const v = a.jetQv / a.jetQ;
        P += ((plungingAirRatio(v) * a.jetQ) / this.jetMeanVol) * T.kJet * T.kJet * 1.0833 * this.jetEnergyTable.total;
      }
      for (const t of a.tiles) P += this.tileLambda[t] * this.tileEnergy[t];
      a.power = Math.sqrt(Math.max(0, P)) * Math.max(a.gain, 0.05);
      qSum += a.power;
    }
    for (let e = 0; e < this.E; e++) {
      const a = this.acc[e];
      if (!em[e].active && a.jetQ <= 0 && a.tiles.length === 0) continue;
      const allowance = qSum > 0 ? (remaining * a.power) / qSum : 0;
      this.emitterPopulations(e, a, allowance, simDt, realDt, p, powerScale, nyqF);
    }

    // ---- 7. Emitter parameters.
    const S = LAYOUT.EMITTER_STRIDE;
    const ep = this.emitterParams;
    let active = 0;
    for (let e = 0; e < this.E; e++) {
      const a = this.acc[e];
      const s = em[e];
      const o = e * S;
      const on = s.active || a.rumbleP + a.midP + a.hissP > 0;
      if (on) active++;
      const lvl = (P: number) => (on && live ? Math.sqrt(Math.max(0, P)) : 0);
      ep[o + LAYOUT.E_RUMBLE] = lvl(a.rumbleP);
      ep[o + LAYOUT.E_RUMBLE_F] = a.rumbleP > 0 ? Math.exp(a.rumbleF / a.rumbleP) : 120 * p;
      ep[o + LAYOUT.E_MID] = lvl(a.midP);
      ep[o + LAYOUT.E_MID_F] = a.midP > 0 ? Math.exp(a.midF / a.midP) : 700 * p;
      ep[o + LAYOUT.E_HISS] = lvl(a.hissP);
      ep[o + LAYOUT.E_HISS_F] = a.hissP > 0 ? Math.exp(a.hissF / a.hissP) : 3000 * p;
      ep[o + LAYOUT.E_BRIGHT] = T.hissBrightness * p;
      const tot = a.rumbleP + a.midP + a.hissP;
      ep[o + LAYOUT.E_MOD] = 0.1 + 0.18 * (tot > 0 ? a.rumbleP / tot : 0);
      ep[o + LAYOUT.E_GAIN] = s.active ? a.gain : 0;
      ep[o + LAYOUT.E_AIR] = AudioMapper.airCutoff(a.dist);
      ep[o + LAYOUT.E_REV] = s.active ? T.reverbSend * Math.sqrt(Math.min(1, a.gain)) : 0;
      ep[o + LAYOUT.E_ACTIVE] = s.active ? 1 : 0;
      this.diag.rumblePower += a.rumbleP;
      this.diag.midPower += a.midP;
      this.diag.hissPower += a.hissP;
    }
    this.diag.activeEmitters = active;

    // ---- 8. Globals.
    const g = this.globals;
    g[LAYOUT.G_BUBBLES] = clamp(fin(inp.gains.bubbles, 1), 0, 4);
    g[LAYOUT.G_ROAR] = clamp(fin(inp.gains.roar, 1), 0, 4);
    g[LAYOUT.G_AMBIENCE] = clamp(fin(inp.gains.ambience, 0.35), 0, 2);
    g[LAYOUT.G_WATER_FADE] = inp.paused ? 0 : 1;
    g[LAYOUT.G_AMB_FADE] = inp.paused ? 0 : 1;
    g[LAYOUT.G_WIND] = 0.5;
    g[LAYOUT.G_BIRDS] = 0.12;
    g[LAYOUT.G_OUT_GAIN] = 1;
  }

  // ------------------------------------------------------------------------------------------------

  private binEvents(ev: ParsedEvents, simDt: number) {
    const T = TUNING;
    const n = ev.n;
    if (this.nextEvent.length < n) this.nextEvent = new Int32Array(Math.max(n, 256));
    const d = ev.data;
    const inv = 1 / T.binSize;
    let readVol = 0;
    let readPoolSolid = 0;
    for (let i = 0; i < n; i++) {
      const o = i * EVENT_FLOATS;
      const x = d[o];
      const y = d[o + 1];
      const z = d[o + 2];
      const kind = Math.round(d[o + 3]);
      if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) || kind < 0 || kind > 2) {
        this.nextEvent[i] = -2;
        continue;
      }
      let vol = d[o + 7];
      if (!(vol > 0) || !Number.isFinite(vol)) vol = this.defaultEventVolume;
      vol = Math.min(vol, 1e-3);
      const sp = Math.min(Math.hypot(fin(d[o + 4]), fin(d[o + 5]), fin(d[o + 6])), 30);
      const key = (Math.floor(x * inv) + 2048) * 16777216 + (Math.floor(y * inv) + 2048) * 4096 + (Math.floor(z * inv) + 2048);
      let bi = this.binIndex.get(key);
      if (bi === undefined) {
        bi = this.nBins++;
        if (bi >= this.bins.length) this.bins.push(newBin());
        const b = this.bins[bi];
        b.key = key;
        b.sx = b.sy = b.sz = b.sv = 0;
        b.n.fill(0);
        b.vol.fill(0);
        b.volV.fill(0);
        b.volV2.fill(0);
        b.denseVol.fill(0);
        b.denseVolV.fill(0);
        b.denseVolV2.fill(0);
        b.denseN.fill(0);
        b.head = -1;
        b.power = 0;
        this.binIndex.set(key, bi);
      }
      const b = this.bins[bi];
      b.sx += vol * x;
      b.sy += vol * y;
      b.sz += vol * z;
      b.sv += vol;
      b.n[kind]++;
      b.vol[kind] += vol;
      b.volV[kind] += vol * sp;
      b.volV2[kind] += vol * sp * sp;
      this.nextEvent[i] = b.head;
      b.head = i;
      readVol += vol;
      if (kind !== IMPACT_DROP) readPoolSolid += vol;
    }
    this.diag.events = n;
    // Unread / unrecorded impacts: spread their volume over the bins as dense flow.
    const meanVol = n > 0 ? readVol / n : this.defaultEventVolume;
    const missingVol = (ev.droppedVolume > 0 ? ev.droppedVolume : ev.notRecorded * meanVol) + ev.unread * meanVol;
    if (missingVol > 0 && readPoolSolid > 0) {
      const s = missingVol / readPoolSolid;
      for (let bi = 0; bi < this.nBins; bi++) {
        const b = this.bins[bi];
        for (const k of [IMPACT_POOL, IMPACT_SOLID]) {
          if (b.vol[k] <= 0) continue;
          const add = b.vol[k] * s;
          const vm = b.volV[k] / b.vol[k];
          const v2 = b.volV2[k] / b.vol[k];
          b.denseVol[k] += add;
          b.denseVolV[k] += add * vm;
          b.denseVolV2[k] += add * v2;
          b.denseN[k] += (b.n[k] * add) / b.vol[k];
        }
      }
    }
    // Bin power estimates (for clustering priority).
    for (let bi = 0; bi < this.nBins; bi++) {
      const b = this.bins[bi];
      let P = 0;
      for (let k = 0; k < 3; k++) {
        if (b.n[k] <= 0) continue;
        const v = b.volV[k] / b.vol[k];
        const D = Math.min(equivalentDiameter(b.vol[k] / b.n[k]), MAX_DROP_DIAMETER);
        if (k === IMPACT_SOLID) P += (b.n[k] / simDt) * AudioMapper.splatEnergy(D, v);
        else P += (b.n[k] / simDt) * AudioMapper.dropEventEnergy(D, v, k);
        if (k === IMPACT_POOL) P += TUNING.rumbleK ** 2 * 0.5 * WATER_DENSITY * (b.volV2[k] / simDt);
      }
      b.power = P;
    }
  }

  private updateFlux(simDt: number) {
    const T = TUNING;
    if (!(simDt > 0)) return;
    const k = 1 - Math.exp(-simDt / T.fluxTau);
    for (let bi = 0; bi < this.nBins; bi++) {
      const b = this.bins[bi];
      let m = this.flux.get(b.key);
      if (!m) {
        m = { q: new Float64Array(3), seen: 0 };
        // Start from the instantaneous flux so a new sheet is classified as dense immediately.
        for (let j = 0; j < 3; j++) m.q[j] = b.vol[j] / simDt;
        this.flux.set(b.key, m);
      }
      m.seen = -1; // updated this frame
      for (let j = 0; j < 3; j++) m.q[j] += (b.vol[j] / simDt - m.q[j]) * k;
      b.wDense[IMPACT_POOL] = smoothstep(T.densePool[0], T.densePool[1], m.q[IMPACT_POOL]);
      b.wDense[IMPACT_SOLID] = smoothstep(T.denseSolid[0], T.denseSolid[1], m.q[IMPACT_SOLID]);
      b.wDense[IMPACT_DROP] = smoothstep(T.denseDrop[0], T.denseDrop[1], m.q[IMPACT_DROP]);
    }
    for (const [key, m] of this.flux) {
      if (m.seen < 0) {
        m.seen = 0;
        continue;
      }
      m.seen += simDt;
      for (let j = 0; j < 3; j++) m.q[j] *= 1 - k;
      if (m.seen > 1.5) this.flux.delete(key);
    }
  }

  private analyseTiles(st: ParsedStats): number {
    const T = TUNING;
    const tl = st.tiles;
    let count = 0;
    const a2 = 2 * T.alpha;
    for (let t = 0; t < NUM_TILES; t++) {
      const o = t * TILE_FLOATS;
      const area = clamp(fin(tl[o + 3]), 0, 10);
      const u = clamp(fin(tl[o + 4]), 0, 6);
      const turb = clamp(fin(tl[o + 5]), 0, 3);
      const jump = clamp(fin(tl[o + 6]), 0, 3);
      const h = clamp(fin(tl[o + 7]), 0, 2);
      this.tilePower[t] = 0;
      this.tileLambda[t] = 0;
      this.tileJump[t] = 0;
      this.tileRush[t] = 0;
      if (!(area > 0) || !(Number.isFinite(tl[o]) && Number.isFinite(tl[o + 1]) && Number.isFinite(tl[o + 2]))) continue;
      const lam = T.babbleRate * turb * area * u;
      const rlo = Math.max(0.5e-3, 0.9e-3 / (1 + 0.4 * u));
      const rhi = Math.max(1.5 * rlo, clamp(0.4 * h, 1.5e-3, 8e-3));
      this.tileRange[t * 2] = rlo;
      this.tileRange[t * 2 + 1] = rhi;
      // Mean bubble energy over a r^-2.2 spectrum (8-point quadrature in log r).
      let wSum = 0;
      let eSum = 0;
      let fSum = 0;
      for (let q = 0; q < 8; q++) {
        const r = rlo * Math.pow(rhi / rlo, (q + 0.5) / 8);
        const w = Math.pow(r, -1.2); // r^-2.2 · r (log-spaced measure)
        const f = minnaertFrequency(r);
        const eps = 0.5 + 0.5 * Math.min(1, u / 1.2);
        const en = (T.kBabble * T.kBabble * eps * eps * 1.0833 * Math.pow(r / 1e-3, a2)) / (4 * bubbleDamping(f));
        wSum += w;
        eSum += w * en;
        fSum += w * en * Math.log(f);
      }
      this.tileEnergy[t] = eSum / wSum;
      this.tileFreq[t] = eSum > 0 ? Math.exp(fSum / eSum) : 1200;
      this.tileLambda[t] = lam;
      this.tileJump[t] = T.jumpRate * jump * area;
      this.tileRush[t] = T.rushK * T.rushK * area * u * u * u * (0.3 + turb);
      const jumpE = (T.kJump * T.kJump * 1.1 * Math.pow(6, a2)) / (4 * bubbleDamping(minnaertFrequency(6e-3)));
      this.tilePower[t] = lam * this.tileEnergy[t] + this.tileJump[t] * jumpE + this.tileRush[t];
      if (this.tilePower[t] > 0) count++;
    }
    return count;
  }

  private discreteEvents(ev: ParsedEvents, simDt: number, realDt: number, p: number, nyqF: number) {
    const T = TUNING;
    const d = ev.data;
    const synthSecondaries = this.dropEventRate < 1;
    let dropN = 0;
    for (let bi = 0; bi < this.nBins; bi++) {
      const bin = this.bins[bi];
      const e = this.assign[bi];
      const g = e >= 0 ? this.acc[e].gain : 0;
      for (let i = bin.head; i >= 0; i = this.nextEvent[i]) {
        const o = i * EVENT_FLOATS;
        const kind = Math.round(d[o + 3]);
        let vol = d[o + 7];
        if (!(vol > 0) || !Number.isFinite(vol)) vol = this.defaultEventVolume;
        vol = Math.min(vol, 1e-3);
        const v = Math.min(Math.hypot(fin(d[o + 4]), fin(d[o + 5]), fin(d[o + 6])), 30);
        if (kind === IMPACT_DROP) dropN++;
        if (this.rand() < bin.wDense[kind]) {
          bin.denseVol[kind] += vol;
          bin.denseVolV[kind] += vol * v;
          bin.denseVolV2[kind] += vol * v * v;
          bin.denseN[kind]++;
          continue;
        }
        if (e < 0) continue;
        this.diag.sparseEvents++;
        const t0 = this.rand() * realDt;
        if (kind === IMPACT_SOLID) this.splat(e, g, vol, v, t0, p);
        else this.drop(e, g, kind, vol, v, t0, p, nyqF, synthSecondaries);
      }
    }
    this.dropEventRate += (dropN / Math.max(simDt, 1e-4) - this.dropEventRate) * 0.1;
  }

  private addBubble(list: RecordList, t: number, e: number, f0: number, d: number, sigma: number, A: number, kind: number, pri: number) {
    const i = list.add();
    const o = i * LAYOUT.BUBBLE_STRIDE;
    const r = list.data;
    r[o + LAYOUT.B_T] = t;
    r[o + LAYOUT.B_EM] = e;
    r[o + LAYOUT.B_F0] = f0;
    r[o + LAYOUT.B_D] = d;
    r[o + LAYOUT.B_CHIRP] = sigma;
    r[o + LAYOUT.B_AMP] = A;
    r[o + LAYOUT.B_KIND] = kind;
    r[o + 7] = 0;
    list.pri[i] = pri;
    list.energy[i] = (A * A) / (4 * Math.max(d, 1e-3));
    list.freq[i] = f0;
  }

  private addBurst(list: RecordList, t: number, e: number, A: number, decay: number, fc: number, q: number, attack: number, kind: number, pri: number) {
    const i = list.add();
    const o = i * LAYOUT.BURST_STRIDE;
    const r = list.data;
    r[o + LAYOUT.U_T] = t;
    r[o + LAYOUT.U_EM] = e;
    r[o + LAYOUT.U_AMP] = A;
    r[o + LAYOUT.U_DECAY] = decay;
    r[o + LAYOUT.U_FC] = fc;
    r[o + LAYOUT.U_Q] = q;
    r[o + LAYOUT.U_ATTACK] = attack;
    r[o + LAYOUT.U_KIND] = kind;
    list.pri[i] = pri;
    list.energy[i] = (A * A * 0.85) / (2 * Math.max(decay, 1));
    list.freq[i] = fc;
  }

  /** One sparse drop / small parcel entering water: impact transient + (maybe) a ringing bubble + secondaries. */
  private drop(e: number, g: number, kind: number, vol: number, v: number, t0: number, p: number, nyqF: number, secondaries: boolean) {
    const T = TUNING;
    const D = clamp(equivalentDiameter(vol), 0.3e-3, kind === IMPACT_DROP ? 4e-3 : MAX_DROP_DIAMETER);
    const vv = Math.max(v, 0.05);
    // Impact transient (initial contact): short broadband click, brighter for small drops.
    const at = T.kTransient * Math.pow(D / 5e-3, 0.8) * Math.pow(vv / 2.5, 1.5) * this.sampleUniform(0.6, 1.3) * (kind === IMPACT_DROP ? 0.6 : 1);
    const sq = Math.sqrt(5e-3 / D);
    if (at > 1e-5) {
      this.addBurst(this.candBursts, t0, e, at, clamp(1200 * sq, 500, 4000) * p, clamp(4500 * sq, 1200, 14000) * p, 0.7, 0.0002 / p, 0, at * g);
    }
    // Entrained bubble (crater collapse / regular entrainment).
    const P = dropEntrainmentProbability(D, vv);
    if (this.rand() < P) {
      const [lo, hi] = dropBubbleRadiusRange(D);
      const r = lo * Math.pow(hi / lo, this.rand());
      const f0 = minnaertFrequency(r);
      if (f0 * p < nyqF) {
        const d = bubbleDamping(f0);
        const eps = this.sampleUniform(0.6, 1.4) * clamp(Math.sqrt(vv / 2.5), 0.5, 1.6);
        const A = (kind === IMPACT_DROP ? T.kDrop * 0.6 : T.kDrop) * eps * Math.pow(r / 1e-3, T.alpha);
        const xi = this.sampleUniform(0.08, 0.2);
        this.addBubble(this.cand, t0 + dropBubbleDelay(D) / p, e, f0 * p, d * p, xi * d * p, A, BUBBLE_KIND.drop, A * g);
      }
    }
    // Secondary droplets from the splash crown / Worthington jet (only if the particle sim does not emit them).
    if (secondaries && kind === IMPACT_POOL) {
      const n2 = poisson(0.4 * smoothstep(1.5, 4, vv) * Math.min(1, D / 3e-3), this.rand);
      for (let k = 0; k < n2; k++) {
        const t2 = t0 + this.sampleUniform(0.04, 0.16) / p;
        const D2 = this.sampleUniform(0.6e-3, 1.6e-3);
        const v2 = this.sampleUniform(1.4, 2.8);
        const a2 = T.kTransient * 0.5 * Math.pow(D2 / 5e-3, 0.8) * Math.pow(v2 / 2.5, 1.5);
        const sq2 = Math.sqrt(5e-3 / D2);
        this.addBurst(this.candBursts, t2, e, a2, clamp(1200 * sq2, 500, 4000) * p, clamp(4500 * sq2, 1200, 14000) * p, 0.7, 0.0002 / p, 1, a2 * g);
        if (this.rand() < dropEntrainmentProbability(D2, v2)) {
          const [lo, hi] = dropBubbleRadiusRange(D2);
          const r = lo * Math.pow(hi / lo, this.rand());
          const f0 = minnaertFrequency(r);
          if (f0 * p < nyqF) {
            const d = bubbleDamping(f0);
            const A = T.kSecondary * this.sampleUniform(0.5, 1.3) * Math.pow(r / 1e-3, T.alpha);
            this.addBubble(this.cand, t2 + dropBubbleDelay(D2) / p, e, f0 * p, d * p, this.sampleUniform(0.08, 0.2) * d * p, A, BUBBLE_KIND.secondary, A * g);
          }
        }
      }
    }
  }

  /** Water striking rock: a few broadband splatter clicks, louder and brighter at higher speed. */
  private splat(e: number, g: number, vol: number, v: number, t0: number, p: number) {
    const T = TUNING;
    const D = clamp(equivalentDiameter(vol), 0.3e-3, 12e-3);
    const as = T.kSplat * Math.pow(D / 5e-3, 0.8) * Math.pow(Math.max(v, 0.05) / 2.5, 1.5) * this.sampleUniform(0.6, 1.3);
    if (as < 1e-5) return;
    const nSub = 1 + (this.rand() < 0.4 ? 1 : 0) + (v > 3 && this.rand() < 0.4 ? 1 : 0);
    const bright = clamp(Math.pow(v / 2.5, 0.3), 0.7, 1.4);
    for (let k = 0; k < nSub; k++) {
      const fc = Math.exp(Math.log(1500) + (Math.log(7000) - Math.log(1500)) * this.rand()) * bright * p;
      const amp = (as * this.sampleUniform(0.5, 1)) / Math.sqrt(nSub);
      this.addBurst(this.candBursts, t0 + this.sampleUniform(0, 0.005) / p, e, amp, this.sampleUniform(500, 1500) * p, fc, this.sampleUniform(0.6, 1.2), 0.0001 / p, 2, amp * g);
    }
  }

  /** Keeps the `cap` highest-priority candidates; the rest is rendered as noise of equal power. */
  private selectCandidates(cand: RecordList, out: RecordList, cap: number, simDt: number, kind: 'bubble' | 'burst') {
    // Candidate energies already include the slow-motion time stretch (their damping is × pitch factor).
    const powerScale = this.rateScale;
    const n = cand.n;
    if (n === 0) return;
    const stride = cand.stride;
    let keep: number[];
    if (n <= cap) keep = Array.from({ length: n }, (_, i) => i);
    else {
      const idx = Array.from({ length: n }, (_, i) => i);
      idx.sort((a, b) => cand.pri[b] - cand.pri[a]);
      keep = idx.slice(0, cap);
      for (let k = cap; k < n; k++) {
        const i = idx[k];
        const e = cand.data[i * stride + 1] | 0;
        const P = (cand.energy[i] / Math.max(simDt, 1e-4)) * powerScale;
        const f = cand.freq[i];
        this.addNoise(e, kind === 'burst' || f > 2500 ? 'hiss' : 'mid', P, kind === 'burst' ? Math.min(f, 6000) : f);
        this.diag.lumpedPower += P;
      }
      keep.sort((a, b) => a - b);
    }
    for (const i of keep) {
      const j = out.add();
      out.data.set(cand.data.subarray(i * stride, i * stride + stride), j * stride);
      out.energy[j] = cand.energy[i];
      this.diag.individualPower += (cand.energy[i] / Math.max(simDt, 1e-4)) * powerScale;
      if (kind === 'bubble') this.diag.bubbles[cand.data[i * stride + LAYOUT.B_KIND] | 0]++;
      else this.diag.bursts++;
    }
  }

  private emitterPopulations(e: number, a: EmitterAcc, allowance: number, simDt: number, realDt: number, p: number, powerScale: number, nyqF: number) {
    const T = TUNING;
    const em = this.clusterer.emitters[e];
    // Plunging jet / sheet.
    let pJet = 0;
    let pBab = 0;
    for (const t of a.tiles) pBab += this.tileLambda[t] * this.tileEnergy[t];
    let jetLambda = 0;
    let vJet = 0;
    if (a.jetQ > 0) {
      vJet = a.jetQv / a.jetQ;
      jetLambda = (plungingAirRatio(vJet) * a.jetQ) / this.jetMeanVol;
      pJet = jetLambda * T.kJet * T.kJet * 1.0833 * this.jetEnergyTable.total;
    }
    const allowJet = pJet + pBab > 0 ? (allowance * pJet) / (pJet + pBab) : 0;
    const allowBab = Math.max(0, allowance - allowJet);
    if (a.jetQ > 0) {
      const v2 = a.jetQv2 / a.jetQ;
      const kinetic = 0.5 * WATER_DENSITY * a.jetQ * v2; // W
      this.diag.jetRate += jetLambda;
      if (jetLambda > 0) {
        const dist = this.jetDist;
        const nExp = jetLambda * simDt;
        const rNyq = minnaertFrequency(nyqF / p);
        let rs = Math.max(dist.min, rNyq);
        if (nExp * (1 - dist.cdf(rs)) > allowJet) rs = dist.quantile(1 - allowJet / Math.max(nExp, 1e-9));
        const nInd = poisson(nExp * (1 - dist.cdf(rs)), this.rand);
        for (let k = 0; k < nInd; k++) {
          const r = dist.sample(this.rand(), rs, dist.max);
          const f0 = minnaertFrequency(r);
          const d = bubbleDamping(f0);
          const A = T.kJet * this.sampleUniform(0.5, 1.5) * Math.pow(r / 1e-3, T.alpha);
          const xi = this.sampleUniform(0.03, 0.12);
          const i = this.bubbles.add();
          const o = i * LAYOUT.BUBBLE_STRIDE;
          const R = this.bubbles.data;
          R[o + LAYOUT.B_T] = this.rand() * realDt;
          R[o + LAYOUT.B_EM] = e;
          R[o + LAYOUT.B_F0] = f0 * p;
          R[o + LAYOUT.B_D] = d * p;
          R[o + LAYOUT.B_CHIRP] = xi * d * p;
          R[o + LAYOUT.B_AMP] = A;
          R[o + LAYOUT.B_KIND] = BUBBLE_KIND.jet;
          R[o + 7] = 0;
          this.diag.bubbles[BUBBLE_KIND.jet]++;
          this.diag.individualPower += ((A * A) / (4 * d * p) / Math.max(simDt, 1e-4)) * this.rateScale;
        }
        // Lumped part: bubbles smaller than rs (above Nyquist excluded). Its power spectral density peaks
        // just above f(rs) and falls steeply (GLF population × damping), hence the band-pass layer.
        const Ecum = this.jetEnergyTable.at(rs) - this.jetEnergyTable.at(rNyq);
        const Pl = jetLambda * T.kJet * T.kJet * 1.0833 * Math.max(0, Ecum) * powerScale;
        this.addNoise(e, 'mid', Pl, clamp(1.3 * minnaertFrequency(rs) * p, 400, 9000));
        this.diag.lumpedPower += Pl;
      }
      // Low rumble (bubble-cloud collective oscillation + impact pressure) and splash hiss.
      const L = Math.max(em.ext, Math.sqrt(a.jetQ / Math.max(vJet, 0.3)), 0.06);
      const rumble = T.rumbleK * T.rumbleK * kinetic * smoothstep(0.8, 2.0, vJet) * powerScale;
      this.addNoise(e, 'rumble', rumble, clamp(30 / L, 55, 240) * p);
      const hiss = T.plungeHissK * T.plungeHissK * kinetic * clamp(vJet / 3, 0, 2) * powerScale;
      this.addNoise(e, 'hiss', hiss, 2200 * p);
    }
    // Stream babble + jumps + rush from SWE tiles.
    if (a.tiles.length) {
      let nExp = 0;
      for (const t of a.tiles) nExp += this.tileLambda[t] * simDt;
      const phi = nExp > allowBab ? allowBab / nExp : 1;
      let lumpP = 0;
      let lumpF = 0;
      let jumpsLeft = Math.max(2, Math.round(30 * realDt));
      for (const t of a.tiles) {
        const lam = this.tileLambda[t];
        this.diag.babbleRate += lam;
        const o = t * TILE_FLOATS;
        const st = this.lastTiles;
        const u = st ? clamp(fin(st[o + 4]), 0, 6) : 0;
        const rlo = this.tileRange[t * 2];
        const rhi = this.tileRange[t * 2 + 1];
        const nInd = poisson(lam * simDt * phi, this.rand);
        const eps0 = 0.5 + 0.5 * Math.min(1, u / 1.2);
        const ex = -1.2; // r^-2.2 density → inverse CDF with exponent a+1 = -1.2
        const l0 = Math.pow(rlo, ex);
        const l1 = Math.pow(rhi, ex);
        for (let k = 0; k < nInd; k++) {
          const r = Math.pow(l0 + (l1 - l0) * this.rand(), 1 / ex);
          const f0 = minnaertFrequency(r);
          if (f0 * p >= nyqF) continue;
          const d = bubbleDamping(f0);
          const A = T.kBabble * eps0 * this.sampleUniform(0.5, 1.5) * Math.pow(r / 1e-3, T.alpha);
          const i = this.bubbles.add();
          const oo = i * LAYOUT.BUBBLE_STRIDE;
          const R = this.bubbles.data;
          R[oo + LAYOUT.B_T] = this.rand() * realDt;
          R[oo + LAYOUT.B_EM] = e;
          R[oo + LAYOUT.B_F0] = f0 * p;
          R[oo + LAYOUT.B_D] = d * p;
          R[oo + LAYOUT.B_CHIRP] = this.sampleUniform(0.05, 0.25) * d * p;
          R[oo + LAYOUT.B_AMP] = A;
          R[oo + LAYOUT.B_KIND] = BUBBLE_KIND.babble;
          R[oo + 7] = 0;
          this.diag.bubbles[BUBBLE_KIND.babble]++;
          this.diag.individualPower += ((A * A) / (4 * d * p) / Math.max(simDt, 1e-4)) * this.rateScale;
        }
        if (phi < 1) {
          const P = (1 - phi) * lam * this.tileEnergy[t] * powerScale;
          lumpP += P;
          lumpF += P * Math.log(this.tileFreq[t] * p);
        }
        // Hydraulic jump gurgles: large, strongly chirping bubbles.
        const nJ = Math.min(jumpsLeft, poisson(this.tileJump[t] * simDt, this.rand));
        jumpsLeft -= nJ;
        for (let k = 0; k < nJ; k++) {
          const r = this.sampleUniform(3e-3, 10e-3);
          const f0 = minnaertFrequency(r);
          const d = bubbleDamping(f0);
          const A = T.kJump * this.sampleUniform(0.6, 1.4) * Math.pow(r / 1e-3, T.alpha) * 0.35;
          const i = this.bubbles.add();
          const oo = i * LAYOUT.BUBBLE_STRIDE;
          const R = this.bubbles.data;
          R[oo + LAYOUT.B_T] = this.rand() * realDt;
          R[oo + LAYOUT.B_EM] = e;
          R[oo + LAYOUT.B_F0] = f0 * p;
          R[oo + LAYOUT.B_D] = d * p;
          R[oo + LAYOUT.B_CHIRP] = this.sampleUniform(0.2, 0.45) * d * p;
          R[oo + LAYOUT.B_AMP] = A;
          R[oo + LAYOUT.B_KIND] = BUBBLE_KIND.jump;
          R[oo + 7] = 0;
          this.diag.bubbles[BUBBLE_KIND.jump]++;
          this.diag.individualPower += ((A * A) / (4 * d * p) / Math.max(simDt, 1e-4)) * this.rateScale;
        }
        // Rush of fast shallow water (subtle, broadband mid).
        const rushF = clamp(300 + 500 * u, 300, 1800) * p;
        this.addNoise(e, 'mid', this.tileRush[t] * powerScale, rushF);
      }
      if (lumpP > 0) {
        this.addNoise(e, 'mid', lumpP, Math.exp(lumpF / lumpP));
        this.diag.lumpedPower += lumpP;
      }
    }
  }

  /** Tile data of the stats used for the current batch (per-tile speed lookups). */
  private lastTiles: Float32Array | null = null;
}
