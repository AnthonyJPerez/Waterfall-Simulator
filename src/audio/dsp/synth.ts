/**
 * Water-audio DSP core: bubble-resonance bank, noise-burst bank (impact transients, rock splatter),
 * per-emitter continuous noise layers (rumble / rush / hiss) with distance gain + air absorption,
 * and a stereo forest ambience (wind in leaves, gusts, distant FM bird calls).
 *
 * IMPORTANT: `defineSynth` is serialised with Function.prototype.toString() and evaluated inside the
 * AudioWorkletGlobalScope (see ../worklet.ts). It must therefore be completely self-contained: no
 * imports, no references to module-level identifiers, only globals that exist in every JS realm
 * (Math, typed arrays, ...). The very same code is used by the offline unit tests.
 *
 * Realtime rules: no allocation in render(), all parameter changes are smoothed (no zipper noise),
 * voices start at zero phase / with an attack ramp and stop below −60 dB (no clicks).
 */
export function defineSynth() {
  /** Message / parameter record layouts shared with the main thread (see protocol.ts). */
  const LAYOUT = {
    NUM_EMITTERS_MAX: 16,
    /** Bubble record: start time (s, relative to the batch), emitter, f0 (Hz), damping (1/s), chirp σ (1/s), amplitude, kind, unused. */
    BUBBLE_STRIDE: 8,
    B_T: 0,
    B_EM: 1,
    B_F0: 2,
    B_D: 3,
    B_CHIRP: 4,
    B_AMP: 5,
    B_KIND: 6,
    /** Noise-burst record: start time (s), emitter, amplitude (rms at onset), decay (1/s), centre (Hz), Q, attack (s), kind. */
    BURST_STRIDE: 8,
    U_T: 0,
    U_EM: 1,
    U_AMP: 2,
    U_DECAY: 3,
    U_FC: 4,
    U_Q: 5,
    U_ATTACK: 6,
    U_KIND: 7,
    /** Per-emitter continuous parameters. Levels are RMS in output units (1 = full scale at the reference distance). */
    EMITTER_STRIDE: 12,
    E_RUMBLE: 0,
    E_RUMBLE_F: 1,
    E_MID: 2,
    E_MID_F: 3,
    E_HISS: 4,
    E_HISS_F: 5,
    E_BRIGHT: 6,
    E_MOD: 7,
    E_GAIN: 8,
    E_AIR: 9,
    E_REV: 10,
    E_ACTIVE: 11,
    /** Global parameters. */
    GLOBAL_STRIDE: 12,
    G_BUBBLES: 0,
    G_ROAR: 1,
    G_AMBIENCE: 2,
    G_WATER_FADE: 3,
    G_AMB_FADE: 4,
    G_WIND: 5,
    G_BIRDS: 6,
    G_OUT_GAIN: 7,
  };

  const TAU = 6.283185307179586;
  /** Emitter fields smoothed linearly (levels) and in the log domain (frequencies). */
  const LEVEL_IDX = [LAYOUT.E_RUMBLE, LAYOUT.E_MID, LAYOUT.E_HISS, LAYOUT.E_GAIN, LAYOUT.E_REV, LAYOUT.E_MOD];
  const FREQ_IDX = [LAYOUT.E_RUMBLE_F, LAYOUT.E_MID_F, LAYOUT.E_HISS_F, LAYOUT.E_BRIGHT, LAYOUT.E_AIR];
  const PEND = 10; // pending record: start, type, em, p0..p6
  const MAX_BLOCK = 1024;
  const SQRT3 = 1.7320508075688772;

  /** xorshift32; [0, 1). */
  class Rng {
    s: number;
    constructor(seed: number) {
      this.s = seed >>> 0 || 0x2545f491;
    }
    next(): number {
      let x = this.s;
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      this.s = x >>> 0;
      return this.s / 4294967296;
    }
    range(a: number, b: number): number {
      return a + (b - a) * this.next();
    }
    gauss(): number {
      const u = Math.max(1e-12, this.next());
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * this.next());
    }
  }

  const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);
  const finite = (x: number) => x === x && x !== Infinity && x !== -Infinity;

  /** Topology-preserving-transform state-variable filter coefficients (Zavalishin / Simper). */
  function svf(fc: number, q: number, fs: number, out: Float64Array, o: number) {
    const f = clamp(fc, 5, fs * 0.47);
    const g = Math.tan((Math.PI * f) / fs);
    const k = 1 / Math.max(q, 0.05);
    const a1 = 1 / (1 + g * (g + k));
    out[o] = a1;
    out[o + 1] = g * a1;
    out[o + 2] = g * g * a1;
    out[o + 3] = k;
  }

  /** Bird call: list of notes (start s, dur s, f0, f1, fm ratio, fm index (fraction of f), amp, harmonic). */
  const NOTE = 8;
  const MAX_NOTES = 24;
  function makeBirdCall(rng: Rng, notes: Float64Array): number {
    const kind = Math.floor(rng.next() * 4);
    let n = 0;
    let t = 0;
    const push = (dur: number, f0: number, f1: number, fmr: number, fmi: number, amp: number, harm: number) => {
      if (n >= MAX_NOTES) return;
      const o = n * NOTE;
      notes[o] = t;
      notes[o + 1] = dur;
      notes[o + 2] = f0;
      notes[o + 3] = f1;
      notes[o + 4] = fmr;
      notes[o + 5] = fmi;
      notes[o + 6] = amp;
      notes[o + 7] = harm;
      n++;
    };
    if (kind === 0) {
      // Two- or three-note descending whistle ("fee-bee").
      const base = rng.range(3600, 4400);
      const k = rng.next() < 0.3 ? 3 : 2;
      for (let i = 0; i < k; i++) {
        const f = base * (1 - 0.11 * i) * rng.range(0.98, 1.02);
        const dur = rng.range(0.22, 0.34);
        push(dur, f * 1.01, f * 0.985, 0.0, 0.0, i === 0 ? 1 : 0.85, 0.06);
        t += dur + rng.range(0.04, 0.09);
      }
    } else if (kind === 1) {
      // Trill: fast repeated down-chirps.
      const cnt = 7 + Math.floor(rng.next() * 9);
      const rate = rng.range(11, 18);
      const hi = rng.range(5000, 6200);
      const lo = hi * rng.range(0.72, 0.82);
      for (let i = 0; i < cnt; i++) {
        push(0.6 / rate, hi, lo, 0, 0, 0.7 + 0.3 * Math.sin((Math.PI * i) / cnt), 0.12);
        t += 1 / rate;
      }
    } else if (kind === 2) {
      // Warble phrase: FM-modulated notes of varying pitch.
      const cnt = 3 + Math.floor(rng.next() * 4);
      for (let i = 0; i < cnt; i++) {
        const f = rng.range(2200, 3600);
        const dur = rng.range(0.1, 0.22);
        push(dur, f, f * rng.range(0.85, 1.2), rng.range(0.008, 0.02), rng.range(0.04, 0.09), rng.range(0.6, 1), 0.2);
        t += dur + rng.range(0.03, 0.08);
      }
    } else {
      // Chips: a few short sharp calls.
      const cnt = 2 + Math.floor(rng.next() * 3);
      for (let i = 0; i < cnt; i++) {
        const f = rng.range(5200, 6800);
        push(0.018, f, f * 0.62, 0, 0, 1, 0.1);
        t += rng.range(0.16, 0.35);
      }
    }
    return n;
  }

  class Synth {
    readonly fs: number;
    readonly E: number;
    /** Samples rendered so far (the synth's own clock). */
    frame = 0;
    // --- scheduling
    private cursor = 0;
    minLatency: number;
    maxLatency: number;
    private pend: Float64Array;
    private nPend = 0;
    private maxPend: number;
    // --- bubble voices (struct of arrays)
    private maxV: number;
    private vZr: Float64Array;
    private vZi: Float64Array;
    private vWr: Float64Array;
    private vWi: Float64Array;
    private vCr: Float64Array;
    private vCi: Float64Array;
    private vKill: Float64Array;
    private vOff: Int32Array;
    private vEm: Int32Array;
    private vList: Int32Array;
    private nV = 0;
    private vFree: Int32Array;
    private nVFree: number;
    // --- burst voices
    private maxU: number;
    private uEnv: Float64Array;
    private uDec: Float64Array;
    private uAtt: Int32Array;
    private uAttInc: Float64Array;
    private uKill: Float64Array;
    private uS1: Float64Array;
    private uS2: Float64Array;
    private uCo: Float64Array; // a1,a2,a3,k,norm
    private uOff: Int32Array;
    private uEm: Int32Array;
    private uList: Int32Array;
    private nU = 0;
    private uFree: Int32Array;
    private nUFree: number;
    private uRng: Rng;
    // --- per emitter buffers & continuous layers
    private drops: Float32Array[]; // per-emitter scratch for bubbles + bursts
    private target: Float64Array; // E * EMITTER_STRIDE
    private cur: Float64Array; // smoothed
    private eState: Float64Array; // filter states: rumble s1,s2, lp1; mid s1,s2; hiss s1,s2, bright; air; mod x, mod cur
    private eCo: Float64Array; // per emitter coefficients
    private eRng: Rng[];
    private eSilent: Uint8Array;
    // --- globals
    private gTarget: Float64Array;
    private gCur: Float64Array;
    // --- ambience
    private aRng: Rng;
    private gust = 0.4;
    private gustX = 0;
    private gustEvt = 0;
    private gustEvtVel = 0;
    private aState: Float64Array;
    private aCo: Float64Array;
    private envL = 0;
    private envR = 0;
    private birdNotes: Float64Array[];
    private birdN: Int32Array;
    private birdT: Float64Array;
    private birdPh: Float64Array;
    private birdMph: Float64Array;
    private birdPan: Float64Array;
    private birdAmp: Float64Array;
    private birdLp: Float64Array;
    private birdLpC: Float64Array;
    private birdOn: Uint8Array;
    // --- stats
    st = { bubblesStarted: 0, burstsStarted: 0, stolen: 0, droppedPending: 0, peak: 0, nanResets: 0, birds: 0, maxVoices: 0 };

    constructor(fs: number, numEmitters: number, seed = 1, opts?: { maxVoices?: number; maxBursts?: number; maxPending?: number }) {
      this.fs = fs > 0 ? fs : 48000;
      this.E = clamp(Math.floor(numEmitters) || 1, 1, LAYOUT.NUM_EMITTERS_MAX);
      const E = this.E;
      this.minLatency = Math.round(0.012 * this.fs);
      this.maxLatency = Math.round(0.15 * this.fs);
      this.maxPend = (opts && opts.maxPending) || 8192;
      this.pend = new Float64Array(this.maxPend * PEND);
      const V = (this.maxV = (opts && opts.maxVoices) || 384);
      this.vZr = new Float64Array(V);
      this.vZi = new Float64Array(V);
      this.vWr = new Float64Array(V);
      this.vWi = new Float64Array(V);
      this.vCr = new Float64Array(V);
      this.vCi = new Float64Array(V);
      this.vKill = new Float64Array(V);
      this.vOff = new Int32Array(V);
      this.vEm = new Int32Array(V);
      this.vList = new Int32Array(V);
      this.vFree = new Int32Array(V);
      for (let i = 0; i < V; i++) this.vFree[i] = V - 1 - i;
      this.nVFree = V;
      const U = (this.maxU = (opts && opts.maxBursts) || 128);
      this.uEnv = new Float64Array(U);
      this.uDec = new Float64Array(U);
      this.uAtt = new Int32Array(U);
      this.uAttInc = new Float64Array(U);
      this.uKill = new Float64Array(U);
      this.uS1 = new Float64Array(U);
      this.uS2 = new Float64Array(U);
      this.uCo = new Float64Array(U * 5);
      this.uOff = new Int32Array(U);
      this.uEm = new Int32Array(U);
      this.uList = new Int32Array(U);
      this.uFree = new Int32Array(U);
      for (let i = 0; i < U; i++) this.uFree[i] = U - 1 - i;
      this.nUFree = U;
      this.uRng = new Rng(seed * 7919 + 13);

      this.drops = [];
      for (let e = 0; e < E; e++) this.drops.push(new Float32Array(MAX_BLOCK));
      const S = LAYOUT.EMITTER_STRIDE;
      this.target = new Float64Array(E * S);
      this.cur = new Float64Array(E * S);
      for (let e = 0; e < E; e++) {
        for (const arr of [this.target, this.cur]) {
          arr[e * S + LAYOUT.E_RUMBLE_F] = 120;
          arr[e * S + LAYOUT.E_MID_F] = 700;
          arr[e * S + LAYOUT.E_HISS_F] = 3000;
          arr[e * S + LAYOUT.E_BRIGHT] = 12000;
          arr[e * S + LAYOUT.E_AIR] = 20000;
          arr[e * S + LAYOUT.E_GAIN] = 0;
        }
      }
      this.eState = new Float64Array(E * 16);
      this.eCo = new Float64Array(E * 20);
      this.eRng = [];
      for (let e = 0; e < E; e++) this.eRng.push(new Rng(seed * 104729 + e * 7727 + 1));
      this.eSilent = new Uint8Array(E).fill(1);

      this.gTarget = new Float64Array(LAYOUT.GLOBAL_STRIDE);
      this.gCur = new Float64Array(LAYOUT.GLOBAL_STRIDE);
      const g0 = [1, 1, 0.35, 1, 1, 0.5, 0.12, 1];
      for (let i = 0; i < g0.length; i++) this.gTarget[i] = g0[i];
      this.gCur.set(this.gTarget);
      this.gCur[LAYOUT.G_WATER_FADE] = 0; // fade in
      this.gCur[LAYOUT.G_AMB_FADE] = 0;

      this.aRng = new Rng(seed * 31337 + 5);
      this.aState = new Float64Array(32);
      this.aCo = new Float64Array(32);
      this.birdNotes = [];
      const NB = 3;
      for (let b = 0; b < NB; b++) this.birdNotes.push(new Float64Array(MAX_NOTES * NOTE));
      this.birdN = new Int32Array(NB);
      this.birdT = new Float64Array(NB);
      this.birdPh = new Float64Array(NB);
      this.birdMph = new Float64Array(NB);
      this.birdPan = new Float64Array(NB * 2);
      this.birdAmp = new Float64Array(NB);
      this.birdLp = new Float64Array(NB);
      this.birdLpC = new Float64Array(NB);
      this.birdOn = new Uint8Array(NB);
    }

    get activeVoices() {
      return this.nV;
    }
    get activeBursts() {
      return this.nU;
    }
    get pending() {
      return this.nPend;
    }

    /** Silences everything immediately (used after numerical trouble). */
    reset() {
      this.nV = 0;
      this.nU = 0;
      this.nPend = 0;
      for (let i = 0; i < this.maxV; i++) this.vFree[i] = this.maxV - 1 - i;
      this.nVFree = this.maxV;
      for (let i = 0; i < this.maxU; i++) this.uFree[i] = this.maxU - 1 - i;
      this.nUFree = this.maxU;
      this.eState.fill(0);
      this.aState.fill(0);
      this.envL = this.envR = 0;
      this.birdOn.fill(0);
      this.cursor = this.frame;
    }

    /**
     * Schedules a batch of bubbles/bursts covering `duration` seconds of real time. Batches are laid
     * end-to-end on the synth clock (jitter buffer) so irregular message arrival neither bunches nor gaps
     * the events; the backlog is bounded by maxLatency.
     */
    schedule(bubbles: Float32Array | null, nBubbles: number, bursts: Float32Array | null, nBursts: number, duration: number) {
      const fs = this.fs;
      const now = this.frame;
      let start = Math.max(this.cursor, now + this.minLatency);
      if (start > now + this.maxLatency) start = now + this.maxLatency;
      const dur = clamp(finite(duration) ? duration : 0, 0, 0.25);
      this.cursor = start + dur * fs;
      const L = LAYOUT;
      if (bubbles) {
        const nb = Math.min(nBubbles | 0, Math.floor(bubbles.length / L.BUBBLE_STRIDE));
        for (let i = 0; i < nb; i++) {
          const o = i * L.BUBBLE_STRIDE;
          const t = bubbles[o + L.B_T];
          const f0 = bubbles[o + L.B_F0];
          const d = bubbles[o + L.B_D];
          const ch = bubbles[o + L.B_CHIRP];
          const a = bubbles[o + L.B_AMP];
          if (!(finite(t) && finite(f0) && finite(d) && finite(ch) && finite(a)) || a <= 0 || f0 <= 0) continue;
          this.push(start + clamp(t, 0, 1) * fs, 0, bubbles[o + L.B_EM], f0, d, ch, a, 0, 0, 0);
        }
      }
      if (bursts) {
        const nu = Math.min(nBursts | 0, Math.floor(bursts.length / L.BURST_STRIDE));
        for (let i = 0; i < nu; i++) {
          const o = i * L.BURST_STRIDE;
          const t = bursts[o + L.U_T];
          const a = bursts[o + L.U_AMP];
          const dec = bursts[o + L.U_DECAY];
          const fc = bursts[o + L.U_FC];
          const q = bursts[o + L.U_Q];
          const att = bursts[o + L.U_ATTACK];
          if (!(finite(t) && finite(a) && finite(dec) && finite(fc) && finite(q) && finite(att)) || a <= 0) continue;
          this.push(start + clamp(t, 0, 1) * fs, 1, bursts[o + L.U_EM], a, dec, fc, q, att, 0, 0);
        }
      }
    }

    private push(t: number, type: number, em: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number) {
      if (this.nPend >= this.maxPend) {
        this.st.droppedPending++;
        return;
      }
      const o = this.nPend * PEND;
      const P = this.pend;
      P[o] = t;
      P[o + 1] = type;
      P[o + 2] = clamp(Math.floor(em) || 0, 0, this.E - 1);
      P[o + 3] = p0;
      P[o + 4] = p1;
      P[o + 5] = p2;
      P[o + 6] = p3;
      P[o + 7] = p4;
      P[o + 8] = p5;
      P[o + 9] = p6;
      this.nPend++;
    }

    setEmitters(params: Float32Array | Float64Array) {
      const n = Math.min(params.length, this.target.length);
      for (let i = 0; i < n; i++) {
        const v = params[i];
        if (finite(v)) this.target[i] = v;
      }
    }

    setGlobals(g: Float32Array | Float64Array) {
      const n = Math.min(g.length, this.gTarget.length);
      for (let i = 0; i < n; i++) if (finite(g[i])) this.gTarget[i] = g[i];
    }

    private startBubble(off: number, em: number, f0: number, d: number, sigma: number, amp: number) {
      const fs = this.fs;
      const nyq = 0.45 * fs;
      if (f0 >= nyq || f0 < 15) return;
      let i: number;
      if (this.nVFree > 0) {
        i = this.vFree[--this.nVFree];
        this.vList[this.nV++] = i;
      } else {
        // Steal the quietest voice if the new one is louder.
        let best = -1;
        let bestA = Infinity;
        for (let a = 0; a < this.nV; a++) {
          const j = this.vList[a];
          const m = this.vZr[j] * this.vZr[j] + this.vZi[j] * this.vZi[j];
          if (m < bestA) {
            bestA = m;
            best = j;
          }
        }
        if (best < 0 || bestA > amp * amp * 0.25) return;
        i = best;
        this.st.stolen++;
      }
      const dd = clamp(d, 1, 40000);
      const life = 6.9 / dd;
      // Keep the chirp below Nyquist over the audible life.
      const sMax = Math.max(0, (nyq / f0 - 1) / life);
      const s = clamp(sigma, 0, sMax);
      const w = (TAU * f0) / fs;
      const rho = Math.exp(-dd / fs);
      this.vZr[i] = amp;
      this.vZi[i] = 0;
      this.vWr[i] = rho * Math.cos(w);
      this.vWi[i] = rho * Math.sin(w);
      const dw = (TAU * f0 * s) / (fs * fs);
      this.vCr[i] = Math.cos(dw);
      this.vCi[i] = Math.sin(dw);
      const k = Math.max(amp * 1e-3, 3e-6);
      this.vKill[i] = k * k;
      this.vOff[i] = off;
      this.vEm[i] = em;
      this.st.bubblesStarted++;
    }

    private startBurst(off: number, em: number, amp: number, decay: number, fc: number, q: number, attack: number) {
      const fs = this.fs;
      if (this.nUFree <= 0) {
        // Steal the quietest burst.
        let best = -1;
        let bestA = Infinity;
        for (let a = 0; a < this.nU; a++) {
          const j = this.uList[a];
          if (this.uAtt[j] > 0) continue;
          if (this.uEnv[j] < bestA) {
            bestA = this.uEnv[j];
            best = j;
          }
        }
        if (best < 0 || bestA > amp * 0.5) return;
        // Recycle in place (remains in the active list).
        this.initBurst(best, off, em, amp, decay, fc, q, attack);
        this.st.stolen++;
        return;
      }
      const i = this.uFree[--this.nUFree];
      this.uList[this.nU++] = i;
      this.initBurst(i, off, em, amp, decay, fc, q, attack);
    }

    private initBurst(i: number, off: number, em: number, amp: number, decay: number, fc: number, q: number, attack: number) {
      const fs = this.fs;
      const f = clamp(fc, 40, 0.45 * fs);
      const Q = clamp(q, 0.3, 20);
      svf(f, Q, fs, this.uCo, i * 5);
      // Normalise so unit-variance white noise gives unit-rms band noise.
      this.uCo[i * 5 + 4] = Math.min(12, Math.sqrt((0.5 * fs) / ((Math.PI / 2) * (f / Q)))) * SQRT3;
      const attS = Math.max(4, Math.round(clamp(attack, 0, 0.05) * fs));
      this.uAtt[i] = attS;
      this.uAttInc[i] = amp / attS;
      this.uEnv[i] = 0;
      this.uDec[i] = Math.exp(-clamp(decay, 5, 50000) / fs);
      this.uKill[i] = Math.max(amp * 1e-3, 1e-6);
      this.uS1[i] = 0;
      this.uS2[i] = 0;
      this.uOff[i] = off;
      this.uEm[i] = em;
      this.st.burstsStarted++;
    }

    private activate(n: number) {
      const P = this.pend;
      const end = this.frame + n;
      let w = 0;
      for (let r = 0; r < this.nPend; r++) {
        const o = r * PEND;
        if (P[o] < end) {
          const off = clamp(Math.floor(P[o] - this.frame), 0, n - 1);
          const em = P[o + 2] | 0;
          if (P[o + 1] === 0) this.startBubble(off, em, P[o + 3], P[o + 4], P[o + 5], P[o + 6]);
          else this.startBurst(off, em, P[o + 3], P[o + 4], P[o + 5], P[o + 6], P[o + 7]);
        } else {
          if (w !== r) P.copyWithin(w * PEND, o, o + PEND);
          w++;
        }
      }
      this.nPend = w;
    }

    private runBubbles(n: number) {
      const zrA = this.vZr;
      const ziA = this.vZi;
      const wrA = this.vWr;
      const wiA = this.vWi;
      let a = 0;
      while (a < this.nV) {
        const i = this.vList[a];
        const out = this.drops[this.vEm[i]];
        let zr = zrA[i];
        let zi = ziA[i];
        let wr = wrA[i];
        let wi = wiA[i];
        const cr = this.vCr[i];
        const ci = this.vCi[i];
        for (let k = this.vOff[i]; k < n; k++) {
          out[k] += zi;
          const tr = zr * wr - zi * wi;
          zi = zr * wi + zi * wr;
          zr = tr;
          const ur = wr * cr - wi * ci;
          wi = wr * ci + wi * cr;
          wr = ur;
        }
        this.vOff[i] = 0;
        if (zr * zr + zi * zi < this.vKill[i] || !(zr === zr)) {
          this.vFree[this.nVFree++] = i;
          this.vList[a] = this.vList[--this.nV];
          continue;
        }
        zrA[i] = zr;
        ziA[i] = zi;
        wrA[i] = wr;
        wiA[i] = wi;
        a++;
      }
    }

    private runBursts(n: number) {
      let rs = this.uRng.s;
      let a = 0;
      const co = this.uCo;
      while (a < this.nU) {
        const i = this.uList[a];
        const out = this.drops[this.uEm[i]];
        const a1 = co[i * 5];
        const a2 = co[i * 5 + 1];
        const a3 = co[i * 5 + 2];
        const kq = co[i * 5 + 3];
        const norm = co[i * 5 + 4] * kq;
        let s1 = this.uS1[i];
        let s2 = this.uS2[i];
        let env = this.uEnv[i];
        let att = this.uAtt[i];
        const inc = this.uAttInc[i];
        const dec = this.uDec[i];
        for (let k = this.uOff[i]; k < n; k++) {
          rs ^= rs << 13;
          rs ^= rs >>> 17;
          rs ^= rs << 5;
          const x = (rs >>> 0) / 2147483648 - 1;
          const v3 = x - s2;
          const v1 = a1 * s1 + a2 * v3;
          const v2 = s2 + a2 * s1 + a3 * v3;
          s1 = 2 * v1 - s1;
          s2 = 2 * v2 - s2;
          if (att > 0) {
            env += inc;
            att--;
          } else env *= dec;
          out[k] += v1 * norm * env;
        }
        this.uOff[i] = 0;
        if ((att <= 0 && env < this.uKill[i]) || !(s1 === s1)) {
          this.uFree[this.nUFree++] = i;
          this.uList[a] = this.uList[--this.nU];
          continue;
        }
        this.uS1[i] = s1;
        this.uS2[i] = s2;
        this.uEnv[i] = env;
        this.uAtt[i] = att;
        a++;
      }
      this.uRng.s = rs >>> 0 || 1;
    }

    /**
     * Renders one block. outs[e] receives emitter e's mono signal (post distance gain + air absorption),
     * ambL/ambR the stereo ambience, rev the mono water reverb send. Any output may be null.
     */
    render(outs: (Float32Array | null | undefined)[], ambL: Float32Array | null, ambR: Float32Array | null, rev: Float32Array | null, nIn: number) {
      const n = clamp(nIn | 0, 0, MAX_BLOCK);
      if (n === 0) return;
      const fs = this.fs;
      const E = this.E;
      const L = LAYOUT;
      const S = L.EMITTER_STRIDE;
      for (let e = 0; e < E; e++) this.drops[e].fill(0, 0, n);
      if (rev) rev.fill(0, 0, n);

      this.activate(n);
      if (this.nV > this.st.maxVoices) this.st.maxVoices = this.nV;
      this.runBubbles(n);
      this.runBursts(n);

      // Globals: block-rate exponential smoothing, per-sample linear interpolation.
      const kFast = 1 - Math.exp(-n / (0.05 * fs));
      const kFade = 1 - Math.exp(-n / (0.6 * fs));
      const g = this.gCur;
      const gt = this.gTarget;
      const gB0 = g[L.G_BUBBLES];
      const gR0 = g[L.G_ROAR] * g[L.G_WATER_FADE];
      const gO0 = g[L.G_OUT_GAIN];
      for (let i = 0; i < gt.length; i++) {
        const k = i === L.G_WATER_FADE || i === L.G_AMB_FADE ? kFade : kFast;
        g[i] += (gt[i] - g[i]) * k;
      }
      const gB1 = g[L.G_BUBBLES];
      const gR1 = g[L.G_ROAR] * g[L.G_WATER_FADE];
      const gO1 = g[L.G_OUT_GAIN];
      const inv = 1 / n;

      const kPar = 1 - Math.exp(-n / (0.08 * fs));
      const kFreq = 1 - Math.exp(-n / (0.12 * fs));
      const tgt = this.target;
      const cur = this.cur;
      const st = this.eState;
      const co = this.eCo;
      let peak = this.st.peak;
      for (let e = 0; e < E; e++) {
        const b = e * S;
        const out = outs[e];
        const drops = this.drops[e];
        // A (re)spawned emitter was silent: jump straight to its distance gain / filters (no fade-in that
        // would swallow the first plink); its noise levels still ramp up from zero.
        if (tgt[b + L.E_ACTIVE] > 0.5 && cur[b + L.E_ACTIVE] < 0.5) {
          cur[b + L.E_GAIN] = tgt[b + L.E_GAIN];
          cur[b + L.E_REV] = tgt[b + L.E_REV];
          cur[b + L.E_AIR] = clamp(tgt[b + L.E_AIR], 10, 0.47 * fs);
        }
        cur[b + L.E_ACTIVE] = tgt[b + L.E_ACTIVE];
        // Level ramps (start → end of block).
        const r0 = cur[b + L.E_RUMBLE];
        const m0 = cur[b + L.E_MID];
        const h0 = cur[b + L.E_HISS];
        const gain0 = cur[b + L.E_GAIN];
        const rv0 = cur[b + L.E_REV];
        for (let j = 0; j < LEVEL_IDX.length; j++) {
          const idx = b + LEVEL_IDX[j];
          cur[idx] += (tgt[idx] - cur[idx]) * kPar;
        }
        for (let j = 0; j < FREQ_IDX.length; j++) {
          const idx = b + FREQ_IDX[j];
          const tv = clamp(tgt[idx], 10, 0.47 * fs);
          // Smooth frequencies in the log domain.
          const lc = Math.log(Math.max(cur[idx], 10));
          cur[idx] = Math.exp(lc + (Math.log(tv) - lc) * kFreq);
        }
        const r1 = cur[b + L.E_RUMBLE];
        const m1 = cur[b + L.E_MID];
        const h1 = cur[b + L.E_HISS];
        const gain1 = cur[b + L.E_GAIN];
        const rv1 = cur[b + L.E_REV];
        const noiseOn = r0 + r1 + m0 + m1 + h0 + h1 > 1e-7 && gR0 + gR1 > 1e-6;

        // Coefficients (block rate).
        const fr = cur[b + L.E_RUMBLE_F];
        const fm = cur[b + L.E_MID_F];
        const fh = cur[b + L.E_HISS_F];
        const fb = cur[b + L.E_BRIGHT];
        const fa = cur[b + L.E_AIR];
        const c = e * 20;
        svf(fr, 0.707, fs, co, c);
        svf(fm, 0.8, fs, co, c + 4);
        svf(fh, 0.707, fs, co, c + 8);
        const cLpR = 1 - Math.exp((-TAU * fr) / fs);
        const cBright = 1 - Math.exp((-TAU * Math.min(fb, 0.45 * fs)) / fs);
        const cAir = fa >= 0.45 * fs ? 1 : 1 - Math.exp((-TAU * fa) / fs);
        // Noise normalisation (unit-variance white in → unit-rms band out), see docs in the file header.
        const nyq = 0.5 * fs;
        const nR = Math.sqrt(nyq / (0.785 * Math.min(fr, nyq)));
        const nM = Math.sqrt(nyq / ((Math.PI / 2) * (Math.min(fm, nyq) / 0.8)));
        const fbE = Math.min(fb, nyq);
        const hissBand = Math.max(50, fbE * (Math.atan(nyq / fbE) - Math.atan(Math.min(fh, nyq) / fbE)));
        const nH = Math.min(30, Math.sqrt(nyq / hissBand));
        // Surging modulation: OU process at block rate.
        const s = e * 16;
        const dtB = n / fs;
        const ou = this.eRng[e];
        const mx0 = st[s + 12];
        let mx = mx0 + (-mx0 * dtB) / 0.35 + Math.sqrt((2 * dtB) / 0.35) * ou.gauss();
        mx = clamp(mx, -2.5, 2.5);
        st[s + 12] = mx;
        const md = clamp(cur[b + L.E_MOD], 0, 0.8);
        const mod0 = 1 + md * mx0;
        const mod1 = 1 + md * mx;

        let rs1 = st[s];
        let rs2 = st[s + 1];
        let rlp = st[s + 2];
        let ms1 = st[s + 3];
        let ms2 = st[s + 4];
        let hs1 = st[s + 5];
        let hs2 = st[s + 6];
        let hb = st[s + 7];
        let air = st[s + 8];
        const ra1 = co[c],
          ra2 = co[c + 1],
          ra3 = co[c + 2];
        const ma1 = co[c + 4],
          ma2 = co[c + 5],
          ma3 = co[c + 6],
          mk = co[c + 7];
        const ha1 = co[c + 8],
          ha2 = co[c + 9],
          ha3 = co[c + 10],
          hk = co[c + 11];
        let rsd = ou.s;
        for (let k = 0; k < n; k++) {
          const t = k * inv;
          let y = drops[k] * (gB0 + (gB1 - gB0) * t);
          if (noiseOn) {
            rsd ^= rsd << 13;
            rsd ^= rsd >>> 17;
            rsd ^= rsd << 5;
            const x = ((rsd >>> 0) / 2147483648 - 1) * SQRT3;
            // Rumble: 2nd-order LP + 1-pole LP.
            let v3 = x - rs2;
            let v1 = ra1 * rs1 + ra2 * v3;
            let v2 = rs2 + ra2 * rs1 + ra3 * v3;
            rs1 = 2 * v1 - rs1;
            rs2 = 2 * v2 - rs2;
            rlp += (v2 - rlp) * cLpR;
            // Mid (rush / dense babble): band-pass.
            v3 = x - ms2;
            v1 = ma1 * ms1 + ma2 * v3;
            v2 = ms2 + ma2 * ms1 + ma3 * v3;
            ms1 = 2 * v1 - ms1;
            ms2 = 2 * v2 - ms2;
            const mid = mk * v1;
            // Hiss: high-pass + brightness LP.
            v3 = x - hs2;
            v1 = ha1 * hs1 + ha2 * v3;
            v2 = hs2 + ha2 * hs1 + ha3 * v3;
            hs1 = 2 * v1 - hs1;
            hs2 = 2 * v2 - hs2;
            hb += (x - hk * v1 - v2 - hb) * cBright;
            const mod = mod0 + (mod1 - mod0) * t;
            const noise =
              (rlp * nR * (r0 + (r1 - r0) * t) + mid * nM * (m0 + (m1 - m0) * t) + hb * nH * (h0 + (h1 - h0) * t)) * mod;
            y += noise * (gR0 + (gR1 - gR0) * t);
          }
          if (rev) rev[k] += y * (rv0 + (rv1 - rv0) * t);
          air += (y - air) * cAir;
          const o = air * (gain0 + (gain1 - gain0) * t) * (gO0 + (gO1 - gO0) * t);
          if (out) out[k] = o;
          const ao = o < 0 ? -o : o;
          if (ao > peak) peak = ao;
        }
        ou.s = rsd >>> 0 || 1;
        if (!(air === air && rs1 === rs1 && ms1 === ms1 && hs1 === hs1 && hb === hb && rlp === rlp)) {
          st.fill(0, s, s + 12);
          this.st.nanResets++;
          if (out) out.fill(0, 0, n);
        } else {
          st[s] = rs1;
          st[s + 1] = rs2;
          st[s + 2] = rlp;
          st[s + 3] = ms1;
          st[s + 4] = ms2;
          st[s + 5] = hs1;
          st[s + 6] = hs2;
          st[s + 7] = hb;
          st[s + 8] = air;
        }
      }
      for (let e = E; e < outs.length; e++) {
        const o = outs[e];
        if (o) o.fill(0, 0, n);
      }
      if (rev) {
        const gOut = g[L.G_OUT_GAIN];
        for (let k = 0; k < n; k++) rev[k] *= gOut;
      }
      this.renderAmbience(ambL, ambR, n);
      // Final safety: never emit NaN/Inf, hard bound at ±4 (the master limiter follows).
      let bad = false;
      for (let e = 0; e < outs.length; e++) {
        const o = outs[e];
        if (!o) continue;
        for (let k = 0; k < n; k++) {
          const v = o[k];
          if (!(v === v) || v > 4 || v < -4) {
            o[k] = v === v ? (v > 0 ? 4 : -4) : 0;
            if (!(v === v)) bad = true;
          }
        }
      }
      if (bad) {
        this.st.nanResets++;
        this.reset();
      }
      this.st.peak = peak;
      this.frame += n;
    }

    private renderAmbience(ambL: Float32Array | null, ambR: Float32Array | null, n: number) {
      const L = LAYOUT;
      const fs = this.fs;
      const g = this.gCur;
      const level = g[L.G_AMBIENCE] * g[L.G_AMB_FADE] * g[L.G_OUT_GAIN];
      const st = this.aState;
      const prevLevel = st[20];
      st[20] = level;
      if (level < 1e-5 && prevLevel < 1e-5) {
        if (ambL) ambL.fill(0, 0, n);
        if (ambR) ambR.fill(0, 0, n);
        return;
      }
      const rng = this.aRng;
      const dtB = n / fs;
      // Gusts: Ornstein–Uhlenbeck base + sporadic gust events (attack ~2 s, release ~4 s).
      const tau = 3.5;
      this.gustX += (-this.gustX * dtB) / tau + Math.sqrt((2 * dtB) / tau) * rng.gauss();
      this.gustX = clamp(this.gustX, -2.5, 2.5);
      if (rng.next() < dtB / 11) this.gustEvtVel = rng.range(0.25, 0.6);
      if (this.gustEvtVel > 0) {
        this.gustEvt += this.gustEvtVel * dtB;
        if (this.gustEvt > 0.6 + this.gustEvtVel) this.gustEvtVel = 0;
      } else this.gustEvt *= Math.exp(-dtB / 3.5);
      const wind = clamp(g[L.G_WIND], 0, 2);
      const g0 = this.gust;
      const g1 = clamp((0.34 + 0.15 * this.gustX + this.gustEvt) * (0.4 + wind), 0.1, 2);
      this.gust = g1;
      // Filters.
      const co = this.aCo;
      const leafF = 1600 + 1600 * Math.min(g1, 1.5);
      svf(leafF, 0.55, fs, co, 0);
      svf(leafF * 1.13, 0.55, fs, co, 4);
      svf(380 + 420 * Math.min(g1, 1.5), 0.7, fs, co, 8);
      svf(110, 0.707, fs, co, 12);
      const nyq = fs / 2;
      const nLeaf = Math.sqrt(nyq / ((Math.PI / 2) * (leafF / 0.55)));
      const nWh = Math.sqrt(nyq / ((Math.PI / 2) * ((380 + 420 * Math.min(g1, 1.5)) / 0.7)));
      const nLow = Math.sqrt(nyq / (1.11 * 110));
      // Rustle: shot-noise envelope (leaves knocking), rate ∝ gust².
      const rate = 120 + 1600 * g1 * g1;
      const pHit = Math.min(0.5, rate / fs);
      const decR = Math.exp(-1 / (0.004 * fs));
      const tauS = 0.004;
      const envRms = Math.sqrt(rate * tauS * 0.5 * (1 / 3) + Math.pow(rate * tauS * 0.5, 2)) + 1e-9; // amp ~ U(0,1)
      const envNorm = 1 / envRms;
      const leafLvl0 = 0.07 * g0;
      const leafLvl1 = 0.07 * g1;
      const whLvl0 = 0.02 * g0 * g0;
      const whLvl1 = 0.02 * g1 * g1;
      const lowLvl0 = 0.03 * g0;
      const lowLvl1 = 0.03 * g1;
      const inv = 1 / n;
      let l1 = st[0],
        l2 = st[1],
        q1 = st[2],
        q2 = st[3],
        w1 = st[4],
        w2 = st[5],
        o1 = st[6],
        o2 = st[7],
        w1r = st[8],
        w2r = st[9];
      let envL = this.envL;
      let envR = this.envR;
      let rs = rng.s;
      const SQ = SQRT3;
      for (let k = 0; k < n; k++) {
        const t = k * inv;
        rs ^= rs << 13;
        rs ^= rs >>> 17;
        rs ^= rs << 5;
        const xL = ((rs >>> 0) / 2147483648 - 1) * SQ;
        rs ^= rs << 13;
        rs ^= rs >>> 17;
        rs ^= rs << 5;
        const xR = ((rs >>> 0) / 2147483648 - 1) * SQ;
        rs ^= rs << 13;
        rs ^= rs >>> 17;
        rs ^= rs << 5;
        const u = (rs >>> 0) / 4294967296;
        // Leaf band (L/R independent).
        let v3 = xL - l2;
        let v1 = co[0] * l1 + co[1] * v3;
        let v2 = l2 + co[1] * l1 + co[2] * v3;
        l1 = 2 * v1 - l1;
        l2 = 2 * v2 - l2;
        const leafL = co[3] * v1 * nLeaf;
        v3 = xR - q2;
        v1 = co[4] * q1 + co[5] * v3;
        v2 = q2 + co[5] * q1 + co[6] * v3;
        q1 = 2 * v1 - q1;
        q2 = 2 * v2 - q2;
        const leafR = co[7] * v1 * nLeaf;
        // Shot-noise envelopes.
        envL *= decR;
        envR *= decR;
        if (u < pHit) envL += u / pHit;
        else if (u > 1 - pHit) envR += (1 - u) / pHit;
        // Whoosh (correlated + decorrelated parts).
        v3 = (xL + xR) * 0.7071 - w2;
        v1 = co[8] * w1 + co[9] * v3;
        v2 = w2 + co[9] * w1 + co[10] * v3;
        w1 = 2 * v1 - w1;
        w2 = 2 * v2 - w2;
        const wh = co[11] * v1 * nWh;
        v3 = xR - w2r;
        v1 = co[8] * w1r + co[9] * v3;
        v2 = w2r + co[9] * w1r + co[10] * v3;
        w1r = 2 * v1 - w1r;
        w2r = 2 * v2 - w2r;
        const whR = co[11] * v1 * nWh;
        // Low wind.
        v3 = xL - o2;
        v1 = co[12] * o1 + co[13] * v3;
        v2 = o2 + co[13] * o1 + co[14] * v3;
        o1 = 2 * v1 - o1;
        o2 = 2 * v2 - o2;
        const low = v2 * nLow;
        const ll = leafLvl0 + (leafLvl1 - leafLvl0) * t;
        const wl = whLvl0 + (whLvl1 - whLvl0) * t;
        const lo = lowLvl0 + (lowLvl1 - lowLvl0) * t;
        const lv = prevLevel + (level - prevLevel) * t;
        const sL = (leafL * envL * envNorm * ll * 0.8 + leafL * ll * 0.35 + wh * wl + low * lo) * lv;
        const sR = (leafR * envR * envNorm * ll * 0.8 + leafR * ll * 0.35 + (0.6 * wh + 0.8 * whR) * wl + low * lo * 0.9) * lv;
        if (ambL) ambL[k] = sL;
        if (ambR) ambR[k] = sR;
      }
      rng.s = rs >>> 0 || 1;
      if (!(l1 === l1 && q1 === q1 && w1 === w1 && o1 === o1 && w1r === w1r && envL === envL && envR === envR)) {
        st.fill(0, 0, 10);
        envL = envR = 0;
      } else {
        st[0] = l1;
        st[1] = l2;
        st[2] = q1;
        st[3] = q2;
        st[4] = w1;
        st[5] = w2;
        st[6] = o1;
        st[7] = o2;
        st[8] = w1r;
        st[9] = w2r;
      }
      this.envL = envL;
      this.envR = envR;
      this.renderBirds(ambL, ambR, n, level);
    }

    private renderBirds(ambL: Float32Array | null, ambR: Float32Array | null, n: number, level: number) {
      const L = LAYOUT;
      const fs = this.fs;
      const rng = this.aRng;
      const rate = clamp(this.gCur[L.G_BIRDS], 0, 2);
      if (rate > 0 && rng.next() < (rate * n) / fs) {
        for (let b = 0; b < this.birdOn.length; b++) {
          if (this.birdOn[b]) continue;
          this.birdN[b] = makeBirdCall(rng, this.birdNotes[b]);
          this.birdT[b] = 0;
          this.birdPh[b] = 0;
          this.birdMph[b] = 0;
          const pan = rng.range(-0.95, 0.95);
          const ang = ((pan + 1) * Math.PI) / 4;
          this.birdPan[b * 2] = Math.cos(ang);
          this.birdPan[b * 2 + 1] = Math.sin(ang);
          // Distant birds: −34 … −22 dBFS peak, duller with distance.
          const dist = rng.next();
          this.birdAmp[b] = Math.pow(10, (-22 - 12 * dist) / 20);
          this.birdLpC[b] = 1 - Math.exp((-TAU * (9000 - 5000 * dist)) / fs);
          this.birdLp[b] = 0;
          this.birdOn[b] = 1;
          this.st.birds++;
          break;
        }
      }
      if (!ambL || !ambR) return;
      for (let b = 0; b < this.birdOn.length; b++) {
        if (!this.birdOn[b]) continue;
        const notes = this.birdNotes[b];
        const nn = this.birdN[b];
        let t = this.birdT[b];
        let ph = this.birdPh[b];
        let mph = this.birdMph[b];
        let lp = this.birdLp[b];
        const lpc = this.birdLpC[b];
        const amp = this.birdAmp[b] * level;
        const pl = this.birdPan[b * 2];
        const pr = this.birdPan[b * 2 + 1];
        const dt = 1 / fs;
        let note = 0;
        for (let k = 0; k < n; k++) {
          while (note < nn && t > notes[note * NOTE] + notes[note * NOTE + 1]) note++;
          let y = 0;
          if (note < nn && t >= notes[note * NOTE]) {
            const o = note * NOTE;
            const dur = notes[o + 1];
            const tn = (t - notes[o]) / dur;
            const f = notes[o + 2] + (notes[o + 3] - notes[o + 2]) * tn;
            const fmr = notes[o + 4];
            let inst = f;
            if (fmr > 0) {
              mph += TAU * (fmr * f) * dt;
              inst = f * (1 + notes[o + 5] * Math.sin(mph));
            }
            ph += TAU * inst * dt;
            if (ph > TAU) ph -= TAU;
            // Raised-cosine attack/release (≈ 8 ms / 25 ms).
            const ta = t - notes[o];
            const tr = notes[o] + dur - t;
            const ea = ta < 0.008 ? 0.5 - 0.5 * Math.cos((Math.PI * ta) / 0.008) : 1;
            const er = tr < 0.025 ? 0.5 - 0.5 * Math.cos((Math.PI * Math.max(tr, 0)) / 0.025) : 1;
            y = (Math.sin(ph) + notes[o + 7] * Math.sin(2 * ph)) * ea * er * notes[o + 6];
          }
          lp += (y - lp) * lpc;
          ambL[k] += lp * amp * pl;
          ambR[k] += lp * amp * pr;
          t += dt;
        }
        this.birdT[b] = t;
        this.birdPh[b] = ph;
        this.birdMph[b] = mph;
        this.birdLp[b] = lp;
        const endT = nn > 0 ? notes[(nn - 1) * NOTE] + notes[(nn - 1) * NOTE + 1] + 0.05 : 0;
        if (t > endT || !(lp === lp)) this.birdOn[b] = 0;
      }
    }
  }

  return { LAYOUT, Synth, Rng };
}

export type SynthModule = ReturnType<typeof defineSynth>;
export type SynthInstance = InstanceType<SynthModule['Synth']>;
