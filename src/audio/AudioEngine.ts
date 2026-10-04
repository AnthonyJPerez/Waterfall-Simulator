/**
 * Procedural, simulation-driven water audio (AudioModule).
 *
 * Per frame:
 *   encodeReadback: copy world.simEvents (adaptive prefix) and world.sweStats into mappable staging buffers
 *   afterSubmit:    smooth the listener (= camera), update the 8 HRTF panners, start the async maps; when the
 *                   data arrives, AudioMapper turns it into bubbles / noise bursts / continuous layer
 *                   parameters which are posted to the AudioWorklet synth (dsp/synth.ts).
 * Lag and dropped readbacks are tolerated: every frame produces exactly one batch (with or without
 * events), and the worklet lays batches end-to-end on its own clock (jitter buffer).
 *
 * Lifetime: the AudioContext, worklet module, master chain and reverb are a page-wide singleton
 * (shared.ts). destroy() fades out and disconnects only this world's graph, so the App's
 * destroy/recreate on preset or quality changes is cheap and click-free.
 */
import type { AudioModule, FrameContext, ModuleContext } from '../app/modules';
import { ReadbackRing } from '../gpu/util';
import { IMPACT_EVENT_BYTES, MAX_IMPACT_EVENTS, SIM_EVENTS_BYTES, SIM_EVENTS_HEADER_BYTES, SWE_STATS_BYTES } from '../world/wgsl';
import { AudioMapper, type ListenerPose, type MapperDiagnostics } from './mapping';
import { LAYOUT, NUM_EMITTERS, OUT_AMBIENCE, OUT_REVERB, PROCESSOR_NAME, type FrameMessage, type WorkletStats } from './protocol';
import { parseSimEvents, parseSweStats, type ParsedEvents, type ParsedStats } from './readback';
import { SharedAudio } from './shared';

/** Silent fallback (kept for environments without Web Audio; not used by the registry). */
export class StubAudioEngine implements AudioModule {
  running = false;
  constructor(private ctx: ModuleContext) {}
  async start() {
    this.running = true;
  }
  encodeReadback(_encoder: GPUCommandEncoder, _frame: FrameContext) {}
  afterSubmit(_frame: FrameContext) {}
  destroy() {}
}

interface WorldGraph {
  node: AudioWorkletNode;
  panners: PannerNode[];
  dry: GainNode;
  rev: GainNode;
  ambSend: GainNode;
  generations: number[];
}

interface FrameMeta {
  dt: number;
  realDt: number;
  timeScale: number;
  paused: boolean;
  frameIndex: number;
}

interface PendingReadback {
  meta: FrameMeta;
  ev: (() => Promise<ArrayBuffer | null>) | null;
  st: (() => Promise<ArrayBuffer | null>) | null;
}

type V3 = [number, number, number];

const norm = (v: V3): V3 => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 1e-9 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, -1];
};
const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export class WaterAudioEngine implements AudioModule {
  private shared: SharedAudio | null = null;
  private graph: WorldGraph | null = null;
  private startPromise: Promise<void> | null = null;
  private destroyed = false;
  private readonly mapper: AudioMapper;
  private evRing: ReadbackRing | null = null;
  private stRing: ReadbackRing | null = null;
  private pending: PendingReadback | null = null;
  private latestStats: ParsedStats | null = null;
  private lastEventCount = 0;
  private lastEventSig = new Float32Array(9);
  private inFlight = 0;
  private listener: { pos: V3; fwd: V3; up: V3; right: V3 } | null = null;
  private lastReverb = -1;
  private sent = { bubbles: 0, bursts: 0, batches: 0, staleEvents: 0, droppedReadbacks: 0 };
  /** Latest stats posted by the worklet (≈ 2 Hz). */
  workletStats: WorkletStats | null = null;

  constructor(private ctx: ModuleContext) {
    this.mapper = new AudioMapper(NUM_EMITTERS, 0x5eed);
    this.mapper.defaultEventVolume = ctx.world.particleVolume;
  }

  get running(): boolean {
    return !this.destroyed && (this.graph !== null || this.startPromise !== null);
  }

  /** Debug / automation snapshot (window.__wf.app.modules.audio.stats). */
  get stats() {
    return {
      running: this.running,
      state: this.shared?.ctx.state ?? 'none',
      sampleRate: this.shared?.ctx.sampleRate ?? 0,
      sent: { ...this.sent },
      diag: this.mapper.diag as MapperDiagnostics,
      emitters: this.mapper.clusterer.emitters.filter((e) => e.active).map((e) => [+e.x.toFixed(3), +e.y.toFixed(3), +e.z.toFixed(3)]),
      worklet: this.workletStats,
    };
  }

  start(): Promise<void> {
    if (this.destroyed) return Promise.resolve();
    // Create / resume the context synchronously: this call is made from a user gesture.
    let shared: SharedAudio;
    try {
      shared = SharedAudio.get();
    } catch (e) {
      return Promise.reject(e);
    }
    this.shared = shared;
    shared.resume();
    if (!this.startPromise) {
      this.startPromise = this.build(shared).catch((e) => {
        this.startPromise = null;
        throw e;
      });
    }
    return this.startPromise;
  }

  private async build(shared: SharedAudio) {
    await shared.workletReady;
    if (this.destroyed || this.graph) return;
    const ctx = shared.ctx;
    const E = NUM_EMITTERS;
    const node = new AudioWorkletNode(ctx, PROCESSOR_NAME, {
      numberOfInputs: 0,
      numberOfOutputs: E + 2,
      outputChannelCount: [...Array(E).fill(1), 2, 1],
      processorOptions: { numEmitters: E, seed: (Math.random() * 1e9) | 0 },
    });
    node.port.onmessage = (ev) => {
      if (ev.data?.type === 'stats') this.workletStats = ev.data as WorkletStats;
    };
    node.onprocessorerror = () => console.warn('[audio] worklet processor error');
    const dry = new GainNode(ctx, { gain: 0 });
    const rev = new GainNode(ctx, { gain: 0 });
    const ambSend = new GainNode(ctx, { gain: 0.5 });
    const panners: PannerNode[] = [];
    for (let e = 0; e < E; e++) {
      const p = new PannerNode(ctx, {
        panningModel: 'HRTF',
        distanceModel: 'inverse',
        refDistance: 1,
        maxDistance: 10000,
        rolloffFactor: 0, // distance attenuation + air absorption are done per emitter in the worklet
        coneInnerAngle: 360,
        coneOuterAngle: 360,
        coneOuterGain: 1,
        positionX: 0,
        positionY: 0,
        positionZ: -1,
      });
      node.connect(p, e, 0);
      p.connect(dry);
      panners.push(p);
    }
    node.connect(dry, OUT_AMBIENCE, 0);
    node.connect(ambSend, OUT_AMBIENCE, 0);
    ambSend.connect(rev);
    node.connect(rev, OUT_REVERB, 0);
    dry.connect(shared.busIn);
    rev.connect(shared.revIn);
    const now = ctx.currentTime;
    dry.gain.setTargetAtTime(1, now, 0.05);
    this.lastReverb = -1;
    this.mapper.sampleRate = ctx.sampleRate;
    this.mapper.reset();
    this.graph = { node, panners, dry, rev, ambSend, generations: new Array(E).fill(-1) };
  }

  encodeReadback(encoder: GPUCommandEncoder, frame: FrameContext) {
    this.pending = null;
    if (!this.graph || this.destroyed) return;
    if (!frame.params.audio.enabled || !this.shared?.audible) return;
    const device = this.ctx.gpu.device;
    if (!this.evRing) this.evRing = new ReadbackRing(device, SIM_EVENTS_BYTES, 4, 'audio.events');
    if (!this.stRing) this.stRing = new ReadbackRing(device, SWE_STATS_BYTES, 3, 'audio.sweStats');
    const meta: FrameMeta = {
      dt: frame.dt,
      realDt: frame.realDt,
      timeScale: frame.params.sim.timeScale,
      paused: frame.params.sim.paused,
      frameIndex: frame.frameIndex,
    };
    // Bound the backlog: if the GPU is far behind, skip this frame's copies (events are lost, the
    // continuous layers carry on from the last known state).
    let ev: PendingReadback['ev'] = null;
    let st: PendingReadback['st'] = null;
    if (this.inFlight < 6) {
      // Adaptive prefix copy: only as many events as recently used (+ headroom).
      let cap = 256;
      while (cap < Math.min(MAX_IMPACT_EVENTS, this.lastEventCount * 1.5 + 64)) cap *= 2;
      cap = Math.min(cap, MAX_IMPACT_EVENTS);
      ev = this.evRing.encodeCopy(encoder, frame.world.simEvents, 0, SIM_EVENTS_HEADER_BYTES + cap * IMPACT_EVENT_BYTES);
      if (frame.frameIndex % 2 === 0 || !this.latestStats) st = this.stRing.encodeCopy(encoder, frame.world.sweStats);
    }
    if (!ev) this.sent.droppedReadbacks++;
    this.pending = { meta, ev, st };
  }

  afterSubmit(frame: FrameContext) {
    if (!this.graph || this.destroyed || !this.shared) return;
    const a = frame.params.audio;
    this.shared.setOutput(a.volume, a.enabled);
    this.updateListener(frame);
    const g = this.graph;
    const reverb = Math.min(1, Math.max(0, Number.isFinite(a.reverb) ? a.reverb : 0));
    if (Math.abs(reverb - this.lastReverb) > 1e-3) {
      g.rev.gain.setTargetAtTime(reverb, this.shared.ctx.currentTime, 0.08);
      this.lastReverb = reverb;
    }
    const pend = this.pending;
    this.pending = null;
    if (!pend) return;
    this.inFlight++;
    const evP = pend.ev ? pend.ev() : Promise.resolve(null);
    const stP = pend.st ? pend.st() : Promise.resolve(null);
    Promise.all([evP, stP])
      .then(([evBuf, stBuf]) => this.processFrame(pend.meta, evBuf, stBuf, frame))
      .catch((e) => console.warn('[audio] frame processing failed', e))
      .finally(() => this.inFlight--);
  }

  private processFrame(meta: FrameMeta, evBuf: ArrayBuffer | null, stBuf: ArrayBuffer | null, frame: FrameContext) {
    const g = this.graph;
    if (this.destroyed || !g || !this.listener) return;
    if (stBuf) {
      const s = parseSweStats(stBuf);
      if (s) this.latestStats = s;
    }
    let events: ParsedEvents | null = null;
    if (evBuf) {
      events = parseSimEvents(evBuf);
      this.lastEventCount = Math.min(events.count, MAX_IMPACT_EVENTS);
      if (this.isStale(events)) {
        this.sent.staleEvents++;
        events = null;
      }
    }
    const p = frame.params.audio;
    const pose: ListenerPose = this.listener;
    const m = this.mapper;
    m.process({
      simDt: meta.dt,
      realDt: meta.realDt,
      timeScale: meta.timeScale,
      paused: meta.paused,
      listener: pose,
      events,
      stats: this.latestStats,
      gains: { bubbles: p.bubbles, roar: p.roar, ambience: p.ambience },
    });
    const msg: FrameMessage = {
      type: 'frame',
      duration: Math.min(0.25, Math.max(0, meta.realDt)),
      bubbles: m.bubbles.data.slice(0, m.bubbles.n * LAYOUT.BUBBLE_STRIDE),
      nBubbles: m.bubbles.n,
      bursts: m.bursts.data.slice(0, m.bursts.n * LAYOUT.BURST_STRIDE),
      nBursts: m.bursts.n,
      emitters: m.emitterParams.slice(),
      globals: m.globals.slice(),
    };
    g.node.port.postMessage(msg, [msg.bubbles.buffer, msg.bursts.buffer, msg.emitters.buffer, msg.globals.buffer]);
    this.sent.bubbles += m.bubbles.n;
    this.sent.bursts += m.bursts.n;
    this.sent.batches++;
    this.updatePanners(false);
  }

  /** The particle sim clears simEvents every step; identical content two frames in a row means a stale list. */
  private isStale(ev: ParsedEvents): boolean {
    const sig = this.lastEventSig;
    const n = Math.min(8, ev.data.length);
    let same = ev.n > 0 && sig[8] === ev.count;
    for (let i = 0; i < n; i++) if (sig[i] !== ev.data[i]) same = false;
    for (let i = 0; i < 8; i++) sig[i] = i < n ? ev.data[i] : NaN;
    sig[8] = ev.count;
    return same;
  }

  private updateListener(frame: FrameContext) {
    const cam = frame.camera;
    const pos: V3 = [cam.position[0], cam.position[1], cam.position[2]];
    const fwd = norm([cam.target[0] - pos[0], cam.target[1] - pos[1], cam.target[2] - pos[2]]);
    if (![...pos, ...fwd].every(Number.isFinite)) return;
    if (!this.listener) {
      const right = norm(cross(fwd, [0, 1, 0]));
      this.listener = { pos, fwd, up: norm(cross(right, fwd)), right };
    } else {
      const L = this.listener;
      const k = 1 - Math.exp(-Math.max(0, frame.realDt) / 0.06);
      for (let i = 0; i < 3; i++) {
        L.pos[i] += (pos[i] - L.pos[i]) * k;
        L.fwd[i] += (fwd[i] - L.fwd[i]) * k;
      }
      L.fwd = norm(L.fwd);
      let right = cross(L.fwd, [0, 1, 0]);
      if (Math.hypot(...right) < 1e-4) right = [1, 0, 0];
      L.right = norm(right);
      L.up = norm(cross(L.right, L.fwd));
    }
    this.updatePanners(true);
  }

  /** Emitter positions in listener space (listener at the origin, facing −Z, +Y up). */
  private updatePanners(smooth: boolean) {
    const g = this.graph;
    const L = this.listener;
    if (!g || !L || !this.shared) return;
    const now = this.shared.ctx.currentTime;
    const em = this.mapper.clusterer.emitters;
    for (let e = 0; e < g.panners.length; e++) {
      const s = em[e];
      if (!s.active) continue;
      const rel = [s.x - L.pos[0], s.y - L.pos[1], s.z - L.pos[2]];
      let x = dot(rel, L.right);
      let y = dot(rel, L.up);
      let z = -dot(rel, L.fwd);
      // Keep a minimum distance so the HRTF direction stays well defined right next to the camera.
      const d = Math.hypot(x, y, z);
      if (d < 0.05) {
        const s2 = d > 1e-6 ? 0.05 / d : 0;
        x = d > 1e-6 ? x * s2 : 0;
        y = d > 1e-6 ? y * s2 : 0;
        z = d > 1e-6 ? z * s2 : -0.05;
      }
      if (![x, y, z].every(Number.isFinite)) continue;
      const p = g.panners[e];
      const jump = g.generations[e] !== s.generation;
      g.generations[e] = s.generation;
      if (jump) {
        p.positionX.cancelScheduledValues(now);
        p.positionY.cancelScheduledValues(now);
        p.positionZ.cancelScheduledValues(now);
        p.positionX.setValueAtTime(x, now);
        p.positionY.setValueAtTime(y, now);
        p.positionZ.setValueAtTime(z, now);
      } else if (smooth) {
        p.positionX.setTargetAtTime(x, now, 0.03);
        p.positionY.setTargetAtTime(y, now, 0.03);
        p.positionZ.setTargetAtTime(z, now, 0.03);
      }
    }
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.pending = null;
    this.evRing?.destroy();
    this.stRing?.destroy();
    this.evRing = this.stRing = null;
    const g = this.graph;
    this.graph = null;
    this.startPromise = null;
    if (g && this.shared) {
      const now = this.shared.ctx.currentTime;
      g.dry.gain.cancelScheduledValues(now);
      g.rev.gain.cancelScheduledValues(now);
      g.dry.gain.setTargetAtTime(0, now, 0.02);
      g.rev.gain.setTargetAtTime(0, now, 0.02);
      // Disconnect after the fade (the reverb tail keeps ringing in the shared convolver).
      setTimeout(() => {
        try {
          g.node.port.postMessage({ type: 'dispose' });
          g.node.port.onmessage = null;
          g.node.disconnect();
          g.panners.forEach((p) => p.disconnect());
          g.dry.disconnect();
          g.rev.disconnect();
          g.ambSend.disconnect();
        } catch {
          /* already disconnected */
        }
      }, 150);
    }
  }
}
