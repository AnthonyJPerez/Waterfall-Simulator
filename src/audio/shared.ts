/**
 * Process-wide audio singleton: ONE AudioContext for the whole page, the AudioWorklet module (loaded once
 * from a Blob URL), the master chain and the convolution reverb. World rebuilds (preset / quality change)
 * only tear down and recreate the per-world graph (see AudioEngine), which is cheap.
 *
 *   per-world dry  ─▶ busIn ─┐
 *   reverb return ───────────┴▶ master (volume · enabled · visible) ▶ glue compressor ▶ limiter ▶ trim ▶ soft clip ▶ out
 *   per-world sends ▶ revIn ▶ convolver (procedural IR) ▶ reverb return
 *
 * Output gating: the master gain fades to 0 when audio is disabled or the tab is hidden; the context is
 * then suspended after the fade (saves CPU) and resumed (with a fade-in) when needed again.
 */
import { generateImpulseResponse, softClipCurve } from './reverb';
import { workletSource } from './worklet';

export class SharedAudio {
  private static instance: SharedAudio | null = null;

  /** Returns the singleton, creating the AudioContext synchronously (call from a user gesture). */
  static get(): SharedAudio {
    if (!SharedAudio.instance || SharedAudio.instance.ctx.state === 'closed') SharedAudio.instance = new SharedAudio();
    return SharedAudio.instance;
  }

  /** The singleton if it already exists (never creates one). */
  static peek(): SharedAudio | null {
    return SharedAudio.instance;
  }

  readonly ctx: AudioContext;
  readonly busIn: GainNode;
  readonly revIn: GainNode;
  readonly master: GainNode;
  readonly workletReady: Promise<void>;
  private hidden = typeof document !== 'undefined' ? document.hidden : false;
  private enabled = true;
  private volume = 0.64;
  private lastTarget = -1;
  private suspendTimer: ReturnType<typeof setTimeout> | null = null;
  private onVisibility = () => {
    this.hidden = document.hidden;
    this.apply();
  };

  private constructor() {
    const AC: typeof AudioContext | undefined = (globalThis as any).AudioContext ?? (globalThis as any).webkitAudioContext;
    if (!AC) throw new Error('Web Audio is not supported in this browser');
    this.ctx = new AC({ latencyHint: 'interactive' });
    // Must happen synchronously inside the user gesture.
    this.ctx.resume().catch(() => {});
    const ctx = this.ctx;

    this.busIn = new GainNode(ctx, { gain: 1 });
    this.revIn = new GainNode(ctx, { gain: 1 });
    this.master = new GainNode(ctx, { gain: 0 });
    const glue = new DynamicsCompressorNode(ctx, { threshold: -20, knee: 12, ratio: 3, attack: 0.008, release: 0.3 });
    const limiter = new DynamicsCompressorNode(ctx, { threshold: -4, knee: 0, ratio: 20, attack: 0.001, release: 0.12 });
    const trim = new GainNode(ctx, { gain: 0.8 });
    const clip = new WaveShaperNode(ctx, { curve: softClipCurve(), oversample: '2x' });
    const convolver = new ConvolverNode(ctx, { disableNormalization: true });
    const [l, r] = generateImpulseResponse(ctx.sampleRate);
    const ir = ctx.createBuffer(2, l.length, ctx.sampleRate);
    ir.copyToChannel(l as Float32Array<ArrayBuffer>, 0);
    ir.copyToChannel(r as Float32Array<ArrayBuffer>, 1);
    convolver.buffer = ir;

    this.busIn.connect(this.master);
    this.revIn.connect(convolver).connect(this.master);
    this.master.connect(glue).connect(limiter).connect(trim).connect(clip).connect(ctx.destination);

    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', this.onVisibility);
    this.workletReady = this.loadWorklet();
    // Avoid unhandled-rejection noise; callers await workletReady themselves.
    this.workletReady.catch(() => {});
  }

  private async loadWorklet(): Promise<void> {
    const wl = (this.ctx as any).audioWorklet as AudioWorklet | undefined;
    if (!wl) throw new Error('AudioWorklet is not available (requires a secure context)');
    const url = URL.createObjectURL(new Blob([workletSource()], { type: 'text/javascript' }));
    try {
      await wl.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  /** Master volume (0..1 slider, perceptual square law) and enable flag; call every frame (cheap). */
  setOutput(volume: number, enabled: boolean) {
    const v = Number.isFinite(volume) ? Math.min(1, Math.max(0, volume)) : 0;
    this.volume = v * v;
    this.enabled = enabled;
    this.apply();
  }

  /** Resumes the context if output is wanted (e.g. after a world rebuild or a user gesture). */
  resume() {
    if (this.ctx.state === 'suspended' && this.enabled && !this.hidden) this.ctx.resume().catch(() => {});
  }

  get audible() {
    return this.enabled && !this.hidden && this.ctx.state === 'running';
  }

  private apply() {
    const want = this.enabled && !this.hidden;
    const target = want ? this.volume : 0;
    const now = this.ctx.currentTime;
    if (Math.abs(target - this.lastTarget) > 1e-4) {
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setTargetAtTime(target, now, want ? 0.08 : 0.04);
      this.lastTarget = target;
    }
    if (want) {
      if (this.suspendTimer) {
        clearTimeout(this.suspendTimer);
        this.suspendTimer = null;
      }
      if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    } else if (!this.suspendTimer && this.ctx.state === 'running') {
      this.suspendTimer = setTimeout(() => {
        this.suspendTimer = null;
        if (!(this.enabled && !this.hidden) && this.ctx.state === 'running') this.ctx.suspend().catch(() => {});
      }, 400);
    }
  }
}
