/**
 * AudioWorklet processor wrapping the DSP core.
 *
 * Loading strategy (robust in Vite dev AND production builds, no extra asset file): the processor source
 * is assembled at runtime from `workletMain.toString()` + `defineSynth.toString()` and loaded from a Blob
 * URL. Both functions are self-contained (they only touch their parameters and AudioWorkletGlobalScope
 * globals), so minification / bundling cannot break them. tests/audio.dsp.test.ts evaluates the generated
 * source in a fake worklet scope to guard this.
 */
import { defineSynth, type SynthModule } from './dsp/synth';
import { PROCESSOR_NAME } from './protocol';

/** Runs inside AudioWorkletGlobalScope. MUST stay self-contained (it is stringified). */
export function workletMain(mod: SynthModule, name: string) {
  const scope = globalThis as any;
  const Base = scope.AudioWorkletProcessor as { new (): { port: MessagePort } };
  class WaterSynthProcessor extends Base {
    E: number;
    synth: InstanceType<SynthModule['Synth']>;
    alive = true;
    statsEvery: number;
    sinceStats = 0;
    constructor(options?: { processorOptions?: { numEmitters?: number; seed?: number } }) {
      super();
      const po = (options && options.processorOptions) || {};
      this.E = Math.max(1, Math.min(16, po.numEmitters || 8));
      const fs = typeof scope.sampleRate === 'number' && scope.sampleRate > 0 ? scope.sampleRate : 48000;
      this.synth = new mod.Synth(fs, this.E, po.seed || 1);
      this.statsEvery = Math.round(fs * 0.5);
      this.port.onmessage = (ev: MessageEvent) => this.onMessage(ev.data);
    }
    onMessage(m: any) {
      if (!m || typeof m !== 'object') return;
      try {
        if (m.type === 'frame') {
          if (m.emitters) this.synth.setEmitters(m.emitters);
          if (m.globals) this.synth.setGlobals(m.globals);
          this.synth.schedule(m.bubbles || null, m.nBubbles || 0, m.bursts || null, m.nBursts || 0, m.duration || 0);
        } else if (m.type === 'globals') {
          if (m.globals) this.synth.setGlobals(m.globals);
        } else if (m.type === 'reset') {
          this.synth.reset();
        } else if (m.type === 'dispose') {
          this.alive = false;
        }
      } catch (e) {
        // Never let a malformed message kill the audio thread.
      }
    }
    process(_inputs: Float32Array[][], outputs: Float32Array[][]) {
      if (!this.alive) return false;
      const E = this.E;
      const outs: (Float32Array | null)[] = this.outs || (this.outs = []);
      outs.length = 0;
      let n = 128;
      for (let e = 0; e < E; e++) {
        const o = outputs[e] && outputs[e][0];
        outs.push(o || null);
        if (o) n = o.length;
      }
      const amb = outputs[E];
      const rev = outputs[E + 1];
      this.synth.render(outs, (amb && amb[0]) || null, (amb && amb[1]) || null, (rev && rev[0]) || null, n);
      this.sinceStats += n;
      if (this.sinceStats >= this.statsEvery) {
        this.sinceStats = 0;
        const s = this.synth.st;
        this.port.postMessage({
          type: 'stats',
          voices: this.synth.activeVoices,
          bursts: this.synth.activeBursts,
          pending: this.synth.pending,
          bubblesStarted: s.bubblesStarted,
          burstsStarted: s.burstsStarted,
          stolen: s.stolen,
          droppedPending: s.droppedPending,
          peak: s.peak,
          nanResets: s.nanResets,
          birds: s.birds,
          maxVoices: s.maxVoices,
        });
        s.peak = 0;
        s.maxVoices = 0;
      }
      return true;
    }
    outs: (Float32Array | null)[] | null = null;
  }
  scope.registerProcessor(name, WaterSynthProcessor);
}

/** Full AudioWorklet module source. */
export function workletSource(): string {
  return `"use strict";\n(${workletMain.toString()})((${defineSynth.toString()})(), ${JSON.stringify(PROCESSOR_NAME)});\n`;
}
