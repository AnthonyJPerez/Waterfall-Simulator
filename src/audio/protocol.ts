/**
 * Main thread ⇄ AudioWorklet protocol. The record layouts are defined once, inside the self-contained
 * DSP core (dsp/synth.ts), and re-exported here for the main-thread code.
 */
import { defineSynth } from './dsp/synth';

/** Evaluated once on the main thread: the same classes the worklet runs (also used by offline tests). */
export const SYNTH = defineSynth();
export const LAYOUT = SYNTH.LAYOUT;

export const PROCESSOR_NAME = 'creekside-water-synth';

/** Number of spatial emitters (PannerNodes) per world. */
export const NUM_EMITTERS = 8;

/** Worklet node outputs: NUM_EMITTERS mono emitters, then stereo ambience, then mono water reverb send. */
export const OUT_AMBIENCE = NUM_EMITTERS;
export const OUT_REVERB = NUM_EMITTERS + 1;

export interface FrameMessage {
  type: 'frame';
  /** Real-time duration (s) covered by this batch. */
  duration: number;
  bubbles: Float32Array;
  nBubbles: number;
  bursts: Float32Array;
  nBursts: number;
  emitters: Float32Array;
  globals: Float32Array;
}

export interface WorkletStats {
  type: 'stats';
  voices: number;
  bursts: number;
  pending: number;
  bubblesStarted: number;
  burstsStarted: number;
  stolen: number;
  droppedPending: number;
  peak: number;
  nanResets: number;
  birds: number;
  maxVoices: number;
}
