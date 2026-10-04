/**
 * Procedural, simulation-driven water audio.
 *
 * STUB IMPLEMENTATION — silent; reads back nothing. To be replaced by the audio agent
 * (keep the AudioModule API).
 */
import type { AudioModule, FrameContext, ModuleContext } from '../app/modules';

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
