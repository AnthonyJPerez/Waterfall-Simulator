/** Small WebGPU helpers shared by all modules. */

export function createBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBufferView,
): GPUBuffer {
  const aligned = Math.max(16, Math.ceil(size / 16) * 16);
  const buffer = device.createBuffer({ label, size: aligned, usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: !!data });
  if (data) {
    const dst = new Uint8Array(buffer.getMappedRange());
    dst.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
  }
  return buffer;
}

export function createStorageBuffer(device: GPUDevice, label: string, size: number, extraUsage: GPUBufferUsageFlags = 0, data?: ArrayBufferView) {
  return createBuffer(device, label, size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | extraUsage, data);
}

export function createUniformBuffer(device: GPUDevice, label: string, size: number) {
  return createBuffer(device, label, size, GPUBufferUsage.UNIFORM);
}

export function shaderModule(device: GPUDevice, label: string, code: string): GPUShaderModule {
  const module = device.createShaderModule({ label, code });
  // Surface compilation problems with readable line numbers (async; non-blocking).
  if ((module as any).getCompilationInfo) {
    module.getCompilationInfo().then((info) => {
      const msgs = info.messages.filter((m) => m.type === 'error' || m.type === 'warning');
      if (!msgs.length) return;
      const lines = code.split('\n');
      for (const m of msgs) {
        const ctx = lines.slice(Math.max(0, m.lineNum - 3), m.lineNum + 1).map((l, i) => `${Math.max(1, m.lineNum - 2) + i}: ${l}`).join('\n');
        const fn = m.type === 'error' ? console.error : console.warn;
        fn(`[wgsl-${m.type}] ${label}:${m.lineNum}:${m.linePos} ${m.message}\n${ctx}`);
      }
    });
  }
  return module;
}

export function computePipeline(
  device: GPUDevice,
  label: string,
  code: string,
  entryPoint = 'main',
  layout: GPUPipelineLayout | 'auto' = 'auto',
  constants?: Record<string, number>,
): GPUComputePipeline {
  return device.createComputePipeline({
    label,
    layout,
    compute: { module: shaderModule(device, label, code), entryPoint, constants },
  });
}

export const divUp = (n: number, d: number) => Math.ceil(n / d);

/** Records a compute pass with a single dispatch. */
export function dispatch(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  x: number,
  y = 1,
  z = 1,
  label?: string,
) {
  const pass = encoder.beginComputePass(label ? { label } : undefined);
  pass.setPipeline(pipeline);
  bindGroups.forEach((bg, i) => pass.setBindGroup(i, bg));
  pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
  pass.end();
}

/**
 * Writes typed fields into a CPU mirror of a uniform/storage struct.
 * Offsets are in floats (4 bytes). Keep layouts in sync with WGSL structs.
 */
export class StructWriter {
  readonly f32: Float32Array;
  readonly u32: Uint32Array;
  readonly i32: Int32Array;
  constructor(public readonly byteSize: number) {
    const buf = new ArrayBuffer(Math.ceil(byteSize / 16) * 16);
    this.f32 = new Float32Array(buf);
    this.u32 = new Uint32Array(buf);
    this.i32 = new Int32Array(buf);
  }
  setMat4(offsetFloats: number, m: ArrayLike<number>) {
    for (let i = 0; i < 16; i++) this.f32[offsetFloats + i] = m[i];
  }
  setVec4(offsetFloats: number, x: number, y = 0, z = 0, w = 0) {
    this.f32[offsetFloats] = x;
    this.f32[offsetFloats + 1] = y;
    this.f32[offsetFloats + 2] = z;
    this.f32[offsetFloats + 3] = w;
  }
  upload(device: GPUDevice, buffer: GPUBuffer, offsetBytes = 0) {
    device.queue.writeBuffer(buffer, offsetBytes, this.f32.buffer, 0, this.f32.byteLength);
  }
}

/** Ring of mappable staging buffers for asynchronous GPU→CPU readback without stalls. */
export class ReadbackRing {
  private ring: { buffer: GPUBuffer; busy: boolean }[] = [];
  constructor(private device: GPUDevice, public readonly size: number, count = 3, private label = 'readback') {
    for (let i = 0; i < count; i++) {
      this.ring.push({
        buffer: device.createBuffer({ label: `${label}-${i}`, size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        busy: false,
      });
    }
  }
  /**
   * Encodes a copy from `src` into a free staging buffer. Returns a callback to invoke
   * after `queue.submit`, which resolves with the data (or null if no buffer was free).
   */
  encodeCopy(encoder: GPUCommandEncoder, src: GPUBuffer, srcOffset = 0, size = this.size): (() => Promise<ArrayBuffer | null>) | null {
    const slot = this.ring.find((s) => !s.busy);
    if (!slot) return null;
    slot.busy = true;
    encoder.copyBufferToBuffer(src, srcOffset, slot.buffer, 0, size);
    return async () => {
      try {
        await slot.buffer.mapAsync(GPUMapMode.READ, 0, size);
        const copy = slot.buffer.getMappedRange(0, size).slice(0);
        slot.buffer.unmap();
        return copy;
      } catch {
        return null;
      } finally {
        slot.busy = false;
      }
    };
  }
  destroy() {
    this.ring.forEach((s) => s.buffer.destroy());
  }
}

/** Optional GPU timestamp profiler; no-ops when timestamp-query is unavailable. */
export class GpuProfiler {
  private querySet?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private readback?: ReadbackRing;
  private labels: string[] = [];
  private pending = false;
  readonly results = new Map<string, number>();
  constructor(private device: GPUDevice, enabled: boolean, private capacity = 32) {
    if (!enabled) return;
    this.querySet = device.createQuerySet({ type: 'timestamp', count: capacity * 2 });
    this.resolveBuf = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readback = new ReadbackRing(device, capacity * 16, 2, 'profiler');
  }
  get enabled() {
    return !!this.querySet;
  }
  beginFrame() {
    this.labels = [];
  }
  /** Returns timestampWrites for a pass descriptor, or undefined when disabled / full. */
  timestamp(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.querySet || this.pending || this.labels.length >= this.capacity) return undefined;
    const i = this.labels.length;
    this.labels.push(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 };
  }
  resolve(encoder: GPUCommandEncoder): (() => void) | null {
    if (!this.querySet || !this.labels.length || this.pending) return null;
    const n = this.labels.length;
    encoder.resolveQuerySet(this.querySet, 0, n * 2, this.resolveBuf!, 0);
    const read = this.readback!.encodeCopy(encoder, this.resolveBuf!, 0, n * 16);
    if (!read) return null;
    const labels = this.labels.slice();
    this.pending = true;
    return () => {
      read().then((data) => {
        this.pending = false;
        if (!data) return;
        const t = new BigInt64Array(data);
        const sums = new Map<string, number>();
        for (let i = 0; i < labels.length; i++) {
          const ms = Number(t[i * 2 + 1] - t[i * 2]) / 1e6;
          if (ms >= 0 && ms < 1000) sums.set(labels[i], (sums.get(labels[i]) ?? 0) + ms);
        }
        for (const [k, v] of sums) this.results.set(k, (this.results.get(k) ?? v) * 0.9 + v * 0.1);
      });
    };
  }
}
