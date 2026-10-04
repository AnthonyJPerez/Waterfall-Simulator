/**
 * WebGPU device bootstrap. Owns the adapter/device/canvas context and
 * reports optional features that modules may take advantage of.
 */

export interface GpuFeatures {
  /** r32float / rgba32float textures can be sampled with a filtering sampler. */
  float32Filterable: boolean;
  /** GPU timestamp queries (profiling). */
  timestampQuery: boolean;
  /** Subgroup operations in WGSL. */
  subgroups: boolean;
  /** rg11b10ufloat can be a render attachment. */
  rg11b10Renderable: boolean;
}

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  features: GpuFeatures;
  /** Software rasterizer (e.g. SwiftShader) — callers may lower quality defaults. */
  isSoftware: boolean;
  /**
   * Offscreen mode (automation): the canvas context is NOT configured; frames are rendered into
   * `offscreenTarget` (rgba8unorm) and can be read back with readOffscreenPng().
   */
  offscreen: boolean;
}

export class WebGPUUnavailableError extends Error {}

export async function initGpu(canvas: HTMLCanvasElement, offscreen = false): Promise<GpuContext> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new WebGPUUnavailableError(
      'WebGPU is not available in this browser. Use a recent Chrome, Edge, Safari (26+) or Firefox (141+).',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGPUUnavailableError('No suitable GPU adapter found.');

  const wanted: GPUFeatureName[] = [];
  const has = (f: string) => adapter.features.has(f);
  if (has('float32-filterable')) wanted.push('float32-filterable');
  if (has('timestamp-query')) wanted.push('timestamp-query');
  if (has('rg11b10ufloat-renderable')) wanted.push('rg11b10ufloat-renderable');
  // Subgroups are not requested by default: WGSL that uses them must `enable subgroups;`
  // and modules must provide a fallback path.

  const limits: Record<string, number> = {};
  const want = (name: keyof GPUSupportedLimits, min: number) => {
    const v = adapter.limits[name] as number;
    if (typeof v === 'number') limits[name as string] = Math.max(min, Math.min(v, Number.MAX_SAFE_INTEGER));
  };
  // Ask for the adapter's maxima on limits that matter for large simulations.
  want('maxStorageBufferBindingSize', 128 << 20);
  want('maxBufferSize', 256 << 20);
  want('maxStorageBuffersPerShaderStage', 8);
  want('maxComputeWorkgroupStorageSize', 16384);
  want('maxComputeInvocationsPerWorkgroup', 256);
  want('maxComputeWorkgroupSizeX', 256);
  want('maxStorageTexturesPerShaderStage', 4);
  want('maxColorAttachmentBytesPerSample', 32);
  for (const k of Object.keys(limits)) {
    const v = adapter.limits[k as keyof GPUSupportedLimits] as number;
    limits[k] = v; // always request exactly the adapter's limit
  }

  const device = await adapter.requestDevice({ requiredFeatures: wanted, requiredLimits: limits });
  device.lost.then((info) => {
    console.error(`[gpu] device lost: ${info.reason} ${info.message}`);
    (window as any).__wfDeviceLost = info;
  });
  device.addEventListener('uncapturederror', (ev: Event) => {
    const err = (ev as GPUUncapturedErrorEvent).error;
    console.error(`[gpu-error] ${err.constructor.name}: ${err.message}`);
  });

  const context = canvas.getContext('webgpu');
  if (!context) throw new WebGPUUnavailableError('Could not create a WebGPU canvas context.');
  const presentationFormat: GPUTextureFormat = offscreen ? 'rgba8unorm' : navigator.gpu.getPreferredCanvasFormat();
  if (!offscreen) context.configure({ device, format: presentationFormat, alphaMode: 'opaque' });

  const info = (adapter as any).info ?? {};
  const isSoftware =
    String(info.architecture ?? '').toLowerCase().includes('swiftshader') ||
    String(info.vendor ?? '').toLowerCase() === 'google' && String(info.architecture ?? '') === 'swiftshader' ||
    (adapter as any).isFallbackAdapter === true;

  return {
    adapter,
    device,
    canvas,
    context,
    presentationFormat,
    isSoftware,
    offscreen,
    features: {
      float32Filterable: device.features.has('float32-filterable'),
      timestampQuery: device.features.has('timestamp-query'),
      subgroups: device.features.has('subgroups'),
      rg11b10Renderable: device.features.has('rg11b10ufloat-renderable'),
    },
  };
}

/** Reads an rgba8unorm texture back and encodes it as a PNG data URL (automation / screenshots). */
export async function readTexturePng(device: GPUDevice, texture: GPUTexture): Promise<string> {
  const w = texture.width;
  const h = texture.height;
  const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) pixels.set(src.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
  buf.unmap();
  buf.destroy();
  for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d')!.putImageData(new ImageData(pixels, w, h), 0, 0);
  return c.toDataURL('image/png');
}
