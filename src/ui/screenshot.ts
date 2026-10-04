/**
 * Screenshot of the rendered view (without editor overlays) as a PNG download.
 *  - Normal mode: the WebGPU canvas content is only readable in the same task that rendered
 *    it, so the capture runs in a requestAnimationFrame callback queued after the app's
 *    frame (the app's rAF callback was registered first) and retries until a new frame
 *    has been rendered.
 *  - Offscreen mode (automation): App.readOffscreenPng().
 */
import { overlayControl } from '../editor/protocol';

export interface ScreenshotHost {
  readonly frameIndex: number;
  readonly gpu: { offscreen: boolean; canvas: HTMLCanvasElement };
  readOffscreenPng(): Promise<string | null>;
  waitFrames(n: number): Promise<void>;
}

export function screenshotFileName(date = new Date()) {
  const p = (n: number) => String(n).padStart(2, '0');
  return `creekside-${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}.png`;
}

function canvasBlob(host: ScreenshotHost): Promise<Blob | null> {
  return new Promise((resolve) => {
    const start = host.frameIndex;
    let tries = 0;
    const attempt = () => {
      tries++;
      if (host.frameIndex === start && tries < 30) {
        requestAnimationFrame(attempt);
        return;
      }
      try {
        host.gpu.canvas.toBlob((b) => resolve(b), 'image/png');
      } catch {
        resolve(null);
      }
    };
    requestAnimationFrame(attempt);
  });
}

/** Captures the next frame without overlays. Resolves to an object URL / data URL, or null. */
export async function captureScreenshot(host: ScreenshotHost): Promise<string | null> {
  overlayControl.suppressed++;
  try {
    if (host.gpu.offscreen) {
      await Promise.race([host.waitFrames(1), new Promise((r) => setTimeout(r, 5000))]);
      return await host.readOffscreenPng();
    }
    const blob = await canvasBlob(host);
    return blob ? URL.createObjectURL(blob) : null;
  } finally {
    overlayControl.suppressed = Math.max(0, overlayControl.suppressed - 1);
  }
}

export async function downloadScreenshot(host: ScreenshotHost): Promise<boolean> {
  const url = await captureScreenshot(host);
  if (!url) return false;
  const a = document.createElement('a');
  a.href = url;
  a.download = screenshotFileName();
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (url.startsWith('blob:')) setTimeout(() => URL.revokeObjectURL(url), 10000);
  return true;
}
