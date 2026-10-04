import { App } from './app/App';
import type { Quality } from './app/params';
import { WebGPUUnavailableError } from './gpu/device';

/**
 * URL parameters (also used by scripts/shot.mjs):
 *   preset=<id>        scene preset
 *   quality=low|medium|high|ultra
 *   fixedDt=<seconds>  deterministic simulation step per frame
 *   cam=x,y,z,tx,ty,tz camera pose
 *   set=path:value;... parameter overrides, e.g. set=flow.rate:5;debug.view:velocity
 *   noui=1             hide the control panel
 *   offscreen=1        render offscreen (headless automation; window.__wf.capture() returns a PNG data URL)
 *   skip=a,b           debug: skip modules/passes (terrain,swe,particles,audio,env,shadow,caustics,terrainDraw,sky,water,particlesDraw,editor)
 */
async function main() {
  const q = new URLSearchParams(location.search);
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const ui = document.getElementById('ui');
  const app = new App(canvas, ui, {
    fixedDt: q.has('fixedDt') ? Number(q.get('fixedDt')) : undefined,
    noUi: q.get('noui') === '1',
    offscreen: q.get('offscreen') === '1',
  });
  (window as any).__wf = { app, ready: false };
  try {
    await app.init({ preset: q.get('preset') ?? undefined, quality: (q.get('quality') as Quality) ?? undefined });
  } catch (e) {
    const el = document.getElementById('error')!;
    el.style.display = 'flex';
    el.querySelector('p')!.textContent =
      e instanceof WebGPUUnavailableError ? e.message : `Failed to start: ${(e as Error)?.message ?? e}`;
    console.error(e);
    return;
  }
  const cam = q.get('cam');
  if (cam) {
    const v = cam.split(',').map(Number);
    if (v.length === 6 && v.every(Number.isFinite)) app.orbit.setPose([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
  }
  const set = q.get('set');
  if (set) {
    for (const kv of set.split(';')) {
      const [path, raw] = kv.split(':');
      if (!path || raw === undefined) continue;
      const cur = app.params.get(path);
      const val = typeof cur === 'number' ? Number(raw) : typeof cur === 'boolean' ? raw === 'true' || raw === '1' : raw;
      app.params.set(path, val);
    }
  }
  app.start();
  Object.assign((window as any).__wf, {
    ready: true,
    setParam: (path: string, v: unknown) => app.params.set(path, v),
    waitFrames: (n: number) => app.waitFrames(n),
    capture: () => app.readOffscreenPng(),
    setCamera: (pos: [number, number, number], target: [number, number, number]) => app.orbit.setPose(pos, target),
    stats: () => ({ swe: app.modules.swe.stats, particles: app.modules.particles.stats, frame: app.frameIndex, simTime: app.simTime }),
  });
  document.getElementById('hint')?.classList.add('show');
}

main();
