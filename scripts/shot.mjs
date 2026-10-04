#!/usr/bin/env node
/**
 * Headless capture harness (WebGPU via SwiftShader in headless Chromium).
 *
 *   node scripts/shot.mjs [options]
 *
 * Options:
 *   --preset <id>            scene preset (default: app default)
 *   --quality <q>            low|medium|high|ultra (default: low — software GPU is slow)
 *   --frames <n>             frames to simulate before the capture (default 90)
 *   --fixedDt <s>            fixed sim dt per frame (default 0.0166667)
 *   --cam x,y,z,tx,ty,tz     camera pose
 *   --set "a.b:v;c.d:w"      parameter overrides (see src/main.ts)
 *   --out <file.png>         output path (default shots/shot.png)
 *   --size WxH               viewport (default 960x540)
 *   --eval "<js>"            JS evaluated in the page (with `wf` = window.__wf, awaited) before waiting frames
 *   --multi "<spec>"         several captures in one session: "frames@x,y,z,tx,ty,tz@out.png|frames@...|..."
 *                            (frames are cumulative waits; camera optional — use "-" to keep)
 *   --url <base>             use an already running dev server instead of starting one
 *   --timeout <ms>           overall timeout (default 600000)
 *   --ui                     show the control panel (hidden by default; not visible in offscreen captures)
 *   --skip a,b               debug: skip modules/passes (see src/main.ts)
 *
 * Prints a JSON summary (errors, warnings, stats) to stdout. Exit code 1 if GPU/WGSL/JS errors occurred.
 */
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = args[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
};

const quality = opt('quality', 'low');
const frames = Number(opt('frames', 90));
const fixedDt = Number(opt('fixedDt', 1 / 60));
const out = resolve(root, opt('out', 'shots/shot.png'));
const [vw, vh] = String(opt('size', '960x540')).split('x').map(Number);
const timeout = Number(opt('timeout', 600000));

let server;
let base = opt('url', null);
if (!base) {
  const { createServer } = await import('vite');
  server = await createServer({ root, logLevel: 'error', server: { port: 0, host: '127.0.0.1' } });
  await server.listen();
  const addr = server.httpServer.address();
  base = `http://127.0.0.1:${addr.port}/`;
}

const qs = new URLSearchParams();
qs.set('quality', quality);
qs.set('fixedDt', String(fixedDt));
// Headless Chromium cannot present WebGPU to a canvas here, so render offscreen and read back.
qs.set('offscreen', '1');
if (!opt('ui', false)) qs.set('noui', '1');
for (const k of ['preset', 'cam', 'set', 'skip']) {
  const v = opt(k, null);
  if (v) qs.set(k, v);
}

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface', '--autoplay-policy=no-user-gesture-required'],
});
const killer = setTimeout(async () => {
  console.error('TIMEOUT');
  await browser.close().catch(() => {});
  await server?.close();
  process.exit(2);
}, timeout);

const page = await browser.newPage({ viewport: { width: vw, height: vh } });
const errors = [];
const warnings = [];
const logs = [];
page.on('console', (m) => {
  const t = m.text();
  if (m.type() === 'error') errors.push(t);
  else if (m.type() === 'warning') warnings.push(t);
  else logs.push(t);
});
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('crash', () => errors.push('page crashed'));
page.on('framenavigated', (f) => { if (f === page.mainFrame()) logs.push('navigated: ' + f.url()); });

const url = `${base}?${qs.toString()}`;
await page.goto(url);
await page.waitForFunction(() => window.__wf && (window.__wf.ready || document.getElementById('error')?.style.display === 'flex'), null, {
  timeout: 120000,
});
const failed = await page.evaluate(() => !window.__wf.ready);
if (failed) errors.push('app failed to initialise: ' + (await page.evaluate(() => document.querySelector('#error p')?.textContent)));

const evalJs = opt('eval', null);
if (!failed && evalJs) {
  await page.evaluate(`(async () => { const wf = window.__wf; ${evalJs} })()`);
}

const captures = [];
if (!failed) {
  const multi = opt('multi', null);
  const plan = multi
    ? String(multi)
        .split('|')
        .map((s) => {
          const [f, cam, file] = s.split('@');
          return { frames: Number(f), cam: cam && cam !== '-' ? cam.split(',').map(Number) : null, out: resolve(root, file) };
        })
    : [{ frames, cam: null, out }];
  for (const step of plan) {
    if (step.cam) await page.evaluate((c) => window.__wf.setCamera([c[0], c[1], c[2]], [c[3], c[4], c[5]]), step.cam);
    const t0 = Date.now();
    await page.evaluate((n) => window.__wf.waitFrames(n), Math.max(1, step.frames));
    mkdirSync(dirname(step.out), { recursive: true });
    const dataUrl = await page.evaluate(() => window.__wf.capture());
    if (dataUrl) writeFileSync(step.out, Buffer.from(dataUrl.split(',')[1], 'base64'));
    else errors.push('capture failed (no offscreen frame)');
    captures.push({ out: step.out, frames: step.frames, seconds: (Date.now() - t0) / 1000 });
  }
}

const stats = failed ? null : await page.evaluate(() => window.__wf.stats());
const gpuErrors = errors.filter((e) => /gpu-error|wgsl-error|device lost|pageerror|\[frame\]/i.test(e));
const summary = { url, captures, stats, errors, warnings: warnings.slice(0, 30), logs: logs.slice(-30) };
console.log(JSON.stringify(summary, null, 2));
clearTimeout(killer);
await browser.close();
await server?.close();
process.exit(gpuErrors.length || failed ? 1 : 0);
