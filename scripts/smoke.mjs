#!/usr/bin/env node
/**
 * Smoke test: loads every preset headlessly, simulates a few frames, captures a PNG per preset and
 * fails if any GPU validation / WGSL / JS error is reported.
 *
 *   node scripts/smoke.mjs [--frames 40] [--size 640x360] [--presets ledge,trickle]
 */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (n, d) => {
  const i = args.indexOf(`--${n}`);
  return i < 0 ? d : args[i + 1];
};
const src = readFileSync(resolve(root, 'src/terrain/presets.ts'), 'utf8');
const all = [...src.matchAll(/^\s{4}id:\s*'([^']+)'/gm)].map((m) => m[1]);
const presets = (opt('presets', '') || all.join(',')).split(',').filter(Boolean);
let failed = 0;
for (const p of presets) {
  const r = spawnSync(
    'node',
    ['scripts/shot.mjs', '--preset', p, '--frames', opt('frames', '40'), '--size', opt('size', '640x360'), '--out', `shots/smoke-${p}.png`],
    { cwd: root, encoding: 'utf8', timeout: 900000 },
  );
  let summary = null;
  try {
    summary = JSON.parse(r.stdout.slice(r.stdout.indexOf('{')));
  } catch {}
  const ok = r.status === 0;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${p}`, ok ? '' : JSON.stringify(summary?.errors ?? r.stderr?.slice(-2000)));
}
process.exit(failed ? 1 : 0);
