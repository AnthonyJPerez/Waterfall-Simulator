# Creekside — Waterfall Simulator

Real-time WebGPU simulator of small waterfalls, creeks and rapids with procedural audio.
Read `docs/ARCHITECTURE.md` before changing anything that crosses a module boundary.

## Commands
- `npm run dev` — Vite dev server (http://127.0.0.1:5173)
- `npm test` — vitest unit tests; `npm run typecheck`; `npm run build`
- `node scripts/shot.mjs --frames 90 --out shots/x.png [--preset id] [--cam x,y,z,tx,ty,tz] [--set "flow.rate:5"]`
  headless WebGPU (SwiftShader) capture; non-zero exit on GPU/WGSL/JS errors. Slow: use `quality=low`
  (default) and ≤ 200 frames. Look at the PNG to judge visuals. `--multi` captures several views in one run.

## Hard conventions
- Units metres/seconds; +Y up; stream flows +X; domain x∈[0,sizeX], z∈[0,sizeZ].
- Reversed-Z for the main camera (depth32float, clear 0, compare `greater`); shadow map is standard Z.
- Render pipelines: `@group(0)` = `frameBindingsWGSL`, `@group(1)` = `lightingBindingsWGSL`
  (+ `lightingFunctionsWGSL`), module bindings from `@group(2)`. Don't change those layouts or the
  lighting function signatures without updating every user.
- Shared GPU resources and their formats/semantics are defined in `src/world/World.ts` and
  `src/world/wgsl.ts`; module contracts in `src/app/modules.ts`. Treat them as APIs.
- Only `src/app/registry.ts` chooses module implementations.
- WGSL lives in TS template strings (`/* wgsl */` tagged) next to the code that uses it.
- Must stay robust: no NaNs/explosions for any slider value, any obstacle placement, any frame dt
  (0 … 1/30 s × timeScale). Clamp, guard divisions, sanitize state on the GPU.
- Performance target: 60 fps at `medium` quality on a mid-range discrete GPU; keep work proportional
  to *active* content (e.g. indirect dispatch over live particles), because the software test GPU is slow.
