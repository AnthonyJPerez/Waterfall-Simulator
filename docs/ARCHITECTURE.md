# Architecture

Creekside is a real-time, close-up simulator of small waterfalls, creek riffles and small rapids,
running entirely on the GPU with **WebGPU** (raw WGSL, no engine). Audio is synthesized procedurally
from simulation events with **Web Audio** (AudioWorklet).

## Physical model (hybrid)

Water is split into two coupled representations, chosen per region by the physics:

| Regime | Representation | Owner |
|---|---|---|
| Flow over gentle beds: riffles, rapids, pools, flow around/over rocks, standing waves, hydraulic jumps, eddies | **2-D shallow-water equations** on a heightfield grid (`h`, `hu`, `hv`) | `src/sim/swe` |
| Water that leaves the bed: over ledges/lips, down steep rock faces, jets, sheets that split into strands, trickles that cling to rock and curve, splashes | **3-D particles** (position-based fluid with cohesion + adhesion) | `src/sim/particles` |
| Sub-resolution detail: spray, mist, foam, entrained bubbles | diffuse particles + advected foam/aeration fields | particles / swe |

**Hand-off SWE → particles** (`world.overflow`): where the flow would detach from the bed (bed drops
faster than the ballistic trajectory / centripetal criterion `u²κ > g cosθ`, or the bed is steeper than
the SWE validity threshold), the SWE removes the outgoing flux and accumulates it per cell
(volume, momentum, lip surface height). The particle system emits particles from it.

**Hand-off particles → SWE** (`world.deposit`): particles that enter SWE water, or come to rest on a
gentle (SWE-valid) bed, are absorbed: volume, horizontal momentum and impact energy are atomically
added (fixed point) to the cell, and the SWE consumes and clears the buffer at its next step.
Impact energy drives foam, aeration (sub-surface white plume) and ripples.

**Obstacles** (rocks, logs) live in `SceneModel`. The terrain module bakes them into:
- `world.bed` (r32float heightfield = max(base terrain, obstacle tops)) — the SWE bed, so water flows
  around rocks or over them depending on depth;
- `world.sdf` (rgba16float 3-D: distance + outward normal) — particle collisions/adhesion and renderers.
Moving a rock re-bakes the affected region; both simulations react on the next step. Water displaced
by a moved rock is preserved (the SWE keeps depth), so it surges outward naturally.

## Units & conventions

- Metres, seconds, kilograms. Gravity 9.81 m/s². Water density 1000 kg/m³.
- Right-handed, **+Y up**, stream flows mainly **+X**. Domain: x ∈ [0, sizeX], z ∈ [0, sizeZ], y ∈ [minY, maxY].
- Heightfield cell (i, j) centre = ((i + .5)·cellSize, (j + .5)·cellSize); linear index `j * nx + i`.
- SDF voxel (a, b, c) centre = (0, minY, 0) + (a + .5, b + .5, c + .5)·sdfVoxel.
- **Reversed-Z** depth everywhere in the main view: `depth32float`, clear 0, compare `greater`.
  `linearDepth(d)` / `worldFromDepth(uv, d)` in `frameBindingsWGSL`. The sun shadow map uses standard Z
  (clear 1, compare `less`).
- HDR linear colour in `rgba16float`; tonemapping only in the post pass.

## Frame order (App.frame)

```
camera → World.updateUniforms → Renderer.updateFrameUniforms
encoder:
  editor.update
  terrain.update        (re-bake bed/SDF if scene changed; wetness)
  swe.step              (consume deposit, sub-step, write water/waterAux, accumulate overflow, stats)
  particles.step        (clear simEvents, emit from overflow, simulate, deposit, events, diffuse)
  audio.encodeReadback  (copy simEvents/sweStats to staging)
  renderer.render:
    env.update → shadow pass (terrain.drawShadow) → water.computeCaustics
    → opaque pass [hdr, normal] (terrain.drawOpaque, env.drawSky)
    → copy hdr→sceneColor, depth→opaqueDepth → water.draw → copy hdr→sceneColor
    → particlesRenderer.draw → env.post (→ swapchain) → editor.drawOverlay
submit → audio.afterSubmit
```

At most two frames are in flight (App frame pacing).

## Shared GPU resources (`src/world/World.ts`)

| Resource | Format | Producer | Content |
|---|---|---|---|
| `bed` | r32float [nx, nz] | terrain | bed height incl. obstacle tops (m) |
| `sdf` | rgba16float 3-D | terrain | (signed distance m, outward normal xyz) |
| `water` | rgba32float [nx, nz] | swe | (depth h, ux, uz, foam 0..1) |
| `waterAux` | rgba16float [nx, nz] | swe | (turbulence 0..1, aeration 0..1, surface divergence, unused) |
| `wetness` | rgba8unorm [nx, nz] | terrain | (wetness, splash wetness, -, -) |
| `caustics` | rgba16float [cnx, cnz] | water renderer | caustic irradiance multiplier on bed (1 = none) |
| `deposit` | atomic<i32>×4 / cell | particles | (volume ×1e12, mom x ×1e12, mom z ×1e12, energy ×1e9) |
| `overflow` | vec4f / cell | swe | (pending volume m³, mom x, mom z, lip surface y) — particles subtract what they emit |
| `simEvents` | `SimEvents` | particles | impact list for audio (cleared by particles at step start) |
| `sweStats` | `SweStats` | swe | totals + 16×8 tile stats for audio |
| `uniforms` | `WorldUniforms` | App | domain, flow params, physics constants, time, tuning |

WGSL declarations for all of these live in `src/world/wgsl.ts` (`worldUniformsWGSL`, `couplingWGSL`).

## Render bindings

All render pipelines use `@group(0)` = frame (`frameBindingsWGSL` in `src/render/frame.ts`:
FrameUniforms, WorldUniforms, samplers, bed/water/waterAux/sdf/wetness) and `@group(1)` = lighting
(`lightingBindingsWGSL`: env cube, shadow map + comparison sampler, caustics, env sampler).
Lighting *functions* (`skyRadiance`, `envRadiance`, `ambientIrradiance`, `sunIrradiance`,
`shadowFactor`, `canopyShade`, `causticsAt`, `sunVisibility`) are in `src/render/env/lighting.ts`;
their signatures are a contract. Module-specific bindings start at `@group(2)`.

Opaque pass targets: `hdr` (rgba16float) + `normal` (rgba16float: world normal xyz, roughness w).

## Module map

| Path | Contract (`src/app/modules.ts`) | Responsibility |
|---|---|---|
| `src/terrain/` + `src/render/terrain/` | `TerrainSystem` | presets, base terrain, rock/log shapes, bake bed + SDF, wetness, CPU raycast, terrain & rock rendering |
| `src/sim/swe/` | `WaterSim` | shallow-water solver, inflow/outflow, overflow detection, deposit consumption, foam/aeration/turbulence, stats |
| `src/sim/particles/` | `ParticleSim` | falling water (PBF + cohesion/adhesion), emission, absorption, diffuse spray/foam/bubbles, impact events |
| `src/render/water/` | `WaterRendererModule` | SWE surface shading (refraction, reflection, absorption, foam, flow-mapped ripples), caustics |
| `src/render/particles/` | `ParticleRendererModule` | screen-space fluid rendering of particles + diffuse particles |
| `src/render/env/` | `EnvRenderer` | sky, sun, env cube, shadows, canopy light, post (bloom, tonemap, AA, DoF) |
| `src/audio/` | `AudioModule` | bubble-resonance synthesis (AudioWorklet), roar/hiss layers, spatialization, ambience |
| `src/editor/`, `src/ui/` | `EditorModule` | rock add/move/rotate/scale/delete with gizmos, control panel |

`src/app/registry.ts` is the only place that picks implementations.

## Testing

- `npm test` — vitest unit tests (CPU reference implementations, math, mapping logic).
- `npm run typecheck`, `npm run build`.
- `node scripts/shot.mjs …` — headless Chromium with WebGPU on SwiftShader (CPU). Renders offscreen
  and writes PNGs; reports GPU validation / WGSL errors (exit code 1). It is slow (~1–5 fps at
  `quality=low`, 960×540), so keep frame counts modest. See the script header for options
  (`--preset`, `--frames`, `--cam`, `--set`, `--multi`, `--skip`, `--eval`).
