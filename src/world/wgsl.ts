/**
 * WGSL declarations shared across modules. These strings are concatenated into
 * shader sources; keep the TS-side layouts (World.ts) in sync.
 */

/** Fixed-point scales for atomic accumulation in the particle→SWE deposit buffer. */
export const DEPOSIT_VOLUME_SCALE = 1e12; // 1 unit = 1e-12 m³
export const DEPOSIT_MOMENTUM_SCALE = 1e12; // 1 unit = 1e-12 m⁴/s (volume × velocity)
export const DEPOSIT_ENERGY_SCALE = 1e9; // 1 unit = 1e-9 J (kinetic energy delivered)

export const MAX_IMPACT_EVENTS = 4096;
export const STAT_TILES_X = 16;
export const STAT_TILES_Z = 8;

/** Byte sizes of the coupling buffers (see World.ts). */
export const IMPACT_EVENT_BYTES = 32;
export const SIM_EVENTS_HEADER_BYTES = 16;
export const SIM_EVENTS_BYTES = SIM_EVENTS_HEADER_BYTES + MAX_IMPACT_EVENTS * IMPACT_EVENT_BYTES;
export const TILE_STAT_BYTES = 32;
export const SWE_STATS_HEADER_BYTES = 32;
export const SWE_STATS_BYTES = SWE_STATS_HEADER_BYTES + STAT_TILES_X * STAT_TILES_Z * TILE_STAT_BYTES;

export const WORLD_UNIFORMS_BYTES = 8 * 16;

export const worldUniformsWGSL = /* wgsl */ `
struct WorldUniforms {
  domainSize: vec4f, // sizeX, sizeZ, minY, maxY (m)
  grid: vec4f,       // nx, nz, cellSize (m), 1/cellSize
  sdfDims: vec4f,    // sdfNx, sdfNy, sdfNz, sdfVoxel (m)
  flow: vec4f,       // main inflow rate (m^3/s), inflow velocity (m/s), turbulence 0..1, Manning n
  physics: vec4f,    // gravity (m/s^2, positive), water density (kg/m^3), kinematic viscosity (m^2/s), surface tension (N/m)
  time: vec4f,       // sim time (s), frame dt (sim s), frame index, timeScale
  tuning: vec4f,     // cohesion mult, adhesion mult, foam mult, ripple mult
  particle: vec4f,   // particle radius (m), particle volume (m^3), maxParticles, maxDiffuseParticles
}
`;

export const couplingWGSL = /* wgsl */ `
const DEPOSIT_VOLUME_SCALE: f32 = ${DEPOSIT_VOLUME_SCALE.toExponential()};
const DEPOSIT_MOMENTUM_SCALE: f32 = ${DEPOSIT_MOMENTUM_SCALE.toExponential()};
const DEPOSIT_ENERGY_SCALE: f32 = ${DEPOSIT_ENERGY_SCALE.toExponential()};
const MAX_IMPACT_EVENTS: u32 = ${MAX_IMPACT_EVENTS}u;
const STAT_TILES_X: u32 = ${STAT_TILES_X}u;
const STAT_TILES_Z: u32 = ${STAT_TILES_Z}u;

// Hand-off thresholds shared by the SWE and particle modules (hysteresis: detach > absorb).
// Bed gradient magnitude (rise/run) above which SWE outflow is diverted to particles.
const COUPLING_DETACH_SLOPE: f32 = 1.0;
// Particles may only come to rest on / be absorbed into beds flatter than this gradient.
const COUPLING_ABSORB_SLOPE: f32 = 0.6;
// SWE depth (m) above which a particle entering the water column is absorbed.
const COUPLING_ABSORB_MIN_DEPTH: f32 = 0.002;

// Impact kinds
const IMPACT_POOL: u32 = 0u;   // falling water entering standing/flowing water
const IMPACT_SOLID: u32 = 1u;  // falling water striking rock / bed
const IMPACT_DROP: u32 = 2u;   // small droplet / spray re-entering water

struct ImpactEvent {
  posKind: vec4f, // world xyz, kind (as f32)
  velVol: vec4f,  // impact velocity xyz (m/s), represented volume (m^3)
}

// Particle → audio. 'count' is an append counter (may exceed MAX_IMPACT_EVENTS; clamp on read).
struct SimEvents {
  count: atomic<u32>,
  droppedCount: atomic<u32>,
  // Fixed-point (DEPOSIT_VOLUME_SCALE) total volume of impacts not recorded because the list was full.
  droppedVolume: atomic<u32>,
  _pad: u32,
  events: array<ImpactEvent, ${MAX_IMPACT_EVENTS}>,
}

// SWE → audio / renderer. One tile = (nx/STAT_TILES_X) x (nz/STAT_TILES_Z) cells.
struct TileStat {
  centroid: vec4f, // energy-weighted world xyz of the tile's activity, w = wetted area (m^2)
  flow: vec4f,     // mean speed (m/s), turbulence / entrainment intensity (>=0), hydraulic jump intensity (>=0), mean depth (m)
}

struct SweStats {
  totalVolume: f32,     // water volume in the SWE grid (m^3)
  inflowRate: f32,      // m^3/s entering this frame
  outflowRate: f32,     // m^3/s leaving through open boundaries
  overflowRate: f32,    // m^3/s converted to particles (falling water)
  depositRate: f32,     // m^3/s absorbed back from particles
  maxSpeed: f32,        // m/s
  substeps: f32,
  _pad: f32,
  tiles: array<TileStat, ${STAT_TILES_X * STAT_TILES_Z}>,
}
`;

/** Helpers for sampling the shared heightfield / SDF. Requires bindings named as below. */
export const worldSamplingWGSL = /* wgsl */ `
// Expects in scope:
//   var<uniform> world: WorldUniforms;
//   var bedTex: texture_2d<f32>;        // r32float bed height
//   var sdfTex: texture_3d<f32>;        // rgba16float (distance, normal.xyz)
//   var linearClamp: sampler;
fn gridCoord(xz: vec2f) -> vec2f { return xz * world.grid.w - 0.5; } // continuous cell coords (cell centres at integers)
fn bedAt(c: vec2i) -> f32 {
  let n = vec2i(i32(world.grid.x), i32(world.grid.y));
  return textureLoad(bedTex, clamp(c, vec2i(0), n - 1), 0).r;
}
fn bedHeight(xz: vec2f) -> f32 {
  let g = gridCoord(xz);
  let f = floor(g);
  let t = g - f;
  let c = vec2i(f);
  let a = mix(bedAt(c), bedAt(c + vec2i(1, 0)), t.x);
  let b = mix(bedAt(c + vec2i(0, 1)), bedAt(c + vec2i(1, 1)), t.x);
  return mix(a, b, t.y);
}
fn sdfUvw(p: vec3f) -> vec3f {
  let ext = world.sdfDims.xyz * world.sdfDims.w;
  return (p - vec3f(0.0, world.domainSize.z, 0.0)) / ext;
}
/** Signed distance (m, negative inside solid) and outward unit normal. Outside the volume: distance to bed plane approximation. */
fn sampleSdf(p: vec3f) -> vec4f {
  let uvw = sdfUvw(p);
  let s = textureSampleLevel(sdfTex, linearClamp, clamp(uvw, vec3f(0.0), vec3f(1.0)), 0.0);
  let n = s.yzw;
  let l = length(n);
  return vec4f(s.x, select(vec3f(0.0, 1.0, 0.0), n / l, l > 1e-5));
}
`;
