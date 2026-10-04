/**
 * Per-frame render state shared by every render module: the FrameUniforms
 * layout, the group(0) "frame" and group(1) "lighting" bind group layouts,
 * and the main render targets.
 *
 * Bind group conventions for ALL render pipelines:
 *   @group(0) frame     — see frameBindingsWGSL (uniforms, samplers, world textures)
 *   @group(1) lighting  — see lightingBindingsWGSL (env cube, shadow map, caustics), provided by the env module
 *   @group(2+)          — module specific
 *
 * Depth convention: REVERSED-Z. depth32float, clear to 0.0, depthCompare 'greater' (or 'greater-equal').
 * Use linearDepth() from frameBindingsWGSL to recover view distance.
 */
import { worldUniformsWGSL } from '../world/wgsl';

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
export const NORMAL_FORMAT: GPUTextureFormat = 'rgba16float';
export const DEPTH_FORMAT: GPUTextureFormat = 'depth32float';

/** Float offsets inside the FrameUniforms struct. */
export const FU = {
  viewProj: 0,
  invViewProj: 16,
  view: 32,
  proj: 48,
  invProj: 64,
  prevViewProj: 80,
  shadowViewProj: 96,
  cameraPos: 112,
  cameraFwd: 116,
  sunDir: 120,
  sunColor: 124,
  ambient: 128,
  screen: 132,
  time: 136,
  water: 140,
  waterAbsorb: 144,
  waterScatter: 148,
  post: 152,
  jitter: 156,
  invView: 160,
} as const;
export const FRAME_UNIFORMS_BYTES = 176 * 4;

export const frameUniformsWGSL = /* wgsl */ `
struct FrameUniforms {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  view: mat4x4f,
  proj: mat4x4f,
  invProj: mat4x4f,
  prevViewProj: mat4x4f,
  shadowViewProj: mat4x4f,
  cameraPos: vec4f,    // xyz, w = near plane (m)
  cameraFwd: vec4f,    // xyz forward unit vector, w = far plane (m)
  sunDir: vec4f,       // xyz unit vector TOWARDS the sun, w = sun angular radius (rad)
  sunColor: vec4f,     // rgb sun irradiance at the ground (HDR, cloud attenuated), w = canopy 0..1
  ambient: vec4f,      // rgb hemispherical sky irradiance, w = cloudiness 0..1
  screen: vec4f,       // width, height, 1/width, 1/height (render target pixels)
  time: vec4f,         // sim time (s), sim dt (s), real time (s), frame index
  water: vec4f,        // clarity 0..1, turbidity 0..1, foam mult, ripple mult
  waterAbsorb: vec4f,  // rgb absorption coefficient sigma_a (1/m), w = scattering coefficient sigma_s (1/m)
  waterScatter: vec4f, // rgb single-scattering albedo tint, w = index of refraction
  post: vec4f,         // exposure (linear multiplier), debug view id, dof on (0/1), aperture f-stop
  jitter: vec4f,       // subpixel jitter (ndc) xy, unused zw
  invView: mat4x4f,
}
`;

/** Debug view ids written to FrameUniforms.post.y. */
export const DEBUG_VIEW_IDS: Record<string, number> = {
  none: 0, depth: 1, velocity: 2, foam: 3, particles: 4, sdf: 5, caustics: 6, wetness: 7, normals: 8,
};

export const frameBindingsWGSL = /* wgsl */ `
${frameUniformsWGSL}
${worldUniformsWGSL}
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> world: WorldUniforms;
@group(0) @binding(2) var linearClamp: sampler;
@group(0) @binding(3) var linearRepeat: sampler;
@group(0) @binding(4) var nearestClamp: sampler;
@group(0) @binding(5) var bedTex: texture_2d<f32>;      // r32float (unfilterable: use textureLoad / bedHeight())
@group(0) @binding(6) var waterTex: texture_2d<f32>;    // rgba32float (h, ux, uz, foam) (unfilterable)
@group(0) @binding(7) var waterAuxTex: texture_2d<f32>; // rgba16float (turbulence, aeration, divergence, -)
@group(0) @binding(8) var sdfTex: texture_3d<f32>;      // rgba16float (distance, normal)
@group(0) @binding(9) var wetnessTex: texture_2d<f32>;  // rgba8unorm (wetness, splash wetness, -, -)

const PI: f32 = 3.14159265359;

/** Reversed-Z depth buffer value → positive view-space distance along the view axis (m). */
fn linearDepth(d: f32) -> f32 {
  let n = frame.cameraPos.w;
  let f = frame.cameraFwd.w;
  // perspectiveReverseZ: d = n * (f - z) / (z * (f - n))  ⇒  z = n * f / (d * (f - n) + n)
  return n * f / (d * (f - n) + n);
}
/** Reconstruct world position from uv (0..1, y down) and reversed-Z depth. */
fn worldFromDepth(uv: vec2f, d: f32) -> vec3f {
  let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
  let w = frame.invViewProj * ndc;
  return w.xyz / w.w;
}
fn waterLoad(c: vec2i) -> vec4f {
  let n = vec2i(i32(world.grid.x), i32(world.grid.y));
  return textureLoad(waterTex, clamp(c, vec2i(0), n - 1), 0);
}
/** Bilinear sample of the SWE state at world xz: (h, ux, uz, foam). */
fn waterSample(xz: vec2f) -> vec4f {
  let g = xz * world.grid.w - 0.5;
  let f = floor(g);
  let t = g - f;
  let c = vec2i(f);
  let a = mix(waterLoad(c), waterLoad(c + vec2i(1, 0)), t.x);
  let b = mix(waterLoad(c + vec2i(0, 1)), waterLoad(c + vec2i(1, 1)), t.x);
  return mix(a, b, t.y);
}
fn gridUv(xz: vec2f) -> vec2f { return xz / world.domainSize.xy; }
fn bedAt(c: vec2i) -> f32 {
  let n = vec2i(i32(world.grid.x), i32(world.grid.y));
  return textureLoad(bedTex, clamp(c, vec2i(0), n - 1), 0).r;
}
/** Bilinear bed height (m) at world xz (clamped to the domain). */
fn bedHeight(xz: vec2f) -> f32 {
  let g = xz * world.grid.w - 0.5;
  let f = floor(g);
  let t = g - f;
  let c = vec2i(f);
  let a = mix(bedAt(c), bedAt(c + vec2i(1, 0)), t.x);
  let b = mix(bedAt(c + vec2i(0, 1)), bedAt(c + vec2i(1, 1)), t.x);
  return mix(a, b, t.y);
}
/** SDF sample (distance m, outward normal). */
fn sampleSdf(p: vec3f) -> vec4f {
  let ext = world.sdfDims.xyz * world.sdfDims.w;
  let uvw = (p - vec3f(0.0, world.domainSize.z, 0.0)) / ext;
  let s = textureSampleLevel(sdfTex, linearClamp, clamp(uvw, vec3f(0.0), vec3f(1.0)), 0.0);
  let l = length(s.yzw);
  return vec4f(s.x, select(vec3f(0.0, 1.0, 0.0), s.yzw / l, l > 1e-5));
}
fn insideDomainXZ(xz: vec2f) -> bool {
  return all(xz >= vec2f(0.0)) && all(xz <= world.domainSize.xy);
}
`;

/** group(1) lighting bindings. Function implementations come from the env module (lightingFunctionsWGSL). */
export const lightingBindingsWGSL = /* wgsl */ `
@group(1) @binding(0) var envCube: texture_cube<f32>;        // prefiltered radiance, mip = roughness * (mips - 1)
@group(1) @binding(1) var shadowMap: texture_depth_2d;       // standard-Z orthographic sun shadow map
@group(1) @binding(2) var shadowSampler: sampler_comparison; // compare 'less-equal'
@group(1) @binding(3) var causticsTex: texture_2d<f32>;      // world.caustics (rgba16float, linear filterable), uv = xz / domainSize
@group(1) @binding(4) var envSampler: sampler;               // trilinear, clamp
`;

export interface RenderTargets {
  width: number;
  height: number;
  /** Linear HDR scene colour. */
  hdr: GPUTexture;
  hdrView: GPUTextureView;
  /** Opaque-pass world normal (xyz, signed) + roughness (w); zero for sky. */
  normal: GPUTexture;
  normalView: GPUTextureView;
  /** Reversed-Z depth. */
  depth: GPUTexture;
  depthView: GPUTextureView;
  /** Copy of `hdr` taken after the opaque pass (and refreshed after the water pass) for refraction. */
  sceneColor: GPUTexture;
  sceneColorView: GPUTextureView;
  /** Copy of the opaque-only depth (r32float, reversed-Z values) — valid during water & particle passes. */
  opaqueDepth: GPUTexture;
  opaqueDepthView: GPUTextureView;
}

export function createRenderTargets(device: GPUDevice, width: number, height: number): RenderTargets {
  const t = (label: string, format: GPUTextureFormat, usage: GPUTextureUsageFlags) =>
    device.createTexture({ label, size: [width, height], format, usage });
  const RA = GPUTextureUsage.RENDER_ATTACHMENT;
  const TB = GPUTextureUsage.TEXTURE_BINDING;
  const CS = GPUTextureUsage.COPY_SRC;
  const CD = GPUTextureUsage.COPY_DST;
  const SB = GPUTextureUsage.STORAGE_BINDING;
  const hdr = t('rt.hdr', HDR_FORMAT, RA | TB | CS | SB);
  const normal = t('rt.normal', NORMAL_FORMAT, RA | TB);
  const depth = t('rt.depth', DEPTH_FORMAT, RA | TB);
  const sceneColor = t('rt.sceneColor', HDR_FORMAT, TB | CD | RA);
  const opaqueDepth = t('rt.opaqueDepth', 'r32float', TB | RA | SB);
  return {
    width,
    height,
    hdr,
    hdrView: hdr.createView(),
    normal,
    normalView: normal.createView(),
    depth,
    depthView: depth.createView(),
    sceneColor,
    sceneColorView: sceneColor.createView(),
    opaqueDepth,
    opaqueDepthView: opaqueDepth.createView(),
  };
}

export function destroyRenderTargets(t: RenderTargets) {
  [t.hdr, t.normal, t.depth, t.sceneColor, t.opaqueDepth].forEach((x) => x.destroy());
}
