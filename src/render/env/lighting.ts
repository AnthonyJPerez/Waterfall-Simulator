/**
 * Lighting functions available to every lit shader (requires frameBindingsWGSL +
 * lightingBindingsWGSL in scope). The env module owns these implementations;
 * the SIGNATURES are a contract used by terrain, water and particle shaders.
 */

export const lightingFunctionsWGSL = /* wgsl */ `
/** Analytic sky radiance along a world direction (HDR). Used for backgrounds and as reference. */
fn skyRadiance(dir: vec3f) -> vec3f {
  let d = normalize(dir);
  let sun = frame.sunDir.xyz;
  let h = clamp(d.y, -1.0, 1.0);
  let zenith = vec3f(0.18, 0.32, 0.62);
  let horizon = vec3f(0.62, 0.70, 0.78);
  var c = mix(horizon, zenith, pow(max(h, 0.0), 0.45));
  c = mix(c, vec3f(0.22, 0.20, 0.17), smoothstep(0.0, -0.25, h)); // ground bounce below horizon
  let mu = max(dot(d, sun), 0.0);
  c += vec3f(1.0, 0.85, 0.6) * (0.25 * pow(mu, 8.0) + 0.6 * pow(mu, 64.0));
  let overcast = vec3f(0.55, 0.58, 0.6) * (0.6 + 0.4 * max(h, 0.0));
  return mix(c, overcast, frame.ambient.w) * 3.0;
}

/** Prefiltered environment radiance for reflections. roughness in [0, 1]. */
fn envRadiance(dir: vec3f, roughness: f32) -> vec3f {
  let mips = f32(textureNumLevels(envCube));
  return textureSampleLevel(envCube, envSampler, dir, roughness * (mips - 1.0)).rgb;
}

/** Diffuse irradiance from the sky hemisphere around normal n (cosine weighted, / PI applied by caller as needed). */
fn ambientIrradiance(n: vec3f) -> vec3f {
  let up = 0.5 + 0.5 * n.y;
  return frame.ambient.rgb * mix(0.35, 1.0, up);
}

/** Sun irradiance (HDR rgb) reaching worldPos, excluding shadowing. */
fn sunIrradiance() -> vec3f { return frame.sunColor.rgb; }

/** Directional shadow visibility 0..1 for the sun at worldPos (normal used for slope bias). */
fn shadowFactor(worldPos: vec3f, n: vec3f) -> f32 {
  let p = frame.shadowViewProj * vec4f(worldPos + n * 0.004, 1.0);
  let uv = p.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0) { return 1.0; }
  return textureSampleCompareLevel(shadowMap, shadowSampler, uv, p.z - 0.0015);
}

/** Dappled forest-canopy light modulation 0..1 at worldPos (multiplies sun). */
fn canopyShade(worldPos: vec3f) -> f32 {
  return 1.0;
}

/** Caustic irradiance multiplier on the bed at worldPos (1 = no caustics). */
fn causticsAt(worldPos: vec3f) -> vec3f {
  let uv = worldPos.xz / world.domainSize.xy;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec3f(1.0); }
  return textureSampleLevel(causticsTex, envSampler, uv, 0.0).rgb;
}

/** Total direct sun light factor at a point: shadow * canopy. */
fn sunVisibility(worldPos: vec3f, n: vec3f) -> f32 {
  return shadowFactor(worldPos, n) * canopyShade(worldPos);
}
`;
