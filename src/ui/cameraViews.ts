/**
 * Camera presets computed from the scene: the preset's own view plus views framed on the
 * main drop ("lip"), the plunge pool, an eye-level view just above the water and an
 * overview of the whole domain. Pure functions (tested) — the terrain is only queried
 * through a height function.
 */

export type V3 = [number, number, number];

export interface CameraView {
  id: 'preset' | 'lip' | 'pool' | 'eye' | 'overview';
  label: string;
  position: V3;
  target: V3;
}

export interface SceneFeatures {
  /** Stream centre line z used for sampling. */
  z: number;
  lipX: number;
  lipY: number;
  poolX: number;
  poolY: number;
  /** Height difference across the lip (m, >= 0). */
  drop: number;
  /** Estimated water surface height in the pool. */
  waterY: number;
}

export interface CameraViewInput {
  extent: { sizeX: number; sizeZ: number; minY: number; maxY: number };
  presetCamera: { position: readonly number[]; target: readonly number[] };
  heightAt: (x: number, z: number) => number;
  /** z of the main inflow (defaults to the domain centre line). */
  inflowZ?: number;
  /** Initial water regions of the preset (to estimate the pool water level). */
  initialWater?: readonly { x0: number; z0: number; x1: number; z1: number; level: number }[];
  /** Vertical field of view (rad) and aspect for framing the overview. */
  fovY?: number;
  aspect?: number;
}

const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);
const finite = (v: number, fb: number) => (Number.isFinite(v) ? v : fb);

/** Finds the steepest drop along the stream centre line and the pool below it. */
export function analyzeStream(inp: CameraViewInput): SceneFeatures {
  const { sizeX, sizeZ, minY } = inp.extent;
  const z = clamp(finite(inp.inflowZ ?? sizeZ / 2, sizeZ / 2), 0, sizeZ);
  const band = Math.min(0.08, sizeZ * 0.1);
  const h = (x: number) => {
    let s = 0;
    let n = 0;
    for (const dz of [-band, 0, band]) {
      const v = inp.heightAt(clamp(x, 0, sizeX), clamp(z + dz, 0, sizeZ));
      if (Number.isFinite(v)) {
        s += v;
        n++;
      }
    }
    return n ? s / n : minY;
  };
  const N = 160;
  const xs: number[] = [];
  const hs: number[] = [];
  for (let i = 0; i <= N; i++) {
    const x = sizeX * (0.04 + (0.92 * i) / N);
    xs.push(x);
    hs.push(h(x));
  }
  const win = Math.max(1, Math.round(N * 0.05));
  let best = 0;
  let bestDrop = -Infinity;
  for (let i = 0; i + win <= N; i++) {
    const d = hs[i] - hs[i + win];
    if (d > bestDrop) {
      bestDrop = d;
      best = i;
    }
  }
  // Lip = last high point before the steepest part of the window.
  let lipI = best;
  for (let i = best; i < best + win; i++) if (hs[i] - hs[i + 1] > (hs[best] - hs[best + win]) / (win * 2)) {
    lipI = i;
    break;
  }
  const lipX = xs[lipI];
  const lipY = hs[lipI];
  // Pool = lowest point within a reach downstream of the lip.
  let poolI = Math.min(N, lipI + 1);
  const reach = Math.min(N, lipI + Math.round(N * 0.4));
  for (let i = lipI + 1; i <= reach; i++) if (hs[i] < hs[poolI]) poolI = i;
  const poolX = xs[poolI];
  const poolY = hs[poolI];
  let waterY = poolY + 0.05;
  for (const w of inp.initialWater ?? []) {
    if (poolX >= Math.min(w.x0, w.x1) && poolX <= Math.max(w.x0, w.x1) && z >= Math.min(w.z0, w.z1) && z <= Math.max(w.z0, w.z1) && w.level > poolY) {
      waterY = Math.max(waterY, w.level);
    }
  }
  return { z, lipX, lipY, poolX, poolY, drop: Math.max(0, lipY - poolY), waterY };
}

function fromAngles(target: V3, dist: number, azimuth: number, elevation: number): V3 {
  // azimuth measured from +X (downstream) towards +Z
  return [
    target[0] + dist * Math.cos(elevation) * Math.cos(azimuth),
    target[1] + dist * Math.sin(elevation),
    target[2] + dist * Math.cos(elevation) * Math.sin(azimuth),
  ];
}

function keepAbove(p: V3, heightAt: (x: number, z: number) => number, ext: CameraViewInput['extent'], clearance: number): V3 {
  const x = clamp(p[0], -ext.sizeX, 2 * ext.sizeX);
  const z = clamp(p[2], -ext.sizeZ, 2 * ext.sizeZ);
  const g = heightAt(clamp(x, 0, ext.sizeX), clamp(z, 0, ext.sizeZ));
  const y = Number.isFinite(g) ? Math.max(p[1], g + clearance) : p[1];
  return [x, clamp(y, ext.minY, ext.maxY + 3 * Math.max(ext.sizeX, ext.sizeZ)), z];
}

export function computeCameraViews(inp: CameraViewInput): CameraView[] {
  const ext = inp.extent;
  const f = analyzeStream(inp);
  const H = inp.heightAt;
  const scale = clamp(f.drop * 2.2, 0.18, 0.7);
  const views: CameraView[] = [];
  const pc = inp.presetCamera;
  views.push({ id: 'preset', label: 'Preset view', position: [pc.position[0], pc.position[1], pc.position[2]], target: [pc.target[0], pc.target[1], pc.target[2]] });

  // Close-up on the lip: slightly downstream and to the side, looking back at the sheet as it parts.
  const lipT: V3 = [f.lipX + 0.01, f.lipY - Math.min(0.03, f.drop * 0.2), f.z];
  views.push({ id: 'lip', label: 'Close-up on the lip', position: keepAbove(fromAngles(lipT, scale * 0.9, 0.55, 0.42), H, ext, 0.03), target: lipT });

  // Plunge pool: from downstream, above, looking at the impact zone.
  const poolT: V3 = [Math.max(f.lipX + 0.04, f.poolX - 0.03), Math.max(f.poolY, f.waterY - 0.02), f.z];
  views.push({ id: 'pool', label: 'Plunge pool', position: keepAbove(fromAngles(poolT, scale * 1.5, 0.35, 0.62), H, ext, 0.05), target: poolT });

  // Eye level: a few cm above the water surface, downstream, looking up at the falls.
  const eyeX = clamp(f.lipX + Math.max(0.35, f.drop * 3), 0.05, ext.sizeX - 0.03);
  const eyeZ = clamp(f.z + 0.1, 0.02, ext.sizeZ - 0.02);
  const eyeY = Math.max(f.waterY + 0.025, finite(H(eyeX, eyeZ), f.poolY) + 0.03);
  const eyeT: V3 = [f.lipX, f.lipY - f.drop * 0.45, f.z];
  views.push({ id: 'eye', label: 'Eye level', position: [eyeX, eyeY, eyeZ], target: eyeT });

  // Overview: whole domain from the downstream side.
  const fov = clamp(inp.fovY ?? (50 * Math.PI) / 180, 0.2, 2.5);
  const aspect = clamp(inp.aspect ?? 16 / 9, 0.3, 4);
  let meanY = 0;
  let n = 0;
  for (let i = 1; i < 8; i++)
    for (let j = 1; j < 4; j++) {
      const v = H((ext.sizeX * i) / 8, (ext.sizeZ * j) / 4);
      if (Number.isFinite(v)) {
        meanY += v;
        n++;
      }
    }
  meanY = n ? meanY / n : (ext.minY + ext.maxY) / 2;
  const ovT: V3 = [ext.sizeX / 2, meanY, ext.sizeZ / 2];
  const radius = 0.5 * Math.hypot(ext.sizeX, ext.sizeZ);
  const halfFov = Math.min(fov / 2, Math.atan(Math.tan(fov / 2) * aspect));
  const dist = (radius / Math.sin(halfFov)) * 0.85;
  views.push({ id: 'overview', label: 'Overview', position: keepAbove(fromAngles(ovT, dist, 0.95, 0.75), H, ext, 0.1), target: ovT });
  return views.map((v) => ({ ...v, position: v.position.map((x, i) => finite(x, v.target[i] + 1)) as V3 }));
}
