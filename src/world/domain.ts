/**
 * Simulation domain: physical extent plus grid resolutions derived from the
 * quality setting. Everything that is resolution dependent derives from here.
 */
import type { Quality } from '../app/params';

export interface DomainExtent {
  /** Length along the stream (x), metres. */
  sizeX: number;
  /** Width across the stream (z), metres. */
  sizeZ: number;
  /** Lowest / highest world Y the simulation must represent (m). */
  minY: number;
  maxY: number;
}

export interface Domain extends DomainExtent {
  quality: Quality;
  /** Heightfield / shallow-water cell size (m). Cells are square. */
  cellSize: number;
  /** Heightfield grid dimensions (cells). Cell (i, j) centre = ((i + .5) * cellSize, (j + .5) * cellSize). */
  nx: number;
  nz: number;
  /** 3-D signed-distance-field voxel size (m) and dimensions. Voxel (a, b, c) centre = (minX + (a + .5) * sdfVoxel, minY + (b + .5) * sdfVoxel, ...). */
  sdfVoxel: number;
  sdfNx: number;
  sdfNy: number;
  sdfNz: number;
  /** Falling-water particle radius (m) and capacity. */
  particleRadius: number;
  maxParticles: number;
  maxDiffuseParticles: number;
}

export const QUALITY_SETTINGS: Record<Quality, { cellSize: number; sdfVoxel: number; particleRadius: number; maxParticles: number; maxDiffuse: number }> = {
  low: { cellSize: 0.008, sdfVoxel: 0.016, particleRadius: 0.004, maxParticles: 32768, maxDiffuse: 16384 },
  medium: { cellSize: 0.006, sdfVoxel: 0.012, particleRadius: 0.003, maxParticles: 98304, maxDiffuse: 49152 },
  high: { cellSize: 0.004, sdfVoxel: 0.008, particleRadius: 0.0025, maxParticles: 196608, maxDiffuse: 98304 },
  ultra: { cellSize: 0.003, sdfVoxel: 0.006, particleRadius: 0.002, maxParticles: 393216, maxDiffuse: 196608 },
};

export function makeDomain(extent: DomainExtent, quality: Quality): Domain {
  const q = QUALITY_SETTINGS[quality];
  const nx = Math.max(16, Math.round(extent.sizeX / q.cellSize));
  const nz = Math.max(16, Math.round(extent.sizeZ / q.cellSize));
  const cellSize = extent.sizeX / nx;
  // Keep cells square: adjust sizeZ to an exact multiple of cellSize.
  const sizeZ = nz * cellSize;
  const sdfNx = Math.ceil(extent.sizeX / q.sdfVoxel);
  const sdfNy = Math.ceil((extent.maxY - extent.minY) / q.sdfVoxel);
  const sdfNz = Math.ceil(sizeZ / q.sdfVoxel);
  return {
    ...extent,
    sizeZ,
    quality,
    cellSize,
    nx,
    nz,
    sdfVoxel: q.sdfVoxel,
    sdfNx,
    sdfNy,
    sdfNz,
    particleRadius: q.particleRadius,
    maxParticles: q.maxParticles,
    maxDiffuseParticles: q.maxDiffuse,
  };
}
