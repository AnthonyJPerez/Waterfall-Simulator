/**
 * Obstacle editing: add / select / move / rotate / scale / delete rocks and logs.
 *
 * STUB IMPLEMENTATION — click-to-add and click-to-delete only, no overlay.
 * To be replaced by the editor agent (keep the EditorModule API).
 */
import type { EditorModule, FrameContext, ModuleContext, TerrainSystem } from '../app/modules';
import type { Camera, OrbitController } from '../core/camera';

export class StubEditor implements EditorModule {
  private onPointerDown = (e: PointerEvent) => {
    const tool = this.ctx.params.values.editor.tool;
    if (e.button !== 0 || tool === 'orbit') return;
    const rect = this.canvas.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = 1 - ((e.clientY - rect.top) / rect.height) * 2;
    const ray = this.camera.rayFromNdc(nx, ny);
    const hit = this.terrain.raycast(ray.origin, ray.dir);
    if (!hit) return;
    (e as any).__consumed = true;
    const ed = this.ctx.params.values.editor;
    if (tool === 'add') {
      const s = ed.addSize;
      this.ctx.scene.add({
        kind: ed.addKind,
        position: [hit.point[0], hit.point[1] - s * 0.25, hit.point[2]],
        rotation: [0, 0, 0, 1],
        scale: [s, s * 0.7, s * 0.85],
        seed: Math.floor(Math.random() * 1e6),
        roughness: 0.18,
      });
    } else if (tool === 'delete' && hit.obstacleId !== undefined) {
      this.ctx.scene.remove(hit.obstacleId);
    }
  };

  constructor(
    private ctx: ModuleContext,
    private canvas: HTMLCanvasElement,
    private camera: Camera,
    private orbit: OrbitController,
    private terrain: TerrainSystem,
  ) {
    canvas.addEventListener('pointerdown', this.onPointerDown, { capture: true });
  }

  update(_frame: FrameContext) {}
  drawOverlay(_encoder: GPUCommandEncoder, _frame: FrameContext, _target: GPUTextureView) {}
  destroy() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown, { capture: true });
  }
}
