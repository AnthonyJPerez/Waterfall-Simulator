/**
 * Small DOM-event protocol between the editor (recreated on every world rebuild) and the
 * persistent UI (panel, toolbar, hint line). Both sides talk through CustomEvents on the
 * canvas, so neither holds a reference to the other.
 */
import type { EditorTool, ObstacleKind } from '../app/params';

/** editor → UI: detail = EditorStatus */
export const EDITOR_STATUS_EVENT = 'wf-editor-status';
/** UI → editor: detail = { cmd: EditorCommand } */
export const EDITOR_COMMAND_EVENT = 'wf-editor-command';

export type EditorCommand =
  | 'undo'
  | 'redo'
  | 'delete'
  | 'duplicate'
  | 'deselect'
  | 'focus'
  | 'reroll'
  /** Ask the editor to re-broadcast its status. */
  | 'status';

export interface EditorStatus {
  tool: EditorTool;
  addKind: ObstacleKind;
  addSize: number;
  selected: { id: number; kind: ObstacleKind; size: number } | null;
  hoveredId: number | null;
  dragging: 'none' | 'slide' | 'lift';
  canUndo: boolean;
  canRedo: boolean;
  undoLabel?: string;
  redoLabel?: string;
  obstacleCount: number;
  /** Context-sensitive one-line help for the current tool / state. */
  hint: string;
}

export function sendEditorCommand(target: EventTarget, cmd: EditorCommand) {
  target.dispatchEvent(new CustomEvent(EDITOR_COMMAND_EVENT, { detail: { cmd } }));
}

/** While > 0 the editor draws no overlay (clean screenshots). */
export const overlayControl = { suppressed: 0 };

/** Size range (m) of new obstacles (params.editor.addSize). */
export const ADD_SIZE_RANGE: [number, number] = [0.02, 0.5];

export const isMac = () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
export const modKey = () => (isMac() ? '⌘' : 'Ctrl+');
