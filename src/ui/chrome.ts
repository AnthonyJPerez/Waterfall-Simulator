/**
 * On-canvas UI chrome around the control panel: tool bar (tools, object kinds, undo/redo,
 * help), contextual hint line, keyboard-shortcut help overlay (H or ?), the
 * "enable sound" call-to-action and an optional stats HUD. Styles live in index.html.
 *
 * Talks to the (re-creatable) editor only through the canvas DOM-event protocol and to the
 * parameters through the ParamStore, so it survives world rebuilds.
 */
import type { EditorTool, ObstacleKind, ParamStore } from '../app/params';
import { EDITOR_STATUS_EVENT, isMac, modKey, sendEditorCommand, type EditorCommand, type EditorStatus } from '../editor/protocol';

const ICONS: Record<string, string> = {
  orbit: '<path d="M12 4a8 8 0 1 1-7.4 5"/><path d="M4 4v5h5"/><circle cx="12" cy="12" r="2.2"/>',
  add: '<path d="M4 17c1-4 3.6-7 7.5-7.6C15 8.8 18.6 10.6 20 14.5c.4 1.1-.3 2.5-1.6 2.5H5.3C4.5 17 3.8 17.6 4 17z"/><path d="M12 2.5v5M9.5 5h5"/>',
  move: '<path d="M12 3v18M3 12h18"/><path d="M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
  delete: '<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 13h9l1-13"/><path d="M10 11v6M14 11v6"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.2a2.5 2.5 0 1 1 3.4 2.3c-.7.3-1 .8-1 1.5v.6"/><circle cx="12" cy="17.2" r=".6" fill="currentColor"/>',
  sound: '<path d="M4 9.5h3.5L12 5v14l-4.5-4.5H4z"/><path d="M15.5 9a4.5 4.5 0 0 1 0 6M18 6.5a8 8 0 0 1 0 11"/>',
  boulder: '<path d="M3.5 18c.6-4.2 3.2-8.4 7.4-9.2 4.4-.8 8.4 2.4 9.6 7 .3 1.2-.4 2.2-1.6 2.2H4.6c-.7 0-1.2-.4-1.1-1z"/>',
  cobble: '<ellipse cx="12" cy="15" rx="8" ry="4.2"/>',
  slab: '<path d="M3 15.5l4-4h13l-3 4z"/><path d="M3 15.5v1.8h14l3-3.8v-2"/>',
  log: '<ellipse cx="6" cy="13" rx="2.6" ry="4"/><path d="M6 9h12.5a2.6 4 0 0 1 0 8H6"/><ellipse cx="6" cy="13" rx="1" ry="1.6"/>',
};

const svg = (name: string) =>
  `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] ?? ''}</svg>`;

const TOOL_DEFS: { tool: EditorTool; label: string; key: string; tip: string }[] = [
  { tool: 'orbit', label: 'Orbit', key: '1', tip: 'Orbit the camera' },
  { tool: 'add', label: 'Add', key: '2', tip: 'Add rocks and logs' },
  { tool: 'move', label: 'Move', key: '3', tip: 'Select, move, rotate and scale' },
  { tool: 'delete', label: 'Delete', key: '4', tip: 'Remove rocks and logs' },
];

const KIND_DEFS: { kind: ObstacleKind; label: string }[] = [
  { kind: 'boulder', label: 'Boulder' },
  { kind: 'cobble', label: 'Cobble' },
  { kind: 'slab', label: 'Slab' },
  { kind: 'log', label: 'Log' },
];

export function helpSections(): { title: string; rows: [string, string][] }[] {
  const M = modKey();
  const alt = isMac() ? '⌥' : 'Alt';
  return [
    {
      title: 'Camera (any tool)',
      rows: [
        ['Drag empty space', 'Orbit'],
        ['Right-drag · Shift+drag', 'Pan'],
        ['Wheel · pinch', 'Zoom towards the cursor'],
        ['Double-click', 'Focus on that point'],
        ['F', 'Focus on the selection / cursor'],
        ['Touch', 'One finger orbits, two fingers pinch & pan'],
      ],
    },
    {
      title: 'Tools',
      rows: [
        ['1  2  3  4', 'Orbit · Add · Move · Delete'],
        ['Esc', 'Cancel drag · deselect · back to Orbit'],
        [`${M}Z`, 'Undo'],
        [`${M}Shift+Z · ${M}Y`, 'Redo'],
        ['P', 'Pause / resume the water'],
        ['H · ?', 'Show / hide this help'],
      ],
    },
    {
      title: 'Add',
      rows: [
        ['Click', 'Place the ghost'],
        [`${alt}+wheel · [ ]`, 'Size'],
        ['Q / E · Shift+wheel', 'Rotate'],
        ['R', 'New random shape'],
        ['K · Shift+K', 'Next / previous kind'],
      ],
    },
    {
      title: 'Move (selected)',
      rows: [
        ['Drag', 'Slide over the bed'],
        [`Shift+drag · ${alt}+drag`, 'Raise / lower'],
        ['Q / E', 'Rotate (Shift: fine)'],
        ['T / G', 'Tilt'],
        [`[ ] · ${alt}+wheel`, 'Scale'],
        ['Arrows · PgUp / PgDn', 'Nudge 1 cm · raise / lower'],
        ['Del · Backspace', 'Delete'],
        [`${M}D`, 'Duplicate'],
      ],
    },
  ];
}

export interface ChromeOptions {
  canvas: HTMLCanvasElement;
  store: ParamStore;
  enableAudio(): void;
  /** undefined when unknown (no host). */
  audioRunning(): boolean | undefined;
}

export class EditorChrome {
  readonly root: HTMLElement;
  private toolbar: HTMLElement;
  private kinds: HTMLElement;
  private help: HTMLElement;
  private sound: HTMLButtonElement;
  private statsBox: HTMLElement;
  private hintEl: HTMLElement | null;
  private status: EditorStatus | null = null;
  private unsubs: (() => unknown)[] = [];

  constructor(private opts: ChromeOptions) {
    const doc = document;
    this.root = doc.createElement('div');
    this.root.id = 'wf-chrome';

    this.toolbar = doc.createElement('div');
    this.toolbar.className = 'wf-toolbar';
    this.toolbar.setAttribute('role', 'toolbar');
    this.toolbar.setAttribute('aria-label', 'Editing tools');
    const tools = doc.createElement('div');
    tools.className = 'wf-group';
    for (const t of TOOL_DEFS) {
      const b = this.button(`${svg(t.tool)}<span class="wf-label">${t.label}</span><kbd>${t.key}</kbd>`, `${t.tip} (${t.key})`);
      b.dataset.tool = t.tool;
      b.addEventListener('click', () => opts.store.set('editor.tool', t.tool));
      tools.appendChild(b);
    }
    this.kinds = doc.createElement('div');
    this.kinds.className = 'wf-group wf-kinds';
    for (const k of KIND_DEFS) {
      const b = this.button(`${svg(k.kind)}<span class="wf-label">${k.label}</span>`, `${k.label} (K cycles)`);
      b.dataset.kind = k.kind;
      b.addEventListener('click', () => {
        opts.store.set('editor.addKind', k.kind);
        if (opts.store.values.editor.tool !== 'add') opts.store.set('editor.tool', 'add');
      });
      this.kinds.appendChild(b);
    }
    const actions = doc.createElement('div');
    actions.className = 'wf-group';
    const M = modKey();
    const cmd = (name: EditorCommand, icon: string, tip: string) => {
      const b = this.button(svg(icon), tip);
      b.dataset.cmd = name;
      b.addEventListener('click', () => sendEditorCommand(opts.canvas, name));
      actions.appendChild(b);
      return b;
    };
    cmd('undo', 'undo', `Undo (${M}Z)`);
    cmd('redo', 'redo', `Redo (${M}Shift+Z)`);
    const hb = this.button(svg('help'), 'Keyboard shortcuts (H)');
    hb.dataset.cmd = 'help';
    hb.addEventListener('click', () => this.toggleHelp());
    actions.appendChild(hb);
    this.toolbar.append(tools, this.kinds, actions);

    this.help = doc.createElement('div');
    this.help.className = 'wf-help';
    this.help.hidden = true;
    this.help.setAttribute('role', 'dialog');
    this.help.setAttribute('aria-label', 'Keyboard shortcuts');
    this.help.innerHTML =
      `<div class="wf-help-card"><div class="wf-help-head"><h2>Controls</h2><button class="wf-close" aria-label="Close">×</button></div><div class="wf-help-grid">` +
      helpSections()
        .map((s) => `<section><h3>${s.title}</h3><dl>${s.rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></section>`)
        .join('') +
      `</div><p class="wf-help-foot">Rocks and logs change the flow immediately: water parts around them, piles up behind them and spills over them.</p></div>`;
    this.help.addEventListener('click', (e) => {
      if (e.target === this.help || (e.target as HTMLElement).closest('.wf-close')) this.toggleHelp(false);
    });

    this.sound = doc.createElement('button');
    this.sound.className = 'wf-sound';
    this.sound.type = 'button';
    this.sound.innerHTML = `${svg('sound')}<span>Sound is off — <b>click to hear the creek</b></span>`;
    this.sound.title = 'Enable audio (browsers need a click before playing sound)';
    this.sound.hidden = true;
    this.sound.addEventListener('click', () => opts.enableAudio());

    this.statsBox = doc.createElement('pre');
    this.statsBox.className = 'wf-stats';
    this.statsBox.hidden = true;

    this.root.append(this.toolbar, this.help, this.sound, this.statsBox);
    doc.body.appendChild(this.root);
    this.hintEl = doc.getElementById('hint');

    const onStatus = (e: Event) => {
      this.status = (e as CustomEvent<EditorStatus>).detail;
      this.render();
    };
    opts.canvas.addEventListener(EDITOR_STATUS_EVENT, onStatus);
    this.unsubs.push(() => opts.canvas.removeEventListener(EDITOR_STATUS_EVENT, onStatus));
    this.unsubs.push(opts.store.on('editor', () => this.render()));
    this.unsubs.push(opts.store.on('debug.showStats', () => this.update()));
    const onKey = (e: KeyboardEvent) => this.onKey(e);
    window.addEventListener('keydown', onKey, { capture: true });
    this.unsubs.push(() => window.removeEventListener('keydown', onKey, { capture: true }));
    sendEditorCommand(opts.canvas, 'status');
    this.render();
    this.update();
  }

  private button(html: string, title: string) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'wf-btn';
    b.title = title;
    b.setAttribute('aria-label', title);
    b.innerHTML = html;
    return b;
  }

  private onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'h' || e.key === 'H' || e.key === '?') {
      e.preventDefault();
      this.toggleHelp();
    } else if (e.key === 'Escape' && !this.help.hidden) {
      e.preventDefault();
      e.stopImmediatePropagation();
      this.toggleHelp(false);
    }
  }

  toggleHelp(show = this.help.hidden) {
    this.help.hidden = !show;
  }

  get helpVisible() {
    return !this.help.hidden;
  }

  private render() {
    const ed = this.opts.store.values.editor;
    const st = this.status;
    const tool = ed.tool;
    this.toolbar.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((b) => {
      const on = b.dataset.tool === tool;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    });
    this.kinds.classList.toggle('show', tool === 'add');
    this.kinds.querySelectorAll<HTMLButtonElement>('[data-kind]').forEach((b) => {
      const on = b.dataset.kind === ed.addKind;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    });
    const undo = this.toolbar.querySelector<HTMLButtonElement>('[data-cmd="undo"]')!;
    const redo = this.toolbar.querySelector<HTMLButtonElement>('[data-cmd="redo"]')!;
    const M = modKey();
    undo.disabled = !st?.canUndo;
    redo.disabled = !st?.canRedo;
    undo.title = st?.undoLabel ? `Undo ${st.undoLabel} (${M}Z)` : `Undo (${M}Z)`;
    redo.title = st?.redoLabel ? `Redo ${st.redoLabel} (${M}Shift+Z)` : `Redo (${M}Shift+Z)`;
    if (this.hintEl && st?.hint && this.hintEl.textContent !== st.hint) this.hintEl.textContent = st.hint;
  }

  /** Periodic refresh (sound CTA, stats HUD visibility). */
  update() {
    const running = this.opts.audioRunning();
    this.sound.hidden = running !== false || !this.opts.store.values.audio.enabled;
    this.statsBox.hidden = !this.opts.store.values.debug.showStats;
  }

  setStats(text: string) {
    if (!this.statsBox.hidden && this.statsBox.textContent !== text) this.statsBox.textContent = text;
  }

  destroy() {
    this.unsubs.forEach((u) => u());
    this.root.remove();
  }
}
