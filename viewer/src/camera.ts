// Camera math and input controls from antimatter15/splat (MIT), verbatim
// semantics: OpenCV convention (+z forward, y down), view matrices stored
// flat column-major with translation in elements 12..14.

export function getProjectionMatrix(fx: number, fy: number, width: number, height: number): number[] {
  const znear = 0.2;
  const zfar = 200;
  return [
    [(2 * fx) / width, 0, 0, 0],
    [0, -(2 * fy) / height, 0, 0],
    [0, 0, zfar / (zfar - znear), 1],
    [0, 0, -(zfar * znear) / (zfar - znear), 0],
  ].flat();
}

export function multiply4(a: number[], b: number[]): number[] {
  return [
    b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
    b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
    b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
    b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
    b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
    b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
    b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
    b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
    b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
    b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
    b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
    b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
    b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
    b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
    b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
    b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
  ];
}

export function invert4(a: number[]): number[] | null {
  const b00 = a[0] * a[5] - a[1] * a[4];
  const b01 = a[0] * a[6] - a[2] * a[4];
  const b02 = a[0] * a[7] - a[3] * a[4];
  const b03 = a[1] * a[6] - a[2] * a[5];
  const b04 = a[1] * a[7] - a[3] * a[5];
  const b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12];
  const b07 = a[8] * a[14] - a[10] * a[12];
  const b08 = a[8] * a[15] - a[11] * a[12];
  const b09 = a[9] * a[14] - a[10] * a[13];
  const b10 = a[9] * a[15] - a[11] * a[13];
  const b11 = a[10] * a[15] - a[11] * a[14];
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  return [
    (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
    (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
    (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
    (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
    (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
    (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
    (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
    (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
    (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
    (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
    (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
    (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
    (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
    (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
    (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
    (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
  ];
}

export function rotate4(a: number[], rad: number, x: number, y: number, z: number): number[] {
  const len = Math.hypot(x, y, z);
  x /= len;
  y /= len;
  z /= len;
  const s = Math.sin(rad);
  const c = Math.cos(rad);
  const t = 1 - c;
  const b00 = x * x * t + c;
  const b01 = y * x * t + z * s;
  const b02 = z * x * t - y * s;
  const b10 = x * y * t - z * s;
  const b11 = y * y * t + c;
  const b12 = z * y * t + x * s;
  const b20 = x * z * t + y * s;
  const b21 = y * z * t - x * s;
  const b22 = z * z * t + c;
  return [
    a[0] * b00 + a[4] * b01 + a[8] * b02,
    a[1] * b00 + a[5] * b01 + a[9] * b02,
    a[2] * b00 + a[6] * b01 + a[10] * b02,
    a[3] * b00 + a[7] * b01 + a[11] * b02,
    a[0] * b10 + a[4] * b11 + a[8] * b12,
    a[1] * b10 + a[5] * b11 + a[9] * b12,
    a[2] * b10 + a[6] * b11 + a[10] * b12,
    a[3] * b10 + a[7] * b11 + a[11] * b12,
    a[0] * b20 + a[4] * b21 + a[8] * b22,
    a[1] * b20 + a[5] * b21 + a[9] * b22,
    a[2] * b20 + a[6] * b21 + a[10] * b22,
    a[3] * b20 + a[7] * b21 + a[11] * b22,
    ...a.slice(12, 16),
  ];
}

export function translate4(a: number[], x: number, y: number, z: number): number[] {
  return [
    ...a.slice(0, 12),
    a[0] * x + a[4] * y + a[8] * z + a[12],
    a[1] * x + a[5] * y + a[9] * z + a[13],
    a[2] * x + a[6] * y + a[10] * z + a[14],
    a[3] * x + a[7] * y + a[11] * z + a[15],
  ];
}

export interface CamHint {
  position: number[];
  target: number[];
  up?: number[];
  fov?: number; // vertical degrees
}

/** lookAt in the reference's y-down world -> their flat viewMatrix layout
 *  (validated against the reference viewer via the compare harness). */
export function viewMatrixFromHint(hint: CamHint): number[] {
  const e = hint.position;
  const t = hint.target;
  const sub = (a: number[], b: number[]) => a.map((v, i) => v - b[i]);
  const norm = (a: number[]) => {
    const l = Math.hypot(a[0], a[1], a[2]);
    return a.map((v) => v / l);
  };
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const dp = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const Z = norm(sub(t, e));
  const X = norm(cross([0, 1, 0], Z));
  const Y = cross(Z, X);
  return [
    X[0], Y[0], Z[0], 0,
    X[1], Y[1], Z[1], 0,
    X[2], Y[2], Z[2], 0,
    -dp(e, X), -dp(e, Y), -dp(e, Z), 1,
  ];
}

/** Mouse/keyboard/touch controls transcribed from the reference viewer:
 *  drag = orbit, ctrl/cmd-drag = pan, shift-drag/right-drag = pan,
 *  wheel = orbit/zoom, keys wasd/arrows/jkli/q,e. Operates on a viewMatrix. */
export class SplatControls {
  viewMatrix: number[];
  private activeKeys: string[] = [];
  private down: number | boolean = false;
  private startX = 0;
  private startY = 0;

  constructor(canvas: HTMLCanvasElement, initial: number[]) {
    this.viewMatrix = initial;

    window.addEventListener('keydown', (e) => {
      if (document.activeElement !== document.body && document.activeElement !== null
          && ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes((document.activeElement as HTMLElement).tagName)) return;
      if (!this.activeKeys.includes(e.code)) this.activeKeys.push(e.code);
    });
    window.addEventListener('keyup', (e) => {
      this.activeKeys = this.activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener('blur', () => (this.activeKeys = []));

    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const lineHeight = 10;
        const scale = e.deltaMode === 1 ? lineHeight : e.deltaMode === 2 ? innerHeight : 1;
        let inv = invert4(this.viewMatrix)!;
        if (e.shiftKey) {
          inv = translate4(inv, (e.deltaX * scale) / innerWidth, (e.deltaY * scale) / innerHeight, 0);
        } else if (e.ctrlKey || e.metaKey) {
          inv = translate4(inv, 0, 0, (-10 * (e.deltaY * scale)) / innerHeight);
        } else {
          const d = 4;
          inv = translate4(inv, 0, 0, d);
          inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
          inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
          inv = translate4(inv, 0, 0, -d);
        }
        this.viewMatrix = invert4(inv)!;
      },
      { passive: false }
    );

    canvas.addEventListener('mousedown', (e) => {
      e.preventDefault();
      this.startX = e.clientX;
      this.startY = e.clientY;
      this.down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.startX = e.clientX;
      this.startY = e.clientY;
      this.down = 2;
    });
    canvas.addEventListener('mousemove', (e) => {
      e.preventDefault();
      if (this.down === 1) {
        let inv = invert4(this.viewMatrix)!;
        const dx = (5 * (e.clientX - this.startX)) / innerWidth;
        const dy = (5 * (e.clientY - this.startY)) / innerHeight;
        const d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = rotate4(inv, dx, 0, 1, 0);
        inv = rotate4(inv, -dy, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);
        this.viewMatrix = invert4(inv)!;
        this.startX = e.clientX;
        this.startY = e.clientY;
      } else if (this.down === 2) {
        let inv = invert4(this.viewMatrix)!;
        inv = translate4(
          inv,
          (-10 * (e.clientX - this.startX)) / innerWidth,
          0,
          (10 * (e.clientY - this.startY)) / innerHeight
        );
        this.viewMatrix = invert4(inv)!;
        this.startX = e.clientX;
        this.startY = e.clientY;
      }
    });
    canvas.addEventListener('mouseup', (e) => {
      e.preventDefault();
      this.down = false;
    });

    // touch: one finger orbit, two finger pinch/pan (reference semantics)
    let altX = 0,
      altY = 0;
    canvas.addEventListener(
      'touchstart',
      (e) => {
        e.preventDefault();
        if (e.touches.length === 1) {
          this.startX = e.touches[0].clientX;
          this.startY = e.touches[0].clientY;
          this.down = 1;
        } else if (e.touches.length === 2) {
          this.startX = e.touches[0].clientX;
          altX = e.touches[1].clientX;
          this.startY = e.touches[0].clientY;
          altY = e.touches[1].clientY;
          this.down = 1;
        }
      },
      { passive: false }
    );
    canvas.addEventListener(
      'touchmove',
      (e) => {
        e.preventDefault();
        if (e.touches.length === 1 && this.down) {
          let inv = invert4(this.viewMatrix)!;
          const dx = (4 * (e.touches[0].clientX - this.startX)) / innerWidth;
          const dy = (4 * (e.touches[0].clientY - this.startY)) / innerHeight;
          const d = 4;
          inv = translate4(inv, 0, 0, d);
          inv = rotate4(inv, dx, 0, 1, 0);
          inv = rotate4(inv, -dy, 1, 0, 0);
          inv = translate4(inv, 0, 0, -d);
          this.viewMatrix = invert4(inv)!;
          this.startX = e.touches[0].clientX;
          this.startY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
          const dtheta =
            Math.atan2(this.startY - altY, this.startX - altX) -
            Math.atan2(e.touches[0].clientY - e.touches[1].clientY, e.touches[0].clientX - e.touches[1].clientX);
          const dscale =
            Math.hypot(this.startX - altX, this.startY - altY) /
            Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
          const dx = (e.touches[0].clientX + e.touches[1].clientX - (this.startX + altX)) / 2;
          const dy = (e.touches[0].clientY + e.touches[1].clientY - (this.startY + altY)) / 2;
          let inv = invert4(this.viewMatrix)!;
          inv = rotate4(inv, dtheta, 0, 0, 1);
          inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);
          inv = translate4(inv, 0, 0, 3 * (1 - dscale));
          this.viewMatrix = invert4(inv)!;
          this.startX = e.touches[0].clientX;
          this.startY = e.touches[0].clientY;
          altX = e.touches[1].clientX;
          altY = e.touches[1].clientY;
        }
      },
      { passive: false }
    );
    canvas.addEventListener(
      'touchend',
      (e) => {
        e.preventDefault();
        this.down = false;
        this.startX = 0;
        this.startY = 0;
      },
      { passive: false }
    );
  }

  /** per-frame key handling, reference semantics */
  update() {
    let inv = invert4(this.viewMatrix)!;
    let changed = false;
    if (this.activeKeys.includes('ArrowUp')) {
      inv = translate4(inv, 0, 0, 0.1);
      changed = true;
    }
    if (this.activeKeys.includes('ArrowDown')) {
      inv = translate4(inv, 0, 0, -0.1);
      changed = true;
    }
    if (this.activeKeys.includes('ArrowLeft')) {
      inv = translate4(inv, -0.03, 0, 0);
      changed = true;
    }
    if (this.activeKeys.includes('ArrowRight')) {
      inv = translate4(inv, 0.03, 0, 0);
      changed = true;
    }
    if (['KeyJ', 'KeyK', 'KeyL', 'KeyI'].some((k) => this.activeKeys.includes(k))) {
      const d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = rotate4(inv, this.activeKeys.includes('KeyJ') ? -0.05 : this.activeKeys.includes('KeyL') ? 0.05 : 0, 0, 1, 0);
      inv = rotate4(inv, this.activeKeys.includes('KeyI') ? 0.05 : this.activeKeys.includes('KeyK') ? -0.05 : 0, 1, 0, 0);
      inv = translate4(inv, 0, 0, -d);
      changed = true;
    }
    if (this.activeKeys.includes('KeyA')) {
      inv = translate4(inv, -0.03, 0, 0);
      changed = true;
    }
    if (this.activeKeys.includes('KeyD')) {
      inv = translate4(inv, 0.03, 0, 0);
      changed = true;
    }
    if (this.activeKeys.includes('KeyW')) {
      inv = rotate4(inv, 0.005, 1, 0, 0);
      changed = true;
    }
    if (this.activeKeys.includes('KeyS')) {
      inv = rotate4(inv, -0.005, 1, 0, 0);
      changed = true;
    }
    if (changed) this.viewMatrix = invert4(inv)!;
  }
}
