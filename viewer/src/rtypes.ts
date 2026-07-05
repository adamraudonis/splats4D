// Structural interface shared by the WebGPU renderer (renderer.ts) and the
// WebGL2 fallback (renderer_gl.ts) so the streaming pipeline is backend-agnostic.

export interface RSet {
  count: number;
  visible: boolean;
  uploadTexture(texdata: Uint32Array, texwidth: number, texheight: number): void;
  uploadTexRows(band: Uint32Array, yStart: number, rows: number): void;
  setIndices(indices: Uint32Array, count: number): void;
  setCamera(projection: ArrayLike<number>, view: ArrayLike<number>, fx: number, fy: number, vw: number, vh: number): void;
  setClip(divider: number, side: number): void;
  dispose(): void;
}

export interface R {
  backend: string;
  createSet(): RSet;
  render(sets: RSet[]): void;
}
