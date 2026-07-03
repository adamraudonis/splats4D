// Loads ORIGINAL (uncompressed) .splat frames for the comparison view.
// Frames are cached pre-permutation (raw bytes), then gathered into the
// encoder's splat order so both meshes share one index space (and one sort).

export interface OrigFrame {
  pos: Float32Array; // n*4
  scale: Float32Array; // n*4
  quat: Uint32Array; // n
  rgba: Uint32Array; // n
}

export class OrigLoader {
  private cache = new Map<number, Uint8Array>(); // frame -> raw .splat bytes
  private lru: number[] = [];
  private perm: Uint32Array | null = null;
  private inflight = new Map<number, Promise<Uint8Array | null>>();

  private baseUrl: string;
  public n: number;
  private maxCached: number;

  constructor(baseUrl: string, n: number, maxCached = 12) {
    this.baseUrl = baseUrl;
    this.n = n;
    this.maxCached = maxCached;
  }

  hasPerm(): boolean {
    return this.perm !== null;
  }

  /** switch to a different sequence: new frame source, drop all cached state */
  reset(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.cache.clear();
    this.lru = [];
    this.perm = null;
    this.inflight.clear();
  }

  async setPermUrl(url: string) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`perm fetch failed: ${r.status}`);
    this.perm = new Uint32Array(await r.arrayBuffer());
  }

  private async fetchRaw(frame: number): Promise<Uint8Array | null> {
    const hit = this.cache.get(frame);
    if (hit) return hit;
    let p = this.inflight.get(frame);
    if (!p) {
      const name = `frame_${String(frame).padStart(4, '0')}.splat`;
      p = fetch(`${this.baseUrl}/${name}`)
        .then(async (r) => {
          if (!r.ok) return null;
          const buf = new Uint8Array(await r.arrayBuffer());
          this.cache.set(frame, buf);
          this.lru.push(frame);
          while (this.lru.length > this.maxCached) {
            const evict = this.lru.shift()!;
            if (evict !== frame) this.cache.delete(evict);
          }
          return buf;
        })
        .finally(() => this.inflight.delete(frame));
      this.inflight.set(frame, p);
    }
    return p;
  }

  /** Fetch + permute one frame into encoder order. Returns null on failure. */
  async load(frame: number): Promise<OrigFrame | null> {
    if (!this.perm || this.n === 0 || this.perm.length !== this.n) return null;
    const raw = await this.fetchRaw(frame);
    if (!raw || raw.length !== this.n * 32) return null;
    const perm = this.perm;
    const n = this.n;
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
    const u32 = new Uint32Array(raw.buffer, raw.byteOffset, raw.length / 4);
    const pos = new Float32Array(n * 4);
    const scale = new Float32Array(n * 4);
    const quat = new Uint32Array(n);
    const rgba = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const src = perm[i] * 8; // 8 words of 4 bytes per splat
      pos[i * 4] = f32[src];
      pos[i * 4 + 1] = f32[src + 1];
      pos[i * 4 + 2] = f32[src + 2];
      scale[i * 4] = f32[src + 3];
      scale[i * 4 + 1] = f32[src + 4];
      scale[i * 4 + 2] = f32[src + 5];
      rgba[i] = u32[src + 6]; // bytes 24..27 = r,g,b,a (LE)
      quat[i] = u32[src + 7]; // bytes 28..31 = w,x,y,z (LE)
    }
    return { pos, scale, quat, rgba };
  }
}
