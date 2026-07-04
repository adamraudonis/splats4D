// Data worker: streams the .splat4d file, decodes GOPs, and maintains the
// splat data in the EXACT texture layout of the reference renderer
// (antimatter15/splat): per splat, two rgba32uint texels —
//   texel A: x, y, z as f32 bits + (unused)
//   texel B: 3x packHalf2x16(4*sigma pairs) + rgba bytes
// Sigma is recomputed on the CPU for splats whose rotation/scale change,
// exactly like the reference worker's generateTexture(). Sorting is the
// reference's 16-bit counting sort (ascending depth = front-to-back).
//
// Protocol (main -> worker):
//   {type:'load', url}
//   {type:'frame', frame}
//   {type:'view', viewProj: Float32Array}        every frame (worker dedups)
//   {type:'perm', perm: ArrayBuffer}             encoder permutation (compare)
//   {type:'origframe', frame, buffer}            raw .splat frame (compare)
//   {type:'return', kind, buffer}                pooled buffer return
//
// (worker -> main):
//   {type:'meta', ...}
//   {type:'static', texdata, texwidth, texheight, staticMs}
//   {type:'frame', frame, band, rowStart, rows, approximate, decodeMs}
//   {type:'buffered', gop, bytesLoaded}
//   {type:'miss', frame, gop}
//   {type:'sorted', indices, count, sortMs}
//   {type:'origtex', frame, texdata, texwidth, texheight}
//   {type:'error', message}

import {
  parseHeader,
  parseStatic,
  parseGop,
  parseGopKeys,
  gopKeysPrefixLen,
  unzigzag,
  type Header,
  type GopData,
  type StaticState,
  ATTR_POS,
  ATTR_ROT,
  ATTR_RGB,
  ATTR_ALPHA,
  ATTR_SCALE,
} from './format';

export const TEXWIDTH = 2048;

let header: Header | null = null;
let stat: StaticState | null = null;
let url = '';
let fileSize = 0;
let rangeMode = false;
let bytesLoaded = 0;

let fileBuf: Uint8Array | null = null;
const gopReady: boolean[] = [];
const gopCache = new Map<number, GopData>();
const gopKeys = new Map<number, GopData>();
const gopFetching = new Set<number>();

const CH = [3, 4, 3, 1, 3];
let dynIdx: Uint32Array[] = [];
let cur: Int32Array[] = [];
let curFrame = -1;

// current state per splat (file order)
let posF: Float32Array; // n*3
let scaleF: Float32Array; // n*3 linear
let quatB: Uint8Array; // n*4 raw u8 (w,x,y,z), .splat convention
let rgbaB: Uint8Array; // n*4

// the texture (reference layout)
let texdata: Uint32Array;
let texF: Float32Array;
let texC: Uint8Array;
let texHeight = 0;
let dirtyRowStart = 0; // dynamic band start row (constant per file)
let sigmaIdx: Uint32Array; // splats needing sigma recompute per frame
let permArr: Uint32Array | null = null;

let pendingFrame = -1;

const pools: Record<string, ArrayBuffer[]> = { band: [], sort: [], origtex: [] };

function post(msg: unknown, transfer?: Transferable[]) {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

// ---- half float packing (reference implementation, verbatim) --------------
const _floatView = new Float32Array(1);
const _int32View = new Int32Array(_floatView.buffer);

function floatToHalf(float: number): number {
  _floatView[0] = float;
  const f = _int32View[0];
  const sign = (f >> 31) & 0x0001;
  const exp = (f >> 23) & 0x00ff;
  let frac = f & 0x007fffff;
  let newExp;
  if (exp === 0) {
    newExp = 0;
  } else if (exp < 113) {
    newExp = 0;
    frac |= 0x00800000;
    frac = frac >> (113 - exp);
    if (frac & 0x01000000) {
      newExp = 1;
      frac = 0;
    }
  } else if (exp < 142) {
    newExp = exp - 112;
  } else {
    newExp = 31;
    frac = 0;
  }
  return (sign << 15) | (newExp << 10) | (frac >> 13);
}

function packHalf2x16(x: number, y: number): number {
  return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
}

/** sigma words for splat i from quatB/scaleF (reference generateTexture math) */
function writeSigma(i: number) {
  const sx = scaleF[i * 3];
  const sy = scaleF[i * 3 + 1];
  const sz = scaleF[i * 3 + 2];
  const r0 = (quatB[i * 4] - 128) / 128;
  const r1 = (quatB[i * 4 + 1] - 128) / 128;
  const r2 = (quatB[i * 4 + 2] - 128) / 128;
  const r3 = (quatB[i * 4 + 3] - 128) / 128;

  const m0 = (1 - 2 * (r2 * r2 + r3 * r3)) * sx;
  const m1 = 2 * (r1 * r2 + r0 * r3) * sx;
  const m2 = 2 * (r1 * r3 - r0 * r2) * sx;
  const m3 = 2 * (r1 * r2 - r0 * r3) * sy;
  const m4 = (1 - 2 * (r1 * r1 + r3 * r3)) * sy;
  const m5 = 2 * (r2 * r3 + r0 * r1) * sy;
  const m6 = 2 * (r1 * r3 + r0 * r2) * sz;
  const m7 = 2 * (r2 * r3 - r0 * r1) * sz;
  const m8 = (1 - 2 * (r1 * r1 + r2 * r2)) * sz;

  const s0 = m0 * m0 + m3 * m3 + m6 * m6;
  const s1 = m0 * m1 + m3 * m4 + m6 * m7;
  const s2 = m0 * m2 + m3 * m5 + m6 * m8;
  const s3 = m1 * m1 + m4 * m4 + m7 * m7;
  const s4 = m1 * m2 + m4 * m5 + m7 * m8;
  const s5 = m2 * m2 + m5 * m5 + m8 * m8;

  texdata[8 * i + 4] = packHalf2x16(4 * s0, 4 * s1);
  texdata[8 * i + 5] = packHalf2x16(4 * s2, 4 * s3);
  texdata[8 * i + 6] = packHalf2x16(4 * s4, 4 * s5);
}

function writePos(i: number) {
  texF[8 * i] = posF[i * 3];
  texF[8 * i + 1] = posF[i * 3 + 1];
  texF[8 * i + 2] = posF[i * 3 + 2];
}

function writeColor(i: number) {
  texC[4 * (8 * i + 7)] = rgbaB[i * 4];
  texC[4 * (8 * i + 7) + 1] = rgbaB[i * 4 + 1];
  texC[4 * (8 * i + 7) + 2] = rgbaB[i * 4 + 2];
  texC[4 * (8 * i + 7) + 3] = rgbaB[i * 4 + 3];
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      url = m.url;
      void load();
    } else if (m.type === 'frame') {
      requestFrame(m.frame);
    } else if (m.type === 'view') {
      runSort(m.viewProj);
    } else if (m.type === 'perm') {
      permArr = new Uint32Array(m.perm);
    } else if (m.type === 'origframe') {
      buildOrigTex(m.frame, m.buffer);
    } else if (m.type === 'return') {
      const pool = pools[m.kind];
      if (pool && pool.length < 4) pool.push(m.buffer);
    } else if (m.type === 'dump') {
      post({ type: 'dump', stats: dumpStats() });
    }
  } catch (err) {
    post({ type: 'error', message: String(err) });
  }
};

async function load() {
  const t0 = performance.now();
  const probe = await fetch(url, { headers: { Range: 'bytes=0-262143' } });
  rangeMode = probe.status === 206;
  const contentRange = probe.headers.get('Content-Range');
  if (rangeMode && contentRange) fileSize = parseInt(contentRange.split('/')[1], 10);

  let head: Uint8Array;
  if (rangeMode) {
    head = new Uint8Array(await probe.arrayBuffer());
  } else {
    fileSize = parseInt(probe.headers.get('Content-Length') ?? '0', 10);
    void streamBody(probe);
    head = await waitBytes(262144);
  }
  const { header: h } = parseHeader(head);
  header = h;
  for (let g = 0; g < h.gops.length; g++) gopReady.push(false);
  post({
    type: 'meta',
    n: h.n,
    t: h.t,
    fps: h.fps,
    gop: h.gop,
    aabb: h.aabb,
    bounds: h.bounds,
    gops: h.gops.map((g) => ({ f0: g.f0, f1: g.f1, t0: g.t0, t1: g.t1, offset: g.offset, len: g.len })),
    fileSize,
    denoised: h.denoised,
    headerMs: performance.now() - t0,
  });
  if (!rangeMode) checkSequentialGops();

  const so = h.static_section.offset;
  const sl = h.static_section.len;
  let sbuf: Uint8Array;
  if (rangeMode) {
    sbuf = so + sl <= head.length ? head.subarray(so, so + sl) : await fetchRange(so, so + sl - 1);
  } else {
    sbuf = (await waitBytes(so + sl)).subarray(so, so + sl);
  }
  stat = parseStatic(sbuf, h);
  buildInitialState();
  const texCopy = texdata.slice(0);
  post(
    {
      type: 'static',
      texdata: texCopy,
      texwidth: TEXWIDTH,
      texheight: texHeight,
      staticMs: performance.now() - t0,
    },
    [texCopy.buffer]
  );

  if (rangeMode) void prefetchLoop();
}

// ---- transport (unchanged) -----------------------------------------------
let streamResolvers: { bytes: number; resolve: (b: Uint8Array) => void }[] = [];

async function streamBody(resp: Response) {
  let fb = new Uint8Array(fileSize || 256 << 20);
  fileBuf = fb;
  const reader = resp.body!.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      checkSequentialGops();
      break;
    }
    if (bytesLoaded + value.length > fb.length) {
      const bigger = new Uint8Array(Math.max(fb.length * 2, bytesLoaded + value.length));
      bigger.set(fb.subarray(0, bytesLoaded));
      fb = bigger;
      fileBuf = fb;
    }
    fb.set(value, bytesLoaded);
    bytesLoaded += value.length;
    streamResolvers = streamResolvers.filter((r) => {
      if (bytesLoaded >= r.bytes) {
        r.resolve(fb);
        return false;
      }
      return true;
    });
    checkSequentialGops();
  }
}

function waitBytes(bytes: number): Promise<Uint8Array> {
  if (bytesLoaded >= bytes) return Promise.resolve(fileBuf!);
  return new Promise((resolve) => streamResolvers.push({ bytes, resolve }));
}

async function fetchRange(start: number, end: number): Promise<Uint8Array> {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  const buf = new Uint8Array(await r.arrayBuffer());
  bytesLoaded += buf.length;
  return buf;
}

function checkSequentialGops() {
  if (!header) return;
  for (let g = 0; g < header.gops.length; g++) {
    const span = header.gops[g];
    if (!gopReady[g] && bytesLoaded >= span.offset + span.len) {
      gopCache.set(g, parseGop(fileBuf!.subarray(span.offset, span.offset + span.len)));
      gopReady[g] = true;
      post({ type: 'buffered', gop: g, bytesLoaded });
      servePending();
    }
  }
}

async function prefetchLoop() {
  if (!header) return;
  for (;;) {
    let target = -1;
    if (pendingFrame >= 0) {
      const g = Math.floor(pendingFrame / header.gop);
      if (!gopReady[g] && !gopFetching.has(g)) target = g;
    }
    if (target < 0) {
      const from = curFrame >= 0 ? Math.floor(curFrame / header.gop) : 0;
      for (let d = 0; d < header.gops.length; d++) {
        const g = (from + d) % header.gops.length;
        if (!gopReady[g] && !gopFetching.has(g)) {
          target = g;
          break;
        }
      }
    }
    if (target < 0) return;
    gopFetching.add(target);
    const span = header.gops[target];
    const buf = await fetchRange(span.offset, span.offset + span.len - 1);
    gopCache.set(target, parseGop(buf));
    gopReady[target] = true;
    gopFetching.delete(target);
    post({ type: 'buffered', gop: target, bytesLoaded });
    servePending();
  }
}

// ---- state ----------------------------------------------------------------
function clamp8(v: number) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function buildInitialState() {
  const h = header!;
  const s = stat!;
  const n = h.n;
  const st = h.steps;

  posF = new Float32Array(n * 3);
  scaleF = new Float32Array(n * 3);
  quatB = new Uint8Array(n * 4);
  rgbaB = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    posF[i * 3] = s.basePos[i * 3] * st.base_pos;
    posF[i * 3 + 1] = s.basePos[i * 3 + 1] * st.base_pos;
    posF[i * 3 + 2] = s.basePos[i * 3 + 2] * st.base_pos;
    scaleF[i * 3] = Math.exp(s.baseLs[i * 3] * st.base_scale_log);
    scaleF[i * 3 + 1] = Math.exp(s.baseLs[i * 3 + 1] * st.base_scale_log);
    scaleF[i * 3 + 2] = Math.exp(s.baseLs[i * 3 + 2] * st.base_scale_log);
    quatB[i * 4] = clamp8(s.baseRot[i * 4] + 128);
    quatB[i * 4 + 1] = clamp8(s.baseRot[i * 4 + 1] + 128);
    quatB[i * 4 + 2] = clamp8(s.baseRot[i * 4 + 2] + 128);
    quatB[i * 4 + 3] = clamp8(s.baseRot[i * 4 + 3] + 128);
    rgbaB[i * 4] = s.baseRgb[i * 3];
    rgbaB[i * 4 + 1] = s.baseRgb[i * 3 + 1];
    rgbaB[i * 4 + 2] = s.baseRgb[i * 3 + 2];
    rgbaB[i * 4 + 3] = s.baseAlpha[i];
  }

  dynIdx = [];
  cur = [];
  for (let a = 0; a < 5; a++) {
    const mask = s.masks[a];
    let count = 0;
    for (let i = 0; i < n; i++) if (!mask[i]) count++;
    const idx = new Uint32Array(count);
    let j = 0;
    for (let i = 0; i < n; i++) if (!mask[i]) idx[j++] = i;
    dynIdx.push(idx);
    cur.push(new Int32Array(count * CH[a]));
  }
  // splats whose sigma changes over time = rot- or scale-dynamic
  const sig = new Set<number>();
  for (const i of dynIdx[ATTR_ROT]) sig.add(i);
  for (const i of dynIdx[ATTR_SCALE]) sig.add(i);
  sigmaIdx = Uint32Array.from([...sig].sort((a, b) => a - b));

  // dirty band: rows covering every dynamic splat (contiguous tail by design)
  let firstDyn = n;
  for (let a = 0; a < 5; a++) if (dynIdx[a].length) firstDyn = Math.min(firstDyn, dynIdx[a][0]);
  dirtyRowStart = firstDyn >= n ? 0 : (2 * firstDyn) >> 11; // texel col pair -> row = (2i)/2048

  texHeight = Math.ceil((2 * n) / TEXWIDTH);
  texdata = new Uint32Array(TEXWIDTH * texHeight * 4);
  texF = new Float32Array(texdata.buffer);
  texC = new Uint8Array(texdata.buffer);
  for (let i = 0; i < n; i++) {
    writePos(i);
    writeSigma(i);
    writeColor(i);
  }
  curFrame = -1;
}

function applyKey(a: number, keys: GopData['keys']) {
  const k = keys[a];
  if (!k) return;
  const c = cur[a];
  for (let i = 0; i < c.length; i++) c[i] = unzigzag(k[i]);
}

function applyDeltaRow(a: number, deltas: GopData['deltas'], row: number) {
  const d = deltas[a];
  if (!d) return;
  const c = cur[a];
  const off = row * c.length;
  for (let i = 0; i < c.length; i++) c[i] += unzigzag(d[off + i]);
}

function rollTo(frame: number): boolean {
  const h = header!;
  const g = Math.floor(frame / h.gop);
  if (!gopReady[g]) return false;
  const gopData = gopCache.get(g)!;
  const f0 = h.gops[g].f0;
  let from: number;
  if (curFrame >= f0 && curFrame <= frame) {
    from = curFrame;
  } else {
    for (let a = 0; a < 5; a++) applyKey(a, gopData.keys);
    from = f0;
  }
  for (let f = from + 1; f <= frame; f++) {
    for (let a = 0; a < 5; a++) applyDeltaRow(a, gopData.deltas, f - f0 - 1);
  }
  curFrame = frame;
  return true;
}

/** dequantize current bins into state arrays + texture texels */
function updateState() {
  const st = header!.steps;
  {
    const idx = dynIdx[ATTR_POS];
    const c = cur[ATTR_POS];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      posF[i * 3] = c[j * 3] * st.pos;
      posF[i * 3 + 1] = c[j * 3 + 1] * st.pos;
      posF[i * 3 + 2] = c[j * 3 + 2] * st.pos;
      writePos(i);
    }
  }
  {
    const idx = dynIdx[ATTR_ROT];
    const c = cur[ATTR_ROT];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      quatB[i * 4] = clamp8(c[j * 4] * st.rot + 128);
      quatB[i * 4 + 1] = clamp8(c[j * 4 + 1] * st.rot + 128);
      quatB[i * 4 + 2] = clamp8(c[j * 4 + 2] * st.rot + 128);
      quatB[i * 4 + 3] = clamp8(c[j * 4 + 3] * st.rot + 128);
    }
  }
  {
    const idx = dynIdx[ATTR_SCALE];
    const c = cur[ATTR_SCALE];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      scaleF[i * 3] = Math.exp(c[j * 3] * st.scale_log);
      scaleF[i * 3 + 1] = Math.exp(c[j * 3 + 1] * st.scale_log);
      scaleF[i * 3 + 2] = Math.exp(c[j * 3 + 2] * st.scale_log);
    }
  }
  for (let k = 0; k < sigmaIdx.length; k++) writeSigma(sigmaIdx[k]);
  {
    const idx = dynIdx[ATTR_RGB];
    const c = cur[ATTR_RGB];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      rgbaB[i * 4] = clamp8(c[j * 3] * st.rgb);
      rgbaB[i * 4 + 1] = clamp8(c[j * 3 + 1] * st.rgb);
      rgbaB[i * 4 + 2] = clamp8(c[j * 3 + 2] * st.rgb);
      writeColor(i);
    }
  }
  {
    const idx = dynIdx[ATTR_ALPHA];
    const c = cur[ATTR_ALPHA];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      rgbaB[i * 4 + 3] = clamp8(c[j] * st.alpha);
      writeColor(i);
    }
  }
}

function emitFrame(frame: number, t0: number, approximate: boolean) {
  updateState();
  const rows = texHeight - dirtyRowStart;
  const bandLen = TEXWIDTH * rows * 4; // u32s
  const bb = pools.band.pop() ?? new ArrayBuffer(bandLen * 4);
  const band = new Uint32Array(bb, 0, bandLen);
  band.set(texdata.subarray(dirtyRowStart * TEXWIDTH * 4, (dirtyRowStart + rows) * TEXWIDTH * 4));
  post(
    { type: 'frame', frame, band: bb, rowStart: dirtyRowStart, rows, approximate, decodeMs: performance.now() - t0 },
    [bb]
  );
}

function requestFrame(frame: number) {
  if (!header || !stat) return;
  const t0 = performance.now();
  if (!rollTo(frame)) {
    pendingFrame = frame;
    const g = Math.floor(frame / header.gop);
    post({ type: 'miss', frame, gop: g });
    if (rangeMode) void priorityFetch(g);
    const keysOnly = gopKeys.get(g);
    if (keysOnly && curFrame !== header.gops[g].f0) {
      for (let a = 0; a < 5; a++) applyKey(a, keysOnly.keys);
      curFrame = header.gops[g].f0;
      emitFrame(curFrame, t0, true);
    }
    return;
  }
  pendingFrame = -1;
  emitFrame(frame, t0, false);
}

async function priorityFetch(g: number) {
  if (!header || gopReady[g] || gopFetching.has(g)) return;
  gopFetching.add(g);
  try {
    const span = header.gops[g];
    const head = await fetchRange(span.offset, span.offset + Math.min(4095, span.len - 1));
    const prefixLen = gopKeysPrefixLen(head);
    let prefix = head;
    if (prefixLen > head.length) {
      const more = await fetchRange(span.offset + head.length, span.offset + prefixLen - 1);
      prefix = new Uint8Array(prefixLen);
      prefix.set(head);
      prefix.set(more, head.length);
    }
    if (!gopReady[g]) {
      gopKeys.set(g, parseGopKeys(prefix));
      if (pendingFrame >= 0 && Math.floor(pendingFrame / header.gop) === g) {
        const keysOnly = gopKeys.get(g)!;
        for (let a = 0; a < 5; a++) applyKey(a, keysOnly.keys);
        curFrame = span.f0;
        emitFrame(curFrame, performance.now(), true);
      }
    }
    if (!gopReady[g]) {
      const rest = await fetchRange(span.offset + prefix.length, span.offset + span.len - 1);
      const full = new Uint8Array(span.len);
      full.set(prefix);
      full.set(rest, prefix.length);
      gopCache.set(g, parseGop(full));
      gopReady[g] = true;
      post({ type: 'buffered', gop: g, bytesLoaded });
      servePending();
    }
  } finally {
    gopFetching.delete(g);
  }
}

function servePending() {
  if (pendingFrame >= 0) requestFrame(pendingFrame);
}

// ---- depth sort (reference 16-bit counting sort, ascending = front-to-back)
let lastProj: Float32Array | null = null;
let lastSortedFrame = -2;

function runSort(viewProj: Float32Array) {
  if (!header || !posF) return;
  // reference redundancy check: skip if view direction barely changed AND the
  // frame (splat positions) didn't change since the last sort
  if (lastProj && lastSortedFrame === curFrame) {
    const dot = lastProj[2] * viewProj[2] + lastProj[6] * viewProj[6] + lastProj[10] * viewProj[10];
    if (Math.abs(dot - 1) < 0.01) return;
  }
  const t0 = performance.now();
  const vertexCount = header.n;
  let maxDepth = -Infinity;
  let minDepth = Infinity;
  const sizeList = new Int32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    const depth = ((viewProj[2] * posF[i * 3] + viewProj[6] * posF[i * 3 + 1] + viewProj[10] * posF[i * 3 + 2]) * 4096) | 0;
    sizeList[i] = depth;
    // range from visible splats only: alpha-0 ghosts can sit at extrapolated
    // positions and would collapse the bucket resolution of the real scene
    if (rgbaB[i * 4 + 3] !== 0) {
      if (depth > maxDepth) maxDepth = depth;
      if (depth < minDepth) minDepth = depth;
    }
  }
  if (minDepth > maxDepth) {
    minDepth = 0;
    maxDepth = 1;
  }
  const depthInv = (256 * 256 - 1) / (maxDepth - minDepth || 1);
  const counts0 = new Uint32Array(256 * 256);
  for (let i = 0; i < vertexCount; i++) {
    let b = ((sizeList[i] - minDepth) * depthInv) | 0;
    if (b < 0) b = 0;
    else if (b > 65535) b = 65535;
    sizeList[i] = b;
    counts0[b]++;
  }
  const starts0 = new Uint32Array(256 * 256);
  for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
  const ib = pools.sort.pop() ?? new ArrayBuffer(vertexCount * 4);
  const depthIndex = new Uint32Array(ib, 0, vertexCount);
  for (let i = 0; i < vertexCount; i++) depthIndex[starts0[sizeList[i]]++] = i;

  lastProj = viewProj.slice(0);
  lastSortedFrame = curFrame;
  post({ type: 'sorted', indices: ib, count: vertexCount, sortMs: performance.now() - t0 }, [ib]);
}

// ---- diagnostics -----------------------------------------------------------
function halfToFloat(h: number): number {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

function dumpStats() {
  if (!header || !texdata) return null;
  const n = header.n;
  let maxSigma = 0, nanSigma = 0, maxAbsPos = 0, badPos = 0;
  for (let i = 0; i < n; i++) {
    for (let w = 4; w <= 6; w++) {
      const v = texdata[8 * i + w];
      for (const h of [v & 0xffff, v >>> 16]) {
        const f = halfToFloat(h);
        if (Number.isNaN(f) || !Number.isFinite(f)) nanSigma++;
        else if (Math.abs(f) > maxSigma) maxSigma = Math.abs(f);
      }
    }
    for (let k = 0; k < 3; k++) {
      const p = texF[8 * i + k];
      if (!Number.isFinite(p)) badPos++;
      else if (Math.abs(p) > maxAbsPos) maxAbsPos = Math.abs(p);
    }
  }
  return { n, curFrame, maxSigma, nanSigma, maxAbsPos, badPos, dirtyRowStart, texHeight };
}

// ---- compare: original .splat frame -> full texture -----------------------
function buildOrigTex(frame: number, buffer: ArrayBuffer) {
  if (!header || !permArr) return;
  const n = header.n;
  const raw = new Uint8Array(buffer);
  if (raw.length !== n * 32) return;
  const f32 = new Float32Array(buffer);
  const rows = texHeight;
  const ob = pools.origtex.pop() ?? new ArrayBuffer(TEXWIDTH * rows * 16);
  const otex = new Uint32Array(ob, 0, TEXWIDTH * rows * 4);
  const oF = new Float32Array(ob);
  const oC = new Uint8Array(ob);
  for (let i = 0; i < n; i++) {
    const src = permArr[i]; // file order i <- original index perm[i]
    const s8 = src * 32;
    oF[8 * i] = f32[src * 8];
    oF[8 * i + 1] = f32[src * 8 + 1];
    oF[8 * i + 2] = f32[src * 8 + 2];
    const sx = f32[src * 8 + 3];
    const sy = f32[src * 8 + 4];
    const sz = f32[src * 8 + 5];
    const r0 = (raw[s8 + 28] - 128) / 128;
    const r1 = (raw[s8 + 29] - 128) / 128;
    const r2 = (raw[s8 + 30] - 128) / 128;
    const r3 = (raw[s8 + 31] - 128) / 128;
    const m0 = (1 - 2 * (r2 * r2 + r3 * r3)) * sx;
    const m1 = 2 * (r1 * r2 + r0 * r3) * sx;
    const m2 = 2 * (r1 * r3 - r0 * r2) * sx;
    const m3 = 2 * (r1 * r2 - r0 * r3) * sy;
    const m4 = (1 - 2 * (r1 * r1 + r3 * r3)) * sy;
    const m5 = 2 * (r2 * r3 + r0 * r1) * sy;
    const m6 = 2 * (r1 * r3 + r0 * r2) * sz;
    const m7 = 2 * (r2 * r3 - r0 * r1) * sz;
    const m8 = (1 - 2 * (r1 * r1 + r2 * r2)) * sz;
    otex[8 * i + 4] = packHalf2x16(4 * (m0 * m0 + m3 * m3 + m6 * m6), 4 * (m0 * m1 + m3 * m4 + m6 * m7));
    otex[8 * i + 5] = packHalf2x16(4 * (m0 * m2 + m3 * m5 + m6 * m8), 4 * (m1 * m1 + m4 * m4 + m7 * m7));
    otex[8 * i + 6] = packHalf2x16(4 * (m1 * m2 + m4 * m5 + m7 * m8), 4 * (m2 * m2 + m5 * m5 + m8 * m8));
    oC[4 * (8 * i + 7)] = raw[s8 + 24];
    oC[4 * (8 * i + 7) + 1] = raw[s8 + 25];
    oC[4 * (8 * i + 7) + 2] = raw[s8 + 26];
    oC[4 * (8 * i + 7) + 3] = raw[s8 + 27];
  }
  post({ type: 'origtex', frame, texdata: ob, texwidth: TEXWIDTH, texheight: rows }, [ob]);
}
