// Data worker: streams the .splat4d file, decodes GOPs, maintains the current
// frame state (integer bins -> dequantized GPU-ready arrays), and depth-sorts.
//
// Protocol (main -> worker):
//   {type:'load', url}
//   {type:'frame', frame}                   request a specific frame's state
//   {type:'sort', viewProj: Float32Array}   request a depth sort of current positions
//   {type:'return', kind, buffer}           return a pooled buffer
//
// (worker -> main):
//   {type:'meta', ...header summary}
//   {type:'static', pos, scale, quat, rgba, stats}      full initial arrays (transfer)
//   {type:'frame', frame, pos, quat, rgba, decodeMs}    updated full arrays (transfer)
//   {type:'buffered', gop, bytesLoaded}                 a GOP became available
//   {type:'miss', frame, gop}                           seek target not buffered yet
//   {type:'sorted', indices, sortMs}
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

let header: Header | null = null;
let stat: StaticState | null = null;
let url = '';
let fileSize = 0;
let rangeMode = false;
let bytesLoaded = 0;

// full file buffer (sequential mode) or sparse chunks (range mode)
let fileBuf: Uint8Array | null = null;
const gopReady: boolean[] = [];
const gopCache = new Map<number, GopData>();
const gopKeys = new Map<number, GopData>(); // keys-only (seek fast path)
const gopFetching = new Set<number>();

// dynamic-attr index lists and current integer state
const CH = [3, 4, 3, 1, 3];
let dynIdx: Uint32Array[] = [];
let cur: Int32Array[] = []; // current bins per attr
let curFrame = -1;

// persistent dequantized full arrays
let posF: Float32Array; // n*4
let quatU: Uint32Array; // n   packed (w,x,y,z) u8
let rgbaU: Uint32Array; // n   packed (r,g,b,a) u8
let scaleF: Float32Array; // n*4 (static in v1 files; rebuilt per frame if dynamic)

let pendingFrame = -1;

// simple buffer pools keyed by kind
const pools: Record<string, ArrayBuffer[]> = { pos: [], quat: [], rgba: [], sort: [] };

function post(msg: unknown, transfer?: Transferable[]) {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      url = m.url;
      void load();
    } else if (m.type === 'frame') {
      requestFrame(m.frame);
    } else if (m.type === 'sort') {
      sort(m.viewProj);
    } else if (m.type === 'return') {
      const pool = pools[m.kind];
      if (pool && pool.length < 4) pool.push(m.buffer);
    }
  } catch (err) {
    post({ type: 'error', message: String(err) });
  }
};

async function load() {
  const t0 = performance.now();
  // probe: range request for the first 256 KB
  const probe = await fetch(url, { headers: { Range: 'bytes=0-262143' } });
  rangeMode = probe.status === 206;
  const contentRange = probe.headers.get('Content-Range');
  if (rangeMode && contentRange) fileSize = parseInt(contentRange.split('/')[1], 10);

  let head: Uint8Array;
  if (rangeMode) {
    head = new Uint8Array(await probe.arrayBuffer());
  } else {
    // server ignored Range: stream the whole body progressively
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
  // bytes may have fully streamed before the header existed — re-check now
  if (!rangeMode) checkSequentialGops();

  // static section
  const so = h.static_section.offset;
  const sl = h.static_section.len;
  let sbuf: Uint8Array;
  if (rangeMode) {
    sbuf = so + sl <= head.length ? head.subarray(so, so + sl) : await fetchRange(so, so + sl - 1);
  } else {
    sbuf = (await waitBytes(so + sl)).subarray(so, so + sl);
  }
  stat = parseStatic(sbuf, h);
  buildInitialArrays();
  post(
    {
      type: 'static',
      pos: posF.buffer,
      scale: scaleF.buffer,
      quat: quatU.buffer,
      rgba: rgbaU.buffer,
      staticMs: performance.now() - t0,
    },
    // NOTE: keep worker-side copies — clone before transfer
  );

  // background: fetch GOPs sequentially (range mode); sequential stream handles it otherwise
  if (rangeMode) void prefetchLoop();
  else void watchSequential();
}

// ---- transport ----------------------------------------------------------
let streamResolvers: { bytes: number; resolve: (b: Uint8Array) => void }[] = [];

async function streamBody(resp: Response) {
  let fb = new Uint8Array(fileSize || 256 << 20);
  fileBuf = fb;
  const reader = resp.body!.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      checkSequentialGops(); // final sweep in case the header arrived late
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
      const data = parseGop(fileBuf!.subarray(span.offset, span.offset + span.len));
      gopCache.set(g, data);
      gopReady[g] = true;
      post({ type: 'buffered', gop: g, bytesLoaded });
      servePending();
    }
  }
}

async function watchSequential() {
  // gop availability is driven by streamBody -> checkSequentialGops
}

async function prefetchLoop() {
  if (!header) return;
  for (;;) {
    // priority: pending seek target's GOP, then nearest un-fetched after playhead
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
    if (target < 0) return; // everything fetched
    gopFetching.add(target);
    const span = header.gops[target];
    const buf = await fetchRange(span.offset, span.offset + span.len - 1);
    const data = parseGop(buf);
    gopCache.set(target, data);
    gopReady[target] = true;
    gopFetching.delete(target);
    post({ type: 'buffered', gop: target, bytesLoaded });
    servePending();
  }
}

// ---- state --------------------------------------------------------------
function buildInitialArrays() {
  const h = header!;
  const s = stat!;
  const n = h.n;
  const st = h.steps;
  posF = new Float32Array(n * 4);
  scaleF = new Float32Array(n * 4);
  quatU = new Uint32Array(n);
  rgbaU = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    posF[i * 4] = s.basePos[i * 3] * st.base_pos;
    posF[i * 4 + 1] = s.basePos[i * 3 + 1] * st.base_pos;
    posF[i * 4 + 2] = s.basePos[i * 3 + 2] * st.base_pos;
    scaleF[i * 4] = Math.exp(s.baseLs[i * 3] * st.base_scale_log);
    scaleF[i * 4 + 1] = Math.exp(s.baseLs[i * 3 + 1] * st.base_scale_log);
    scaleF[i * 4 + 2] = Math.exp(s.baseLs[i * 3 + 2] * st.base_scale_log);
    const w = clamp8(s.baseRot[i * 4] + 128);
    const x = clamp8(s.baseRot[i * 4 + 1] + 128);
    const y = clamp8(s.baseRot[i * 4 + 2] + 128);
    const z = clamp8(s.baseRot[i * 4 + 3] + 128);
    quatU[i] = (w | (x << 8) | (y << 16) | (z << 24)) >>> 0;
    rgbaU[i] =
      (s.baseRgb[i * 3] | (s.baseRgb[i * 3 + 1] << 8) | (s.baseRgb[i * 3 + 2] << 16) | (s.baseAlpha[i] << 24)) >>> 0;
  }
  // dynamic index lists
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
  // static arrays show frame 0 (bases hold frame-0 values for dynamic attrs),
  // but track bins are unseeded until GOP 0's keys are applied
  curFrame = -1;
}

function clamp8(v: number) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
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
  // roll forward from current state when it lies in [f0, frame]; otherwise reseed from the keyframe
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

function updateArrays() {
  const st = header!.steps;
  // positions
  {
    const idx = dynIdx[ATTR_POS];
    const c = cur[ATTR_POS];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      posF[i * 4] = c[j * 3] * st.pos;
      posF[i * 4 + 1] = c[j * 3 + 1] * st.pos;
      posF[i * 4 + 2] = c[j * 3 + 2] * st.pos;
    }
  }
  // rotations
  {
    const idx = dynIdx[ATTR_ROT];
    const c = cur[ATTR_ROT];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      const w = clamp8(c[j * 4] * st.rot + 128);
      const x = clamp8(c[j * 4 + 1] * st.rot + 128);
      const y = clamp8(c[j * 4 + 2] * st.rot + 128);
      const z = clamp8(c[j * 4 + 3] * st.rot + 128);
      quatU[i] = (w | (x << 8) | (y << 16) | (z << 24)) >>> 0;
    }
  }
  // colors (keep static alpha byte)
  {
    const idx = dynIdx[ATTR_RGB];
    const c = cur[ATTR_RGB];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      const r = clamp8(c[j * 3] * st.rgb);
      const g = clamp8(c[j * 3 + 1] * st.rgb);
      const b = clamp8(c[j * 3 + 2] * st.rgb);
      rgbaU[i] = ((rgbaU[i] & 0xff000000) | (r | (g << 8) | (b << 16))) >>> 0;
    }
  }
  // dynamic alpha
  {
    const idx = dynIdx[ATTR_ALPHA];
    const c = cur[ATTR_ALPHA];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      rgbaU[i] = ((rgbaU[i] & 0x00ffffff) | (clamp8(c[j] * st.alpha) << 24)) >>> 0;
    }
  }
  // dynamic scale
  {
    const idx = dynIdx[ATTR_SCALE];
    const c = cur[ATTR_SCALE];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      scaleF[i * 4] = Math.exp(c[j * 3] * st.scale_log);
      scaleF[i * 4 + 1] = Math.exp(c[j * 3 + 1] * st.scale_log);
      scaleF[i * 4 + 2] = Math.exp(c[j * 3 + 2] * st.scale_log);
    }
  }
}

function requestFrame(frame: number) {
  if (!header || !stat) return;
  const t0 = performance.now();
  if (!rollTo(frame)) {
    pendingFrame = frame;
    const g = Math.floor(frame / header.gop);
    post({ type: 'miss', frame, gop: g });
    if (rangeMode) void priorityFetch(g);
    // fast path: if this GOP's keys are already here, show its keyframe now
    const keysOnly = gopKeys.get(g);
    if (keysOnly && curFrame !== header.gops[g].f0) {
      for (let a = 0; a < 5; a++) applyKey(a, keysOnly.keys);
      curFrame = header.gops[g].f0;
      emitFrame(curFrame, t0, /*approximate*/ true);
    }
    return;
  }
  pendingFrame = -1;
  emitFrame(frame, t0, false);
}

function emitFrame(frame: number, t0: number, approximate: boolean) {
  updateArrays();
  // copy into pooled transfer buffers
  const pb = pools.pos.pop() ?? new ArrayBuffer(posF.byteLength);
  const qb = pools.quat.pop() ?? new ArrayBuffer(quatU.byteLength);
  const cb = pools.rgba.pop() ?? new ArrayBuffer(rgbaU.byteLength);
  new Float32Array(pb).set(posF);
  new Uint32Array(qb).set(quatU);
  new Uint32Array(cb).set(rgbaU);
  post(
    { type: 'frame', frame, approximate, pos: pb, quat: qb, rgba: cb, decodeMs: performance.now() - t0 },
    [pb, qb, cb]
  );
}

/** Two-phase priority fetch for a seek target: TOC+keys prefix first (show the
 *  keyframe immediately), then the delta payloads (roll to the exact frame). */
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
      // show the keyframe for a still-pending seek right away
      if (pendingFrame >= 0 && Math.floor(pendingFrame / header.gop) === g) {
        const keysOnly = gopKeys.get(g)!;
        for (let a = 0; a < 5; a++) applyKey(a, keysOnly.keys);
        curFrame = span.f0;
        emitFrame(curFrame, performance.now(), true);
      }
    }
    // phase 2: rest of the chunk
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

// ---- depth sort (16-bit counting sort, antimatter15 style) ---------------
let counts = new Uint32Array(65536);

function sort(viewProj: Float32Array) {
  if (!header || !posF) {
    // static state not built yet — tell main so it can retry (avoids a stuck sortInFlight)
    post({ type: 'sorted-skip' });
    return;
  }
  const t0 = performance.now();
  const n = header.n;
  const vz0 = viewProj[2],
    vz1 = viewProj[6],
    vz2 = viewProj[10];
  const depths = new Int32Array(n);
  let mn = Infinity,
    mx = -Infinity;
  for (let i = 0; i < n; i++) {
    const d = ((vz0 * posF[i * 4] + vz1 * posF[i * 4 + 1] + vz2 * posF[i * 4 + 2]) * 4096) | 0;
    depths[i] = d;
    if (d < mn) mn = d;
    if (d > mx) mx = d;
  }
  const range = mx - mn || 1;
  const scale = 65535 / range;
  counts.fill(0);
  const buckets = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const b = ((depths[i] - mn) * scale) | 0;
    buckets[i] = b;
    counts[b]++;
  }
  // back-to-front: far (large positive view-space z after proj row? we want descending depth)
  // prefix sum from the FAR end so larger depth draws first
  let acc = 0;
  for (let b = 65535; b >= 0; b--) {
    const c = counts[b];
    counts[b] = acc;
    acc += c;
  }
  const ib = pools.sort.pop() ?? new ArrayBuffer(n * 4);
  const indices = new Uint32Array(ib);
  for (let i = 0; i < n; i++) indices[counts[buckets[i]]++] = i;
  post({ type: 'sorted', indices: ib, sortMs: performance.now() - t0 }, [ib]);
}
