// .splat4d container parsing (mirrors converter/src/{streams,decode}.rs).
// Everything after quantization is integer math — decode is bit-deterministic.

import { decompress } from 'fzstd';

export interface Bounds {
  pos_m: number;
  scale_rel: number;
  rgb: number;
  alpha: number;
  rot: number;
}

export interface Steps {
  pos: number;
  base_pos: number;
  scale_log: number;
  base_scale_log: number;
  rgb: number;
  alpha: number;
  rot: number;
}

export interface GopSpan {
  offset: number;
  len: number;
  f0: number;
  f1: number;
  t0: number;
  t1: number;
}

export interface Header {
  n: number;
  t: number;
  fps: number;
  gop: number;
  bounds: Bounds;
  steps: Steps;
  dyn: { pos: number; rot: number; rgb: number; alpha: number; scale: number };
  groups: { mask: number; count: number }[];
  aabb: number[];
  denoised: boolean;
  static_section: { offset: number; len: number };
  gops: GopSpan[];
}

export const ATTR_POS = 0,
  ATTR_ROT = 1,
  ATTR_RGB = 2,
  ATTR_ALPHA = 3,
  ATTR_SCALE = 4;
export const KIND_KEY = 0,
  KIND_DELTA = 1;

export function parseHeader(buf: Uint8Array): { header: Header; headerEnd: number } {
  if (buf.length < 12) throw new Error('too short');
  if (buf[0] !== 0x53 || buf[1] !== 0x50 || buf[2] !== 0x34 || buf[3] !== 0x44)
    throw new Error('not a .splat4d file');
  const dv = new DataView(buf.buffer, buf.byteOffset);
  const version = dv.getUint16(4, true);
  if (version !== 1) throw new Error(`unsupported version ${version}`);
  const jsonLen = dv.getUint32(8, true);
  if (buf.length < 12 + jsonLen) throw new Error('need more header bytes');
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(12, 12 + jsonLen))) as Header;
  return { header, headerEnd: 12 + jsonLen };
}

export interface Stream {
  attr: number;
  kind: number;
  syms: Uint8Array | Uint16Array | Uint32Array; // zigzagged symbols
}

const STREAM_HDR = 11;

/** Read one stream at `off` within `buf`; returns [stream, nextOffset]. */
export function readStream(buf: Uint8Array, off: number): [Stream, number] {
  const dv = new DataView(buf.buffer, buf.byteOffset);
  const attr = buf[off];
  const kind = buf[off + 1];
  const width = buf[off + 2];
  const elems = dv.getUint32(off + 3, true);
  const comp = dv.getUint32(off + 7, true);
  const start = off + STREAM_HDR;
  const planes = decompress(buf.subarray(start, start + comp), new Uint8Array(elems * width));
  let syms: Uint8Array | Uint16Array | Uint32Array;
  if (width === 1) {
    syms = planes;
  } else if (width === 2) {
    const out = new Uint16Array(elems);
    for (let i = 0; i < elems; i++) out[i] = planes[i] | (planes[elems + i] << 8);
    syms = out;
  } else {
    const out = new Uint32Array(elems);
    for (let i = 0; i < elems; i++)
      out[i] =
        (planes[i] | (planes[elems + i] << 8) | (planes[2 * elems + i] << 16) | (planes[3 * elems + i] << 24)) >>> 0;
    syms = out;
  }
  return [{ attr, kind, syms }, start + comp];
}

export function unzigzag(z: number): number {
  return (z >>> 1) ^ -(z & 1);
}

export interface StaticState {
  masks: Uint8Array[]; // per attr: 1 = static, per splat
  basePos: Int32Array; // n*3 fine bins
  baseLs: Int32Array; // n*3
  baseRgb: Uint8Array; // n*3
  baseAlpha: Uint8Array; // n
  baseRot: Int16Array; // n*4 centered
}

function unbitmap(syms: ArrayLike<number>, n: number): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (syms[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

function unSpatialDelta(syms: ArrayLike<number>, c: number): Int32Array {
  const nn = syms.length / c;
  const out = new Int32Array(syms.length);
  const prev = new Int32Array(c);
  for (let i = 0; i < nn; i++)
    for (let k = 0; k < c; k++) {
      prev[k] += unzigzag(syms[i * c + k]);
      out[i * c + k] = prev[k];
    }
  return out;
}

export function parseStatic(buf: Uint8Array, header: Header): StaticState {
  // `buf` is the static section bytes (offset already applied by caller)
  const n = header.n;
  let off = 0;
  const next = () => {
    const [s, no] = readStream(buf, off);
    off = no;
    return s.syms;
  };
  const masks = [
    unbitmap(next(), n),
    unbitmap(next(), n),
    unbitmap(next(), n),
    unbitmap(next(), n),
    unbitmap(next(), n),
  ];
  const basePos = unSpatialDelta(next(), 3);
  const baseLs = unSpatialDelta(next(), 3);
  const baseRgbS = next();
  const baseRgb = new Uint8Array(baseRgbS.length);
  baseRgb.set(baseRgbS as Uint8Array);
  const baseAlphaS = next();
  const baseAlpha = new Uint8Array(baseAlphaS.length);
  baseAlpha.set(baseAlphaS as Uint8Array);
  const baseRotS = next();
  const baseRot = new Int16Array(baseRotS.length);
  for (let i = 0; i < baseRotS.length; i++) baseRot[i] = baseRotS[i] - 128;
  return { masks, basePos, baseLs, baseRgb, baseAlpha, baseRot };
}

export interface GopData {
  keys: (Uint8Array | Uint16Array | Uint32Array | null)[]; // per attr, zigzagged abs bins
  deltas: (Uint8Array | Uint16Array | Uint32Array | null)[]; // per attr, zigzagged, frame-major
}

/** Byte length of the chunk prefix holding the TOC + all key streams
 *  (keys always precede deltas within a chunk). `buf` needs only the TOC. */
export function gopKeysPrefixLen(buf: Uint8Array): number {
  const dv = new DataView(buf.buffer, buf.byteOffset);
  const count = dv.getUint16(0, true);
  let end = 2 + count * STREAM_HDR;
  let off = 2;
  let payload = end;
  for (let i = 0; i < count; i++) {
    const kind = buf[off + 1];
    const comp = dv.getUint32(off + 7, true);
    if (kind === KIND_KEY) end = payload + comp;
    payload += comp;
    off += STREAM_HDR;
  }
  return end;
}

/** Parse only the key streams from a chunk prefix (see gopKeysPrefixLen). */
export function parseGopKeys(buf: Uint8Array): GopData {
  const dv = new DataView(buf.buffer, buf.byteOffset);
  const count = dv.getUint16(0, true);
  let off = 2;
  const metas: { attr: number; kind: number; width: number; elems: number; comp: number }[] = [];
  for (let i = 0; i < count; i++) {
    metas.push({
      attr: buf[off],
      kind: buf[off + 1],
      width: buf[off + 2],
      elems: dv.getUint32(off + 3, true),
      comp: dv.getUint32(off + 7, true),
    });
    off += STREAM_HDR;
  }
  const keys: GopData['keys'] = [null, null, null, null, null];
  for (const m of metas) {
    if (m.kind !== KIND_KEY) break; // keys are a contiguous prefix
    const planes = decompress(buf.subarray(off, off + m.comp), new Uint8Array(m.elems * m.width));
    off += m.comp;
    keys[m.attr] = widen(planes, m.width, m.elems);
  }
  return { keys, deltas: [null, null, null, null, null] };
}

function widen(planes: Uint8Array, width: number, elems: number): Uint8Array | Uint16Array | Uint32Array {
  if (width === 1) return planes;
  if (width === 2) {
    const out = new Uint16Array(elems);
    for (let i = 0; i < elems; i++) out[i] = planes[i] | (planes[elems + i] << 8);
    return out;
  }
  const out = new Uint32Array(elems);
  for (let i = 0; i < elems; i++)
    out[i] =
      (planes[i] | (planes[elems + i] << 8) | (planes[2 * elems + i] << 16) | (planes[3 * elems + i] << 24)) >>> 0;
  return out;
}

export function parseGop(buf: Uint8Array): GopData {
  const dv = new DataView(buf.buffer, buf.byteOffset);
  const count = dv.getUint16(0, true);
  let off = 2;
  const metas: { attr: number; kind: number; width: number; elems: number; comp: number }[] = [];
  for (let i = 0; i < count; i++) {
    metas.push({
      attr: buf[off],
      kind: buf[off + 1],
      width: buf[off + 2],
      elems: dv.getUint32(off + 3, true),
      comp: dv.getUint32(off + 7, true),
    });
    off += STREAM_HDR;
  }
  const keys: GopData['keys'] = [null, null, null, null, null];
  const deltas: GopData['deltas'] = [null, null, null, null, null];
  for (const m of metas) {
    const planes = decompress(buf.subarray(off, off + m.comp), new Uint8Array(m.elems * m.width));
    off += m.comp;
    let syms: Uint8Array | Uint16Array | Uint32Array;
    if (m.width === 1) syms = planes;
    else if (m.width === 2) {
      const out = new Uint16Array(m.elems);
      for (let i = 0; i < m.elems; i++) out[i] = planes[i] | (planes[m.elems + i] << 8);
      syms = out;
    } else {
      const out = new Uint32Array(m.elems);
      for (let i = 0; i < m.elems; i++)
        out[i] =
          (planes[i] |
            (planes[m.elems + i] << 8) |
            (planes[2 * m.elems + i] << 16) |
            (planes[3 * m.elems + i] << 24)) >>>
          0;
      syms = out;
    }
    if (m.kind === KIND_KEY) keys[m.attr] = syms;
    else deltas[m.attr] = syms;
  }
  return { keys, deltas };
}
