#!/usr/bin/env python3
"""Prototype of the .splat4d codec on a directory of antimatter15 .splat frames.

Measures every design alternative on real data and verifies deterministic
error bounds end-to-end. The Rust converter implements the winning design.

Core mechanisms (all deterministic L-inf guarantees):
  * quantize with step 2b (floats) / 2b+1 (ints) => reconstruction error <= b
  * static classification: candidate = quantized midrange of true values;
    static iff candidate is within bound of BOTH true min and max (exact check)
  * deadband "hold" encoding for dynamic tracks: stored bin only changes when
    the true value would violate the bound vs the held value (suppresses
    quantization flicker; deltas become mostly zero; bound holds by check)
  * GOP structure: keyframe = absolute held bins, P-frames = exact integer
    deltas of held bins (quantize-then-delta => zero drift)

Usage:
  uv run --with numpy --with zstandard python tools/prototype_codec.py data/frames/juggle \
      --pos-mm 5 --color-levels 2 --rot-steps 0 --scale-pct 2 --gop 30
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import zstandard

ZSTD_LEVEL = 19


# ---------------------------------------------------------------- helpers
def zz(d):
    """zigzag encode signed ints -> unsigned"""
    d = d.astype(np.int64)
    return np.where(d >= 0, d * 2, -d * 2 - 1)


def entropy_bytes(symbols) -> float:
    """order-0 Shannon entropy lower bound in BYTES for an int array"""
    if symbols.size == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()) * symbols.size / 8


def byte_planes(a: np.ndarray) -> bytes:
    """split little-endian int array into byte planes (Blosc-style shuffle)"""
    b = np.ascontiguousarray(a).view(np.uint8).reshape(a.size, a.dtype.itemsize)
    return b.T.tobytes()


def zst(data: bytes, level=ZSTD_LEVEL) -> int:
    return len(zstandard.ZstdCompressor(level=level, threads=8).compress(data))


def part1by2(x):
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton3(p: np.ndarray) -> np.ndarray:
    q = p - p.min(axis=0)
    return part1by2(q[:, 0]) | (part1by2(q[:, 1]) << np.uint64(1)) | (part1by2(q[:, 2]) << np.uint64(2))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--pos-mm", type=float, default=5.0, help="max abs position error, millimeters")
    ap.add_argument("--color-levels", type=int, default=2, help="max abs color/alpha error, 8-bit levels")
    ap.add_argument("--rot-steps", type=int, default=0, help="max abs quat component error, 1/128 units (0 = lossless vs input grid)")
    ap.add_argument("--scale-pct", type=float, default=2.0, help="max relative scale error, percent")
    ap.add_argument("--gop", type=int, default=30, help="frames per GOP (keyframe interval)")
    ap.add_argument("--fast", action="store_true", help="zstd level 3 for quick iteration")
    args = ap.parse_args()
    level = 3 if args.fast else ZSTD_LEVEL

    t_start = time.time()
    fdir = Path(args.frames_dir)
    manifest = json.loads((fdir / "frames.json").read_text())
    files = [fdir / f["file"] for f in manifest["frames"]]
    T = len(files)

    # bounds and steps
    b_pos = args.pos_mm / 1000.0
    s_pos = 2 * b_pos
    b_col = args.color_levels          # integer domain
    s_col = 2 * b_col + 1
    b_rot = args.rot_steps             # integer domain, u8-128 centered
    s_rot = 2 * b_rot + 1
    b_ls = np.log(1 + args.scale_pct / 100.0)
    s_ls = 2 * b_ls

    # ---- parse all frames into true-value arrays ----
    t0 = time.time()
    n_probe = Path(files[0]).stat().st_size // 32
    N = n_probe
    pos = np.empty((T, N, 3), dtype=np.float32)
    ls = np.empty((T, N, 3), dtype=np.float32)
    col = np.empty((T, N, 4), dtype=np.int16)
    rot = np.empty((T, N, 4), dtype=np.int16)   # sign-canonicalized, centered at 0
    for t, f in enumerate(files):
        r = np.frombuffer(f.read_bytes(), dtype=np.uint8).reshape(-1, 32)
        assert r.shape[0] == N, "v1 requires equal splat count per frame"
        pos[t] = r[:, 0:12].view(np.float32).reshape(N, 3)
        ls[t] = np.log(np.maximum(r[:, 12:24].view(np.float32).reshape(N, 3), 1e-9))
        col[t] = r[:, 24:28].astype(np.int16)
        rt = r[:, 28:32].astype(np.int16) - 128
        if t > 0:
            flip = (rt * rot[t - 1]).sum(axis=1) < 0
            rt[flip] *= -1
        rot[t] = rt
    raw_total = T * N * 32
    print(f"parsed T={T} frames N={N} raw={raw_total/1e6:.1f}MB in {time.time()-t0:.1f}s")

    sizes, ent = {}, {}

    # ---- static classification: candidate = quantized midrange, exact check ----
    def classify_f(v, step, bound):
        vmin, vmax = v.min(axis=0), v.max(axis=0)
        cand = np.round((vmin + vmax) / 2 / step) * step
        ok = ((cand >= vmax - bound - 1e-7) & (cand <= vmin + bound + 1e-7)).all(axis=1)
        return ok, np.round(cand / step).astype(np.int32)

    def classify_i(v, step, bound):
        vmin, vmax = v.min(axis=0), v.max(axis=0)
        cand_bin = np.floor_divide((vmin + vmax) // 2 + step // 2, step)
        cand = cand_bin * step
        ok = ((cand >= vmax - bound) & (cand <= vmin + bound)).all(axis=1)
        return ok, cand_bin.astype(np.int32)

    pos_stat, pos_base = classify_f(pos, s_pos, b_pos)
    ls_stat, ls_base = classify_f(ls, s_ls, b_ls)
    col_stat, col_base = classify_i(col, s_col, b_col)
    rot_stat, rot_base = classify_i(rot, s_rot, b_rot)
    print(f"static: pos {pos_stat.mean()*100:.1f}%  col {col_stat.mean()*100:.1f}%  "
          f"rot {rot_stat.mean()*100:.1f}%  scale {ls_stat.mean()*100:.1f}%")

    # ---- ordering: dynamism group, morton within group ----
    dyn_key = ((~pos_stat).astype(np.uint8) * 4 + (~rot_stat).astype(np.uint8) * 2
               + (~col_stat).astype(np.uint8) + (~ls_stat).astype(np.uint8) * 8)
    q0 = np.round(pos[0] / s_pos).astype(np.int64)
    order = np.lexsort((morton3(q0), dyn_key))
    pos, ls, col, rot = pos[:, order], ls[:, order], col[:, order], rot[:, order]
    pos_stat, ls_stat, col_stat, rot_stat = pos_stat[order], ls_stat[order], col_stat[order], rot_stat[order]
    pos_base, ls_base, col_base, rot_base = pos_base[order], ls_base[order], col_base[order], rot_base[order]
    groups = {int(g): int((dyn_key[order] == g).sum()) for g in np.unique(dyn_key)}
    print(f"  dynamism groups (bits scale,pos,rot,col): { {f'{g:04b}': n for g, n in groups.items()} }")

    # ---- hold-encoding (deadband) for a dynamic attribute ----
    def hold_encode_f(v, base_bins, dyn_mask, step, bound):
        """float attr: returns held bins (T,Nd,C) int32"""
        vd = v[:, dyn_mask]
        held = np.empty(vd.shape, dtype=np.int32)
        held[0] = np.round(vd[0] / step).astype(np.int32)
        cur = held[0].copy()
        for t in range(1, vd.shape[0]):
            viol = (np.abs(vd[t] - cur * step) > bound).any(axis=1)
            fresh = np.round(vd[t][viol] / step).astype(np.int32)
            cur[viol] = fresh
            held[t] = cur
        return held

    def hold_encode_i(v, base_bins, dyn_mask, step, bound):
        vd = v[:, dyn_mask]
        held = np.empty(vd.shape, dtype=np.int32)
        held[0] = np.floor_divide(vd[0].astype(np.int32) + step // 2, step)
        cur = held[0].copy()
        for t in range(1, vd.shape[0]):
            viol = (np.abs(vd[t] - cur * step) > bound).any(axis=1)
            fresh = np.floor_divide(vd[t][viol].astype(np.int32) + step // 2, step)
            cur[viol] = fresh
            held[t] = cur
        return held

    # ---- GOP encode dense tracks ----
    def gop_encode(held, name):
        total, e_total, key_bytes = 0, 0.0, 0
        for g0 in range(0, T, args.gop):
            g1 = min(g0 + args.gop, T)
            key = zz(held[g0])
            kb = zst(byte_planes(key.astype(np.uint32)), level)
            key_bytes += kb
            total += kb
            e_total += entropy_bytes(key)
            if g1 - g0 > 1:
                deltas = zz(np.diff(held[g0:g1], axis=0))
                mx = deltas.max() if deltas.size else 0
                dt = np.uint8 if mx < 256 else (np.uint16 if mx < 65536 else np.uint32)
                total += zst(byte_planes(deltas.astype(dt)), level)
                e_total += entropy_bytes(deltas)
        sizes[name] = total
        ent[name] = e_total
        return key_bytes

    t0 = time.time()
    pos_held = hold_encode_f(pos, pos_base, ~pos_stat, s_pos, b_pos)
    kb = gop_encode(pos_held, "dyn.pos")
    print(f"dyn.pos: {int((~pos_stat).sum())} splats, {sizes['dyn.pos']/1e6:.2f}MB (keys {kb/1e6:.2f}MB)  [{time.time()-t0:.0f}s]")

    t0 = time.time()
    rot_held = hold_encode_i(rot, rot_base, ~rot_stat, s_rot, b_rot)
    kb = gop_encode(rot_held, "dyn.rot")
    print(f"dyn.rot: {int((~rot_stat).sum())} splats, {sizes['dyn.rot']/1e6:.2f}MB (keys {kb/1e6:.2f}MB)  [{time.time()-t0:.0f}s]")

    t0 = time.time()
    col_held = hold_encode_i(col, col_base, ~col_stat, s_col, b_col)
    kb = gop_encode(col_held, "dyn.col")
    print(f"dyn.col: {int((~col_stat).sum())} splats, {sizes['dyn.col']/1e6:.2f}MB (keys {kb/1e6:.2f}MB)  [{time.time()-t0:.0f}s]")

    ls_held = None
    if not ls_stat.all():
        ls_held = hold_encode_f(ls, ls_base, ~ls_stat, s_ls, b_ls)
        kb = gop_encode(ls_held, "dyn.scale")
        print(f"dyn.scale: {int((~ls_stat).sum())} splats, {sizes['dyn.scale']/1e6:.2f}MB")

    # ---- sparse color alternative: events at violations only ----
    t0 = time.time()
    n_events = 0
    idx_streams, val_streams = [], []
    cur = col_held[0].copy()
    for t in range(1, T):
        diff = (col_held[t] != cur).any(axis=1)
        idx = np.nonzero(diff)[0]
        n_events += idx.size
        idx_streams.append(np.diff(idx.astype(np.int64), prepend=0))
        val_streams.append(zz(col_held[t][idx] - cur[idx]))
        cur[idx] = col_held[t][idx]
    if n_events:
        ev_idx = zz(np.concatenate(idx_streams))
        ev_val = np.concatenate(val_streams)
        sparse_bytes = (zst(byte_planes(ev_idx.astype(np.uint32)), level)
                        + zst(byte_planes(ev_val.astype(np.uint16)), level))
        sizes["dyn.col.sparse(alt)"] = sparse_bytes
        print(f"sparse color: {n_events} events ({n_events/(T-1):.0f}/frame), "
              f"{sparse_bytes/1e6:.2f}MB vs dense {sizes['dyn.col']/1e6:.2f}MB  [{time.time()-t0:.0f}s]")

    # ---- base streams ----
    t0 = time.time()
    dpos = np.diff(pos_base, axis=0, prepend=pos_base[:1])
    sizes["base.pos"] = zst(byte_planes(zz(dpos).astype(np.uint32)), level)
    ent["base.pos"] = entropy_bytes(zz(dpos))
    sizes["base.col"] = zst(byte_planes(col_base.astype(np.int16)), level)
    ent["base.col"] = entropy_bytes(col_base)
    sizes["base.rot"] = zst(byte_planes(rot_base.astype(np.int16)), level)
    ent["base.rot"] = entropy_bytes(rot_base)
    dls = np.diff(ls_base, axis=0, prepend=ls_base[:1])
    sizes["base.scale"] = zst(byte_planes(zz(dls).astype(np.uint16)), level)
    ent["base.scale"] = entropy_bytes(zz(dls))
    sizes["masks"] = zst(np.packbits(np.stack([pos_stat, col_stat, rot_stat, ls_stat])).tobytes(), level)
    print(f"base streams encoded [{time.time()-t0:.0f}s]")

    # ---- totals ----
    dense_total = sum(v for k, v in sizes.items() if "(alt)" not in k)
    ent_total = sum(ent.values())
    print(f"\n==== stream sizes (zstd-{level}) ====")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
        e = f"  (entropy min {ent[k]/1e6:6.2f}MB)" if k in ent else ""
        print(f"  {k:22s} {v/1e6:8.2f} MB{e}")
    print(f"\nRAW INPUT: {raw_total/1e6:.1f} MB")
    print(f"TOTAL .splat4d (dense col): {dense_total/1e6:.2f} MB   ratio {raw_total/dense_total:.0f}x")
    if "dyn.col.sparse(alt)" in sizes:
        alt = dense_total - sizes["dyn.col"] + sizes["dyn.col.sparse(alt)"]
        print(f"TOTAL .splat4d (sparse col): {alt/1e6:.2f} MB   ratio {raw_total/alt:.0f}x")
    print(f"order-0 entropy lower bound of emitted symbols: {ent_total/1e6:.2f} MB")
    print(f"baselines: zstd-19 --long full series = 642.0 MB (2.5x); zstd-19 per frame ~ 10.16/10.77 MB")

    # ---- end-to-end bound verification (exact decoder semantics) ----
    t0 = time.time()
    max_e = dict(pos=0.0, col=0, rot=0, ls=0.0)
    dyn_p, dyn_c, dyn_r, dyn_s = ~pos_stat, ~col_stat, ~rot_stat, ~ls_stat
    rp = np.empty((N, 3), np.float32)
    rc = np.empty((N, 4), np.int32)
    rr = np.empty((N, 4), np.int32)
    rs = np.empty((N, 3), np.float32)
    for t in range(T):
        rp[:] = pos_base * s_pos
        rp[dyn_p] = pos_held[t] * s_pos
        rc[:] = col_base * s_col
        rc[dyn_c] = col_held[t] * s_col
        rr[:] = rot_base * s_rot
        rr[dyn_r] = rot_held[t] * s_rot
        rs[:] = ls_base * s_ls
        if ls_held is not None:
            rs[dyn_s] = ls_held[t] * s_ls
        max_e["pos"] = max(max_e["pos"], float(np.abs(rp - pos[t]).max()))
        max_e["col"] = max(max_e["col"], int(np.abs(rc - col[t]).max()))
        max_e["rot"] = max(max_e["rot"], int(np.abs(rr - rot[t]).max()))
        max_e["ls"] = max(max_e["ls"], float(np.abs(rs - ls[t]).max()))
    print(f"\n==== bound verification, all {T} frames ({time.time()-t0:.0f}s) ====")
    print(f"  pos   max err {max_e['pos']*1000:.3f} mm      (bound {args.pos_mm} mm)")
    print(f"  color max err {max_e['col']} levels        (bound {b_col})")
    print(f"  rot   max err {max_e['rot']} /128 units    (bound {b_rot})")
    print(f"  scale max rel err {(np.exp(max_e['ls'])-1)*100:.2f}%     (bound {args.scale_pct}%)")
    ok = (max_e["pos"] <= b_pos + 1e-6 and max_e["col"] <= b_col
          and max_e["rot"] <= b_rot and max_e["ls"] <= b_ls + 1e-6)
    print("  BOUNDS " + ("VERIFIED ✓" if ok else "VIOLATED ✗"))
    print(f"\ntotal prototype time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
