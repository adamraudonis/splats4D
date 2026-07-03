#!/usr/bin/env python3
"""Codec design experiments: parameter sweep + color-coding strategies.

Parses frames once, then evaluates many codec configs cheaply.
Decides the defaults for the Rust implementation.

Usage:
  uv run --with numpy --with zstandard python tools/experiments.py data/frames/juggle
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import zstandard


def zz(d):
    d = d.astype(np.int64)
    return np.where(d >= 0, d * 2, -d * 2 - 1)


def entropy_bytes(symbols) -> float:
    if symbols.size == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()) * symbols.size / 8


def byte_planes(a: np.ndarray) -> bytes:
    b = np.ascontiguousarray(a).view(np.uint8).reshape(a.size, a.dtype.itemsize)
    return b.T.tobytes()


def zst(data: bytes, level=3) -> int:
    return len(zstandard.ZstdCompressor(level=level, threads=8).compress(data))


def part1by2(x):
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton3(p):
    q = p - p.min(axis=0)
    return part1by2(q[:, 0]) | (part1by2(q[:, 1]) << np.uint64(1)) | (part1by2(q[:, 2]) << np.uint64(2))


# ---------------- generic per-attribute codec (classify + hold + GOP) ----------
def encode_attr_f(v, step, bound, gop, level):
    """float attr (T,N,C). returns (static_mask, base_bins, total_bytes)"""
    T = v.shape[0]
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    cand = np.round((vmin + vmax) / 2 / step) * step
    stat = ((cand >= vmax - bound - 1e-7) & (cand <= vmin + bound + 1e-7)).all(axis=1)
    base = np.round(cand / step).astype(np.int32)
    vd = v[:, ~stat]
    total = 0
    if vd.shape[1]:
        held = np.empty(vd.shape, dtype=np.int32)
        held[0] = np.round(vd[0] / step)
        cur = held[0].copy()
        for t in range(1, T):
            viol = (np.abs(vd[t] - cur * step) > bound).any(axis=1)
            cur[viol] = np.round(vd[t][viol] / step)
            held[t] = cur
        for g0 in range(0, T, gop):
            g1 = min(g0 + gop, T)
            total += zst(byte_planes(zz(held[g0]).astype(np.uint32)), level)
            if g1 - g0 > 1:
                d = zz(np.diff(held[g0:g1], axis=0))
                dt = np.uint8 if d.max() < 256 else np.uint16
                total += zst(byte_planes(d.astype(dt)), level)
    return stat, base, total


def encode_attr_i(v, step, bound, gop, level, weight=None, wbound=None):
    """int attr (T,N,C). weight/wbound: optional per-splat contribution weighting:
    violation iff |err| > bound OR (weighted: |err|*w > wbound with per-splat w in [0,1])
    -> effective per-splat bound = max(bound, wbound/w). Deterministic."""
    T = v.shape[0]
    if weight is not None:
        eff = np.maximum(bound, np.floor(wbound / np.maximum(weight, 1e-3)))
        eff = np.minimum(eff, 64)[:, None]  # cap: never worse than +-64 levels
    else:
        eff = np.full((v.shape[1], 1), bound, dtype=np.float64)
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    # weighted mode keeps the GLOBAL quantization step (from `bound`) but uses the
    # per-splat effective bound for the deadband/static checks.
    cand_bin = np.floor_divide((vmin + vmax) // 2 + step // 2, step)
    cand = cand_bin * step
    stat = ((cand >= vmax - eff) & (cand <= vmin + eff)).all(axis=1)
    base = cand_bin.astype(np.int32)
    vd = v[:, ~stat]
    effd = eff[~stat]
    total = 0
    if vd.shape[1]:
        held = np.empty(vd.shape, dtype=np.int32)
        held[0] = np.floor_divide(vd[0].astype(np.int32) + step // 2, step)
        cur = held[0].copy()
        for t in range(1, T):
            viol = (np.abs(vd[t] - cur * step) > effd).any(axis=1)
            cur[viol] = np.floor_divide(vd[t][viol].astype(np.int32) + step // 2, step)
            held[t] = cur
        for g0 in range(0, T, gop):
            g1 = min(g0 + gop, T)
            total += zst(byte_planes(zz(held[g0]).astype(np.uint32)), level)
            if g1 - g0 > 1:
                d = zz(np.diff(held[g0:g1], axis=0))
                dt = np.uint8 if d.max() < 256 else np.uint16
                total += zst(byte_planes(d.astype(dt)), level)
    return stat, base, total


def base_size(base, spatial_delta, level, dtype=np.int16):
    if spatial_delta:
        d = np.diff(base, axis=0, prepend=base[:1])
        return zst(byte_planes(zz(d).astype(np.uint32)), level)
    return zst(byte_planes(base.astype(dtype)), level)


def main():
    fdir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/frames/juggle")
    level = 3
    manifest = json.loads((fdir / "frames.json").read_text())
    files = [fdir / f["file"] for f in manifest["frames"]]
    T = len(files)
    t0 = time.time()
    N = Path(files[0]).stat().st_size // 32
    pos = np.empty((T, N, 3), dtype=np.float32)
    ls = np.empty((T, N, 3), dtype=np.float32)
    rgb = np.empty((T, N, 3), dtype=np.int16)
    alp = np.empty((T, N, 1), dtype=np.int16)
    rot = np.empty((T, N, 4), dtype=np.int16)
    for t, f in enumerate(files):
        r = np.frombuffer(f.read_bytes(), dtype=np.uint8).reshape(-1, 32)
        pos[t] = r[:, 0:12].view(np.float32).reshape(N, 3)
        ls[t] = np.log(np.maximum(r[:, 12:24].view(np.float32).reshape(N, 3), 1e-9))
        rgb[t] = r[:, 24:27].astype(np.int16)
        alp[t] = r[:, 27:28].astype(np.int16)
        rt = r[:, 28:32].astype(np.int16) - 128
        if t > 0:
            flip = (rt * rot[t - 1]).sum(axis=1) < 0
            rt[flip] *= -1
        rot[t] = rt
    raw_total = T * N * 32
    print(f"parsed {raw_total/1e6:.0f}MB in {time.time()-t0:.1f}s")

    # order once (approx: 5mm grid morton; grouping differs per config but effect is small)
    order = np.argsort(morton3(np.round(pos[0] / 0.01).astype(np.int64)))
    pos, ls, rgb, alp, rot = pos[:, order], ls[:, order], rgb[:, order], alp[:, order], rot[:, order]

    a0 = alp[0, :, 0].astype(np.float32) / 255.0
    print(f"alpha distribution: <0.1: {(a0<0.1).mean()*100:.1f}%  <0.35: {(a0<0.35).mean()*100:.1f}%  "
          f"<0.7: {(a0<0.7).mean()*100:.1f}%  >=0.7: {(a0>=0.7).mean()*100:.1f}%")

    # ---- YCoCg-R diagnostic: per-frame delta entropy comparison ----
    R, G, B = rgb[..., 0].astype(np.int32), rgb[..., 1].astype(np.int32), rgb[..., 2].astype(np.int32)
    Co = R - B
    tmp = B + (Co >> 1)
    Cg = G - tmp
    Y = tmp + (Cg >> 1)
    for name, chans in [("RGB", [R, G, B]), ("YCoCg", [Y, Co, Cg])]:
        e = sum(entropy_bytes(zz(np.diff(c[:, ::7], axis=0))) for c in chans) * 7
        print(f"unquantized per-frame delta entropy {name}: {e/1e6:.1f} MB (est, 1/7 sample)")

    GOP = 30

    def run_config(pmm, clv, rst, spct, weighted=None, label=""):
        t0 = time.time()
        b_pos = pmm / 1000
        b_ls = np.log(1 + spct / 100)
        stat_p, base_p, dyn_p = encode_attr_f(pos, 2 * b_pos, b_pos, GOP, level)
        stat_s, base_s, dyn_s = encode_attr_f(ls, 2 * b_ls, b_ls, GOP, level)
        stat_r, base_r, dyn_r = encode_attr_i(rot, 2 * rst + 1, rst, GOP, level)
        if weighted:
            stat_c, base_c, dyn_c = encode_attr_i(rgb, 2 * clv + 1, clv, GOP, level, weight=a0, wbound=weighted)
        else:
            stat_c, base_c, dyn_c = encode_attr_i(rgb, 2 * clv + 1, clv, GOP, level)
        stat_a, base_a, dyn_a = encode_attr_i(alp, 2 * clv + 1, clv, GOP, level)
        base_total = (base_size(base_p, True, level) + base_size(base_s, True, level)
                      + base_size(base_r, False, level) + base_size(base_c, False, level)
                      + base_size(base_a, False, level))
        masks = zst(np.packbits(np.stack([stat_p, stat_s, stat_r, stat_c, stat_a])).tobytes(), level)
        total = base_total + masks + dyn_p + dyn_s + dyn_r + dyn_c + dyn_a
        print(f"pos±{pmm}mm col±{clv} rot±{rst} scale±{spct}% {label:14s} | "
              f"stat p/r/c: {stat_p.mean()*100:.0f}/{stat_r.mean()*100:.0f}/{stat_c.mean()*100:.0f}% | "
              f"dyn p/r/c/a: {dyn_p/1e6:.1f}/{dyn_r/1e6:.1f}/{dyn_c/1e6:.1f}/{dyn_a/1e6:.2f} | "
              f"base {base_total/1e6:.1f} | TOTAL {total/1e6:6.1f} MB  {raw_total/total:5.0f}x  [{time.time()-t0:.0f}s]")
        return total

    print("\n---- axis sweeps around center (pos5, col4, rot1, scale2) ----")
    for pmm in [2, 5, 10]:
        run_config(pmm, 4, 1, 2)
    for clv in [2, 4, 8]:
        run_config(5, clv, 1, 2)
    for rst in [0, 1, 2]:
        run_config(5, 4, rst, 2)

    print("\n---- contribution-weighted color (|alpha*err| <= beta, cap ±64) ----")
    for beta in [2, 4]:
        run_config(5, 2, 1, 2, weighted=beta, label=f"beta={beta}")
        run_config(5, 4, 1, 2, weighted=beta, label=f"beta={beta}")

    print("\n---- GOP sensitivity at defaults ----")
    for g in [15, 30, 60, 150]:
        GOP = g
        run_config(5, 4, 1, 2, label=f"gop={g}")


if __name__ == "__main__":
    main()
