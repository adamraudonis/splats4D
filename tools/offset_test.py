#!/usr/bin/env python3
"""Test: does a per-frame global color offset shrink the dominant color stream?"""
import json
import time
from pathlib import Path

import numpy as np
import zstandard


def zz(d):
    d = d.astype(np.int64)
    return np.where(d >= 0, d * 2, -d * 2 - 1)


def byte_planes(a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(a.size, a.dtype.itemsize)
    return b.T.tobytes()


def zst(data, level=3):
    return len(zstandard.ZstdCompressor(level=level, threads=8).compress(data))


fdir = Path("data/frames/juggle")
manifest = json.loads((fdir / "frames.json").read_text())
files = [fdir / f["file"] for f in manifest["frames"]]
T = len(files)
N = files[0].stat().st_size // 32
rgb = np.empty((T, N, 3), dtype=np.int16)
for t, f in enumerate(files):
    r = np.frombuffer(f.read_bytes(), dtype=np.uint8).reshape(-1, 32)
    rgb[t] = r[:, 24:27].astype(np.int16)

b = 4
s = 2 * b + 1
GOP = 30


def encode(use_offset):
    t0 = time.time()
    # global per-frame offset vs frame 0 (median of per-splat difference)
    if use_offset:
        g = np.array([np.median(rgb[t] - rgb[0], axis=0) for t in range(T)]).round().astype(np.int16)
    else:
        g = np.zeros((T, 3), np.int16)
    # static classification with offset-adjusted values
    adj = rgb - g[:, None, :]
    vmin, vmax = adj.min(axis=0), adj.max(axis=0)
    cand = (vmin + vmax) // 2
    stat = ((cand >= vmax - b) & (cand <= vmin + b)).all(axis=1)
    vd = adj[:, ~stat]
    held = np.empty(vd.shape, dtype=np.int32)
    held[0] = np.floor_divide(vd[0].astype(np.int32) + s // 2, s)
    cur = held[0].copy()
    for t in range(1, T):
        viol = (np.abs(vd[t] - cur * s) > b).any(axis=1)
        cur[viol] = np.floor_divide(vd[t][viol].astype(np.int32) + s // 2, s)
        held[t] = cur
    total = 0
    for g0 in range(0, T, GOP):
        g1 = min(g0 + GOP, T)
        total += zst(byte_planes(zz(held[g0]).astype(np.uint32)))
        d = zz(np.diff(held[g0:g1], axis=0))
        total += zst(byte_planes(d.astype(np.uint8 if d.max() < 256 else np.uint16)))
    print(f"offset={use_offset}: static {stat.mean()*100:.1f}%  dyn.col {total/1e6:.1f} MB  [{time.time()-t0:.0f}s]")
    return total


base = encode(False)
off = encode(True)
print(f"win: {(1-off/base)*100:.1f}%")
