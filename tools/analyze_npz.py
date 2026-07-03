#!/usr/bin/env python3
"""Temporal-redundancy analysis of a Dynamic3DGaussians params.npz.

Answers the questions that drive the .splat4d codec design:
  1. What fraction of splats is static under a given position bound?
  2. For dynamic splats, what is the entropy of quantized per-frame deltas?
  3. How much do colors / rotations actually change over time?
  4. Projected .splat4d size at various error bounds (Shannon lower bound).

Usage: uv run --with numpy tools/analyze_npz.py data/raw/output/pretrained/juggle/params.npz
"""
import sys
import time

import numpy as np


def entropy_bits(symbols: np.ndarray) -> float:
    """Order-0 Shannon entropy in bits/symbol of an integer array."""
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def zigzag(d: np.ndarray) -> np.ndarray:
    return np.where(d >= 0, d.astype(np.int64) * 2, -d.astype(np.int64) * 2 - 1)


def main(npz_path: str) -> None:
    t0 = time.time()
    data = np.load(npz_path)
    means = data["means3D"].astype(np.float32)  # (T,N,3)
    T, N = means.shape[0], means.shape[1]
    print(f"T={T} frames, N={N} splats, load {time.time()-t0:.1f}s")

    seg_fg = data["seg_colors"][:, 0] > 0.5
    print(f"seg foreground fraction: {seg_fg.mean():.4f} ({seg_fg.sum()} splats)")

    # ---- scene scale / units sanity ----
    lo, hi = means[0].min(axis=0), means[0].max(axis=0)
    print(f"frame0 AABB: min {lo}, max {hi}, extent {hi-lo} (CMU panoptic => meters)")

    # ---- position temporal statistics ----
    ref = means[0]
    max_disp = np.zeros(N, dtype=np.float32)
    step_mag_sum = np.zeros(N, dtype=np.float32)
    for t in range(1, T):
        d = means[t] - ref
        np.maximum(max_disp, np.abs(d).max(axis=1), out=max_disp)  # L-inf vs frame 0
        step_mag_sum += np.abs(means[t] - means[t - 1]).max(axis=1)
    print("\n-- static fraction (L-inf displacement vs frame0 stays within bound over ALL frames) --")
    for bound_mm in [1, 2, 5, 10, 20, 50]:
        b = bound_mm / 1000.0
        frac = float((max_disp <= b).mean())
        print(f"  bound +-{bound_mm:3d}mm: {frac*100:6.2f}% static ({int((max_disp<=b).sum())} splats)")

    # agreement between motion-static and seg background
    for bound_mm in [5, 10]:
        b = bound_mm / 1000.0
        static = max_disp <= b
        print(f"  bound {bound_mm}mm: static&fg {(static & seg_fg).sum()}, moving&bg {((~static) & ~seg_fg).sum()}")

    # ---- delta entropy for dynamic splats at various quantization steps ----
    print("\n-- per-frame position delta entropy for DYNAMIC splats (quantize-then-delta, zigzag) --")
    for bound_mm in [2, 5, 10]:
        step = 2 * bound_mm / 1000.0
        dyn = max_disp > bound_mm / 1000.0
        nd = int(dyn.sum())
        if nd == 0:
            continue
        # subsample frames for speed if huge
        q_prev = np.round(means[0][dyn] / step).astype(np.int64)
        all_syms = []
        for t in range(1, T):
            q = np.round(means[t][dyn] / step).astype(np.int64)
            all_syms.append(zigzag(q - q_prev).ravel())
            q_prev = q
        syms = np.concatenate(all_syms)
        H = entropy_bits(syms)
        zero_frac = float((syms == 0).mean())
        big = float((syms > 30).mean())
        bytes_pf = nd * 3 * H / 8
        print(f"  bound +-{bound_mm}mm: dyn={nd} H={H:.3f} bits/sym zero={zero_frac*100:.1f}% |sym|>15steps={big*100:.2f}% => ~{bytes_pf/1e3:.1f} KB/frame, {bytes_pf*(T-1)/1e6:.1f} MB total deltas")

    del means

    # ---- rotation temporal statistics ----
    rots = data["unnorm_rotations"].astype(np.float32)
    rots /= np.linalg.norm(rots, axis=2, keepdims=True)
    # canonicalize sign (q and -q identical): align to previous frame
    ref = rots[0].copy()
    max_dq = np.zeros(N, dtype=np.float32)
    for t in range(1, T):
        q = rots[t]
        sign = np.sign((q * ref).sum(axis=1, keepdims=True))
        sign[sign == 0] = 1
        q = q * sign
        np.maximum(max_dq, np.abs(q - ref).max(axis=1), out=max_dq)
    print("\n-- rotation: max per-component quat drift vs frame0 (input .splat grid = 1/128 = 0.0078) --")
    for q8 in [1, 2, 4, 8]:
        b = q8 / 128.0
        print(f"  within {q8}/128: {(max_dq <= b).mean()*100:6.2f}%")

    # delta entropy of 8-bit quantized quat components for dynamic-position splats
    dyn = max_disp > 0.005
    q_prev = np.round(rots[0][dyn] * 128).astype(np.int64)
    all_syms = []
    for t in range(1, T):
        q = rots[t][dyn]
        sign = np.sign((q * rots[t - 1][dyn]).sum(axis=1, keepdims=True))
        sign[sign == 0] = 1
        qq = np.round(q * sign * 128).astype(np.int64)
        all_syms.append(zigzag(qq - q_prev).ravel())
        q_prev = qq
    syms = np.concatenate(all_syms)
    print(f"  dyn splats 8-bit quat delta entropy: {entropy_bits(syms):.3f} bits/sym, zero {float((syms==0).mean())*100:.1f}%")
    del rots

    # ---- color temporal statistics ----
    colors = data["rgb_colors"].astype(np.float32)
    c0 = np.clip(np.round(colors[0] * 255), 0, 255).astype(np.int16)
    max_dc = np.zeros(N, dtype=np.int16)
    for t in range(1, T):
        c = np.clip(np.round(colors[t] * 255), 0, 255).astype(np.int16)
        np.maximum(max_dc, np.abs(c - c0).max(axis=1).astype(np.int16), out=max_dc)
    print("\n-- color: max per-channel change vs frame0 (8-bit levels) --")
    for lv in [0, 1, 2, 4, 8, 16]:
        print(f"  within +-{lv:2d} levels: {(max_dc <= lv).mean()*100:6.2f}%")

    print(f"\ntotal analysis time {time.time()-t0:.1f}s")
    raw = T * N * 32
    print(f"raw .splat series size: {raw/1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/output/pretrained/juggle/params.npz")
