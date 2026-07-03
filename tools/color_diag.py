#!/usr/bin/env python3
"""Diagnose color noise structure + measure hierarchical (per-GOP) static potential."""
import sys
import time
from pathlib import Path

import numpy as np


def main():
    t0 = time.time()
    npz = np.load("data/raw/output/pretrained/juggle/params.npz")
    cf = npz["rgb_colors"]  # (T,N,3) float
    T, N = cf.shape[0], cf.shape[1]
    seg_fg = npz["seg_colors"][:, 0] > 0.5
    alpha = 1 / (1 + np.exp(-npz["logit_opacities"][:, 0]))
    print(f"float colors: min {cf.min():.3f} max {cf.max():.3f}, outside [0,1]: {((cf<0)|(cf>1)).mean()*100:.2f}% of values")

    # per-splat temporal std in u8 levels (float domain, x255) — sample channels R
    r = cf[:, :, 0] * 255.0
    std = r.std(axis=0)
    print(f"per-splat R-channel temporal std (levels): median {np.median(std):.2f}  "
          f"p25 {np.percentile(std,25):.2f}  p75 {np.percentile(std,75):.2f}  p95 {np.percentile(std,95):.2f}")
    print(f"  bg only: median {np.median(std[~seg_fg]):.2f} p95 {np.percentile(std[~seg_fg],95):.2f}")
    print(f"  fg only: median {np.median(std[seg_fg]):.2f} p95 {np.percentile(std[seg_fg],95):.2f}")
    print(f"  high-alpha(>0.7) bg: median {np.median(std[(~seg_fg)&(alpha>0.7)]):.2f}")

    # is it white noise or drift? lag-1 autocorrelation of deltas
    d = np.diff(r[:, ::13], axis=0)
    ac = (d[:-1] * d[1:]).mean() / (d * d).mean()
    print(f"lag-1 autocorr of per-frame color deltas: {ac:.3f}  (-0.5 = pure white noise around mean, 0 = random walk)")

    # global per-frame mean drift?
    gm = r.mean(axis=1)
    print(f"global mean R per frame: min {gm.min():.2f} max {gm.max():.2f} (range {gm.max()-gm.min():.3f} levels)")

    # hierarchical static: fraction static per window size (range <= 2*bound)
    print("\n-- windowed color-static fraction (all 3 channels, range<=2b in window) --")
    c255 = cf * 255.0
    for W in [15, 30, 50, 75, 150]:
        for b in [2, 4]:
            fracs = []
            for w0 in range(0, T - W + 1, W):
                win = c255[w0:w0 + W]
                rng = (win.max(axis=0) - win.min(axis=0)).max(axis=1)
                fracs.append((rng <= 2 * b).mean())
            print(f"  window {W:3d} frames, ±{b}: {np.mean(fracs)*100:5.1f}% static-in-window")

    # same for positions at 5mm
    print("\n-- windowed pos-static fraction --")
    m = npz["means3D"]
    for W in [30, 150]:
        fracs = []
        for w0 in range(0, T - W + 1, W):
            win = m[w0:w0 + W]
            rng = (win.max(axis=0) - win.min(axis=0)).max(axis=2 - 1)  # max over xyz of range
            fracs.append((rng <= 0.010).mean())
        print(f"  window {W:3d} frames, ±5mm: {np.mean(fracs)*100:5.1f}%")

    print(f"\n{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
