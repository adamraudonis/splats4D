#!/usr/bin/env python3
"""Generate a series of antimatter15 .splat files from a Dynamic3DGaussians params.npz.

The params.npz (Luiten et al., Dynamic 3D Gaussians) contains tracked gaussians:
  means3D          (T, N, 3) float32  - positions per timestep
  rgb_colors       (T, N, 3) float32  - RGB in [0,1] per timestep (soft-fixed over time)
  unnorm_rotations (T, N, 4) float32  - quaternions (w,x,y,z), unnormalized
  log_scales       (N, 3)    float32  - static, exp() -> linear scale
  logit_opacities  (N, 1)    float32  - static, sigmoid() -> opacity
  seg_colors       (N, 3)    float32  - channel 0 > 0.5 means foreground (dynamic)

antimatter15 .splat record = 32 bytes:
  float32[3] position, float32[3] scale, uint8[4] RGBA, uint8[4] quaternion
  quaternion components stored as round(q*128 + 128), order (w,x,y,z), normalized.

Outputs frame_%04d.splat plus a frames.json manifest with timestamps (20 fps).

Usage: uv run --with numpy tools/gen_splat_frames.py data/raw/output/pretrained/juggle/params.npz data/frames/juggle
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

FPS = 20.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(npz_path: str, out_dir: str, max_frames: int | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = np.load(npz_path)
    means = data["means3D"]  # (T,N,3)
    colors = data["rgb_colors"]  # (T,N,3)
    rots = data["unnorm_rotations"]  # (T,N,4)
    log_scales = data["log_scales"]  # (N,3)
    logit_op = data["logit_opacities"]  # (N,1)
    seg = data["seg_colors"]  # (N,3)
    T, N = means.shape[0], means.shape[1]
    if max_frames:
        T = min(T, max_frames)
    print(f"loaded {npz_path}: T={means.shape[0]} N={N} (using T={T}), load {time.time()-t0:.1f}s")

    scales = np.exp(log_scales).astype(np.float32)  # (N,3) static
    alpha = np.clip(np.round(sigmoid(logit_op[:, 0]) * 255.0), 0, 255).astype(np.uint8)  # (N,)
    fg_frac = float((seg[:, 0] > 0.5).mean())
    print(f"foreground (dynamic) fraction: {fg_frac:.3f}")

    rec = np.zeros((N, 32), dtype=np.uint8)
    rec[:, 12:24] = scales.view(np.uint8).reshape(N, 12)
    rec[:, 27] = alpha

    manifest = {"fps": FPS, "count": T, "num_splats": int(N), "frames": []}
    gen_t0 = time.time()
    for t in range(T):
        rec[:, 0:12] = np.ascontiguousarray(means[t]).view(np.uint8).reshape(N, 12)
        c = np.clip(np.round(colors[t] * 255.0), 0, 255).astype(np.uint8)
        rec[:, 24:27] = c
        q = rots[t].astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rec[:, 28:32] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
        name = f"frame_{t:04d}.splat"
        (out / name).write_bytes(rec.tobytes())
        manifest["frames"].append({"file": name, "timestamp": round(t / FPS, 6)})
        if t % 25 == 0:
            print(f"  frame {t}/{T}")

    (out / "frames.json").write_text(json.dumps(manifest, indent=1))
    total_bytes = T * N * 32
    print(f"wrote {T} frames x {N} splats = {total_bytes/1e6:.1f} MB in {time.time()-gen_t0:.1f}s -> {out}")


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "data/raw/output/pretrained/juggle/params.npz"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/frames/juggle"
    mf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(npz, dst, mf)
