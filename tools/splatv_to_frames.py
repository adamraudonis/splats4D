#!/usr/bin/env python3
"""Convert an antimatter15 .splatv (SpacetimeGaussians-derived dynamic splat)
into a series of antimatter15 .splat frames by evaluating the motion model at
sampled timestamps.

.splatv layout (from splaTV hybrid.js, MIT):
  u32 magic 0x674b, u32 json_len, json chunks [{type:"splat", size, texwidth,
  texheight, cameras}], texdata (texwidth*texheight*16 bytes, 16 u32/splat):
    w0..w2   x,y,z float32
    w3,w4    quat (w,x,y,z) as 4 half floats
    w5,w6    linear scale (sx,sy),(sz,0) as halves
    w7       RGBA u8 (color 0..255, alpha = sigmoid(opacity)*255)
    w8..w12  motion_0..8 as halves: cubic position coeffs v1,v2,v3
    w13,w14  omega_0..3 as halves: linear quaternion velocity
    w15      (trbf_center, exp(trbf_scale)) as halves

  Evaluation at normalized time t in [0,1]:
    dt   = t - trbf_center
    pos  = xyz + v1*dt + v2*dt^2 + v3*dt^3
    quat = normalize(quat + omega*dt)
    alpha *= exp(-(dt/trbf_scale_exp)^2)     (viewer culls below 0.02)

Usage: uv run --with numpy python tools/splatv_to_frames.py data/raw/splatv/flame.splatv data/frames/flame_2s 40
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

FPS = 20.0


def main(src: str, out_dir: str, t_frames: int = 40) -> None:
    t0 = time.time()
    raw = Path(src).read_bytes()
    magic, json_len = struct.unpack_from("<II", raw, 0)
    assert magic == 0x674B, f"bad magic {magic:#x}"
    chunks = json.loads(raw[8 : 8 + json_len])
    splat_chunk = next(c for c in chunks if c["type"] == "splat")
    n = splat_chunk["size"] // 64
    data = np.frombuffer(raw, dtype=np.uint32, count=n * 16, offset=8 + json_len).reshape(n, 16)
    print(f"{src}: {n} splats, tex {splat_chunk['texwidth']}x{splat_chunk['texheight']}")

    f32 = data.view(np.float32)
    u8 = data.view(np.uint8).reshape(n, 64)
    halves = data.view(np.float16).reshape(n, 32)  # word w -> halves [2w]=lo, [2w+1]=hi

    xyz = f32[:, 0:3].astype(np.float64)
    quat = halves[:, 6:10].astype(np.float64)  # w3,w4 -> (w,x,y,z)
    scale = halves[:, 10:13].astype(np.float32)  # w5,w6 -> (sx,sy,sz) linear
    rgba = u8[:, 28:32].copy()  # w7
    motion = halves[:, 16:25].astype(np.float64)  # w8..w12 lo/hi -> motion_0..8
    omega = halves[:, 26:30].astype(np.float64)  # w13,w14 -> omega_0..3
    trbf_c = halves[:, 30].astype(np.float64)  # w15 lo
    trbf_s = np.maximum(halves[:, 31].astype(np.float64), 1e-6)  # w15 hi (already exp'ed)

    v1, v2, v3 = motion[:, 0:3], motion[:, 3:6], motion[:, 6:9]
    base_alpha = rgba[:, 3].astype(np.float64)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = np.zeros((n, 32), dtype=np.uint8)
    rec[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
    rec[:, 24:27] = rgba[:, 0:3]

    manifest = {"fps": FPS, "count": t_frames, "num_splats": int(n), "frames": []}
    # camera hint for the viewer, from the first embedded camera
    cams = splat_chunk.get("cameras") or []
    if cams:
        c = cams[0]
        R = np.array(c["rotation"], dtype=np.float64)
        p = np.array(c["position"], dtype=np.float64)
        fwd = R[2] if abs(np.linalg.norm(R[2]) - 1) < 0.1 else R[:, 2]
        manifest["camera"] = {
            "position": [round(float(v), 4) for v in p],
            "target": [round(float(v), 4) for v in (p + fwd * 2.0)],
            "up": [0, -1, 0],
        }

    for i in range(t_frames):
        t = i / max(t_frames - 1, 1)
        dt = t - trbf_c
        pos = xyz + v1 * dt[:, None] + v2 * (dt**2)[:, None] + v3 * (dt**3)[:, None]
        q = quat + omega * dt[:, None]
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        topacity = np.exp(-((dt / trbf_s) ** 2))
        # the reference splaTV shader HARD-discards splats below 0.02 temporal
        # opacity; without this, splats extrapolated far along their motion
        # polynomials linger as faint giant streaks
        vis = topacity >= 0.02
        alpha = np.where(vis, np.clip(np.round(base_alpha * topacity), 0, 255), 0).astype(np.uint8)

        rec[:, 0:12] = pos.astype(np.float32).view(np.uint8).reshape(n, 12)
        rec[:, 27] = alpha
        rec[:, 28:32] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
        name = f"frame_{i:04d}.splat"
        (out / name).write_bytes(rec.tobytes())
        manifest["frames"].append({"file": name, "timestamp": round(i / FPS, 6)})

    (out / "frames.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {t_frames} frames x {n} splats = {t_frames*n*32/1e6:.1f} MB in {time.time()-t0:.1f}s -> {out}")
    if "camera" in manifest:
        print(f"camera hint: {manifest['camera']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
