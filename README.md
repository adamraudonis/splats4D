# splats4D

**`.splat4d`** — a streamable 4D gaussian splat format with **deterministic,
tunable error bounds**, a fast Rust encoder, and a **three.js WebGPU**
streaming viewer with YouTube-style buffering and instant scrubbing.

Takes a time series of [antimatter15 `.splat`](https://github.com/antimatter15/splat)
frames and produces one small seekable file.

**Headline numbers** (juggle, 150 frames × 336,568 splats, 1,615 MB raw):

| | |
|---|---|
| output size | **63.9 MB (25.3×)** default · **33.4 MB (48.3×)** coarse+denoise |
| best generic lossless baseline (zstd-19 `--long`) | 642 MB (2.5×) |
| encode time (10 cores) | 8–12 s (~130–150 MB/s input) |
| bounds, verified on every splat of every frame | ±5 mm pos · ±4/255 color · ±1/128 rot · ±2 % scale |
| viewer: full scene first view | 141 ms local · **791 ms @ 50 Mbps** |
| viewer: scrub into unbuffered region | **keyframe in 145 ms** |
| playback | 60 fps @ 336k splats, WebGPU only (no WebGL fallback) |

## How it works

- **Per-attribute static/dynamic split** — 80 % of splats (the background)
  never move within the bound; they're stored once. Ground-truth check:
  the split agrees perfectly with the dataset's fg/bg segmentation.
- **Error-bounded quantization** (SZ/ZFP-style): step = 2×bound, so the bound
  holds by construction. Everything after quantization is integer math —
  no drift, bit-identical decode in Rust and JS.
- **Deadband "hold" tracks**: a stored value only changes when the true value
  would violate the bound against it — kills quantization flicker, makes
  temporal deltas mostly zero.
- **H.265-style closed GOPs** (default 30 frames): keyframe + integer delta
  P-frames per chunk; each chunk decodes independently → HTTP-range seeking.
  Keys are laid out before deltas inside each chunk, so a seek can show the
  keyframe after fetching ~10 % of the chunk.
- **Morton ordering + byte-plane shuffle + zstd** per stream; output lands at
  ~100 % of the order-0 entropy of its own symbols.

## Layout

```
converter/   Rust CLI: splat4d encode | verify | decode | info
viewer/      Vite + three.js r185 WebGPURenderer viewer (worker decode + sort)
tools/       Python prototypes/analysis, frame generator, throttled test server
docs/        FORMAT.md (spec) · BENCHMARKS.md (numbers & methodology)
```

## Quick start

```bash
# 1. get data (Dynamic 3D Gaussians pretrained output, ~2 GB for one sequence)
uv run --with remotezip python - <<'EOF'
from remotezip import RemoteZip
with RemoteZip('https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip') as z:
    z.extract('output/pretrained/juggle/params.npz', 'data/raw/')
EOF
uv run --with numpy python tools/gen_splat_frames.py  # -> data/frames/juggle/*.splat

# 2. encode (prints sizes, entropy floor, timings; verifies every bound)
cargo build --release --manifest-path converter/Cargo.toml
./converter/target/release/splat4d encode -i data/frames/juggle -o data/out/juggle.splat4d \
    --pos-mm 5 --color-levels 4 --rot-steps 1 --scale-pct 2

# 3. view (WebGPU required: Chrome 113+, Safari 26+, Firefox 141+)
cd viewer && npm install && npm run dev
```

The dev viewer includes:
- **compression sliders** (position/color/rotation/scale bounds, GOP, denoise,
  zstd effort) — "Encode & load" re-runs the Rust encoder via a Vite dev API
  (~1–3 s at the fast effort, cached per parameter set) and hot-swaps the
  result into the player, showing size/ratio and the verified max errors;
- **⇔ compare** (or press `C`) — split-screen against the original
  uncompressed `.splat` frames with a draggable divider, rendered through the
  same WebGPU pipeline with a shared camera and depth sort.

## Serving from an object store (S3 / GCS / R2 / any HTTP host)

`.splat4d` is designed for plain **HTTP Range requests** — no server logic, no
manifests, no video container. The viewer (and any conforming client) fetches:

1. `bytes=0-262143` → magic + header JSON (all offsets in the file are absolute);
2. one range for the static section (a few MB) → the complete scene renders;
3. one range per GOP chunk for playback/prefetch;
4. on seek into unbuffered time: the chunk-prefix range (TOC + key streams,
   ~10 % of the chunk) → keyframe on screen immediately, then the delta
   payloads → exact frame.

If the server ignores Range (returns 200), the client transparently falls back
to progressive sequential streaming — still playable, just no random-access
fetch. For browser access to a bucket, CORS must allow the `Range` header and
expose `Content-Range`:

```json
[{ "AllowedMethods": ["GET", "HEAD"],
   "AllowedOrigins": ["https://your-site"],
   "AllowedHeaders": ["Range"],
   "ExposeHeaders":  ["Content-Range", "Content-Length", "Accept-Ranges"] }]
```

Store the object with **no `Content-Encoding`** — payloads are already
zstd-compressed inside the container, so range math stays byte-exact.
Point the viewer at it with `?file=https://bucket.example.com/scene.splat4d`.

## Python package

`pip install splat4d` — a pure-Python (numpy) encoder producing the identical
format, for pipelines where the Rust binary isn't convenient. Same CLI:
`splat4d encode -i frames_dir -o out.splat4d`. Source in [python/](python/).

See [docs/FORMAT.md](docs/FORMAT.md) for the byte-level spec and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full results, including
per-sequence gzip baselines. Project page:
[adamraudonis.github.io/splats4D](https://adamraudonis.github.io/splats4D/).
