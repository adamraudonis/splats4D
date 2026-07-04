# `.splat4d`

**A streamable 4D gaussian splat format with tunable error bounds.**

**[Project page & live demo →](https://adamraudonis.github.io/splats4D/)** ·
[PyPI (`splats4d`)](https://pypi.org/project/splats4d/) ·
[Format spec](docs/FORMAT.md) ·
[Benchmarks](docs/BENCHMARKS.md)

A time series of [antimatter15 `.splat`](https://github.com/antimatter15/splat)
frames → one small, seekable file. Every attribute of every splat in every
decoded frame stays within a user-chosen bound of the source — pointwise and
deterministic, verified by the encoder on every run.

**Headline numbers** (juggle, 150 frames × 336,568 splats, 1,615 MB raw):

| | |
|---|---|
| output size | **83.4 MB (19.4×)** default · **40.9 MB (39.5×)** coarse+denoise |
| best generic lossless baseline (zstd-19 `--long`) | 642 MB (2.5×) |
| encode time (10 cores) | 1–3 s (~640 MB/s of input) |
| default bounds | ±2 mm position · ±4/255 color · **exact** rotation · ±2 % scale |
| viewer: full first view | 141 ms local · **791 ms @ 50 Mbps** |
| viewer: seek into unbuffered region | **keyframe in 145 ms** |
| playback | 60 fps @ 336k splats (WebGPU) |

## How it works

- **Per-attribute static/dynamic split** — most splats are background that
  never moves within the bound; they're stored once. The entire background of
  a 1.6 GB sequence costs a few MB.
- **Error-bounded quantization** (SZ/ZFP-style): step = 2×bound, so the bound
  holds by construction. Everything after quantization is integer math —
  no drift, bit-identical decode in Rust and JS.
- **Deadband "hold" tracks**: a stored value only changes when the true value
  would violate the bound against it — kills quantization flicker, makes
  temporal deltas mostly zero.
- **H.265-style closed GOPs** (default 30 frames): keyframe + integer-delta
  P-frames per chunk; each chunk decodes independently → HTTP-range seeking.
  Keys are laid out before deltas, so a seek can show the keyframe after
  fetching ~10 % of a chunk.
- **Morton ordering + byte-plane shuffle + zstd** per stream; output lands at
  ~100 % of the order-0 entropy of its own symbols.

The bundled viewer is raw WebGPU — a line-by-line port of the
[antimatter15/splat](https://github.com/antimatter15/splat) renderer,
pixel-diff verified against it.

## Repo layout

```
converter/   Rust CLI: splat4d encode | verify | decode | info  (also the PyPI package)
viewer/      Vite + raw-WebGPU streaming viewer (worker decode + sort)
tools/       dataset converters, benchmark runner, throttled test server
docs/        project page · FORMAT.md (spec) · BENCHMARKS.md (numbers & methodology)
```

## Quick start

```bash
# encode a folder of frame_0000.splat, frame_0001.splat, …
pip install splats4d
splat4d encode -i frames_dir -o out.splat4d
# defaults: --pos-mm 2 --color-levels 4 --rot-steps 0 --scale-pct 2 --gop 30 --zstd-level 3
# every encode verifies its own bounds and prints sizes, ratios, and the entropy floor

# or build from source
cargo build --release --manifest-path converter/Cargo.toml

# run the dev viewer (WebGPU: Chrome 113+, Safari 26+, Firefox 141+ on Windows)
cd viewer && npm install && npm run dev
```

The dev viewer includes live **compression sliders** (re-encodes through a dev
API and hot-swaps the result, showing size, ratio, and verified max errors)
and a **⇔ compare** mode — split-screen against the original uncompressed
frames with a draggable divider, rendered through the same pipeline.

## Stream from an object store (S3 / GCS / R2 / any HTTP host)

`.splat4d` is designed for plain **HTTP Range requests** — no server logic, no
manifests, no video container. A client fetches:

1. `bytes=0-262143` → magic + header JSON (all offsets in the file are absolute);
2. one range for the static section (a few MB) → the complete scene renders;
3. one range per GOP chunk for playback/prefetch;
4. on seek into unbuffered time: the chunk-prefix range (TOC + key streams)
   → keyframe on screen immediately, then the delta payloads → exact frame.

If the server ignores Range (returns 200), the client falls back to
progressive sequential streaming. For browser access to a bucket, CORS must
allow the `Range` header and expose `Content-Range`:

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

`pip install splats4d` ships the native Rust CLI as a platform wheel (maturin
`bin` bindings) — after install, the `splat4d` executable is on your PATH at
full native speed. Packaging lives in
[converter/pyproject.toml](converter/pyproject.toml); wheels are built and
published by [.github/workflows/wheels.yml](.github/workflows/wheels.yml).

---

MIT · built on the shoulders of: antimatter15/splat (format),
[Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/) (data),
SZ/ZFP (error-bounded quantization), H.264/H.265 (GOP structure), zstd.
