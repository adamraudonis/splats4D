# splat4d Benchmarks

Test asset: **juggle** from Dynamic 3D Gaussians (Luiten et al., 3DV 2024) —
tracked gaussians of a person juggling inside the CMU Panoptic dome.
**150 frames @ 20 fps (7.5 s) × 336,568 splats**, converted to antimatter15
`.splat` frames (32 B/splat).

Hardware: Apple Silicon, 10 cores, 16 GB RAM. All encodes via
`converter` (Rust, rayon + zstd), all bounds verified exhaustively
(decode every splat of every frame, compare against source).

## Baselines

| baseline | size | ratio |
|---|---:|---:|
| raw `.splat` series (input) | 1,615.5 MB | 1.0× |
| gzip -9, single frame | 10.22 / 10.77 MB | 1.05× |
| zstd-19, single frame (per-frame shipping) | 10.16 / 10.77 MB | 1.06× |
| zstd-3 of concatenated series | 1,519.2 MB | 1.06× |
| **zstd-19 `--long=27` of concatenated series** (best generic lossless) | **642.0 MB** | **2.5×** |
| TC3DGS (published, same data class, no hard bounds, needs GPU opt.) | — | 40–67× |

Raw splat data barely compresses losslessly — f32 mantissas are effectively
random. Cross-frame matching (`--long`) finds the static background but still
pays for float-identical re-encoding.

## Per-sequence results (including gzip baselines)

Eight sequences from three independent sources: Dynamic 3D Gaussians (CMU
Panoptic dome: juggle/boxes/softball/tennis), Neu3D cooking scenes via
SpacetimeGaussians/splaTV (flame = backyard BBQ at night, sear = indoor
kitchen chef), and Technicolor (birthday = party table, 659k splats).
2-second clips @ 20 fps, plus the full juggle. "each frame gzip-9" = the sum
of gzipping every .splat file individually (i.e. shipping gzipped frames);
concatenated gzip is no better — gzip's 32 KB window can't see across 10 MB
frames. splat4d rows are the default preset (±5 mm / ±4 / ±1/128 / ±2 %),
bounds verified on every splat of every frame. Machine-readable copy:
[benchmarks.json](benchmarks.json).

| sequence | frames×splats | raw | each frame gzip-9 | concat gzip-9 | concat zstd-19 --long | splat4d (±5mm/±4) | vs gzip | splat4d +denoise |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| birthday_2s | 40×659,456 | 844 MB | 724 MB (1.17×) | 724 MB (1.17×) | 334 MB (2.5×) | **23.3 MB (36.3×)** | 31.1× smaller | 23.3 MB (36.3×) |
| boxes_2s | 40×351,422 | 450 MB | 423 MB (1.06×) | 423 MB (1.06×) | 184 MB (2.5×) | **21.5 MB (20.9×)** | 19.7× smaller | 16.0 MB (28.2×) |
| flame_2s | 40×333,824 | 427 MB | 367 MB (1.16×) | 367 MB (1.16×) | 146 MB (2.9×) | **10.4 MB (41.0×)** | 35.2× smaller | 10.4 MB (41.0×) |
| juggle (full 7.5 s) | 150×336,568 | 1616 MB | 1515 MB (1.07×) | 1515 MB (1.07×) | 642 MB (2.5×) | **63.9 MB (25.3×)** | 23.7× smaller | 43.5 MB (37.1×) |
| juggle_2s | 40×336,568 | 431 MB | 405 MB (1.06×) | 405 MB (1.06×) | 177 MB (2.4×) | **20.7 MB (20.8×)** | 19.5× smaller | 15.1 MB (28.6×) |
| sear_2s | 40×108,544 | 139 MB | 119 MB (1.16×) | 119 MB (1.16×) | 47 MB (3.0×) | **5.8 MB (23.8×)** | 20.5× smaller | 5.8 MB (23.8×) |
| softball_2s | 40×335,567 | 430 MB | 404 MB (1.06×) | 404 MB (1.06×) | 175 MB (2.4×) | **20.6 MB (20.9×)** | 19.6× smaller | 15.1 MB (28.5×) |
| tennis_2s | 40×333,076 | 426 MB | 401 MB (1.06×) | 401 MB (1.06×) | 176 MB (2.4×) | **20.8 MB (20.5×)** | 19.3× smaller | 15.2 MB (28.0×) |

Note the split: the CMU sequences carry heavy per-frame color noise (a
training artifact of that dataset), which the default bound faithfully
preserves — `--denoise-colors` buys them ~1.4×. The splaTV-derived scenes
(flame/sear/birthday) have clean, stable colors, land at 24–41× out of the
box, and gain nothing from denoising — confirming color noise, not the codec,
limits the CMU numbers.

## splat4d results (this project)

| preset | bounds (pos / color / rot / scale) | size | ratio | encode time* |
|---|---|---:|---:|---:|
| fine | ±2 mm / ±2 lvl / lossless / ±1 % | 86.2 MB | 18.7× | 10.8 s |
| **default** | ±5 mm / ±4 lvl / ±1/128 / ±2 % | **63.9 MB** | **25.3×** | 11.6 s |
| default, `--zstd-level 19` | ±5 mm / ±4 / ±1/128 / ±2 % | 60.2 MB | 26.9× | 31.9 s |
| default + `--denoise-colors` | same, vs median-5 color reference | 43.5 MB | 37.1× | 8.4 s |
| coarse | ±10 mm / ±8 lvl / ±2/128 / ±5 % | 50.5 MB | 32.0× | 10.1 s |
| **coarse + denoise** | same, vs median-5 color reference | **33.4 MB** | **48.3×** | 7.9 s |

\* wall-clock on 10 cores, excluding the optional verification pass (+0.8 s).
Throughput at default: **~132–157 MB/s of input**; peak RSS 4.1 GB.
The transform pipeline alone (no entropy coding) runs at 1.7 GB/s — encode
time is ~95 % zstd.

Stream breakdown at default (63.9 MB): color deltas 47.7 MB (75 % — this
dataset's colors carry heavy training noise), position deltas 4.2 MB,
rotation deltas 4.4 MB, static base (full background + frame-0 state)
3.78 MB, masks ~0.
Static fractions: position 80.1 %, rotation 80.5 %, scale 100 %, opacity
100 %, color 4.1 %.

## Theoretical minimum

Order-0 Shannon entropy of the emitted byte planes (the residual information
after quantize → hold → temporal/spatial delta → zigzag → shuffle):

| preset | entropy floor | actual | efficiency |
|---|---:|---:|---:|
| default | 66.3 MB | 63.9 MB | **104 %** (zstd's context modeling beats the order-0 bound) |
| default + denoise | 43.4 MB | 43.5 MB | 100 % |
| coarse + denoise | 31.5 MB | 33.4 MB | 94 % |

The codec is at (or beyond) the memoryless limit of its own symbol streams —
further gains require better transforms (motion prediction, context/rANS
coding), not a better back-end compressor.

## Error-bound verification (the headline guarantee)

`splat4d encode` decodes every frame and compares all 336,568 × 150 splats
against the source; `splat4d verify` does the same standalone. Default preset:

| attribute | bound | max observed error |
|---|---|---|
| position | ±5 mm | 5.000 mm ✓ |
| color | ±4 levels | 4 ✓ |
| opacity | ±4 levels | 0 ✓ |
| rotation | ±1/128 per quat component | 1 ✓ |
| scale | ±2 % relative | 0.995 % ✓ |

Errors are structural maxima of the quantizer — never exceeded, by
construction (quantize-then-delta on integers; deadband checks enforce the
bound before every emitted symbol).

## Interoperability with existing splat viewers

Single frames reconstructed from a `.splat4d` (`splat4d decode --frame N`) are
standard antimatter15 `.splat` files. Verified by loading a decoded flame
frame (333,824 splats, 10.7 MB) into third-party viewers:

| viewer | result |
|---|---|
| antimatter15/splat (the format's reference WebGL viewer) | loads via `?url=`, correct splat count, renders the scene correctly at 77 fps |
| PlayCanvas SuperSplat editor (local build of `playcanvas/supersplat@main`) | imports via `?load=`, status bar reports 333,824 splats, renders and orbits with no console errors |

So any `.splat4d` can be exploded back into frames consumable by the existing
ecosystem, and conversely anything that emits `.splat` frames can feed the
encoder.

## Viewer (raw WebGPU, no WebGL fallback)

The renderer is a line-by-line WebGPU port of the antimatter15/splat WebGL
viewer (`viewer/public/webgpu`, generated by `tools/make_webgpu_port.py`),
verified pixel-identical to it with `viewer/public/compare.html`. The
streaming 4D viewer reuses that exact pipeline (same WGSL math, rgba32uint
texture layout, under-blending, ascending counting sort) and is itself
diffed against the port rendering Rust-decoded frames
(`viewer/public/compare4d.html`): across wide/person/flame poses and frames
0/29/30/39 of the flame scene, mean |Δ| ≤ 0.74 with zero pixels off by more
than 16/255 (residual = sort tie-breaking between equal-depth splats).

Measured in Chrome, 336,568 splats:

| metric | local | throttled 50 Mbps |
|---|---:|---:|
| time to full first view (header + 3.78 MB static section) | **141–157 ms** | **791 ms** |
| seek into unbuffered region → keyframe on screen | — | **145 ms** |
| seek into unbuffered region → exact frame | — | 2.5 s |
| full 63.9 MB buffered | ~1 s | ~10.5 s |

Runtime: **60–61 fps** during playback; worker frame decode 2.5–27 ms;
worker depth sort (16-bit counting sort) 1–25 ms, triggered only on view
rotation or frame change; per-frame GPU upload is one `writeTexture` of the
dynamic band of the rgba32uint data texture (only rows containing dynamic
splats — the encoder orders fully-static splats first).

The 145 ms scrub response comes from the keys-first chunk layout: a seek
fetches the chunk's TOC+keys prefix (~10 % of the chunk), shows that
keyframe instantly, and rolls forward when the delta payloads land —
the same trick video players use.

## Pass/fail vs the project goals

- **Conversion speed**: 1.6 GB → one file in 8–12 s (≥130 MB/s). PASS
- **Massive size win**: 25–48× vs raw, 10–19× vs the best generic lossless
  baseline, at or past the order-0 entropy floor of its streams. PASS
- **Fast loading**: full scene on screen in 0.14 s local / 0.79 s @ 50 Mbps,
  playback starts immediately, YouTube-style buffer bar + instant scrubbing. PASS
- **Deterministic bounds**: verified exhaustively on every splat of every
  frame, tunable per attribute from the CLI. PASS

## Reproduce

```bash
# data
uv run --with remotezip python - <<'EOF'
from remotezip import RemoteZip
with RemoteZip('https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip') as z:
    z.extract('output/pretrained/juggle/params.npz', 'data/raw/')
EOF
uv run --with numpy python tools/gen_splat_frames.py data/raw/output/pretrained/juggle/params.npz data/frames/juggle

# encode + verify + report
cargo build --release --manifest-path converter/Cargo.toml
./converter/target/release/splat4d encode -i data/frames/juggle -o data/out/juggle.splat4d --report report.json

# view
cp data/out/juggle.splat4d viewer/public/
cd viewer && npm install && npm run dev
# throttled streaming demo:
node tools/serve_throttled.mjs data/out 8901 50
# open http://localhost:5173/?file=http%3A%2F%2Flocalhost%3A8901%2Fjuggle.splat4d
```
