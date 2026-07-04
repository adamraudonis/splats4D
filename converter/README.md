# splat4d

Compress a folder of [antimatter15 `.splat`](https://github.com/antimatter15/splat)
gaussian-splat frames into **one small, streamable `.splat4d` file** with
**deterministic, tunable error bounds**. This package ships the native Rust CLI —
`pip install splats4d` puts the `splat4d` executable on your PATH.

```bash
pip install splats4d

# defaults: --pos-mm 2 --color-levels 4 --rot-steps 0 (exact rotation) --scale-pct 2
splat4d encode -i frames_dir -o scene.splat4d
```

`frames_dir` contains `frame_0000.splat`, `frame_0001.splat`, … plus an
optional `frames.json` manifest (`{"fps": 20, "frames": [{"file": "...",
"timestamp": 0.0}, …]}`); without a manifest, files are taken in sorted order
at `--fps`.

## What you get

- **25–48× smaller** than the raw frame series on real captures
  (10–19× smaller than gzipping the same frames), encoding at ~140 MB/s.
- **Hard error bounds, verified**: every attribute of every splat in every
  decoded frame is within your chosen bound (±mm position, ±levels color,
  ±quaternion steps, ±% scale). Each encode decodes everything back and
  asserts the bounds before it reports success.
- **Streaming & seeking**: closed keyframe GOPs indexed by absolute byte
  ranges — clients play and scrub via plain HTTP Range requests straight from
  an object store (S3/GCS/R2). A WebGPU viewer is included in the repo.

## Commands

```bash
splat4d encode -i <dir> -o <file>   # compress (see --help for all bounds/tuning flags)
splat4d verify <file> -i <dir>      # exhaustive bound check against the source frames
splat4d decode <file> --frame 42 -o f42.splat   # reconstruct one frame
splat4d info <file>                 # print header / chunk index
```

Format spec, benchmarks, and the browser viewer:
[github.com/adamraudonis/splats4D](https://github.com/adamraudonis/splats4D) ·
[project page](https://adamraudonis.github.io/splats4D/)
