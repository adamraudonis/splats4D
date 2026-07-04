#!/usr/bin/env python3
"""Benchmark every frame sequence: raw size, gzip baselines (each .splat
gzipped individually, and the concatenated series gzipped), zstd-19 --long
baseline, and splat4d encodes (default + denoised) with verified bounds.

Writes docs/benchmarks.json (consumed by the GitHub Pages site) and prints a
markdown table.

Usage: uv run python tools/benchmark_all.py [seq ...]   (default: all in data/frames)
"""
import gzip
import json
import zlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONVERTER = ROOT / "converter/target/release/splat4d"
FRAMES = ROOT / "data/frames"
OUT = ROOT / "data/out/bench"
DOCS = ROOT / "docs"


def gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9))


def gzip_concat_size(files: list[Path]) -> int:
    co = zlib.compressobj(9, zlib.DEFLATED, 31)  # wbits 31 = gzip container
    total = 0
    for f in files:
        total += len(co.compress(f.read_bytes()))
    return total + len(co.flush())


def zstd_long_size(files: list[Path]) -> int:
    cat = subprocess.Popen(["cat", *[str(f) for f in files]], stdout=subprocess.PIPE)
    z = subprocess.run(
        ["zstd", "-19", "-T0", "--long=27", "-c"],
        stdin=cat.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    cat.wait()
    return len(z.stdout)


def encode(seq_dir: Path, out: Path, extra: list[str]) -> dict:
    report = out.with_suffix(".json")
    subprocess.run(
        [str(CONVERTER), "encode", "-i", str(seq_dir), "-o", str(out), "--report", str(report), *extra],
        check=True,
        capture_output=True,
    )
    return json.loads(report.read_text())


def bench_sequence(seq: str, cached: dict | None = None) -> dict:
    seq_dir = FRAMES / seq
    manifest = json.loads((seq_dir / "frames.json").read_text())
    files = [seq_dir / f["file"] for f in manifest["frames"]]
    t, n = len(files), manifest["num_splats"]
    raw = sum(f.stat().st_size for f in files)
    print(f"[{seq}] {t} frames x {n} splats = {raw/1e6:.0f} MB raw")

    # generic baselines are deterministic — reuse them when re-benchmarking
    # after an encoder-preset change (pass --fresh-baselines to recompute)
    if cached and cached.get("raw_bytes") == raw:
        per_frame_gz = cached["gzip_per_frame_bytes"]
        concat_gz = cached["gzip_concat_bytes"]
        zstd_long = cached["zstd19_long_bytes"]
        print(f"[{seq}] baselines reused from docs/benchmarks.json")
    else:
        t0 = time.time()
        with ProcessPoolExecutor() as ex:
            per_frame_gz = sum(ex.map(gzip_size, files))
        print(f"[{seq}] per-frame gzip-9: {per_frame_gz/1e6:.1f} MB ({time.time()-t0:.0f}s)")

        t0 = time.time()
        concat_gz = gzip_concat_size(files)
        print(f"[{seq}] concat gzip-9: {concat_gz/1e6:.1f} MB ({time.time()-t0:.0f}s)")

        t0 = time.time()
        zstd_long = zstd_long_size(files)
        print(f"[{seq}] concat zstd-19 --long: {zstd_long/1e6:.1f} MB ({time.time()-t0:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    default = encode(seq_dir, OUT / f"{seq}_default.splat4d", [])
    denoised = encode(seq_dir, OUT / f"{seq}_denoised.splat4d", ["--denoise-colors"])
    print(f"[{seq}] splat4d default: {default['output']['bytes']/1e6:.1f} MB "
          f"({default['output']['ratio']:.1f}x, {default['times_s']['total']:.1f}s, "
          f"verified={default['verify']['ok'] if default.get('verify') else '?'})")
    print(f"[{seq}] splat4d denoised: {denoised['output']['bytes']/1e6:.1f} MB "
          f"({denoised['output']['ratio']:.1f}x)")

    return {
        "seq": seq,
        "frames": t,
        "splats": n,
        "fps": manifest["fps"],
        "raw_bytes": raw,
        "gzip_per_frame_bytes": per_frame_gz,
        "gzip_concat_bytes": concat_gz,
        "zstd19_long_bytes": zstd_long,
        "splat4d_default": {
            "bytes": default["output"]["bytes"],
            "encode_s": default["times_s"]["total"],
            "entropy_min_bytes": default["entropy_min_bytes"],
            "verify": default.get("verify"),
            "static_fracs": default["static_fracs"],
        },
        "splat4d_denoised": {
            "bytes": denoised["output"]["bytes"],
            "encode_s": denoised["times_s"]["total"],
            "verify": denoised.get("verify"),
        },
    }


def main():
    fresh = "--fresh-baselines" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--fresh-baselines"]
    seqs = args or sorted(
        d.name for d in FRAMES.iterdir() if (d / "frames.json").exists()
    )
    results = []
    existing = {}
    bench_path = DOCS / "benchmarks.json"
    if bench_path.exists():
        existing = {r["seq"]: r for r in json.loads(bench_path.read_text())["sequences"]}
    for seq in seqs:
        results.append(bench_sequence(seq, None if fresh else existing.get(seq)))
        existing[results[-1]["seq"]] = results[-1]

    all_results = sorted(existing.values(), key=lambda r: r["seq"])
    bench_path.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d"), "sequences": all_results}, indent=1))
    print(f"\nwrote {bench_path}")

    # markdown table
    print("\n| sequence | frames×splats | raw | each frame gzip-9 | concat gzip-9 | concat zstd-19 --long | splat4d (default: ±2mm/±4 color/rot exact/±2%) | vs gzip | splat4d +denoise |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        d = r["splat4d_default"]["bytes"]
        dn = r["splat4d_denoised"]["bytes"]
        print(
            f"| {r['seq']} | {r['frames']}×{r['splats']:,} | {r['raw_bytes']/1e6:.0f} MB "
            f"| {r['gzip_per_frame_bytes']/1e6:.0f} MB ({r['raw_bytes']/r['gzip_per_frame_bytes']:.2f}×) "
            f"| {r['gzip_concat_bytes']/1e6:.0f} MB ({r['raw_bytes']/r['gzip_concat_bytes']:.2f}×) "
            f"| {r['zstd19_long_bytes']/1e6:.0f} MB ({r['raw_bytes']/r['zstd19_long_bytes']:.1f}×) "
            f"| **{d/1e6:.1f} MB ({r['raw_bytes']/d:.1f}×)** "
            f"| {r['gzip_per_frame_bytes']/d:.1f}× smaller "
            f"| {dn/1e6:.1f} MB ({r['raw_bytes']/dn:.1f}×) |"
        )


if __name__ == "__main__":
    main()
