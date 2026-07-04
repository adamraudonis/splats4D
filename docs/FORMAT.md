# .splat4d v1 — 4D Gaussian Splat Format

A single-file, streamable, seekable container that lossy-compresses a time
series of [antimatter15 `.splat`](https://github.com/antimatter15/splat) frames
with **deterministic, user-tunable error bounds**.

Design lineage: quantize-then-delta with bin width 2ε (SZ/ZFP error-bounded
compression), closed GOPs with I/P frames (H.264/HEVC), per-attribute SoA
streams + Morton ordering + byte-plane shuffle + zstd (SPZ/SOGS/Blosc), and a
byte-range chunk index for YouTube-style buffering/seek (which no existing 4D
splat format provides).

## Guarantees

For every splat in every frame, the decoded value differs from the encoder's
reference value by at most the bound:

| attribute | bound parameter | domain | default |
|---|---|---|---|
| position   | `eps_pos` meters (L∞ per axis) | f32 | 2 mm |
| scale      | `eps_scale` relative (per axis) | log domain: \|ln(ŝ/s)\| ≤ ln(1+r) | 2% |
| color RGB  | `b_rgb` 8-bit levels (L∞ per channel) | u8 | ±4 |
| opacity    | `b_alpha` 8-bit levels | u8 | ±4 |
| rotation   | `b_rot` units of 1/128 per quat component, up to sign q≡−q | u8 grid | 0 (exact) |

The reference is the raw input bytes, except when the optional temporal color
denoise prefilter is enabled (header flag bit 0), in which case color bounds
hold against the median-filtered color signal (deviation stats are reported by
the encoder).

Reconstruction is bit-deterministic: after quantization every stage operates on
integers (temporal delta, spatial delta, zigzag, byte-plane shuffle, zstd), so
Rust and JS decoders produce identical integers and errors can never drift or
accumulate (quantize-then-delta, never delta-then-quantize).

## Encoding pipeline

1. **Parse** T frames × N splats × 32 B. v1 requires constant N and index
   correspondence across frames (true for tracked-gaussian pipelines).
   Rotation sign is canonicalized temporally: flip q_t when dot(q_t, q_{t−1}) < 0.
2. **Quantize** each attribute: floats with step `s = 2ε` (`bin = round(v/s)`),
   integers with step `s = 2b+1` (`bin = floor((v + s>>1)/s)`, values centered:
   rot uses v−128). Max error ≤ ε (resp. ≤ b) by construction.
3. **Static classification per attribute per splat** (candidate check):
   candidate = midrange of the true values over the whole clip, snapped to a
   *fine* grid (pitch ε for floats, pitch 1 for ints). Static iff the candidate
   is within the bound of both the min and max true value (exact, per
   component). Static attributes are stored once in the base and emit no
   temporal data.
4. **Order** splats by (dynamism group, Morton code of frame-0 position).
   Dynamism group = bitmask (scale,pos,rot,rgb,alpha dynamic). Fully static
   splats come first; each attribute's dynamic set is a small union of
   contiguous ranges, so per-frame GPU uploads are contiguous.
5. **Hold (deadband) encoding** for dynamic attributes: the stored bin changes
   only when the true value would violate the bound against the currently held
   bin (`|v_t − held·s| > bound` on any component → held = fresh quantized bin).
   Suppresses quantization flicker; deltas become mostly zero; the check itself
   enforces the bound.
6. **GOP structure** (closed, default 30 frames): I-frame = absolute held bins
   (zigzag), P-frames = integer deltas of held bins vs previous frame (zigzag).
   Each GOP chunk decodes independently from the static section → seek = fetch
   one chunk.
7. **Serialize** each stream: byte-plane shuffle (all low bytes, then next
   bytes…) → zstd (level configurable, window ≤ 8 MB for browser friendliness).

## Container layout (all little-endian)

```
[0..4)   magic "SP4D" (0x53 0x50 0x34 0x44)
[4..6)   version u16 = 1
[6..8)   flags   u16 (bit0: colors denoised)
[8..12)  json_len u32
[12..12+json_len) header JSON (UTF-8)
[...]    STATIC section (binary streams, offsets in header JSON)
[...]    GOP chunk 0, GOP chunk 1, ...  (offsets in header JSON)
```

### Header JSON

```jsonc
{
  "n": 336568, "t": 150, "fps": 20.0, "gop": 30,
  "bounds": { "pos_m": 0.005, "scale_rel": 0.02, "rgb": 4, "alpha": 4, "rot": 1 },
  "steps":  { "pos": 0.01, "scale_log": 0.039221, "rgb": 9, "alpha": 9, "rot": 3,
              "base_pos": 0.005, "base_scale_log": 0.019611 },
  "dyn": { "pos": 108426, "rot": 154655, "rgb": 323773, "alpha": 0, "scale": 0 },
  "groups": [ { "mask": 0, "count": 7752 }, ... ],   // dynamism-ordered ranges
  "aabb": [minx,miny,minz, maxx,maxy,maxz],
  "static_section": { "offset": 1234, "len": 3456789 },
  "gops": [ { "offset": ..., "len": ..., "f0": 0, "f1": 29, "t0": 0.0, "t1": 1.45 }, ... ]
}
```

`offset` is absolute within the file, so a client fetches
`[0, header_end)` → `static_section` → any GOP chunk via HTTP Range.

### Binary stream encoding

Every stream: `u8 sym_width (1|2|4)`, `u32 element_count`, `u32 comp_len`,
then `comp_len` bytes of zstd. Payload = byte-plane-shuffled little-endian
unsigned ints (`element_count × sym_width` bytes raw).

### STATIC section — streams in order

| stream | contents |
|---|---|
| `mask_pos, mask_rot, mask_rgb, mask_alpha, mask_scale` | N-bit bitmaps (1 = static), packed MSB-first |
| `base_pos` | N×3 fine-grid bins, spatial delta along order, zigzag (dynamic splats: frame-0 value) |
| `base_scale` | N×3 fine-grid log bins, spatial delta, zigzag |
| `base_rgb` | N×3 u8 |
| `base_alpha` | N×1 u8 |
| `base_rot` | N×4 u8 (canonicalized sign) |

### GOP chunk — mini-TOC then streams

```
u16 stream_count
per stream: u8 attr_id (0=pos 1=rot 2=rgb 3=alpha 4=scale), u8 kind (0=key 1=delta),
            u8 sym_width, u32 element_count, u32 comp_len
stream payloads (zstd), in TOC order
```

Key stream: `dyn_count × channels` zigzagged absolute bins of the GOP's first
frame. Delta stream: `(frames−1) × dyn_count × channels` zigzagged temporal
deltas, frame-major.

Within a chunk, all key streams precede all delta streams, so a seeking client
can fetch just the chunk prefix (TOC + keys, typically ~10 % of the chunk),
display the keyframe immediately, and roll to the exact frame when the delta
payloads arrive — the same trick video players use for instant scrubbing.

### Decoding

1. Fetch header + static section → dequantize bases → full N-splat state
   (this alone renders the complete first frame: dynamic attrs use their
   frame-0 base values... and ~80 % of a typical scene is fully static).
2. For frame t: chunk = gops[t / gop]; state of dynamic attr =
   key + Σ deltas up to t (integer prefix sum); dequantize:
   pos = bin·step; scale = exp(bin·step_log); rgb/alpha = clamp(bin·step, 0, 255);
   quat_i = (bin·step)/128, then normalize.
3. Seek(t′) = fetch chunk containing t′ (plus static section if not cached);
   decode ≤ gop−1 delta frames of integer adds. No dependence on other chunks.

## Why these mechanisms (measured on juggle, 150×336,568 splats, 1,615 MB raw)

* Static split: 80 % of splats fully static in position at ±5 mm → the entire
  background costs ~3 MB once instead of ~1.3 GB.
* Hold encoding: raises rotation static fraction from 54 % (naive bins at ±0)
  to 71 % at ±1/128 and halves dynamic-rotation bytes.
* Quantize-then-delta: position deltas of dynamic splats have H₀ ≈ 2.1 bits
  at ±5 mm (52 % zeros) → ~8 MB for all motion, drift-free.
* GOP 30: keyframes cost <15 % of dynamic bytes — seeking is nearly free
  (GOP 150 saves only ~3 % total).
* Byte-plane shuffle + zstd lands within ~10 % of the order-0 entropy bound of
  the emitted symbols.
