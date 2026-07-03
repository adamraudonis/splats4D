//! .splat4d encoder: classify -> order -> hold-encode -> GOP streams -> container.

use crate::model::Frames;
use crate::streams::*;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Bounds {
    pub pos_m: f64,
    pub scale_rel: f64,
    pub rgb: i32,
    pub alpha: i32,
    pub rot: i32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Steps {
    pub pos: f64,
    pub base_pos: f64,
    pub scale_log: f64,
    pub base_scale_log: f64,
    pub rgb: i32,
    pub alpha: i32,
    pub rot: i32,
}

impl Bounds {
    pub fn steps(&self) -> Steps {
        let eps_ls = (1.0 + self.scale_rel).ln();
        Steps {
            pos: 2.0 * self.pos_m,
            base_pos: self.pos_m,
            scale_log: 2.0 * eps_ls,
            base_scale_log: eps_ls,
            rgb: 2 * self.rgb + 1,
            alpha: 2 * self.alpha + 1,
            rot: 2 * self.rot + 1,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Span {
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GopSpan {
    pub offset: u64,
    pub len: u64,
    pub f0: usize,
    pub f1: usize,
    pub t0: f32,
    pub t1: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Group {
    pub mask: u8,
    pub count: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DynCounts {
    pub pos: usize,
    pub rot: usize,
    pub rgb: usize,
    pub alpha: usize,
    pub scale: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Header {
    pub n: usize,
    pub t: usize,
    pub fps: f32,
    pub gop: usize,
    pub bounds: Bounds,
    pub steps: Steps,
    #[serde(rename = "dyn")]
    pub dyn_counts: DynCounts,
    pub groups: Vec<Group>,
    pub aabb: [f32; 6],
    pub denoised: bool,
    pub static_section: Span,
    pub gops: Vec<GopSpan>,
}

pub struct StreamInfo {
    pub name: String,
    pub raw_bytes: usize,
    pub comp_bytes: usize,
    pub entropy_bytes: f64,
}

pub struct Report {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub streams: Vec<StreamInfo>,
    pub static_fracs: [f64; 5],
}

/// Everything the encoder derives before stream emission; verify() re-derives
/// this to know the splat permutation.
pub struct FrontEnd {
    pub perm: Vec<u32>,
    pub groups: Vec<Group>,
    pub masks: [Vec<bool>; 5],    // static flags per attr (permuted order)
    pub base_pos: Vec<i32>,       // n*3 fine-grid bins (permuted)
    pub base_ls: Vec<i32>,        // n*3
    pub base_rgb: Vec<u8>,        // n*3
    pub base_alpha: Vec<u8>,      // n
    pub base_rot: Vec<i16>,       // n*4 centered
    pub aabb: [f32; 6],
}

fn part1by2(x: u64) -> u64 {
    let mut x = x & 0x1F_FFFF;
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8)) & 0x100F00F00F00F00F;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

fn qf(v: f64, step: f64) -> i32 {
    (v / step).round() as i32
}

fn qi(v: i32, step: i32) -> i32 {
    (v + (step >> 1)).div_euclid(step)
}

fn minmax_f32(a: &[f32], t: usize, nc: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mn = a[..nc].to_vec();
    let mut mx = a[..nc].to_vec();
    for ti in 1..t {
        let src = &a[ti * nc..(ti + 1) * nc];
        mn.par_iter_mut()
            .zip(mx.par_iter_mut())
            .zip(src.par_iter())
            .for_each(|((lo, hi), v)| {
                if *v < *lo {
                    *lo = *v;
                }
                if *v > *hi {
                    *hi = *v;
                }
            });
    }
    (mn, mx)
}

fn minmax_i16(a: &[i16], t: usize, nc: usize) -> (Vec<i16>, Vec<i16>) {
    let mut mn = a[..nc].to_vec();
    let mut mx = a[..nc].to_vec();
    for ti in 1..t {
        let src = &a[ti * nc..(ti + 1) * nc];
        mn.par_iter_mut()
            .zip(mx.par_iter_mut())
            .zip(src.par_iter())
            .for_each(|((lo, hi), v)| {
                if *v < *lo {
                    *lo = *v;
                }
                if *v > *hi {
                    *hi = *v;
                }
            });
    }
    (mn, mx)
}

fn minmax_u8(a: &[u8], t: usize, nc: usize) -> (Vec<u8>, Vec<u8>) {
    let mut mn = a[..nc].to_vec();
    let mut mx = a[..nc].to_vec();
    for ti in 1..t {
        let src = &a[ti * nc..(ti + 1) * nc];
        mn.par_iter_mut()
            .zip(mx.par_iter_mut())
            .zip(src.par_iter())
            .for_each(|((lo, hi), v)| {
                if *v < *lo {
                    *lo = *v;
                }
                if *v > *hi {
                    *hi = *v;
                }
            });
    }
    (mn, mx)
}

fn permute<T: Copy + Send + Sync>(a: &mut Vec<T>, t: usize, n: usize, c: usize, perm: &[u32]) {
    let mut out = Vec::with_capacity(a.len());
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(a.len());
    }
    out.par_chunks_mut(n * c)
        .zip(a.par_chunks(n * c))
        .for_each(|(dst, src)| {
            for (i, &p) in perm.iter().enumerate() {
                let p = p as usize;
                dst[i * c..i * c + c].copy_from_slice(&src[p * c..p * c + c]);
            }
        });
    *a = out;
}

fn permute1<T: Copy>(a: &[T], perm: &[u32]) -> Vec<T> {
    perm.iter().map(|&p| a[p as usize]).collect()
}

/// Classification + ordering. Mutates `frames` by permuting splats.
pub fn front_end(frames: &mut Frames, bounds: &Bounds) -> FrontEnd {
    let (t, n) = (frames.t, frames.n);
    let steps = bounds.steps();

    // --- per-splat min/max over time ---
    let (pmin, pmax) = minmax_f32(&frames.pos, t, n * 3);
    let (smin, smax) = minmax_f32(&frames.lscale, t, n * 3);
    let (cmin, cmax) = minmax_u8(&frames.rgb, t, n * 3);
    let (amin, amax) = minmax_u8(&frames.alpha, t, n);
    let (rmin, rmax) = minmax_i16(&frames.rot, t, n * 4);

    let mut aabb = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
    for i in 0..n {
        for k in 0..3 {
            aabb[k] = aabb[k].min(pmin[i * 3 + k]);
            aabb[3 + k] = aabb[3 + k].max(pmax[i * 3 + k]);
        }
    }

    // --- candidate-check static classification + bases (original order) ---
    let eps_p = bounds.pos_m;
    let eps_s = (1.0 + bounds.scale_rel).ln();

    let classify_f = |mn: &[f32], mx: &[f32], v0: &[f32], c: usize, base_step: f64, eps: f64| {
        let nn = mn.len() / c;
        let mut stat = vec![false; nn];
        let mut base = vec![0i32; nn * c];
        stat.par_iter_mut()
            .zip(base.par_chunks_mut(c))
            .enumerate()
            .for_each(|(i, (st, bs))| {
                let mut ok = true;
                for k in 0..c {
                    let lo = mn[i * c + k] as f64;
                    let hi = mx[i * c + k] as f64;
                    let cand = qf((lo + hi) / 2.0, base_step);
                    let cv = cand as f64 * base_step;
                    if !(cv >= hi - eps - 1e-9 && cv <= lo + eps + 1e-9) {
                        ok = false;
                    }
                    bs[k] = cand;
                }
                if !ok {
                    for k in 0..c {
                        bs[k] = qf(v0[i * c + k] as f64, base_step);
                    }
                }
                *st = ok;
            });
        (stat, base)
    };

    let (pos_stat, base_pos) = classify_f(&pmin, &pmax, &frames.pos[..n * 3], 3, steps.base_pos, eps_p);
    let (ls_stat, base_ls) = classify_f(&smin, &smax, &frames.lscale[..n * 3], 3, steps.base_scale_log, eps_s);

    let classify_i = |mn: &[i32], mx: &[i32], v0: &[i32], c: usize, b: i32| {
        let nn = mn.len() / c;
        let mut stat = vec![false; nn];
        let mut base = vec![0i32; nn * c];
        stat.par_iter_mut()
            .zip(base.par_chunks_mut(c))
            .enumerate()
            .for_each(|(i, (st, bs))| {
                let mut ok = true;
                for k in 0..c {
                    let lo = mn[i * c + k];
                    let hi = mx[i * c + k];
                    let cand = (lo + hi).div_euclid(2);
                    if !(cand >= hi - b && cand <= lo + b) {
                        ok = false;
                    }
                    bs[k] = cand;
                }
                if !ok {
                    bs.copy_from_slice(&v0[i * c..i * c + c]);
                }
                *st = ok;
            });
        (stat, base)
    };

    let to_i32_u8 = |v: &[u8]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
    let to_i32_i16 = |v: &[i16]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();

    let (rgb_stat, base_rgb_i) = classify_i(
        &to_i32_u8(&cmin), &to_i32_u8(&cmax), &to_i32_u8(&frames.rgb[..n * 3]), 3, bounds.rgb);
    let (alp_stat, base_alp_i) = classify_i(
        &to_i32_u8(&amin), &to_i32_u8(&amax), &to_i32_u8(&frames.alpha[..n]), 1, bounds.alpha);
    let (rot_stat, base_rot_i) = classify_i(
        &to_i32_i16(&rmin), &to_i32_i16(&rmax), &to_i32_i16(&frames.rot[..n * 4]), 4, bounds.rot);

    // --- ordering: (dynamism group mask, morton of frame-0 position) ---
    let ox = aabb[0] as f64;
    let oy = aabb[1] as f64;
    let oz = aabb[2] as f64;
    let key: Vec<u64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mask = (!pos_stat[i] as u64)
                | ((!rot_stat[i] as u64) << 1)
                | ((!rgb_stat[i] as u64) << 2)
                | ((!alp_stat[i] as u64) << 3)
                | ((!ls_stat[i] as u64) << 4);
            let bx = (((frames.pos[i * 3] as f64 - ox) / steps.pos) as u64).min(0x1F_FFFF);
            let by = (((frames.pos[i * 3 + 1] as f64 - oy) / steps.pos) as u64).min(0x1F_FFFF);
            let bz = (((frames.pos[i * 3 + 2] as f64 - oz) / steps.pos) as u64).min(0x1F_FFFF);
            let m = part1by2(bx) | (part1by2(by) << 1) | (part1by2(bz) << 2);
            (mask << 58) | (m & 0x03FF_FFFF_FFFF_FFFF)
        })
        .collect();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.par_sort_unstable_by_key(|&i| key[i as usize]);

    let mut groups: Vec<Group> = Vec::new();
    for &p in &perm {
        let mask = (key[p as usize] >> 58) as u8;
        match groups.last_mut() {
            Some(g) if g.mask == mask => g.count += 1,
            _ => groups.push(Group { mask, count: 1 }),
        }
    }

    // --- permute everything ---
    permute(&mut frames.pos, t, n, 3, &perm);
    permute(&mut frames.lscale, t, n, 3, &perm);
    permute(&mut frames.rgb, t, n, 3, &perm);
    permute(&mut frames.alpha, t, n, 1, &perm);
    permute(&mut frames.rot, t, n, 4, &perm);

    let g3: Vec<u32> = perm.iter().flat_map(|&p| [p * 3, p * 3 + 1, p * 3 + 2]).collect();
    let g4: Vec<u32> = perm.iter().flat_map(|&p| [p * 4, p * 4 + 1, p * 4 + 2, p * 4 + 3]).collect();

    FrontEnd {
        masks: [
            permute1(&pos_stat, &perm),
            permute1(&rot_stat, &perm),
            permute1(&rgb_stat, &perm),
            permute1(&alp_stat, &perm),
            permute1(&ls_stat, &perm),
        ],
        base_pos: permute1(&base_pos, &g3),
        base_ls: permute1(&base_ls, &g3),
        base_rgb: permute1(&base_rgb_i, &g3).iter().map(|&v| v as u8).collect(),
        base_alpha: permute1(&base_alp_i, &perm).iter().map(|&v| v as u8).collect(),
        base_rot: permute1(&base_rot_i, &g4).iter().map(|&v| v as i16).collect(),
        perm,
        groups,
        aabb,
    }
}

/// Deadband hold-encoding of one dynamic attribute. Float variant.
pub fn hold_f(vals: &[f32], t: usize, n: usize, c: usize, idx: &[u32], step: f64, eps: f64) -> Vec<i16> {
    let nd = idx.len();
    let mut held = vec![0i16; t * nd * c];
    // frame 0
    {
        let h0 = &mut held[..nd * c];
        h0.par_chunks_mut(c).zip(idx.par_iter()).for_each(|(h, &i)| {
            for k in 0..c {
                h[k] = qf(vals[i as usize * c + k] as f64, step) as i16;
            }
        });
    }
    for ti in 1..t {
        let (past, cur) = held.split_at_mut(ti * nd * c);
        let prev = &past[(ti - 1) * nd * c..];
        let frame = &vals[ti * n * c..(ti + 1) * n * c];
        cur[..nd * c]
            .par_chunks_mut(c)
            .zip(prev.par_chunks(c))
            .zip(idx.par_iter())
            .for_each(|((h, hp), &i)| {
                let base = i as usize * c;
                let mut viol = false;
                for k in 0..c {
                    if ((frame[base + k] as f64) - (hp[k] as f64) * step).abs() > eps {
                        viol = true;
                        break;
                    }
                }
                if viol {
                    for k in 0..c {
                        h[k] = qf(frame[base + k] as f64, step) as i16;
                    }
                } else {
                    h.copy_from_slice(hp);
                }
            });
    }
    held
}

/// Deadband hold-encoding, integer variant (values already i32-representable).
pub fn hold_i(get: &(dyn Fn(usize, usize, usize) -> i32 + Sync), t: usize, c: usize, idx: &[u32], step: i32, b: i32) -> Vec<i16> {
    let nd = idx.len();
    let mut held = vec![0i16; t * nd * c];
    {
        let h0 = &mut held[..nd * c];
        h0.par_chunks_mut(c).zip(idx.par_iter()).for_each(|(h, &i)| {
            for k in 0..c {
                h[k] = qi(get(0, i as usize, k), step) as i16;
            }
        });
    }
    for ti in 1..t {
        let (past, cur) = held.split_at_mut(ti * nd * c);
        let prev = &past[(ti - 1) * nd * c..];
        cur[..nd * c]
            .par_chunks_mut(c)
            .zip(prev.par_chunks(c))
            .zip(idx.par_iter())
            .for_each(|((h, hp), &i)| {
                let mut viol = false;
                for k in 0..c {
                    if (get(ti, i as usize, k) - hp[k] as i32 * step).abs() > b {
                        viol = true;
                        break;
                    }
                }
                if viol {
                    for k in 0..c {
                        h[k] = qi(get(ti, i as usize, k), step) as i16;
                    }
                } else {
                    h.copy_from_slice(hp);
                }
            });
    }
    held
}

pub fn dyn_indices(mask: &[bool]) -> Vec<u32> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &s)| (!s).then_some(i as u32))
        .collect()
}

fn bitmap(mask: &[bool]) -> Vec<u32> {
    let mut out = vec![0u32; mask.len().div_ceil(8)];
    for (i, &m) in mask.iter().enumerate() {
        if m {
            out[i / 8] |= 128 >> (i % 8);
        }
    }
    out
}

pub struct Encoded {
    pub file: Vec<u8>,
    pub report: Report,
    pub header: Header,
}

pub fn encode(frames: &mut Frames, bounds: &Bounds, gop: usize, level: i32, denoised: bool) -> Result<(Encoded, FrontEnd)> {
    let (t, n) = (frames.t, frames.n);
    let steps = bounds.steps();
    let fe = front_end(frames, bounds);

    // --- hold-encode dynamic tracks ---
    let idx_pos = dyn_indices(&fe.masks[0]);
    let idx_rot = dyn_indices(&fe.masks[1]);
    let idx_rgb = dyn_indices(&fe.masks[2]);
    let idx_alp = dyn_indices(&fe.masks[3]);
    let idx_ls = dyn_indices(&fe.masks[4]);

    let held_pos = hold_f(&frames.pos, t, n, 3, &idx_pos, steps.pos, bounds.pos_m);
    let held_ls = hold_f(&frames.lscale, t, n, 3, &idx_ls, steps.scale_log, (1.0 + bounds.scale_rel).ln());
    let rgb_ref = &frames.rgb;
    let held_rgb = hold_i(&|ti, i, k| rgb_ref[ti * n * 3 + i * 3 + k] as i32, t, 3, &idx_rgb, steps.rgb, bounds.rgb);
    let alp_ref = &frames.alpha;
    let held_alp = hold_i(&|ti, i, _| alp_ref[ti * n + i] as i32, t, 1, &idx_alp, steps.alpha, bounds.alpha);
    let rot_ref = &frames.rot;
    let held_rot = hold_i(&|ti, i, k| rot_ref[ti * n * 4 + i * 4 + k] as i32, t, 4, &idx_rot, steps.rot, bounds.rot);

    // --- static section raw streams ---
    let mut static_raw: Vec<(String, RawStream)> = Vec::new();
    for (a, name) in [(0usize, "mask_pos"), (1, "mask_rot"), (2, "mask_rgb"), (3, "mask_alpha"), (4, "mask_scale")] {
        static_raw.push((name.into(), RawStream { attr: a as u8, kind: KIND_STATIC, syms: bitmap(&fe.masks[a]) }));
    }
    let spatial_delta = |bins: &[i32], c: usize| -> Vec<u32> {
        let nn = bins.len() / c;
        let mut out = Vec::with_capacity(bins.len());
        for i in 0..nn {
            for k in 0..c {
                let prev = if i == 0 { 0 } else { bins[(i - 1) * c + k] };
                out.push(zigzag(bins[i * c + k] - prev));
            }
        }
        out
    };
    static_raw.push(("base_pos".into(), RawStream { attr: ATTR_POS, kind: KIND_STATIC, syms: spatial_delta(&fe.base_pos, 3) }));
    static_raw.push(("base_scale".into(), RawStream { attr: ATTR_SCALE, kind: KIND_STATIC, syms: spatial_delta(&fe.base_ls, 3) }));
    static_raw.push(("base_rgb".into(), RawStream { attr: ATTR_RGB, kind: KIND_STATIC, syms: fe.base_rgb.iter().map(|&v| v as u32).collect() }));
    static_raw.push(("base_alpha".into(), RawStream { attr: ATTR_ALPHA, kind: KIND_STATIC, syms: fe.base_alpha.iter().map(|&v| v as u32).collect() }));
    static_raw.push(("base_rot".into(), RawStream { attr: ATTR_ROT, kind: KIND_STATIC, syms: fe.base_rot.iter().map(|&v| (v as i32 + 128) as u32).collect() }));

    // --- GOP raw stream jobs ---
    let n_gops = t.div_ceil(gop);
    struct Job {
        gop: usize,
        name: String,
        raw: RawStream,
    }
    let tracks: [(&str, u8, &Vec<i16>, usize, usize); 5] = [
        ("pos", ATTR_POS, &held_pos, idx_pos.len(), 3),
        ("rot", ATTR_ROT, &held_rot, idx_rot.len(), 4),
        ("rgb", ATTR_RGB, &held_rgb, idx_rgb.len(), 3),
        ("alpha", ATTR_ALPHA, &held_alp, idx_alp.len(), 1),
        ("scale", ATTR_SCALE, &held_ls, idx_ls.len(), 3),
    ];
    let jobs: Vec<Job> = (0..n_gops)
        .into_par_iter()
        .flat_map_iter(|g| {
            let f0 = g * gop;
            let f1 = ((g + 1) * gop - 1).min(t - 1);
            let mut out = Vec::new();
            for (nm, attr, held, nd, c) in tracks.iter() {
                if *nd == 0 {
                    continue;
                }
                let ndc = nd * c;
                let key: Vec<u32> = held[f0 * ndc..(f0 + 1) * ndc].iter().map(|&v| zigzag(v as i32)).collect();
                out.push(Job { gop: g, name: format!("g{g}.{nm}.key"), raw: RawStream { attr: *attr, kind: KIND_KEY, syms: key } });
                if f1 > f0 {
                    let mut deltas = Vec::with_capacity((f1 - f0) * ndc);
                    for f in f0 + 1..=f1 {
                        let cur = &held[f * ndc..(f + 1) * ndc];
                        let prev = &held[(f - 1) * ndc..f * ndc];
                        deltas.extend(cur.iter().zip(prev.iter()).map(|(&a, &b)| zigzag(a as i32 - b as i32)));
                    }
                    out.push(Job { gop: g, name: format!("g{g}.{nm}.delta"), raw: RawStream { attr: *attr, kind: KIND_DELTA, syms: deltas } });
                }
            }
            out
        })
        .collect();

    // --- compress everything in parallel ---
    let packed_static: Vec<(String, PackedStream)> = static_raw
        .into_par_iter()
        .map(|(name, raw)| Ok((name, pack(raw, level)?)))
        .collect::<Result<_>>()?;
    let packed_jobs: Vec<(usize, String, PackedStream)> = jobs
        .into_par_iter()
        .map(|j| Ok((j.gop, j.name, pack(j.raw, level)?)))
        .collect::<Result<_>>()?;

    // --- assemble sections ---
    let mut static_bytes = Vec::new();
    for (_, s) in &packed_static {
        write_stream(&mut static_bytes, s);
    }
    let mut gop_bytes: Vec<Vec<u8>> = vec![Vec::new(); n_gops];
    for g in 0..n_gops {
        // keys first: a seek can fetch the chunk prefix (TOC + keys) and show
        // the keyframe immediately, before the delta payloads arrive
        let mut streams: Vec<&PackedStream> = packed_jobs.iter().filter(|(gg, _, _)| *gg == g).map(|(_, _, s)| s).collect();
        streams.sort_by_key(|s| s.kind);
        let chunk = &mut gop_bytes[g];
        chunk.extend_from_slice(&(streams.len() as u16).to_le_bytes());
        for s in &streams {
            chunk.push(s.attr);
            chunk.push(s.kind);
            chunk.push(s.sym_width);
            chunk.extend_from_slice(&s.elems.to_le_bytes());
            chunk.extend_from_slice(&(s.payload.len() as u32).to_le_bytes());
        }
        for s in &streams {
            chunk.extend_from_slice(&s.payload);
        }
    }

    // --- header (iterate JSON length to fixpoint) ---
    let fps = frames.fps;
    let mk_header = |json_len: usize| -> Header {
        let mut off = 12 + json_len as u64;
        let static_section = Span { offset: off, len: static_bytes.len() as u64 };
        off += static_bytes.len() as u64;
        let mut gops = Vec::new();
        for (g, gb) in gop_bytes.iter().enumerate() {
            let f0 = g * gop;
            let f1 = ((g + 1) * gop - 1).min(t - 1);
            gops.push(GopSpan { offset: off, len: gb.len() as u64, f0, f1, t0: f0 as f32 / fps, t1: f1 as f32 / fps });
            off += gb.len() as u64;
        }
        Header {
            n,
            t,
            fps,
            gop,
            bounds: *bounds,
            steps,
            dyn_counts: DynCounts { pos: idx_pos.len(), rot: idx_rot.len(), rgb: idx_rgb.len(), alpha: idx_alp.len(), scale: idx_ls.len() },
            groups: fe.groups.clone(),
            aabb: fe.aabb,
            denoised,
            static_section,
            gops,
        }
    };
    let mut json_len = serde_json::to_vec(&mk_header(0))?.len();
    let (header, json) = loop {
        let h = mk_header(json_len);
        let j = serde_json::to_vec(&h)?;
        if j.len() == json_len {
            break (h, j);
        }
        json_len = j.len();
    };

    let mut file = Vec::with_capacity(12 + json.len() + static_bytes.len() + gop_bytes.iter().map(|g| g.len()).sum::<usize>());
    file.extend_from_slice(b"SP4D");
    file.extend_from_slice(&1u16.to_le_bytes());
    file.extend_from_slice(&(denoised as u16).to_le_bytes());
    file.extend_from_slice(&(json.len() as u32).to_le_bytes());
    file.extend_from_slice(&json);
    file.extend_from_slice(&static_bytes);
    for gb in &gop_bytes {
        file.extend_from_slice(gb);
    }

    // --- report ---
    let mut streams = Vec::new();
    for (name, s) in &packed_static {
        streams.push(StreamInfo { name: name.clone(), raw_bytes: s.raw_bytes, comp_bytes: s.payload.len() + STREAM_HDR, entropy_bytes: s.entropy_bytes });
    }
    for (_, name, s) in &packed_jobs {
        streams.push(StreamInfo { name: name.clone(), raw_bytes: s.raw_bytes, comp_bytes: s.payload.len() + STREAM_HDR, entropy_bytes: s.entropy_bytes });
    }
    let report = Report {
        input_bytes: t * n * crate::model::REC,
        output_bytes: file.len(),
        streams,
        static_fracs: [
            fe.masks[0].iter().filter(|&&m| m).count() as f64 / n as f64,
            fe.masks[1].iter().filter(|&&m| m).count() as f64 / n as f64,
            fe.masks[2].iter().filter(|&&m| m).count() as f64 / n as f64,
            fe.masks[3].iter().filter(|&&m| m).count() as f64 / n as f64,
            fe.masks[4].iter().filter(|&&m| m).count() as f64 / n as f64,
        ],
    };

    Ok((Encoded { file, report, header }, fe))
}
