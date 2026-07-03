//! .splat4d decoder: container parsing, frame reconstruction, and the
//! exhaustive error-bound verifier (decode every frame, compare to reference).

use crate::encode::{dyn_indices, Header};
use crate::model::Frames;
use crate::streams::*;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;

pub struct StaticState {
    pub masks: [Vec<bool>; 5], // pos, rot, rgb, alpha, scale (1 = static)
    pub base_pos: Vec<i32>,    // n*3 fine bins
    pub base_ls: Vec<i32>,     // n*3
    pub base_rgb: Vec<u8>,     // n*3
    pub base_alpha: Vec<u8>,   // n
    pub base_rot: Vec<i16>,    // n*4 centered
}

pub struct Decoder {
    pub header: Header,
    pub data: Vec<u8>,
    pub stat: StaticState,
}

fn unbitmap(syms: &[u32], n: usize) -> Vec<bool> {
    let mut out = vec![false; n];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (syms[i / 8] as u8) & (128 >> (i % 8)) != 0;
    }
    out
}

fn un_spatial_delta(syms: &[u32], c: usize) -> Vec<i32> {
    let nn = syms.len() / c;
    let mut out = vec![0i32; syms.len()];
    let mut prev = vec![0i32; c];
    for i in 0..nn {
        for k in 0..c {
            prev[k] += unzigzag(syms[i * c + k]);
            out[i * c + k] = prev[k];
        }
    }
    out
}

impl Decoder {
    pub fn open(path: &std::path::Path) -> Result<Decoder> {
        let data = std::fs::read(path).with_context(|| path.display().to_string())?;
        if data.len() < 12 || &data[0..4] != b"SP4D" {
            bail!("not a .splat4d file (bad magic)");
        }
        let version = u16::from_le_bytes(data[4..6].try_into()?);
        if version != 1 {
            bail!("unsupported version {version}");
        }
        let json_len = u32::from_le_bytes(data[8..12].try_into()?) as usize;
        let header: Header = serde_json::from_slice(&data[12..12 + json_len])?;
        let n = header.n;

        // static section: fixed stream order per spec
        let mut off = header.static_section.offset as usize;
        let mut next = |data: &[u8], off: &mut usize| -> Result<Vec<u32>> {
            let (s, no) = read_stream(data, *off)?;
            *off = no;
            Ok(s.syms)
        };
        let masks = [
            unbitmap(&next(&data, &mut off)?, n),
            unbitmap(&next(&data, &mut off)?, n),
            unbitmap(&next(&data, &mut off)?, n),
            unbitmap(&next(&data, &mut off)?, n),
            unbitmap(&next(&data, &mut off)?, n),
        ];
        let base_pos = un_spatial_delta(&next(&data, &mut off)?, 3);
        let base_ls = un_spatial_delta(&next(&data, &mut off)?, 3);
        let base_rgb: Vec<u8> = next(&data, &mut off)?.iter().map(|&v| v as u8).collect();
        let base_alpha: Vec<u8> = next(&data, &mut off)?.iter().map(|&v| v as u8).collect();
        let base_rot: Vec<i16> = next(&data, &mut off)?.iter().map(|&v| v as i16 - 128).collect();
        if off != (header.static_section.offset + header.static_section.len) as usize {
            bail!("static section length mismatch");
        }

        Ok(Decoder {
            header,
            data,
            stat: StaticState { masks, base_pos, base_ls, base_rgb, base_alpha, base_rot },
        })
    }

    /// Parse one GOP chunk into (attr, kind) -> symbol vectors.
    pub fn gop_streams(&self, g: usize) -> Result<Vec<(u8, u8, Vec<u32>)>> {
        let span = &self.header.gops[g];
        let mut off = span.offset as usize;
        let count = u16::from_le_bytes(self.data[off..off + 2].try_into()?) as usize;
        off += 2;
        let mut metas = Vec::with_capacity(count);
        for _ in 0..count {
            let attr = self.data[off];
            let kind = self.data[off + 1];
            let width = self.data[off + 2];
            let elems = u32::from_le_bytes(self.data[off + 3..off + 7].try_into()?) as usize;
            let comp = u32::from_le_bytes(self.data[off + 7..off + 11].try_into()?) as usize;
            metas.push((attr, kind, width, elems, comp));
            off += STREAM_HDR;
        }
        let mut out = Vec::with_capacity(count);
        for (attr, kind, width, elems, comp) in metas {
            let planes = zstd::bulk::decompress(&self.data[off..off + comp], elems * width as usize)?;
            out.push((attr, kind, from_planes(&planes, width, elems)));
            off += comp;
        }
        if off != (span.offset + span.len) as usize {
            bail!("gop {g} length mismatch");
        }
        Ok(out)
    }
}

pub struct MaxErrs {
    pub pos_m: f64,
    pub scale_log: f64,
    pub rgb: i32,
    pub alpha: i32,
    pub rot: i32,
    pub ok: bool,
}

/// Exhaustively decode every frame and compare against the reference frames
/// (which must be in encoder (permuted) order — i.e. the frames the encoder saw).
pub fn verify(dec: &Decoder, frames: &Frames) -> Result<MaxErrs> {
    let h = &dec.header;
    let (t, n) = (h.t, h.n);
    if frames.t != t || frames.n != n {
        bail!("reference mismatch: file {}x{}, reference {}x{}", t, n, frames.t, frames.n);
    }
    let s = &h.steps;
    let b = &h.bounds;
    let eps_ls = (1.0 + b.scale_rel).ln();

    let idx: [Vec<u32>; 5] = [
        dyn_indices(&dec.stat.masks[0]),
        dyn_indices(&dec.stat.masks[1]),
        dyn_indices(&dec.stat.masks[2]),
        dyn_indices(&dec.stat.masks[3]),
        dyn_indices(&dec.stat.masks[4]),
    ];
    let chans = [3usize, 4, 3, 1, 3];

    // static attribute errors (checked once per frame via max over frames below);
    // dynamic tracks: roll forward through GOPs.
    let mut cur: [Vec<i32>; 5] = [
        vec![0; idx[0].len() * 3],
        vec![0; idx[1].len() * 4],
        vec![0; idx[2].len() * 3],
        vec![0; idx[3].len()],
        vec![0; idx[4].len() * 3],
    ];

    let mut max_pos = 0f64;
    let mut max_ls = 0f64;
    let mut max_rgb = 0i32;
    let mut max_alpha = 0i32;
    let mut max_rot = 0i32;

    for (g, span) in h.gops.iter().enumerate() {
        let streams = dec.gop_streams(g)?;
        for f in span.f0..=span.f1 {
            // advance state
            for a in 0..5 {
                if idx[a].is_empty() {
                    continue;
                }
                let ndc = idx[a].len() * chans[a];
                if f == span.f0 {
                    let key = streams
                        .iter()
                        .find(|(attr, kind, _)| *attr == a as u8 && *kind == KIND_KEY)
                        .map(|(_, _, s)| s)
                        .context("missing key stream")?;
                    for (o, z) in cur[a].iter_mut().zip(key.iter()) {
                        *o = unzigzag(*z);
                    }
                } else {
                    let deltas = streams
                        .iter()
                        .find(|(attr, kind, _)| *attr == a as u8 && *kind == KIND_DELTA)
                        .map(|(_, _, s)| s)
                        .context("missing delta stream")?;
                    let off = (f - span.f0 - 1) * ndc;
                    for (o, z) in cur[a].iter_mut().zip(deltas[off..off + ndc].iter()) {
                        *o += unzigzag(*z);
                    }
                }
            }

            // compare — static splats against bases, dynamic against cur bins
            let fp = &frames.pos[f * n * 3..(f + 1) * n * 3];
            let fls = &frames.lscale[f * n * 3..(f + 1) * n * 3];
            let frgb = &frames.rgb[f * n * 3..(f + 1) * n * 3];
            let falp = &frames.alpha[f * n..(f + 1) * n];
            let frot = &frames.rot[f * n * 4..(f + 1) * n * 4];

            let e_pos = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut e = 0f64;
                    if dec.stat.masks[0][i] {
                        for k in 0..3 {
                            e = e.max((dec.stat.base_pos[i * 3 + k] as f64 * s.base_pos - fp[i * 3 + k] as f64).abs());
                        }
                    }
                    e
                })
                .reduce(|| 0.0, f64::max);
            let e_posd = cur[0]
                .par_chunks(3)
                .zip(idx[0].par_iter())
                .map(|(bins, &i)| {
                    let mut e = 0f64;
                    for k in 0..3 {
                        e = e.max((bins[k] as f64 * s.pos - fp[i as usize * 3 + k] as f64).abs());
                    }
                    e
                })
                .reduce(|| 0.0, f64::max);
            max_pos = max_pos.max(e_pos).max(e_posd);

            let e_ls = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut e = 0f64;
                    if dec.stat.masks[4][i] {
                        for k in 0..3 {
                            e = e.max((dec.stat.base_ls[i * 3 + k] as f64 * s.base_scale_log - fls[i * 3 + k] as f64).abs());
                        }
                    }
                    e
                })
                .reduce(|| 0.0, f64::max);
            let e_lsd = cur[4]
                .par_chunks(3)
                .zip(idx[4].par_iter())
                .map(|(bins, &i)| {
                    let mut e = 0f64;
                    for k in 0..3 {
                        e = e.max((bins[k] as f64 * s.scale_log - fls[i as usize * 3 + k] as f64).abs());
                    }
                    e
                })
                .reduce(|| 0.0, f64::max);
            max_ls = max_ls.max(e_ls).max(e_lsd);

            let e_rgb = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut e = 0i32;
                    if dec.stat.masks[2][i] {
                        for k in 0..3 {
                            e = e.max((dec.stat.base_rgb[i * 3 + k] as i32 - frgb[i * 3 + k] as i32).abs());
                        }
                    }
                    e
                })
                .reduce(|| 0, i32::max);
            let e_rgbd = cur[2]
                .par_chunks(3)
                .zip(idx[2].par_iter())
                .map(|(bins, &i)| {
                    let mut e = 0i32;
                    for k in 0..3 {
                        e = e.max((bins[k] * s.rgb - frgb[i as usize * 3 + k] as i32).abs());
                    }
                    e
                })
                .reduce(|| 0, i32::max);
            max_rgb = max_rgb.max(e_rgb).max(e_rgbd);

            let e_alp = (0..n)
                .into_par_iter()
                .map(|i| {
                    if dec.stat.masks[3][i] {
                        (dec.stat.base_alpha[i] as i32 - falp[i] as i32).abs()
                    } else {
                        0
                    }
                })
                .reduce(|| 0, i32::max);
            let e_alpd = cur[3]
                .par_iter()
                .zip(idx[3].par_iter())
                .map(|(&bin, &i)| (bin * s.alpha - falp[i as usize] as i32).abs())
                .reduce(|| 0, i32::max);
            max_alpha = max_alpha.max(e_alp).max(e_alpd);

            let e_rot = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut e = 0i32;
                    if dec.stat.masks[1][i] {
                        for k in 0..4 {
                            e = e.max((dec.stat.base_rot[i * 4 + k] as i32 - frot[i * 4 + k] as i32).abs());
                        }
                    }
                    e
                })
                .reduce(|| 0, i32::max);
            let e_rotd = cur[1]
                .par_chunks(4)
                .zip(idx[1].par_iter())
                .map(|(bins, &i)| {
                    let mut e = 0i32;
                    for k in 0..4 {
                        e = e.max((bins[k] * s.rot - frot[i as usize * 4 + k] as i32).abs());
                    }
                    e
                })
                .reduce(|| 0, i32::max);
            max_rot = max_rot.max(e_rot).max(e_rotd);
        }
    }

    let ok = max_pos <= b.pos_m + 1e-6
        && max_ls <= eps_ls + 1e-6
        && max_rgb <= b.rgb
        && max_alpha <= b.alpha
        && max_rot <= b.rot;
    Ok(MaxErrs { pos_m: max_pos, scale_log: max_ls, rgb: max_rgb, alpha: max_alpha, rot: max_rot, ok })
}

/// Reconstruct one frame as antimatter15 .splat bytes.
pub fn decode_frame(dec: &Decoder, frame: usize) -> Result<Vec<u8>> {
    let h = &dec.header;
    let (n, s) = (h.n, &h.steps);
    if frame >= h.t {
        bail!("frame {} out of range 0..{}", frame, h.t);
    }
    let g = h.gops.iter().position(|sp| frame >= sp.f0 && frame <= sp.f1).context("no gop")?;
    let span = &h.gops[g];
    let streams = dec.gop_streams(g)?;

    let idx: [Vec<u32>; 5] = [
        dyn_indices(&dec.stat.masks[0]),
        dyn_indices(&dec.stat.masks[1]),
        dyn_indices(&dec.stat.masks[2]),
        dyn_indices(&dec.stat.masks[3]),
        dyn_indices(&dec.stat.masks[4]),
    ];
    let chans = [3usize, 4, 3, 1, 3];
    let mut cur: Vec<Vec<i32>> = (0..5).map(|a| vec![0i32; idx[a].len() * chans[a]]).collect();
    for a in 0..5 {
        if idx[a].is_empty() {
            continue;
        }
        let key = streams
            .iter()
            .find(|(attr, kind, _)| *attr == a as u8 && *kind == KIND_KEY)
            .map(|(_, _, s)| s)
            .context("missing key")?;
        for (o, z) in cur[a].iter_mut().zip(key.iter()) {
            *o = unzigzag(*z);
        }
        if frame > span.f0 {
            let ndc = idx[a].len() * chans[a];
            let deltas = streams
                .iter()
                .find(|(attr, kind, _)| *attr == a as u8 && *kind == KIND_DELTA)
                .map(|(_, _, s)| s)
                .context("missing delta")?;
            for f in span.f0 + 1..=frame {
                let off = (f - span.f0 - 1) * ndc;
                for (o, z) in cur[a].iter_mut().zip(deltas[off..off + ndc].iter()) {
                    *o += unzigzag(*z);
                }
            }
        }
    }

    // start from bases, overwrite dynamic
    let mut pos: Vec<f32> = dec.stat.base_pos.iter().map(|&b| (b as f64 * s.base_pos) as f32).collect();
    let mut lsv: Vec<f32> = dec.stat.base_ls.iter().map(|&b| (b as f64 * s.base_scale_log) as f32).collect();
    let mut rgb = dec.stat.base_rgb.clone();
    let mut alpha = dec.stat.base_alpha.clone();
    let mut rot: Vec<i16> = dec.stat.base_rot.clone();
    for (j, &i) in idx[0].iter().enumerate() {
        for k in 0..3 {
            pos[i as usize * 3 + k] = (cur[0][j * 3 + k] as f64 * s.pos) as f32;
        }
    }
    for (j, &i) in idx[4].iter().enumerate() {
        for k in 0..3 {
            lsv[i as usize * 3 + k] = (cur[4][j * 3 + k] as f64 * s.scale_log) as f32;
        }
    }
    for (j, &i) in idx[2].iter().enumerate() {
        for k in 0..3 {
            rgb[i as usize * 3 + k] = (cur[2][j * 3 + k] * s.rgb).clamp(0, 255) as u8;
        }
    }
    for (j, &i) in idx[3].iter().enumerate() {
        alpha[i as usize] = (cur[3][j] * s.alpha).clamp(0, 255) as u8;
    }
    for (j, &i) in idx[1].iter().enumerate() {
        for k in 0..4 {
            rot[i as usize * 4 + k] = (cur[1][j * 4 + k] * s.rot).clamp(-128, 127) as i16;
        }
    }

    let mut out = Vec::with_capacity(n * 32);
    for i in 0..n {
        for k in 0..3 {
            out.extend_from_slice(&pos[i * 3 + k].to_le_bytes());
        }
        for k in 0..3 {
            out.extend_from_slice(&lsv[i * 3 + k].exp().to_le_bytes());
        }
        for k in 0..3 {
            out.push(rgb[i * 3 + k]);
        }
        out.push(alpha[i]);
        for k in 0..4 {
            out.push((rot[i * 4 + k] + 128).clamp(0, 255) as u8);
        }
    }
    Ok(out)
}
