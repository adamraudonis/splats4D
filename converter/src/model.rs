//! Input parsing: a directory of antimatter15 .splat frames (+ frames.json manifest)
//! into SoA arrays, plus the optional temporal color denoise prefilter.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;

pub const REC: usize = 32; // bytes per splat record

#[derive(Deserialize)]
struct ManifestFrame {
    file: String,
    #[allow(dead_code)]
    timestamp: f64,
}

#[derive(Deserialize)]
struct Manifest {
    fps: f32,
    frames: Vec<ManifestFrame>,
}

/// Frame-major SoA arrays for the whole sequence.
pub struct Frames {
    pub t: usize,
    pub n: usize,
    pub fps: f32,
    pub pos: Vec<f32>,    // t*n*3
    pub lscale: Vec<f32>, // t*n*3  (ln of linear scale)
    pub rgb: Vec<u8>,     // t*n*3
    pub alpha: Vec<u8>,   // t*n
    pub rot: Vec<i16>,    // t*n*4  centered (v-128), temporally sign-canonicalized
}

impl Frames {
    pub fn load(dir: &Path, fps_flag: Option<f32>) -> Result<Frames> {
        let manifest_path = dir.join("frames.json");
        let (files, fps): (Vec<std::path::PathBuf>, f32) = if manifest_path.exists() {
            let m: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
            (m.frames.iter().map(|f| dir.join(&f.file)).collect(), m.fps)
        } else {
            let mut fs: Vec<_> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "splat"))
                .collect();
            fs.sort();
            (fs, fps_flag.unwrap_or(30.0))
        };
        if files.is_empty() {
            bail!("no .splat frames found in {}", dir.display());
        }
        let t = files.len();
        let first = std::fs::metadata(&files[0])?.len() as usize;
        if first % REC != 0 {
            bail!("{} size not a multiple of 32", files[0].display());
        }
        let n = first / REC;

        let mut pos = vec![0f32; t * n * 3];
        let mut lscale = vec![0f32; t * n * 3];
        let mut rgb = vec![0u8; t * n * 3];
        let mut alpha = vec![0u8; t * n];
        let mut rot = vec![0i16; t * n * 4];

        // parse frames in parallel (each frame owns disjoint slices)
        pos.par_chunks_mut(n * 3)
            .zip(lscale.par_chunks_mut(n * 3))
            .zip(rgb.par_chunks_mut(n * 3))
            .zip(alpha.par_chunks_mut(n))
            .zip(rot.par_chunks_mut(n * 4))
            .zip(files.par_iter())
            .try_for_each(|(((((p, ls), c), a), r), f)| -> Result<()> {
                let buf = std::fs::read(f).with_context(|| f.display().to_string())?;
                if buf.len() != n * REC {
                    bail!("{}: expected {} splats, got {}", f.display(), n, buf.len() / REC);
                }
                for i in 0..n {
                    let rec = &buf[i * REC..(i + 1) * REC];
                    for k in 0..3 {
                        p[i * 3 + k] = f32::from_le_bytes(rec[k * 4..k * 4 + 4].try_into().unwrap());
                        let s = f32::from_le_bytes(rec[12 + k * 4..16 + k * 4].try_into().unwrap());
                        ls[i * 3 + k] = s.max(1e-9).ln();
                        c[i * 3 + k] = rec[24 + k];
                    }
                    a[i] = rec[27];
                    for k in 0..4 {
                        r[i * 4 + k] = rec[28 + k] as i16 - 128;
                    }
                }
                Ok(())
            })?;

        // temporal quaternion sign canonicalization (q and -q are the same rotation)
        for ti in 1..t {
            let (past, cur) = rot.split_at_mut(ti * n * 4);
            let prev = &past[(ti - 1) * n * 4..];
            cur[..n * 4]
                .par_chunks_mut(4)
                .zip(prev.par_chunks(4))
                .for_each(|(q, qp)| {
                    let dot: i32 = (0..4).map(|k| q[k] as i32 * qp[k] as i32).sum();
                    if dot < 0 {
                        for v in q.iter_mut() {
                            // saturate: -(-128) = +128 is off the u8 grid; 127
                            // is the nearest representable value of the same
                            // rotation (q and -q are identified anyway)
                            *v = if *v == -128 { 127 } else { -*v };
                        }
                    }
                });
        }

        Ok(Frames { t, n, fps, pos, lscale, rgb, alpha, rot })
    }

    /// Optional prefilter: temporal median-of-5 on each color channel.
    /// Returns (mean, p99) absolute deviation from the raw signal in 8-bit levels.
    pub fn denoise_colors(&mut self) -> (f64, f64) {
        let (t, n) = (self.t, self.n);
        let raw = self.rgb.clone();
        let out: Vec<u8> = (0..t * n * 3)
            .into_par_iter()
            .map(|idx| {
                let ti = idx / (n * 3);
                let j = idx % (n * 3);
                let mut w = [0u8; 5];
                for (k, wi) in w.iter_mut().enumerate() {
                    let tt = (ti as isize + k as isize - 2).clamp(0, t as isize - 1) as usize;
                    *wi = raw[tt * n * 3 + j];
                }
                w.sort_unstable();
                w[2]
            })
            .collect();
        let mut sum = 0u64;
        let mut hist = [0u64; 256];
        for (a, b) in out.iter().zip(raw.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs() as usize;
            sum += d as u64;
            hist[d] += 1;
        }
        let total: u64 = hist.iter().sum();
        let mut acc = 0u64;
        let mut p99 = 0usize;
        for (d, c) in hist.iter().enumerate() {
            acc += c;
            if acc as f64 >= total as f64 * 0.99 {
                p99 = d;
                break;
            }
        }
        self.rgb = out;
        (sum as f64 / total as f64, p99 as f64)
    }
}
