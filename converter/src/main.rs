//! splat4d — error-bounded temporal compressor for gaussian splat sequences.
//!
//! encode: series of antimatter15 .splat frames -> one streamable .splat4d
//! verify: exhaustive per-frame error-bound check against the source frames
//! decode: reconstruct a single frame back to .splat
//! info:   print header / stream statistics

mod decode;
mod encode;
mod model;
mod streams;

use anyhow::Result;
use clap::{Parser, Subcommand};
use encode::Bounds;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "splat4d", about = "4D gaussian splat compressor with deterministic error bounds")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a directory of .splat frames into one .splat4d file
    Encode {
        /// Directory containing frame_*.splat (+ optional frames.json manifest)
        #[arg(short, long)]
        input: PathBuf,
        /// Output .splat4d path
        #[arg(short, long)]
        output: PathBuf,
        /// Max absolute position error, millimeters
        #[arg(long, default_value_t = 2.0)]
        pos_mm: f64,
        /// Max absolute color/opacity error, 8-bit levels
        #[arg(long, default_value_t = 4)]
        color_levels: i32,
        /// Max absolute quaternion-component error, units of 1/128 (0 = lossless vs input grid)
        #[arg(long, default_value_t = 0)]
        rot_steps: i32,
        /// Max relative scale error, percent
        #[arg(long, default_value_t = 2.0)]
        scale_pct: f64,
        /// Frames per GOP (keyframe interval)
        #[arg(long, default_value_t = 30)]
        gop: usize,
        /// zstd compression level (0 = auto: 19, with 13 for very large streams)
        #[arg(long, default_value_t = 3)]
        zstd_level: i32,
        /// Temporal median-of-5 prefilter on colors (bounds then hold vs the smoothed signal)
        #[arg(long)]
        denoise_colors: bool,
        /// Skip the exhaustive decode-and-check pass
        #[arg(long)]
        no_verify: bool,
        /// Frames per second if no frames.json manifest is present
        #[arg(long)]
        fps: Option<f32>,
        /// Write a machine-readable metrics report to this JSON path
        #[arg(long)]
        report: Option<PathBuf>,
        /// Write the splat permutation (file order -> input index, n×u32 LE);
        /// lets a viewer align original frames with the encoded ordering
        #[arg(long)]
        perm_out: Option<PathBuf>,
    },
    /// Verify a .splat4d against its source frames (exhaustive, every splat every frame)
    Verify {
        file: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(long)]
        fps: Option<f32>,
    },
    /// Reconstruct one frame back to a .splat file
    Decode {
        file: PathBuf,
        #[arg(long, default_value_t = 0)]
        frame: usize,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Print header and section statistics
    Info { file: PathBuf },
}

fn human(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes} B")
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Encode {
            input,
            output,
            pos_mm,
            color_levels,
            rot_steps,
            scale_pct,
            gop,
            zstd_level,
            denoise_colors,
            no_verify,
            fps,
            report,
            perm_out,
        } => {
            let bounds = Bounds {
                pos_m: pos_mm / 1000.0,
                scale_rel: scale_pct / 100.0,
                rgb: color_levels,
                alpha: color_levels,
                rot: rot_steps,
            };
            let t_all = Instant::now();

            let t0 = Instant::now();
            let mut frames = model::Frames::load(&input, fps)?;
            let t_parse = t0.elapsed();
            let input_bytes = frames.t * frames.n * model::REC;
            eprintln!(
                "parsed {} frames x {} splats ({}) in {:.2}s",
                frames.t,
                frames.n,
                human(input_bytes),
                t_parse.as_secs_f64()
            );

            let mut denoise_stats = None;
            if denoise_colors {
                let t0 = Instant::now();
                let (mean, p99) = frames.denoise_colors();
                denoise_stats = Some((mean, p99));
                eprintln!(
                    "denoised colors (median-5): mean |dev| {:.2} levels, p99 {:.0} in {:.2}s",
                    mean,
                    p99,
                    t0.elapsed().as_secs_f64()
                );
            }

            let t0 = Instant::now();
            let (enc, fe) = encode::encode(&mut frames, &bounds, gop, zstd_level, denoise_colors)?;
            let t_encode = t0.elapsed();

            let t0 = Instant::now();
            std::fs::write(&output, &enc.file)?;
            if let Some(path) = &perm_out {
                let mut bytes = Vec::with_capacity(fe.perm.len() * 4);
                for p in &fe.perm {
                    bytes.extend_from_slice(&p.to_le_bytes());
                }
                std::fs::write(path, bytes)?;
            }
            let t_write = t0.elapsed();

            // report
            let r = &enc.report;
            eprintln!("\nstatic fractions: pos {:.1}%  rot {:.1}%  rgb {:.1}%  alpha {:.1}%  scale {:.1}%",
                r.static_fracs[0] * 100.0, r.static_fracs[1] * 100.0, r.static_fracs[2] * 100.0,
                r.static_fracs[3] * 100.0, r.static_fracs[4] * 100.0);

            // aggregate stream stats by category
            let mut agg: std::collections::BTreeMap<String, (usize, usize, f64)> = Default::default();
            for s in &r.streams {
                let cat = if s.name.starts_with('g') && s.name.contains('.') {
                    let mut it = s.name.split('.');
                    let _g = it.next();
                    format!("dyn.{}.{}", it.next().unwrap_or("?"), it.next().unwrap_or("?"))
                } else {
                    format!("static.{}", s.name)
                };
                let e = agg.entry(cat).or_default();
                e.0 += s.raw_bytes;
                e.1 += s.comp_bytes;
                e.2 += s.entropy_bytes;
            }
            eprintln!("\n{:<22} {:>12} {:>12} {:>12}", "stream", "raw", "zstd", "entropy-min");
            for (cat, (raw, comp, ent)) in &agg {
                eprintln!("{:<22} {:>12} {:>12} {:>12}", cat, human(*raw), human(*comp), human(*ent as usize));
            }
            let ent_total: f64 = r.streams.iter().map(|s| s.entropy_bytes).sum();
            eprintln!("\ninput  {} ({} frames x {} splats x 32 B)", human(r.input_bytes), frames.t, frames.n);
            eprintln!("output {}   ratio {:.1}x   entropy-min {} ({:.1}x)",
                human(r.output_bytes),
                r.input_bytes as f64 / r.output_bytes as f64,
                human(ent_total as usize),
                r.input_bytes as f64 / ent_total);
            eprintln!("bytes/frame {}   bytes/splat/frame {:.3}",
                human(r.output_bytes / frames.t), r.output_bytes as f64 / (frames.t * frames.n) as f64);
            eprintln!("\ntimes: parse {:.2}s  encode+compress {:.2}s  write {:.2}s  total {:.2}s  ({:.0} MB/s in)",
                t_parse.as_secs_f64(), t_encode.as_secs_f64(), t_write.as_secs_f64(),
                t_all.elapsed().as_secs_f64(), r.input_bytes as f64 / 1e6 / t_all.elapsed().as_secs_f64());

            let mut verify_json = serde_json::json!(null);
            if !no_verify {
                let t0 = Instant::now();
                let dec = decode::Decoder::open(&output)?;
                let errs = decode::verify(&dec, &frames)?;
                eprintln!("\nverify (every splat, every frame) in {:.2}s:", t0.elapsed().as_secs_f64());
                eprintln!("  pos    max err {:.4} mm   (bound {} mm)", errs.pos_m * 1000.0, pos_mm);
                eprintln!("  scale  max err {:.3}%      (bound {}%)", (errs.scale_log.exp() - 1.0) * 100.0, scale_pct);
                eprintln!("  rgb    max err {} levels  (bound {})", errs.rgb, color_levels);
                eprintln!("  alpha  max err {} levels  (bound {})", errs.alpha, color_levels);
                eprintln!("  rot    max err {} /128    (bound {})", errs.rot, rot_steps);
                eprintln!("  BOUNDS {}", if errs.ok { "VERIFIED" } else { "VIOLATED — BUG" });
                verify_json = serde_json::json!({
                    "pos_mm": errs.pos_m * 1000.0,
                    "scale_pct": (errs.scale_log.exp() - 1.0) * 100.0,
                    "rgb_levels": errs.rgb, "alpha_levels": errs.alpha, "rot_units": errs.rot,
                    "ok": errs.ok,
                });
                if !errs.ok {
                    anyhow::bail!("error bounds violated — encoder bug");
                }
            }

            if let Some(path) = report {
                let json = serde_json::json!({
                    "input": { "frames": frames.t, "splats": frames.n, "bytes": r.input_bytes },
                    "output": { "bytes": r.output_bytes, "ratio": r.input_bytes as f64 / r.output_bytes as f64 },
                    "entropy_min_bytes": ent_total,
                    "bounds": bounds,
                    "gop": gop, "zstd_level": zstd_level,
                    "denoise": denoise_stats.map(|(m, p)| serde_json::json!({"mean_dev": m, "p99_dev": p})),
                    "static_fracs": { "pos": r.static_fracs[0], "rot": r.static_fracs[1], "rgb": r.static_fracs[2], "alpha": r.static_fracs[3], "scale": r.static_fracs[4] },
                    "streams": agg.iter().map(|(k, v)| serde_json::json!({"name": k, "raw": v.0, "zstd": v.1, "entropy": v.2})).collect::<Vec<_>>(),
                    "times_s": { "parse": t_parse.as_secs_f64(), "encode": t_encode.as_secs_f64(), "write": t_write.as_secs_f64(), "total": t_all.elapsed().as_secs_f64() },
                    "verify": verify_json,
                });
                std::fs::write(&path, serde_json::to_string_pretty(&json)?)?;
                eprintln!("report written to {}", path.display());
            }
        }

        Cmd::Verify { file, input, fps } => {
            let dec = decode::Decoder::open(&file)?;
            let mut frames = model::Frames::load(&input, fps)?;
            if dec.header.denoised {
                frames.denoise_colors();
            }
            // re-derive the encoder permutation deterministically
            let _fe = encode::front_end(&mut frames, &dec.header.bounds);
            let errs = decode::verify(&dec, &frames)?;
            println!("pos    max err {:.4} mm   (bound {} mm)", errs.pos_m * 1000.0, dec.header.bounds.pos_m * 1000.0);
            println!("scale  max err {:.3}%      (bound {}%)", (errs.scale_log.exp() - 1.0) * 100.0, dec.header.bounds.scale_rel * 100.0);
            println!("rgb    max err {} levels  (bound {})", errs.rgb, dec.header.bounds.rgb);
            println!("alpha  max err {} levels  (bound {})", errs.alpha, dec.header.bounds.alpha);
            println!("rot    max err {} /128    (bound {})", errs.rot, dec.header.bounds.rot);
            println!("BOUNDS {}", if errs.ok { "VERIFIED" } else { "VIOLATED" });
            if !errs.ok {
                std::process::exit(1);
            }
        }

        Cmd::Decode { file, frame, output } => {
            let dec = decode::Decoder::open(&file)?;
            let bytes = decode::decode_frame(&dec, frame)?;
            std::fs::write(&output, &bytes)?;
            println!("wrote {} ({} splats) for frame {}", output.display(), bytes.len() / 32, frame);
        }

        Cmd::Info { file } => {
            let dec = decode::Decoder::open(&file)?;
            let h = &dec.header;
            println!("{}", serde_json::to_string_pretty(&h)?);
            let file_len = dec.data.len();
            println!("file {}   header+static {}   {} GOPs",
                human(file_len),
                human((h.static_section.offset + h.static_section.len) as usize),
                h.gops.len());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode, front_end};
    use crate::model::Frames;

    /// Build a synthetic sequence: half static splats, half orbiting, colors drifting.
    fn synthetic(t: usize, n: usize) -> Frames {
        let mut pos = vec![0f32; t * n * 3];
        let mut lscale = vec![0f32; t * n * 3];
        let mut rgb = vec![0u8; t * n * 3];
        let mut alpha = vec![0u8; t * n];
        let mut rot = vec![0i16; t * n * 4];
        for ti in 0..t {
            for i in 0..n {
                let moving = i % 2 == 0;
                let phase = ti as f32 * 0.1 + i as f32;
                let (x, y, z) = if moving {
                    (phase.sin(), phase.cos(), i as f32 * 0.01)
                } else {
                    ((i % 100) as f32 * 0.05, (i / 100) as f32 * 0.05, 0.0)
                };
                pos[(ti * n + i) * 3] = x;
                pos[(ti * n + i) * 3 + 1] = y;
                pos[(ti * n + i) * 3 + 2] = z;
                for k in 0..3 {
                    lscale[(ti * n + i) * 3 + k] = -3.0 + (i % 7) as f32 * 0.1;
                    rgb[(ti * n + i) * 3 + k] = ((i * 37 + k * 11 + if moving { ti * 3 } else { 0 }) % 256) as u8;
                }
                alpha[ti * n + i] = 200;
                let q = [1.0 + phase.sin() * if moving { 0.3 } else { 0.0 }, 0.1, 0.2, 0.3f32];
                let norm = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
                for k in 0..4 {
                    rot[(ti * n + i) * 4 + k] = ((q[k] / norm * 128.0).round() as i32).clamp(-128, 127) as i16;
                }
            }
        }
        let mut f = Frames { t, n, fps: 30.0, pos, lscale, rgb, alpha, rot };
        // canonicalize rotation signs the same way the loader does
        for ti in 1..t {
            for i in 0..n {
                let dot: i32 = (0..4)
                    .map(|k| f.rot[(ti * n + i) * 4 + k] as i32 * f.rot[((ti - 1) * n + i) * 4 + k] as i32)
                    .sum();
                if dot < 0 {
                    for k in 0..4 {
                        let v = f.rot[(ti * n + i) * 4 + k];
                        f.rot[(ti * n + i) * 4 + k] = if v == -128 { 127 } else { -v };
                    }
                }
            }
        }
        f
    }

    #[test]
    fn roundtrip_bounds_hold() {
        let bounds = Bounds { pos_m: 0.005, scale_rel: 0.02, rgb: 4, alpha: 4, rot: 1 };
        let mut frames = synthetic(25, 400);
        let (enc, _) = encode(&mut frames, &bounds, 10, 3, false).unwrap();
        let tmp = std::env::temp_dir().join("splat4d_test.splat4d");
        std::fs::write(&tmp, &enc.file).unwrap();
        let dec = decode::Decoder::open(&tmp).unwrap();
        let errs = decode::verify(&dec, &frames).unwrap();
        assert!(errs.ok, "bounds violated: pos {} mm rgb {} rot {}", errs.pos_m * 1000.0, errs.rgb, errs.rot);
        // single-frame decode roundtrip sanity
        let f0 = decode::decode_frame(&dec, 7).unwrap();
        assert_eq!(f0.len(), 400 * 32);
    }

    #[test]
    fn front_end_deterministic() {
        let bounds = Bounds { pos_m: 0.005, scale_rel: 0.02, rgb: 4, alpha: 4, rot: 1 };
        let mut f1 = synthetic(10, 300);
        let mut f2 = synthetic(10, 300);
        let fe1 = front_end(&mut f1, &bounds);
        let fe2 = front_end(&mut f2, &bounds);
        assert_eq!(fe1.perm, fe2.perm);
        assert_eq!(fe1.base_pos, fe2.base_pos);
    }
}
