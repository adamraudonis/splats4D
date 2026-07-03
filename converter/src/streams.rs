//! Binary stream primitives: zigzag, byte-plane shuffle, zstd pack/unpack,
//! order-0 entropy estimation.

use anyhow::{bail, Result};

pub fn zigzag(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

pub fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

/// Attribute ids used in stream headers.
pub const ATTR_POS: u8 = 0;
pub const ATTR_ROT: u8 = 1;
pub const ATTR_RGB: u8 = 2;
pub const ATTR_ALPHA: u8 = 3;
pub const ATTR_SCALE: u8 = 4;
pub const ATTR_NAMES: [&str; 5] = ["pos", "rot", "rgb", "alpha", "scale"];

pub const KIND_KEY: u8 = 0;
pub const KIND_DELTA: u8 = 1;
pub const KIND_STATIC: u8 = 2;

/// A stream of unsigned symbols before compression.
pub struct RawStream {
    pub attr: u8,
    pub kind: u8,
    pub syms: Vec<u32>,
}

pub struct PackedStream {
    pub attr: u8,
    pub kind: u8,
    pub sym_width: u8,
    pub elems: u32,
    pub raw_bytes: usize,
    pub entropy_bytes: f64,
    pub payload: Vec<u8>, // zstd-compressed byte planes
}

pub fn sym_width(syms: &[u32]) -> u8 {
    let mx = syms.iter().copied().max().unwrap_or(0);
    if mx < 256 {
        1
    } else if mx < 65536 {
        2
    } else {
        4
    }
}

/// Byte-plane shuffle: plane p holds byte p of every symbol.
pub fn to_planes(syms: &[u32], width: u8) -> Vec<u8> {
    let n = syms.len();
    let w = width as usize;
    let mut out = vec![0u8; n * w];
    for p in 0..w {
        let plane = &mut out[p * n..(p + 1) * n];
        for (i, s) in syms.iter().enumerate() {
            plane[i] = (s >> (8 * p)) as u8;
        }
    }
    out
}

pub fn from_planes(bytes: &[u8], width: u8, n: usize) -> Vec<u32> {
    let w = width as usize;
    let mut out = vec![0u32; n];
    for p in 0..w {
        let plane = &bytes[p * n..(p + 1) * n];
        for (i, b) in plane.iter().enumerate() {
            out[i] |= (*b as u32) << (8 * p);
        }
    }
    out
}

/// Order-0 Shannon entropy of the byte planes, in bytes (the honest
/// "theoretical minimum" for a memoryless coder on our emitted bytes).
pub fn planes_entropy_bytes(planes: &[u8]) -> f64 {
    let mut hist = [0u64; 256];
    for b in planes {
        hist[*b as usize] += 1;
    }
    let total = planes.len() as f64;
    if total == 0.0 {
        return 0.0;
    }
    let mut bits = 0.0;
    for c in hist {
        if c > 0 {
            let p = c as f64 / total;
            bits -= p.log2() * c as f64;
        }
    }
    bits / 8.0
}

pub fn pack(s: RawStream, level: i32) -> Result<PackedStream> {
    let width = sym_width(&s.syms);
    let planes = to_planes(&s.syms, width);
    let entropy = planes_entropy_bytes(&planes);
    // level 0 = auto: 19 everywhere except very large (noise-dominated) streams,
    // where 19 costs ~10x the CPU of 13 for ~4% size — a bad trade by default.
    let level = if level == 0 {
        if planes.len() > 8 << 20 {
            13
        } else {
            19
        }
    } else {
        level
    };
    // multithread zstd for large streams — they dominate the encode wall-time.
    // JobSize must be set explicitly: the default (4x window) exceeds our
    // stream sizes and silently disables parallelism.
    let payload = if planes.len() > 4 << 20 {
        let mut enc = zstd::stream::write::Encoder::new(Vec::new(), level)?;
        enc.multithread(rayon::current_num_threads() as u32)?;
        enc.set_parameter(zstd::zstd_safe::CParameter::JobSize(2 << 20))?;
        std::io::Write::write_all(&mut enc, &planes)?;
        enc.finish()?
    } else {
        zstd::bulk::compress(&planes, level)?
    };
    Ok(PackedStream {
        attr: s.attr,
        kind: s.kind,
        sym_width: width,
        elems: s.syms.len() as u32,
        raw_bytes: planes.len(),
        entropy_bytes: entropy,
        payload,
    })
}

pub const STREAM_HDR: usize = 11; // attr u8, kind u8, width u8, elems u32, comp_len u32

pub fn write_stream(out: &mut Vec<u8>, s: &PackedStream) {
    out.push(s.attr);
    out.push(s.kind);
    out.push(s.sym_width);
    out.extend_from_slice(&s.elems.to_le_bytes());
    out.extend_from_slice(&(s.payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&s.payload);
}

pub struct ReadStream {
    pub attr: u8,
    pub kind: u8,
    pub syms: Vec<u32>,
}

/// Reads one stream at `off`; returns (stream, next offset).
pub fn read_stream(buf: &[u8], off: usize) -> Result<(ReadStream, usize)> {
    if buf.len() < off + STREAM_HDR {
        bail!("truncated stream header");
    }
    let attr = buf[off];
    let kind = buf[off + 1];
    let width = buf[off + 2];
    let elems = u32::from_le_bytes(buf[off + 3..off + 7].try_into()?) as usize;
    let comp = u32::from_le_bytes(buf[off + 7..off + 11].try_into()?) as usize;
    let start = off + STREAM_HDR;
    if buf.len() < start + comp {
        bail!("truncated stream payload");
    }
    let planes = zstd::bulk::decompress(&buf[start..start + comp], elems * width as usize)?;
    let syms = from_planes(&planes, width, elems);
    Ok((ReadStream { attr, kind, syms }, start + comp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrip() {
        for v in [-5i32, -1, 0, 1, 7, 1000, -70000, i32::MIN / 2] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn stream_roundtrip() {
        let syms: Vec<u32> = (0..10000u32).map(|i| (i * 2654435761) % 300).collect();
        let packed = pack(RawStream { attr: 2, kind: 1, syms: syms.clone() }, 3).unwrap();
        let mut buf = Vec::new();
        write_stream(&mut buf, &packed);
        let (rs, next) = read_stream(&buf, 0).unwrap();
        assert_eq!(next, buf.len());
        assert_eq!(rs.syms, syms);
        assert_eq!(rs.attr, 2);
    }
}
