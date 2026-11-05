//! First-stage scoring of 1-bit codes against a float query with lookup tables.
//!
//! The query is split into groups of 4 dims. A code's group can only be one of 16 sign
//! patterns, so each group gets a 16-entry table of partial dot products, quantized to u8.
//! A code's score is the u16 sum of one table entry per group: the dot product up to one
//! scale and one offset shared by every code, so the ranking is kept.
//!
//! Codes are stored in blocks of 32, transposed: row j of a block is byte j of its 32 codes,
//! the shape a 16-way byte shuffle wants. `pshufb` / `vpshufb` (x86) and `tbl` (ARM) are
//! 16-entry table lookups, so with a table in one register and 32 docs' nibbles in another,
//! one instruction does 32 lookups. Every SIMD kernel must equal `scan_scalar` bit for bit.

use anyhow::{Result, bail};
use serde::Serialize;
use std::sync::OnceLock;

/// Codes per block.
pub const BLOCK: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Kernel {
    Scalar,
    Avx2,
    /// AVX-512BW: needs both avx512f and avx512bw.
    Avx512,
    Neon,
}

impl Kernel {
    /// Fastest first.
    const ALL: [Kernel; 4] = [Kernel::Avx512, Kernel::Avx2, Kernel::Neon, Kernel::Scalar];

    pub fn is_available(self) -> bool {
        match self {
            Kernel::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => {
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            }
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Every kernel this CPU can run, fastest first.
    pub fn available() -> Vec<Kernel> {
        Self::ALL.into_iter().filter(|k| k.is_available()).collect()
    }

    /// The kernel searches use unless a request picks one: the one `TELUSUR_KERNEL` names
    /// (scalar, avx2, avx512, neon), else the fastest available. Read once per process.
    pub fn default_for_process() -> Result<Kernel> {
        static CHOICE: OnceLock<Result<Kernel, String>> = OnceLock::new();
        let choice = CHOICE.get_or_init(|| match std::env::var("TELUSUR_KERNEL") {
            Err(_) => Ok(Self::available()[0]),
            Ok(name) => {
                let k = <Kernel as clap::ValueEnum>::from_str(&name, true).map_err(|_| {
                    format!("TELUSUR_KERNEL={name} is not a kernel (scalar, avx2, avx512, neon)")
                })?;
                if k.is_available() {
                    Ok(k)
                } else {
                    Err(format!("TELUSUR_KERNEL={name}: this CPU cannot run it"))
                }
            }
        });
        choice.clone().map_err(anyhow::Error::msg)
    }
}

/// Scores every slot of `blocks` into `out` with `kernel`, which must be available.
pub fn scan(kernel: Kernel, lut: &QueryLut, blocks: &[u8], out: &mut [u16]) -> Result<()> {
    if !kernel.is_available() {
        bail!("this CPU cannot run the {kernel:?} kernel");
    }
    assert_eq!(
        out.len() * lut.tables.len() / 2,
        blocks.len(),
        "one u16 per slot"
    );
    match kernel {
        Kernel::Scalar => scan_scalar(lut, blocks, out),
        // SAFETY: each target-feature kernel runs only after `is_available` found its features.
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { x86::scan_avx2(lut, blocks, out) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => unsafe { x86::scan_avx512(lut, blocks, out) },
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => unsafe { arm::scan_neon(lut, blocks, out) },
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_available is false for other architectures' kernels"),
    }
    Ok(())
}

pub struct QueryLut {
    /// One table per group of 4 query dims: entry n is the partial dot product with the
    /// sign pattern n (bit k set = dim 4g + k is +1), shifted to start at 0 and scaled.
    pub tables: Vec<[u8; 16]>,
}

impl QueryLut {
    /// `q` is the normalized code-dims prefix of the query, a multiple of 64 dims long.
    pub fn new(q: &[f32]) -> Self {
        let raw: Vec<[f32; 16]> = q
            .as_chunks::<4>()
            .0
            .iter()
            .map(|g| {
                std::array::from_fn(|n| {
                    (0..4)
                        .map(|k| if n >> k & 1 == 1 { g[k] } else { -g[k] })
                        .sum()
                })
            })
            .collect();
        let range = |t: &[f32; 16]| {
            let (lo, hi) = t
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            (lo, hi - lo)
        };
        // One scale for all groups keeps the sum a dot product; each group's own offset is a
        // constant shared by every code, so dropping it keeps the ranking.
        let widest = raw.iter().map(|t| range(t).1).fold(0.0, f32::max);
        let scale = if widest > 0.0 { 255.0 / widest } else { 0.0 };
        let tables = raw
            .iter()
            .map(|t| {
                let lo = range(t).0;
                std::array::from_fn(|n| ((t[n] - lo) * scale).round() as u8)
            })
            .collect();
        Self { tables }
    }
}

/// Packs codes of `bytes` bytes each, back to back, into transposed blocks. The last block is
/// padded with zero codes.
pub fn pack_blocks(codes: &[u8], bytes: usize) -> Vec<u8> {
    let n = codes.len() / bytes;
    let mut out = vec![0u8; n.div_ceil(BLOCK) * BLOCK * bytes];
    for (i, code) in codes.chunks_exact(bytes).enumerate() {
        let block = &mut out[i / BLOCK * BLOCK * bytes..][..BLOCK * bytes];
        for (j, &b) in code.iter().enumerate() {
            block[j * BLOCK + i % BLOCK] = b;
        }
    }
    out
}

/// Scores every slot of `blocks` into `out` (32 per block). The reference every other kernel
/// must match bit for bit.
pub fn scan_scalar(lut: &QueryLut, blocks: &[u8], out: &mut [u16]) {
    let bytes = lut.tables.len() / 2;
    for (block, out) in blocks
        .chunks_exact(bytes * BLOCK)
        .zip(out.as_chunks_mut::<BLOCK>().0)
    {
        let mut acc = [0u16; BLOCK];
        for (j, row) in block.as_chunks::<BLOCK>().0.iter().enumerate() {
            let (lo, hi) = (&lut.tables[2 * j], &lut.tables[2 * j + 1]);
            for (a, &b) in acc.iter_mut().zip(row) {
                *a += u16::from(lo[usize::from(b & 15)]) + u16::from(hi[usize::from(b >> 4)]);
            }
        }
        out.copy_from_slice(&acc);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{BLOCK, QueryLut};
    use std::arch::x86_64::*;

    /// Writes 32 docs' u16 sums: `even` holds docs 0, 2, .., 30 and `odd` docs 1, 3, .., 31,
    /// as 16 u16 lanes each.
    #[target_feature(enable = "avx2")]
    fn store_interleaved(even: __m256i, odd: __m256i, out: &mut [u16; BLOCK]) {
        // unpack works within 128-bit lanes: a = docs 0..8 | 16..24, b = docs 8..16 | 24..32.
        let a = _mm256_unpacklo_epi16(even, odd);
        let b = _mm256_unpackhi_epi16(even, odd);
        // SAFETY: `out` is 32 u16 = 64 bytes, exactly two unaligned 32-byte stores.
        unsafe {
            _mm256_storeu_si256(
                out.as_mut_ptr().cast(),
                _mm256_permute2x128_si256(a, b, 0x20),
            );
            _mm256_storeu_si256(
                out.as_mut_ptr().add(16).cast(),
                _mm256_permute2x128_si256(a, b, 0x31),
            );
        }
    }

    /// One row (32 bytes) per step. `vpshufb` looks up within each 128-bit lane, so each
    /// 16-byte table is broadcast to both lanes. Sums widen to u16 by viewing the register as
    /// 16 u16 lanes: `x & 0x00FF` is the even docs, `x >> 8` the odd ones.
    #[target_feature(enable = "avx2")]
    pub(super) fn scan_avx2(lut: &QueryLut, blocks: &[u8], out: &mut [u16]) {
        let bytes = lut.tables.len() / 2;
        let nibble = _mm256_set1_epi8(0x0F);
        let low_byte = _mm256_set1_epi16(0x00FF);
        for (block, out) in blocks
            .chunks_exact(bytes * BLOCK)
            .zip(out.as_chunks_mut::<BLOCK>().0)
        {
            let (mut even, mut odd) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            for (row, t) in block
                .as_chunks::<BLOCK>()
                .0
                .iter()
                .zip(lut.tables.as_chunks::<2>().0)
            {
                // SAFETY: `row` is 32 bytes and each table 16, the sizes of these unaligned loads.
                let (r, t_lo, t_hi) = unsafe {
                    (
                        _mm256_loadu_si256(row.as_ptr().cast()),
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(t[0].as_ptr().cast())),
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(t[1].as_ptr().cast())),
                    )
                };
                let lo = _mm256_shuffle_epi8(t_lo, _mm256_and_si256(r, nibble));
                let hi =
                    _mm256_shuffle_epi8(t_hi, _mm256_and_si256(_mm256_srli_epi16(r, 4), nibble));
                even = _mm256_add_epi16(
                    even,
                    _mm256_add_epi16(
                        _mm256_and_si256(lo, low_byte),
                        _mm256_and_si256(hi, low_byte),
                    ),
                );
                odd = _mm256_add_epi16(
                    odd,
                    _mm256_add_epi16(_mm256_srli_epi16(lo, 8), _mm256_srli_epi16(hi, 8)),
                );
            }
            store_interleaved(even, odd, out);
        }
    }

    /// Two rows (64 bytes) per step: row 2p in the low 256 bits, row 2p + 1 in the high. The
    /// low-nibble tables are `[g(4p), g(4p), g(4p+2), g(4p+2)]` per 128-bit lane and the
    /// high-nibble ones `[g(4p+1), g(4p+1), g(4p+3), g(4p+3)]`; both halves sum the same 32
    /// docs, so they are added together at the end.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) fn scan_avx512(lut: &QueryLut, blocks: &[u8], out: &mut [u16]) {
        let bytes = lut.tables.len() / 2; // a multiple of 8, so rows pair up
        let lane = |t: &[u8; 16]| {
            // SAFETY: a table is 16 bytes, one unaligned 128-bit load.
            _mm512_broadcast_i32x4(unsafe { _mm_loadu_si128(t.as_ptr().cast()) })
        };
        // Per row pair: the low-nibble and high-nibble table registers.
        let pairs: Vec<(__m512i, __m512i)> = lut
            .tables
            .as_chunks::<4>()
            .0
            .iter()
            .map(|t| {
                let half = |a: &[u8; 16], b: &[u8; 16]| {
                    _mm512_inserti64x4(lane(a), _mm512_castsi512_si256(lane(b)), 1)
                };
                (half(&t[0], &t[2]), half(&t[1], &t[3]))
            })
            .collect();
        let nibble = _mm512_set1_epi8(0x0F);
        let low_byte = _mm512_set1_epi16(0x00FF);
        for (block, out) in blocks
            .chunks_exact(bytes * BLOCK)
            .zip(out.as_chunks_mut::<BLOCK>().0)
        {
            let (mut even, mut odd) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            for (rows, &(t_lo, t_hi)) in block.as_chunks::<{ 2 * BLOCK }>().0.iter().zip(&pairs) {
                // SAFETY: `rows` is 64 bytes, one unaligned 512-bit load.
                let r = unsafe { _mm512_loadu_si512(rows.as_ptr().cast()) };
                let lo = _mm512_shuffle_epi8(t_lo, _mm512_and_si512(r, nibble));
                let hi =
                    _mm512_shuffle_epi8(t_hi, _mm512_and_si512(_mm512_srli_epi16(r, 4), nibble));
                even = _mm512_add_epi16(
                    even,
                    _mm512_add_epi16(
                        _mm512_and_si512(lo, low_byte),
                        _mm512_and_si512(hi, low_byte),
                    ),
                );
                odd = _mm512_add_epi16(
                    odd,
                    _mm512_add_epi16(_mm512_srli_epi16(lo, 8), _mm512_srli_epi16(hi, 8)),
                );
            }
            let fold = |v: __m512i| {
                _mm256_add_epi16(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1))
            };
            store_interleaved(fold(even), fold(odd), out);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{BLOCK, QueryLut};
    use std::arch::aarch64::*;

    /// A row is 32 bytes, two 16-byte registers. `vqtbl1q_u8` is the 16-entry lookup;
    /// `vaddl_u8` / `vaddl_high_u8` add the two lookups while widening to u16, which keeps
    /// docs in their natural order.
    #[target_feature(enable = "neon")]
    pub(super) fn scan_neon(lut: &QueryLut, blocks: &[u8], out: &mut [u16]) {
        let bytes = lut.tables.len() / 2;
        let nibble = vdupq_n_u8(0x0F);
        for (block, out) in blocks
            .chunks_exact(bytes * BLOCK)
            .zip(out.as_chunks_mut::<BLOCK>().0)
        {
            let mut acc = [vdupq_n_u16(0); 4]; // docs 0..8, 8..16, 16..24, 24..32
            for (row, t) in block
                .as_chunks::<BLOCK>()
                .0
                .iter()
                .zip(lut.tables.as_chunks::<2>().0)
            {
                // SAFETY: each table is 16 bytes and `row` 32, so all four loads are in bounds.
                let (t_lo, t_hi, r0, r1) = unsafe {
                    (
                        vld1q_u8(t[0].as_ptr()),
                        vld1q_u8(t[1].as_ptr()),
                        vld1q_u8(row.as_ptr()),
                        vld1q_u8(row.as_ptr().add(16)),
                    )
                };
                for (h, r) in [r0, r1].into_iter().enumerate() {
                    let lo = vqtbl1q_u8(t_lo, vandq_u8(r, nibble));
                    let hi = vqtbl1q_u8(t_hi, vshrq_n_u8::<4>(r));
                    acc[2 * h] = vaddq_u16(acc[2 * h], vaddl_u8(vget_low_u8(lo), vget_low_u8(hi)));
                    acc[2 * h + 1] = vaddq_u16(acc[2 * h + 1], vaddl_high_u8(lo, hi));
                }
            }
            for (i, a) in acc.into_iter().enumerate() {
                // SAFETY: `out` is 32 u16 and each store writes 8 of them at 8 * i.
                unsafe { vst1q_u16(out.as_mut_ptr().add(8 * i), a) };
            }
        }
    }
}
