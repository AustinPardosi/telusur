//! First-stage scoring of 1-bit codes against a float query with lookup tables.
//!
//! The query is split into groups of 4 dims. A code's group can only be one of 16 sign
//! patterns, so each group gets a 16-entry table of partial dot products, quantized to u8.
//! A code's score is the u16 sum of one table entry per group: the dot product up to one
//! scale and one offset shared by every code, so the ranking is kept.
//!
//! Codes are stored in blocks of 32, transposed: row j of a block is byte j of its 32 codes,
//! the shape a 16-way byte shuffle wants (the SIMD kernels come later; this is the scalar one).

/// Codes per block.
pub const BLOCK: usize = 32;

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
