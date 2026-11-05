//! Neural retrieval index: 1-bit codes in blocks of 32 for the first stage, f16 rerank vectors
//! for exact scoring. All docs sit in one cluster until clustering lands.
//!
//! Files: `vectors.json` (dims), `codes.bin` (transposed blocks), `ids.bin` (u32 doc id per
//! slot, `u32::MAX` for padding), `rerank.f16` (d_rerank f16 per doc, in doc id order).

use super::{
    kernel::{BLOCK, QueryLut, pack_blocks, scan_scalar},
    quant::{binarize, dot_f16, truncate_normalize},
};
use crate::util::{Fbin, MmapSlice, TopK, mmap};
use anyhow::{Result, ensure};
use half::f16;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const PADDING: u32 = u32::MAX;

#[derive(Serialize, Deserialize)]
struct Meta {
    /// Dims of the full vectors the index was built from.
    dims: usize,
    d_code: usize,
    d_rerank: usize,
}

/// Builds the vector files in `out`, one row of `vectors` per doc.
pub fn build(vectors: &Fbin, out: &Path, d_code: usize, d_rerank: Option<usize>) -> Result<()> {
    let d_rerank = d_rerank.unwrap_or(vectors.d.min(1024));
    // u16 sums of 255 per group overflow past 1024 dims (256 groups x 255 = 65,280).
    ensure!(
        d_code.is_multiple_of(64) && (64..=1024).contains(&d_code),
        "code dims must be a multiple of 64, at most 1024; got {d_code}"
    );
    ensure!(
        d_code <= vectors.d,
        "code dims {d_code} exceed the vectors' {} dims",
        vectors.d
    );
    ensure!(
        (1..=vectors.d).contains(&d_rerank),
        "rerank dims must be 1 to {}; got {d_rerank}",
        vectors.d
    );

    let bytes = d_code / 8;
    let mut codes = vec![0u8; vectors.n * bytes];
    let mut rerank = BufWriter::new(File::create(out.join("rerank.f16"))?);
    for (i, code) in codes.chunks_exact_mut(bytes).enumerate() {
        let v = vectors.row(i);
        binarize(&truncate_normalize(v, d_code), code);
        let r: Vec<f16> = truncate_normalize(v, d_rerank)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        rerank.write_all(bytemuck::cast_slice(&r))?;
    }
    rerank.flush()?;
    fs::write(out.join("codes.bin"), pack_blocks(&codes, bytes))?;
    let slots = vectors.n.div_ceil(BLOCK) * BLOCK;
    let ids: Vec<u32> = (0..slots as u32)
        .map(|i| if (i as usize) < vectors.n { i } else { PADDING })
        .collect();
    fs::write(out.join("ids.bin"), bytemuck::cast_slice(&ids))?;
    let meta = Meta {
        dims: vectors.d,
        d_code,
        d_rerank,
    };
    fs::write(out.join("vectors.json"), serde_json::to_vec(&meta)?)?;
    Ok(())
}

pub struct VectorIndex {
    meta: Meta,
    codes: Mmap,
    ids: MmapSlice<u32>,
    rerank: MmapSlice<f16>,
}

/// Milliseconds per stage of one neural search.
pub struct StageTimes {
    pub route: f64,
    pub scan: f64,
    pub rerank: f64,
}

impl VectorIndex {
    /// The vector part of the index in `dir`, if it was built with vectors.
    pub fn open(dir: &Path) -> Result<Option<Self>> {
        let Ok(meta) = fs::read(dir.join("vectors.json")) else {
            return Ok(None);
        };
        Ok(Some(Self {
            meta: serde_json::from_slice(&meta)?,
            codes: mmap(&dir.join("codes.bin"))?,
            ids: MmapSlice::open(&dir.join("ids.bin"))?,
            rerank: MmapSlice::open(&dir.join("rerank.f16"))?,
        }))
    }

    /// Top k (doc, score) for a full query vector: scan the codes, keep the `rerank_depth`
    /// best (at least k), then score those exactly with rerank vectors. Ties go to the smaller
    /// doc id.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        rerank_depth: usize,
    ) -> Result<(Vec<(u32, f32)>, StageTimes)> {
        let Meta {
            dims,
            d_code,
            d_rerank,
        } = self.meta;
        ensure!(
            query.len() == dims,
            "the query vector has {} dims; the index was built from {dims}",
            query.len()
        );

        let t = Instant::now();
        let ids = self.ids.as_slice(); // one cluster: every slot
        let route = ms(t);

        let t = Instant::now();
        let lut = QueryLut::new(&truncate_normalize(query, d_code));
        let mut scores = vec![0u16; ids.len()];
        scan_scalar(&lut, &self.codes, &mut scores);
        let mut candidates = TopK::new(rerank_depth.max(k));
        let mut min_accept = 0u16;
        for (&doc, &s) in ids.iter().zip(&scores) {
            // Cheap compare first; ties may still get in on doc id, so TopK has the last word.
            if s < min_accept || doc == PADDING {
                continue;
            }
            candidates.push(doc, f32::from(s));
            min_accept = candidates.threshold().max(0.0) as u16;
        }
        let scan = ms(t);

        let t = Instant::now();
        let q = truncate_normalize(query, d_rerank);
        let rerank = self.rerank.as_slice();
        let mut top = TopK::new(k);
        for (doc, _) in candidates.into_sorted() {
            let at = doc as usize * d_rerank;
            top.push(doc, dot_f16(&rerank[at..at + d_rerank], &q));
        }
        let rerank = ms(t);
        Ok((
            top.into_sorted(),
            StageTimes {
                route,
                scan,
                rerank,
            },
        ))
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
