//! Neural retrieval index: docs split into clusters by spherical k-means on their code-dims
//! prefix; 1-bit codes in blocks of 32 for the first stage; f16 rerank vectors for exact
//! scoring. A query scores only the docs in the clusters nearest to it.
//!
//! Files: `vectors.json` (dims, cluster count), `centroids.f32` (nlist x d_code),
//! `clusters.bin` (nlist + 1 u64 slot offsets), `codes.bin` (each cluster's codes as
//! transposed blocks, back to back), `ids.bin` (u32 doc id per slot, `u32::MAX` for padding),
//! `rerank.f16` (d_rerank f16 per doc, in doc id order).

use super::{
    kernel::{BLOCK, Kernel, QueryLut, dot_f16, pack_blocks, prefetch, scan},
    kmeans,
    quant::{binarize, dot, truncate_normalize},
};
use crate::util::{Fbin, MmapSlice, TopK, mmap};
use anyhow::{Result, ensure};
use half::f16;
use memmap2::Mmap;
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const PADDING: u32 = u32::MAX;
/// Clusters probed when a request does not say: the smallest count that reached recall@10 of
/// 0.95 against rerank ground truth on DBpedia 1M (B3, bench-results/b3-ann-dbpedia-*), which
/// it does at the default rerank depth of 500.
pub const DEFAULT_NPROBE: usize = 256;
/// A filter allowing at most this many docs skips the clusters and scores them all exactly.
pub const EXACT_FILTER_MAX: u64 = 20_000;
/// Rerank rows are prefetched this many docs ahead of the one being scored. On DBpedia's 1%
/// filter (B5, #24), 1 and 2 were fastest and 3, 4, 8 and 16 slower; 2 also gave the fastest
/// unfiltered search.
const PREFETCH_AHEAD: usize = 2;
/// k-means trains on this many sample points per cluster, for this many rounds.
const SAMPLE_PER_CLUSTER: usize = 64;
const ITERS: usize = 12;

#[derive(Serialize, Deserialize)]
struct Meta {
    /// Dims of the full vectors the index was built from.
    dims: usize,
    d_code: usize,
    d_rerank: usize,
    nlist: usize,
}

/// About 2 x sqrt(n), rounded to the nearest power of two (2048 for 1M docs).
pub fn default_nlist(n: usize) -> usize {
    let target = 2.0 * (n.max(1) as f64).sqrt();
    1 << target.log2().round() as u32
}

/// Builds the vector files in `out`, one row of `vectors` per doc.
pub fn build(
    vectors: &Fbin,
    out: &Path,
    d_code: usize,
    d_rerank: Option<usize>,
    nlist: Option<usize>,
) -> Result<()> {
    let n = vectors.n;
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
    let nlist = nlist.unwrap_or_else(|| default_nlist(n)).clamp(1, n.max(1));

    // Codes in doc order, and rerank vectors streamed to disk.
    let bytes = d_code / 8;
    let mut codes = vec![0u8; n * bytes];
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

    // Clusters: train on a sample of the code-dims prefixes, then assign every doc.
    let prefix = |i: usize| truncate_normalize(vectors.row(i), d_code);
    let (centroids, assigned) = if nlist == 1 {
        (vec![0f32; d_code], vec![0u32; n])
    } else {
        let mut rng = StdRng::seed_from_u64(42);
        let picks = sample(&mut rng, n, (SAMPLE_PER_CLUSTER * nlist).min(n));
        let train: Vec<f32> = picks.into_iter().flat_map(prefix).collect();
        let centroids = kmeans::train(&train, d_code, nlist, ITERS, &mut rng);
        let assigned = kmeans::assign(n, d_code, prefix, &centroids);
        (centroids, assigned)
    };

    // Lay clusters out back to back, each padded to whole blocks; docs in id order inside.
    let mut members = vec![Vec::new(); nlist];
    for (doc, &c) in assigned.iter().enumerate() {
        members[c as usize].push(doc as u32);
    }
    let (mut blocks, mut ids, mut offsets) = (Vec::new(), Vec::new(), vec![0u64]);
    for docs in &members {
        let cluster: Vec<u8> = docs
            .iter()
            .flat_map(|&d| &codes[d as usize * bytes..][..bytes])
            .copied()
            .collect();
        blocks.extend(pack_blocks(&cluster, bytes));
        ids.extend(docs);
        ids.resize(ids.len().div_ceil(BLOCK) * BLOCK, PADDING);
        offsets.push(ids.len() as u64);
    }
    fs::write(out.join("codes.bin"), blocks)?;
    fs::write(out.join("ids.bin"), bytemuck::cast_slice(&ids))?;
    fs::write(out.join("clusters.bin"), bytemuck::cast_slice(&offsets))?;
    fs::write(out.join("centroids.f32"), bytemuck::cast_slice(&centroids))?;
    let meta = Meta {
        dims: vectors.d,
        d_code,
        d_rerank,
        nlist,
    };
    fs::write(out.join("vectors.json"), serde_json::to_vec(&meta)?)?;
    Ok(())
}

pub struct VectorIndex {
    meta: Meta,
    centroids: MmapSlice<f32>,
    offsets: MmapSlice<u64>,
    codes: Mmap,
    ids: MmapSlice<u32>,
    rerank: MmapSlice<f16>,
}

/// What one neural search did.
pub struct Trace {
    pub route: f64,
    pub scan: f64,
    pub rerank: f64,
    pub clusters_probed: usize,
    /// Slots scanned, padding included: the kernel's work.
    pub codes_scanned: usize,
}

impl VectorIndex {
    /// The vector part of the index in `dir`, if it was built with vectors.
    pub fn open(dir: &Path) -> Result<Option<Self>> {
        let Ok(meta) = fs::read(dir.join("vectors.json")) else {
            return Ok(None);
        };
        Ok(Some(Self {
            meta: serde_json::from_slice(&meta)?,
            centroids: MmapSlice::open(&dir.join("centroids.f32"))?,
            offsets: MmapSlice::open(&dir.join("clusters.bin"))?,
            codes: mmap(&dir.join("codes.bin"))?,
            ids: MmapSlice::open(&dir.join("ids.bin"))?,
            rerank: MmapSlice::open(&dir.join("rerank.f16"))?,
        }))
    }

    pub fn nlist(&self) -> usize {
        self.meta.nlist
    }

    pub fn dims(&self) -> usize {
        self.meta.dims
    }

    /// Top k (doc, score) for a full query vector. Route: the `nprobe` clusters whose
    /// centroids best match the query. Scan: score their codes with `kernel`, best cluster
    /// first so the threshold rises early, keeping the `rerank_depth` best (at least k).
    /// Rerank: score those exactly with rerank vectors. A rerank depth of 0 skips the rerank
    /// and returns code scores. Ties go to the smaller doc id.
    ///
    /// With `allowed` (a filter's docs), only those docs can be returned. If there are at most
    /// `EXACT_FILTER_MAX`, they are all scored exactly with rerank vectors and nothing is
    /// probed. Otherwise the probe widens to `nprobe / sqrt(selectivity)` clusters (at most all
    /// of them) and disallowed docs are dropped before the candidate heap. Widening by
    /// `1 / selectivity` probed every cluster at 10% for recall nobody asked for; the square
    /// root kept filtered recall above unfiltered on DBpedia and FineWeb (B5, #25).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        rerank_depth: usize,
        nprobe: usize,
        kernel: Kernel,
        allowed: Option<&RoaringBitmap>,
    ) -> Result<(Vec<(u32, f32)>, Trace)> {
        let Meta {
            dims,
            d_code,
            d_rerank,
            nlist,
        } = self.meta;
        ensure!(
            query.len() == dims,
            "the query vector has {} dims; the index was built from {dims}",
            query.len()
        );
        ensure!(nprobe >= 1, "probe at least one cluster");
        let rerank = self.rerank.as_slice();
        let rerank_row = |doc: u32| &rerank[doc as usize * d_rerank..][..d_rerank];

        if let Some(allowed) = allowed
            && allowed.len() <= EXACT_FILTER_MAX
        {
            let t = Instant::now();
            let q = truncate_normalize(query, d_rerank);
            let mut top = TopK::new(k);
            let mut ahead = allowed.iter().skip(PREFETCH_AHEAD);
            for doc in allowed {
                if let Some(a) = ahead.next() {
                    prefetch(rerank_row(a));
                }
                top.push(doc, dot_f16(rerank_row(doc), &q));
            }
            let trace = Trace {
                route: 0.0,
                scan: 0.0,
                rerank: ms(t),
                clusters_probed: 0,
                codes_scanned: 0,
            };
            return Ok((top.into_sorted(), trace));
        }
        let nprobe = match allowed {
            Some(a) => {
                let docs = (rerank.len() / d_rerank) as f64;
                (nprobe as f64 * (docs / a.len() as f64).sqrt()).ceil() as usize
            }
            None => nprobe,
        };

        let t = Instant::now();
        let q = truncate_normalize(query, d_code);
        let mut nearest = TopK::new(nprobe.min(nlist));
        for (c, centroid) in self.centroids.as_slice().chunks_exact(d_code).enumerate() {
            nearest.push(c as u32, dot(centroid, &q));
        }
        let probe = nearest.into_sorted();
        let route = ms(t);

        let t = Instant::now();
        let lut = QueryLut::new(&q);
        let (offsets, ids) = (self.offsets.as_slice(), self.ids.as_slice());
        let mut candidates = TopK::new(if rerank_depth == 0 {
            k
        } else {
            rerank_depth.max(k)
        });
        let mut min_accept = 0u16;
        let mut scores = Vec::new();
        let mut codes_scanned = 0;
        for &(c, _) in &probe {
            let (from, to) = (
                offsets[c as usize] as usize,
                offsets[c as usize + 1] as usize,
            );
            scores.resize(to - from, 0);
            scan(
                kernel,
                &lut,
                &self.codes[from * d_code / 8..to * d_code / 8],
                &mut scores,
            )?;
            codes_scanned += to - from;
            for (&doc, &s) in ids[from..to].iter().zip(&scores) {
                // Cheap compare first; ties may still get in on doc id, so TopK has the last word.
                if s < min_accept || doc == PADDING || allowed.is_some_and(|a| !a.contains(doc)) {
                    continue;
                }
                candidates.push(doc, f32::from(s));
                min_accept = candidates.threshold().max(0.0) as u16;
            }
        }
        let scan = ms(t);

        let t = Instant::now();
        let hits = if rerank_depth == 0 {
            candidates.into_sorted()
        } else {
            let q = truncate_normalize(query, d_rerank);
            let mut top = TopK::new(k);
            let docs = candidates.into_sorted();
            for (i, &(doc, _)) in docs.iter().enumerate() {
                if let Some(&(a, _)) = docs.get(i + PREFETCH_AHEAD) {
                    prefetch(rerank_row(a));
                }
                top.push(doc, dot_f16(rerank_row(doc), &q));
            }
            top.into_sorted()
        };
        let rerank = ms(t);
        Ok((
            hits,
            Trace {
                route,
                scan,
                rerank,
                clusters_probed: probe.len(),
                codes_scanned,
            },
        ))
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
