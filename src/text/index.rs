//! Keyword index: tiered posting lists, an FST dictionary, one-byte doc length norms, and
//! exhaustive BM25 scored the way Lucene does it.
//!
//! Files:
//! - `terms.fst`: term to value. A singleton (df 1) packs `SINGLETON | tf << 32 | doc`; any
//!   other term gives the byte offset of its postings in `postings.bin`.
//! - `postings.bin`: per term `varint df`, `u16 tier bitset` (bit t = the tier with tf == t,
//!   t in 1..=15; bit 0 = the overflow tier, tf >= 16), then each present tier in the order
//!   1..=15, overflow: `varint n_docs, varint byte_len`, then n_docs x `varint doc_delta`
//!   (overflow tier: plus `varint tf`). The first doc_delta of a tier is doc + 1, later ones
//!   are gaps, so every varint is at least 1.
//! - `norms.bin`: one SmallFloat byte per doc.
//! - `keyword.json`: collection stats for idf and avgdl.

use super::{
    analyzer::analyze,
    codec::{byte4_to_int, get_varint, int_to_byte4, put_varint},
};
use crate::util::mmap;
use anyhow::Result;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::BufWriter,
    path::Path,
};

pub const K1: f32 = 0.9;
pub const B: f32 = 0.4;
const SINGLETON: u64 = 1 << 63;
const OVERFLOW: usize = 0;
/// Order tiers are written in: tf 1..=15, then overflow.
const TIERS: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, OVERFLOW];

#[derive(Default, Serialize, Deserialize)]
struct Stats {
    /// Docs with at least one token, Lucene's `docCount`.
    doc_count: u64,
    /// Tokens over all docs, Lucene's `sumTotalTermFreq`.
    total_len: u64,
}

// ponytail: every posting list sits in RAM until finish(), which caps a build at a few
// million FineWeb docs on 16 GB. Segment-and-merge is the upgrade path.
#[derive(Default)]
pub struct Builder {
    terms: HashMap<String, Vec<(u32, u32)>>,
    norms: Vec<u8>,
    stats: Stats,
}

impl Builder {
    /// Adds the next doc. Doc ids count up from 0 in the order docs are added.
    pub fn add(&mut self, text: &str) {
        let doc = self.norms.len() as u32;
        let tokens = analyze(text);
        let len = tokens.len();
        let mut tf = HashMap::<String, u32>::new();
        for t in tokens {
            *tf.entry(t).or_default() += 1;
        }
        for (term, tf) in tf {
            self.terms.entry(term).or_default().push((doc, tf));
        }
        self.norms.push(int_to_byte4(len as u32));
        self.stats.doc_count += u64::from(len > 0);
        self.stats.total_len += len as u64;
    }

    /// Writes the index files into `dir` and returns the number of terms.
    pub fn finish(self, dir: &Path) -> Result<usize> {
        let mut terms: Vec<_> = self.terms.into_iter().collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut fst = fst::MapBuilder::new(BufWriter::new(File::create(dir.join("terms.fst"))?))?;
        let mut postings = Vec::new();
        let mut tiers: [Vec<u8>; 16] = Default::default();
        let mut counts = [0u64; 16];
        for (term, list) in &terms {
            if let [(doc, tf)] = list[..] {
                fst.insert(term, SINGLETON | u64::from(tf) << 32 | u64::from(doc))?;
                continue;
            }
            fst.insert(term, postings.len() as u64)?;
            let mut prev = [u32::MAX; 16];
            for &(doc, tf) in list {
                let t = if tf <= 15 { tf as usize } else { OVERFLOW };
                put_varint(&mut tiers[t], u64::from(doc.wrapping_sub(prev[t])));
                if t == OVERFLOW {
                    put_varint(&mut tiers[t], u64::from(tf));
                }
                prev[t] = doc;
                counts[t] += 1;
            }
            put_varint(&mut postings, list.len() as u64);
            let bitset = TIERS
                .iter()
                .filter(|&&t| counts[t] > 0)
                .fold(0u16, |m, &t| m | 1 << t);
            postings.extend(bitset.to_le_bytes());
            for t in TIERS {
                if counts[t] == 0 {
                    continue;
                }
                put_varint(&mut postings, counts[t]);
                put_varint(&mut postings, tiers[t].len() as u64);
                postings.append(&mut tiers[t]);
                counts[t] = 0;
            }
        }
        fst.finish()?;
        fs::write(dir.join("postings.bin"), postings)?;
        fs::write(dir.join("norms.bin"), &self.norms)?;
        fs::write(dir.join("keyword.json"), serde_json::to_vec(&self.stats)?)?;
        Ok(terms.len())
    }
}

pub struct KeywordIndex {
    terms: fst::Map<Mmap>,
    postings: Mmap,
    norms: Mmap,
    doc_count: f64,
    /// Lucene's per-norm cache: `1 / (k1 * (1 - b + b * dl / avgdl))` for each norm byte.
    norm_inverse: [f32; 256],
}

impl KeywordIndex {
    pub fn open(dir: &Path) -> Result<Self> {
        let stats: Stats = serde_json::from_slice(&fs::read(dir.join("keyword.json"))?)?;
        let avgdl = (stats.total_len as f64 / stats.doc_count as f64) as f32;
        Ok(Self {
            terms: fst::Map::new(mmap(&dir.join("terms.fst"))?)?,
            postings: mmap(&dir.join("postings.bin"))?,
            norms: mmap(&dir.join("norms.bin"))?,
            doc_count: stats.doc_count as f64,
            norm_inverse: std::array::from_fn(|i| {
                1.0 / (K1 * ((1.0 - B) + B * byte4_to_int(i as u8) as f32 / avgdl))
            }),
        })
    }

    /// Top k (doc, score) by BM25, best first, ties to the smaller doc id. Exhaustive: every
    /// posting of every query term is scored into a dense accumulator.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        // Like Anserini's bag-of-words query: one clause per distinct term, boosted by its
        // count. A BTreeMap keeps the summing order, and so the scores, deterministic.
        let mut boosts = BTreeMap::<String, u32>::new();
        for t in analyze(query) {
            *boosts.entry(t).or_default() += 1;
        }
        let p = &self.postings[..];
        let mut acc = vec![0f32; self.norms.len()];
        for (term, boost) in boosts {
            let Some(v) = self.terms.get(&term) else {
                continue;
            };
            let mut pos = v as usize;
            let df = if v & SINGLETON != 0 {
                1
            } else {
                get_varint(p, &mut pos)
            };
            let idf = (1.0 + (self.doc_count - df as f64 + 0.5) / (df as f64 + 0.5)).ln() as f32;
            let w = boost as f32 * idf;
            let mut score = |doc: u32, tf: u32| {
                let inv = self.norm_inverse[self.norms[doc as usize] as usize];
                acc[doc as usize] += w - w / (1.0 + tf as f32 * inv);
            };
            if v & SINGLETON != 0 {
                score(v as u32, (v >> 32) as u32 & !(1 << 31));
                continue;
            }
            let bitset = u16::from_le_bytes([p[pos], p[pos + 1]]);
            pos += 2;
            for t in TIERS.into_iter().filter(|&t| bitset & 1 << t != 0) {
                let n = get_varint(p, &mut pos);
                get_varint(p, &mut pos); // byte_len, only needed to jump over a tier
                let mut doc = u32::MAX;
                for _ in 0..n {
                    doc = doc.wrapping_add(get_varint(p, &mut pos) as u32);
                    let tf = if t == OVERFLOW {
                        get_varint(p, &mut pos) as u32
                    } else {
                        t as u32
                    };
                    score(doc, tf);
                }
            }
        }

        let mut hits: Vec<(u32, f32)> = (0..acc.len() as u32)
            .zip(acc)
            .filter(|&(_, s)| s > 0.0)
            .collect();
        let by_rank = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if hits.len() > k {
            hits.select_nth_unstable_by(k, by_rank);
            hits.truncate(k);
        }
        hits.sort_unstable_by(by_rank);
        hits
    }
}
