//! Keyword index: tiered posting lists, an FST dictionary, one-byte doc length norms, and
//! BM25 scored the way Lucene does it, either exhaustively or with MaxScore over tiers.
//!
//! Files:
//! - `terms.fst`: term to value. A singleton (df 1) packs `SINGLETON | tf << 32 | doc`; any
//!   other term gives the byte offset of its postings in `postings.bin`.
//! - `postings.bin`: per term `varint df`, `u16 tier bitset` (bit t = the tier with tf == t,
//!   t in 1..=15; bit 0 = the overflow tier, tf >= 16), then each present tier in the order
//!   1..=15, overflow: `varint n_docs, varint byte_len, u8 min_norm`, `varint max_tf` (overflow
//!   tier only), `varint skip_start` (tiers of more than 128 docs only), then n_docs x
//!   `varint doc_delta` (overflow tier: plus `varint tf`). The first doc_delta of a tier is
//!   doc + 1, later ones are gaps, so every varint is at least 1.
//! - `skips.bin`: skip entries, `u32 prev_doc, u32 byte_offset` (little endian). A tier of n
//!   docs owns (n - 1) / 128 of them from `skip_start`; entry i points at posting 128 (i + 1),
//!   its offset counted from the start of the tier's data, `prev_doc` being the doc before it.
//! - `norms.bin`: one SmallFloat byte per doc.
//! - `keyword.json`: collection stats for idf and avgdl.

use super::{
    analyzer::analyze,
    codec::{byte4_to_int, get_varint, int_to_byte4, put_varint},
};
use crate::util::{TopK, mmap};
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
/// Postings per skip interval.
const SKIP: u32 = 128;
const END: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    /// Score every posting of every query term.
    Exhaustive,
    /// MaxScore over (term, tier) cursors: same top-k scores, far fewer postings decoded.
    #[default]
    Maxscore,
}

#[derive(Default, Serialize, Deserialize)]
struct Stats {
    /// Docs with at least one token, Lucene's `docCount`.
    doc_count: u64,
    /// Tokens over all docs, Lucene's `sumTotalTermFreq`.
    total_len: u64,
}

/// Bytes one posting format takes for a whole corpus, by part.
#[derive(Debug, Serialize)]
pub struct Footprint {
    pub format: &'static str,
    pub postings: u64,
    /// Per-term entries: a hash table, or nothing once the FST holds the offsets.
    pub term_table: u64,
    /// The term text: heap strings, or the FST.
    pub term_strings: u64,
    pub doc_lengths: u64,
}

impl Footprint {
    pub fn total(&self) -> u64 {
        self.postings + self.term_table + self.term_strings + self.doc_lengths
    }
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

    /// Distinct terms held so far.
    pub fn terms(&self) -> usize {
        self.terms.len()
    }

    /// Postings held so far.
    pub fn postings(&self) -> u64 {
        self.terms.values().map(|l| l.len() as u64).sum()
    }

    /// The first four steps of the memory ablation, measured on the postings this builder
    /// holds. Hash table bytes come from the builder's own map (hashbrown keeps capacity at
    /// 7/8 of its buckets) plus one control byte per bucket; Vec and String bytes are their
    /// real capacities; every encoded format is really encoded, term by term. The last step,
    /// the shipped format, is the size of the files `finish` writes.
    ///
    /// Step 3 keeps a u32 tf per posting, so each step changes one thing. Also returns the
    /// postings bytes of step 3 with a varint tf instead, to show what tiers save over that.
    pub fn ablation(&self) -> (Vec<Footprint>, u64) {
        let buckets = (self.terms.capacity() / 7 * 8) as u64;
        let table = |entry: usize| buckets * (entry as u64 + 1) + 16;
        let term_strings: u64 = self.terms.keys().map(|t| t.capacity() as u64).sum();
        let doc_lengths = 4 * self.norms.len() as u64;
        let step = |format, postings, term_table| Footprint {
            format,
            postings,
            term_table,
            term_strings,
            doc_lengths,
        };

        let (mut naive, mut flat, mut delta, mut delta_varint_tf, mut tiered) = (0, 0, 0, 0, 0);
        let (mut buf, mut skips, mut enc) = (Vec::new(), Vec::new(), TierEncoder::default());
        for list in self.terms.values() {
            naive += 8 * list.capacity() as u64;
            flat += 8 * list.len() as u64;
            buf.clear();
            let mut prev = u32::MAX;
            for &(doc, _) in list {
                put_varint(&mut buf, u64::from(doc.wrapping_sub(prev)));
                prev = doc;
            }
            delta += (buf.len() + 4 * list.len()) as u64; // plus a u32 tf per posting
            for &(_, tf) in list {
                put_varint(&mut buf, u64::from(tf));
            }
            delta_varint_tf += buf.len() as u64;
            buf.clear();
            skips.clear();
            enc.encode(list, &self.norms, &mut buf, &mut skips);
            tiered += (buf.len() + skips.len()) as u64;
        }
        let flat_table = table(size_of::<(String, (u64, u32))>());
        let steps = vec![
            step(
                "HashMap<String, Vec<(u32, u32)>>",
                naive,
                table(size_of::<(String, Vec<(u32, u32)>)>()),
            ),
            step(
                "+ one flat buffer, (offset, len) per term",
                flat,
                flat_table,
            ),
            step("+ delta varint doc ids", delta, flat_table),
            step("+ tf tiers (with headers and skips)", tiered, flat_table),
        ];
        (steps, delta_varint_tf)
    }

    /// Writes the index files into `dir` and returns the number of terms.
    pub fn finish(self, dir: &Path) -> Result<usize> {
        let mut terms: Vec<_> = self.terms.into_iter().collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut fst = fst::MapBuilder::new(BufWriter::new(File::create(dir.join("terms.fst"))?))?;
        let mut postings = Vec::new();
        let mut skips = Vec::new();
        let mut enc = TierEncoder::default();
        for (term, list) in &terms {
            if let [(doc, tf)] = list[..] {
                fst.insert(term, SINGLETON | u64::from(tf) << 32 | u64::from(doc))?;
                continue;
            }
            fst.insert(term, postings.len() as u64)?;
            enc.encode(list, &self.norms, &mut postings, &mut skips);
        }
        fst.finish()?;
        fs::write(dir.join("postings.bin"), postings)?;
        fs::write(dir.join("skips.bin"), skips)?;
        fs::write(dir.join("norms.bin"), &self.norms)?;
        fs::write(dir.join("keyword.json"), serde_json::to_vec(&self.stats)?)?;
        Ok(terms.len())
    }
}

/// Writes one term's posting list in the tiered format: `varint df`, the tier bitset, then
/// each tier's header and data. Keeps its per-tier buffers between terms.
#[derive(Default)]
struct TierEncoder {
    data: [Vec<u8>; 16],
    skips: [Vec<u8>; 16],
}

impl TierEncoder {
    fn encode(
        &mut self,
        list: &[(u32, u32)],
        norms: &[u8],
        postings: &mut Vec<u8>,
        skips: &mut Vec<u8>,
    ) {
        let (data, tier_skips) = (&mut self.data, &mut self.skips);
        let mut prev = [u32::MAX; 16];
        let mut count = [0u32; 16];
        let mut min_norm = [u8::MAX; 16];
        let mut max_tf = 0;
        for &(doc, tf) in list {
            let t = if tf <= 15 { tf as usize } else { OVERFLOW };
            if count[t] > 0 && count[t] % SKIP == 0 {
                tier_skips[t].extend(prev[t].to_le_bytes());
                tier_skips[t].extend((data[t].len() as u32).to_le_bytes());
            }
            put_varint(&mut data[t], u64::from(doc.wrapping_sub(prev[t])));
            if t == OVERFLOW {
                put_varint(&mut data[t], u64::from(tf));
                max_tf = max_tf.max(tf);
            }
            prev[t] = doc;
            count[t] += 1;
            min_norm[t] = min_norm[t].min(norms[doc as usize]);
        }
        put_varint(postings, list.len() as u64);
        let bitset = TIERS
            .iter()
            .filter(|&&t| count[t] > 0)
            .fold(0u16, |m, &t| m | 1 << t);
        postings.extend(bitset.to_le_bytes());
        for t in TIERS {
            if count[t] == 0 {
                continue;
            }
            put_varint(postings, u64::from(count[t]));
            put_varint(postings, data[t].len() as u64);
            postings.push(min_norm[t]);
            if t == OVERFLOW {
                put_varint(postings, u64::from(max_tf));
            }
            if count[t] > SKIP {
                put_varint(postings, skips.len() as u64 / 8);
                skips.append(&mut tier_skips[t]);
            }
            postings.append(&mut data[t]);
        }
    }
}

pub struct KeywordIndex {
    terms: fst::Map<Mmap>,
    postings: Mmap,
    skips: Mmap,
    norms: Mmap,
    doc_count: f64,
    /// Lucene's per-norm cache: `1 / (k1 * (1 - b + b * dl / avgdl))` for each norm byte.
    norm_inverse: [f32; 256],
}

/// Lucene's BM25 for one term in one doc, with the `(k1 + 1)` factor dropped. Rises with tf
/// and falls with `norm`, so a tier's best possible score is its smallest norm.
fn bm25(w: f32, tf: u32, norm_inverse: f32) -> f32 {
    w - w / (1.0 + tf as f32 * norm_inverse)
}

/// One run of postings that share a weight: a tier of one query term, or a singleton.
struct Cursor<'a> {
    buf: &'a [u8],
    /// Byte position of the next posting.
    pos: usize,
    /// Where the tier's data starts; skip offsets count from here.
    start: usize,
    skips: &'a [u8],
    next_skip: usize,
    /// Index of the next posting to decode, out of `n`.
    idx: u32,
    n: u32,
    /// The current posting, or `END` once the run is used up.
    doc: u32,
    tf: u32,
    overflow: bool,
    /// Which query term this run belongs to.
    term: usize,
    w: f32,
    /// Highest score any posting in this run can get.
    bound: f32,
}

impl Cursor<'_> {
    /// Decodes the next posting.
    fn next(&mut self, decoded: &mut u64) {
        if self.idx == self.n {
            self.doc = END;
            return;
        }
        self.doc = self
            .doc
            .wrapping_add(get_varint(self.buf, &mut self.pos) as u32);
        if self.overflow {
            self.tf = get_varint(self.buf, &mut self.pos) as u32;
        }
        self.idx += 1;
        *decoded += 1;
    }

    /// Moves to the first posting at or after `target`, jumping over whole skip intervals
    /// whose docs are all below it.
    fn advance(&mut self, target: u32, decoded: &mut u64) {
        if self.doc >= target {
            return;
        }
        let mut jump = None;
        while let Some(e) = self.skips.get(self.next_skip * 8..self.next_skip * 8 + 8) {
            let prev = u32::from_le_bytes(e[..4].try_into().unwrap());
            if prev >= target {
                break;
            }
            jump = Some((
                self.next_skip as u32,
                prev,
                u32::from_le_bytes(e[4..].try_into().unwrap()),
            ));
            self.next_skip += 1;
        }
        if let Some((i, prev, offset)) = jump {
            let at = (i + 1) * SKIP;
            if at >= self.idx {
                self.idx = at;
                self.pos = self.start + offset as usize;
                self.doc = prev;
            }
        }
        while self.doc < target {
            self.next(decoded);
        }
    }
}

struct QueryTerm {
    value: u64,
    w: f32,
}

impl KeywordIndex {
    pub fn open(dir: &Path) -> Result<Self> {
        let stats: Stats = serde_json::from_slice(&fs::read(dir.join("keyword.json"))?)?;
        let avgdl = (stats.total_len as f64 / stats.doc_count as f64) as f32;
        Ok(Self {
            terms: fst::Map::new(mmap(&dir.join("terms.fst"))?)?,
            postings: mmap(&dir.join("postings.bin"))?,
            skips: mmap(&dir.join("skips.bin"))?,
            norms: mmap(&dir.join("norms.bin"))?,
            doc_count: stats.doc_count as f64,
            norm_inverse: std::array::from_fn(|i| {
                1.0 / (K1 * ((1.0 - B) + B * byte4_to_int(i as u8) as f32 / avgdl))
            }),
        })
    }

    /// Top k (doc, score) by BM25, best first, ties to the smaller doc id, plus the number
    /// of postings decoded. The strategies add a doc's per-term scores in different orders,
    /// so their top-k scores agree within float rounding, not bit for bit.
    pub fn search(&self, query: &str, k: usize, strategy: Strategy) -> (Vec<(u32, f32)>, u64) {
        let terms = self.query_terms(query);
        let mut cursors = Vec::new();
        let mut decoded = 0;
        for (i, t) in terms.iter().enumerate() {
            self.cursors(i, t, &mut cursors, &mut decoded);
        }
        let mut top = TopK::new(k);
        match strategy {
            Strategy::Exhaustive => {
                let mut acc = vec![0f32; self.norms.len()];
                for c in &mut cursors {
                    while c.doc != END {
                        acc[c.doc as usize] += self.score(c);
                        c.next(&mut decoded);
                    }
                }
                for (doc, s) in (0..).zip(acc) {
                    if s > 0.0 {
                        top.push(doc, s);
                    }
                }
            }
            Strategy::Maxscore => self.maxscore(&mut cursors, terms.len(), &mut top, &mut decoded),
        }
        (top.into_sorted(), decoded)
    }

    /// Like Anserini's bag-of-words query: one clause per distinct term, boosted by its
    /// count, in term order so sums are deterministic.
    fn query_terms(&self, query: &str) -> Vec<QueryTerm> {
        let mut boosts = BTreeMap::<String, u32>::new();
        for t in analyze(query) {
            *boosts.entry(t).or_default() += 1;
        }
        boosts
            .into_iter()
            .filter_map(|(term, boost)| {
                let value = self.terms.get(&term)?;
                let df = if value & SINGLETON != 0 {
                    1
                } else {
                    get_varint(&self.postings, &mut (value as usize))
                };
                let idf =
                    (1.0 + (self.doc_count - df as f64 + 0.5) / (df as f64 + 0.5)).ln() as f32;
                Some(QueryTerm {
                    value,
                    w: boost as f32 * idf,
                })
            })
            .collect()
    }

    /// Opens one cursor per tier of query term `term`, each on its first posting.
    fn cursors<'a>(
        &'a self,
        term: usize,
        t: &QueryTerm,
        out: &mut Vec<Cursor<'a>>,
        decoded: &mut u64,
    ) {
        let p = &self.postings[..];
        let mut c = Cursor {
            buf: p,
            pos: 0,
            start: 0,
            skips: &[],
            next_skip: 0,
            idx: 0,
            n: 0,
            doc: u32::MAX,
            tf: 0,
            overflow: false,
            term,
            w: t.w,
            bound: 0.0,
        };
        if t.value & SINGLETON != 0 {
            c.doc = t.value as u32;
            c.tf = (t.value >> 32) as u32 & !(1 << 31);
            (c.idx, c.n) = (1, 1);
            c.bound = self.score(&c);
            *decoded += 1;
            out.push(c);
            return;
        }
        let mut pos = t.value as usize;
        get_varint(p, &mut pos); // df, already folded into `t.w`
        let bitset = u16::from_le_bytes([p[pos], p[pos + 1]]);
        pos += 2;
        for tier in TIERS.into_iter().filter(|&tier| bitset & 1 << tier != 0) {
            let n = get_varint(p, &mut pos) as u32;
            let byte_len = get_varint(p, &mut pos) as usize;
            let min_norm = p[pos];
            pos += 1;
            let overflow = tier == OVERFLOW;
            let max_tf = if overflow {
                get_varint(p, &mut pos) as u32
            } else {
                tier as u32
            };
            let skips = if n > SKIP {
                let from = get_varint(p, &mut pos) as usize * 8;
                &self.skips[from..from + (n as usize - 1) / SKIP as usize * 8]
            } else {
                &[]
            };
            let mut tc = Cursor {
                pos,
                start: pos,
                skips,
                n,
                tf: max_tf,
                overflow,
                bound: bm25(t.w, max_tf, self.norm_inverse[min_norm as usize]),
                ..c
            };
            tc.next(decoded);
            out.push(tc);
            pos += byte_len;
        }
    }

    fn score(&self, c: &Cursor) -> f32 {
        bm25(
            c.w,
            c.tf,
            self.norm_inverse[self.norms[c.doc as usize] as usize],
        )
    }

    /// MaxScore in windows of doc ids, like Lucene's `MaxScoreBulkScorer`. Cursors are sorted
    /// by bound; the lowest ones whose combined bound cannot beat the current k-th score are
    /// non-essential. Per window, the essential cursors score everything they hold into a
    /// small array, then each doc they found, in doc order, probes the non-essential cursors
    /// (with skips) only while it could still make the top k. Docs found by non-essential
    /// cursors alone are never scored.
    fn maxscore(&self, cursors: &mut [Cursor], n_terms: usize, top: &mut TopK, decoded: &mut u64) {
        const WINDOW: usize = 4096;
        cursors.sort_by(|a, b| a.bound.total_cmp(&b.bound));
        // prefix[i]: the most cursors 0..=i can add to one doc. A doc sits in one tier per
        // term, so each term counts once, with its highest bound among them.
        let mut best = vec![0f32; n_terms];
        let prefix: Vec<f32> = cursors
            .iter()
            .map(|c| {
                best[c.term] = best[c.term].max(c.bound);
                best.iter().sum()
            })
            .collect();
        // Which query terms a doc in the window has matched, so other tiers of the same term
        // are not probed for it. Only kept for queries of up to 64 terms.
        let track = n_terms <= 64;
        let mut scores = vec![0f32; WINDOW];
        let mut hits = vec![0u64; WINDOW];
        let mut seen = [0u64; WINDOW / 64];
        let mut first_essential = 0;
        loop {
            let theta = top.threshold();
            while first_essential < cursors.len() && prefix[first_essential] <= theta {
                first_essential += 1;
            }
            let (lower, essential) = cursors.split_at_mut(first_essential);
            let Some(lo) = essential.iter().map(|c| c.doc).min().filter(|&d| d != END) else {
                break;
            };
            let hi = lo.saturating_add(WINDOW as u32);
            for c in essential.iter_mut() {
                while c.doc < hi {
                    let i = (c.doc - lo) as usize;
                    scores[i] += self.score(c);
                    hits[i] |= 1 << (c.term % 64);
                    seen[i / 64] |= 1 << (i % 64);
                    c.next(decoded);
                }
            }
            // Doc order matters: a later doc has a larger id, so it must beat theta outright.
            for (w, word) in seen.iter_mut().enumerate() {
                let mut bits = std::mem::take(word);
                while bits != 0 {
                    let i = w * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let doc = lo + i as u32;
                    let mut score = std::mem::take(&mut scores[i]);
                    let mut hit = std::mem::take(&mut hits[i]);
                    let theta = top.threshold();
                    let mut possible = true;
                    for (j, c) in lower.iter_mut().enumerate().rev() {
                        if score + prefix[j] <= theta {
                            possible = false;
                            break;
                        }
                        if track && hit & 1 << c.term != 0 {
                            continue;
                        }
                        c.advance(doc, decoded);
                        if c.doc == doc {
                            score += self.score(c);
                            hit |= 1 << (c.term % 64);
                        }
                    }
                    if possible && score > theta {
                        top.push(doc, score);
                    }
                }
            }
        }
    }
}
