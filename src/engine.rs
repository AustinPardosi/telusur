//! The main seam: build an index directory from a corpus, open it, search it.

use crate::{
    docstore::{self, DocStore},
    filter::{self, FilterIndex},
    text::index::{self as keyword, KeywordIndex},
    util::Fbin,
    vector::{
        ivf::{self, VectorIndex},
        kernel::Kernel,
    },
};
use anyhow::{Context, Result, bail};
pub use keyword::Strategy;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

pub const MAX_K: usize = 1000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Keyword,
    Neural,
    #[default]
    Hybrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    /// The index was built without vectors.
    NoVectors,
    /// There is no query vector and nothing to embed the query text with.
    NoEmbedder,
    /// The embedder failed or sent back something unusable.
    EmbedderError,
    /// The embedder did not answer in time.
    EmbedderTimeout,
}

pub struct Request {
    pub query: String,
    pub mode: Mode,
    pub k: usize,
    /// Adds `stats` to the response.
    pub debug: bool,
    /// Debug option: how keyword search scores.
    pub strategy: Strategy,
    /// A full query vector, for neural search without an embedder.
    pub vector: Option<Vec<f32>>,
    /// How many of the best codes neural search rescores exactly (at least k). The default,
    /// 500, is what reached recall@10 of 0.95 with the default clusters probed in B3.
    pub rerank_depth: usize,
    /// Debug option: the scoring kernel, instead of the process default.
    pub kernel: Option<Kernel>,
    /// Clusters neural search probes [default: 256, from B3; at most all of them].
    pub nprobe: Option<usize>,
    /// Every entry must hold; both retrievers apply them.
    pub filters: filter::Filters,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            query: String::new(),
            mode: Mode::default(),
            k: 10,
            debug: false,
            strategy: Strategy::default(),
            vector: None,
            rerank_depth: 500,
            kernel: None,
            nprobe: None,
            filters: filter::Filters::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub mode_used: Mode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<FallbackReason>,
    pub results: Vec<Hit>,
    pub timings_ms: Timings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<Stats>,
}

/// How much work a query did. Only with `debug`.
#[derive(Debug, Default, Serialize)]
pub struct Stats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postings_decoded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<Kernel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clusters_probed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codes_scanned: Option<usize>,
    /// Docs that pass the filters, when there are any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs_allowed: Option<u64>,
}

/// One result. `id` is the external id; doc ids never leave the engine.
#[derive(Debug, Serialize)]
pub struct Hit {
    pub id: String,
    pub score: f32,
    pub title: String,
    pub url: String,
    pub snippet: String,
    /// Hybrid only: where each retriever ranked the doc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranks: Option<Ranks>,
}

/// 1-based ranks in each retriever's list; absent when the doc was not in that list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Ranks {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neural: Option<usize>,
}

/// RRF's k: a doc at rank r adds 1 / (60 + r).
pub const RRF_K: f64 = 60.0;
/// Hybrid takes the top max(k, this) of each retriever.
pub const FUSE_DEPTH: usize = 100;

/// Reciprocal rank fusion of two best-first lists: score = sum of 1 / (60 + rank). Top k,
/// best first, ties to the smaller doc id.
pub fn rrf(keyword: &[(u32, f32)], neural: &[(u32, f32)], k: usize) -> Vec<(u32, f64, Ranks)> {
    let mut ranks: HashMap<u32, Ranks> = HashMap::new();
    for (i, &(doc, _)) in keyword.iter().enumerate() {
        ranks.entry(doc).or_default().keyword = Some(i + 1);
    }
    for (i, &(doc, _)) in neural.iter().enumerate() {
        ranks.entry(doc).or_default().neural = Some(i + 1);
    }
    let term = |r: Option<usize>| r.map_or(0.0, |r| 1.0 / (RRF_K + r as f64));
    let mut fused: Vec<_> = ranks
        .into_iter()
        .map(|(doc, r)| (doc, term(r.keyword) + term(r.neural), r))
        .collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused.truncate(k);
    fused
}

/// Milliseconds per stage. Only the stages that ran appear.
#[derive(Debug, Default, Serialize)]
pub struct Timings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuse: Option<f64>,
    pub docs: f64,
    pub total: f64,
}

/// What to build besides the keyword index.
#[derive(Clone, Debug)]
pub struct BuildOptions {
    /// Full vectors, one `.fbin` row per doc. Without them the index is keyword only.
    pub vectors: Option<PathBuf>,
    /// Matryoshka prefix kept as the 1-bit code.
    pub d_code: usize,
    /// Prefix kept as the f16 rerank vector; defaults to the smaller of 1024 and the dims.
    pub d_rerank: Option<usize>,
    /// Clusters; defaults to about 2 x sqrt(docs), rounded to a power of two.
    pub nlist: Option<usize>,
    /// Corpus fields to filter on by value (a string or a list of strings per doc).
    pub keyword_fields: Vec<String>,
    /// The corpus field holding the crawl date (`YYYY-MM-DD` or a UTC timestamp).
    pub date_field: Option<String>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            vectors: None,
            d_code: 256,
            d_rerank: None,
            nlist: None,
            keyword_fields: Vec::new(),
            date_field: None,
        }
    }
}

/// One line of `corpus.jsonl`. Other fields are ignored for now.
#[derive(Deserialize)]
struct CorpusDoc {
    #[serde(alias = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
    #[serde(default)]
    url: String,
    /// Everything else, where filter fields come from.
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

/// Builds an index in `out` from `corpus` (one JSON doc per line, line order = doc id, which
/// is also the row in the vectors file). Returns the number of docs.
pub fn build(corpus: &Path, out: &Path, opts: &BuildOptions) -> Result<usize> {
    build_with(corpus, out, opts, |_| {})
}

/// `build`, letting `inspect` look at the keyword postings before they are written.
pub fn build_with(
    corpus: &Path,
    out: &Path,
    opts: &BuildOptions,
    inspect: impl FnOnce(&keyword::Builder),
) -> Result<usize> {
    let vectors = opts.vectors.as_deref().map(Fbin::open).transpose()?;
    fs::create_dir_all(out)?;
    let file = File::open(corpus).with_context(|| format!("open {}", corpus.display()))?;
    let mut keyword = keyword::Builder::default();
    let mut docs = docstore::Writer::create(out)?;
    let mut filters = (!opts.keyword_fields.is_empty() || opts.date_field.is_some())
        .then(|| filter::Builder::new(&opts.keyword_fields, opts.date_field.as_deref()))
        .transpose()?;
    let mut n = 0;
    for line in BufReader::new(file).lines() {
        let at = || format!("{} line {}", corpus.display(), n + 1);
        let d: CorpusDoc = serde_json::from_str(&line?).with_context(at)?;
        // Title and text are indexed as one field, like Anserini's BEIR flat indexes.
        keyword.add(&format!("{}\n{}", d.title, d.text));
        docs.add([&d.id, &d.title, &d.url, docstore::snippet(&d.text)])?;
        if let Some(f) = &mut filters {
            f.add(n as u32, &d.rest).with_context(at)?;
        }
        n += 1;
    }
    if let Some(f) = filters {
        f.finish(out)?;
    }
    if let Some(v) = &vectors {
        // Every doc has a vector, or none do.
        if v.n != n {
            bail!(
                "the vectors file has {} rows but the corpus has {n} docs",
                v.n
            );
        }
        ivf::build(v, out, opts.d_code, opts.d_rerank, opts.nlist)?;
    }
    inspect(&keyword);
    keyword.finish(out)?;
    docs.finish(out)?;
    Ok(n)
}

pub struct Index {
    keyword: KeywordIndex,
    docs: DocStore,
    vectors: Option<VectorIndex>,
    filters: FilterIndex,
}

impl Index {
    pub fn open(dir: &Path) -> Result<Self> {
        Ok(Self {
            keyword: KeywordIndex::open(dir)?,
            docs: DocStore::open(dir)?,
            vectors: VectorIndex::open(dir)?,
            filters: FilterIndex::open(dir)?,
        })
    }

    pub fn docs(&self) -> usize {
        self.docs.count()
    }

    /// The external id of `doc`.
    pub fn external_id(&self, doc: u32) -> Result<&str> {
        Ok(self.docs.get(doc)?[0])
    }

    /// Dims of the full vectors the index was built from, if it has vectors.
    pub fn vector_dims(&self) -> Option<usize> {
        self.vectors.as_ref().map(VectorIndex::dims)
    }

    pub fn search(&self, req: &Request) -> Result<Response> {
        let start = Instant::now();
        if req.k > MAX_K {
            bail!("k must be at most {MAX_K}, got {}", req.k);
        }
        let (neural, fallback_reason) = match (req.mode, &self.vectors, req.vector.as_deref()) {
            (Mode::Keyword, ..) => (None, None),
            (Mode::Neural | Mode::Hybrid, Some(index), Some(q)) => (Some((index, q)), None),
            (Mode::Neural, None, _) => {
                bail!("neural search needs vectors, and this index was built without them")
            }
            (Mode::Neural, Some(_), None) => {
                bail!("neural search needs a query vector, and no embedder is configured")
            }
            (Mode::Hybrid, None, _) => (None, Some(FallbackReason::NoVectors)),
            (Mode::Hybrid, Some(_), None) => (None, Some(FallbackReason::NoEmbedder)),
        };
        let mode_used = match (req.mode, neural) {
            (Mode::Neural, _) => Mode::Neural,
            (Mode::Hybrid, Some(_)) => Mode::Hybrid,
            _ => Mode::Keyword,
        };
        let depth = if mode_used == Mode::Hybrid {
            req.k.max(FUSE_DEPTH)
        } else {
            req.k
        };

        let mut timings = Timings::default();
        let mut stats = Stats::default();
        let allowed = self.filters.allowed(&req.filters)?;
        stats.docs_allowed = allowed.as_ref().map(|a| a.len());
        let allowed = allowed.as_ref();
        let run_neural = || -> Result<_> {
            let Some((index, q)) = neural else {
                return Ok(None);
            };
            let kernel = req.kernel.map_or_else(Kernel::default_for_process, Ok)?;
            let nprobe = req.nprobe.unwrap_or(ivf::DEFAULT_NPROBE);
            let (hits, trace) =
                index.search(q, depth, req.rerank_depth, nprobe, kernel, allowed)?;
            Ok(Some((hits, trace, kernel)))
        };
        let run_keyword = || {
            (mode_used != Mode::Neural).then(|| {
                let t = Instant::now();
                let (hits, decoded) = self
                    .keyword
                    .search(&req.query, depth, req.strategy, allowed);
                (hits, decoded, ms(t))
            })
        };
        // Hybrid runs both retrievers in parallel; a single retriever runs on this thread.
        let (neural_hits, keyword_hits) = if mode_used == Mode::Hybrid {
            rayon::join(run_neural, run_keyword)
        } else {
            (run_neural(), run_keyword())
        };
        let neural_hits = neural_hits?.map(|(hits, t, kernel)| {
            stats.kernel = Some(kernel);
            stats.clusters_probed = Some(t.clusters_probed);
            stats.codes_scanned = Some(t.codes_scanned);
            (timings.route, timings.scan, timings.rerank) =
                (Some(t.route), Some(t.scan), Some(t.rerank));
            hits
        });
        let keyword_hits = keyword_hits.map(|(hits, decoded, t)| {
            stats.postings_decoded = Some(decoded);
            timings.keyword = Some(t);
            hits
        });
        let hits: Vec<(u32, f32, Option<Ranks>)> = match (keyword_hits, neural_hits) {
            (Some(kw), Some(nn)) => {
                let t = Instant::now();
                let fused = rrf(&kw, &nn, req.k);
                timings.fuse = Some(ms(t));
                fused
                    .into_iter()
                    .map(|(doc, score, r)| (doc, score as f32, Some(r)))
                    .collect()
            }
            (Some(hits), None) | (None, Some(hits)) => {
                hits.into_iter().map(|(d, s)| (d, s, None)).collect()
            }
            (None, None) => unreachable!("one retriever always runs"),
        };

        let t = Instant::now();
        let results = hits
            .into_iter()
            .map(|(doc, score, ranks)| {
                let [id, title, url, snippet] = self.docs.get(doc)?.map(String::from);
                Ok(Hit {
                    id,
                    score,
                    title,
                    url,
                    snippet,
                    ranks,
                })
            })
            .collect::<Result<_>>()?;
        timings.docs = ms(t);
        timings.total = ms(start);

        Ok(Response {
            mode_used,
            fallback_reason,
            results,
            timings_ms: timings,
            stats: req.debug.then_some(stats),
        })
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
