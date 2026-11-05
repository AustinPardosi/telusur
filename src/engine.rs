//! The main seam: build an index directory from a corpus, open it, search it.

use crate::{
    docstore::{self, DocStore},
    text::index::{self as keyword, KeywordIndex},
};
use anyhow::{Context, Result, bail};
pub use keyword::Strategy;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
    time::Instant,
};

pub const MAX_K: usize = 1000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
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
    NoVectors,
}

pub struct Request {
    pub query: String,
    pub mode: Mode,
    pub k: usize,
    /// Adds `stats` to the response.
    pub debug: bool,
    /// Debug option: how keyword search scores.
    pub strategy: Strategy,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            query: String::new(),
            mode: Mode::default(),
            k: 10,
            debug: false,
            strategy: Strategy::default(),
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
#[derive(Debug, Serialize)]
pub struct Stats {
    pub postings_decoded: u64,
}

/// One result. `id` is the external id; doc ids never leave the engine.
#[derive(Debug, Serialize)]
pub struct Hit {
    pub id: String,
    pub score: f32,
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Serialize)]
pub struct Timings {
    pub keyword: f64,
    pub docs: f64,
    pub total: f64,
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
}

/// Builds an index in `out` from `corpus` (one JSON doc per line, line order = doc id).
/// Returns the number of docs.
pub fn build(corpus: &Path, out: &Path) -> Result<usize> {
    build_with(corpus, out, |_| {})
}

/// `build`, letting `inspect` look at the keyword postings before they are written.
pub fn build_with(
    corpus: &Path,
    out: &Path,
    inspect: impl FnOnce(&keyword::Builder),
) -> Result<usize> {
    fs::create_dir_all(out)?;
    let file = File::open(corpus).with_context(|| format!("open {}", corpus.display()))?;
    let mut keyword = keyword::Builder::default();
    let mut docs = docstore::Writer::create(out)?;
    let mut n = 0;
    for line in BufReader::new(file).lines() {
        let d: CorpusDoc = serde_json::from_str(&line?)
            .with_context(|| format!("{} line {}", corpus.display(), n + 1))?;
        // Title and text are indexed as one field, like Anserini's BEIR flat indexes.
        keyword.add(&format!("{}\n{}", d.title, d.text));
        docs.add([&d.id, &d.title, &d.url, docstore::snippet(&d.text)])?;
        n += 1;
    }
    inspect(&keyword);
    keyword.finish(out)?;
    docs.finish(out)?;
    Ok(n)
}

pub struct Index {
    keyword: KeywordIndex,
    docs: DocStore,
}

impl Index {
    pub fn open(dir: &Path) -> Result<Self> {
        Ok(Self {
            keyword: KeywordIndex::open(dir)?,
            docs: DocStore::open(dir)?,
        })
    }

    pub fn search(&self, req: &Request) -> Result<Response> {
        let start = Instant::now();
        if req.k > MAX_K {
            bail!("k must be at most {MAX_K}, got {}", req.k);
        }
        let fallback_reason = match req.mode {
            Mode::Neural => {
                bail!("neural search needs vectors, and this index was built without them")
            }
            Mode::Hybrid => Some(FallbackReason::NoVectors),
            Mode::Keyword => None,
        };

        let t = Instant::now();
        let (hits, postings_decoded) = self.keyword.search(&req.query, req.k, req.strategy);
        let keyword = ms(t);

        let t = Instant::now();
        let results = hits
            .into_iter()
            .map(|(doc, score)| {
                let [id, title, url, snippet] = self.docs.get(doc)?.map(String::from);
                Ok(Hit {
                    id,
                    score,
                    title,
                    url,
                    snippet,
                })
            })
            .collect::<Result<_>>()?;
        let docs = ms(t);

        Ok(Response {
            mode_used: Mode::Keyword,
            fallback_reason,
            results,
            timings_ms: Timings {
                keyword,
                docs,
                total: ms(start),
            },
            stats: req.debug.then_some(Stats { postings_decoded }),
        })
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
