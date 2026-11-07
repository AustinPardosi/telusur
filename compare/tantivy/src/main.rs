//! B9: Tantivy on the corpora and queries of telusur's B7. Docs go through telusur's own
//! analyzer (a custom Tantivy tokenizer), so both engines index exactly the same terms; what
//! differs is the index format and the query engine. One text field (title, newline, text) with
//! term frequencies and no positions, merged into one segment, plus the external id stored so a
//! search can return it, as telusur's does. A query is one SHOULD clause per distinct term,
//! scored with Tantivy's BM25 (k1 1.2 and b 0.75, which it does not let you change), top k by
//! TopDocs (block-max WAND).

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    thread,
    time::Instant,
};
use tantivy::{
    Index, IndexWriter, TantivyDocument, Term,
    collector::TopDocs,
    doc,
    indexer::NoMergePolicy,
    query::{BooleanQuery, Occur, Query, TermQuery},
    schema::{IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions, Value as _},
    tokenizer::{TextAnalyzer, Token, TokenStream, Tokenizer},
};
use telusur::{
    engine::{self, Mode, Request},
    eval::load_beir,
    text::analyzer::analyze,
    util::{hardware, percentile},
};

/// Telusur's analyzer as a Tantivy tokenizer.
#[derive(Clone)]
struct Telusur;

struct Stream {
    tokens: std::vec::IntoIter<String>,
    token: Token,
}

impl Tokenizer for Telusur {
    type TokenStream<'a> = Stream;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> Stream {
        Stream {
            tokens: analyze(text).into_iter(),
            token: Token::default(),
        }
    }
}

impl TokenStream for Stream {
    fn advance(&mut self) -> bool {
        let Some(text) = self.tokens.next() else {
            return false;
        };
        self.token.position = self.token.position.wrapping_add(1);
        self.token.text = text;
        true
    }
    fn token(&self) -> &Token {
        &self.token
    }
    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

const TOKENIZER: &str = "telusur";

fn register(index: &Index) {
    index
        .tokenizers()
        .register(TOKENIZER, TextAnalyzer::from(Telusur));
}

#[derive(Parser)]
#[command(about)]
enum Cmd {
    /// Index a corpus.jsonl (id or _id, title, text) into one Tantivy segment
    Index {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        threads: usize,
        /// Indexing memory budget, shared by the threads
        #[arg(long, default_value_t = 4000)]
        mem_mb: usize,
    },
    /// Latency and QPS on the test queries of a BEIR directory, measured like telusur's B7
    Bench {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, value_delimiter = ',', default_value = "1,4,10")]
        threads: Vec<usize>,
        /// Also report how many of each query's top k telusur returns too (a sanity check:
        /// the BM25 parameters differ, so not all)
        #[arg(long)]
        telusur_index: Option<PathBuf>,
    },
}

/// Bytes by file extension, and in total.
fn sizes(dir: &Path) -> Result<(BTreeMap<String, u64>, u64)> {
    let mut by_ext = BTreeMap::new();
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        let ext = p
            .extension()
            .map_or("", |e| e.to_str().unwrap_or(""))
            .to_string();
        *by_ext.entry(ext).or_insert(0) += fs::metadata(&p)?.len();
    }
    let total = by_ext.values().sum();
    Ok((by_ext, total))
}

fn index(corpus: &Path, out: &Path, threads: usize, mem_mb: usize) -> Result<Value> {
    let mut schema = Schema::builder();
    // Stored, not indexed: telusur keeps the external id in its doc store, not its postings.
    let id = schema.add_text_field("id", STORED);
    let indexing = TextFieldIndexing::default()
        .set_tokenizer(TOKENIZER)
        .set_index_option(IndexRecordOption::WithFreqs);
    let body = schema.add_text_field(
        "body",
        TextOptions::default().set_indexing_options(indexing),
    );
    fs::create_dir_all(out)?;
    let index = Index::create_in_dir(out, schema.build())?;
    register(&index);
    let mut writer: IndexWriter = index.writer_with_num_threads(threads, mem_mb << 20)?;
    // No background merges: one explicit merge into a single segment at the end.
    writer.set_merge_policy(Box::new(NoMergePolicy));

    let start = Instant::now();
    let mut n = 0u64;
    for line in
        BufReader::new(File::open(corpus).with_context(|| format!("open {}", corpus.display()))?)
            .lines()
    {
        let d: Value = serde_json::from_str(&line?)?;
        let ext = d
            .get("id")
            .or_else(|| d.get("_id"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let text = format!(
            "{}\n{}",
            d["title"].as_str().unwrap_or(""),
            d["text"].as_str().unwrap_or("")
        );
        writer.add_document(doc!(id => ext, body => text))?;
        n += 1;
    }
    writer.commit()?;
    let indexed = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let segments = index.searchable_segment_ids()?;
    if segments.len() > 1 {
        writer.merge(&segments).wait()?;
    }
    writer.wait_merging_threads()?;
    let merged = start.elapsed().as_secs_f64();
    let (files, total) = sizes(out)?;
    Ok(json!({
        "bench": "tantivy-index",
        "corpus": corpus.display().to_string(),
        "docs": n,
        "segments_before_merge": segments.len(),
        "index_seconds": indexed,
        "merge_seconds": merged,
        "bytes_by_extension": files,
        "bytes": total,
        "threads": threads,
        "mem_mb": mem_mb,
        "machine": hardware(),
    }))
}

fn bench(
    dir: &Path,
    data: &Path,
    k: usize,
    threads: &[usize],
    telusur: Option<&Path>,
) -> Result<Value> {
    const MIN_TIMED: usize = 1000;
    const QPS_PASSES: usize = 5;
    let index = Index::open_in_dir(dir)?;
    register(&index);
    let schema = index.schema();
    let (id, body) = (schema.get_field("id")?, schema.get_field("body")?);
    let searcher = index.reader()?.searcher();
    let (_, queries) = load_beir(data)?;
    let texts: Vec<&str> = queries.values().map(String::as_str).collect();

    let query = |q: &str| -> Box<dyn Query> {
        let mut terms = analyze(q);
        terms.sort_unstable();
        terms.dedup();
        let clauses = terms
            .iter()
            .map(|t| {
                let term = Term::from_field_text(body, t);
                let q: Box<dyn Query> =
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
                (Occur::Should, q)
            })
            .collect();
        Box::new(BooleanQuery::new(clauses))
    };
    let top = |q: &str| searcher.search(&*query(q), &TopDocs::with_limit(k).order_by_score());
    // A whole search also reads the external ids, as telusur's does.
    let whole = |q: &str| -> Result<Vec<String>> {
        top(q)?
            .into_iter()
            .map(|(_, addr)| {
                let d: TantivyDocument = searcher.doc(addr)?;
                Ok(d.get_first(id)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string())
            })
            .collect()
    };

    // Warm-up pass, also counting results and the overlap with telusur.
    let telusur = telusur.map(engine::Index::open).transpose()?;
    let (mut results, mut shared) = (0, 0);
    for q in &texts {
        let ids = whole(q)?;
        results += ids.len();
        if let Some(t) = &telusur {
            let req = Request {
                query: q.to_string(),
                mode: Mode::Keyword,
                k,
                ..Request::default()
            };
            shared += t
                .search(&req)?
                .results
                .iter()
                .filter(|h| ids.contains(&h.id))
                .count();
        }
    }
    let timed = texts.len() * MIN_TIMED.div_ceil(texts.len());
    let mut lat = Vec::with_capacity(timed);
    for q in texts.iter().cycle().take(timed) {
        let start = Instant::now();
        top(q)?;
        lat.push(start.elapsed().as_secs_f64() * 1e3);
    }
    lat.sort_by(f64::total_cmp);

    let mut qps = BTreeMap::new();
    for &n in threads {
        let total = timed * QPS_PASSES;
        let start = Instant::now();
        thread::scope(|s| -> Result<()> {
            let workers: Vec<_> = (0..n)
                .map(|t| {
                    let (texts, whole) = (&texts, &whole);
                    s.spawn(move || -> Result<()> {
                        for i in (t..total).step_by(n) {
                            whole(texts[i % texts.len()])?;
                        }
                        Ok(())
                    })
                })
                .collect();
            workers
                .into_iter()
                .try_for_each(|w| w.join().expect("bench thread panicked"))
        })?;
        qps.insert(n, total as f64 / start.elapsed().as_secs_f64());
    }
    let (files, bytes) = sizes(dir)?;
    Ok(json!({
        "bench": "tantivy-bm25",
        "dataset": data.file_name().unwrap_or_default().to_string_lossy(),
        "docs": searcher.num_docs(),
        "segments": searcher.segment_readers().len(),
        "k": k,
        "queries": texts.len(),
        "timed_queries": timed,
        "results_per_query": results as f64 / texts.len() as f64,
        "top_k_shared_with_telusur": telusur.map(|_| shared as f64 / (k * texts.len()) as f64),
        "p50_ms": percentile(&lat, 50.0),
        "p99_ms": percentile(&lat, 99.0),
        "mean_ms": lat.iter().sum::<f64>() / lat.len() as f64,
        "qps": qps,
        "bytes_by_extension": files,
        "bytes": bytes,
        "machine": hardware(),
    }))
}

fn main() -> Result<()> {
    let report = match Cmd::parse() {
        Cmd::Index {
            corpus,
            out,
            threads,
            mem_mb,
        } => index(&corpus, &out, threads, mem_mb)?,
        Cmd::Bench {
            index,
            data,
            k,
            threads,
            telusur_index,
        } => bench(&index, &data, k, &threads, telusur_index.as_deref())?,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
