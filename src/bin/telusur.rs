use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{path::PathBuf, time::Instant};
use telusur::{
    bench,
    engine::{self, BuildOptions, Index, Mode, Request, Strategy},
    eval,
    filter::{Filter, Filters},
    ingest, server, synth,
    util::{Fbin, read_ibin, write_ibin},
    vector::{kernel::Kernel, kmeans},
};

#[derive(Parser)]
#[command(version, about, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write synthetic data for tests and quick speed checks
    #[command(subcommand)]
    Synth(Synth),
    /// Build an index from a corpus
    Index {
        /// One JSON doc per line: id (or _id), title, text, url
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Full vectors (.fbin), one row per doc in corpus order; enables neural search
        #[arg(long)]
        vectors: Option<PathBuf>,
        /// Dims kept as the 1-bit code: a multiple of 64, at most 1024
        #[arg(long, default_value_t = 256)]
        d_code: usize,
        /// Dims kept as the f16 rerank vector [default: the smaller of 1024 and the dims]
        #[arg(long)]
        d_rerank: Option<usize>,
        /// Clusters [default: about 2 x sqrt(docs), rounded to a power of two]
        #[arg(long)]
        nlist: Option<usize>,
        /// Corpus fields to filter on by value, e.g. domain
        #[arg(long, value_delimiter = ',')]
        keyword_fields: Vec<String>,
        /// Corpus field holding the crawl date (YYYY-MM-DD or a UTC timestamp)
        #[arg(long)]
        date_field: Option<String>,
    },
    /// Exact ground truth: the top k base rows per query by full vectors (gt_full.ibin) and
    /// by rerank vectors (gt_rerank.ibin)
    Gt {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long, default_value_t = 100)]
        k: usize,
        /// Rerank dims, as given to `index` [default: the smaller of 1024 and the dims]
        #[arg(long)]
        d_rerank: Option<usize>,
        /// Directory to write the two .ibin files to
        #[arg(long)]
        out: PathBuf,
    },
    /// Search an index and print the response as JSON
    Search {
        #[arg(long)]
        index: PathBuf,
        /// Query text (keyword and hybrid modes)
        #[arg(default_value = "")]
        query: String,
        /// Take the query vector from this .fbin file (neural and hybrid modes)
        #[arg(long)]
        query_vectors: Option<PathBuf>,
        /// Row of --query-vectors to use
        #[arg(long, default_value_t = 0)]
        query_row: usize,
        /// How many of the best codes neural search rescores exactly (0: no rerank)
        #[arg(long, default_value_t = 500)]
        rerank_depth: usize,
        /// Clusters neural search probes [default: 256, from B3]
        #[arg(long)]
        nprobe: Option<usize>,
        /// Debug option: scoring kernel [default: TELUSUR_KERNEL, else the fastest]
        #[arg(long, value_enum)]
        kernel: Option<Kernel>,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, value_enum, default_value_t)]
        mode: Mode,
        /// Add stats (postings decoded) to the response
        #[arg(long)]
        debug: bool,
        /// Debug option: how keyword search scores
        #[arg(long, value_enum, default_value_t)]
        strategy: Strategy,
        /// Repeatable: field=a,b (any of the values) or crawl_date=2023-01-01..2025-12-31
        /// (inclusive, either end may be left open)
        #[arg(long = "filter", value_parser = parse_filter)]
        filters: Vec<(String, Filter)>,
    },
    /// Serve an index over HTTP: POST /search, GET /health, GET /stats
    Serve {
        #[arg(long)]
        index: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Embedder sidecar URL (scripts/embed_server.py); without it, neural search needs a
        /// vector in the request
        #[arg(long)]
        embedder: Option<String>,
    },
    /// Turn a crawl into corpus.jsonl and print a JSON report of how fast each stage ran
    #[command(subcommand)]
    Ingest(Ingest),
    /// Measure retrieval quality and print a JSON report
    #[command(subcommand)]
    Eval(Eval),
    /// Measure speed and print a JSON report
    #[command(subcommand)]
    Bench(Bench),
}

#[derive(Subcommand)]
enum Bench {
    /// B5 for neural and hybrid search: the same filters as keyword-filters, recall against
    /// exact search within each filter
    HybridFilters {
        #[arg(long)]
        index: PathBuf,
        /// The vectors the index was built from
        #[arg(long)]
        base: PathBuf,
        /// One JSON object per line, query text in --field
        #[arg(long)]
        queries: PathBuf,
        #[arg(long, default_value = "text")]
        field: String,
        /// Row i is the vector of query i
        #[arg(long)]
        query_vectors: PathBuf,
        #[arg(long, default_value = "domain")]
        domain_field: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// B5 for keyword search: domain and crawl date filters at 0.1% to 50% selectivity
    KeywordFilters {
        #[arg(long)]
        index: PathBuf,
        /// BEIR directory whose test queries to run (queries.jsonl and qrels/test.tsv)
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "domain")]
        domain_field: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// B12: HTTP latency per mode, with the server's per-stage split, against a running serve
    Http {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// One JSON object per line
        #[arg(long)]
        queries: PathBuf,
        /// The field of each line that holds the query text
        #[arg(long, default_value = "text")]
        field: String,
        /// Send row i as the vector of query i (neural and hybrid); without it the server
        /// embeds the query text
        #[arg(long)]
        query_vectors: Option<PathBuf>,
        #[arg(
            long,
            value_enum,
            value_delimiter = ',',
            default_value = "keyword,neural,hybrid"
        )]
        modes: Vec<Mode>,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// B7: exhaustive vs MaxScore keyword search on a BEIR test set
    Bm25 {
        /// Directory with queries.jsonl and qrels/test.tsv
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        index: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Thread counts to measure QPS at
        #[arg(long, value_delimiter = ',', default_value = "1,4,10")]
        threads: Vec<usize>,
    },
    /// B8: bytes per posting at each step of the posting format; builds the index in --out
    AblatePostings {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// B2: recall@100 of Hamming, float-table and u8-table first-stage scoring
    AblateQuant {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        /// Use the first this many base rows
        #[arg(long, default_value_t = 200_000)]
        docs: usize,
        #[arg(long, value_delimiter = ',', default_value = "64,128,256,512")]
        dims: Vec<usize>,
        /// Dims of the cosine the truth is ranked by [default: the smaller of 1024 and the dims]
        #[arg(long)]
        d_rerank: Option<usize>,
    },
    /// B5: filtered neural search latency and recall@10 at several selectivities
    Filters {
        #[arg(long)]
        index: PathBuf,
        /// The base vectors the index was built from, for the exact filtered truth
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        /// Keyword field holding categories c0, c1, ... (see scripts/add_categories.py)
        #[arg(long, default_value = "category")]
        field: String,
        #[arg(long, default_value_t = 1000)]
        categories: usize,
        #[arg(long, value_delimiter = ',', default_value = "0.001,0.01,0.1,0.5")]
        selectivity: Vec<f64>,
    },
    /// B3: neural recall@10, latency and QPS over clusters probed x rerank depth
    Ann {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        gt_full: PathBuf,
        #[arg(long)]
        gt_rerank: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "8,16,32,64,128")]
        nprobe: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "0,100,200,500")]
        rerank_depth: Vec<usize>,
        /// Thread counts to measure QPS at
        #[arg(long, value_delimiter = ',', default_value = "4,10")]
        threads: Vec<usize>,
    },
}

#[derive(Subcommand)]
enum Ingest {
    /// B11: FineWeb parquet (a file or a directory of them) to corpus.jsonl, with domain and
    /// crawl_date fields, exact duplicates dropped
    Fineweb {
        #[arg(long)]
        parquet: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Stop after writing this many docs
        #[arg(long)]
        limit: Option<u64>,
    },
}

#[derive(Subcommand)]
enum Eval {
    /// nDCG@10 and Recall@100 on a BEIR test set, the way Anserini's regressions measure them
    Beir {
        /// Directory with corpus.jsonl, queries.jsonl and qrels/test.tsv
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        index: PathBuf,
        /// Query vectors (.fbin, with query_ids.txt next to it naming each row's query):
        /// also evaluate neural and hybrid mode
        #[arg(long)]
        query_vectors: Option<PathBuf>,
        /// Also write the run in TREC format, for checking with trec_eval (with .<mode>
        /// appended when there are query vectors)
        #[arg(long)]
        run: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum Synth {
    /// Zipfian corpus with topical words, plus 2 to 4 term queries in queries.jsonl next to it
    Text {
        #[arg(long, default_value_t = 100_000)]
        docs: usize,
        #[arg(long, default_value_t = 1000)]
        queries: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Matryoshka-like vectors (.fbin), plus queries in queries.fbin next to them
    Vectors {
        #[arg(long, default_value_t = 100_000)]
        docs: usize,
        #[arg(long, default_value_t = 1000)]
        queries: usize,
        #[arg(long, default_value_t = 768)]
        dims: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Synth(Synth::Text {
            docs,
            queries,
            seed,
            out,
        }) => {
            let qpath = synth::text(&out, docs, queries, seed)?;
            println!(
                "wrote {docs} docs to {} and {queries} queries to {}",
                out.display(),
                qpath.display()
            );
        }
        Cmd::Synth(Synth::Vectors {
            docs,
            queries,
            dims,
            seed,
            out,
        }) => {
            let qpath = synth::vectors(&out, docs, queries, dims, seed)?;
            println!(
                "wrote {docs} vectors to {} and {queries} queries to {}",
                out.display(),
                qpath.display()
            );
        }
        Cmd::Index {
            corpus,
            out,
            vectors,
            d_code,
            d_rerank,
            nlist,
            keyword_fields,
            date_field,
        } => {
            let start = Instant::now();
            let opts = BuildOptions {
                vectors,
                d_code,
                d_rerank,
                nlist,
                keyword_fields,
                date_field,
            };
            let n = engine::build(&corpus, &out, &opts)?;
            println!(
                "indexed {n} docs into {} in {:.1}s",
                out.display(),
                start.elapsed().as_secs_f64()
            );
            memory_report(&out, n)?;
        }
        Cmd::Gt {
            base,
            queries,
            k,
            d_rerank,
            out,
        } => {
            let (base, queries) = (Fbin::open(&base)?, Fbin::open(&queries)?);
            std::fs::create_dir_all(&out)?;
            let d_rerank = d_rerank.unwrap_or(base.d.min(1024));
            for (name, dims) in [("gt_full.ibin", base.d), ("gt_rerank.ibin", d_rerank)] {
                let start = Instant::now();
                write_ibin(
                    &out.join(name),
                    k,
                    &kmeans::ground_truth(&base, &queries, dims, k),
                )?;
                println!(
                    "wrote {} ({dims} dims) in {:.1}s",
                    out.join(name).display(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
        Cmd::Search {
            index,
            query,
            query_vectors,
            query_row,
            rerank_depth,
            nprobe,
            kernel,
            k,
            mode,
            debug,
            strategy,
            filters,
        } => {
            let vector = query_vectors
                .map(|p| -> anyhow::Result<Vec<f32>> {
                    let f = Fbin::open(&p)?;
                    anyhow::ensure!(query_row < f.n, "{} has {} rows", p.display(), f.n);
                    Ok(f.row(query_row).to_vec())
                })
                .transpose()?;
            let req = Request {
                query,
                mode,
                k,
                debug,
                strategy,
                vector,
                rerank_depth,
                kernel,
                nprobe,
                filters: filters.into_iter().collect::<Filters>(),
            };
            let res = Index::open(&index)?.search(&req)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Cmd::Serve {
            index,
            addr,
            embedder,
        } => tokio::runtime::Runtime::new()?.block_on(server::serve(&index, &addr, embedder))?,
        Cmd::Ingest(Ingest::Fineweb {
            parquet,
            out,
            limit,
        }) => {
            let report = ingest::fineweb(&parquet, &out, limit)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Eval(Eval::Beir {
            data,
            index,
            query_vectors,
            run,
        }) => {
            let report = eval::beir(
                &Index::open(&index)?,
                &data,
                query_vectors.as_deref(),
                run.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::HybridFilters {
            index,
            base,
            queries,
            field,
            query_vectors,
            domain_field,
            k,
        }) => {
            let (base, vectors) = (Fbin::open(&base)?, Fbin::open(&query_vectors)?);
            let report =
                bench::hybrid_filters(&index, &base, &queries, &field, &vectors, &domain_field, k)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::KeywordFilters {
            index,
            data,
            domain_field,
            k,
        }) => {
            let report = bench::keyword_filters(&index, &data, &domain_field, k)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::Http {
            url,
            queries,
            field,
            query_vectors,
            modes,
            k,
        }) => {
            let vectors = query_vectors.as_deref().map(Fbin::open).transpose()?;
            let report = bench::http(&url, &queries, &field, vectors.as_ref(), &modes, k)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::Bm25 {
            data,
            index,
            k,
            threads,
        }) => {
            let report = bench::bm25(&Index::open(&index)?, &data, k, &threads)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::AblatePostings { corpus, out }) => {
            let report = bench::ablate_postings(&corpus, &out)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::AblateQuant {
            base,
            queries,
            docs,
            dims,
            d_rerank,
        }) => {
            let base = Fbin::open(&base)?;
            let d_rerank = d_rerank.unwrap_or(base.d.min(1024));
            let report = bench::ablate_quant(&base, &Fbin::open(&queries)?, docs, &dims, d_rerank)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::Filters {
            index,
            base,
            queries,
            field,
            categories,
            selectivity,
        }) => {
            let report = bench::filters(
                &index,
                &Fbin::open(&base)?,
                &Fbin::open(&queries)?,
                &field,
                categories,
                &selectivity,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Bench(Bench::Ann {
            index,
            queries,
            gt_full,
            gt_rerank,
            nprobe,
            rerank_depth,
            threads,
        }) => {
            let gt = [read_ibin(&gt_full)?, read_ibin(&gt_rerank)?];
            let report = bench::ann(
                &index,
                &Fbin::open(&queries)?,
                [&gt[0], &gt[1]],
                &nprobe,
                &rerank_depth,
                &threads,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

/// Bytes per part of the index in `dir`, total and per doc. Every part is memory-mapped
/// or read whole, so file size is memory size.
fn memory_report(dir: &std::path::Path, docs: usize) -> Result<()> {
    const PARTS: [(&str, &str); 12] = [
        ("codes.bin", "1-bit codes (hot)"),
        ("ids.bin", "doc id per code slot (hot)"),
        ("centroids.f32", "centroids"),
        ("clusters.bin", "cluster offsets"),
        ("rerank.f16", "f16 rerank vectors (cold)"),
        ("postings.bin", "postings"),
        ("skips.bin", "skip entries"),
        ("terms.fst", "term dictionary"),
        ("norms.bin", "doc length norms"),
        ("docs.bin", "doc store"),
        ("docs.off", "doc store offsets"),
        ("keyword.json", "keyword stats"),
    ];
    println!("{:<28} {:>14} {:>12}", "part", "bytes", "bytes/doc");
    for (file, part) in PARTS {
        if let Ok(m) = std::fs::metadata(dir.join(file)) {
            println!(
                "{part:<28} {:>14} {:>12.2}",
                m.len(),
                m.len() as f64 / docs.max(1) as f64
            );
        }
    }
    Ok(())
}

/// `field=a,b` is any of the values; `field=from..to` is an inclusive range, either end open.
fn parse_filter(s: &str) -> Result<(String, Filter), String> {
    let (field, value) = s.split_once('=').ok_or("expected field=value")?;
    let filter = match value.split_once("..") {
        Some((from, to)) => Filter::Range {
            gte: (!from.is_empty()).then(|| from.to_string()),
            lte: (!to.is_empty()).then(|| to.to_string()),
        },
        None => Filter::Any(value.split(',').map(String::from).collect()),
    };
    Ok((field.to_string(), filter))
}
