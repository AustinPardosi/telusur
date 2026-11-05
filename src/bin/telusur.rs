use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{path::PathBuf, time::Instant};
use telusur::{
    bench,
    engine::{self, BuildOptions, Index, Mode, Request, Strategy},
    eval, synth,
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
        /// Take the query vector from this .fbin file (neural mode)
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
    },
    /// Measure retrieval quality and print a JSON report
    #[command(subcommand)]
    Eval(Eval),
    /// Measure speed and print a JSON report
    #[command(subcommand)]
    Bench(Bench),
}

#[derive(Subcommand)]
enum Bench {
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
enum Eval {
    /// nDCG@10 and Recall@100 on a BEIR test set, the way Anserini's regressions measure them
    Beir {
        /// Directory with corpus.jsonl, queries.jsonl and qrels/test.tsv
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        index: PathBuf,
        /// Also write the run in TREC format, for checking with trec_eval
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
        } => {
            let start = Instant::now();
            let opts = BuildOptions {
                vectors,
                d_code,
                d_rerank,
                nlist,
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
            };
            let res = Index::open(&index)?.search(&req)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Cmd::Eval(Eval::Beir { data, index, run }) => {
            let report = eval::beir(&Index::open(&index)?, &data, run.as_deref())?;
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
