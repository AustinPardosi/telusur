use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{path::PathBuf, time::Instant};
use telusur::{
    bench,
    engine::{self, Index, Mode, Request, Strategy},
    eval, synth,
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
    },
    /// Search an index and print the response as JSON
    Search {
        #[arg(long)]
        index: PathBuf,
        query: String,
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
        Cmd::Index { corpus, out } => {
            let start = Instant::now();
            let n = engine::build(&corpus, &out)?;
            println!(
                "indexed {n} docs into {} in {:.1}s",
                out.display(),
                start.elapsed().as_secs_f64()
            );
        }
        Cmd::Search {
            index,
            query,
            k,
            mode,
            debug,
            strategy,
        } => {
            let req = Request {
                query,
                mode,
                k,
                debug,
                strategy,
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
    }
    Ok(())
}
