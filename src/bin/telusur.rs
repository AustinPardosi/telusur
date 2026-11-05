use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{path::PathBuf, time::Instant};
use telusur::{
    engine::{self, Index, Mode, Request},
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
    },
    /// Measure retrieval quality and print a JSON report
    #[command(subcommand)]
    Eval(Eval),
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
        } => {
            let res = Index::open(&index)?.search(&Request { query, mode, k })?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Cmd::Eval(Eval::Beir { data, index, run }) => {
            let report = eval::beir(&Index::open(&index)?, &data, run.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
