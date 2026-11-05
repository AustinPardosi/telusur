//! Speed benchmarks. Every report names the machine it ran on.

use crate::{
    engine::{self, BuildOptions, Index, Mode, Request, Response, Strategy},
    eval::load_beir,
    text::{analyzer::analyze, index::Footprint},
    util::{Fbin, hardware, percentile},
    vector::{ivf::VectorIndex, kernel::Kernel},
};
use anyhow::Result;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path, thread, time::Instant};

/// Latency is measured over at least this many queries, cycling through the query set.
const MIN_TIMED: usize = 1000;
/// Each throughput run does this many passes over the timed queries.
const QPS_PASSES: usize = 5;

#[derive(Serialize)]
pub struct Bm25Report {
    pub bench: &'static str,
    pub dataset: String,
    pub k: usize,
    pub queries: usize,
    /// Queries with at least 3 distinct terms after analysis.
    pub queries_3plus_terms: usize,
    pub timed_queries: usize,
    /// Queries whose MaxScore top-k scores differ from exhaustive (beyond 1e-5 relative).
    pub mismatched_queries: usize,
    pub machine: String,
    pub strategies: Vec<StrategyReport>,
}

#[derive(Serialize)]
pub struct StrategyReport {
    pub strategy: Strategy,
    pub postings_decoded_mean: f64,
    pub postings_decoded_mean_3plus_terms: f64,
    /// Keyword stage only, one thread.
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    /// Whole searches per second, by thread count.
    pub qps: BTreeMap<usize, f64>,
}

/// B7: exhaustive vs MaxScore over tiers on the test queries of a BEIR dataset.
pub fn bm25(index: &Index, data: &Path, k: usize, threads: &[usize]) -> Result<Bm25Report> {
    let (_, queries) = load_beir(data)?;
    let texts: Vec<&str> = queries.values().map(String::as_str).collect();
    let long: Vec<bool> = texts
        .iter()
        .map(|q| {
            let mut t = analyze(q);
            t.sort_unstable();
            t.dedup();
            t.len() >= 3
        })
        .collect();
    let n_long = long.iter().filter(|&&l| l).count();
    let timed = texts.len() * MIN_TIMED.div_ceil(texts.len());
    let search = |q: &str, strategy| -> Result<Response> {
        index.search(&Request {
            query: q.into(),
            mode: Mode::Keyword,
            k,
            debug: true,
            strategy,
            ..Request::default()
        })
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;

    let mut strategies = Vec::new();
    let mut tops = Vec::new();
    for strategy in [Strategy::Exhaustive, Strategy::Maxscore] {
        // The warm-up pass also records the work done and the scores returned.
        let (mut decoded, mut decoded_long, mut scores) = (Vec::new(), Vec::new(), Vec::new());
        for (q, &is_long) in texts.iter().zip(&long) {
            let res = search(q, strategy)?;
            let d = res.stats.and_then(|s| s.postings_decoded).unwrap_or(0) as f64;
            decoded.push(d);
            if is_long {
                decoded_long.push(d);
            }
            scores.push(res.results.iter().map(|h| h.score).collect::<Vec<_>>());
        }
        tops.push(scores);

        let mut lat = Vec::with_capacity(timed);
        for q in texts.iter().cycle().take(timed) {
            lat.push(search(q, strategy)?.timings_ms.keyword.unwrap_or(0.0));
        }
        lat.sort_by(f64::total_cmp);

        let mut qps = BTreeMap::new();
        for &n in threads {
            let total = timed * QPS_PASSES;
            let start = Instant::now();
            thread::scope(|s| {
                let workers: Vec<_> = (0..n)
                    .map(|t| {
                        let (texts, search) = (&texts, &search);
                        s.spawn(move || -> Result<()> {
                            for i in (t..total).step_by(n) {
                                search(texts[i % texts.len()], strategy)?;
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

        strategies.push(StrategyReport {
            strategy,
            postings_decoded_mean: mean(&decoded),
            postings_decoded_mean_3plus_terms: mean(&decoded_long),
            p50_ms: percentile(&lat, 50.0),
            p99_ms: percentile(&lat, 99.0),
            mean_ms: mean(&lat),
            qps,
        });
    }

    let close = |a: f32, b: f32| (a - b).abs() <= 1e-5 * a.abs().max(b.abs());
    let mismatched_queries = tops[0]
        .iter()
        .zip(&tops[1])
        .filter(|(e, m)| e.len() != m.len() || e.iter().zip(m.iter()).any(|(&a, &b)| !close(a, b)))
        .count();

    Ok(Bm25Report {
        bench: "bm25",
        dataset: data
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into(),
        k,
        queries: texts.len(),
        queries_3plus_terms: n_long,
        timed_queries: timed,
        mismatched_queries,
        machine: hardware(),
        strategies,
    })
}

#[derive(Serialize)]
pub struct AblationReport {
    pub bench: &'static str,
    pub corpus: String,
    pub docs: usize,
    pub terms: usize,
    pub postings: u64,
    pub steps: Vec<AblationStep>,
    /// Final format's total over the naive one's.
    pub final_vs_naive: f64,
    pub at_least_50_percent_smaller: bool,
    /// Postings bytes of the delta varint step if tf were a varint too, not a u32.
    pub delta_varint_tf_postings: u64,
    pub machine: String,
}

#[derive(Serialize)]
pub struct AblationStep {
    #[serde(flatten)]
    pub footprint: Footprint,
    pub total: u64,
    pub bytes_per_posting: f64,
}

/// B8: bytes per posting at each step of the posting format, like the memory story in Exa's
/// BM25 post. Builds the index in `out`; the last step is the size of its files.
pub fn ablate_postings(corpus: &Path, out: &Path) -> Result<AblationReport> {
    let mut seen = None;
    let docs = engine::build_with(corpus, out, &BuildOptions::default(), |b| {
        seen = Some((b.ablation(), b.terms(), b.postings()))
    })?;
    let ((mut footprints, delta_varint_tf_postings), terms, postings) =
        seen.expect("build_with always inspects");
    let size = |f: &str| fs::metadata(out.join(f)).map(|m| m.len());
    footprints.push(Footprint {
        format: "+ inline singletons, FST dictionary, 1-byte norms (the index files)",
        postings: size("postings.bin")? + size("skips.bin")?,
        term_table: 0,
        term_strings: size("terms.fst")?,
        doc_lengths: size("norms.bin")?,
    });
    let final_vs_naive =
        footprints[footprints.len() - 1].total() as f64 / footprints[0].total() as f64;
    let steps = footprints
        .into_iter()
        .map(|footprint| {
            let total = footprint.total();
            AblationStep {
                footprint,
                total,
                bytes_per_posting: total as f64 / postings as f64,
            }
        })
        .collect();
    Ok(AblationReport {
        bench: "ablate-postings",
        corpus: corpus.display().to_string(),
        docs,
        terms,
        postings,
        steps,
        final_vs_naive,
        at_least_50_percent_smaller: final_vs_naive <= 0.5,
        delta_varint_tf_postings,
        machine: hardware(),
    })
}

#[derive(Serialize)]
pub struct AnnReport {
    pub bench: &'static str,
    pub index: String,
    pub nlist: usize,
    pub k: usize,
    pub queries: usize,
    pub timed_queries: usize,
    pub kernel: Kernel,
    pub machine: String,
    pub runs: Vec<AnnRun>,
}

#[derive(Serialize)]
pub struct AnnRun {
    pub nprobe: usize,
    pub rerank_depth: usize,
    /// Recall@k against the exact top k by full vectors and by rerank vectors.
    pub recall_full: f64,
    pub recall_rerank: f64,
    pub codes_scanned_mean: f64,
    /// Route, scan and rerank, one thread.
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub qps: BTreeMap<usize, f64>,
}

/// B3: recall against both ground truths, latency and QPS for every pair of clusters probed
/// and rerank depth, on the vector part of the index in `dir`.
pub fn ann(
    dir: &Path,
    queries: &Fbin,
    gt: [&[Vec<u32>]; 2],
    nprobes: &[usize],
    depths: &[usize],
    threads: &[usize],
) -> Result<AnnReport> {
    const K: usize = 10;
    let index = VectorIndex::open(dir)?
        .ok_or_else(|| anyhow::anyhow!("{} has no vectors", dir.display()))?;
    let kernel = Kernel::default_for_process()?;
    let qs: Vec<&[f32]> = (0..queries.n).map(|i| queries.row(i)).collect();
    anyhow::ensure!(
        gt.iter().all(|g| g.len() == qs.len()),
        "ground truth rows must match the queries"
    );
    let timed = qs.len() * MIN_TIMED.div_ceil(qs.len().max(1));
    let recall = |hits: &[(u32, f32)], truth: &[u32]| {
        hits.iter()
            .filter(|(d, _)| truth[..K.min(truth.len())].contains(d))
            .count() as f64
            / K as f64
    };

    let mut runs = Vec::new();
    for &nprobe in nprobes {
        for &depth in depths {
            let search = |q: &[f32]| index.search(q, K, depth, nprobe, kernel);
            let (mut recall_full, mut recall_rerank, mut scanned) = (0.0, 0.0, 0.0);
            for (i, q) in qs.iter().enumerate() {
                let (hits, trace) = search(q)?;
                recall_full += recall(&hits, &gt[0][i]);
                recall_rerank += recall(&hits, &gt[1][i]);
                scanned += trace.codes_scanned as f64;
            }
            let mut lat = Vec::with_capacity(timed);
            for q in qs.iter().cycle().take(timed) {
                let start = Instant::now();
                search(q)?;
                lat.push(start.elapsed().as_secs_f64() * 1e3);
            }
            lat.sort_by(f64::total_cmp);
            let mut qps = BTreeMap::new();
            for &n in threads {
                let start = Instant::now();
                thread::scope(|s| {
                    let workers: Vec<_> = (0..n)
                        .map(|t| {
                            let (qs, search) = (&qs, &search);
                            s.spawn(move || -> Result<()> {
                                for i in (t..timed).step_by(n) {
                                    search(qs[i % qs.len()])?;
                                }
                                Ok(())
                            })
                        })
                        .collect();
                    workers
                        .into_iter()
                        .try_for_each(|w| w.join().expect("bench thread panicked"))
                })?;
                qps.insert(n, timed as f64 / start.elapsed().as_secs_f64());
            }
            let n = qs.len() as f64;
            runs.push(AnnRun {
                nprobe,
                rerank_depth: depth,
                recall_full: recall_full / n,
                recall_rerank: recall_rerank / n,
                codes_scanned_mean: scanned / n,
                p50_ms: percentile(&lat, 50.0),
                p99_ms: percentile(&lat, 99.0),
                mean_ms: lat.iter().sum::<f64>() / lat.len() as f64,
                qps,
            });
        }
    }
    Ok(AnnReport {
        bench: "ann",
        index: dir.display().to_string(),
        nlist: index.nlist(),
        k: K,
        queries: qs.len(),
        timed_queries: timed,
        kernel,
        machine: hardware(),
        runs,
    })
}
