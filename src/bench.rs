//! Speed benchmarks. Every report names the machine it ran on.

use crate::{
    engine::{Index, Mode, Request, Response, Strategy},
    eval::load_beir,
    text::analyzer::analyze,
    util::{hardware, percentile},
};
use anyhow::Result;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, thread, time::Instant};

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
            let d = res.stats.map_or(0, |s| s.postings_decoded) as f64;
            decoded.push(d);
            if is_long {
                decoded_long.push(d);
            }
            scores.push(res.results.iter().map(|h| h.score).collect::<Vec<_>>());
        }
        tops.push(scores);

        let mut lat = Vec::with_capacity(timed);
        for q in texts.iter().cycle().take(timed) {
            lat.push(search(q, strategy)?.timings_ms.keyword);
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
