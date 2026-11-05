//! Speed benchmarks. Every report names the machine it ran on.

use crate::{
    engine::{self, BuildOptions, Index, Mode, Request, Response, Strategy},
    eval::load_beir,
    filter::{Filter, FilterIndex, Filters},
    text::{analyzer::analyze, index::Footprint},
    util::{Fbin, TopK, hardware, percentile},
    vector::{
        ivf::{self, VectorIndex},
        kernel::{Kernel, QueryLut, pack_blocks, scan},
        kmeans::exact_topk,
        quant::{binarize, truncate_normalize},
    },
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
    thread,
    time::Instant,
};

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
            let search = |q: &[f32]| index.search(q, K, depth, nprobe, kernel, None);
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
                                for i in (t..timed * QPS_PASSES).step_by(n) {
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
                qps.insert(
                    n,
                    (timed * QPS_PASSES) as f64 / start.elapsed().as_secs_f64(),
                );
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

#[derive(Serialize)]
pub struct QuantReport {
    pub bench: &'static str,
    pub docs: usize,
    pub queries: usize,
    pub k: usize,
    /// What recall is measured against.
    pub reference: String,
    pub kernel: Kernel,
    pub machine: String,
    pub rows: Vec<QuantRow>,
}

/// Recall@100 of each first-stage scorer at one code size.
#[derive(Serialize)]
pub struct QuantRow {
    pub dims: usize,
    /// Binary query against binary codes: agreeing bits.
    pub hamming: f64,
    /// Float query against binary codes, exactly: the dot product with the codes' +1 / -1.
    pub float_table: f64,
    /// The same with the u8-quantized lookup tables the kernels use.
    pub u8_table: f64,
}

/// B2: how much of the true top 100 each first-stage scorer finds in its own top 100, over
/// the first `docs` base rows, scanning every code (no clusters). The truth is the exact top
/// 100 by cosine at the rerank dims: the docs the reranker needs to be handed.
pub fn ablate_quant(
    base: &Fbin,
    queries: &Fbin,
    docs: usize,
    dims: &[usize],
    d_rerank: usize,
) -> Result<QuantReport> {
    const K: usize = 100;
    let n = docs.min(base.n);
    let kernel = Kernel::default_for_process()?;
    let prefix = |v: &[f32], d: usize| truncate_normalize(v, d);
    let flat = |d: usize, f: &dyn Fn(&[f32]) -> Vec<f32>| -> Vec<f32> {
        (0..queries.n)
            .flat_map(|i| f(&queries.row(i)[..d]))
            .collect()
    };
    let truth = exact_topk(
        n,
        d_rerank,
        |i| prefix(base.row(i), d_rerank),
        &flat(d_rerank, &|q| prefix(q, d_rerank)),
        K,
    );
    let recall = |found: &[Vec<u32>]| {
        found
            .iter()
            .zip(&truth)
            .map(|(f, t)| f.iter().filter(|d| t.contains(d)).count())
            .sum::<usize>() as f64
            / (K * truth.len()) as f64
    };
    let signs = |v: &[f32]| {
        v.iter()
            .map(|&x| if x > 0.0 { 1.0 } else { -1.0 })
            .collect::<Vec<f32>>()
    };

    let mut rows = Vec::new();
    for &d in dims {
        anyhow::ensure!(
            d.is_multiple_of(64) && d <= base.d.min(1024),
            "code dims must be multiples of 64, at most 1024"
        );
        let code_signs = |i: usize| signs(&base.row(i)[..d]);
        let hamming = exact_topk(n, d, code_signs, &flat(d, &signs), K);
        let float_table = exact_topk(n, d, code_signs, &flat(d, &|q| prefix(q, d)), K);

        let mut codes = vec![0u8; n * d / 8];
        for (i, code) in codes.chunks_exact_mut(d / 8).enumerate() {
            binarize(&base.row(i)[..d], code);
        }
        let blocks = pack_blocks(&codes, d / 8);
        let mut scores = vec![0u16; n.div_ceil(32) * 32];
        let mut u8_table = Vec::with_capacity(queries.n);
        for qi in 0..queries.n {
            scan(
                kernel,
                &QueryLut::new(&prefix(queries.row(qi), d)),
                &blocks,
                &mut scores,
            )?;
            let mut top = TopK::new(K);
            for (doc, &s) in (0..n as u32).zip(&scores) {
                top.push(doc, f32::from(s));
            }
            u8_table.push(
                top.into_sorted()
                    .into_iter()
                    .map(|(doc, _)| doc)
                    .collect::<Vec<_>>(),
            );
        }
        rows.push(QuantRow {
            dims: d,
            hamming: recall(&hamming),
            float_table: recall(&float_table),
            u8_table: recall(&u8_table),
        });
    }
    Ok(QuantReport {
        bench: "ablate-quant",
        docs: n,
        queries: queries.n,
        k: K,
        reference: format!("exact top {K} by cosine at {d_rerank} dims, over the same {n} docs"),
        kernel,
        machine: hardware(),
        rows,
    })
}

#[derive(Serialize)]
pub struct FilterReport {
    pub bench: &'static str,
    pub index: String,
    pub field: String,
    pub queries: usize,
    pub nprobe: usize,
    pub rerank_depth: usize,
    pub kernel: Kernel,
    pub machine: String,
    pub runs: Vec<FilterRun>,
}

#[derive(Serialize)]
pub struct FilterRun {
    /// Share of docs the filter allows; 1.0 is no filter.
    pub selectivity: f64,
    pub docs_allowed: u64,
    /// "exact" scores every allowed doc; "clusters" probes nprobe / selectivity clusters.
    pub path: &'static str,
    pub clusters_probed_mean: f64,
    /// Against the exact top 10 among the allowed docs, by cosine at the rerank dims.
    pub recall_at_10: f64,
    /// Filter lookup plus neural search, one thread.
    pub p50_ms: f64,
    pub p99_ms: f64,
}

/// B5: filtered neural search on the index in `dir`, whose `field` holds categories c0 to
/// c(categories - 1). Selectivity s picks the first s x categories of them. Each query
/// resolves the filter (bitmap lookups) and searches with it, as the engine does.
pub fn filters(
    dir: &Path,
    base: &Fbin,
    queries: &Fbin,
    field: &str,
    categories: usize,
    selectivities: &[f64],
) -> Result<FilterReport> {
    const K: usize = 10;
    let index = VectorIndex::open(dir)?
        .ok_or_else(|| anyhow::anyhow!("{} has no vectors", dir.display()))?;
    let filter_index = FilterIndex::open(dir)?;
    let kernel = Kernel::default_for_process()?;
    let (nprobe, rerank_depth) = (ivf::DEFAULT_NPROBE, 500);
    let d_rerank = base.d.min(1024);
    let qs: Vec<&[f32]> = (0..queries.n).map(|i| queries.row(i)).collect();
    let q_flat: Vec<f32> = qs
        .iter()
        .flat_map(|q| truncate_normalize(q, d_rerank))
        .collect();

    let mut runs = Vec::new();
    for &s in std::iter::once(&1.0).chain(selectivities) {
        let filters: Filters = if s < 1.0 {
            let m = ((s * categories as f64).round() as usize).max(1);
            [(
                field.to_string(),
                Filter::Any((0..m).map(|c| format!("c{c}")).collect()),
            )]
            .into()
        } else {
            Filters::new()
        };
        let allowed = filter_index.allowed(&filters)?;
        let ids: Vec<u32> = match &allowed {
            Some(a) => a.iter().collect(),
            None => (0..base.n as u32).collect(),
        };
        let truth: Vec<Vec<u32>> = exact_topk(
            ids.len(),
            d_rerank,
            |j| truncate_normalize(base.row(ids[j] as usize), d_rerank),
            &q_flat,
            K,
        )
        .into_iter()
        .map(|t| t.into_iter().map(|j| ids[j as usize]).collect())
        .collect();

        let search = |q: &[f32]| -> Result<(Vec<(u32, f32)>, ivf::Trace)> {
            let allowed = filter_index.allowed(&filters)?;
            index.search(q, K, rerank_depth, nprobe, kernel, allowed.as_ref())
        };
        let (mut found, mut probed) = (0, 0);
        for (q, t) in qs.iter().zip(&truth) {
            let (hits, trace) = search(q)?;
            found += hits.iter().filter(|(d, _)| t.contains(d)).count();
            probed += trace.clusters_probed;
        }
        let mut lat = Vec::with_capacity(qs.len());
        for q in qs
            .iter()
            .cycle()
            .take(qs.len() * MIN_TIMED.div_ceil(qs.len().max(1)))
        {
            let start = Instant::now();
            search(q)?;
            lat.push(start.elapsed().as_secs_f64() * 1e3);
        }
        lat.sort_by(f64::total_cmp);
        let allowed_docs = ids.len() as u64;
        runs.push(FilterRun {
            selectivity: allowed_docs as f64 / base.n as f64,
            docs_allowed: allowed_docs,
            path: if allowed.is_some() && allowed_docs <= ivf::EXACT_FILTER_MAX {
                "exact"
            } else {
                "clusters"
            },
            clusters_probed_mean: probed as f64 / qs.len() as f64,
            recall_at_10: found as f64 / (K * qs.len()) as f64,
            p50_ms: percentile(&lat, 50.0),
            p99_ms: percentile(&lat, 99.0),
        });
    }
    Ok(FilterReport {
        bench: "filters",
        index: dir.display().to_string(),
        field: field.to_string(),
        queries: qs.len(),
        nprobe,
        rerank_depth,
        kernel,
        machine: hardware(),
        runs,
    })
}

#[derive(Serialize)]
pub struct HttpReport {
    pub bench: &'static str,
    pub url: String,
    pub queries: usize,
    pub timed_queries: usize,
    pub k: usize,
    /// Whether requests carried query vectors; without them the server embeds query text.
    pub vectors_in_request: bool,
    /// The server's GET /stats.
    pub server: Value,
    pub machine: String,
    pub modes: Vec<HttpRun>,
}

#[derive(Serialize)]
pub struct HttpRun {
    pub mode: Mode,
    /// Request sent to response read, at the client.
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    /// The server's `timings_ms`, per stage: p50, p99 and mean.
    pub stages: BTreeMap<String, [f64; 3]>,
    /// Responses whose `mode_used` differs from the mode asked for (should be 0).
    pub fallbacks: usize,
}

/// B12: one client, one request at a time, against a running `telusur serve` at `url`. Query
/// text comes from `field` of each line of `queries`; with `vectors`, row i goes with line i
/// in neural and hybrid requests. Bodies are serialized before timing.
pub fn http(
    url: &str,
    queries: &Path,
    field: &str,
    vectors: Option<&Fbin>,
    modes: &[Mode],
    k: usize,
) -> Result<HttpReport> {
    let texts: Vec<String> = fs::read_to_string(queries)?
        .lines()
        .map(|l| -> Result<String> {
            let v: Value = serde_json::from_str(l)?;
            Ok(v[field]
                .as_str()
                .with_context(|| format!("a query has no string field {field}"))?
                .to_string())
        })
        .collect::<Result<_>>()?;
    if let Some(v) = vectors {
        ensure!(
            v.n == texts.len(),
            "{} queries but {} vectors",
            texts.len(),
            v.n
        );
    }
    let timed = texts.len() * MIN_TIMED.div_ceil(texts.len().max(1));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let client = reqwest::Client::new();
    let url = url.trim_end_matches('/');
    let server: Value = rt.block_on(async {
        client
            .get(format!("{url}/stats"))
            .send()
            .await?
            .json()
            .await
    })?;

    let mut runs = Vec::new();
    for &mode in modes {
        let bodies: Vec<String> = texts
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let mut b = json!({ "query": q, "mode": mode, "k": k });
                if let (Some(v), Mode::Neural | Mode::Hybrid) = (vectors, mode) {
                    b["vector"] = json!(v.row(i));
                }
                b.to_string()
            })
            .collect();
        let send = |body: &String| {
            let req = client
                .post(format!("{url}/search"))
                .header("content-type", "application/json")
                .body(body.clone());
            async move {
                let res = req.send().await?;
                ensure!(res.status().is_success(), "HTTP {}", res.status());
                Ok::<Value, anyhow::Error>(res.json().await?)
            }
        };
        let (lat, stages, fallbacks) = rt.block_on(async {
            for b in &bodies {
                send(b).await?; // warm-up
            }
            let (mut lat, mut stages, mut fallbacks) = (Vec::new(), BTreeMap::new(), 0);
            for b in bodies.iter().cycle().take(timed) {
                let start = Instant::now();
                let res = send(b).await?;
                lat.push(start.elapsed().as_secs_f64() * 1e3);
                fallbacks += usize::from(res["mode_used"] != json!(mode));
                for (stage, ms) in res["timings_ms"].as_object().into_iter().flatten() {
                    let ms = ms.as_f64().unwrap_or(0.0);
                    stages
                        .entry(stage.clone())
                        .or_insert_with(Vec::new)
                        .push(ms);
                }
            }
            Ok::<_, anyhow::Error>((lat, stages, fallbacks))
        })?;
        let summary = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
            [percentile(&v, 50.0), percentile(&v, 99.0), mean]
        };
        let [p50_ms, p99_ms, mean_ms] = summary(lat);
        runs.push(HttpRun {
            mode,
            p50_ms,
            p99_ms,
            mean_ms,
            stages: stages.into_iter().map(|(s, v)| (s, summary(v))).collect(),
            fallbacks,
        });
    }
    Ok(HttpReport {
        bench: "http",
        url: url.to_string(),
        queries: texts.len(),
        timed_queries: timed,
        k,
        vectors_in_request: vectors.is_some(),
        server,
        machine: hardware(),
        modes: runs,
    })
}

#[derive(Serialize)]
pub struct KeywordFilterReport {
    pub bench: &'static str,
    pub index: String,
    pub docs: usize,
    pub queries_from: String,
    pub queries: usize,
    pub k: usize,
    pub machine: String,
    pub runs: Vec<KeywordFilterRun>,
}

#[derive(Serialize)]
pub struct KeywordFilterRun {
    /// What the filter asks for, e.g. "the 12 biggest domains" or "crawl_date >= 2022-05-20".
    pub filter: String,
    /// Share of docs the filter allows; 1.0 is no filter.
    pub selectivity: f64,
    pub docs_allowed: u64,
    pub postings_decoded_mean: f64,
    /// Queries whose MaxScore top-k scores differ from exhaustive under the same filter.
    pub mismatched_queries: usize,
    /// Filter lookup plus MaxScore keyword search, one thread.
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
}

/// Filters on web data, each with a label: none, lists of the biggest domains (values of
/// `domain_field`) holding about 0.1% and 1% of the `n` docs (past that a list is thousands of
/// domains), and runs of consecutive crawl days closest to 0.1%, 1%, 10% and 50%.
fn web_filter_specs(
    filter_index: &FilterIndex,
    n: usize,
    domain_field: &str,
) -> Result<Vec<(String, Filters)>> {
    // The shortest prefix of `counts` holding at least `share` of the docs.
    let prefix = |counts: &[(String, u64)], share: f64| {
        let mut sum = 0;
        let len = counts
            .iter()
            .position(|(_, c)| {
                sum += c;
                sum as f64 >= share * n as f64
            })
            .map_or(counts.len(), |i| i + 1);
        counts[..len]
            .iter()
            .map(|(v, _)| v.clone())
            .collect::<Vec<_>>()
    };
    let mut specs = vec![("no filter".to_string(), Filters::new())];
    let mut domains = filter_index.value_counts(domain_field)?;
    domains.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for share in [0.001, 0.01] {
        let values = prefix(&domains, share);
        let label = format!("the {} biggest {domain_field}s", values.len());
        specs.push((
            label,
            [(domain_field.to_string(), Filter::Any(values))].into(),
        ));
    }
    if let Some(date) = filter_index.date_field() {
        let days = filter_index.value_counts(date)?;
        for share in [0.001, 0.01, 0.1, 0.5] {
            // The run of consecutive crawl days whose doc count is closest to the target.
            let target = share * n as f64;
            let mut best = (f64::INFINITY, 0, 0);
            for i in 0..days.len() {
                let mut sum = 0;
                for (j, (_, c)) in days.iter().enumerate().skip(i) {
                    sum += c;
                    let off = (sum as f64 - target).abs();
                    if off < best.0 {
                        best = (off, i, j);
                    }
                    if sum as f64 >= target {
                        break;
                    }
                }
            }
            let (from, to) = (days[best.1].0.clone(), days[best.2].0.clone());
            let label = format!("{date} {from} to {to}");
            let range = Filter::Range {
                gte: Some(from),
                lte: Some(to),
            };
            specs.push((label, [(date.to_string(), range)].into()));
        }
    }
    Ok(specs)
}

/// B5 for keyword search on the index in `dir`: lists of the biggest domains (values of
/// `domain_field`) and ranges of the most recent crawl days, each grown until it allows about
/// 0.1%, 1%, 10% or 50% of the docs (domains only to 1%: past that a list is thousands of
/// domains), searched with the test queries of the BEIR dataset in `data`.
pub fn keyword_filters(
    dir: &Path,
    data: &Path,
    domain_field: &str,
    k: usize,
) -> Result<KeywordFilterReport> {
    let index = Index::open(dir)?;
    let filter_index = FilterIndex::open(dir)?;
    let n = index.docs();
    let (_, queries) = load_beir(data)?;
    let texts: Vec<&str> = queries.values().map(String::as_str).collect();

    let specs = web_filter_specs(&filter_index, n, domain_field)?;

    let timed = texts.len() * MIN_TIMED.div_ceil(texts.len());
    let search = |q: &str, strategy, filters: &Filters| {
        index.search(&Request {
            query: q.into(),
            mode: Mode::Keyword,
            k,
            debug: true,
            strategy,
            filters: filters.clone(),
            ..Request::default()
        })
    };
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-5 * a.abs().max(b.abs());
    let mut runs = Vec::new();
    for (filter, filters) in specs {
        let docs_allowed = filter_index
            .allowed(&filters)?
            .map_or(n as u64, |a| a.len());
        // Warm-up pass, also checking MaxScore against exhaustive.
        let (mut decoded, mut mismatched_queries) = (0, 0);
        for q in &texts {
            let m = search(q, Strategy::Maxscore, &filters)?;
            let e = search(q, Strategy::Exhaustive, &filters)?;
            decoded += m.stats.and_then(|s| s.postings_decoded).unwrap_or(0);
            let differs = m.results.len() != e.results.len()
                || m.results
                    .iter()
                    .zip(&e.results)
                    .any(|(a, b)| !close(a.score, b.score));
            mismatched_queries += usize::from(differs);
        }
        let mut lat = Vec::with_capacity(timed);
        for q in texts.iter().cycle().take(timed) {
            let start = Instant::now();
            search(q, Strategy::Maxscore, &filters)?;
            lat.push(start.elapsed().as_secs_f64() * 1e3);
        }
        lat.sort_by(f64::total_cmp);
        runs.push(KeywordFilterRun {
            filter,
            selectivity: docs_allowed as f64 / n as f64,
            docs_allowed,
            postings_decoded_mean: decoded as f64 / texts.len() as f64,
            mismatched_queries,
            p50_ms: percentile(&lat, 50.0),
            p99_ms: percentile(&lat, 99.0),
            mean_ms: lat.iter().sum::<f64>() / lat.len() as f64,
        });
    }
    Ok(KeywordFilterReport {
        bench: "keyword-filters",
        index: dir.display().to_string(),
        docs: n,
        queries_from: data.display().to_string(),
        queries: texts.len(),
        k,
        machine: hardware(),
        runs,
    })
}

#[derive(Serialize)]
pub struct HybridFilterReport {
    pub bench: &'static str,
    pub index: String,
    pub docs: usize,
    pub queries: usize,
    pub k: usize,
    pub nprobe: usize,
    pub rerank_depth: usize,
    pub machine: String,
    pub runs: Vec<HybridFilterRun>,
}

#[derive(Serialize)]
pub struct HybridFilterRun {
    pub filter: String,
    /// Share of docs the filter allows; 1.0 is no filter.
    pub selectivity: f64,
    pub docs_allowed: u64,
    /// Against the exact top k among the allowed docs, by cosine at the rerank dims.
    pub neural_recall: f64,
    /// Against RRF over exhaustive keyword search and exact neural search, both filtered.
    pub hybrid_recall: f64,
    /// Whole searches (filter lookup included), one thread; hybrid runs its two retrievers
    /// in parallel.
    pub neural_p50_ms: f64,
    pub neural_p99_ms: f64,
    pub hybrid_p50_ms: f64,
    pub hybrid_p99_ms: f64,
    /// Results outside the filter, over every query of both modes (must be 0).
    pub leaked: usize,
}

/// B5 for neural and hybrid search on the index in `dir` (built from `base`), with the filters
/// of `web_filter_specs`. Query text is field `field` of each line of `queries`, and its vector
/// the same row of `vectors`.
pub fn hybrid_filters(
    dir: &Path,
    base: &Fbin,
    queries: &Path,
    field: &str,
    vectors: &Fbin,
    domain_field: &str,
    k: usize,
) -> Result<HybridFilterReport> {
    let index = Index::open(dir)?;
    let filter_index = FilterIndex::open(dir)?;
    let n = index.docs();
    ensure!(
        base.n == n,
        "the index has {n} docs but the base has {} rows",
        base.n
    );
    let texts: Vec<String> = fs::read_to_string(queries)?
        .lines()
        .map(|l| -> Result<String> {
            let v: Value = serde_json::from_str(l)?;
            Ok(v[field]
                .as_str()
                .context("a query has no text")?
                .to_string())
        })
        .collect::<Result<_>>()?;
    ensure!(
        texts.len() == vectors.n,
        "{} queries but {} vectors",
        texts.len(),
        vectors.n
    );
    let doc_of: HashMap<String, u32> = (0..n as u32)
        .map(|d| Ok((index.external_id(d)?.to_string(), d)))
        .collect::<Result<_>>()?;
    let d_rerank = base.d.min(1024);
    let q_flat: Vec<f32> = (0..vectors.n)
        .flat_map(|i| truncate_normalize(vectors.row(i), d_rerank))
        .collect();
    let depth = k.max(engine::FUSE_DEPTH);
    let timed = texts.len() * MIN_TIMED.div_ceil(texts.len().max(1));

    let mut runs = Vec::new();
    for (filter, filters) in web_filter_specs(&filter_index, n, domain_field)? {
        let allowed = filter_index.allowed(&filters)?;
        let ids: Vec<u32> = match &allowed {
            Some(a) => a.iter().collect(),
            None => (0..n as u32).collect(),
        };
        let exact: Vec<Vec<u32>> = exact_topk(
            ids.len(),
            d_rerank,
            |j| truncate_normalize(base.row(ids[j] as usize), d_rerank),
            &q_flat,
            depth,
        )
        .into_iter()
        .map(|t| t.into_iter().map(|j| ids[j as usize]).collect())
        .collect();
        let search = |i: usize, mode, strategy| {
            index.search(&Request {
                query: texts[i].clone(),
                mode,
                k,
                strategy,
                vector: Some(vectors.row(i).to_vec()),
                filters: filters.clone(),
                ..Request::default()
            })
        };
        let docs = |r: &Response| -> Vec<u32> { r.results.iter().map(|h| doc_of[&h.id]).collect() };
        let passes = |d: u32| allowed.as_ref().is_none_or(|a| a.contains(d));

        // Recall pass, which is also the warm-up.
        let (mut neural_found, mut hybrid_found, mut leaked) = (0, 0, 0);
        for i in 0..texts.len() {
            let neural = docs(&search(i, Mode::Neural, Strategy::Maxscore)?);
            let hybrid = docs(&search(i, Mode::Hybrid, Strategy::Maxscore)?);
            leaked += neural
                .iter()
                .chain(&hybrid)
                .filter(|&&d| !passes(d))
                .count();
            let truth = &exact[i][..k.min(exact[i].len())];
            neural_found += neural.iter().filter(|d| truth.contains(d)).count();
            let keyword = index.search(&Request {
                query: texts[i].clone(),
                mode: Mode::Keyword,
                k: depth,
                strategy: Strategy::Exhaustive,
                filters: filters.clone(),
                ..Request::default()
            })?;
            let kw: Vec<(u32, f32)> = docs(&keyword).into_iter().map(|d| (d, 0.0)).collect();
            let nn: Vec<(u32, f32)> = exact[i].iter().map(|&d| (d, 0.0)).collect();
            let fused: Vec<u32> = engine::rrf(&kw, &nn, k).into_iter().map(|f| f.0).collect();
            hybrid_found += hybrid.iter().filter(|d| fused.contains(d)).count();
        }
        let latency = |mode| -> Result<[f64; 2]> {
            let mut lat = Vec::with_capacity(timed);
            for i in (0..texts.len()).cycle().take(timed) {
                let start = Instant::now();
                search(i, mode, Strategy::Maxscore)?;
                lat.push(start.elapsed().as_secs_f64() * 1e3);
            }
            lat.sort_by(f64::total_cmp);
            Ok([percentile(&lat, 50.0), percentile(&lat, 99.0)])
        };
        let [neural_p50_ms, neural_p99_ms] = latency(Mode::Neural)?;
        let [hybrid_p50_ms, hybrid_p99_ms] = latency(Mode::Hybrid)?;
        let wanted = (texts.len() * k.min(ids.len())).max(1) as f64;
        runs.push(HybridFilterRun {
            filter,
            selectivity: ids.len() as f64 / n as f64,
            docs_allowed: ids.len() as u64,
            neural_recall: neural_found as f64 / wanted,
            hybrid_recall: hybrid_found as f64 / wanted,
            neural_p50_ms,
            neural_p99_ms,
            hybrid_p50_ms,
            hybrid_p99_ms,
            leaked,
        });
    }
    Ok(HybridFilterReport {
        bench: "hybrid-filters",
        index: dir.display().to_string(),
        docs: n,
        queries: texts.len(),
        k,
        nprobe: ivf::DEFAULT_NPROBE,
        rerank_depth: Request::default().rerank_depth,
        machine: hardware(),
        runs,
    })
}
