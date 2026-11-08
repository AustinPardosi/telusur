use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf};
use telusur::{
    bench,
    engine::{self, BuildOptions, Index, Mode, Ranks, Request, Response, Strategy},
    filter::{Filter, FilterIndex, Filters},
    synth,
    util::Fbin,
    vector::{kernel::Kernel, kmeans},
};

fn build(name: &str, docs: &[Value]) -> Index {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("corpus.jsonl");
    fs::write(
        &corpus,
        docs.iter().map(|d| format!("{d}\n")).collect::<String>(),
    )
    .unwrap();
    engine::build(&corpus, &dir.join("index"), &BuildOptions::default()).unwrap();
    Index::open(&dir.join("index")).unwrap()
}

fn search(index: &Index, query: &str, mode: Mode, k: usize) -> anyhow::Result<Response> {
    index.search(&Request {
        query: query.into(),
        mode,
        k,
        ..Request::default()
    })
}

fn keyword(index: &Index, query: &str, k: usize, strategy: Strategy) -> Response {
    let req = Request {
        query: query.into(),
        mode: Mode::Keyword,
        k,
        debug: true,
        strategy,
        ..Request::default()
    };
    index.search(&req).unwrap()
}

#[test]
fn maxscore_equals_exhaustive() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("maxscore");
    let queries = synth::text(&dir.join("corpus.jsonl"), 20_000, 200, 7).unwrap();
    engine::build(
        &dir.join("corpus.jsonl"),
        &dir.join("index"),
        &BuildOptions::default(),
    )
    .unwrap();
    let index = Index::open(&dir.join("index")).unwrap();

    // The synthetic 2 to 4 term queries, plus pairs of them joined into 4 to 8 term queries.
    let short: Vec<String> = fs::read_to_string(queries)
        .unwrap()
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["text"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    let long: Vec<String> = short.chunks(2).map(|p| p.join(" ")).collect();

    let (mut decoded_exhaustive, mut decoded_maxscore) = (0, 0);
    for q in short.iter().chain(&long) {
        for k in [1, 10, 100] {
            let e = keyword(&index, q, k, Strategy::Exhaustive);
            let m = keyword(&index, q, k, Strategy::Maxscore);
            assert_eq!(e.results.len(), m.results.len(), "{q} k={k}");
            for (rank, (a, b)) in e.results.iter().zip(&m.results).enumerate() {
                let rel = (a.score - b.score).abs() / a.score;
                assert!(
                    rel <= 1e-5,
                    "{q} k={k} rank {rank}: {} vs {}",
                    a.score,
                    b.score
                );
            }
            decoded_exhaustive += e.stats.unwrap().postings_decoded.unwrap();
            decoded_maxscore += m.stats.unwrap().postings_decoded.unwrap();
        }
    }
    // How much less is B7's question; here it only has to be less.
    assert!(
        decoded_maxscore < decoded_exhaustive,
        "{decoded_maxscore} vs {decoded_exhaustive}"
    );
}

fn words(spec: &[(&str, usize)]) -> String {
    spec.iter()
        .flat_map(|&(w, n)| vec![w; n])
        .collect::<Vec<_>>()
        .join(" ")
}

/// BM25 as Lucene scores it, written out by hand in f64. `docs` are the kept tokens.
fn lucene_bm25(docs: &[Vec<&str>], query: &[&str]) -> Vec<f64> {
    // Lucene's one-byte norm is exact up to 40 tokens, then keeps 4 significant bits.
    let norm = |len: usize| match len {
        0..=40 => len as f64,
        41 => 40.0,
        100 => 96.0,
        _ => unreachable!("add the SmallFloat value for {len}"),
    };
    let (k1, b) = (0.9, 0.4);
    let n = docs.iter().filter(|d| !d.is_empty()).count() as f64;
    let avgdl = docs.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let terms: BTreeSet<&str> = query.iter().copied().collect();
    docs.iter()
        .map(|d| {
            terms
                .iter()
                .map(|term| {
                    let boost = query.iter().filter(|q| *q == term).count() as f64;
                    let df = docs.iter().filter(|d| d.contains(term)).count() as f64;
                    let tf = d.iter().filter(|t| *t == term).count() as f64;
                    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                    boost * idf * tf / (tf + k1 * (1.0 - b + b * norm(d.len()) / avgdl))
                })
                .sum()
        })
        .collect()
}

#[test]
fn bm25_matches_hand_written_lucene_formula() {
    // (title, text). Covers tf 1 to 15 tiers, the overflow tier (tf 20), a singleton term
    // (delta), a doc of only stopwords (not counted in N), title words, and lengths 41 and
    // 100 where the one-byte norm rounds.
    let docs = [
        ("alpha", "beta gamma".to_string()),
        ("", words(&[("alpha", 3), ("beta", 1), ("filler", 37)])),
        ("", words(&[("alpha", 20), ("gamma", 80)])),
        ("", "beta delta".to_string()),
        ("", "the and of".to_string()),
        ("", words(&[("gamma", 25), ("beta", 5)])),
        ("", words(&[("alpha", 2), ("beta", 2)])),
    ];
    let corpus: Vec<Value> = docs
        .iter()
        .enumerate()
        .map(|(i, (title, text))| json!({ "id": format!("d{i}"), "title": title, "text": text }))
        .collect();
    let index = build("bm25", &corpus);
    let tokens: Vec<Vec<&str>> = docs
        .iter()
        .map(|(title, text)| {
            title
                .split_whitespace()
                .chain(text.split_whitespace())
                .filter(|w| !["the", "and", "of"].contains(w))
                .collect()
        })
        .collect();
    assert!(tokens.iter().any(|d| d.len() == 41) && tokens.iter().any(|d| d.len() == 100));

    let queries = ["alpha beta delta", "alpha alpha gamma", "gamma", "delta"];
    for (query, strategy) in queries
        .iter()
        .flat_map(|q| [(q, Strategy::Exhaustive), (q, Strategy::Maxscore)])
    {
        let expected = lucene_bm25(&tokens, &query.split(' ').collect::<Vec<_>>());
        let res = keyword(&index, query, 10, strategy);
        let matching = expected.iter().filter(|&&s| s > 0.0).count();
        assert_eq!(res.results.len(), matching, "{query}");
        for hit in &res.results {
            let want = expected[hit.id[1..].parse::<usize>().unwrap()];
            let rel = (f64::from(hit.score) - want).abs() / want;
            assert!(
                rel < 1e-5,
                "{query}: {} scored {} want {want}",
                hit.id,
                hit.score
            );
        }
        assert!(res.results.windows(2).all(|w| w[0].score >= w[1].score));
    }
}

#[test]
fn results_carry_external_ids_and_snippets() {
    let long = (0..60)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let index = build(
        "external_ids",
        &[
            json!({ "_id": "beir-7", "title": "Short one", "text": "needle here", "url": "https://a.example/x" }),
            json!({ "id": "fw-42", "text": format!("needle {long}") }),
        ],
    );
    let res = search(&index, "needle", Mode::Keyword, 10).unwrap();
    let ids: Vec<&str> = res.results.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, ["beir-7", "fw-42"]);

    let short = &res.results[0];
    assert_eq!(
        (
            short.title.as_str(),
            short.url.as_str(),
            short.snippet.as_str()
        ),
        ("Short one", "https://a.example/x", "needle here")
    );

    let text = format!("needle {long}");
    let snip = &res.results[1].snippet;
    assert!(
        snip.chars().count() <= 200 && snip.chars().count() > 150,
        "{snip}"
    );
    assert!(text.starts_with(snip.as_str()) && text[snip.len()..].starts_with(' '));

    let json = serde_json::to_value(&res.results[0]).unwrap();
    let keys: BTreeSet<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from(["id", "score", "title", "url", "snippet"])
    );
}

#[test]
fn modes_limits_and_ties() {
    let index = build(
        "modes",
        &[
            json!({ "id": "b", "text": "same words" }),
            json!({ "id": "a", "text": "same words" }),
        ],
    );

    let hybrid = serde_json::to_value(search(&index, "same", Mode::Hybrid, 10).unwrap()).unwrap();
    assert_eq!(hybrid["mode_used"], "keyword");
    assert_eq!(hybrid["fallback_reason"], "no_vectors");
    for t in ["keyword", "docs", "total"] {
        assert!(hybrid["timings_ms"][t].is_f64(), "{t}");
    }
    let keyword = serde_json::to_value(search(&index, "same", Mode::Keyword, 10).unwrap()).unwrap();
    assert!(keyword.get("fallback_reason").is_none());

    let err = search(&index, "same", Mode::Neural, 10)
        .unwrap_err()
        .to_string();
    assert!(err.contains("vectors"), "{err}");

    assert!(search(&index, "same", Mode::Keyword, 1000).is_ok());
    assert!(search(&index, "same", Mode::Keyword, 1001).is_err());

    // Equal scores go to the smaller doc id (corpus order), not the external id.
    let res = search(&index, "same", Mode::Keyword, 10).unwrap();
    assert_eq!(res.results[0].score, res.results[1].score);
    let ids: Vec<&str> = res.results.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, ["b", "a"]);
    assert_eq!(
        search(&index, "same", Mode::Keyword, 1).unwrap().results[0].id,
        "b"
    );
}

#[test]
fn ablation_tier_step_is_the_real_encoding() {
    // Every term in two or more docs, so there are no singletons to inline and the tiers step
    // must take exactly the bytes of the postings files.
    let docs: Vec<Value> = (0..300)
        .map(|i| json!({ "id": format!("d{i}"), "text": format!("common words {} more{}", "shared ".repeat(i % 20 + 1), i % 7) }))
        .collect();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ablation");
    fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("corpus.jsonl");
    fs::write(
        &corpus,
        docs.iter().map(|d| format!("{d}\n")).collect::<String>(),
    )
    .unwrap();
    let report = bench::ablate_postings(&corpus, &dir.join("index")).unwrap();
    let steps = &report.steps;
    assert_eq!(steps.len(), 5);
    assert_eq!(steps[3].footprint.postings, steps[4].footprint.postings);
    assert!(steps.windows(2).all(|w| w[1].total < w[0].total));
}

/// A corpus of `n` synthetic docs with synthetic vectors (`dims` each), built into an index.
fn vector_index(name: &str, n: usize, dims: usize) -> (Index, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    synth::text(&dir.join("corpus.jsonl"), n, 0, 3).unwrap();
    let queries = synth::vectors(&dir.join("base.fbin"), n, 100, dims, 3).unwrap();
    let opts = BuildOptions {
        vectors: Some(dir.join("base.fbin")),
        d_code: dims.min(256),
        ..BuildOptions::default()
    };
    engine::build(&dir.join("corpus.jsonl"), &dir.join("index"), &opts).unwrap();
    let rows = |f: Fbin| (0..f.n).map(|i| f.row(i).to_vec()).collect::<Vec<_>>();
    let base = rows(Fbin::open(&dir.join("base.fbin")).unwrap());
    let queries = rows(Fbin::open(&queries).unwrap());
    (Index::open(&dir.join("index")).unwrap(), base, queries)
}

fn neural(index: &Index, q: &[f32], k: usize) -> anyhow::Result<Response> {
    index.search(&Request {
        mode: Mode::Neural,
        k,
        vector: Some(q.to_vec()),
        ..Request::default()
    })
}

#[test]
fn neural_recall_against_exact_search() {
    // 512-dim vectors: 256-dim codes for the first stage, all 512 dims to rerank. 20,000 docs
    // make 256 clusters, of which the default probes 32.
    let (index, base, queries) = vector_index("neural_recall", 20_000, 512);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("neural_recall");
    let (base_f, query_f) = (
        Fbin::open(&dir.join("base.fbin")).unwrap(),
        Fbin::open(&dir.join("queries.fbin")).unwrap(),
    );
    let gt = kmeans::ground_truth(&base_f, &query_f, 512, 10);
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut found = 0;
    for (q, gt) in queries.iter().zip(&gt) {
        // The chunked, parallel ground truth equals a plain sort.
        let mut exact: Vec<(usize, f32)> = base.iter().map(|v| dot(v, q)).enumerate().collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        assert_eq!(
            gt,
            &exact[..10]
                .iter()
                .map(|&(i, _)| i as u32)
                .collect::<Vec<_>>()
        );
        let truth: BTreeSet<String> = gt.iter().map(|i| format!("synth-{i}")).collect();
        let res = neural(&index, q, 10).unwrap();
        assert_eq!(res.results.len(), 10);
        found += res.results.iter().filter(|h| truth.contains(&h.id)).count();
    }
    let recall = found as f64 / (10 * queries.len()) as f64;
    assert!(recall > 0.9, "recall@10 {recall}");
}

#[test]
fn neural_modes_and_vector_checks() {
    let (index, _, queries) = vector_index("neural_modes", 500, 128);
    let res = serde_json::to_value(neural(&index, &queries[0], 5).unwrap()).unwrap();
    assert_eq!(res["mode_used"], "neural");
    for t in ["route", "scan", "rerank", "docs", "total"] {
        assert!(res["timings_ms"][t].is_f64(), "{t}");
    }
    assert!(res["timings_ms"].get("keyword").is_none());

    // No query vector and no embedder: neural fails, hybrid falls back to keyword.
    let err = index
        .search(&Request {
            mode: Mode::Neural,
            ..Request::default()
        })
        .unwrap_err();
    assert!(err.to_string().contains("query vector"), "{err}");
    let hybrid = index
        .search(&Request {
            query: "w1".into(),
            ..Request::default()
        })
        .unwrap();
    assert_eq!(
        serde_json::to_value(&hybrid).unwrap()["fallback_reason"],
        "no_embedder"
    );
    assert!(
        neural(&index, &queries[0][..64], 5).is_err(),
        "wrong query dims"
    );

    // Every doc has a vector, or the build fails.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("neural_modes");
    synth::vectors(&dir.join("short.fbin"), 499, 1, 128, 3).unwrap();
    let opts = BuildOptions {
        vectors: Some(dir.join("short.fbin")),
        ..BuildOptions::default()
    };
    let err = engine::build(&dir.join("corpus.jsonl"), &dir.join("bad"), &opts).unwrap_err();
    assert!(err.to_string().contains("499 rows"), "{err}");
    for d_code in [100, 2048] {
        let opts = BuildOptions {
            vectors: Some(dir.join("base.fbin")),
            d_code,
            ..BuildOptions::default()
        };
        assert!(
            engine::build(&dir.join("corpus.jsonl"), &dir.join("bad"), &opts).is_err(),
            "d_code {d_code}"
        );
    }
}

#[test]
fn every_kernel_returns_identical_results() {
    let (index, _, queries) = vector_index("kernels", 3_000, 256);
    for q in &queries {
        let run = |kernel| {
            let req = Request {
                mode: Mode::Neural,
                vector: Some(q.clone()),
                kernel: Some(kernel),
                debug: true,
                ..Request::default()
            };
            let res = index.search(&req).unwrap();
            assert_eq!(res.stats.unwrap().kernel, Some(kernel));
            res.results
                .into_iter()
                .map(|h| (h.id, h.score.to_bits()))
                .collect::<Vec<_>>()
        };
        let want = run(Kernel::Scalar);
        for kernel in Kernel::available() {
            assert_eq!(run(kernel), want, "{kernel:?}");
        }
    }
}

#[test]
fn probing_every_cluster_equals_one_cluster() {
    let (clustered, _, queries) = vector_index("all_clusters", 5_000, 256);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("all_clusters");
    let opts = BuildOptions {
        vectors: Some(dir.join("base.fbin")),
        nlist: Some(1),
        ..BuildOptions::default()
    };
    engine::build(&dir.join("corpus.jsonl"), &dir.join("one"), &opts).unwrap();
    let one = Index::open(&dir.join("one")).unwrap();
    let run = |index: &Index, q: &[f32], nprobe| {
        let req = Request {
            mode: Mode::Neural,
            k: 20,
            vector: Some(q.to_vec()),
            nprobe: Some(nprobe),
            debug: true,
            ..Request::default()
        };
        index.search(&req).unwrap()
    };
    for q in &queries {
        let all = run(&clustered, q, 1 << 20);
        let single = run(&one, q, 1);
        let ids = |r: &Response| {
            r.results
                .iter()
                .map(|h| (h.id.clone(), h.score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&all), ids(&single));
        let (a, b) = (all.stats.unwrap(), single.stats.unwrap());
        assert_eq!(a.clusters_probed, Some(128)); // default for 5,000 docs: 2 x 70.7, rounded
        assert_eq!(b.clusters_probed, Some(1));
        assert!(
            a.codes_scanned >= b.codes_scanned,
            "more clusters, more padding"
        );
    }
}

#[test]
fn rrf_matches_hand_computed_scores() {
    let kw = [(5, 9.0), (3, 8.0), (9, 7.0)];
    let nn = [(3, 0.9), (7, 0.8)];
    let ranks = |keyword, neural| Ranks { keyword, neural };
    assert_eq!(
        engine::rrf(&kw, &nn, 10),
        [
            (3, 1.0 / 62.0 + 1.0 / 61.0, ranks(Some(2), Some(1))),
            (5, 1.0 / 61.0, ranks(Some(1), None)),
            (7, 1.0 / 62.0, ranks(None, Some(2))),
            (9, 1.0 / 63.0, ranks(Some(3), None)),
        ]
    );
    assert_eq!(engine::rrf(&kw, &nn, 2).len(), 2);
    // Equal scores go to the smaller doc id, whichever list it came from.
    let tie: Vec<u32> = engine::rrf(&[(8, 1.0)], &[(2, 1.0)], 10)
        .iter()
        .map(|f| f.0)
        .collect();
    assert_eq!(tie, [2, 8]);
}

#[test]
fn hybrid_is_rrf_over_both_retrievers() {
    let (index, _, queries) = vector_index("hybrid", 3_000, 128);
    let doc = |id: &str| id.trim_start_matches("synth-").parse::<u32>().unwrap();
    for (i, q) in queries.iter().take(20).enumerate() {
        let run = |mode, k| {
            index
                .search(&Request {
                    query: format!("w{} w{} w0", i + 1, 2 * i + 7),
                    mode,
                    k,
                    vector: Some(q.clone()),
                    debug: true,
                    ..Request::default()
                })
                .unwrap()
        };
        let list = |r: &Response| -> Vec<(u32, f32)> {
            r.results.iter().map(|h| (doc(&h.id), h.score)).collect()
        };
        // Hybrid at k = 10 fuses the top 100 of each.
        let (kw, nn) = (run(Mode::Keyword, 100), run(Mode::Neural, 100));
        assert_eq!(kw.results.len(), 100);
        let want = engine::rrf(&list(&kw), &list(&nn), 10);
        let hybrid = run(Mode::Hybrid, 10);
        let got: Vec<_> = hybrid
            .results
            .iter()
            .map(|h| (doc(&h.id), h.score, h.ranks.unwrap()))
            .collect();
        let want: Vec<_> = want.into_iter().map(|(d, s, r)| (d, s as f32, r)).collect();
        assert_eq!(got, want);

        let json = serde_json::to_value(&hybrid).unwrap();
        assert_eq!(json["mode_used"], "hybrid");
        assert!(json.get("fallback_reason").is_none());
        for t in [
            "keyword", "route", "scan", "rerank", "fuse", "docs", "total",
        ] {
            assert!(json["timings_ms"][t].is_f64(), "{t}");
        }
        for s in ["postings_decoded", "clusters_probed", "codes_scanned"] {
            assert!(json["stats"][s].is_u64(), "{s}");
        }
        assert!(json["results"][0]["ranks"].is_object());
        // Single modes carry no ranks.
        assert!(
            kw.results
                .iter()
                .chain(&nn.results)
                .all(|h| h.ranks.is_none())
        );
    }
}

/// Doc i: group `g{i % 1000}` unless i % 13 == 0, crawl date month 1 + i % 12 (as a UTC
/// timestamp) unless i % 17 == 0.
fn group_of(i: usize) -> Option<usize> {
    (!i.is_multiple_of(13)).then_some(i % 1000)
}
fn month_of(i: usize) -> Option<usize> {
    (!i.is_multiple_of(17)).then_some(1 + i % 12)
}

#[test]
fn filters_never_leak() {
    let n = 50_000;
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("filters");
    synth::text(&dir.join("plain.jsonl"), n, 0, 5).unwrap();
    let queries = synth::vectors(&dir.join("base.fbin"), n, 10, 128, 5).unwrap();
    let corpus: String = fs::read_to_string(dir.join("plain.jsonl"))
        .unwrap()
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let mut d: Value = serde_json::from_str(line).unwrap();
            if let Some(g) = group_of(i) {
                d["group"] = json!(format!("g{g}"));
            }
            if let Some(m) = month_of(i) {
                d["crawl_date"] = json!(format!("2023-{m:02}-{:02}T12:00:00Z", 1 + i % 28));
            }
            format!("{d}\n")
        })
        .collect();
    fs::write(dir.join("corpus.jsonl"), corpus).unwrap();
    let opts = BuildOptions {
        vectors: Some(dir.join("base.fbin")),
        d_code: 128,
        keyword_fields: vec!["group".into()],
        date_field: Some("crawl_date".into()),
        ..BuildOptions::default()
    };
    engine::build(&dir.join("corpus.jsonl"), &dir.join("index"), &opts).unwrap();
    let index = Index::open(&dir.join("index")).unwrap();
    let qs = Fbin::open(&queries).unwrap();
    let doc = |id: &str| id.trim_start_matches("synth-").parse::<usize>().unwrap();

    // 0.1%, 1%, 10% and 50% of groups; under 20,000 docs is scored exactly, over is probed.
    for groups in [1, 10, 100, 500] {
        let filters: Filters = [(
            "group".to_string(),
            Filter::Any((0..groups).map(|g| format!("g{g}")).collect()),
        )]
        .into();
        let passes = |i: usize| group_of(i).is_some_and(|g| g < groups);
        let expected = (0..n).filter(|&i| passes(i)).count() as u64;
        for qi in 0..qs.n {
            let req = Request {
                mode: Mode::Neural,
                k: 50,
                vector: Some(qs.row(qi).to_vec()),
                filters: filters.clone(),
                debug: true,
                ..Request::default()
            };
            let res = index.search(&req).unwrap();
            assert!(
                res.results.iter().all(|h| passes(doc(&h.id))),
                "neural, {groups} groups"
            );
            assert_eq!(res.results.len() as u64, expected.min(50));
            let stats = res.stats.unwrap();
            assert_eq!(stats.docs_allowed, Some(expected));
            assert_eq!(
                stats.clusters_probed == Some(0),
                expected <= 20_000,
                "{groups} groups: path"
            );
            let hybrid = index
                .search(&Request {
                    query: "w1 w2 w3".into(),
                    mode: Mode::Hybrid,
                    ..req
                })
                .unwrap();
            assert_eq!(hybrid.mode_used, Mode::Hybrid);
            assert!(
                hybrid.results.iter().all(|h| passes(doc(&h.id))),
                "hybrid, {groups} groups"
            );
            assert_eq!(hybrid.results.len() as u64, expected.min(50));
        }
        for q in ["w1 w2 w3", "w0 w5", "w1000 w1001 w42"] {
            let run = |strategy| {
                let req = Request {
                    query: q.into(),
                    mode: Mode::Keyword,
                    k: 100,
                    strategy,
                    filters: filters.clone(),
                    ..Request::default()
                };
                index.search(&req).unwrap().results
            };
            let (e, m) = (run(Strategy::Exhaustive), run(Strategy::Maxscore));
            assert!(
                e.iter().chain(&m).all(|h| passes(doc(&h.id))),
                "keyword, {groups} groups"
            );
            assert_eq!(e.len(), m.len());
            for (a, b) in e.iter().zip(&m) {
                assert!(
                    (a.score - b.score).abs() <= 1e-5 * a.score,
                    "{q}: {} vs {}",
                    a.score,
                    b.score
                );
            }
        }
    }

    // A crawl date range, inclusive, and every field must hold.
    let filters: Filters = [
        (
            "crawl_date".to_string(),
            Filter::Range {
                gte: Some("2023-03-01".into()),
                lte: Some("2023-05-28".into()),
            },
        ),
        (
            "group".to_string(),
            Filter::Any((0..500).map(|g| format!("g{g}")).collect()),
        ),
    ]
    .into();
    let passes = |i: usize| {
        month_of(i).is_some_and(|m| (3..=5).contains(&m)) && group_of(i).is_some_and(|g| g < 500)
    };
    let req = Request {
        query: "w1 w2".into(),
        mode: Mode::Keyword,
        k: 1000,
        filters: filters.clone(),
        debug: true,
        ..Request::default()
    };
    let res = index.search(&req).unwrap();
    assert!(!res.results.is_empty() && res.results.iter().all(|h| passes(doc(&h.id))));
    assert_eq!(
        res.stats.unwrap().docs_allowed,
        Some((0..n).filter(|&i| passes(i)).count() as u64)
    );

    // Bad requests and bad corpora fail clearly.
    let bad = |filters: Filters| {
        index.search(&Request {
            query: "w1".into(),
            mode: Mode::Keyword,
            filters,
            ..Request::default()
        })
    };
    assert!(
        bad([("lang".into(), Filter::Any(vec!["en".into()]))].into()).is_err(),
        "undeclared field"
    );
    assert!(
        bad([(
            "group".into(),
            Filter::Range {
                gte: None,
                lte: None
            }
        )]
        .into())
        .is_err(),
        "range on a keyword field"
    );
    assert!(
        bad([(
            "crawl_date".into(),
            Filter::Range {
                gte: Some("2023-3-1".into()),
                lte: None
            }
        )]
        .into())
        .is_err(),
        "bad date"
    );
    fs::write(
        dir.join("numeric.jsonl"),
        "{\"id\":\"a\",\"text\":\"x\",\"group\":3}\n",
    )
    .unwrap();
    let opts = BuildOptions {
        keyword_fields: vec!["group".into()],
        ..BuildOptions::default()
    };
    let err = engine::build(&dir.join("numeric.jsonl"), &dir.join("numeric"), &opts).unwrap_err();
    assert!(format!("{err:#}").contains("no numeric filters"), "{err:#}");
}

/// Doc i of n: a crawl date in block `i * 40 / n`, so dates come in runs along doc id order (as
/// in FineWeb) on odd days only, leaving gaps; none when i % 17 == 0.
fn clustered_day(i: usize, n: usize) -> Option<String> {
    let block = i * 40 / n;
    (!i.is_multiple_of(17))
        .then(|| format!("2023-{:02}-{:02}", 1 + block / 14, 1 + 2 * (block % 14)))
}

#[test]
fn clustered_date_ranges() {
    let n = 50_000;
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("clustered-dates");
    synth::text(&dir.join("plain.jsonl"), n, 0, 7).unwrap();
    let corpus: String = fs::read_to_string(dir.join("plain.jsonl"))
        .unwrap()
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let mut d: Value = serde_json::from_str(line).unwrap();
            if let Some(day) = clustered_day(i, n) {
                d["crawl_date"] = json!(format!("{day}T12:00:00Z"));
            }
            format!("{d}\n")
        })
        .collect();
    fs::write(dir.join("corpus.jsonl"), corpus).unwrap();
    let opts = BuildOptions {
        date_field: Some("crawl_date".into()),
        ..BuildOptions::default()
    };
    engine::build(&dir.join("corpus.jsonl"), &dir.join("index"), &opts).unwrap();
    let index = Index::open(&dir.join("index")).unwrap();
    let filter_index = FilterIndex::open(&dir.join("index")).unwrap();
    let range = |gte: Option<&str>, lte: Option<&str>| -> Filters {
        [(
            "crawl_date".to_string(),
            Filter::Range {
                gte: gte.map(Into::into),
                lte: lte.map(Into::into),
            },
        )]
        .into()
    };

    // Every range, against the docs whose day falls in it: ends on days with docs, on gaps,
    // outside all days, open, and reversed.
    let ends = [
        None,
        Some("2022-12-31"),
        Some("2023-01-01"),
        Some("2023-01-02"),
        Some("2023-01-27"),
        Some("2023-02-14"),
        Some("2023-02-15"),
        Some("2023-03-23"),
        Some("2023-03-24"),
        Some("2023-04-01"),
    ];
    for gte in ends {
        for lte in ends {
            let got = filter_index.allowed(&range(gte, lte)).unwrap().unwrap();
            let want: Vec<u32> = (0..n)
                .filter(|&i| {
                    clustered_day(i, n).is_some_and(|d| {
                        gte.is_none_or(|g| d.as_str() >= g) && lte.is_none_or(|l| d.as_str() <= l)
                    })
                })
                .map(|i| i as u32)
                .collect();
            assert_eq!(got.iter().collect::<Vec<_>>(), want, "{gte:?}..{lte:?}");
        }
    }

    // MaxScore matches exhaustive under filters that are runs of doc ids, and a narrow run
    // decodes fewer postings than no filter: cursors jump to it instead of scoring their way.
    let doc = |id: &str| id.trim_start_matches("synth-").parse::<usize>().unwrap();
    let filters = [
        range(Some("2023-02-03"), Some("2023-02-03")),
        range(Some("2023-01-01"), Some("2023-01-05")),
        range(Some("2023-03-21"), None),
        range(Some("2023-01-09"), Some("2023-03-09")),
    ];
    for q in ["w1 w2 w3", "w0 w5", "w1000 w1001 w42"] {
        for k in [1, 10, 100] {
            let unfiltered = keyword(&index, q, k, Strategy::Maxscore);
            for f in &filters {
                let run = |strategy| {
                    let req = Request {
                        query: q.into(),
                        mode: Mode::Keyword,
                        k,
                        strategy,
                        filters: f.clone(),
                        debug: true,
                        ..Request::default()
                    };
                    index.search(&req).unwrap()
                };
                let (e, m) = (run(Strategy::Exhaustive), run(Strategy::Maxscore));
                let allowed = filter_index.allowed(f).unwrap().unwrap();
                assert!(
                    e.results
                        .iter()
                        .chain(&m.results)
                        .all(|h| allowed.contains(doc(&h.id) as u32)),
                    "{q} k={k} {f:?}: leaked"
                );
                assert_eq!(e.results.len(), m.results.len(), "{q} k={k} {f:?}");
                for (a, b) in e.results.iter().zip(&m.results) {
                    assert!(
                        (a.score - b.score).abs() <= 1e-5 * a.score,
                        "{q} k={k} {f:?}: {} vs {}",
                        a.score,
                        b.score
                    );
                }
                if allowed.len() < n as u64 / 20 {
                    let decoded =
                        |r: &Response| r.stats.as_ref().unwrap().postings_decoded.unwrap();
                    assert!(
                        decoded(&m) < decoded(&unfiltered),
                        "{q} k={k} {f:?}: {} postings decoded, {} with no filter",
                        decoded(&m),
                        decoded(&unfiltered)
                    );
                }
            }
        }
    }
}
