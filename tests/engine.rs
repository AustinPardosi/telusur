use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf};
use telusur::engine::{self, Index, Mode, Request, Response};

fn build(name: &str, docs: &[Value]) -> Index {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("corpus.jsonl");
    fs::write(
        &corpus,
        docs.iter().map(|d| format!("{d}\n")).collect::<String>(),
    )
    .unwrap();
    engine::build(&corpus, &dir.join("index")).unwrap();
    Index::open(&dir.join("index")).unwrap()
}

fn search(index: &Index, query: &str, mode: Mode, k: usize) -> anyhow::Result<Response> {
    index.search(&Request {
        query: query.into(),
        mode,
        k,
    })
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

    for query in ["alpha beta delta", "alpha alpha gamma", "gamma", "delta"] {
        let expected = lucene_bm25(&tokens, &query.split(' ').collect::<Vec<_>>());
        let res = search(&index, query, Mode::Keyword, 10).unwrap();
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
