//! The HTTP contract: JSON in, JSON out, over a real socket, with a fake embedder.

use axum::{Json, Router, http::StatusCode, routing::post};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
use telusur::{
    engine::{self, BuildOptions},
    server, synth,
    util::Fbin,
};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

/// Status and JSON body of POST /search.
async fn search(base: &str, body: &str) -> (u16, Value) {
    let res = reqwest::Client::new()
        .post(format!("{base}/search"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    (res.status().as_u16(), res.json().await.unwrap())
}

async fn get(base: &str, path: &str) -> Value {
    reqwest::get(format!("{base}{path}"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn http_contract() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("server");
    synth::text(&dir.join("corpus.jsonl"), 500, 0, 7).unwrap();
    let queries = synth::vectors(&dir.join("base.fbin"), 500, 1, 128, 7).unwrap();
    let opts = BuildOptions {
        vectors: Some(dir.join("base.fbin")),
        d_code: 128,
        ..BuildOptions::default()
    };
    let index = dir.join("index");
    engine::build(&dir.join("corpus.jsonl"), &index, &opts).unwrap();
    let q = Fbin::open(&queries).unwrap().row(0).to_vec();

    let vector = |v: Vec<f32>| post(move || async move { Json(json!({ "vector": v })) });
    let fake = Router::new()
        .route("/ok/embed", vector(q.clone()))
        .route("/short/embed", vector(q[..3].to_vec()))
        .route(
            "/bad/embed",
            post(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
        )
        .route(
            "/slow/embed",
            post(|| async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                StatusCode::OK
            }),
        );
    let embedder = spawn(fake).await;
    let serve = |path: Option<&str>| {
        let url = path.map(|p| format!("{embedder}/{p}"));
        spawn(server::router(&index, url).unwrap())
    };
    let ok = serve(Some("ok")).await;

    // Hybrid with an embedder: both retrievers, ranks, every stage timed, stats only on debug.
    let (status, res) = search(&ok, r#"{"query": "w1 w2", "k": 5}"#).await;
    assert_eq!(status, 200, "{res}");
    assert_eq!(res["mode_used"], "hybrid");
    assert!(res.get("fallback_reason").is_none());
    assert!(res.get("stats").is_none());
    let results = res["results"].as_array().unwrap();
    assert_eq!(results.len(), 5);
    for r in results {
        assert!(
            r["id"].as_str().unwrap().starts_with("synth-"),
            "external ids"
        );
        for f in ["score", "title", "url", "snippet", "ranks"] {
            assert!(r.get(f).is_some(), "{f}");
        }
    }
    let stages = [
        "embed", "route", "scan", "rerank", "keyword", "fuse", "docs", "total",
    ];
    for t in stages {
        assert!(res["timings_ms"][t].is_f64(), "{t}");
    }
    let (_, res) = search(&ok, r#"{"query": "w1 w2", "debug": true}"#).await;
    assert!(res["stats"]["postings_decoded"].is_u64());
    assert!(res["stats"]["clusters_probed"].is_u64());

    let (status, res) = search(&ok, r#"{"query": "w1", "mode": "neural"}"#).await;
    assert_eq!((status, &res["mode_used"]), (200, &json!("neural")));
    assert!(res["results"][0].get("ranks").is_none());
    let (_, res) = search(&ok, r#"{"query": "w1", "mode": "keyword"}"#).await;
    assert_eq!(res["mode_used"], "keyword");
    assert!(
        res["timings_ms"].get("embed").is_none(),
        "keyword never embeds"
    );

    // Fallbacks: hybrid answers with keyword and says why; neural gets a 503.
    for (path, reason) in [
        (None, "no_embedder"),
        (Some("bad"), "embedder_error"),
        (Some("short"), "embedder_error"),
        (Some("slow"), "embedder_timeout"),
    ] {
        let base = serve(path).await;
        let (status, res) = search(&base, r#"{"query": "w1 w2"}"#).await;
        assert_eq!(status, 200, "{path:?}: {res}");
        assert_eq!(res["mode_used"], "keyword", "{path:?}");
        assert_eq!(res["fallback_reason"], reason, "{path:?}");
        assert!(!res["results"].as_array().unwrap().is_empty());
        if reason == "embedder_timeout" {
            assert!(res["timings_ms"]["embed"].as_f64().unwrap() >= 999.0);
        }
        let (status, res) = search(&base, r#"{"query": "w1", "mode": "neural"}"#).await;
        assert_eq!(status, 503, "{path:?}: {res}");
        assert!(res["error"].is_string());
    }

    // A vector in the request needs no embedder.
    let none = serve(None).await;
    let body = json!({ "mode": "neural", "vector": q }).to_string();
    let (status, res) = search(&none, &body).await;
    assert_eq!((status, &res["mode_used"]), (200, &json!("neural")));

    // Bad requests are 400s with a JSON error.
    for body in [
        r#"{"query": "w1", "k": 1001}"#,
        r#"{"query": "w1""#,
        r#"{"q": "w1"}"#,
        r#"{"query": "w1", "mode": "fuzzy"}"#,
        r#"{"query": "w1", "filters": {"lang": ["en"]}}"#,
        r#"{"mode": "neural", "vector": [1.0, 2.0]}"#,
    ] {
        let (status, res) = search(&none, body).await;
        assert_eq!(status, 400, "{body}: {res}");
        assert!(res["error"].is_string(), "{body}");
    }
    assert_eq!(search(&ok, r#"{"query": "w1", "k": 1000}"#).await.0, 200);

    assert_eq!(get(&ok, "/health").await, json!({ "status": "ok" }));
    let stats = get(&ok, "/stats").await;
    assert_eq!(stats["docs"], 500);
    assert_eq!(stats["vector_dims"], 128);
    assert!(stats["kernel"].is_string());
    assert!(stats["files"]["codes.bin"].as_u64().unwrap() > 0);
    assert!(stats["index_bytes"].as_u64().unwrap() > 0);
}
