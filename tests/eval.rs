use serde::Deserialize;
use std::collections::HashMap;
use telusur::eval::{Qrels, Run, evaluate, query_metrics};

#[derive(Deserialize)]
struct Fixture {
    qrels: Qrels,
    run: HashMap<String, HashMap<String, f64>>,
    expected: HashMap<String, HashMap<String, f64>>,
}

/// tests/data/pytrec_eval.json holds a run with ties and graded relevance, plus pytrec_eval's
/// per-query results for it (made by scripts/pytrec_eval_fixture.py).
#[test]
fn metrics_match_pytrec_eval() {
    let f: Fixture = serde_json::from_str(include_str!("data/pytrec_eval.json")).unwrap();
    let run: Run = f
        .run
        .into_iter()
        .map(|(q, docs)| (q, docs.into_iter().collect()))
        .collect();

    for (q, want) in &f.expected {
        let (ndcg, recall) = query_metrics(&f.qrels[q], &run[q]);
        assert!(
            (ndcg - want["ndcg_cut_10"]).abs() < 1e-12,
            "{q} ndcg {ndcg} want {}",
            want["ndcg_cut_10"]
        );
        assert!(
            (recall - want["recall_100"]).abs() < 1e-12,
            "{q} recall {recall} want {}",
            want["recall_100"]
        );
    }

    // trec_eval -c: the mean covers every judged query; q3 was never searched and scores 0,
    // q5 was searched but not judged and is ignored.
    let n = f.qrels.len() as f64;
    let mean = |m: &str| f.expected.values().map(|v| v[m]).sum::<f64>() / n;
    let (ndcg, recall) = evaluate(&f.qrels, &run);
    assert!((ndcg - mean("ndcg_cut_10")).abs() < 1e-12);
    assert!((recall - mean("recall_100")).abs() < 1e-12);
}
