//! Retrieval quality the way Anserini's BEIR regressions measure it: keyword search for the
//! top 1000, results whose doc id equals the query id dropped (`-removeQuery`), scores
//! rounded to 6 decimals as Anserini's run files write them, then trec_eval's nDCG@10 and
//! Recall@100 averaged over every query in the qrels (`trec_eval -c`).

use crate::{
    engine::{Index, MAX_K, Mode, Request},
    util::hardware,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

/// Query id to (external doc id to relevance grade).
pub type Qrels = HashMap<String, HashMap<String, i32>>;
/// Query id to (external doc id, score), in any order.
pub type Run = HashMap<String, Vec<(String, f64)>>;

/// nDCG@10 and Recall@100 for one query, as trec_eval computes them: ranked by score, ties to
/// the larger doc id; the gain is the grade when it is above 0; relevant means grade >= 1.
pub fn query_metrics(judged: &HashMap<String, i32>, run: &[(String, f64)]) -> (f64, f64) {
    let mut ranked: Vec<&(String, f64)> = run.iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    let grade = |doc: &str| judged.get(doc).copied().unwrap_or(0).max(0);
    let dcg = |gains: &mut dyn Iterator<Item = i32>| -> f64 {
        gains
            .take(10)
            .enumerate()
            .map(|(i, g)| f64::from(g) / (i as f64 + 2.0).log2())
            .sum()
    };

    let mut ideal: Vec<i32> = judged.values().copied().filter(|&g| g > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg = dcg(&mut ideal.iter().copied());
    let ndcg = if idcg > 0.0 {
        dcg(&mut ranked.iter().map(|(d, _)| grade(d))) / idcg
    } else {
        0.0
    };

    let found = ranked
        .iter()
        .take(100)
        .filter(|(d, _)| grade(d) > 0)
        .count();
    let recall = if ideal.is_empty() {
        0.0
    } else {
        found as f64 / ideal.len() as f64
    };
    (ndcg, recall)
}

/// Mean nDCG@10 and Recall@100 over every query in `qrels`; a query with no run scores 0.
pub fn evaluate(qrels: &Qrels, run: &Run) -> (f64, f64) {
    let (ndcg, recall) = qrels
        .iter()
        .map(|(q, judged)| run.get(q).map_or((0.0, 0.0), |r| query_metrics(judged, r)))
        .fold((0.0, 0.0), |(a, b), (n, r)| (a + n, b + r));
    let n = qrels.len().max(1) as f64;
    (ndcg / n, recall / n)
}

#[derive(Serialize)]
pub struct Report {
    pub dataset: String,
    pub mode: Mode,
    pub queries: usize,
    pub ndcg_at_10: f64,
    pub recall_at_100: f64,
    pub seconds: f64,
    pub machine: String,
}

#[derive(Deserialize)]
struct Query {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// Runs the test queries of the BEIR dataset in `data` (corpus.jsonl, queries.jsonl,
/// qrels/test.tsv) against `index`. Optionally writes the run in TREC format to `run_out`.
pub fn beir(index: &Index, data: &Path, run_out: Option<&Path>) -> Result<Report> {
    let mut qrels = Qrels::new();
    let tsv = data.join("qrels/test.tsv");
    for line in fs::read_to_string(&tsv)
        .with_context(|| format!("read {}", tsv.display()))?
        .lines()
        .skip(1)
    {
        let mut f = line.split('\t');
        let (Some(q), Some(d), Some(g)) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        qrels
            .entry(q.into())
            .or_default()
            .insert(d.into(), g.trim().parse()?);
    }
    let mut queries = BTreeMap::new();
    for line in fs::read_to_string(data.join("queries.jsonl"))?.lines() {
        let q: Query = serde_json::from_str(line)?;
        if qrels.contains_key(&q.id) {
            queries.insert(q.id, q.text);
        }
    }

    let start = Instant::now();
    let mut run = Run::new();
    let mut out = run_out.map(File::create).transpose()?.map(BufWriter::new);
    for (qid, text) in &queries {
        let res = index.search(&Request {
            query: text.clone(),
            mode: Mode::Keyword,
            k: MAX_K,
        })?;
        let hits: Vec<(String, f64)> = res
            .results
            .into_iter()
            .filter(|h| h.id != *qid)
            .map(|h| (h.id, (f64::from(h.score) * 1e6).round() / 1e6))
            .collect();
        if let Some(w) = &mut out {
            for (rank, (doc, score)) in hits.iter().enumerate() {
                writeln!(w, "{qid} Q0 {doc} {} {score:.6} telusur", rank + 1)?;
            }
        }
        run.insert(qid.clone(), hits);
    }
    if let Some(mut w) = out {
        w.flush()?;
    }
    let (ndcg_at_10, recall_at_100) = evaluate(&qrels, &run);

    Ok(Report {
        dataset: data
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into(),
        mode: Mode::Keyword,
        queries: qrels.len(),
        ndcg_at_10,
        recall_at_100,
        seconds: start.elapsed().as_secs_f64(),
        machine: hardware(),
    })
}
