//! Synthetic data for tests and quick speed checks. Never quote it as quality results.

use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::index::sample};
use rand_distr::{Distribution, Zipf};
use serde_json::json;
use std::{
    fmt::Write as _,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

/// Background words are `w0..w49999`, Zipfian by rank, so `w0` is the most common.
const VOCAB: usize = 50_000;
const TOPICS: usize = 100;
/// Topic t owns `w(1000 + 100 t)` to `w(1099 + 100 t)`: mid-frequency words, common inside it.
const TOPIC_WORDS: usize = 100;
const TOPIC_BASE: usize = 1000;
/// Share of a doc's tokens drawn from its topic.
const TOPICAL: f64 = 0.3;

fn topic_word(topic: usize, i: usize) -> usize {
    TOPIC_BASE + topic * TOPIC_WORDS + i
}

/// Writes `docs` docs to `out` and `queries` 2 to 4 term queries to `queries.jsonl` next to it,
/// seeded and deterministic. Returns the queries path.
pub fn text(out: &Path, docs: usize, queries: usize, seed: u64) -> Result<PathBuf> {
    let mut rng = StdRng::seed_from_u64(seed);
    let background = Zipf::new(VOCAB as f64, 1.0)?;
    let topical = Zipf::new(TOPIC_WORDS as f64, 1.0)?;
    let rank = |z: &Zipf<f64>, rng: &mut StdRng| z.sample(rng) as usize - 1;

    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(out)?);
    let mut title = String::new();
    let mut text = String::new();
    for i in 0..docs {
        let topic = rng.random_range(0..TOPICS);
        title.clear();
        text.clear();
        for _ in 0..rng.random_range(3..=8) {
            write!(title, "w{} ", topic_word(topic, rank(&topical, &mut rng)))?;
        }
        for _ in 0..rng.random_range(20..=300) {
            let word = if rng.random_bool(TOPICAL) {
                topic_word(topic, rank(&topical, &mut rng))
            } else {
                rank(&background, &mut rng)
            };
            write!(text, "w{word} ")?;
        }
        let doc = json!({
            "id": format!("synth-{i}"),
            "title": title.trim_end(),
            "text": text.trim_end(),
            "url": format!("https://synth.example/{topic}/{i}"),
        });
        writeln!(w, "{doc}")?;
    }
    w.flush()?;

    // One common word plus 1 to 3 distinct words from one topic.
    let path = out.with_file_name("queries.jsonl");
    let mut w = BufWriter::new(File::create(&path)?);
    for i in 0..queries {
        let topic = rng.random_range(0..TOPICS);
        let mut q = format!("w{}", rng.random_range(0..100));
        let n = rng.random_range(1..=3);
        for t in sample(&mut rng, TOPIC_WORDS, n) {
            write!(q, " w{}", topic_word(topic, t))?;
        }
        writeln!(w, "{}", json!({ "_id": format!("q{i}"), "text": q }))?;
    }
    w.flush()?;
    Ok(path)
}
