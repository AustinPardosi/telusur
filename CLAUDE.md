# CLAUDE.md

Guide for Claude Code working in this repo. Read [`PRD.md`](PRD.md) before starting any milestone. The PRD is the spec; this file is how to work.

## What this is

`telusur` is a hybrid search engine in Rust: a binary-quantized Matryoshka IVF vector index with SIMD lookup-table scoring, a compressed BM25 index with MaxScore over tf tiers, and RRF fusion. It rebuilds designs from Exa's engineering blog. It is a personal project. Correctness and honest measurement matter more than features.

## Ground rules

1. **Tests before fast paths.** Every optimized path has a slow, obviously correct reference and a test that compares them:
   - every SIMD kernel == scalar kernel, bit for bit, for dims 64/128/256/512/1024, with a ragged last block
   - MaxScore BM25 == exhaustive BM25 on top-k scores within 1e-5 relative (compare scores, not ids, because of ties; f32 sums differ by order), for k in {1, 10, 100}, with and without filters
   - BM25 scores == a hand-written reference formula on a tiny corpus
   - filtered search never returns a doc outside the filter
2. **Never invent benchmark numbers.** README numbers come only from files in `bench-results/` produced by a command. If a number is not measured yet, leave `TBD`.
3. **Synthetic data is for tests and speed checks only.** Never present it as quality results.
4. `unsafe` **only in kernel code and mmap helpers**, each block with a `// SAFETY:` comment saying why the pointer math is in bounds.
5. **Ask before** adding a dependency not listed in PRD section 10, changing an on-disk format, or changing a public CLI flag.
6. **Prose style** (README, docs, comments, commit messages): plain, short, conversational. Never use em dashes. Use commas, colons, parentheses or separate sentences.

## Commands

```bash
cargo build --release
cargo test --release                 # release: some tests build real indexes
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo bench --bench kernels -- --warm-up-time 1 --measurement-time 3   # quick run
TELUSUR_KERNEL=scalar cargo test --release   # force a kernel

```

`.cargo/config.toml` sets `-C target-cpu=native`. Kernels still use runtime detection, so binaries are safe on the build machine; do not ship them to other CPUs.

## Module map


| Path                   | Owns                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `src/util.rs`          | `MmapSlice<T: Pod>`, `.fbin` / `.ibin` IO, `TopK` heap, percentiles                   |
| `src/vector/quant.rs`  | Matryoshka truncate + normalize, binarize, `dot`, `dot_f16`, Hamming                  |
| `src/vector/kernel.rs` | `QueryLut`, fast-scan block packing, scalar / AVX2 / AVX-512 / NEON kernels, dispatch |
| `src/vector/kmeans.rs` | spherical k-means, sgemm assignment, exact top-k for ground truth                     |
| `src/vector/ivf.rs`    | build, open (mmap), search with filters, memory report                                |
| `src/text/analyzer.rs` | Anserini-matching English analyzer: UAX #29 tokens, hand-written Porter stemmer       |
| `src/text/codec.rs`    | varint, Lucene SmallFloat norms                                                       |
| `src/text/index.rs`    | tiered postings build, FST dictionary, exhaustive + MaxScore search                   |
| `src/filter.rs`        | keyword and crawl date filters as roaring bitmaps (no numeric filters)                |
| `src/docstore.rs`      | id to stored fields (title, url, snippet), mmap                                       |
| `src/engine.rs`        | hybrid search, RRF, per-stage timings                                                 |
| `src/server.rs`        | axum HTTP API, embedder sidecar client                                                |
| `src/eval.rs`          | recall@k, nDCG@10, Recall@100, BEIR loader                                            |
| `src/ingest.rs`        | FineWeb parquet streaming, normalize, dedup                                           |
| `src/synth.rs`         | synthetic vectors and Zipfian text                                                    |
| `src/bin/telusur.rs`   | CLI                                                                                   |


## Design facts that are easy to get wrong

**Kernels**

- Bit layout: byte `j` of a code holds dims `8j..8j+7`, LSB first. Low nibble = group `2j`, high nibble = group `2j+1`.
- Block layout: 32 docs per block, transposed, row `j` = byte `j` of all 32 codes.
- `pshufb` / `vpshufb` look up within each 128-bit lane. Duplicate each 16-byte table into every lane.
- Widening on x86: view the 32-byte register as 16 u16 lanes. `x & 0x00FF` = even docs, `x >> 8` = odd docs. Interleave at the end of the block.
- AVX-512: load two rows (64 bytes) at once; the table for low nibbles is `[g(4p), g(4p), g(4p+2), g(4p+2)]`, high nibbles `[g(4p+1), g(4p+1), g(4p+3), g(4p+3)]`. Add the two 256-bit halves at the end.
- NEON: `vqtbl1q_u8` for lookup, `vaddl_u8` / `vaddl_high_u8` to widen. Shifts take const generics: `vshrq_n_u8::<4>(x)`.
- u16 accumulators: max sum is `(dims / 4) × 255`, so code dims must be ≤ 1024 and a multiple of 64. Keep the 1024-dim all-ones overflow test.
- Rust 2024: inside an `unsafe fn` with `#[target_feature]`, unsafe operations still need an inner `unsafe { }` block. x86 intrinsics accept the old positional immediate form, e.g. `_mm256_srli_epi16(c, 4)`.
- Gate AVX-512 on both `avx512f` and `avx512bw`.

**Vector index**

- k-means and ground truth must use a GEMM (`matrixmultiply::sgemm`) in chunks. Per-query dot loops over the whole base are memory-bound and orders of magnitude slower.
- Build streams the input (`Fbin` is mmapped). Do not load a 12 GB file into RAM.
- Keep the scan loop branch-light: compare the u16 score against a `min_accept` threshold before touching the heap or the filter.
- `TopK` breaks ties toward the smaller doc id so results are deterministic.
- Every doc in an index has a vector, or none do. Indexes are read-only once built (`docs/adr/0001-indexes-are-immutable.md`).

**BM25**

- Lucene BM25 drops the `(k1 + 1)` factor; it does not change ranking. idf is `ln(1 + (N - df + 0.5) / (df + 0.5))`. Defaults k1 = 0.9, b = 0.4.
- Analyzer = Anserini's `DefaultEnglishAnalyzer`: Unicode word boundaries (`unicode-segmentation`), possessive `'s`, lowercase, Lucene's 33 stopwords, original Porter stemmer. Not `rust-stemmers`: it only has Porter2.
- BEIR eval = Anserini's flat regressions: title, newline, text as one field; drop results whose doc id equals the query id (`-removeQuery`); trec_eval rules with `-c`; gain = relevance grade.
- Doc lengths go through `SmallFloat.intToByte4` (exact below 24, then 4 significant bits). avgdl uses exact total length / docs. Needed to match Lucene numbers.
- Varints of values ≥ 1 never contain a 0x00 byte. First doc in a tier is stored as `doc + 1` (wrapping add from `u32::MAX`), later ones as gaps.
- Tiers have explicit headers (no separator bytes) so search can jump to any tier.
- Skip entries store `prev_doc` (last doc of the previous block) so delta decoding can resume mid-tier.
- MaxScore: sort cursors by upper bound ascending, keep prefix sums. A cursor becomes non-essential when the prefix sum up to it is ≤ the current k-th score. Stop when every cursor is non-essential.

**Deps**

- `rand` 0.9 and `rand_distr` 0.5 (0.10 changed the API). Only used for synthetic data and k-means init.
- No xxh3 crate: ingest dedup uses the standard library hasher.

## How to work a milestone

1. Re-read the PRD section for the milestone.
2. Write the tests and the reference implementation first.
3. Implement the fast path until tests pass in release mode.
4. Run clippy and fmt.
5. Run the milestone's benchmark, save raw output to `bench-results/<bench>-<date>.json` plus a short `.md` table.
6. Update the README only from those files. Leave anything unmeasured as `TBD`.
7. Summarize what changed and what the numbers say, in two or three sentences.

## Data

Everything in `data/` and `indexes/` is gitignored. Prep scripts in `scripts/` (Python, `uv` or `pip`): `prep_dbpedia.py`, `prep_beir.py`, `embed_server.py`, `compare_faiss.py`. Formats: `.fbin` (`u32 n, u32 d`, f32 rows), `.ibin` (`u32 n, u32 k`, u32 ids), `corpus.jsonl` where line order = doc id = row in the `.fbin`.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on AustinPardosi/telusur, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
