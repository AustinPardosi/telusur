<p align="center">
  <img src="assets/social-preview.png" alt="telusur: hybrid search engine in Rust. 1-bit Matryoshka vectors, SIMD lookup tables, compressed BM25.">
</p>
<p align="center">
  <a href="#results">Results</a> · <a href="#how-a-query-runs">How it works</a> · <a href="#quick-start">Quick start</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#datasets">Datasets</a>
</p>

---

 

*telusur* is Indonesian for "to search through, to trace". It is pronounced teh-LOO-soor.

 

I wanted to understand how a search engine built for AI actually works under the hood, so I took the designs Exa describes in its engineering blog and rebuilt them from scratch: their [vector database](https://exa.ai/blog/building-web-scale-vector-db), their [BM25 index](https://exa.ai/blog/bm25-optimization), and the [fusion step](https://exa.ai/blog/composing-a-search-engine) that joins them. Then I measured everything on real data.

 

One detail from Exa's vector DB post stuck with me: the lookup-table kernel at the core of it had to be written in C++ instead of Rust. telusur does it in Rust.

 

> Independent project. Not affiliated with or endorsed by Exa.

 

## Results

 

All numbers below come from the commands in [Benchmarks](#benchmarks), on the machines listed. <!-- fill from bench-results/ -->

 

**Machines:** an Apple M4 laptop (4 performance + 6 efficiency cores, 16 GB RAM) for everything, and an x86 cloud VM with AVX-512 (TBD) for the AVX2 and AVX-512 kernel rows.

 


|                                                          |                                       |
| -------------------------------------------------------- | ------------------------------------- |
| Codes scored per second, one core (256-dim, best kernel) | 516M (NEON, Apple M4)                 |
| Speedup over scalar                                      | 11x (NEON, Apple M4)                  |
| Recall@10 on 1M OpenAI embeddings                        | 0.952 at 0.58 ms p50 (0.835 vs full)  |
| Hot memory per vector                                    | 37.2 bytes (vs 12,288 for raw float32) |
| BM25 nDCG@10 vs Anserini (avg of 4 BEIR sets)            | 45.72 vs 45.79                        |
| Hybrid nDCG@10 (avg of 4 BEIR sets)                      | 56.60 (neural 56.14, BM25 45.72)      |
| BM25 index size vs naive                                 | 87% smaller (BEIR NQ)                 |
| Postings decoded, MaxScore vs exhaustive                 | 50% fewer on TREC-COVID, same top-10  |
| Ingest throughput (FineWeb)                              | 40,198 docs/s on one core (3.47B/day) |


 

## How a query runs

 

```mermaid
flowchart LR
    Q[query] --> E[embed]
    Q --> A[analyze]
    E --> R[route to nearest clusters]
    R --> S[scan 1-bit codes<br/>SIMD lookup tables]
    S --> RR[rerank with f16 vectors]
    A --> B[BM25 MaxScore<br/>over tf tiers]
    F[filters] --> S
    F --> B
    RR --> FU[reciprocal rank fusion]
    B --> FU
    FU --> OUT[results + timings]

```

 

Keyword and neural retrieval run in parallel. Every response includes how long each stage took.

 

## The interesting parts

 

### 1. Scoring 1-bit codes against a float query, in registers

 

Documents are stored as the first 256 dims of a Matryoshka embedding, 1 bit per dim. That is 32 bytes per document instead of 12 KB.

 

The query stays in float. Split it into groups of 4 dims. A group of a binary document can only be one of 16 sign patterns, so for each query group I precompute all 16 possible partial dot products. Scoring a document is then one table lookup per 4 dims.

 

The fast part is doing those lookups 32 documents at a time. A byte shuffle (`vpshufb` on x86, `tbl` on ARM) is a 16-way parallel table lookup, so with the table in a register and 32 documents' nibbles in another, one instruction does 32 lookups. Codes are stored transposed in blocks of 32 so the nibbles line up.

 

There are four kernels (scalar, AVX2, AVX-512, NEON), picked at runtime. Every one is tested to match the scalar version bit for bit.

 


| Kernel  | Codes/s per core (256-dim) |
| ------- | -------------------------- |
| scalar  | 46.3M (Apple M4)           |
| AVX2    | TBD                        |
| AVX-512 | TBD                        |
| NEON    | 516M (Apple M4)            |

Measured with `cargo bench --bench kernels` ([B1](bench-results/b1-kernels-2026-11-05.md)). The AVX2 and AVX-512 rows wait for the x86 VM run; on the Mac, NEON is 11x scalar at 256 and 512 dims.


 

Why not Hamming distance? Keeping the query in float loses much less information. On the same codes:

 


| First-stage scorer                | Recall@100 (256-dim codes) |
| --------------------------------- | -------------------------- |
| Hamming (binary query)            | 0.304                      |
| Asymmetric, float table           | 0.425                      |
| Asymmetric, u8 table (what ships) | 0.425                      |

200k DBpedia vectors, the true top 100 by cosine at 1024 dims, Apple M4 ([B2](bench-results/b2-ablate-quant-2026-11-05.md), 64 to 512 dims there). The float query finds 1.25x to 1.7x as many of the true neighbors as Hamming at every code size, and quantizing its tables to u8 costs nothing measurable.


 

### 2. Frequency tiers are impact tiers

 

Exa's BM25 post groups each term's postings by term frequency to save memory: all docs with tf=1 together, tf=2 together, and so on, so tf is stored once per group.

 

It turns out that grouping does a second job. Every doc in a tier has the same tf, so the tier's best possible score is known exactly. Treat each (term, tier) as its own list and run MaxScore over them, and the long tf=1 tiers of common words get skipped unless a document could still make the top k. Same top-k scores as scoring everything, and the bigger the corpus, the less decoding.

 

BEIR test queries, k = 10, Apple M4 ([B7](bench-results/b7-bm25-pruning-2026-11-05.md)). Each cell is exhaustive / MaxScore over tiers; latency is the keyword stage on one thread.


| Dataset    | Postings decoded per query | p50 ms        | p99 ms        | QPS, 1 thread   |
| ---------- | -------------------------- | ------------- | ------------- | --------------- |
| SciFact    | 4,376 / 4,103              | 0.066 / 0.047 | 0.090 / 0.119 | 14,936 / 20,312 |
| NFCorpus   | 671 / 671                  | 0.012 / 0.009 | 0.050 / 0.035 | 55,005 / 78,495 |
| FiQA       | 47,742 / 29,959            | 0.634 / 0.374 | 0.864 / 1.497 | 1,750 / 2,230   |
| TREC-COVID | 182,938 / 91,402           | 1.895 / 1.033 | 2.691 / 5.065 | 540 / 784       |

MaxScore wins on p50 and throughput everywhere, but loses p99 on long queries (9 to 20 terms), where every candidate probes many dense tiers. Block-max bounds inside tiers are the next step.


 

Memory, step by step (like Exa's post), on BEIR NQ: 2.68M docs, 118.6M postings, Apple M4 ([B8](bench-results/b8-ablate-postings-2026-11-05.md)). Each row counts postings, dictionary and doc lengths.

 


| Format                                            | Bytes per posting |
| ------------------------------------------------- | ----------------- |
| `HashMap<String, Vec<(u32, u32)>>`                | 12.69             |
| + one flat buffer                                 | 8.89              |
| + delta varint doc ids                            | 6.20              |
| + tf tiers                                        | 2.42              |
| + inline singletons, FST dictionary, 1-byte norms | 1.64              |

That is 87% smaller than naive. The delta varint row still keeps a 4-byte tf per posting; against a varint tf, tiers cut postings bytes by 34%.


 

### 3. Filters that act like a database

 

Filters (domain, crawl date range, any keyword field) are roaring bitmaps. If a filter is narrow, telusur skips the index and scores the matching docs exactly. If it is broad, it probes more clusters and drops non-matching docs before they reach the heap. Either way, no result ever breaks the filter.

 


| Selectivity | p50 ms | Recall@10 | Path |
| ----------- | ------ | --------- | ---- |
| none        | 0.59   | 0.952     | 256 clusters |
| 0.1%        | 0.14   | 1.000     | exact |
| 1%          | 2.20   | 1.000     | exact |
| 10%         | 4.09   | 0.963     | 2,048 clusters |
| 50%         | 3.51   | 0.961     | 513 clusters |

DBpedia 1M with synthetic categories, neural search at the defaults, Apple M4 ([B5](bench-results/b5-filters-dbpedia-2026-11-05.md)). Recall is against the exact top 10 among the allowed docs. At 1% the exact path takes 3.7x the unfiltered latency, so it misses the 2x target; 10% and 50% probe nearly every cluster.


 

## Quick start

 

```bash
# build
cargo build --release

# try it on synthetic data (no downloads)
./target/release/telusur synth text --docs 100000 --out data/synth/corpus.jsonl
./target/release/telusur index --corpus data/synth/corpus.jsonl --out indexes/synth
./target/release/telusur search --index indexes/synth "w42 w1337"

# real data: BEIR SciFact with embeddings
uv run --with fastembed scripts/prep_beir.py scifact --model nomic-ai/nomic-embed-text-v1.5
./target/release/telusur index --corpus data/scifact/corpus.jsonl \
    --vectors data/scifact/corpus.fbin --out indexes/scifact
./target/release/telusur eval beir --data data/scifact --index indexes/scifact \
    --query-vectors data/scifact/queries.fbin

# serve it
uv run --with fastembed scripts/embed_server.py --model nomic-ai/nomic-embed-text-v1.5 &
./target/release/telusur serve --index indexes/scifact --embedder http://127.0.0.1:8001 &
curl -s localhost:8080/search -d '{"query":"does vitamin d prevent fractures","mode":"hybrid","k":5,"debug":true}'

```

 

## Benchmarks

 

Rules I hold myself to: release build, warm cache, single-thread latency over 1,000+ queries (p50 and p99), all-core throughput as QPS, the hardware printed with every run, and every comparison against another engine on the same machine. Raw outputs live in [`bench-results/`](bench-results/).

 


| What                        | Command                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| Kernel throughput           | `cargo bench --bench kernels`                                      |
| First-stage scorer ablation | `telusur bench ablate-quant --base data/dbpedia/base.fbin ...`     |
| ANN recall vs latency       | `telusur bench ann --index indexes/dbpedia --queries ... --gt ...` |
| Filters                     | `telusur bench filters ...`                                        |
| BM25 quality                | `telusur eval beir --data data/<set> --index indexes/<set>`        |
| Keyword, neural, hybrid quality | `telusur eval beir ... --query-vectors data/<set>/queries.fbin` |
| HTTP latency per stage      | `telusur bench http --queries ...` against `telusur serve`         |
| BM25 speed + memory         | `telusur bench bm25 ...` and `telusur bench ablate-postings ...`   |
| Ingest                      | `telusur ingest fineweb --parquet data/fineweb --out ...`          |


 

### Search quality (nDCG@10)

 


| Dataset    | Anserini BM25 | telusur BM25 | telusur neural | telusur hybrid |
| ---------- | ------------- | ------------ | -------------- | -------------- |
| SciFact    | 67.89         | 67.89        | 70.33          | 73.85          |
| NFCorpus   | 32.18         | 32.05        | 34.56          | 35.83          |
| FiQA       | 23.61         | 23.61        | 37.48          | 35.24          |
| TREC-COVID | 59.47         | 59.35        | 82.19          | 81.50          |


 

Anserini reference: the [BEIR (v1.0.0) flat BM25 regressions](https://github.com/castorini/anserini/tree/master/docs/reproduce/from-document-collection), k1=0.9, b=0.4. telusur's eval follows the same setup: title and text as one field, results matching the query id dropped, `trec_eval -c`. Measured on the Apple M4 ([B6](bench-results/b6-bm25-parity-2026-11-05.md)). Recall@100 matches Anserini on all four sets. NFCorpus and TREC-COVID are 0.13 and 0.12 below on nDCG@10, but plain Lucene 10.5 with the same setup lands where telusur does, so the gap is between today's Lucene and Anserini's reference run, which was recorded with Lucene 8.11 in 2022.

Neural uses nomic-embed-text-v1.5 on CPU (768 dims), searched at the index defaults. Hybrid is RRF over the top 1000 of each ([B10](bench-results/b10-hybrid-beir-2026-11-06.md)). Hybrid beats the better single mode on SciFact and NFCorpus, where BM25 and neural are close. It falls a little behind neural on FiQA and TREC-COVID, where BM25 is 14 and 23 points worse. That misses the PRD target of 3 of 4 sets. Its average (56.60) is still the best of the three modes (neural 56.14, BM25 45.72).

 

### ANN on 1M OpenAI embeddings

 


DBpedia 1M, OpenAI text-embedding-3-large (3072 dims), 2,048 clusters, 256-dim codes, 1024-dim rerank vectors, Apple M4 ([B3](bench-results/b3-ann-dbpedia-2026-11-05.md), full grid there). Recall@10 against rerank ground truth (exact top 10 at the rerank dims) and full ground truth (at all 3072 dims); latency on one thread.

| Clusters probed | Rerank depth | Recall@10 (rerank / full GT) | p50 ms | p99 ms | QPS (4 / 10 threads) |
| --------------- | ------------ | ---------------------------- | ------ | ------ | -------------------- |
| 32              | 0            | 0.457 / 0.441                | 0.08   | 0.09   | 42,910 / 65,843      |
| 32              | 200          | 0.839 / 0.754                | 0.17   | 0.18   | 18,228 / 31,792      |
| 128             | 200          | 0.894 / 0.800                | 0.28   | 0.33   | 11,452 / 17,794      |
| 128             | 500          | 0.932 / 0.821                | 0.40   | 0.44   | 7,761 / 11,863       |
| 256 (default)   | 500          | 0.952 / 0.835                | 0.58   | 0.69   | 5,674 / 8,809        |
| 512             | 1000         | 0.976 / 0.849                | 1.10   | 1.48   | 2,939 / 3,521        |

The defaults, 256 clusters probed and a rerank depth of 500, are the smallest that reach 0.95 against rerank ground truth; at depth 200 recall stops near 0.92 however many clusters are probed. Against full ground truth recall tops out near 0.85: that is the cost of reranking with 1024 of the 3072 dims. QPS at 10 threads is noisy on a laptop with other apps open.


 

Compared with FAISS and hnswlib on the same machine: TBD.

### HTTP latency

One client, one request at a time over localhost, k = 10, 1,000 requests per mode, Apple M4 ([B12](bench-results/b12-http-2026-11-06.md), per-stage split there). On DBpedia 1M the request carries the query vector. On TREC-COVID the server embeds the query text through the nomic sidecar on CPU.

| Mode    | DBpedia 1M p50 / p99 ms | TREC-COVID, server embeds, p50 / p99 ms |
| ------- | ----------------------- | --------------------------------------- |
| keyword | 0.23 / 1.48             | 1.18 / 5.15                             |
| neural  | 1.61 / 2.20             | 23.66 / 77.02                           |
| hybrid  | 1.64 / 3.61             | 24.78 / 78.09                           |

HTTP and JSON add 0.1 to 0.2 ms. Hybrid costs about the same as neural, because the keyword search runs in parallel. When the server embeds, the embedder is about 90% of the time (22 ms p50), and the search itself takes about 1 ms.

 

## Datasets

 


| Dataset                                                                                                                                     | Used for                             | Size               |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------ |
| [DBpedia 1M, OpenAI text-embedding-3-large](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) | ANN speed, recall, memory            | 24.8 GB            |
| [BEIR](https://github.com/beir-cellar/beir): SciFact, NFCorpus, FiQA, TREC-COVID                                                            | Retrieval quality                    | small to 171k docs |
| [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb)                                                                | Ingest throughput, web-style filters | 27.6 GB            |
| Synthetic                                                                                                                                   | Tests and CI                         | generated          |


 

FineWeb is released under ODC-By 1.0 and is subject to Common Crawl's terms of use.

 

## Project layout

 

```
src/vector/   Matryoshka, binary codes, SIMD kernels, k-means, IVF, rerank
src/text/     analyzer, posting codecs, BM25 index and MaxScore
src/          filters, doc store, hybrid engine, HTTP server, eval, ingest
benches/      criterion benchmarks
scripts/      dataset prep and the embedding sidecar

```

 

## What I would do next

 

- Shard by cluster across processes and scatter-gather queries.
- Block-max bounds inside each tier.
- 2-bit codes or product quantization as an alternative first stage.
- A zstd cold tier for rarely used terms.

 

## Credits

 

The design follows Exa's engineering posts: [How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db), [Serving BM25 with 50% memory reduction](https://exa.ai/blog/bm25-optimization), and [Composing a search engine](https://exa.ai/blog/composing-a-search-engine). BM25 settings follow Anserini. Benchmark file formats follow big-ann-benchmarks.

 

## License

 

MIT