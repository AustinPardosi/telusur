# B4: telusur vs FAISS and hnswlib on DBpedia 1M (7 Nov 2026)

Raw outputs: [`b4-ann-vs-others-2026-11-07.json`](b4-ann-vs-others-2026-11-07.json) (FAISS 1.15.1, hnswlib 0.8.0) and [`b4-telusur-2026-11-07.json`](b4-telusur-2026-11-07.json). telusur's file sizes are from [`b12-http-2026-11-07.json`](b12-http-2026-11-07.json), which lists the index's files. Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power.

```bash
telusur bench ann --index indexes/dbpedia --queries data/dbpedia/queries.fbin --gt-full data/dbpedia/gt_full.ibin \
    --gt-rerank data/dbpedia/gt_rerank.ibin --nprobe 256 --rerank-depth 500
uv run --with faiss-cpu --with hnswlib --with numpy scripts/compare_faiss.py data/dbpedia
```

**Setup:** the data, queries and ground truth of [B3](b3-ann-dbpedia-2026-11-05.md). FAISS and hnswlib get what telusur reranks with: the first 1024 dims of each vector, normalized. So the rerank ground truth (exact top 10 at 1024 dims) is the exact answer for every engine.

- **FAISS IndexIVFFlat:** 2,048 lists, like telusur, with float32 vectors.
- **FAISS IndexBinaryIVF:** telusur's 256-bit codes, 2,048 lists, Hamming distance. Run alone, and with the top 500 reranked exactly at 1024 dims. This is telusur's design with Hamming in place of the lookup-table scorer.
- **hnswlib HNSW:** M 16, ef_construction 200.

Each engine sweeps its knob. The table shows the cheapest setting that reaches recall@10 ≥ 0.95 (rerank ground truth). Latency is one thread, one query at a time. QPS sends one query per call from a pool of 4 or 10 threads, with each library's own threading set to 1, which is how telusur runs. Index bytes are the saved index file. The full sweeps are in the JSON.

| Engine                            | Setting                   | Recall@10, rerank / full GT | p50 ms | p99 ms | QPS, 4 / 10 threads | Index bytes          |
| --------------------------------- | ------------------------- | --------------------------- | ------ | ------ | ------------------- | -------------------- |
| telusur                           | 256 clusters, rerank 500  | 0.952 / 0.835               | 0.54   | 0.69   | 6,391 / 10,078      | 2.09 GB (39 MB hot)  |
| hnswlib HNSW                      | ef 64                     | 0.956 / 0.833               | 0.81   | 1.14   | 4,275 / 7,472       | 4.24 GB              |
| FAISS IndexIVFFlat                | 128 lists                 | 0.972 / 0.842               | 4.34   | 7.35   | 273 / 421           | 4.11 GB              |
| FAISS IndexBinaryIVF + rerank 500 | 512 lists (best, < 0.95)  | 0.868 / 0.791               | 0.63   | 0.81   | not run             | 4.13 GB              |
| FAISS IndexBinaryIVF              | 512 lists (best, < 0.95)  | 0.361 / 0.352               | 0.31   | 0.36   | not run             | 40 MB                |

telusur's 2.09 GB is 39 MB of codes, doc ids and centroids, which every query scans or routes with, plus 2.05 GB of f16 rerank vectors. Those are memory-mapped, and a query reads only its 500 candidates. FAISS and hnswlib keep every float32 vector in RAM. The two binary rows never reach 0.95, so they have no QPS run.

What the numbers say:

- **At the same recall, telusur is 1.5x faster than HNSW at p50 and 1.3x at 10-thread QPS.** It also uses half the bytes, and only 39 MB of them need to stay in RAM.
- **IndexIVFFlat is 8x slower at p50.** Each candidate costs it a full 4 KB float vector, against 32 bytes for a telusur code. At 64 lists (recall 0.946), it is still 2.32 ms.
- **The same codes with Hamming distance top out at 0.868, even with the same rerank.** telusur's scan keeps the query in float, which [B2](b2-ablate-quant-2026-11-05.md) showed finds 1.25x to 1.7x as many true neighbors as Hamming. FAISS also trains its binary clusters with Hamming k-means, where telusur routes with float centroids. The two effects are not measured separately here.

Caveats:

- **hnswlib on ARM:** its hand-written SIMD distances are for x86 only (SSE, AVX). On this Mac it uses a plain loop, so it is probably slower here than on x86.
- FAISS and hnswlib are timed from Python, so each query includes a Python call. telusur is timed in its own Rust bench.
- The settings are matched by the 0.95 threshold, not tuned to equal recall, so IndexIVFFlat lands higher (0.972).
- 10-thread QPS on a laptop is noisy.
