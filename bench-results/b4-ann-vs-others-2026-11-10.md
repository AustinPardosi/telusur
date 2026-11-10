# B4 rerun: telusur vs FAISS and hnswlib on DBpedia 1M (10 Nov 2026)

Raw outputs: [`b4-ann-vs-others-2026-11-10.json`](b4-ann-vs-others-2026-11-10.json) (FAISS 1.15.1, hnswlib 0.8.0) and [`b4-telusur-2026-11-10.json`](b4-telusur-2026-11-10.json). Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power. All engines ran in one session, after a cooldown, with no sleep.

Same commands, setup and caveats as [the 7 Nov run](b4-ann-vs-others-2026-11-07.md). The rerun is for telusur's faster rerank ([#24](https://github.com/AustinPardosi/telusur/issues/24), [details](rerank-speedup-2026-11-10.md)): comparing a new telusur number with old numbers for the other engines would mix sessions.

| Engine                            | Setting                   | Recall@10, rerank / full GT | p50 ms | p99 ms | QPS, 4 / 10 threads | Index bytes          |
| --------------------------------- | ------------------------- | --------------------------- | ------ | ------ | ------------------- | -------------------- |
| telusur                           | 256 clusters, rerank 500  | 0.952 / 0.835               | 0.44   | 0.59   | 8,393 / 12,491      | 2.09 GB (39 MB hot)  |
| hnswlib HNSW                      | ef 64                     | 0.959 / 0.834               | 0.82   | 1.14   | 4,354 / 6,733       | 4.24 GB              |
| FAISS IndexIVFFlat                | 128 lists                 | 0.972 / 0.842               | 4.29   | 5.40   | 258 / 352           | 4.11 GB              |
| FAISS IndexBinaryIVF + rerank 500 | 512 lists (best, < 0.95)  | 0.868 / 0.791               | 0.62   | 0.66   | not run             | 4.13 GB              |
| FAISS IndexBinaryIVF              | 512 lists (best, < 0.95)  | 0.361 / 0.352               | 0.29   | 0.32   | not run             | 40 MB                |

**At the same recall, telusur is now 1.8x faster than HNSW at p50 (0.44 vs 0.82 ms) and 1.9x at 10-thread QPS.** On 7 Nov it was 1.5x and 1.3x. The other engines' numbers barely moved: hnswlib 0.81 → 0.82 ms, IndexIVFFlat 4.34 → 4.29 ms, so the change is telusur's.
