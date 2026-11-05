# B3: ANN recall vs latency on DBpedia 1M (5 Nov 2026)

Raw output: [`b3-ann-dbpedia-2026-11-05.json`](b3-ann-dbpedia-2026-11-05.json) (the PRD grid) and [`b3-ann-dbpedia-extended-2026-11-05.json`](b3-ann-dbpedia-extended-2026-11-05.json) (more clusters and deeper reranks, to find where recall reaches 0.95). Machine: Apple M4, 10 cores, 16 GB RAM, macOS, NEON kernel.

```bash
uv run --with pyarrow --with numpy scripts/prep_dbpedia.py
telusur gt --base data/dbpedia/base.fbin --queries data/dbpedia/queries.fbin --out data/dbpedia
telusur index --corpus data/dbpedia/corpus.jsonl --vectors data/dbpedia/base.fbin --out indexes/dbpedia
telusur bench ann --index indexes/dbpedia --queries data/dbpedia/queries.fbin \
    --gt-full data/dbpedia/gt_full.ibin --gt-rerank data/dbpedia/gt_rerank.ibin
telusur bench ann ... --nprobe 256,512,1024 --rerank-depth 200,500,1000
```

**Data:** OpenAI text-embedding-3-large (3072 dims), 999,000 base vectors, the last 1,000 rows held out as queries. **Index:** 2,048 clusters (the default for 1M docs), 256-dim 1-bit codes, 1024-dim f16 rerank vectors. Built in 91 s (the first build, 121 s, peaked at 4.2 GB of memory).

**Ground truth:** exact top 100 by cosine, at 1024 dims (rerank ground truth: what the index loses) and at all 3072 dims (full ground truth: what the whole design loses, shortening included). Both took 61 s.

**Measured:** recall@10 over all 1,000 queries; single-thread latency (route, scan and rerank) over 1,000 queries after a warm-up pass; QPS over five passes at 4 and 10 threads.

| Clusters probed | Rerank depth | Recall@10, rerank GT | Recall@10, full GT | Codes scanned | p50 ms | p99 ms | QPS 4 / 10 threads |
| --------------- | ------------ | -------------------- | ------------------ | ------------- | ------ | ------ | ------------------ |
| 8 | 0 | 0.424 | 0.408 | 4,526 | 0.045 | 0.084 | 60,710 / 105,273 |
| 8 | 100 | 0.712 | 0.651 | 4,526 | 0.091 | 0.331 | 35,484 / 56,161 |
| 8 | 200 | 0.743 | 0.673 | 4,526 | 0.131 | 0.178 | 24,478 / 39,818 |
| 8 | 500 | 0.760 | 0.684 | 4,526 | 0.258 | 0.299 | 11,086 / 19,549 |
| 16 | 0 | 0.444 | 0.428 | 8,773 | 0.054 | 0.062 | 59,909 / 92,971 |
| 16 | 100 | 0.762 | 0.695 | 8,773 | 0.107 | 0.123 | 29,606 / 49,632 |
| 16 | 200 | 0.797 | 0.719 | 8,773 | 0.185 | 0.214 | 15,993 / 33,163 |
| 16 | 500 | 0.821 | 0.733 | 8,773 | 0.283 | 0.372 | 10,916 / 18,007 |
| 32 | 0 | 0.457 | 0.441 | 17,169 | 0.077 | 0.088 | 42,910 / 65,843 |
| 32 | 100 | 0.797 | 0.727 | 17,169 | 0.133 | 0.154 | 24,224 / 41,592 |
| 32 | 200 | 0.839 | 0.754 | 17,169 | 0.166 | 0.183 | 18,228 / 31,792 |
| 32 | 500 | 0.868 | 0.771 | 17,169 | 0.296 | 0.612 | 10,100 / 16,774 |
| 64 | 0 | 0.467 | 0.452 | 33,786 | 0.113 | 0.153 | 23,975 / 47,170 |
| 64 | 100 | 0.825 | 0.753 | 33,786 | 0.180 | 0.224 | 18,062 / 28,620 |
| 64 | 200 | 0.873 | 0.782 | 33,786 | 0.212 | 0.254 | 15,322 / 25,262 |
| 64 | 500 | 0.907 | 0.802 | 33,786 | 0.335 | 0.370 | 8,791 / 14,653 |
| 128 | 0 | 0.472 | 0.458 | 66,903 | 0.185 | 0.236 | 18,325 / 29,300 |
| 128 | 100 | 0.843 | 0.769 | 66,903 | 0.234 | 0.272 | 13,969 / 22,769 |
| 128 | 200 | 0.894 | 0.800 | 66,903 | 0.283 | 0.328 | 11,452 / 17,794 |
| 128 | 500 | 0.932 | 0.821 | 66,903 | 0.404 | 0.438 | 7,761 / 11,863 |
| 256 | 200 | 0.910 | 0.813 | 132,996 | 0.426 | 0.485 | 7,675 / 12,198 |
| **256** | **500** | **0.952** | **0.835** | 132,996 | 0.576 | 0.685 | 5,674 / 8,809 |
| 256 | 1000 | 0.966 | 0.842 | 132,996 | 0.766 | 0.847 | 4,089 / 6,696 |
| 512 | 200 | 0.918 | 0.819 | 263,937 | 0.725 | 0.927 | 4,549 / 6,493 |
| 512 | 500 | 0.960 | 0.841 | 263,937 | 0.822 | 1.091 | 3,591 / 4,995 |
| 512 | 1000 | 0.976 | 0.849 | 263,937 | 1.096 | 1.485 | 2,939 / 3,521 |
| 1024 | 200 | 0.921 | 0.822 | 524,626 | 1.221 | 1.574 | 2,446 / 2,917 |
| 1024 | 500 | 0.964 | 0.845 | 524,626 | 1.328 | 1.825 | 2,065 / 2,359 |
| 1024 | 1000 | 0.981 | 0.852 | 524,626 | 1.593 | 1.905 | 1,433 / 1,950 |

**Defaults: 256 clusters probed, rerank depth 500.** 256 is the smallest count in either sweep to reach recall@10 of 0.95 against rerank ground truth, and it does so at rerank depth 500 (0.952 at 0.58 ms p50), so the default rerank depth went from 200 to 500 with it. That meets the PRD target of at least 0.95 at a p50 of at most 5 ms.

**What the numbers say:**

- At rerank depth 200 recall stops near 0.92 however many clusters are probed: 200 candidates from 256-dim 1-bit codes are not enough. Deeper reranks keep paying off.
- Without a rerank (depth 0), 1-bit codes alone find under half of the true top 10.
- Against full ground truth the best setting reaches 0.85. The gap from rerank ground truth is the cost of reranking with 1024 of the 3072 dims; reranking with all 3072 is the PRD's fallback if that matters.
- Rerank cost grows with depth: at 128 clusters, depth 0 to 500 adds 0.22 ms, about 0.44 us per candidate for a 1024-dim f16 dot product.

**Hot memory per vector:** 37.2 bytes, 33.0 for the code (32 bytes plus block padding) and 4.1 for the doc id, against 12,288 bytes for a raw float32 vector. Centroids add 2.1 bytes per doc. The f16 rerank vectors (2,048 bytes per doc) are memory-mapped and only touched for candidates.

**History:** the first run of this sweep took 2.09 ms p50 at the default setting. Routing and reranking then used plain in-order f32 / f16 sums the compiler could not vectorize; they now add in 16 lanes, the index was rebuilt with that code, and everything here is from the re-run. Recall did not change in any cell.

**Caveats:** single-thread runs use the default QoS (no core pinning). The laptop had other apps open (a load average of 5 to 7), which makes the 10-thread QPS and some p99 values noisy between runs; recall is exact and repeatable.
