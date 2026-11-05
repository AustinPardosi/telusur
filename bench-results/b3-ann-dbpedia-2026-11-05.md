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

**Data:** OpenAI text-embedding-3-large (3072 dims), 999,000 base vectors, the last 1,000 rows held out as queries. **Index:** 2,048 clusters (the default for 1M docs), 256-dim 1-bit codes, 1024-dim f16 rerank vectors. Built in 121 s, 4.2 GB peak memory.

**Ground truth:** exact top 100 by cosine, at 1024 dims (rerank ground truth: what the index loses) and at all 3072 dims (full ground truth: what the whole design loses, shortening included). Both took 61 s.

**Measured:** recall@10 over all 1,000 queries; single-thread latency (route, scan and rerank) over 1,000 queries after a warm-up pass; QPS over five passes at 4 and 10 threads.

| Clusters probed | Rerank depth | Recall@10, rerank GT | Recall@10, full GT | Codes scanned | p50 ms | p99 ms | QPS 4 / 10 threads |
| --------------- | ------------ | -------------------- | ------------------ | ------------- | ------ | ------ | ------------------ |
| 8               | 0            | 0.424                | 0.408              | 4,526         | 0.351  | 0.378  | 11,669 / 18,703    |
| 8               | 100          | 0.712                | 0.651              | 4,526         | 0.537  | 1.077  | 5,433 / 5,707      |
| 8               | 200          | 0.743                | 0.673              | 4,526         | 0.719  | 5.332  | 5,797 / 9,354      |
| 8               | 500          | 0.760                | 0.684              | 4,526         | 1.253  | 1.937  | 3,477 / 4,984      |
| 16              | 0            | 0.444                | 0.428              | 8,773         | 0.384  | 0.631  | 10,960 / 22,532    |
| 16              | 100          | 0.762                | 0.695              | 8,773         | 0.567  | 0.919  | 5,671 / 16,407     |
| 16              | 200          | 0.797                | 0.719              | 8,773         | 0.779  | 1.174  | 5,371 / 8,002      |
| 16              | 500          | 0.821                | 0.733              | 8,773         | 1.306  | 2.052  | 3,158 / 5,445      |
| 32              | 0            | 0.457                | 0.441              | 17,169        | 0.428  | 0.469  | 9,954 / 21,680     |
| 32              | 100          | 0.797                | 0.727              | 17,169        | 0.633  | 0.894  | 6,511 / 10,881     |
| 32              | 200          | 0.839                | 0.754              | 17,169        | 0.790  | 0.845  | 5,141 / 7,635      |
| 32              | 500          | 0.868                | 0.771              | 17,169        | 1.364  | 1.741  | 3,050 / 4,873      |
| 64              | 0            | 0.467                | 0.452              | 33,786        | 0.525  | 0.698  | 8,404 / 18,117     |
| 64              | 100          | 0.825                | 0.753              | 33,786        | 0.725  | 0.950  | 5,735 / 8,567      |
| 64              | 200          | 0.873                | 0.782              | 33,786        | 0.978  | 1.835  | 4,444 / 7,308      |
| 64              | 500          | 0.907                | 0.802              | 33,786        | 1.616  | 2.354  | 2,857 / 4,824      |
| 128             | 0            | 0.472                | 0.458              | 66,903        | 0.740  | 1.211  | 6,484 / 13,634     |
| 128             | 100          | 0.843                | 0.769              | 66,903        | 0.917  | 1.559  | 3,965 / 6,087      |
| 128             | 200          | 0.894                | 0.800              | 66,903        | 1.141  | 1.441  | 3,676 / 4,202      |
| 128             | 500          | 0.932                | 0.821              | 66,903        | 1.690  | 2.520  | 2,426 / 4,014      |
| 256             | 200          | 0.910                | 0.813              | 132,996       | 1.446  | 2.227  | 2,744 / 3,824      |
| **256**         | **500**      | **0.952**            | **0.835**          | 132,996       | 2.085  | 3.272  | 1,941 / 2,784      |
| 256             | 1000         | 0.966                | 0.842              | 132,996       | 3.009  | 5.385  | 1,319 / 2,036      |
| 512             | 200          | 0.918                | 0.819              | 263,937       | 2.633  | 4.327  | 1,809 / 2,641      |
| 512             | 500          | 0.960                | 0.841              | 263,937       | 2.900  | 4.986  | 1,373 / 2,138      |
| 512             | 1000         | 0.976                | 0.849              | 263,937       | 3.611  | 4.590  | 1,093 / 1,737      |
| 1024            | 200          | 0.921                | 0.822              | 524,626       | 3.255  | 4.878  | 1,090 / 1,459      |
| 1024            | 500          | 0.964                | 0.845              | 524,626       | 4.431  | 8.283  | 1,007 / 1,346      |
| 1024            | 1000         | 0.981                | 0.852              | 524,626       | 6.205  | 13.390 | 688 / 1,102        |

**Default clusters probed: 256.** It is the smallest count in either sweep to reach recall@10 of 0.95 against rerank ground truth, and it does so at rerank depth 500 (0.952 at 2.09 ms p50). That meets the PRD target of at least 0.95 at a p50 of at most 5 ms.

**What the numbers say:**

- At rerank depth 200 recall stops near 0.92 however many clusters are probed: 200 candidates from 256-dim 1-bit codes are not enough. Deeper reranks keep paying off.
- Without a rerank (depth 0), 1-bit codes alone find under half of the true top 10.
- Against full ground truth the best setting reaches 0.85. The gap from rerank ground truth is the cost of reranking with 1024 of the 3072 dims; reranking with all 3072 is the PRD's fallback if that matters.
- Rerank cost grows with depth: at 128 clusters, depth 0 to 500 adds about 0.95 ms, roughly 1.9 us per candidate for the scalar f16 dot product. Routing over 2,048 centroids is a fixed cost too. Both are scalar code and the next things to speed up.

**Hot memory per vector:** 37.2 bytes, 33.0 for the code (32 bytes plus block padding) and 4.1 for the doc id, against 12,288 bytes for a raw float32 vector. Centroids add 2.1 bytes per doc. The f16 rerank vectors (2,048 bytes per doc) are memory-mapped and only touched for candidates.

**Caveats:** single-thread runs use the default QoS (no core pinning). The laptop had other apps open (about one core of background load), which makes the 10-thread QPS and some p99 values noisy between runs; recall is exact and repeatable.
