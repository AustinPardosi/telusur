# B5: filtered neural search on DBpedia 1M (5 Nov 2026)

Raw output: [`b5-filters-dbpedia-2026-11-05.json`](b5-filters-dbpedia-2026-11-05.json). Machine: Apple M4, 10 cores, 16 GB RAM, macOS, NEON kernel.

```bash
python scripts/add_categories.py data/dbpedia/corpus.jsonl data/dbpedia/corpus.categories.jsonl
telusur index --corpus data/dbpedia/corpus.categories.jsonl --vectors data/dbpedia/base.fbin \
    --keyword-fields category --out indexes/dbpedia
telusur bench filters --index indexes/dbpedia --base data/dbpedia/base.fbin --queries data/dbpedia/queries.fbin
```

Each doc gets a synthetic category, c0 to c999 from a hash of its line number; selectivity s filters on the first s x 1000 categories. Each query resolves the filter (roaring bitmap lookups) and runs neural search with it, at the defaults (256 clusters probed, rerank depth 500). Recall@10 is against the exact top 10 among the allowed docs by cosine at 1024 dims. Latency is one thread over 1,000 queries.

| Selectivity | Docs     | Path     | Clusters probed | Recall@10 | p50 ms | p99 ms | vs unfiltered |
| ----------- | -------- | -------- | --------------- | --------- | ------ | ------ | ------------- |
| none        |  999,000 | clusters |             256 | 0.952     | 0.588  | 0.771  | 1.0x          |
| 0.10%       |      995 | exact    |               0 | 1.000     | 0.135  | 0.147  | 0.2x          |
| 0.99%       |    9,934 | exact    |               0 | 1.000     | 2.202  | 3.693  | 3.7x          |
| 10.02%      |  100,106 | clusters |           2,048 | 0.963     | 4.091  | 5.134  | 7.0x          |
| 49.94%      |  498,894 | clusters |             513 | 0.961     | 3.512  | 3.858  | 6.0x          |

**Paths:** a filter allowing 20,000 docs or fewer is scored exactly with rerank vectors ("exact"); a broader one probes clusters probed / selectivity clusters, at most all 2,048, and drops disallowed docs before the candidate heap ("clusters").

**Does 1% stay within 2x unfiltered latency? No: 3.7x** (2.20 ms against 0.59 ms). The exact path scores all 9,934 allowed docs with 1024-dim f16 vectors, about 0.2 us each.

**What the numbers say:**

- No result breaks the filter (the engine tests check that), and the exact path is perfect on recall.
- 0.1% is faster than no filter: 995 exact dot products beat scanning 256 clusters.
- 10% and 50% cost 6x to 7x unfiltered for recall 0.96: dividing clusters probed by selectivity probes nearly every cluster at 10%.
- Ways to close the gap, not tried yet: map doc ids to code slots so a narrow filter can use the 1-bit first stage on its own docs before reranking; and widen the probe less than 1 / selectivity.

The first run of this benchmark had 1% at 7.6 ms: the f16 and f32 dot products were plain in-order sums the compiler could not vectorize. They now add in 16 lanes, which is what the numbers above use (B3 was re-run with the same change).
