# B5: filtered neural and hybrid search on FineWeb 1M (10 Nov 2026)

Raw outputs:

- Embed and build log: [`fineweb-neural-build-2026-11-10.txt`](fineweb-neural-build-2026-11-10.txt)
- B5: [`b5-hybrid-filters-fineweb-2026-11-10.json`](b5-hybrid-filters-fineweb-2026-11-10.json)

Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power, NEON kernel.

```bash
telusur ingest fineweb --parquet data/fineweb --out data/fineweb/corpus-1m.jsonl --limit 1000000
uv run --with fastembed scripts/embed_corpus.py data/fineweb/corpus-1m.jsonl data/fineweb/corpus-1m.fbin \
    --limit 1000000 --max-tokens 512
uv run --with fastembed scripts/embed_corpus.py data/nq/queries.jsonl data/fineweb/nq-queries.fbin --queries
telusur index --corpus data/fineweb/corpus-1m.jsonl --vectors data/fineweb/corpus-1m.fbin \
    --out indexes/fineweb-neural --keyword-fields domain --date-field crawl_date
telusur bench hybrid-filters --index indexes/fineweb-neural --base data/fineweb/corpus-1m.fbin \
    --queries data/nq/queries.jsonl --query-vectors data/fineweb/nq-queries.fbin
```

## Build

- **Docs:** the first 1M docs of the deduplicated FineWeb corpus, the same docs the 3M keyword index starts with.
- **Embed:** nomic-embed-text-v1.5 with its document prefix, 768 dims, through fastembed (ONNX) on the CPU. 1,000,000 docs took 210,128 s (58 hours) of wall clock, in one run from 7 Nov 14:46 to 10 Nov 01:09.
- **Why that long:** chunk 1 (docs 50k to 100k) took about 19 h, because the Mac slept with its lid closed from 18:24 that evening to the next morning. Chunks 2 to 19 averaged 2.0 to 2.1 h per 50k docs (from the logged rates, which have one decimal).
- **Index:** built in 140.5 s. It takes 2.1 GB on disk: f16 rerank vectors 1.54 GB, doc store 331 MB, postings 265 MB, term dictionary 42 MB, 1-bit codes 33 MB, filters 26 MB.

## Results

The queries are the 3,452 BEIR NQ test queries, run against FineWeb docs with k = 10 and the defaults (256 clusters probed, rerank depth 500). The filters are the same kind as in the keyword B5: the biggest domains, and the tightest runs of crawl days that hit each selectivity. Latency is one thread over 3,452 queries.

- **Neural recall@10** is against the exact top 10 among the allowed docs, by cosine on the full 768-dim float vectors.
- **Hybrid recall@10** is against an exact hybrid: RRF over the exhaustive keyword top 100 and the exact neural top 100, both with the same filter.
- **Path:** a filter allowing 20,000 docs or fewer is scored exactly with rerank vectors. A broader one probes 256 / selectivity clusters, at most all 2,048.

| Filter                         | Selectivity | Docs allowed | Path           | Neural recall | Neural p50 ms | Neural p99 ms | vs none | Hybrid recall | Hybrid p50 ms | Hybrid p99 ms | vs none |
| ------------------------------ | ----------- | ------------ | -------------- | ------------- | ------------- | ------------- | ------- | ------------- | ------------- | ------------- | ------- |
| none                           | 100%        | 1,000,000    | 256 clusters   | 0.822         | 0.53          | 0.64          | 1.00x   | 0.789         | 2.09          | 7.64          | 1.00x   |
| the biggest domain             | 0.11%       | 1,118        | exact          | 1.000         | 0.12          | 0.13          | 0.22x   | 1.000         | 2.05          | 4.63          | 0.98x   |
| the 20 biggest domains         | 1.02%       | 10,182       | exact          | 0.999         | 1.29          | 2.07          | 2.45x   | 1.000         | 2.40          | 5.08          | 1.15x   |
| crawl date 2020-11-01          | 0.10%       | 1,003        | exact          | 1.000         | 0.10          | 0.11          | 0.19x   | 1.000         | 1.67          | 3.05          | 0.80x   |
| crawl date 2020-10-31 to 11-01 | 1.00%       | 9,978        | exact          | 1.000         | 0.99          | 1.30          | 1.87x   | 1.000         | 2.16          | 4.38          | 1.04x   |
| crawl date 2018-08-20 to 2019-08-21 | 10.0%  | 99,964       | 2,048 clusters | 0.867         | 2.53          | 3.28          | 4.79x   | 0.809         | 2.64          | 4.24          | 1.26x   |
| crawl date 2013-05-26 to 2019-08-17 | 50.0%  | 500,043      | 512 clusters   | 0.831         | 1.03          | 1.21          | 1.96x   | 0.792         | 2.13          | 5.93          | 1.02x   |

On the exact path recall is 0.9992 to 0.9998, not 1. Probably the f16 rerank vectors flip a few near ties that the f32 truth breaks the other way.

**PRD target: latency at 1% selectivity ≤ 2x unfiltered.**

| Mode   | Domains | Crawl dates |
| ------ | ------- | ----------- |
| Hybrid | 1.15x, met | 1.04x, met |
| Neural | 2.45x, missed | 1.87x, met |

**What the numbers say:**

- **No result breaks the filter.** Across 7 filters, 3,452 queries and both modes, the bench counted 0 results outside the filter.
- **Hybrid cost barely moves with a filter.** Every filter stays within 0.80x to 1.26x of unfiltered. Neural alone is 0.53 ms, so most of hybrid's 2 ms is the keyword stage, which a filter barely changes (see the keyword B5 on FineWeb 3M).
- **Narrow neural filters are faster than no filter.** About 1,000 exact dot products beat scanning 256 clusters.
- **Neural at 1% domains misses the 2x target.** The exact path scores all 10,182 allowed docs, about 0.13 us each. Two days of crawl (9,978 docs) take 0.99 ms against 1.29 ms for 20 domains. My guess, not measured: a date range's docs sit in runs of ids, so their rerank rows are next to each other in memory, while a domain's are scattered.
- **10% is the worst case for neural, at 4.8x.** 256 / 0.1 is more than 2,048, so it probes every cluster. That also lifts its recall above unfiltered (0.867 vs 0.822). The DBpedia B5 had the same shape.
- **Unfiltered neural recall is 0.822, against 0.952 on DBpedia 1M** with the same settings. I have not looked into why. The model, the vectors (768 dims, not 1024) and the queries (NQ questions against web pages) all differ. A sweep of clusters probed and rerank depth on this index would be the first step.
- **Hybrid recall sits below neural recall** when the filter is broad. Probably because RRF reads ranks down to 100 in each list, so the approximate neural ranks below the top 10 matter too.

## History

The queue first ran this benchmark at 01:11, two minutes after the 58-hour embed finished. It was rerun at 10:20 after a cooldown and a speed check (the B3 point at 0.514 ms p50, against 0.543 ms on 7 Nov), with no sleep during the run. Recall was identical and every latency was within 3% of the first run. The numbers above are from the rerun.
