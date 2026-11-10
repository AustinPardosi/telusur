# Broad neural filters: a multi-way union and a gentler probe (10 Nov 2026)

[#25](https://github.com/AustinPardosi/telusur/issues/25). Raw outputs:

- The sweep of the probe rule: [`b5-probe-sweep-2026-11-10.json`](b5-probe-sweep-2026-11-10.json)
- B5 DBpedia: [`b5-filters-dbpedia-sqrt-2026-11-10.json`](b5-filters-dbpedia-sqrt-2026-11-10.json) (before: [`b5-filters-dbpedia-2026-11-10.json`](b5-filters-dbpedia-2026-11-10.json))
- B5 FineWeb: [`b5-hybrid-filters-fineweb-sqrt-2026-11-10.json`](b5-hybrid-filters-fineweb-sqrt-2026-11-10.json) (before: [`b5-hybrid-filters-fineweb-fastdot-2026-11-10.json`](b5-hybrid-filters-fineweb-fastdot-2026-11-10.json))

Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power. The B5 runs came after a cooldown, with no sleep; the sweep ran without one, so compare its latencies only with each other.

```bash
telusur bench filters --index indexes/dbpedia --base data/dbpedia/base.fbin --queries data/dbpedia/queries.fbin
telusur bench hybrid-filters --index indexes/fineweb-neural --base data/fineweb/corpus-1m.fbin \
    --queries data/nq/queries.jsonl --query-vectors data/fineweb/nq-queries.fbin
```

## The problem

After [#24](rerank-speedup-2026-11-10.md), narrow filters were fast, but a filter allowing more than 20,000 docs still cost 6.6x to 7.8x unfiltered on DBpedia, and 5.95x at 10% on FineWeb. The ticket blamed the probe: `256 / selectivity` clusters is every cluster at 10%.

The probe was half of it. Timing the filter resolution alone, outside search, found the other half on DBpedia:

| DBpedia filter       | Resolving the filter | Whole query |
| -------------------- | -------------------- | ----------- |
| 10% (100 categories) | 1.48 ms              | 3.62 ms     |
| 50% (500 categories) | 2.31 ms              | 3.09 ms     |

A filter that allows any of several values ORed their bitmaps one at a time. Each `|=` re-merges the growing result, so the cost grows with the number of values times the result size. FineWeb's date ranges were already cheap (0.20 to 0.25 ms) thanks to the segment tree from #22.

## The fixes

**One multi-way union** (`roaring::MultiOps`) for "any of these values": the same set, built once. 100 categories now resolve in 0.21 ms instead of 1.48, 500 in 0.69 instead of 2.31.

**Probe `256 / sqrt(selectivity)` clusters** instead of `256 / selectivity`. A sweep of `256 / selectivity^a`, with the union fix in:

| a    | DBpedia 10%: clusters, recall, p50 | DBpedia 50%      | FineWeb 10%: recall, p50 | FineWeb 50%   |
| ---- | ---------------------------------- | ---------------- | ------------------------ | ------------- |
| 1    | 2,048, 0.963, 2.34 ms              | 513, 0.961, 1.46 | 0.976, 3.75 ms           | 0.957, 1.30   |
| 0.75 | 1,438, 0.963, 1.86                 | 431, 0.959, 1.35 |                          |               |
| 0.5  | 809, 0.960, 1.23                   | 363, 0.957, 1.29 | 0.971, 1.90              | 0.951, 1.07   |
| 0.35 | 573, 0.957, 1.00                   | 327, 0.956, 1.27 | 0.965, 1.52              | 0.948, 0.99   |
| 0.25 | 456, 0.954, 0.87                   | 305, 0.954, 1.22 | 0.960, 1.34              | 0.947, 0.96   |

Unfiltered recall at the same settings is 0.952 on DBpedia and 0.940 on FineWeb. Every value tried keeps filtered recall above it. At a = 0.25 the margin on DBpedia's 50% filter is 0.002, about the noise of 10,000 query results, so I took a = 0.5, the square root, which keeps clear margins on both datasets. 0.25 is the next step if broad filters need to be faster still.

## B5 after both fixes

**DBpedia 1M**, synthetic categories:

| Selectivity | Path                   | Recall@10 | p50 ms, before → after | p99 ms, after | vs none, before → after |
| ----------- | ---------------------- | --------- | ---------------------- | ------------- | ----------------------- |
| none        | 256 clusters           | 0.952     | 0.464 → 0.453          | 0.61          | 1.00x → 1.00x           |
| 0.10%       | exact                  | 1.000     | 0.054 → 0.054          | 0.06          | 0.12x → 0.12x           |
| 0.99%       | exact                  | 1.000     | 0.748 → 0.698          | 1.16          | 1.61x → 1.54x           |
| 10.0%       | 2,048 → 809 clusters   | 0.963 → 0.960 | 3.620 → 1.221      | 1.52          | 7.80x → 2.70x           |
| 49.9%       | 513 → 363 clusters     | 0.961 → 0.957 | 3.086 → 1.280      | 1.63          | 6.65x → 2.83x           |

**FineWeb 1M**, NQ queries, 512-dim codes:

| Filter             | Neural recall  | Neural p50 ms, before → after | vs none, after | Hybrid recall  | Hybrid p50 ms, before → after |
| ------------------ | -------------- | ----------------------------- | -------------- | -------------- | ----------------------------- |
| none               | 0.940          | 0.622 → 0.620                 | 1.00x          | 0.896          | 2.21 → 2.16                   |
| a year of crawl    | 0.976 → 0.971  | 3.699 → 1.902                 | 3.07x          | 0.920 → 0.915  | 3.78 → 2.07                   |
| six years of crawl | 0.957 → 0.951  | 1.297 → 1.052                 | 1.70x          | 0.908 → 0.902  | 2.15 → 2.17                   |

The narrow filters did not change (they skip the probe), and no result left its filter.

**What the numbers say:**

- **Broad filters cost 2.7x to 2.8x unfiltered on DBpedia, down from 6.6x to 7.8x,** and filtered recall stays above unfiltered recall (0.957 to 0.960 against 0.952).
- **On FineWeb the 10% filter halves** (3.70 → 1.90 ms neural), and hybrid at 10% is now as fast as unfiltered hybrid.
- **What is left is mostly the bigger probe.** 809 clusters scan about 3x the codes of 256. Each scanned doc the filter excludes still costs a score, and a bitmap check if it scores high enough.
- The PRD's design said `nprobe / selectivity`; telusur now uses the square root, for the reasons above.
