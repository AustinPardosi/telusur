# A NEON rerank kernel and prefetch: narrow filters within 2x (10 Nov 2026)

[#24](https://github.com/AustinPardosi/telusur/issues/24). Raw outputs:

- Rerank dot product, before and after: [`b1-rerank-dot-2026-11-10.txt`](b1-rerank-dot-2026-11-10.txt)
- B5 DBpedia: [`b5-filters-dbpedia-2026-11-10.json`](b5-filters-dbpedia-2026-11-10.json) (before: [`b5-filters-dbpedia-2026-11-05.json`](b5-filters-dbpedia-2026-11-05.json))
- B5 FineWeb: [`b5-hybrid-filters-fineweb-fastdot-2026-11-10.json`](b5-hybrid-filters-fineweb-fastdot-2026-11-10.json) (before: [`b5-hybrid-filters-fineweb-c512-2026-11-10.json`](b5-hybrid-filters-fineweb-c512-2026-11-10.json))
- B3 DBpedia: [`b3-ann-dbpedia-2026-11-10.json`](b3-ann-dbpedia-2026-11-10.json), [`b3-ann-dbpedia-extended-2026-11-10.json`](b3-ann-dbpedia-extended-2026-11-10.json)

Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power. The after runs came from one script, after a cooldown, with no sleep during it.

```bash
cargo bench --bench kernels -- rerank --warm-up-time 1 --measurement-time 3
telusur bench filters --index indexes/dbpedia --base data/dbpedia/base.fbin --queries data/dbpedia/queries.fbin
telusur bench hybrid-filters --index indexes/fineweb-neural --base data/fineweb/corpus-1m.fbin \
    --queries data/nq/queries.jsonl --query-vectors data/fineweb/nq-queries.fbin
telusur bench ann --index indexes/dbpedia --queries data/dbpedia/queries.fbin \
    --gt-full data/dbpedia/gt_full.ibin --gt-rerank data/dbpedia/gt_rerank.ibin
telusur bench ann ... --nprobe 256,512,1024 --rerank-depth 200,500,1000
```

## The problem

A filter allowing 20,000 docs or fewer skips the index and scores every allowed doc with `dot_f16` on its rerank vector. At 1% selectivity that missed the PRD target (≤ 2x unfiltered latency): 3.7x on DBpedia (2.20 vs 0.59 ms), and 1.81x to 2.45x on FineWeb.

The ticket guessed memory: each allowed doc is a random 1.5 to 2 KB read. A bench of `dot_f16` alone said otherwise:

| 10,000 dot products | Rows in cache | Random rows of 512 MB |
| ------------------- | ------------- | --------------------- |
| 768 dims            | 1.01 ms       | 1.22 ms               |
| 1024 dims           | 1.29 ms       | 1.77 ms               |

Memory added only 20% to 37%. The dot product itself was slow: 100 ns for 768 dims, under 2 dims per cycle. Timing the pieces apart found two causes:

- **The f16 conversion.** `half` converts 4 values at a time through inline assembly and small arrays, which the compiler cannot vectorize or fuse around. Converting a row took as long as the whole f32 dot product.
- **The f32 dot product.** It adds into 4 NEON registers, one add after another, so each add waits on the one before it.

## The fix

**A rerank kernel, `kernel::dot_f16`**, beside the scan kernels:

- **NEON:** `fcvtl` / `fcvtl2` convert 8 halves per pair of instructions (through `asm!`, since the f16 intrinsics are not stable yet), into 8 independent registers of fused multiply-adds.
- **AVX2:** with FMA and F16C, `vcvtph2ps` converts 8 halves at once, into 4 registers of 8 lanes.
- **Scalar:** `dot_f16_scalar`, the reference. It does the same 32 lanes of fused multiply-adds, the same halving sums and the same tail, so the SIMD versions must equal it bit for bit. The test checks that at 1 to 1024 dims, ragged lengths and an all-zero row. It passes with NEON here and with AVX2 under Rosetta.

A try with a branch-free bit trick for the conversion vectorized, but took about 7 instructions per 4 values and came out 30% slower than before.

**Prefetch.** Each rerank row is 24 to 32 cache lines. While scoring doc i, the exact filter path and the rerank ask the CPU to start loading the row of the doc 2 places ahead. On DBpedia's 1% filter, 1 and 2 docs ahead were fastest; 3, 4, 8 and 16 were slower, probably because so many pending lines fill the CPU's queue of outstanding misses.

| 10,000 dot products | Before, in cache | Before, random | After, in cache | After, random |
| ------------------- | ---------------- | -------------- | --------------- | ------------- |
| 768 dims            | 1.01 ms          | 1.22 ms        | 0.27 ms         | 0.52 ms       |
| 1024 dims           | 1.29 ms          | 1.77 ms        | 0.37 ms         | 0.96 ms       |

That bench has no prefetch; the prefetch counts in the searches below.

## B5: filtered neural search

**DBpedia 1M**, synthetic categories, defaults (256 clusters, rerank depth 500):

| Selectivity | Path           | Recall@10 | p50 ms, before → after | p99 ms, after | vs none, before → after |
| ----------- | -------------- | --------- | ---------------------- | ------------- | ----------------------- |
| none        | 256 clusters   | 0.952     | 0.588 → 0.464          | 0.62          | 1.00x → 1.00x           |
| 0.10%       | exact          | 1.000     | 0.135 → 0.054          | 0.06          | 0.23x → 0.12x           |
| 0.99%       | exact          | 1.000     | 2.202 → 0.748          | 1.24          | 3.74x → 1.61x           |
| 10.0%       | 2,048 clusters | 0.963     | 4.091 → 3.620          | 4.25          | 6.96x → 7.80x           |
| 49.9%       | 513 clusters   | 0.961     | 3.512 → 3.086          | 3.30          | 5.97x → 6.65x           |

**FineWeb 1M**, NQ queries, 512-dim codes, defaults:

| Filter                 | Docs allowed | Neural recall | Neural p50 ms, before → after | vs none, after | Hybrid p50 ms, after |
| ---------------------- | ------------ | ------------- | ----------------------------- | -------------- | -------------------- |
| none                   | 1,000,000    | 0.940         | 0.715 → 0.622                 | 1.00x          | 2.21                 |
| the biggest domain     | 1,118        | 1.000         | 0.116 → 0.049                 | 0.08x          | 2.15                 |
| the 20 biggest domains | 10,182       | 0.999         | 1.294 → 0.537                 | 0.86x          | 2.37                 |
| one day of crawl       | 1,003        | 1.000         | 0.097 → 0.044                 | 0.07x          | 1.75                 |
| two days of crawl      | 9,978        | 1.000         | 0.981 → 0.404                 | 0.65x          | 2.20                 |
| a year of crawl        | 99,964       | 0.976         | 3.827 → 3.699                 | 5.95x          | 3.78                 |
| six years of crawl     | 500,043      | 0.957         | 1.372 → 1.297                 | 2.08x          | 2.15                 |

Hybrid recall is unchanged from before (0.896 unfiltered, 1.000 on the narrow filters, 0.920 and 0.908 on the broad ones), and no result left its filter.

**PRD target met on both datasets:** at 1% selectivity neural search takes 1.61x unfiltered on DBpedia and 0.65x to 0.86x on FineWeb. Recall is the same as before to the 4th decimal: fused multiply-adds round differently from a multiply then an add, but not enough to change a top 10 here.

**Broad filters got only a little faster** (10% and 50%), so they look worse next to the faster unfiltered search: they spend their time scanning codes, not reranking. That is [#25](https://github.com/AustinPardosi/telusur/issues/25).

## B3: unfiltered neural search on DBpedia

The rerank is part of every neural search, so unfiltered search got faster too. Same rows as the README:

| Clusters probed | Rerank depth | Recall@10 (rerank / full GT) | p50 ms, before → after | QPS 4 / 10 threads, after |
| --------------- | ------------ | ---------------------------- | ---------------------- | ------------------------- |
| 32              | 0            | 0.457 / 0.441                | 0.08 → 0.07            | 47,337 / 75,497           |
| 32              | 200          | 0.839 / 0.754                | 0.17 → 0.12            | 29,746 / 44,267           |
| 128             | 200          | 0.894 / 0.800                | 0.28 → 0.23            | 15,186 / 23,535           |
| 128             | 500          | 0.932 / 0.821                | 0.40 → 0.31            | 11,644 / 18,475           |
| 256 (default)   | 500          | 0.952 / 0.835                | 0.58 → 0.45            | 7,633 / 12,005            |
| 512             | 1000         | 0.976 / 0.849                | 1.10 → 0.84            | 4,062 / 6,450             |

Recall is identical in every row. The deeper the rerank, the bigger the win: 22% at the defaults, 24% at depth 1000.
