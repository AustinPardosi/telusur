# B9: telusur BM25 vs Tantivy (7 Nov 2026)

Raw outputs:

- telusur: [`b9-telusur-bm25-2026-11-07.json`](b9-telusur-bm25-2026-11-07.json) (BEIR sets), [`b7-bm25-fineweb-2026-11-07.json`](b7-bm25-fineweb-2026-11-07.json) (FineWeb), [`b9-telusur-sizes-2026-11-07.txt`](b9-telusur-sizes-2026-11-07.txt)
- Tantivy 0.27.0: [`b9-tantivy-2026-11-07.json`](b9-tantivy-2026-11-07.json), index builds and searches

Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power.

```bash
telusur bench bm25 --data data/<set> --index indexes/<set>
cd compare/tantivy && cargo build --release
telusur-tantivy index --corpus data/<set>/corpus.jsonl --out indexes/tantivy-<set>
telusur-tantivy bench --index indexes/tantivy-<set> --data data/<set> --telusur-index indexes/<set>
```

FineWeb uses `data/fineweb/corpus.jsonl` (3M docs) with the BEIR NQ test queries, like [B7 on FineWeb](fineweb-3m-2026-11-07.md).

**Setup:**

- **Same terms.** The harness (`compare/tantivy`, its own crate) feeds docs and queries through telusur's analyzer as a custom Tantivy tokenizer. So both engines index exactly the same terms.
- **Same index contents.** One text field (title, newline, text) with term frequencies and no positions, merged into one segment. The external id is stored but not indexed, as in telusur.
- **Query:** one OR clause per distinct term, top 10. Tantivy uses its `TopDocs` collector (block-max WAND). telusur uses MaxScore over tf tiers.
- **Scoring differs.** Tantivy's BM25 is fixed at k1 1.2, b 0.75. telusur uses Anserini's k1 0.9, b 0.4. So the top 10s differ (the last column), and each engine prunes against its own scores. This compares each engine at its own scoring, not identical work.
- **Latency:** analyze plus top-10 search, one thread, at least 1,000 queries. **QPS:** whole searches, including reading the external ids, at 1 and 10 threads.

The laptop was in the middle of a long embedding job. It was paused for this run, and timing started only once a rerun of the B3 point matched the morning's.

Each cell is telusur / Tantivy:

| Dataset    | Docs      | p50 ms        | p99 ms         | QPS, 1 thread    | QPS, 10 threads   | Top 10 shared |
| ---------- | --------- | ------------- | -------------- | ---------------- | ----------------- | ------------- |
| SciFact    | 5,183     | 0.022 / 0.058 | 0.053 / 0.241  | 39,234 / 13,704  | 184,629 / 69,976  | 87%           |
| NFCorpus   | 3,633     | 0.005 / 0.007 | 0.019 / 0.060  | 148,407 / 83,930 | 628,883 / 205,296 | 76%           |
| FiQA       | 57,638    | 0.189 / 0.353 | 0.758 / 1.747  | 4,487 / 2,175    | 29,907 / 13,491   | 65%           |
| TREC-COVID | 171,332   | 0.527 / 0.816 | 2.575 / 5.703  | 1,529 / 969      | 7,617 / 4,851     | 70%           |
| FineWeb 3M | 3,000,000 | 2.39 / 3.12   | 13.96 / 25.86  | 315 / 198        | 1,798 / 1,056     | 62%           |

Index bytes: postings, term dictionary and doc lengths. For telusur that is `postings.bin`, `skips.bin`, `terms.fst` and `norms.bin`. For Tantivy it is `.idx`, `.term` and `.fieldnorm`. Neither count includes stored ids.

| Dataset    | telusur     | Tantivy       | telusur vs Tantivy |
| ---------- | ----------- | ------------- | ------------------ |
| SciFact    | 1,229,544   | 1,167,164     | +5%                |
| NFCorpus   | 921,914     | 882,332       | +4%                |
| FiQA       | 6,231,306   | 6,473,550     | -4%                |
| TREC-COVID | 22,409,851  | 23,351,917    | -4%                |
| FineWeb 3M | 926,352,664 | 1,004,611,598 | -8%                |

What the numbers say:

- **telusur is faster on every set:** 1.3x to 2.7x at p50, and Tantivy's p99 is 1.9x to 4.6x telusur's.
- **On the small sets, the gap is mostly per-query overhead.** There, even telusur's exhaustive search beats Tantivy (0.034 vs 0.058 ms p50 on SciFact).
- **FineWeb is the test of pruning.** Exhaustive search takes 17.03 ms there, and Tantivy's block-max WAND gets that down to 3.12. MaxScore over tiers gets it to 2.39, and its p99 is half of Tantivy's.
- **The indexes are about the same size.** telusur is within 5% on the BEIR sets and 8% smaller on FineWeb. Tantivy bit-packs doc ids and tfs in blocks of 128. telusur stores each tf once per tier and delta-varints the doc ids.

Caveats:

- The different BM25 parameters change the score distribution, and with it how much each engine can skip. Tantivy does not let you change k1 or b.
- Tantivy is a general engine: segments, deletes, many query types. telusur's index does one thing.

**History:** a first run also indexed the id field in Tantivy. That added one term per doc to its dictionary and postings. The harness was fixed before these numbers were taken.
