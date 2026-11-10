# Why FineWeb neural recall was 0.82, and 512-dim codes (10 Nov 2026)

[#23](https://github.com/AustinPardosi/telusur/issues/23). Raw outputs:

- B3 sweeps, 256-dim codes: [`b3-ann-fineweb-nq-2026-11-10.json`](b3-ann-fineweb-nq-2026-11-10.json), [`b3-ann-fineweb-holdout-2026-11-10.json`](b3-ann-fineweb-holdout-2026-11-10.json)
- B3 sweep, 512-dim codes: [`b3-ann-fineweb-nq-c512-2026-11-10.json`](b3-ann-fineweb-nq-c512-2026-11-10.json)
- B2: [`b2-ablate-quant-fineweb-nq-2026-11-10.json`](b2-ablate-quant-fineweb-nq-2026-11-10.json), [`b2-ablate-quant-fineweb-holdout-2026-11-10.json`](b2-ablate-quant-fineweb-holdout-2026-11-10.json)
- B5 on the 512-dim index: [`b5-hybrid-filters-fineweb-c512-2026-11-10.json`](b5-hybrid-filters-fineweb-c512-2026-11-10.json)

Machine: Apple M4, 10 cores, 16 GB RAM, macOS, AC power, NEON kernel.

```bash
# control queries: 1,000 FineWeb docs that are not in the 1M index, with the document prefix
sed -n '1000001,1001000p' data/fineweb/corpus.jsonl > data/fineweb/holdout-1k.jsonl
uv run --with fastembed scripts/embed_corpus.py data/fineweb/holdout-1k.jsonl data/fineweb/holdout-1k.fbin
telusur gt --base data/fineweb/corpus-1m.fbin --queries data/fineweb/nq-queries.fbin --out data/fineweb/gt-nq
telusur bench ann --index indexes/fineweb-neural --queries data/fineweb/nq-queries.fbin \
    --gt-full data/fineweb/gt-nq/gt_full.ibin --gt-rerank data/fineweb/gt-nq/gt_rerank.ibin \
    --nprobe 64,128,256,512,1024,2048 --rerank-depth 0,500,1000,2000,5000 --threads 10
telusur bench ablate-quant --base data/fineweb/corpus-1m.fbin --queries data/fineweb/nq-queries.fbin
# the same for the holdout queries, then the 512-dim build and its sweep
telusur index --corpus data/fineweb/corpus-1m.jsonl --vectors data/fineweb/corpus-1m.fbin \
    --out indexes/fineweb-neural --keyword-fields domain --date-field crawl_date --d-code 512
```

The question: at the defaults (256 clusters probed, rerank depth 500), neural recall@10 on FineWeb 1M was 0.822, against 0.952 on DBpedia 1M. Recall here is against the exact top 10 by cosine at 768 dims, which are both the full and the rerank dims.

## Routing is not the problem

Recall@10 with the NQ queries, 256-dim codes:

| Clusters probed | Depth 0 | Depth 500 | Depth 1000 | Depth 2000 | Depth 5000 |
| --------------- | ------- | --------- | ---------- | ---------- | ---------- |
| 64              | 0.274   | 0.788     | 0.834      | 0.864      | 0.881      |
| 128             | 0.278   | 0.812     | 0.865      | 0.902      | 0.927      |
| 256             | 0.278   | 0.822     | 0.879      | 0.920      | 0.952      |
| 512             | 0.278   | 0.827     | 0.885      | 0.929      | 0.965      |
| 1024            | 0.278   | 0.828     | 0.887      | 0.933      | 0.971      |
| 2048 (all)      | 0.279   | 0.828     | 0.887      | 0.934      | 0.972      |

Probing every cluster at depth 500 only gets to 0.828. Reranking deeper is what helps: at 256 clusters, depth 5,000 reaches 0.952. So the true top 10 is in the probed clusters, but often not among the 500 best code scores. The 1-bit first stage is the weak part.

## The codes are weaker on these vectors

B2 scans the first 200,000 docs with every scorer and asks how much of the exact top 100 each scorer's top 100 holds. The u8 table is what ships.

| Code dims | DBpedia (OpenAI) | FineWeb, NQ queries | FineWeb, holdout docs |
| --------- | ---------------- | ------------------- | --------------------- |
| 64        | 0.102            | 0.099               | 0.111                 |
| 128       | 0.239            | 0.171               | 0.194                 |
| 256       | 0.425            | 0.281               | 0.318                 |
| 512       | 0.622            | 0.438               | 0.477                 |

256 binarized dims of nomic-embed-text-v1.5 keep its 768-dim ranking much worse than 256 binarized dims of text-embedding-3-large keep its 1024-dim ranking. On FineWeb it takes 512 dims to match what 256 dims do on DBpedia.

## The queries are a small part

The holdout queries are 1,000 FineWeb docs that are not in the index, embedded with the document prefix, so they look like the base, as DBpedia's held-out queries do. At the defaults they reach 0.845, against 0.822 for NQ questions. Better, but nowhere near 0.952. Questions as queries cost a little recall; the codes explain most of the gap.

## Fix: 512-dim codes for this index

The same sweep on an index built with `--d-code 512`, NQ queries:

| Clusters probed | Depth 500       | Depth 1000      | Depth 2000      |
| --------------- | --------------- | --------------- | --------------- |
| 64              | 0.879, 0.37 ms  | 0.891, 0.57 ms  | 0.895, 0.94 ms  |
| 128             | 0.916, 0.49 ms  | 0.931, 0.68 ms  | 0.937, 1.07 ms  |
| 256             | 0.940, 0.71 ms  | 0.957, 0.91 ms  | 0.965, 1.31 ms  |
| 512             | 0.955, 1.13 ms  | 0.973, 1.35 ms  | 0.982, 1.75 ms  |
| 1024            | 0.962, 1.95 ms  | 0.981, 2.19 ms  | 0.991, 2.62 ms  |

Compared at the same p50 latency:

| p50     | 256-dim codes, best cell         | 512-dim codes                    |
| ------- | -------------------------------- | -------------------------------- |
| ~0.5 ms | 0.822 (256 clusters, depth 500)  | 0.916 (128 clusters, depth 500)  |
| ~0.7 ms | 0.879 (256, 1000)                | 0.940 (256, 500: the defaults)   |
| ~0.9 ms | 0.902 (128, 2000)                | 0.957 (256, 1000)                |
| 0.95    | needs depth 5,000: 2.39 ms       | 0.91 ms                          |

**512-dim codes win at every latency.** At the defaults they give 0.940 at 0.71 ms. Reaching 0.95 takes 0.91 ms instead of 2.39 ms.

**The cost is memory:** the hot part (codes, ids, centroids) goes from 39.3 MB to 74.4 MB, or 74 bytes per vector instead of 39. The 1.5 GB of rerank vectors stay memory-mapped as before, and the build took 154 s instead of 140.

**Recommendation:** build FineWeb (nomic-embed-text-v1.5) with `--d-code 512`, and keep 256 as the default, since it already reaches 0.952 on DBpedia. The right code size depends on the embedding model, so B2 is the check to run on a new model before building. `indexes/fineweb-neural` is now the 512-dim build.
