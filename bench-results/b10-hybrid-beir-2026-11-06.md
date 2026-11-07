# B10: keyword, neural and hybrid quality on BEIR (6 Nov 2026)

Raw output: [`b10-hybrid-beir-2026-11-06.json`](b10-hybrid-beir-2026-11-06.json). Machine: Apple M4, 10 cores, 16 GB RAM, macOS.

```bash
uv run --with fastembed scripts/prep_beir.py <set> --model nomic-ai/nomic-embed-text-v1.5
telusur index --corpus data/<set>/corpus.jsonl --vectors data/<set>/corpus.fbin --out indexes/<set>
telusur eval beir --data data/<set> --index indexes/<set> --query-vectors data/<set>/queries.fbin
```

Setup:

- **Keyword:** BM25, the same as [B6](b6-bm25-parity-2026-11-05.md).
- **Neural:** nomic-embed-text-v1.5 on CPU, 768 dims, docs cut at 2048 tokens, with nomic's document and query prefixes. It searches the index at its defaults: 256-dim 1-bit codes, nprobe 256, and the top 1000 reranked on the full 768-dim f16 vectors.
- **Hybrid:** RRF (k = 60) over the top 1000 of each.

Eval rules are the same as B6.

nDCG@10:

| Dataset    | Queries | Keyword | Neural | Hybrid | Hybrid vs better single |
| ---------- | ------- | ------- | ------ | ------ | ----------------------- |
| SciFact    | 300     | 67.89   | 70.33  | 73.85  | +3.52                   |
| NFCorpus   | 323     | 32.05   | 34.56  | 35.83  | +1.27                   |
| FiQA       | 648     | 23.61   | 37.48  | 35.24  | -2.24                   |
| TREC-COVID | 50      | 59.35   | 82.19  | 81.50  | -0.69                   |
| Average    |         | 45.72   | 56.14  | 56.60  |                         |

Recall@100:

| Dataset    | Keyword | Neural | Hybrid |
| ---------- | ------- | ------ | ------ |
| SciFact    | 92.53   | 93.00  | 95.50  |
| NFCorpus   | 24.57   | 29.32  | 30.73  |
| FiQA       | 53.95   | 66.89  | 66.77  |
| TREC-COVID | 10.91   | 14.48  | 14.92  |

**PRD target missed:** the target is hybrid ≥ the better of keyword and neural on 3 of 4 sets. Hybrid beats it on 2 of 4 (SciFact and NFCorpus).

Hybrid helps where the two retrievers are close: they are within 2.5 points of each other on SciFact and NFCorpus. It loses a little where BM25 is far behind: 14 points on FiQA and 23 on TREC-COVID. Plain RRF weights both lists the same, so a much weaker list pulls the fused ranking down. Hybrid still has the best average nDCG@10, and the best Recall@100 on 3 of 4 sets.

Nothing here tunes the fusion (weights, depth or k). Doing so on these test queries would be fitting to the test set.
