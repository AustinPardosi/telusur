# B2: first-stage scorer ablation on DBpedia (5 Nov 2026)

Raw output: [`b2-ablate-quant-2026-11-05.json`](b2-ablate-quant-2026-11-05.json). Machine: Apple M4, 10 cores, 16 GB RAM, macOS, NEON kernel.

```bash
telusur bench ablate-quant --base data/dbpedia/base.fbin --queries data/dbpedia/queries.fbin
```

The first 200,000 DBpedia base vectors (OpenAI text-embedding-3-large) and the 1,000 held-out queries. Each scorer scans every code (no clusters) and keeps its top 100; recall@100 is how many of the true top 100 that holds. The truth is the exact top 100 by cosine at 1024 dims, the rerank dims: the docs the first stage must hand the reranker.

- **Hamming:** binary query against binary codes (agreeing bits).
- **Float table:** float query against binary codes, exactly (the dot product with the codes' +1 / -1).
- **u8 table:** the same through the u8-quantized lookup tables the kernels use (what ships).

| Dims | Hamming | Float table | u8 table |
| ---- | ------- | ----------- | -------- |
| 64   | 0.059   | 0.102       | 0.102    |
| 128  | 0.150   | 0.239       | 0.239    |
| 256  | 0.304   | 0.425       | 0.425    |
| 512  | 0.497   | 0.622       | 0.622    |

**What the numbers say:** keeping the query in float finds 1.25x (512 dims) to 1.7x (64 dims) as many of the true top 100 as Hamming at the same memory. Quantizing the tables to u8 costs nothing measurable: float and u8 differ by at most 0.00006. Longer codes keep helping: 256 dims find 42.5% of the top 100, 512 dims 62.2%.
