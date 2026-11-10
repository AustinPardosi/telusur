# B8: BM25 memory, step by step, on BEIR NQ (5 Nov 2026)

Raw output: [`b8-ablate-postings-2026-11-05.json`](b8-ablate-postings-2026-11-05.json). Machine: Apple M4, 10 cores, 16 GB RAM, macOS.

```bash
python scripts/prep_beir.py nq
telusur bench ablate-postings --corpus data/nq/corpus.jsonl --out indexes/nq
```

Corpus: 2,681,468 docs, 996,981 distinct terms, 118,649,493 postings, after the Anserini-matching analyzer (title and text as one field).

| Format                                                   | Postings      | Dictionary  | Doc lengths | Total         | Bytes per posting |
| -------------------------------------------------------- | ------------- | ----------- | ----------- | ------------- | ----------------- |
| `HashMap<String, Vec<(u32, u32)>>`                       | 1,384,299,552 | 111,227,303 | 10,725,872  | 1,506,252,727 | 12.69             |
| + one flat buffer, (offset, len) per term                | 949,195,944   | 94,450,087  | 10,725,872  | 1,054,371,903 | 8.89              |
| + delta varint doc ids                                   | 630,181,466   | 94,450,087  | 10,725,872  | 735,357,425   | 6.20              |
| + tf tiers (with headers and skips)                      | 182,256,965   | 94,450,087  | 10,725,872  | 287,432,924   | 2.42              |
| + inline singletons, FST dictionary, 1-byte norms        | 177,270,386   | 15,118,001  | 2,681,468   | 195,069,855   | 1.64              |

**The final format is 87% smaller than naive** (195 MB vs 1,506 MB, a ratio of 0.130), so it clears the PRD's 50% target.

How each row is measured, all from the same in-memory postings of one build:

- **Naive** is the build's own map: postings at the Vecs' real capacities (growth by doubling leaves slack), hash table bytes from the map's real bucket count, the term strings' real capacities, and a u32 length per doc.
- **Flat buffer, delta varints, tiers** are really encoded, term by term, with the same hash table size (term to offset and length) and term strings. The tiers row runs the index's own encoder, so it includes tier headers, `min_norm` and skip entries.
- **The last row** is the size of the index files the same build wrote: `postings.bin` + `skips.bin`, `terms.fst` (terms and offsets together), `norms.bin`.
- Dictionary = hash table + term strings, or the FST.

**One caveat on the tiers row.** To change one thing per step, the delta varint row still stores tf as a u32. With a varint tf there instead, postings take 274,232,991 bytes (2.31 per posting), so against that the tiers save 34% of the postings bytes, not 71%. Most of the drop in that row comes from not storing a 4-byte tf.

The build is one pass in memory: 113 s and 1.54 GB peak resident memory for NQ (from `/usr/bin/time -l`).
