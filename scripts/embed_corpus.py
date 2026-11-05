"""Embed the first N docs of a corpus.jsonl into an .fbin (row i = line i), resumably.

    uv run --with fastembed scripts/embed_corpus.py data/fineweb/corpus-1m.jsonl \\
        data/fineweb/corpus-1m.fbin --limit 1000000 --max-tokens 512

With --queries, lines are queries and get the query prefix instead (BEIR queries.jsonl).

Same model, document prefix and layer norm as prep_beir.py. Docs are embedded in chunks of
50,000; each finished chunk is saved next to the output, so a stopped run picks up at the
next chunk. The chunks are joined into the .fbin at the end.
"""

import json
import shutil
import struct
import sys
import time
from itertools import islice
from pathlib import Path

from prep_beir import PREFIXES, layer_norm

CHUNK = 50_000


def main(args: list[str]) -> None:
    opt = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
    import numpy as np
    from fastembed import TextEmbedding

    corpus, out = Path(args[0]), Path(args[1])
    limit = int(opt("--limit", "0")) or None
    model_name = opt("--model", "nomic-ai/nomic-embed-text-v1.5")
    model = TextEmbedding(model_name)
    model.model.tokenizer.enable_truncation(max_length=int(opt("--max-tokens", "512")))
    prefix = PREFIXES.get(model_name, ("", ""))["--queries" in args]
    chunk_size = int(opt("--chunk", str(CHUNK)))
    parts = out.with_name(out.name + ".parts")
    parts.mkdir(exist_ok=True)

    start, done = time.time(), 0
    with open(corpus, encoding="utf-8") as f:
        lines = islice(f, limit)
        for c in range(sys.maxsize):
            chunk = list(islice(lines, chunk_size))
            if not chunk:
                break
            part = parts / f"{c:05}.npy"
            if part.exists():
                continue
            docs = [json.loads(line) for line in chunk]
            texts = [f"{prefix}{d.get('title', '')} {d.get('text', '')}".strip() for d in docs]
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            embedded = np.stack(list(model.embed([texts[i] for i in order], batch_size=16)))
            rows = np.empty_like(embedded)
            rows[order] = embedded
            np.save(part.with_suffix(".tmp.npy"), layer_norm(rows))
            part.with_suffix(".tmp.npy").rename(part)
            done += len(texts)
            rate = done / (time.time() - start)
            print(f"chunk {c}: {c * chunk_size + len(texts)} docs, {rate:.1f} docs/s this run", flush=True)

    files = sorted(parts.glob("[0-9][0-9][0-9][0-9][0-9].npy"))
    n = sum(np.load(p, mmap_mode="r").shape[0] for p in files)
    d = np.load(files[0], mmap_mode="r").shape[1]
    with open(out, "wb") as w:
        w.write(struct.pack("<II", n, d))
        for p in files:
            w.write(np.load(p).astype("<f4").tobytes())
    shutil.rmtree(parts)
    print(f"wrote {out}: {n} x {d} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
