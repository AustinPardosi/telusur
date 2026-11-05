"""Download DBpedia 1M with OpenAI text-embedding-3-large vectors and write data/dbpedia/:
base.fbin (the first 999,000 rows), queries.fbin (the last 1,000), corpus.jsonl (the base
docs in row order, so line = doc id = row) and queries.jsonl.

The embedding column is detected: the widest list-of-floats column (the dataset also has a
1536-dim ada-002 column), or name one with --column. Shards are downloaded one at a time and
deleted once converted, so the 24.8 GB of parquet never sits on disk whole.

    uv run --with pyarrow --with numpy scripts/prep_dbpedia.py [--column NAME] [--shards N]

--shards N stops after the first N shards, for a quick try.
"""

import collections
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
API = f"https://huggingface.co/api/datasets/{DATASET}"
FILE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{{}}"
HOLDOUT = 1000


def embedding_column(table: pa.Table) -> str:
    lists = [f.name for f in table.schema if pa.types.is_list(f.type) and pa.types.is_floating(f.type.value_type)]
    if not lists:
        sys.exit("no list-of-floats column found")
    return max(lists, key=lambda name: len(table.column(name)[0]))


def main(column: str | None, shards: int | None) -> None:
    out = Path("data/dbpedia")
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API) as r:
        files = sorted(s["rfilename"] for s in json.load(r)["siblings"] if s["rfilename"].endswith(".parquet"))
    files = files[:shards]

    base = open(out / "base.fbin", "wb")
    base.write(struct.pack("<II", 0, 0))  # patched at the end
    corpus = open(out / "corpus.jsonl", "w")
    held = collections.deque()  # the last HOLDOUT rows seen, written as queries at the end
    n, dims = 0, None
    for i, name in enumerate(files):
        shard = out / "shard.parquet"
        urllib.request.urlretrieve(FILE.format(name), shard)
        table = pq.read_table(shard)
        if column is None:
            column = embedding_column(table)
            print(f"embedding column: {column}", flush=True)
        col = table.column(column).combine_chunks()
        lengths = pc.list_value_length(col)
        if dims is None:
            dims = pc.min(lengths).as_py()
        if pc.min(lengths).as_py() != dims or pc.max(lengths).as_py() != dims:
            sys.exit(f"{name}: vectors are not all {dims} dims")
        vectors = col.flatten().to_numpy(zero_copy_only=False).astype("<f4").reshape(-1, dims)
        for doc_id, title, text, vector in zip(
            table.column("_id").to_pylist(), table.column("title").to_pylist(), table.column("text").to_pylist(), vectors
        ):
            held.append((json.dumps({"_id": doc_id, "title": title or "", "text": text or ""}), vector))
            if len(held) > HOLDOUT:
                doc, v = held.popleft()
                corpus.write(doc + "\n")
                base.write(v.tobytes())
                n += 1
        shard.unlink()
        print(f"{i + 1}/{len(files)} shards, {n} base rows", flush=True)

    base.seek(0)
    base.write(struct.pack("<II", n, dims))
    base.close()
    corpus.close()
    with open(out / "queries.fbin", "wb") as q, open(out / "queries.jsonl", "w") as qj:
        q.write(struct.pack("<II", len(held), dims))
        for doc, v in held:
            q.write(v.tobytes())
            qj.write(doc + "\n")
    print(f"wrote {n} base rows and {len(held)} queries of {dims} dims to {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = lambda flag: args[args.index(flag) + 1] if flag in args else None
    main(opt("--column"), int(opt("--shards")) if opt("--shards") else None)
