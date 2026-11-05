"""Download a BEIR dataset and write data/<name>/corpus.jsonl, queries.jsonl and qrels/test.tsv.

    python scripts/prep_beir.py scifact

With --model, also embed the corpus and the test queries and write corpus.fbin (one row per
corpus line), queries.fbin and query_ids.txt (the query id of each row):

    uv run --with fastembed scripts/prep_beir.py scifact --model nomic-ai/nomic-embed-text-v1.5

nomic-embed-text-v1.5 runs through fastembed's ONNX export (no remote model code), with its
`search_document: ` and `search_query: ` prefixes. Vectors are layer-normed over all dims as
nomic's Matryoshka recipe says, so truncating and normalizing them (which the index does)
gives nomic's shorter embeddings. --limit N embeds only the first N docs, for a quick timing.
"""

import json
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
KEEP = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
PREFIXES = {"nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: ")}


def download(name: str, out: Path) -> None:
    if all((out / m).exists() for m in KEEP):
        return
    archive = Path("data") / f"{name}.zip"
    out.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"downloading {URL.format(name)}")
        urllib.request.urlretrieve(URL.format(name), archive)
    with zipfile.ZipFile(archive) as z:
        for member in KEEP:
            dest = out / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(f"{name}/{member}") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"wrote {dest}")
    archive.unlink()


def write_fbin(path: Path, rows) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<II", *rows.shape))
        f.write(rows.astype("<f4").tobytes())


def embed(out: Path, model_name: str, limit: int | None) -> None:
    import numpy as np
    from fastembed import TextEmbedding

    doc_prefix, query_prefix = PREFIXES.get(model_name, ("", ""))
    model = TextEmbedding(model_name)
    # fastembed allows 8192 tokens; one such doc needs more attention memory than the laptop
    # has. 2048 is the context nomic trained at, and only cuts a few TREC-COVID and FiQA docs.
    model.model.tokenizer.enable_truncation(max_length=2048)

    def run(texts: list[str], what: str):
        start = time.time()
        # Sorted by length, batches carry little padding. Batches of 8 keep 2048-token
        # attention near 1.6 GB.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_rows = []
        for i, row in enumerate(model.embed([texts[i] for i in order], batch_size=8), 1):
            sorted_rows.append(row)
            if i % 10_000 == 0:
                print(f"  {i}/{len(texts)} {what}, {time.time() - start:.0f}s", flush=True)
        rows = np.empty((len(texts), len(sorted_rows[0])), dtype=np.float32)
        rows[order] = sorted_rows
        mean = rows.mean(axis=1, keepdims=True)
        rows = (rows - mean) / np.sqrt(rows.var(axis=1, keepdims=True) + 1e-5)  # layer norm
        print(f"embedded {len(texts)} {what} in {time.time() - start:.0f}s", flush=True)
        return rows

    docs = [json.loads(line) for line in open(out / "corpus.jsonl", encoding="utf-8")][:limit]
    texts = [f"{doc_prefix}{d.get('title', '')} {d.get('text', '')}".strip() for d in docs]
    judged = {line.split("\t")[0] for line in list(open(out / "qrels/test.tsv", encoding="utf-8"))[1:]}
    queries = sorted(
        (q["_id"], q["text"]) for q in map(json.loads, open(out / "queries.jsonl", encoding="utf-8")) if q["_id"] in judged
    )
    if limit:
        run(texts, "docs")
        return
    write_fbin(out / "corpus.fbin", run(texts, "docs"))
    write_fbin(out / "queries.fbin", run([f"{query_prefix}{t}" for _, t in queries], "queries"))
    (out / "query_ids.txt").write_text("".join(f"{qid}\n" for qid, _ in queries))
    print(f"wrote {out}/corpus.fbin, queries.fbin and query_ids.txt")


def main(args: list[str]) -> None:
    opt = lambda flag: args[args.index(flag) + 1] if flag in args else None
    name = args[0]
    out = Path("data") / name
    download(name, out)
    if model := opt("--model"):
        embed(out, model, int(opt("--limit")) if opt("--limit") else None)


if __name__ == "__main__":
    if not sys.argv[1:] or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    main(sys.argv[1:])
