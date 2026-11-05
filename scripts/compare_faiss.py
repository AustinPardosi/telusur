"""B4: FAISS and hnswlib on DBpedia 1M, the queries and ground truth of B3, on this machine.

    uv run --with faiss-cpu --with hnswlib --with numpy scripts/compare_faiss.py data/dbpedia \\
        > bench-results/b4-ann-vs-others-<date>.json

Every engine gets what telusur reranks with: the first 1024 dims of each vector, normalized
(inner product = cosine), so gt_rerank.ibin is the exact answer for all of them.

- FAISS IndexIVFFlat, 2048 lists (telusur's default for 1M), nprobe swept.
- FAISS IndexBinaryIVF on telusur's codes (first 256 dims, bit = value > 0, LSB first),
  2048 lists, nprobe swept, alone and with the top 500 reranked exactly at 1024 dims (telusur's
  design, with FAISS's Hamming scan in place of the lookup-table kernel).
- hnswlib HNSW, M 16, ef_construction 200, ef swept.

Per setting: recall@10 against both ground truths and single-thread latency (one query at a
time). At the cheapest setting reaching recall@10 >= 0.95 (telusur's default reaches 0.952):
QPS at 4 and 10 threads, one query per call from a thread pool, the library's own threads
set to 1 (how telusur runs), and index memory. Configs run one at a time to bound memory.
--limit N uses the first N base rows and computes its own ground truth, for a quick check.
"""

import concurrent.futures as cf
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import faiss
import hnswlib
import numpy as np

D, D_CODE, NLIST, K, DEPTH, TARGET = 1024, 256, 2048, 10, 500, 0.95


def fbin(path: Path):
    n, d = struct.unpack("<II", open(path, "rb").read(8))
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(n, d))


def ibin(path: Path):
    n, k = struct.unpack("<II", open(path, "rb").read(8))
    return np.fromfile(path, dtype="<u4", offset=8).reshape(n, k)


def prefix_normalized(rows, limit=None, chunk=100_000):
    n = min(limit or len(rows), len(rows))
    out = np.empty((n, D), dtype=np.float32)
    for i in range(0, n, chunk):
        x = np.asarray(rows[i : min(i + chunk, n), :D], dtype=np.float32)
        out[i : i + len(x)] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out


def codes(x):
    return np.packbits(x[:, :D_CODE] > 0, axis=1, bitorder="little")


def recall(found, truth):
    return float(np.mean([len(set(f[:K]) & set(t[:K])) / K for f, t in zip(found, truth)]))


def latency(one, queries):
    for q in queries[:50]:
        one(q)  # warm-up
    times = []
    for q in queries:
        t = time.perf_counter()
        one(q)
        times.append((time.perf_counter() - t) * 1e3)
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.99)]


def qps(one, queries, threads, passes=5):
    work = [q for _ in range(passes) for q in queries]
    with cf.ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, work[: len(queries)]))  # warm-up
        t = time.perf_counter()
        list(pool.map(one, work, chunksize=64))
    return len(work) / (time.perf_counter() - t)


def file_bytes(write):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        path = os.path.join(d, "index")
        write(path)
        return os.path.getsize(path)


def sweep(name, settings, set_knob, one, queries, gts, extra_bytes=0, write=None):
    rows = []
    for s in settings:
        set_knob(s)
        found = np.stack([one(q) for q in queries])
        p50, p99 = latency(one, queries)
        rows.append({"setting": s, "recall_at_10_rerank_gt": recall(found, gts[0]),
                     "recall_at_10_full_gt": recall(found, gts[1]) if gts[1] is not None else None,
                     "p50_ms": p50, "p99_ms": p99})
        print(f"{name} {s}: recall {rows[-1]['recall_at_10_rerank_gt']:.3f}, p50 {p50:.3f} ms",
              file=sys.stderr, flush=True)
    out = {"engine": name, "sweep": rows}
    best = next((r for r in rows if r["recall_at_10_rerank_gt"] >= TARGET), None)
    if best:
        set_knob(best["setting"])
        out["at_target"] = {"setting": best["setting"],
                            "qps": {t: qps(one, queries, t) for t in (4, 10)}}
    if write:
        out["index_bytes"] = file_bytes(write) + extra_bytes
    return out


def machine():
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0)
    return f"{cpu or platform.processor()}, {os.cpu_count()} cores, {ram // 2**30} GB RAM, {platform.system()} {platform.machine()}"


def main(args):
    data = Path(args[0])
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    faiss.omp_set_num_threads(1)
    x = prefix_normalized(fbin(data / "base.fbin"), limit)
    q = prefix_normalized(fbin(data / "queries.fbin"))
    if limit:
        gts = (np.argsort(-(q @ x.T), axis=1)[:, :100], None)
    else:
        gts = (ibin(data / "gt_rerank.ibin"), ibin(data / "gt_full.ibin"))
    n, nlist = len(x), min(NLIST, len(x) // 39)
    report = {"bench": "ann-vs-others", "base": len(x), "queries": len(q), "dims": D, "k": K,
              "faiss": faiss.__version__, "hnswlib": metadata.version("hnswlib"),
              "machine": machine(), "engines": []}
    probes = [p for p in (8, 16, 32, 64, 128, 256, 512) if p <= nlist]

    queries = list(range(len(q)))  # each engine's search takes a query's row number
    t = time.time()
    quantizer = faiss.IndexFlatIP(D)  # kept referenced: the index does not own it
    ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss.omp_set_num_threads(os.cpu_count())
    ivf.train(x)
    ivf.add(x)
    faiss.omp_set_num_threads(1)
    build = time.time() - t
    one = lambda i: ivf.search(q[i : i + 1], K)[1][0]
    r = sweep("faiss IndexIVFFlat", probes, lambda p: setattr(ivf, "nprobe", p), one, queries, gts,
              write=lambda p: faiss.write_index(ivf, p))
    r["build_s"] = build
    report["engines"].append(r)
    del ivf, quantizer

    t = time.time()
    xb, qb = codes(x), codes(q)
    bquantizer = faiss.IndexBinaryFlat(D_CODE)
    bivf = faiss.IndexBinaryIVF(bquantizer, D_CODE, nlist)
    faiss.omp_set_num_threads(os.cpu_count())
    bivf.train(xb)
    bivf.add(xb)
    faiss.omp_set_num_threads(1)
    build = time.time() - t
    def binary(i, depth):
        ids = bivf.search(qb[i : i + 1], depth)[1][0]
        ids = ids[ids >= 0]
        if depth > K:
            ids = ids[np.argsort(-(x[ids] @ q[i]), kind="stable")[:K]]
        return np.pad(ids, (0, K - len(ids)), constant_values=-1)
    write = lambda p: faiss.write_index_binary(bivf, p)
    for depth, label in ((K, "faiss IndexBinaryIVF"), (DEPTH, f"faiss IndexBinaryIVF + rerank {DEPTH}")):
        r = sweep(label, probes, lambda p: setattr(bivf, "nprobe", p), lambda i: binary(i, depth),
                  queries, gts, extra_bytes=x.nbytes if depth > K else 0, write=write)
        r["build_s"] = build
        report["engines"].append(r)
    del bivf, bquantizer

    t = time.time()
    hnsw = hnswlib.Index(space="ip", dim=D)
    hnsw.init_index(max_elements=n, M=16, ef_construction=200, random_seed=42)
    hnsw.add_items(x, np.arange(n), num_threads=os.cpu_count())
    hnsw.set_num_threads(1)
    build = time.time() - t
    one = lambda i: hnsw.knn_query(q[i : i + 1], k=K, num_threads=1)[0][0].astype(np.int64)
    r = sweep("hnswlib HNSW (M 16, ef_construction 200)", [16, 32, 64, 128, 256, 512],
              hnsw.set_ef, one, queries, gts, write=hnsw.save_index)
    r["build_s"] = build
    report["engines"].append(r)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
