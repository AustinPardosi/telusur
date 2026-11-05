"""Add a synthetic "category" field, c0 to c999 from a hash of the line number, to a corpus
for the filter benchmark: picking categories c0..c(m-1) selects about m / 1000 of the docs.

    python scripts/add_categories.py data/dbpedia/corpus.jsonl data/dbpedia/corpus.categories.jsonl
"""

import json
import sys

CATEGORIES = 1000
MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def main(src: str, dst: str) -> None:
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            doc = json.loads(line)
            doc["category"] = f"c{splitmix64(i) % CATEGORIES}"
            out.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
