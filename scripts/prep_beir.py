"""Download a BEIR dataset and write data/<name>/corpus.jsonl, queries.jsonl and qrels/test.tsv.

    python scripts/prep_beir.py scifact
"""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
KEEP = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]


def main(name: str) -> None:
    out = Path("data") / name
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
