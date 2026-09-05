"""Load the scraped corpus, preferring the compressed copy.

The corpus is committed as `data/docs.json.gz` (1.0 MB, versus 4.6 MB raw) so
that cloning the repo is enough to run the eval and the demo. Without it a
reader would have to spend twelve minutes scraping before seeing a number,
which is the fastest way to lose them.

`data/docs.json` is gitignored and takes precedence when present, so a fresh
scrape is picked up automatically without re-compressing.
"""

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "docs.json"
PACKED = ROOT / "data" / "docs.json.gz"


def load_docs() -> list[dict]:
    if RAW.exists():
        return json.loads(RAW.read_text(encoding="utf-8"))
    if PACKED.exists():
        with gzip.open(PACKED, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    raise SystemExit(
        "no corpus found — run `python ingest/scrape.py` "
        f"(expected {RAW.name} or {PACKED.name} in data/)"
    )


def pack() -> None:
    """Compress a fresh scrape for committing."""
    docs = json.loads(RAW.read_text(encoding="utf-8"))
    blob = json.dumps(docs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    PACKED.write_bytes(gzip.compress(blob, 9))
    print(f"{len(docs)} pages · {RAW.stat().st_size / 1e6:.1f} MB "
          f"-> {PACKED.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    pack()
