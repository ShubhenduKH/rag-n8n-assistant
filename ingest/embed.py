"""Embed chunks and build the search index.

Deliberately has no vector-database dependency. At this corpus size (~2k pages,
~5k chunks) a numpy matrix is faster than a network round-trip to a hosted
vector store, costs nothing, and makes the whole project runnable with a single
API key. Swap in pgvector or Qdrant when the corpus outgrows memory — the
retriever interface stays identical.

    python ingest/embed.py --strategy recursive-800-overlap-100
"""

import sys

# Windows defaults stdout to cp1252, which raises UnicodeEncodeError as soon as
# a scraped title contains an arrow or a smart quote — but only when output is
# redirected to a file, so it passes interactively and fails in CI.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import argparse
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.providers import embed, embed_model, provider  # noqa: E402
from ingest.corpus import load_docs                     # noqa: E402
from ingest.chunk import STRATEGIES, chunk_documents    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs.json"
BATCH = 64        # Gemini batchEmbedContents caps the request size


def index_paths(strategy: str) -> tuple[Path, Path]:
    safe = strategy.replace("/", "-")
    return (ROOT / "data" / f"index-{safe}.npy",
            ROOT / "data" / f"chunks-{safe}.json")


def main(strategy: str) -> None:
    docs = load_docs()
    chunks = chunk_documents(docs, strategy)
    print(f"{len(docs)} docs -> {len(chunks)} chunks  [{strategy}]")

    chars = sum(len(c.text) for c in chunks)
    print(f"provider={provider()}  model={embed_model()}")
    print(f"~{chars // 4:,} tokens to embed\n")

    parts = []
    for i in range(0, len(chunks), BATCH):
        batch = [c.text for c in chunks[i:i + BATCH]]
        parts.append(embed(batch))          # provider layer returns L2-normalised rows
        print(f"  {min(i + BATCH, len(chunks)):>5}/{len(chunks)}")

    matrix = np.vstack(parts)

    index_path, chunks_path = index_paths(strategy)
    np.save(index_path, matrix)
    chunks_path.write_text(
        json.dumps([{"text": c.text, "source_url": c.source_url,
                     "title": c.title, "index": c.index} for c in chunks],
                   ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nwrote {index_path.name}  {matrix.shape}")
    print(f"wrote {chunks_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="recursive-800-overlap-100",
                    choices=list(STRATEGIES))
    main(**vars(ap.parse_args()))
