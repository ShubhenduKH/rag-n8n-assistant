"""Embed chunks and build the search index.

Deliberately has no vector-database dependency. At this corpus size (~2k pages,
~5k chunks) a numpy matrix is faster than a network round-trip to a hosted
vector store, costs nothing, and makes the whole project runnable with a single
API key. Swap in pgvector or Qdrant when the corpus outgrows memory — the
retriever interface stays identical.

    python ingest/embed.py --strategy recursive-800-overlap-100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import STRATEGIES, chunk_documents  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs.json"
BATCH = 128


def index_paths(strategy: str) -> tuple[Path, Path]:
    safe = strategy.replace("/", "-")
    return (ROOT / "data" / f"index-{safe}.npy",
            ROOT / "data" / f"chunks-{safe}.json")


def embed_batch(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:                      # noqa: BLE001
            wait = 2 ** attempt
            print(f"    retry in {wait}s — {type(exc).__name__}: {exc}")
            time.sleep(wait)
    raise SystemExit("embedding failed after 5 attempts")


def main(strategy: str) -> None:
    if not DOCS.exists():
        raise SystemExit("run ingest/scrape.py first — data/docs.json missing")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — copy .env.example to .env")

    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI()

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    chunks = chunk_documents(docs, strategy)
    print(f"{len(docs)} docs -> {len(chunks)} chunks  [{strategy}]")

    chars = sum(len(c.text) for c in chunks)
    print(f"~{chars // 4:,} tokens, about ${chars / 4 / 1_000_000 * 0.02:.3f} to embed\n")

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH):
        batch = [c.text for c in chunks[i:i + BATCH]]
        vectors.extend(embed_batch(client, batch, model))
        print(f"  {min(i + BATCH, len(chunks)):>5}/{len(chunks)}")

    matrix = np.asarray(vectors, dtype=np.float32)
    # Pre-normalise so retrieval is a single dot product instead of a division
    # per query. Same result, measurably faster at query time.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

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
