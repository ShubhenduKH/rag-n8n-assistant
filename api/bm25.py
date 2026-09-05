"""BM25 lexical retrieval — no API key, no model download, no network.

This is the baseline the embedding retriever has to beat. It exists for three
reasons:

  1. It runs anywhere, immediately, for free. The project is demoable and the
     eval is runnable before any credential exists.
  2. BM25 is a genuinely strong baseline on technical documentation, where
     users search with the same nouns the docs use ("HTTP Request node",
     "GENERIC_TIMEZONE"). Publishing a RAG accuracy number without comparing
     against it is how people accidentally claim credit for lexical matching.
  3. It is the honest half of a hybrid retriever. Dense embeddings lose exact
     identifiers; BM25 loses paraphrase. Production wants both.

Implementation is Okapi BM25 over numpy sparse counts — roughly 60 lines, no
scikit-learn, no rank_bm25 dependency.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

K1 = 1.5      # term-frequency saturation
B = 0.75      # length normalisation

TOKEN = re.compile(r"[a-z0-9_]+")

# Dropping these costs nothing and stops "how do I" matching every page.
STOP = frozenset("""
a an and are as at be by for from has have how i if in into is it its of on or
that the this to was what when where which who why will with you your do does
did can could should would there their them then than these those
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed.

    Underscores are kept inside tokens so environment variables like
    N8N_ENCRYPTION_KEY survive as one term instead of fragmenting.
    """
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Hit:
    text: str
    source_url: str
    title: str
    score: float


class BM25:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # Title text is worth repeating: a page titled "Error handling" should
        # rank for "error" even if the body says "failure" throughout.
        docs = [tokenize((c.get("title", "") + " ") * 2 + c["text"]) for c in chunks]

        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, count))

        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        norm = K1 * (1 - B + B * self.lengths / (self.avg_len or 1.0))

        for term in tokenize(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_id, freq in plist:
                scores[doc_id] += idf * (freq * (K1 + 1)) / (freq + norm[doc_id])

        top = np.argsort(-scores)[:k]
        return [
            Hit(text=self.chunks[i]["text"],
                source_url=self.chunks[i]["source_url"],
                title=self.chunks[i]["title"],
                score=float(scores[i]))
            for i in top if scores[i] > 0
        ]


def build(strategy: str) -> BM25:
    """Build straight from docs.json — no embedding step required."""
    import sys
    sys.path.insert(0, str(ROOT))
    from ingest.chunk import chunk_documents

    docs_path = ROOT / "data" / "docs.json"
    if not docs_path.exists():
        raise SystemExit("run ingest/scrape.py first")

    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    chunks = chunk_documents(docs, strategy)
    return BM25([{"text": c.text, "source_url": c.source_url,
                  "title": c.title, "index": c.index} for c in chunks])


if __name__ == "__main__":
    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "recursive-800-overlap-100"
    index = build(strategy)
    print(f"{len(index.chunks)} chunks · {len(index.postings)} unique terms\n")

    for q in ["how do I retry a failed HTTP request node",
              "self host with docker compose",
              "GENERIC_TIMEZONE environment variable"]:
        print(f"  {q}")
        for hit in index.search(q, 3):
            print(f"     {hit.score:6.2f}  {hit.source_url[:70]}")
        print()
