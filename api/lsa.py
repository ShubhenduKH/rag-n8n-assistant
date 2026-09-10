"""Dense retrieval without an API key: latent semantic analysis over the corpus.

BM25 fails on this gold set for one specific reason — conversational questions
("how can I use now() in an expression") share only generic terms with the page
that answers them, so the right page ranks 165th out of 6,748. That is a
vocabulary-mismatch problem, and vocabulary mismatch is exactly what a dense
representation is supposed to fix.

Hosted embeddings are the usual answer. LSA is the answer you can run offline:
factor the TF-IDF matrix with a truncated SVD and both documents and queries
live in a ~256-dimensional space where terms that co-occur across the corpus
collapse onto the same axes. It is weaker than a modern sentence encoder, but it
needs no key, no download and no network, and it establishes whether *any* dense
method helps here before spending money to find out.

Implemented on scipy's sparse SVD rather than scikit-learn: one light dependency
instead of a heavy one, and the linear algebra stays visible.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from api.bm25 import Hit, tokenize

ROOT = Path(__file__).resolve().parents[1]
DIMS = 256


@dataclass
class Vocab:
    index: dict          # term -> column
    idf: np.ndarray      # per-column inverse document frequency


class LSA:
    """Drop-in for BM25: same `.search(query, k) -> list[Hit]` interface, so the
    eval harness scores it through the identical code path."""

    def __init__(self, chunks: list[dict], dims: int = DIMS):
        self.chunks = chunks
        docs = [tokenize((c.get("title", "") + " ") * 2 + c["text"]) for c in chunks]

        terms = sorted({t for d in docs for t in d})
        index = {t: i for i, t in enumerate(terms)}

        rows, cols, vals = [], [], []
        df = np.zeros(len(terms), dtype=np.float32)
        for i, doc in enumerate(docs):
            counts: dict[int, int] = {}
            for t in doc:
                counts[index[t]] = counts.get(index[t], 0) + 1
            for col, n in counts.items():
                rows.append(i)
                cols.append(col)
                # Sublinear term frequency: a term appearing 40 times is not 40x
                # more informative than one appearing once, and docs pages repeat
                # node names heavily.
                vals.append(1.0 + math.log(n))
                df[col] += 1

        n_docs = len(docs)
        idf = np.log((1 + n_docs) / (1 + df)).astype(np.float32) + 1.0
        self.vocab = Vocab(index=index, idf=idf)

        matrix = csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(n_docs, len(terms)),
        )
        matrix = matrix.multiply(idf).tocsr()
        matrix = self._l2(matrix)

        # svds returns singular values ascending; reverse so component 0 is the
        # strongest.
        k = min(dims, min(matrix.shape) - 1)
        u, s, vt = svds(matrix, k=k)
        order = np.argsort(-s)
        self.singular = s[order]
        self.terms_to_topics = vt[order].T.astype(np.float32)   # terms x k

        embeddings = (u[:, order] * self.singular).astype(np.float32)
        self.doc_vectors = self._normalise(embeddings)
        self.dims = k

    @staticmethod
    def _l2(matrix: csr_matrix) -> csr_matrix:
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        return csr_matrix(matrix.multiply(1.0 / norms[:, None]))

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab.index), dtype=np.float32)
        counts: dict[int, int] = {}
        for term in tokenize(query):
            col = self.vocab.index.get(term)
            if col is not None:
                counts[col] = counts.get(col, 0) + 1
        for col, n in counts.items():
            vec[col] = (1.0 + math.log(n)) * self.vocab.idf[col]

        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm

        projected = vec @ self.terms_to_topics
        norm = np.linalg.norm(projected)
        return projected / norm if norm else projected

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.doc_vectors @ self.embed_query(query)
        top = np.argsort(-scores)[:k]
        return [
            Hit(text=self.chunks[i]["text"],
                source_url=self.chunks[i]["source_url"],
                title=self.chunks[i]["title"],
                score=float(scores[i]))
            for i in top
        ]


def reciprocal_rank_fusion(*rankings: list[Hit], k: int = 5, c: int = 60) -> list[Hit]:
    """Combine rankings by reciprocal rank, the standard hybrid-retrieval merge.

    Score-level fusion would need BM25 and cosine scores to be commensurable,
    which they are not. RRF only uses positions, so it needs no calibration:
    each list contributes 1/(c + rank) to every document it ranks.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = f"{hit.source_url}#{hit.text[:60]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
            if key not in best or hit.score > best[key].score:
                best[key] = hit

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [best[key] for key, _ in ordered]


class Hybrid:
    """BM25 for exact identifiers, LSA for paraphrase, fused by rank."""

    def __init__(self, bm25, lsa, pool: int = 30):
        self.bm25, self.lsa, self.pool = bm25, lsa, pool

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return reciprocal_rank_fusion(
            self.bm25.search(query, self.pool),
            self.lsa.search(query, self.pool),
            k=k,
        )


def build(strategy: str = "recursive-800", dims: int = DIMS) -> LSA:
    import sys
    sys.path.insert(0, str(ROOT))
    from ingest.chunk import chunk_documents
    from ingest.corpus import load_docs

    chunks = chunk_documents(load_docs(), strategy)
    return LSA([{"text": c.text, "source_url": c.source_url,
                 "title": c.title, "index": c.index} for c in chunks], dims)
