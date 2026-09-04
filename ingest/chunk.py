"""Chunking strategies.

Four variants so the eval can compare them. This comparison table is the whole
point of the project — it's what turns "I built a RAG bot" into "I measured
four approaches and here's which won."
"""

from dataclasses import dataclass
from typing import Callable, Iterator


MIN_CHARS = 80   # trailing fragments retrieve noisily; drop them


@dataclass
class Chunk:
    text: str
    source_url: str
    title: str
    index: int


def _clean(text: str) -> str:
    return " ".join(text.split())


def fixed(text: str, size: int = 512, overlap: int = 0) -> Iterator[str]:
    """Split on a fixed character window. Fast, dumb, breaks mid-sentence."""
    text = _clean(text)
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than size")
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            yield piece


def recursive(text: str, size: int = 800, overlap: int = 100) -> Iterator[str]:
    """Split on the largest natural boundary that fits.

    Tries paragraph breaks, then sentences, then words. Keeps semantic units
    intact far more often than `fixed`, which is usually worth several points
    of retrieval accuracy.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split(chunk: str, seps: list) -> list:
        if len(chunk) <= size:
            return [chunk] if chunk.strip() else []
        if not seps:
            return [chunk[i:i + size] for i in range(0, len(chunk), size)]

        sep, rest = seps[0], seps[1:]
        parts, buf, out = chunk.split(sep), "", []

        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    out.extend(split(buf, rest))
                buf = part
        if buf:
            out.extend(split(buf, rest))
        return out

    pieces = split(_clean(text), separators)

    if overlap and len(pieces) > 1:
        overlapped = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            overlapped.append(prev[-overlap:] + " " + cur)
        pieces = overlapped

    yield from (p for p in pieces if p.strip())


# The four variants the eval compares. Add your own and it appears in the
# results table automatically.
STRATEGIES: dict[str, Callable[[str], Iterator[str]]] = {
    "fixed-512": lambda t: fixed(t, 512, 0),
    "fixed-512-overlap-64": lambda t: fixed(t, 512, 64),
    "recursive-800": lambda t: recursive(t, 800, 0),
    "recursive-800-overlap-100": lambda t: recursive(t, 800, 100),
}


def chunk_documents(docs: list[dict], strategy: str) -> list[Chunk]:
    """docs: [{"url": ..., "title": ..., "text": ...}, ...]"""
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; have {list(STRATEGIES)}")

    fn = STRATEGIES[strategy]
    chunks = []
    for doc in docs:
        kept = 0
        for piece in fn(doc["text"]):
            if len(piece) < MIN_CHARS:
                continue
            chunks.append(Chunk(
                text=piece,
                source_url=doc["url"],
                title=doc.get("title", ""),
                index=kept,
            ))
            kept += 1
    return chunks


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    docs_path = Path("data/docs.json")
    if not docs_path.exists():
        sys.exit("run ingest/scrape.py first — data/docs.json not found")

    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    print(f"{len(docs)} documents\n")
    for name in STRATEGIES:
        chunks = chunk_documents(docs, name)
        avg = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
        print(f"  {name:<28} {len(chunks):>6} chunks   avg {avg:>6.0f} chars")
