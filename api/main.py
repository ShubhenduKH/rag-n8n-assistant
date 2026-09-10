"""FastAPI app: serves the demo page and the /query endpoint.

    uvicorn api.main:app --reload
"""

import json
import os
import time
from pathlib import Path

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.retrieve import get_index

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "index.html"
RESULTS = ROOT / "eval" / "results.json"
STRATEGY = os.getenv("STRATEGY", "recursive-800-overlap-100")

app = FastAPI(title="n8n docs assistant", docs_url="/api-docs")


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    title: str
    url: str
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
    mode: str


@app.get("/")
def home():
    if not WEB.exists():
        raise HTTPException(404, "web/index.html not built")
    return FileResponse(WEB)


@app.get("/scorecard")
def scorecard():
    """The published eval numbers, read straight from the harness output.

    Served from disk rather than hardcoded so the page can never claim an
    accuracy the eval did not actually produce.
    """
    if not RESULTS.exists():
        return {"status": "not measured yet"}

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    best = max(results, key=lambda r: r.get("retrieval_any_at_k", r["retrieval_at_k"]))
    return {
        "status": "measured",
        "strategy": best["strategy"],
        "questions": best["n"],
        "retrieval_at_k": best["retrieval_at_k"],
        "retrieval_any_at_k": best.get("retrieval_any_at_k"),
        "answer_correct": best.get("answer_correct"),
        "measured_on": best["measured_on"],
        "all_strategies": [
            {k: r.get(k) for k in
             ("strategy", "retrieval_at_k", "retrieval_any_at_k", "answer_correct")}
            for r in sorted(results, key=lambda r: -r.get("retrieval_any_at_k", 0))
        ],
    }


@lru_cache(maxsize=2)
def _bm25(strategy: str):
    from api.bm25 import build
    return build(strategy)


def _extractive(question: str, k: int):
    """BM25 + top passage, no generation.

    Lets the demo run with no credentials at all. The passage is returned
    verbatim rather than paraphrased, so nothing here can hallucinate — the
    honest trade is that it answers with a chunk of documentation instead of
    a sentence.
    """
    hits = _bm25(STRATEGY).search(question, k)
    if not hits:
        return "I don't know based on the n8n docs I have.", []

    best = hits[0]
    passage = _trim_to_boundary(best.text)
    text = (f"{passage}\n\n— from “{best.title}”\n\n"
            "(Retrieval-only mode: this is the best-matching passage, not a "
            "generated answer. Set a model provider in .env for generated answers.)")
    return text, hits


def _trim_to_boundary(passage: str, look: int = 200) -> str:
    """Start the passage at a sentence, or failing that a word.

    Overlapping chunks begin mid-word — a demo that opens with "equest node:"
    reads as broken even though retrieval was correct. Prefer the first sentence
    boundary within `look` characters; otherwise drop the partial first word.
    """
    passage = passage.strip()

    # Already starts cleanly — leave it alone. Trimming a good passage loses
    # its first sentence, which is usually the one that answers the question.
    if not passage or not passage[0].islower():
        return passage

    cuts = [passage.find(m, 0, look) + len(m) for m in (". ", "? ", "! ")
            if passage.find(m, 0, look) != -1]
    if cuts:
        return passage[min(cuts):].strip()

    if " " in passage[:80]:
        return passage.split(" ", 1)[1].strip()
    return passage


@app.post("/query", response_model=Answer)
def query(body: Query):
    started = time.perf_counter()
    mode = "generated"
    try:
        index = get_index(STRATEGY)
        text, hits = index.answer(body.question, body.k)
    except SystemExit:
        # No dense index or no API key — fall back rather than 503 the demo.
        mode = "extractive"
        try:
            text, hits = _extractive(body.question, body.k)
        except SystemExit as exc:
            raise HTTPException(503, str(exc)) from exc

    seen, sources = set(), []
    for h in hits:
        if h.source_url in seen:
            continue
        seen.add(h.source_url)
        sources.append(Source(title=h.title or h.source_url, url=h.source_url,
                              score=round(h.score, 3)))

    return Answer(
        answer=text,
        sources=sources,
        latency_ms=int((time.perf_counter() - started) * 1000),
        mode=mode,
    )


@app.get("/health")
def health():
    return {"ok": True, "strategy": STRATEGY}
