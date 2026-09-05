"""FastAPI app: serves the demo page and the /query endpoint.

    uvicorn api.main:app --reload
"""

import json
import os
import time
from pathlib import Path

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
    best = max(results, key=lambda r: r["retrieval_at_k"])
    return {
        "status": "measured",
        "strategy": best["strategy"],
        "questions": best["n"],
        "retrieval_at_k": best["retrieval_at_k"],
        "answer_correct": best["answer_correct"],
        "measured_on": best["measured_on"],
        "all_strategies": [
            {k: r[k] for k in ("strategy", "retrieval_at_k", "answer_correct")}
            for r in sorted(results, key=lambda r: -r["retrieval_at_k"])
        ],
    }


@app.post("/query", response_model=Answer)
def query(body: Query):
    started = time.perf_counter()
    try:
        index = get_index(STRATEGY)
        text, hits = index.answer(body.question, body.k)
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
    )


@app.get("/health")
def health():
    return {"ok": True, "strategy": STRATEGY}
