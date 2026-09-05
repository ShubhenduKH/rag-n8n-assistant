"""Retrieval and answer generation.

Kept separate from the web layer so the eval harness can import it directly and
score the exact same code path that serves users. If the eval tests a different
pipeline than production, the number it prints is fiction.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from api.providers import chat, chat_model, embed

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = """You answer questions about n8n using only the documentation excerpts provided.

Rules:
- Answer only from the excerpts. Do not use outside knowledge about n8n.
- If the excerpts do not contain the answer, say exactly: "I don't know based on the n8n docs I have."
- Be concise and concrete. Name the setting, node or environment variable involved.
- Cite the source URLs you used at the end as a plain list.

An honest "I don't know" is correct behaviour, not a failure."""


@dataclass
class Retrieved:
    text: str
    source_url: str
    title: str
    score: float


class Index:
    def __init__(self, strategy: str):
        safe = strategy.replace("/", "-")
        matrix_path = ROOT / "data" / f"index-{safe}.npy"
        chunks_path = ROOT / "data" / f"chunks-{safe}.json"

        if not matrix_path.exists():
            raise SystemExit(f"missing {matrix_path.name} — run ingest/embed.py --strategy {strategy}")

        self.strategy = strategy
        self.matrix = np.load(matrix_path)
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chat_model = chat_model()

    def embed(self, text: str) -> np.ndarray:
        # is_query=True selects the asymmetric query task type where the
        # provider supports it.
        return embed([text], is_query=True)[0]

    def search(self, question: str, k: int = 5) -> list[Retrieved]:
        # Index rows are pre-normalised, so a dot product IS cosine similarity.
        scores = self.matrix @ self.embed(question)
        top = np.argsort(-scores)[:k]
        return [
            Retrieved(
                text=self.chunks[i]["text"],
                source_url=self.chunks[i]["source_url"],
                title=self.chunks[i]["title"],
                score=float(scores[i]),
            )
            for i in top
        ]

    def answer(self, question: str, k: int = 5) -> tuple[str, list[Retrieved]]:
        hits = self.search(question, k)

        context = "\n\n---\n\n".join(
            f"[{i + 1}] {h.title}\nURL: {h.source_url}\n{h.text}"
            for i, h in enumerate(hits)
        )

        reply = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        return reply.choices[0].message.content.strip(), hits


@lru_cache(maxsize=4)
def get_index(strategy: str = "recursive-800-overlap-100") -> Index:
    return Index(strategy)
