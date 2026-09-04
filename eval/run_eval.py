"""The eval harness. This file is the differentiator.

Two metrics:
  Retrieval@k      did the gold source_url appear in the top-k retrieved chunks?
                   Objective, no LLM involved, cannot be gamed.
  Answer correct   does the generated answer match the gold answer?
                   Judged by an LLM, then spot-checked by hand.

Publish whatever number this prints. An honest 68% with a stated method beats
a fabricated 94% — and a serious client will ask to see the gold set.
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

GOLD_PATH = Path(__file__).parent / "gold_set.csv"
RESULTS_PATH = Path(__file__).parent / "results.json"


@dataclass
class Question:
    id: str
    question: str
    gold_answer: str
    source_url: str
    difficulty: str
    category: str


def load_gold() -> list[Question]:
    if not GOLD_PATH.exists():
        raise SystemExit(f"missing {GOLD_PATH} — collect the 50 questions first")

    with GOLD_PATH.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise SystemExit("gold_set.csv has a header but no rows")

    questions = [Question(
        id=r["id"],
        question=r["question"],
        gold_answer=r.get("gold_answer", ""),
        source_url=r["source_url"],
        difficulty=r.get("difficulty", ""),
        category=r.get("category", ""),
    ) for r in rows]

    missing = [q.id for q in questions if not q.source_url.strip()]
    if missing:
        raise SystemExit(f"rows missing source_url: {missing}")

    return questions


def normalise_url(url: str) -> str:
    """docs.n8n.io/foo/ and docs.n8n.io/foo#section must compare equal."""
    url = url.strip().split("#")[0].split("?")[0]
    return url.rstrip("/").lower()


def retrieval_at_k(retrieved: list, gold_url: str, k: int = 5) -> bool:
    gold = normalise_url(gold_url)
    return any(normalise_url(c.source_url) == gold for c in retrieved[:k])


JUDGE_PROMPT = """You are grading a support answer against a reference answer.

Question: {question}

Reference answer: {gold}

Candidate answer: {candidate}

Does the candidate convey the same substantive guidance as the reference?
Ignore differences in wording, length, formatting and politeness. Judge only
whether someone following the candidate would achieve what the reference
describes.

A candidate that says it does not know is INCORRECT.
A candidate that is correct but adds extra detail is CORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def judge(question: str, gold: str, candidate: str, client) -> bool:
    """LLM-as-judge. Verify at least 10 of these by hand before publishing."""
    reply = client.chat.completions.create(
        model=os.getenv("JUDGE_MODEL", "gpt-5.6-luna"),
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, gold=gold, candidate=candidate
            ),
        }],
        temperature=0,
    )
    return reply.choices[0].message.content.strip().upper().startswith("CORRECT")


def run(strategy: str, retriever, answerer, judge_client, k: int = 5) -> dict:
    """retriever(question) -> list[Chunk];  answerer(question) -> str"""
    questions = load_gold()
    hits = correct = 0
    per_question = []

    for q in questions:
        retrieved = retriever(q.question)
        hit = retrieval_at_k(retrieved, q.source_url, k)
        hits += hit

        answer = answerer(q.question)
        ok = judge(q.question, q.gold_answer, answer, judge_client) if q.gold_answer else None
        correct += bool(ok)

        per_question.append({
            "id": q.id,
            "difficulty": q.difficulty,
            "category": q.category,
            "retrieval_hit": hit,
            "answer_correct": ok,
        })
        print(f"  {q.id:>3}  retrieval {'HIT ' if hit else 'MISS'}  answer {ok}")

    n = len(questions)
    result = {
        "strategy": strategy,
        "n": n,
        "k": k,
        "retrieval_at_k": round(100 * hits / n, 1),
        "answer_correct": round(100 * correct / n, 1),
        "measured_on": date.today().isoformat(),
        "per_question": per_question,
    }

    print(f"\n{strategy}")
    print(f"  Retrieval@{k}:    {result['retrieval_at_k']}%   ({hits}/{n})")
    print(f"  Answer correct: {result['answer_correct']}%   ({correct}/{n})")
    return result


def breakdown(result: dict, field: str) -> dict:
    """Per-category or per-difficulty accuracy. This is what looks rigorous
    in a README, and it costs nothing extra to compute."""
    buckets: dict[str, list] = {}
    for row in result["per_question"]:
        buckets.setdefault(row[field] or "unspecified", []).append(row)

    return {
        key: {
            "n": len(rows),
            "retrieval": round(100 * sum(r["retrieval_hit"] for r in rows) / len(rows), 1),
            "answer": round(100 * sum(bool(r["answer_correct"]) for r in rows) / len(rows), 1),
        }
        for key, rows in sorted(buckets.items())
    }


def save(results: list[dict]) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH}")


def markdown_table(results: list[dict]) -> str:
    """Paste the output of this straight into the README."""
    lines = [
        "| Chunking | Retrieval@5 | Answer correct |",
        "|---|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: -x["retrieval_at_k"]):
        lines.append(
            f"| {r['strategy']} | {r['retrieval_at_k']}% | {r['answer_correct']}% |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(
        "Import run() from your pipeline and pass a retriever + answerer.\n"
        "See README for the wiring."
    )
