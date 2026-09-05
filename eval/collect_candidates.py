"""Build gold-set candidates from real community.n8n.io threads.

Pulls solved questions from the Discourse API, extracts the accepted answer and
any docs.n8n.io links cited in the thread, and writes a CSV for manual review.

The output is CANDIDATES, not a finished gold set. Every row still needs a human
to confirm the accepted answer is actually correct and the docs URL actually
answers the question. That verification is what makes the eval trustworthy — and
it is the part you cannot outsource.

    python eval/collect_candidates.py --target 120
"""

import argparse
import csv
import html
import json
import re
import time
from pathlib import Path

import requests

BASE = "https://community.n8n.io"
QUESTIONS_CATEGORY = 12
OUT = Path(__file__).parent / "candidates.csv"
DELAY = 0.6
HEADERS = {"User-Agent": "rag-n8n-assistant/0.1 (portfolio project)"}

DOCS_RE = re.compile(r"https?://docs\.n8n\.io/[^\s\"'<>)\]]+")
TAG_RE = re.compile(r"<[^>]+>")

# Thread titles that will not produce a clean factual question.
SKIP_TITLE = re.compile(
    r"\b(feature request|roadmap|hiring|job|announce|release|survey|poll|"
    r"fraud|scam|down|outage|opinion|thoughts on|which is better)\b",
    re.I,
)

CATEGORY_HINTS = [
    ("error-handling", r"\b(error|fail|retry|catch|exception|timeout|crash)\b"),
    ("deployment", r"\b(docker|self-host|deploy|install|upgrade|kubernetes|env|queue mode)\b"),
    ("auth-credentials", r"\b(oauth|credential|api key|token|auth|permission|scope)\b"),
    ("data-transform", r"\b(json|item|array|split|merge|expression|transform|map|loop)\b"),
    ("scheduling", r"\b(cron|schedule|trigger|interval|webhook)\b"),
    ("integrations", r"\b(google|slack|notion|airtable|whatsapp|telegram|openai|http request)\b"),
    ("nodes", r"\b(node|function|code node|set node|switch|if node)\b"),
]


def strip_html(raw: str) -> str:
    text = TAG_RE.sub(" ", raw or "")
    return " ".join(html.unescape(text).split())


def guess_category(title: str, body: str) -> str:
    """Score every category and take the best match.

    First-match-wins put 'error-handling' on half the corpus, because almost
    every thread body mentions an error somewhere. Title matches are weighted
    3x since the title is what the question is actually about.
    """
    best, best_score = "other", 0
    for name, pattern in CATEGORY_HINTS:
        score = 3 * len(re.findall(pattern, title, re.I)) + len(re.findall(pattern, body, re.I))
        if score > best_score:
            best, best_score = name, score
    return best


def assign_difficulty(rows: list[dict]) -> None:
    """Rank by thread length and docs breadth, then split into thirds.

    Absolute thresholds put everything in one bucket — solved threads are long
    by nature. Relative ranking within the collected set is the honest version,
    and it guarantees a usable spread.
    """
    ranked = sorted(rows, key=lambda r: (len(r["all_docs_urls"].split("|")), r["replies"]))
    third = max(len(ranked) // 3, 1)
    for i, row in enumerate(ranked):
        row["difficulty"] = "easy" if i < third else "medium" if i < 2 * third else "hard"


def fetch(session: requests.Session, url: str) -> dict | None:
    try:
        resp = session.get(url, timeout=20)
    except requests.RequestException as exc:
        print(f"    ! {exc}")
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def solved_topics(session: requests.Session, pages: int) -> list[dict]:
    """Page through the questions category collecting solved threads."""
    found, seen = [], set()

    for listing in ("top.json?period=yearly&", "top.json?period=all&", "latest.json?"):
        for page in range(pages):
            url = f"{BASE}/c/questions/{QUESTIONS_CATEGORY}/l/{listing}page={page}"
            data = fetch(session, url)
            time.sleep(DELAY)
            if not data:
                break

            topics = data.get("topic_list", {}).get("topics", [])
            if not topics:
                break

            for t in topics:
                if t["id"] in seen:
                    continue
                seen.add(t["id"])
                if not t.get("has_accepted_answer"):
                    continue
                if SKIP_TITLE.search(t["title"]):
                    continue
                found.append(t)

            print(f"  {listing.split('.')[0]:<6} page {page}: {len(found)} solved so far")

    return found


def extract(session: requests.Session, topic: dict) -> dict | None:
    data = fetch(session, f"{BASE}/t/{topic['id']}.json")
    time.sleep(DELAY)
    if not data:
        return None

    posts = data.get("post_stream", {}).get("posts", [])
    if len(posts) < 2:
        return None

    question = strip_html(posts[0].get("cooked", ""))

    accepted = next(
        (p for p in posts[1:] if p.get("accepted_answer") or p.get("is_accepted_answer")),
        None,
    )
    if accepted is None:
        return None

    answer = strip_html(accepted.get("cooked", ""))
    if len(answer) < 40:
        return None

    docs_links = set()
    for p in posts:
        docs_links.update(DOCS_RE.findall(p.get("cooked", "")))
    docs_links = {u.rstrip(".,);") for u in docs_links}

    return {
        "id": topic["id"],
        "question": topic["title"].strip(),
        "question_body": question[:600],
        "gold_answer": answer[:600],
        "source_url": sorted(docs_links)[0] if docs_links else "",
        "all_docs_urls": " | ".join(sorted(docs_links)),
        "thread_url": f"{BASE}/t/{topic['slug']}/{topic['id']}",
        "difficulty": "",
        "category": guess_category(topic["title"], question),
        "replies": topic.get("posts_count", 0),
        "views": topic.get("views", 0),
        "verified": "no",
    }


def main(target: int, pages: int) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    print("collecting solved threads from the questions category\n")
    topics = solved_topics(session, pages)
    print(f"\n{len(topics)} solved candidate threads\n")

    rows = []
    for topic in topics:
        if len(rows) >= target:
            break
        row = extract(session, topic)
        if row:
            rows.append(row)
            flag = "docs" if row["source_url"] else " -- "
            print(f"  [{len(rows):>3}] {flag}  {row['question'][:64]}")

    assign_difficulty(rows)
    rows.sort(key=lambda r: (r["source_url"] == "", -r["views"]))

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with_docs = sum(1 for r in rows if r["source_url"])
    print(f"\nwrote {OUT}")
    print(f"  {len(rows)} candidates · {with_docs} with a docs.n8n.io link")
    print("\nby difficulty:", {d: sum(1 for r in rows if r["difficulty"] == d)
                              for d in ("easy", "medium", "hard")})
    print("by category:  ", {c: sum(1 for r in rows if r["category"] == c)
                             for c in sorted({r["category"] for r in rows})})
    print("\nNEXT: open the CSV, keep the 50 best rows with a real docs URL,")
    print("verify each answer against that page, set verified=yes, then copy")
    print("the columns id,question,gold_answer,source_url,difficulty,category")
    print("into eval/gold_set.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--pages", type=int, default=4)
    main(**vars(ap.parse_args()))
