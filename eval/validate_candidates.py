"""Clean and rank the candidate pool before human review.

Three problems the collector cannot avoid, all fixed here:

  1. Asset URLs.        Link previews embed favicon.ico and screenshots, so the
                        "docs link" is sometimes an image.
  2. Dead doc URLs.     n8n restructured its docs; old `.html` and `#/` paths
                        still appear in years-old threads and now 404.
  3. Chatty answers.    An accepted answer that opens "Thanks! I had a look..."
                        is a conversation turn, not a reference answer.

Output: candidates_clean.csv, ranked best-first, with a `needs_rewrite` flag on
answers that a human has to rewrite before they can serve as gold.

    python eval/validate_candidates.py
"""

import csv
import re
import time
from pathlib import Path

import requests

IN = Path(__file__).parent / "candidates.csv"
OUT = Path(__file__).parent / "candidates_clean.csv"
DELAY = 0.25
HEADERS = {"User-Agent": "rag-n8n-assistant/0.1 (portfolio project)"}

ASSET = re.compile(r"\.(ico|png|jpe?g|gif|svg|css|js|woff2?|zip|pdf)(\?|$)", re.I)

# Old docs layouts that no longer resolve.
STALE = re.compile(r"(/#/|\.html(\?|#|$)|/_images/)", re.I)

# Accepted posts that are a pasted log or stack trace rather than an answer.
LOGDUMP = re.compile(
    r"(httpCode|timestamp:\s*\d{10}|at Object\.|at Function\.|"
    r"stack:|node_modules/|\d{4}-\d{2}-\d{2}T\d{2}:\d{2})",
    re.I,
)

# Accepted answers that are conversation, not reference material.
CHATTY = re.compile(
    r"^\s*(thanks|thank you|ah |ah,|oh |sorry|glad|great|perfect|awesome|"
    r"you're welcome|no problem|sure[,.]|hi |hey |hello|welcome)",
    re.I,
)


def pick_url(row: dict) -> str:
    """Prefer a real docs page over an asset, and a live path over a stale one."""
    urls = [u.strip() for u in row["all_docs_urls"].split("|") if u.strip()]
    real = [u for u in urls if not ASSET.search(u)]
    fresh = [u for u in real if not STALE.search(u)]
    # No fallback to stale: `docs.n8n.io/#/x` never sends the fragment to the
    # server, so it resolves to the homepage and passes a 200 check while
    # pointing at nothing. A row with only stale URLs is unusable.
    return fresh[0] if fresh else ""


def answer_quality(answer: str) -> tuple[bool, str]:
    """Returns (needs_rewrite, reason)."""
    text = answer.strip()
    if len(text) < 80:
        return True, "too short"
    if CHATTY.match(text):
        return True, "conversational opener"
    if text.count("http") >= 3 and len(text) < 200:
        return True, "mostly links"
    if LOGDUMP.search(text):
        return True, "log or stack trace"
    if text.count("|") >= 6:
        return True, "log or stack trace"
    return False, ""


def url_alive(session: requests.Session, url: str, cache: dict) -> bool:
    if url in cache:
        return cache[url]
    ok = False
    try:
        resp = session.head(url, timeout=15, allow_redirects=True)
        if resp.status_code in (403, 405):        # some hosts reject HEAD
            resp = session.get(url, timeout=15, stream=True)
        ok = resp.status_code == 200
    except requests.RequestException:
        ok = False
    cache[url] = ok
    time.sleep(DELAY)
    return ok


def main() -> None:
    rows = list(csv.DictReader(IN.open(encoding="utf-8")))
    session = requests.Session()
    session.headers.update(HEADERS)
    cache: dict[str, bool] = {}

    kept, dropped = [], {"no_url": 0, "dead_url": 0}

    for row in rows:
        url = pick_url(row)
        if not url:
            dropped["no_url"] += 1
            continue

        if not url_alive(session, url, cache):
            dropped["dead_url"] += 1
            continue

        needs_rewrite, reason = answer_quality(row["gold_answer"])
        row["source_url"] = url
        row["needs_rewrite"] = "yes" if needs_rewrite else "no"
        row["rewrite_reason"] = reason
        kept.append(row)
        print(f"  {'RW ' if needs_rewrite else 'ok '} {row['question'][:58]:<58} {url[:52]}")

    # Best first: clean answers, then most-viewed.
    kept.sort(key=lambda r: (r["needs_rewrite"] == "yes", -int(r["views"] or 0)))

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
        writer.writeheader()
        writer.writerows(kept)

    ready = sum(1 for r in kept if r["needs_rewrite"] == "no")
    print(f"\nwrote {OUT}")
    print(f"  {len(kept)} live candidates — {ready} usable as-is, "
          f"{len(kept) - ready} need the answer rewritten")
    print(f"  dropped: {dropped['no_url']} no docs URL, {dropped['dead_url']} dead URL")

    from collections import Counter
    print("\n  difficulty:", dict(Counter(r["difficulty"] for r in kept)))
    print("  category:  ", dict(Counter(r["category"] for r in kept)))


if __name__ == "__main__":
    main()
