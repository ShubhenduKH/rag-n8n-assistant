"""Turn validated candidates into a gold set.

Two jobs the collector cannot do:

  1. Resolve every docs URL the thread cited, not just the first, into the live
     post-redirect form. A thread that cites three pages has three acceptable
     answers; scoring only the first counts correct retrievals as misses.
  2. Preserve verification decisions across rebuilds. Rows already judged by a
     human keep their verdict and note, so widening the candidate pool never
     silently discards that work.

    python eval/build_gold.py --limit 120
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import normalise_url       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "eval" / "candidates_clean.csv"
GOLD = ROOT / "eval" / "gold_set.csv"
URLS = ROOT / "data" / "urls.json"

DELAY = 0.25
ASSET = re.compile(r"\.(ico|png|jpe?g|gif|svg|css|js|woff2?|zip|pdf)(\?|$)", re.I)

FIELDS = ["id", "question", "gold_answer", "source_url", "acceptable_urls",
          "n_acceptable", "original_url", "difficulty", "category",
          "verified", "verify_note", "thread_url"]


def load_previous() -> dict:
    """Existing verdicts, keyed by thread URL so ids can be renumbered freely."""
    if not GOLD.exists():
        return {}
    with GOLD.open(encoding="utf-8") as fh:
        return {r["thread_url"]: (r.get("verified", "no"), r.get("verify_note", ""))
                for r in csv.DictReader(fh)}


def main(limit: int) -> None:
    live = {normalise_url(u) for u in json.loads(URLS.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(CLEAN.open(encoding="utf-8")))
    previous = load_previous()

    session = requests.Session()
    session.headers.update({"User-Agent": "rag-n8n-assistant/0.1"})
    cache: dict[str, str] = {}

    def resolve(url: str) -> str:
        """Post-redirect URL, or "" if it does not land inside the corpus."""
        if url in cache:
            return cache[url]
        out = ""
        if normalise_url(url) in live:
            out = url
        else:
            try:
                resp = session.get(url, timeout=20, allow_redirects=True)
                if resp.status_code == 200 and normalise_url(resp.url) in live:
                    out = resp.url
            except requests.RequestException:
                pass
            time.sleep(DELAY)
        cache[url] = out
        return out

    usable = [r for r in rows if r.get("needs_rewrite") == "no"][:limit]
    print(f"{len(usable)} candidates with a usable answer\n")

    out, dropped = [], 0
    for row in usable:
        primary = resolve(row["source_url"])
        if not primary:
            dropped += 1
            continue

        acceptable = {normalise_url(primary)}
        for cited in row.get("all_docs_urls", "").split("|"):
            cited = cited.strip()
            if not cited or ASSET.search(cited):
                continue
            landed = resolve(cited)
            if landed:
                acceptable.add(normalise_url(landed))

        verified, note = previous.get(row["thread_url"], ("no", ""))
        out.append({
            "id": len(out) + 1,
            "question": row["question"],
            "gold_answer": " ".join(row["gold_answer"].split())[:400],
            "source_url": primary,
            "acceptable_urls": " | ".join(sorted(acceptable)),
            "n_acceptable": len(acceptable),
            "original_url": row["source_url"],
            "difficulty": row["difficulty"],
            "category": row["category"],
            "verified": verified,
            "verify_note": note,
            "thread_url": row["thread_url"],
        })
        if len(out) % 20 == 0:
            print(f"  {len(out)} built")

    with GOLD.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)

    carried = sum(1 for r in out if r["verified"] in ("yes", "no") and r["verify_note"])
    print(f"\nwrote {GOLD.name}")
    print(f"  {len(out)} rows · {dropped} dropped (URL not in corpus)")
    print(f"  {sum(1 for r in out if r['verified'] == 'yes')} already verified yes")
    print(f"  {carried} carried a previous decision")
    print(f"  {sum(1 for r in out if not r['verify_note'])} still need review")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    main(**vars(ap.parse_args()))
