"""Pre-screen the gold set so human verification takes minutes, not hours.

Verifying 39 rows by hand means opening 39 pages and reading each one. Most of
that is mechanical: does the page still exist, is it in the corpus, and does it
actually contain the terms the accepted answer relies on?

This script does the mechanical part and writes review.md — a checklist ordered
worst-first, with the evidence inline. What it cannot do is judge whether the
answer is *correct*; that judgement is what makes the eval trustworthy and it
stays with a human.

    python eval/prepare_review.py
"""

import sys

# Windows defaults stdout to cp1252, which raises UnicodeEncodeError as soon as
# a scraped title contains an arrow or a smart quote — but only when output is
# redirected to a file, so it passes interactively and fails in CI.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import csv
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.bm25 import tokenize                 # noqa: E402
from eval.run_eval import normalise_url       # noqa: E402
from ingest.corpus import load_docs           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "review.md"

# Terms too generic to prove a page is the right one.
GENERIC = frozenset("""
n8n node nodes workflow workflows data set get use using used add added
value values page docs documentation example examples option options
""".split())


def content_terms(text: str) -> set:
    return {t for t in tokenize(text) if t not in GENERIC and len(t) > 2}


def score_subsets(all_rows, clean, flagged) -> dict:
    """Score retrieval separately on the clean and flagged halves.

    The gap between them is the share of the headline number that is gold-set
    noise rather than retriever behaviour. Reported here so the split in the
    README is reproducible rather than asserted.
    """
    from api.bm25 import build as build_bm25

    index = build_bm25("recursive-800")

    def run(rows):
        if not rows:
            return (0.0, 0.0, 0)
        strict = lenient = 0
        for row in rows:
            urls = {normalise_url(h.source_url)
                    for h in index.search(row["question"], 5)}
            if normalise_url(row["source_url"]) in urls:
                strict += 1
            acceptable = {normalise_url(u) for u in row["acceptable_urls"].split("|")}
            if urls & acceptable:
                lenient += 1
        n = len(rows)
        return (round(100 * strict / n, 1), round(100 * lenient / n, 1), n)

    return {
        "consistent rows only": run(clean),
        "all rows": run(all_rows),
        "flagged rows only": run(flagged),
    }


def main() -> None:
    docs = load_docs()
    by_url = {normalise_url(d["url"]): d for d in docs}
    gold = list(csv.DictReader((ROOT / "eval" / "gold_set.csv").open(encoding="utf-8")))

    rows = []
    for row in gold:
        url = normalise_url(row["source_url"])
        page = by_url.get(url)

        answer_terms = content_terms(row["gold_answer"])
        question_terms = content_terms(row["question"])

        if page:
            page_terms = content_terms(page["title"] + " " + page["text"])
            answer_overlap = answer_terms & page_terms
            question_overlap = question_terms & page_terms
            coverage = len(answer_overlap) / max(len(answer_terms), 1)
        else:
            page_terms = set()
            answer_overlap = question_overlap = set()
            coverage = 0.0

        problems = []
        if not page:
            problems.append("page not in corpus")
        if coverage < 0.25:
            problems.append(f"answer terms barely on page ({coverage:.0%})")
        if not question_overlap:
            problems.append("no question term appears on the page")
        if len(row["gold_answer"]) < 120:
            problems.append("answer very short")
        if re.search(r"\bhttps?://(?!docs\.n8n\.io)", row["gold_answer"]):
            problems.append("answer points off-site")

        rows.append({
            **row,
            "page_title": page["title"] if page else "",
            "coverage": coverage,
            "shared_terms": sorted(answer_overlap)[:12],
            "problems": problems,
            "risk": len(problems),
        })

    rows.sort(key=lambda r: (-r["risk"], -r["coverage"]))

    clean = [r for r in rows if not r["problems"]]
    flagged = [r for r in rows if r["problems"]]

    subset_scores = score_subsets(rows, clean, flagged)

    lines = [
        "# Gold set review",
        "",
        f"{len(rows)} rows · **{len(flagged)} need attention** · {len(clean)} look consistent",
        "",
        "Automated checks confirm the page exists, is in the scraped corpus, and shares",
        "vocabulary with the accepted answer. They cannot confirm the answer is *correct*.",
        "",
        "For each row: open the page, decide whether it genuinely answers the question,",
        "then set `verified=yes` in `gold_set.csv` (or delete the row).",
        "",
        "---",
        "",
        "## Needs attention",
        "",
    ]

    for r in flagged:
        lines += [
            f"### {r['id']}. {r['question']}",
            "",
            f"- **Problems:** {', '.join(r['problems'])}",
            f"- **Gold page:** [{r['page_title'] or 'MISSING'}]({r['source_url']})",
            f"- **Answer terms on page:** {r['coverage']:.0%}"
            + (f" — {', '.join(r['shared_terms'])}" if r["shared_terms"] else ""),
            f"- **Source thread:** {r['thread_url']}",
            f"- **Accepted answer:** {r['gold_answer'][:220]}...",
            "",
        ]

    lines += ["---", "", "## Look consistent (still needs a human read)", ""]
    for r in clean:
        lines += [
            f"- [ ] **{r['id']}.** {r['question']}  ",
            f"  [{r['page_title']}]({r['source_url']}) · answer terms on page: {r['coverage']:.0%}",
        ]

    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(rows)} rows checked")
    print(f"  {len(flagged)} flagged")
    print(f"  {len(clean)} look consistent")
    print(f"\nwrote {OUT}")

    from collections import Counter
    reasons = Counter(p for r in flagged for p in r["problems"])
    if reasons:
        print("\nreasons:")
        for reason, count in reasons.most_common():
            print(f"  {count:>3}  {reason}")

    print("\nRetrieval@5 by subset — this is the number that matters:")
    for label, (strict, lenient, n) in subset_scores.items():
        print(f"  {label:<24} n={n:<3} strict {strict:5.1f}%  lenient {lenient:5.1f}%")
    print("\nIf the consistent subset scores far higher, the headline number was")
    print("measuring gold-set noise as much as retriever quality.")


if __name__ == "__main__":
    main()
