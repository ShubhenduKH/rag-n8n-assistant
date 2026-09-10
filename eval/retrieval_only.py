"""Score retrieval across every strategy, including BM25. No API key required.

Retrieval@k is the half of the eval that needs no model: either the page that
answers the question was retrieved, or it was not. Running this first tells you
whether the pipeline can find the right page before you spend anything on
generation — and BM25 gives the baseline any embedding retriever has to beat.

    python eval/retrieval_only.py
"""

import sys

# Windows defaults stdout to cp1252, which raises UnicodeEncodeError as soon as
# a scraped title contains an arrow or a smart quote — but only when output is
# redirected to a file, so it passes interactively and fails in CI.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.bm25 import build as build_bm25              # noqa: E402
from ingest.chunk import STRATEGIES                   # noqa: E402
from eval.run_eval import breakdown, markdown_table, run, save  # noqa: E402


def main() -> None:
    results = []

    for strategy in STRATEGIES:
        print(f"\n{'=' * 58}\nBM25 · {strategy}\n{'=' * 58}")
        index = build_bm25(strategy)
        print(f"{len(index.chunks)} chunks, {len(index.postings)} terms\n")
        result = run(f"bm25 · {strategy}", retriever=lambda q, i=index: i.search(q, 5))
        result["by_difficulty"] = breakdown(result, "difficulty")
        result["by_category"] = breakdown(result, "category")
        results.append(result)

    save(results)
    print("\n\nREADME table:\n")
    print(markdown_table(results))

    best = max(results, key=lambda r: r["retrieval_at_k"])
    print(f"\nBest retriever: {best['strategy']}  ({best['retrieval_at_k']}%)")
    print("\nBy difficulty:")
    for name, row in best["by_difficulty"].items():
        print(f"  {name:<10} n={row['n']:<4} retrieval {row['retrieval']:>5}%")
    print("\nBy category:")
    for name, row in best["by_category"].items():
        print(f"  {name:<18} n={row['n']:<4} retrieval {row['retrieval']:>5}%")


if __name__ == "__main__":
    main()
