"""Run the eval across every chunking strategy and write the results table.

Scores the same retriever the API serves, so the published number describes
production and not a parallel pipeline built for the benchmark.

    python ingest/embed.py --strategy fixed-512
    python ingest/embed.py --strategy recursive-800-overlap-100
    python eval/compare.py --strategies fixed-512 recursive-800-overlap-100
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.retrieve import Index                    # noqa: E402
from eval.run_eval import breakdown, markdown_table, run, save  # noqa: E402


def main(strategies: list[str], k: int) -> None:
    judge_client = OpenAI()
    results = []

    for strategy in strategies:
        print(f"\n{'=' * 60}\n{strategy}\n{'=' * 60}")
        try:
            index = Index(strategy)
        except SystemExit as exc:
            print(f"  skipped — {exc}")
            continue

        result = run(
            strategy=strategy,
            retriever=lambda q, idx=index: idx.search(q, k),
            answerer=lambda q, idx=index: idx.answer(q, k)[0],
            judge_client=judge_client,
            k=k,
        )
        result["by_difficulty"] = breakdown(result, "difficulty")
        result["by_category"] = breakdown(result, "category")
        results.append(result)

    if not results:
        raise SystemExit("no strategies scored — did you run ingest/embed.py?")

    save(results)

    print("\n\nPaste this into the README:\n")
    print(markdown_table(results))

    best = max(results, key=lambda r: r["retrieval_at_k"])
    print(f"\nBest: {best['strategy']}")
    print("\nBy difficulty:")
    for name, row in best["by_difficulty"].items():
        print(f"  {name:<10} n={row['n']:<4} retrieval {row['retrieval']:>5}%  answer {row['answer']:>5}%")
    print("\nBy category:")
    for name, row in best["by_category"].items():
        print(f"  {name:<18} n={row['n']:<4} retrieval {row['retrieval']:>5}%  answer {row['answer']:>5}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+",
                    default=["fixed-512", "fixed-512-overlap-64",
                             "recursive-800", "recursive-800-overlap-100"])
    ap.add_argument("-k", type=int, default=5)
    main(**vars(ap.parse_args()))
