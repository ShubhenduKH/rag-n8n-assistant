"""BM25 vs LSA vs hybrid, on the verified rows and on everything.

Answers the question the README leaves open — does dense retrieval actually help
here, or was the lexical baseline already most of the way there? — without an API
key, by using a dense representation derived from the corpus itself.

Reports the verified subset separately because that is the only subset where a
miss is unambiguously the retriever's fault.

    python eval/compare_retrievers.py
"""

import csv
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.bm25 import build as build_bm25       # noqa: E402
from api.lsa import Hybrid, LSA                # noqa: E402
from eval.run_eval import normalise_url        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "retriever_comparison.json"
STRATEGY = "recursive-800"


def wilson(hits: int, n: int) -> tuple[float, float]:
    """Wilson interval — honest at the small n and extreme rates seen here,
    where the normal approximation can run past 0% or 100%."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * max(0.0, centre - spread), 1),
            round(100 * min(1.0, centre + spread), 1))


def score(retriever, rows, k=5) -> dict:
    strict = lenient = 0
    misses = []
    for row in rows:
        urls = {normalise_url(h.source_url) for h in retriever.search(row["question"], k)}
        hit = normalise_url(row["source_url"]) in urls
        any_hit = bool(urls & {normalise_url(u) for u in row["acceptable_urls"].split("|")})
        strict += hit
        lenient += any_hit
        if not any_hit:
            misses.append(row["question"][:60])
    n = len(rows)
    lo, hi = wilson(strict, n)
    return {"n": n,
            "strict": round(100 * strict / n, 1),
            "lenient": round(100 * lenient / n, 1),
            "ci": [lo, hi],
            "misses": misses}


def main() -> None:
    rows = list(csv.DictReader((ROOT / "eval" / "gold_set.csv").open(encoding="utf-8")))
    verified = [r for r in rows if r["verified"] == "yes"]

    print("building retrievers...")
    bm25 = build_bm25(STRATEGY)
    lsa = LSA(bm25.chunks)
    hybrid = Hybrid(bm25, lsa)
    print(f"  {len(bm25.chunks)} chunks · {lsa.dims} LSA dims\n")

    retrievers = {"BM25": bm25, "LSA": lsa, "Hybrid (RRF)": hybrid}
    results = {}

    for label, retriever in retrievers.items():
        results[label] = {
            "verified": score(retriever, verified),
            "all": score(retriever, rows),
        }
        v, a = results[label]["verified"], results[label]["all"]
        print(f"{label:<14} verified n={v['n']:<4} strict {v['strict']:5.1f}% "
              f"[{v['ci'][0]}–{v['ci'][1]}]  lenient {v['lenient']:5.1f}%")
        print(f"{'':<14} all      n={a['n']:<4} strict {a['strict']:5.1f}% "
              f"[{a['ci'][0]}–{a['ci'][1]}]  lenient {a['lenient']:5.1f}%")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT.name}\n")
    print("| Retriever | Verified strict | Verified lenient | All strict | All lenient |")
    print("|---|---:|---:|---:|---:|")
    for label, r in results.items():
        print(f"| {label} | {r['verified']['strict']}% | {r['verified']['lenient']}% "
              f"| {r['all']['strict']}% | {r['all']['lenient']}% |")

    base = results["BM25"]["verified"]["strict"]
    for label in ("LSA", "Hybrid (RRF)"):
        delta = results[label]["verified"]["strict"] - base
        print(f"\n{label} vs BM25 on verified rows: {delta:+.1f}pp")

    both = set(results["BM25"]["verified"]["misses"]) & set(results["LSA"]["verified"]["misses"])
    print(f"\nquestions every retriever misses: {len(both)}")
    for q in sorted(both)[:6]:
        print(f"  - {q}")


if __name__ == "__main__":
    main()
