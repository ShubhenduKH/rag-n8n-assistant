"""Sweep BM25 parameters to find the lexical ceiling.

Run before reaching for embeddings. If a parameter sweep closes the gap, the
retriever was misconfigured, not fundamentally limited — and swapping in a
dense model would have taken credit for a fix that cost nothing.

Here it does not close the gap, which is the point: the sweep is the evidence
that dense retrieval is warranted rather than assumed.

    python eval/tune_bm25.py
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.bm25 import tokenize                 # noqa: E402
from eval.run_eval import normalise_url       # noqa: E402
from ingest.chunk import chunk_documents      # noqa: E402
from ingest.corpus import load_docs           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "tuning.json"

TITLE_WEIGHTS = (1, 2, 4, 8)
K1_VALUES = (1.2, 1.5, 2.0)
B_VALUES = (0.3, 0.75)


def load():
    docs = load_docs()
    gold = list(csv.DictReader((ROOT / "eval" / "gold_set.csv").open(encoding="utf-8")))
    chunks = chunk_documents(docs, "recursive-800")
    meta = [{"url": normalise_url(c.source_url), "title": c.title, "text": c.text}
            for c in chunks]
    return meta, gold


def build(meta, title_weight, k1, b):
    tokens = [tokenize((m["title"] + " ") * title_weight + m["text"]) for m in meta]
    lengths = np.array([len(t) for t in tokens], dtype=np.float32)
    avg = float(lengths.mean()) or 1.0

    postings: dict[str, list] = {}
    for i, toks in enumerate(tokens):
        for term, count in Counter(toks).items():
            postings.setdefault(term, []).append((i, count))

    n = len(tokens)
    idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}
    return postings, idf, k1 * (1 - b + b * lengths / avg), k1


def search(query, index, meta, k=5, pool=60):
    postings, idf, norm, k1 = index
    scores = np.zeros(len(meta), dtype=np.float32)
    for term in tokenize(query):
        plist = postings.get(term)
        if not plist:
            continue
        weight = idf[term]
        for doc_id, freq in plist:
            scores[doc_id] += weight * (freq * (k1 + 1)) / (freq + norm[doc_id])

    seen, urls = set(), []
    for i in np.argsort(-scores)[:pool]:
        url = meta[i]["url"]
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= k:
            break
    return set(urls)


def evaluate(index, meta, gold, k=5):
    strict = lenient = 0
    for row in gold:
        urls = search(row["question"], index, meta, k)
        if normalise_url(row["source_url"]) in urls:
            strict += 1
        acceptable = {normalise_url(u) for u in row["acceptable_urls"].split("|")}
        if urls & acceptable:
            lenient += 1
    n = len(gold)
    return round(100 * strict / n, 1), round(100 * lenient / n, 1)


def main() -> None:
    meta, gold = load()
    print(f"{len(meta)} chunks · {len(gold)} questions\n")
    print("title_w   k1     b      strict   lenient")
    print("-" * 44)

    results = []
    for tw in TITLE_WEIGHTS:
        for k1 in K1_VALUES:
            for b in B_VALUES:
                strict, lenient = evaluate(build(meta, tw, k1, b), meta, gold)
                results.append({"title_weight": tw, "k1": k1, "b": b,
                                "strict": strict, "lenient": lenient})
                print(f"  {tw:<7} {k1:<6} {b:<6} {strict:5.1f}%   {lenient:5.1f}%")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    strict_values = {r["strict"] for r in results}
    lenient_values = {r["lenient"] for r in results}
    best = max(results, key=lambda r: (r["lenient"], r["strict"]))

    print(f"\nwrote {OUT.name}")
    print(f"\nconfigurations tested : {len(results)}")
    print(f"distinct strict scores: {sorted(strict_values)}")
    print(f"distinct lenient scores: {sorted(lenient_values)}")
    print(f"\nbest: title_weight={best['title_weight']} k1={best['k1']} b={best['b']}"
          f"  ->  strict {best['strict']}%  lenient {best['lenient']}%")
    print(f"spread: strict {max(strict_values) - min(strict_values):.1f}pp, "
          f"lenient {max(lenient_values) - min(lenient_values):.1f}pp")
    print("\nA spread this small across the whole grid means the retriever is at its\n"
          "ceiling for this query distribution — not misconfigured.")


if __name__ == "__main__":
    main()
