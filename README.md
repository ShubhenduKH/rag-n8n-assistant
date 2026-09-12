# rag-n8n-assistant

A retrieval assistant over the [n8n documentation](https://docs.n8n.io), built to
answer the question most RAG demos skip: **how often does it actually retrieve the
right page?**

The interesting part of this repo is not the chatbot. It is the evaluation —
a gold set built from real solved threads on the n8n community forum, and a
lexical baseline measured *before* reaching for embeddings.

## Results

1,310 pages · 6,314 chunks · 108 questions, 18 hand-verified · measured 2026-09-12

| Retriever | Verified rows (n=18) | All rows (n=108) |
|---|---:|---:|
| BM25 | 22.2% | **18.5%** |
| LSA, 64d | **38.9%** | 17.6% |
| LSA, 256d | **33.3%** | 16.7% |
| Hybrid RRF, 64d | 33.3% | 16.7% |

### The honest headline: at this sample size, the ordering is not resolvable

Read the table again. **On the verified rows LSA wins at every dimensionality.
On all 108 rows BM25 wins at every dimensionality.** The 95% confidence interval
on the BM25 verified figure is [9%, 45%] — wide enough to contain almost every
other number in the table.

I could have published "BM25 beats dense retrieval" or "dense retrieval beats
BM25" from the same experiment by choosing which subset to report. Both would
have been defensible-sounding and both would have been noise. So this repo
publishes neither.

It gets worse, and the worse part is the useful part. An earlier revision of this
README *did* claim BM25 won at every dimensionality — that was true before the
corpus was cleaned. Stripping boilerplate that appeared on 100% of pages (see
below) shifted every chunk boundary and flipped the verified-row ordering. **A
6.7% change in corpus text reversed the conclusion.** A result that fragile is
not a result.

What can be said with the evidence available:

- Both retrievers land in the **15–40%** band on this corpus. Neither is close to
  solved.
- The gap between them is **smaller than the gap between subsets**, which means
  gold-set composition currently matters more than retriever choice.
- Separating them would need roughly **n=150 verified rows**, against the 18 here.
  At a ~3% yield from forum threads that is ~5,000 candidate threads.

### Cleaning the corpus: right thing to do, no measurable gain

Two phrases appeared on **100% of 1,338 pages** (`For the complete documentation
index, see llms.txt`, `This page is also available as Markdown`), as did the
`Last updated … Was this helpful?` footer. Removing them cut 6.7% of corpus text
and dropped 28 pages that turned out to be nothing but chrome.

The effect on retrieval, measured on the larger n=108 sample:

| | Before | After |
|---|---:|---:|
| BM25 | 20.4% | 18.5% |
| LSA (256d) | 13.9% | 16.7% |
| Hybrid | 15.7% | 13.9% |

All three moves sit inside the ±7pp interval. **The cleaning was principled and
made no measurable difference** — worth doing because the demo no longer answers
questions with "Was this helpful?", not because it moved the number. Reporting it
as an improvement would have been a third way to manufacture a result from noise.

### Most of a forum-sourced gold set is unusable

Every row was verified by hand: read the cited page, read the accepted answer,
decide whether the page genuinely answers the question. **18 of 108 passed.**

| Stage | Rows | Survival |
|---|---:|---:|
| Solved threads collected from the forum API | 600 | — |
| Cited docs URL resolves and is live | 183 | 31% |
| Answer is prose, not a screenshot or a log | 111 | 19% |
| Reached the gold set | 108 | 18% |
| **Passed human verification** | **18** | **3.0%** |

The 90 rejections are not hard questions. They are rows where the accepted answer
was never a documentation answer:

| Why rejected | Example |
|---|---|
| Answer is a retraction | *"This post was written a long time ago… do not follow these steps anymore"* |
| Answer is a bug report | *"got it working — that appears to be a Claude-side bug"* |
| Answer is conversation | *"tried it today and it wasnt working"* |
| Answer is a version bump | *"supposed to be fixed in 2.8.0 (pre-release)"* |
| Fix is not on the cited page | answer sets `N8N_ENDPOINT_HEALTH`; the page never mentions it |
| Wrong page entirely | question uses the HTTP Request node; page is the API for *building* nodes |

A thread gets marked solved when the asker is unblocked — by a version bump, a
screenshot, or someone's pasted config. **None of that corresponds to a docs
page, so no retriever can score on it.** Every decision and its reason is in
`gold_set.csv` under `verified` and `verify_note`.

**A correction this repo made to itself.** An earlier pass verified 39 rows, got
7, and measured 42.9% verified vs 8.3% rejected — a 5x gap. Widening the pool to
108 rows and 18 verified moved that to **33.3% vs 17.8%**, and the confidence
intervals now nearly touch. The 5x gap was mostly small-sample noise, which is
precisely the failure this README warns about elsewhere. The direction survived;
the magnitude did not.

**The honest conclusion is about method, not score.** Sourcing a gold set from
solved forum threads has a ~3% yield. Reaching n=50 verified would need roughly
1,700 candidate threads. Anyone publishing a RAG accuracy number from an
unverified forum-scraped gold set is reporting mostly noise.

### Parameter tuning does not rescue it

24 BM25 configurations — title weight ×1–8, k1 ∈ {1.2, 1.5, 2.0}, b ∈ {0.3, 0.75}:

| | spread across the whole grid |
|---|---|
| strict | 2.6pp (20.5% – 23.1%) |
| lenient | 7.7pp (23.1% – 30.8%) |

The grid's best config is not adopted as the headline. Picking the argmax of a
24-cell sweep on n=39 selects noise — the same mistake this README criticises
elsewhere — so the defaults stand and the sweep is reported as evidence of a
ceiling rather than as a result. Reproduce with `python eval/tune_bm25.py`.

### Where lexical retrieval fails, and why

| Works | Fails |
|---|---|
| scheduling · error-handling | data-transform · auth-credentials |

Questions that name a thing (`GENERIC_TIMEZONE`, "Error Trigger") retrieve well.
Conversational ones — *"How can I use now() in an expression"* — do not, because
`use`, `now` and `expression` appear on hundreds of pages. The gold page for that
question **is** in the corpus and **does** contain all three terms; its best chunk
ranks 165th of 6,748. That is vocabulary mismatch, and vocabulary mismatch is
what dense retrieval is supposed to fix.

### The dense retriever, and why it settles nothing

`api/lsa.py` builds a dense representation from the corpus itself — TF-IDF
factored by a truncated SVD, so documents and queries share a latent space. No
API key, no model download. `api/lsa.py` also implements reciprocal-rank fusion
for a BM25 + LSA hybrid.

Scored on the 18 verified rows and on all 108, across four dimensionalities:

| Retriever | Verified strict | Verified lenient | All strict | All lenient |
|---|---:|---:|---:|---:|
| **BM25** | **33.3%** | **44.4%** | **20.4%** | **29.6%** |
| LSA (64d) | 33.3% | 38.9% | 14.8% | 20.4% |
| LSA (128d) | 33.3% | 38.9% | 14.8% | 22.2% |
| LSA (256d) | 27.8% | 33.3% | 13.9% | 22.2% |
| LSA (512d) | 27.8% | 27.8% | 14.8% | 21.3% |
| Hybrid RRF (128d) | 33.3% | 33.3% | 17.6% | 25.9% |

See the results table above: LSA wins on verified rows, loses on all rows, and
the intervals overlap throughout. The experiment does not settle which retriever
is better on this corpus.

What it does establish is a **floor**. Any hosted-embedding result must beat
18.5% overall and 22.2% verified to have earned anything — not zero, which is
what a RAG demo with no baseline implicitly claims. And LSA can only learn
co-occurrence present in 1,310 pages: "now()" never lands near "Luxon" here, no
matter the dimensionality. That is the specific thing a model trained on far more
text is being paid to fix, and isolating it is more useful than assuming it.

## Two measurement bugs that would have faked this number

Both were found while building the eval, and both would have produced a
confidently wrong result.

**1. 33 of 39 gold URLs pointed at pages that no longer exist.** n8n restructured
its docs (`/hosting/` → `/deploy/`, `/data/` → `/build/work-with-data/`). Every
old path still answers `200` — *after a redirect*. A naive liveness check passes
them, and then retrieval is scored against a URL that is not in the corpus, so
every correct hit is counted as a miss. The fix is to record where the redirect
lands, not whether the request succeeded.

**2. Scoring one "correct" page when the thread cited several.** 25 of 39 threads
link more than one docs page, averaging 2.3. For *"how do I self-host n8n"*, both
the cloud-provider guide and the one-line-setup page are right answers. The eval
now reports strict (the page the accepted answer pointed at) and lenient (any
page the thread cited) side by side rather than picking one and hiding the choice.

## Method

The gold set is built from threads on
[community.n8n.io](https://community.n8n.io) that have an accepted answer, via
the public Discourse API. `collect_candidates.py` pulls them;
`validate_candidates.py` drops threads whose cited docs URL is dead, resolves
redirects, and flags accepted answers that are conversational replies or pasted
stack traces rather than reference answers.

`eval/gold_set.csv` carries a `verified` column and a `verify_note` giving the
reason for every decision. 7 rows passed; 32 did not, and the notes say why.

Retrieval@5 needs no model, so the eval runs offline and for free.

## Run it

```bash
pip install -r requirements.txt

python ingest/scrape.py            # 1,338 pages via sitemap (~12 min)
python eval/retrieval_only.py      # BM25 eval — no API key needed
python eval/prepare_review.py      # gold-set checklist -> eval/review.md
python eval/tune_bm25.py           # parameter sweep
python eval/compare_retrievers.py  # BM25 vs LSA vs hybrid — still no key
uvicorn api.main:app --reload      # demo at localhost:8000 (no key needed)
```

Optional, for generated answers rather than retrieved passages:

```bash
cp .env.example .env               # add a free Gemini key
python -m api.providers --list     # confirm the key works
python ingest/embed.py --strategy recursive-800-overlap-100
python eval/compare.py             # dense retrieval + answer grading
```

## Layout

```
ingest/   scrape.py               sitemap crawl (docs nav is client-rendered,
                                  so link-following reaches ~5% of the site)
          chunk.py                four chunking strategies
          embed.py                dense index (optional)
api/      bm25.py                 Okapi BM25 in numpy — no key, no download
          lsa.py                  TF-IDF + truncated SVD dense retrieval, RRF hybrid
          providers.py            gemini | groq | openai behind one interface
          retrieve.py             dense search + grounded answer
          main.py                 FastAPI: /query, /scorecard
eval/     collect_candidates.py   pull solved threads from the forum API
          validate_candidates.py  resolve redirects, flag unusable answers
          prepare_review.py       pre-screen the gold set -> review.md
          tune_bm25.py            24-config sweep to establish the ceiling
          gold_set.csv            39 questions  <- the actual asset
          run_eval.py             Retrieval@k, strict and lenient
          retrieval_only.py       key-free eval across all strategies
          compare_retrievers.py   BM25 vs LSA vs hybrid, with Wilson intervals
          build_gold.py           rebuild the gold set, preserving verdicts
web/      index.html              demo page; scorecard read from eval output
```

## Design notes

**No vector database.** At 6,748 chunks a normalised numpy matrix beats a network
round-trip, and it keeps the project runnable with no credentials.

**No vendor lock-in.** `providers.py` speaks raw HTTP to Gemini, Groq or OpenAI,
chosen by one environment variable — so the same gold set can be scored across
providers to compare cost per point of accuracy.

**The eval imports the same retriever the API serves.** If the benchmark exercises
a different code path than production, the number it prints is fiction.
