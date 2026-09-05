# rag-n8n-assistant

A retrieval assistant over the [n8n documentation](https://docs.n8n.io), built to
answer the question most RAG demos skip: **how often does it actually retrieve the
right page?**

The interesting part of this repo is not the chatbot. It is the evaluation —
a gold set built from real solved threads on the n8n community forum, and a
lexical baseline measured *before* reaching for embeddings.

## Results — BM25 baseline

| Retriever | Retrieval@5 (strict) | Retrieval@5 (any cited page) |
|---|---:|---:|
| bm25 · recursive-800 | **20.5%** | **25.6%** |
| bm25 · recursive-800-overlap-100 | 20.5% | 25.6% |
| bm25 · fixed-512 | 20.5% | 23.1% |
| bm25 · fixed-512-overlap-64 | 15.4% | 17.9% |

39 questions · 1,338 pages · 6,748 chunks · measured 2026-09-05
95% CI ±12.7pp (strict) — n=39 is small, so treat the ranking between the top
three as a tie rather than a result.

**This is the number a dense retriever has to beat.** Publishing a RAG accuracy
figure without a lexical baseline means you cannot tell how much of it your
embeddings actually earned — on developer documentation, where users search with
the same nouns the docs use, BM25 alone is often most of the way there.

Here it is not, and the eval says why:

| Where BM25 works | Where it fails |
|---|---|
| scheduling 50% · error-handling 40% | data-transform 0% · auth-credentials 0% |

Questions that name a thing (`GENERIC_TIMEZONE`, "Error Trigger") retrieve well.
Questions phrased conversationally — *"How can I use now() in an expression"* —
do not, because `use`, `now` and `expression` appear on hundreds of pages. The
gold page for that question **is** in the corpus and **does** contain all three
terms; it just ranks below the noise. That is a textbook case for dense
retrieval, and it is the next commit.

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

`eval/gold_set.csv` carries `verified=no` on every row. The rows are real and the
URLs resolve, but each answer still needs a human to confirm it against the page
before the number is trustworthy. That column is the honest state of this repo.

Retrieval@5 needs no model, so the eval runs offline and for free.

## Run it

```bash
pip install -r requirements.txt

python ingest/scrape.py            # 1,338 pages via sitemap (~12 min)
python eval/retrieval_only.py      # BM25 eval — no API key needed
uvicorn api.main:app --reload      # demo at localhost:8000
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
          providers.py            gemini | groq | openai behind one interface
          retrieve.py             dense search + grounded answer
          main.py                 FastAPI: /query, /scorecard
eval/     collect_candidates.py   pull solved threads from the forum API
          validate_candidates.py  resolve redirects, flag unusable answers
          gold_set.csv            39 questions  <- the actual asset
          run_eval.py             Retrieval@k, strict and lenient
          retrieval_only.py       key-free eval across all strategies
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
