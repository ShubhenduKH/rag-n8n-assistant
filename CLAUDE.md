# rag-n8n-assistant

RAG over the n8n docs. **The evaluation is the point of this repo, not the chatbot.**

## Run it

```bash
pip install -r requirements.txt
python eval/retrieval_only.py      # BM25 eval — no API key
uvicorn api.main:app --reload      # demo — no API key
```

The corpus ships committed as `data/docs.json.gz` (1.0 MB), so a fresh clone runs
immediately. `ingest/scrape.py` only needs re-running to refresh it; after that,
`python -m ingest.corpus` re-packs the gz.

## Current state

- **Raw headline: Retrieval@5 = 20.5% strict / 25.6% lenient** on 39 questions,
  1,338 pages, 6,748 chunks. BM25 only; no dense retrieval yet.
- **All 39 rows have been verified by hand. 7 passed.** Each decision and its
  reason is in `gold_set.csv` (`verified`, `verify_note`). On those 7 rows
  retrieval is 42.9% / 57.1% — but n=7 means ±37pp, so quote it as a direction,
  never as a headline.
- Dense retrieval (`ingest/embed.py`, `eval/compare.py`) is written but unrun:
  it needs a free Gemini key in `.env`.
- A wider collection (600 threads) is the way to grow the verified set; yield is
  ~3.5%, so n=50 verified needs roughly 1,400 candidates.

## Facts that are easy to get wrong here

**The corpus and the gold URLs drift.** n8n restructured its docs — `/hosting/` →
`/deploy/`, `/data/` → `/build/work-with-data/`. Old URLs return `200` *after a
redirect*, so a naive liveness check passes them while scoring against a URL that
is not in the corpus. Always record the post-redirect URL.

**Retrieval is scored two ways.** `source_url` is the page the accepted answer
pointed at; `acceptable_urls` is every docs page the thread cited (25 of 39 cite
more than one). Report both — picking one silently hides a judgement call.

**96.5% of a naively-sourced gold set is unusable, and that is the real story.**
The funnel is 200 threads -> 59 live URLs -> 39 gold -> 15 pre-screened -> 7
verified. Threads get marked solved by a version bump, a screenshot, a retraction
or a bug report, none of which corresponds to a docs page. Retrieval is 42.9% on
verified rows against 8.3% on rejected ones, so the 20.5% headline is mostly
gold-set noise rather than retriever behaviour.

**Do not adopt the tuning sweep's argmax.** `eval/tune_bm25.py` tries 24 configs;
the spread is 2.6pp strict. Picking the best cell on n=39 fits noise. Defaults
stand; the sweep is evidence of a ceiling.

**The scraper must use the sitemap.** The docs nav is client-rendered — link
crawling reaches ~5% of the site.

**The eval imports the same retriever the API serves.** Keep it that way; a
benchmark on a different code path is fiction.

**Reconfigure stdout to UTF-8 in any script that prints scraped text.** Windows
defaults to cp1252 and raises UnicodeEncodeError on an arrow or smart quote, but
only when output is redirected — so it passes interactively and dies in CI.

## Conventions

- Publish whatever number the eval prints. An honest 20% with a stated method
  beats a flattering number, and the README says so explicitly.
- Every claim in the README maps to a script someone can rerun.
- `providers.py` wraps gemini/groq/openai in raw HTTP — no vendor SDKs.
- No vector database: 6,748 chunks fit in a numpy matrix.
