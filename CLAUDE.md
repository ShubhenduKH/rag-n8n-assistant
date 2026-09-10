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

- **Headline: Retrieval@5 = 33.3% on 18 verified rows; 20.4% across all 108.**
  1,338 pages, 6,748 chunks. BM25 only; no dense retrieval yet.
- **All 108 rows verified by hand. 18 passed.** Each decision and its reason is
  in `gold_set.csv` (`verified`, `verify_note`). n=18 means ±21.8pp — quote it
  with the interval, never bare.
- **Dense retrieval was tried and lost.** `api/lsa.py` (TF-IDF + truncated SVD,
  no API key) scores 27.8–33.3% verified against BM25's 33.3%, at every
  dimensionality from 64 to 512. The RRF hybrid does not rescue it. Run
  `python eval/compare_retrievers.py`.
- Hosted-embedding retrieval (`ingest/embed.py`, `eval/compare.py`) is written
  but unrun — it needs a free Gemini key in `.env`. Its baseline to beat is
  33.3% verified / 20.4% overall, not zero.
- Yield from forum threads is ~3%, so n=50 verified needs ~1,700 candidates.
  `eval/build_gold.py` rebuilds the set and preserves existing verdicts by
  thread URL, so widening the pool never discards verification work.

## Facts that are easy to get wrong here

**The corpus and the gold URLs drift.** n8n restructured its docs — `/hosting/` →
`/deploy/`, `/data/` → `/build/work-with-data/`. Old URLs return `200` *after a
redirect*, so a naive liveness check passes them while scoring against a URL that
is not in the corpus. Always record the post-redirect URL.

**Retrieval is scored two ways.** `source_url` is the page the accepted answer
pointed at; `acceptable_urls` is every docs page the thread cited (25 of 39 cite
more than one). Report both — picking one silently hides a judgement call.

**97% of a naively-sourced gold set is unusable, and that is the real story.**
Funnel: 600 threads -> 183 live URLs -> 111 prose answers -> 108 gold -> 18
verified. Threads get marked solved by a version bump, a screenshot, a retraction
or a bug report, none of which corresponds to a docs page.

**Beware small-sample gaps here.** An earlier pass measured 42.9% verified vs
8.3% rejected on n=7/n=24 and read it as a 5x effect. At n=18/n=90 it is 33.3%
vs 17.8% with nearly-touching intervals. The direction held; the magnitude was
noise. Re-check any subset claim against the current n before repeating it.

**Do not adopt the tuning sweep's argmax.** `eval/tune_bm25.py` tries 24 configs;
the spread is 2.6pp strict. Picking the best cell on n=39 fits noise. Defaults
stand; the sweep is evidence of a ceiling.

**Do not claim "embeddings don't help" from the LSA result.** LSA only learns
co-occurrence inside these 1,338 pages. It shows dense-ness alone is not the fix;
it says nothing about a model trained on far more text. Keep that distinction.

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
