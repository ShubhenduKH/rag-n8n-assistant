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

- **Headline: Retrieval@5 = 20.5% strict / 25.6% lenient** on 39 questions, 1,338
  pages, 6,748 chunks. BM25 only; no dense retrieval yet.
- **Every gold row is `verified=no`.** The rows are real and the URLs resolve, but
  a human has not confirmed each answer against its page. Do not present the
  number as final until that is done — `eval/prepare_review.py` writes the
  checklist.
- Dense retrieval (`ingest/embed.py`, `eval/compare.py`) is written but unrun:
  it needs a free Gemini key in `.env`.

## Facts that are easy to get wrong here

**The corpus and the gold URLs drift.** n8n restructured its docs — `/hosting/` →
`/deploy/`, `/data/` → `/build/work-with-data/`. Old URLs return `200` *after a
redirect*, so a naive liveness check passes them while scoring against a URL that
is not in the corpus. Always record the post-redirect URL.

**Retrieval is scored two ways.** `source_url` is the page the accepted answer
pointed at; `acceptable_urls` is every docs page the thread cited (25 of 39 cite
more than one). Report both — picking one silently hides a judgement call.

**60% of the gold set is unusable and that is the real story.** Threads get marked
solved by version bumps and screenshots, which correspond to no docs page. On the
15 consistent rows retrieval is 40.0%; on the 24 flagged rows it is 8.3%. The
20.5% headline is half gold-set noise.

**Do not adopt the tuning sweep's argmax.** `eval/tune_bm25.py` tries 24 configs;
the spread is 2.6pp strict. Picking the best cell on n=39 fits noise. Defaults
stand; the sweep is evidence of a ceiling.

**The scraper must use the sitemap.** The docs nav is client-rendered — link
crawling reaches ~5% of the site.

**The eval imports the same retriever the API serves.** Keep it that way; a
benchmark on a different code path is fiction.

## Conventions

- Publish whatever number the eval prints. An honest 20% with a stated method
  beats a flattering number, and the README says so explicitly.
- Every claim in the README maps to a script someone can rerun.
- `providers.py` wraps gemini/groq/openai in raw HTTP — no vendor SDKs.
- No vector database: 6,748 chunks fit in a numpy matrix.
