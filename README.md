# rag-n8n-assistant

A retrieval-augmented assistant over the [n8n](https://docs.n8n.io) documentation,
built to answer a question most RAG demos avoid: **how often is it actually right?**

## Results

<!-- Replace after running the eval. Publish the real numbers, whatever they are. -->

| Chunking | Retrieval@5 | Answer correct |
|---|---:|---:|
| _pending_ | — | — |

**Method.** 50 questions taken from real threads on community.n8n.io, each with a
verified answer and the docs URL that answers it. Retrieval@5 asks whether that
URL appears in the top 5 retrieved chunks — objective, no LLM involved. Answer
correctness is graded by an LLM judge against the reference answer, with 10
judgements verified by hand.

Gold set: [`eval/gold_set.csv`](eval/gold_set.csv) · measured `<date>`

## Live demo

<!-- your deployed URL -->

## Stack

Python · FastAPI · numpy vector index · pluggable model provider

**No vector database.** At this corpus size a normalised numpy matrix beats a
network round-trip to a hosted store, and it keeps the project runnable with a
single free API key.

**No vendor lock-in.** `api/providers.py` speaks raw HTTP to Gemini, Groq or
OpenAI, selected by one environment variable. The same gold set can therefore be
scored across providers to compare cost per point of accuracy.

Default is the **Gemini free tier**, which needs no credit card.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env                                # then fill in your keys

python -m api.providers --list                            # what your key can use
python ingest/scrape.py                                   # docs -> data/docs.json (~10 min)
python ingest/embed.py --strategy recursive-800-overlap-100   # build the index
python eval/compare.py                                    # score every strategy
uvicorn api.main:app --reload                             # demo at localhost:8000
```

## Layout

```
ingest/   scrape.py              crawl docs.n8n.io
          chunk.py               four chunking strategies
          embed.py               build the numpy index
api/      providers.py           gemini | groq | openai behind one interface
          retrieve.py            search + grounded answer (shared with the eval)
          main.py                FastAPI: /query, /scorecard
eval/     collect_candidates.py  pull solved threads from the forum API
          validate_candidates.py drop dead URLs, flag unusable answers
          gold_set.csv           50 verified questions  <- the actual asset
          run_eval.py            Retrieval@k + answer correctness
          compare.py             score every strategy, emit the table
web/      index.html             demo page, scorecard served from eval output
```

## Why the eval matters

Anyone can wire an LLM to a vector store. The part that is hard to fake is
knowing whether the result is correct — and being willing to publish the number.
