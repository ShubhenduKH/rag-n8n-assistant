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

Python · FastAPI · pgvector (Supabase) · `text-embedding-3-small` · `gpt-5.6-luna`

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env                                # then fill in your keys

python ingest/scrape.py      # docs.n8n.io -> data/docs.json  (~10 min)
python ingest/chunk.py       # compare chunk counts per strategy
python ingest/embed.py       # embed + store
uvicorn api.main:app --reload
```

## Layout

```
ingest/   scrape.py, chunk.py, embed.py
api/      main.py            FastAPI: /query -> answer + citations
eval/     gold_set.csv       50 real questions  <- the actual asset
          run_eval.py        Retrieval@k and answer correctness
web/      single-page UI with the scorecard printed on it
```

## Why the eval matters

Anyone can wire an LLM to a vector store. The part that is hard to fake is
knowing whether the result is correct — and being willing to publish the number.
