FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The corpus is committed so the image boots with a working index and needs no
# credentials. Without data/docs.json the app still starts, but /query returns
# 503 until you run ingest/scrape.py.
ENV PORT=8000
EXPOSE 8000

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
