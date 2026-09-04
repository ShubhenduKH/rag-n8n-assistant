"""Scrape docs.n8n.io into data/docs.json.

Polite by default: honours a delay between requests, sets a real User-Agent,
and stays inside docs.n8n.io. Run once — the output is cached to disk.
"""

import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = "https://docs.n8n.io/"
OUT = Path(__file__).resolve().parents[1] / "data" / "docs.json"
DELAY = 0.5          # seconds between requests — do not lower this
MAX_PAGES = 2000
HEADERS = {"User-Agent": "rag-n8n-assistant/0.1 (portfolio project; contact: you@example.com)"}

# Nav, sidebar and footer text repeats on every page and poisons retrieval.
STRIP = ["nav", "header", "footer", "script", "style", "aside", "noscript"]


def in_scope(url: str) -> bool:
    p = urlparse(url)
    if p.netloc != "docs.n8n.io":
        return False
    if any(url.lower().endswith(ext) for ext in (".png", ".jpg", ".svg", ".pdf", ".zip")):
        return False
    return True


def normalise(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def extract(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(STRIP):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return None

    text = main.get_text(" ", strip=True)
    if len(text) < 200:          # nav stubs and redirect pages
        return None

    title = soup.find("h1")
    return {
        "url": normalise(url),
        "title": title.get_text(strip=True) if title else "",
        "text": text,
    }


def links_from(html: str, base: str) -> set:
    soup = BeautifulSoup(html, "lxml")
    found = set()
    for a in soup.find_all("a", href=True):
        url = normalise(urljoin(base, a["href"]))
        if in_scope(url):
            found.add(url)
    return found


def crawl(root: str = ROOT, limit: int = MAX_PAGES) -> list[dict]:
    seen: set = set()
    queue = [normalise(root)]
    docs: list[dict] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and len(docs) < limit:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            resp = session.get(url, timeout=20)
        except requests.RequestException as exc:
            print(f"  skip {url} — {exc}")
            continue

        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            continue

        doc = extract(resp.text, url)
        if doc:
            docs.append(doc)
            print(f"  [{len(docs):>4}] {doc['title'][:60] or url}")

        for link in links_from(resp.text, url):
            if link not in seen:
                queue.append(link)

        time.sleep(DELAY)

    return docs


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"crawling {ROOT} (delay {DELAY}s, cap {MAX_PAGES})\n")

    docs = crawl()

    OUT.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    chars = sum(len(d["text"]) for d in docs)
    print(f"\n{len(docs)} pages · ~{chars:,} chars · ~{chars // 4:,} tokens")
    print(f"embedding cost at $0.02/1M: about ${chars / 4 / 1_000_000 * 0.02:.3f}")
    print(f"wrote {OUT}")
