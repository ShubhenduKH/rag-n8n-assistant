"""Fetch docs.n8n.io into data/docs.json.

Uses the sitemap rather than following links. The docs nav is client-rendered,
so a link crawler sees roughly 17 links per page and never reaches most of the
site; the sitemap enumerates all 1,300+ pages directly and cannot drift out of
sync with what the site actually publishes.

robots.txt allows this (`Content-Signal: ai-train=yes, search=yes`). The delay
below is deliberate — keep it.

    python ingest/scrape.py                # everything
    python ingest/scrape.py --limit 50     # quick sample while developing
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SITEMAP = "https://docs.n8n.io/sitemap.xml"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "docs.json"
URLS = ROOT / "data" / "urls.json"
DELAY = 0.4
RETRIES = 3
HEADERS = {"User-Agent": "rag-n8n-assistant/0.1 (portfolio project)"}

STRIP = ["nav", "header", "footer", "script", "style", "aside", "noscript", "svg"]
LOC = re.compile(r"<loc>(.*?)</loc>")


def get(session: requests.Session, url: str) -> requests.Response | None:
    """The docs host intermittently resets connections; retry rather than lose a page."""
    for attempt in range(RETRIES):
        try:
            resp = session.get(url, timeout=25)
            if resp.status_code == 200:
                return resp
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def collect_urls(session: requests.Session) -> list[str]:
    index = get(session, SITEMAP)
    if index is None:
        raise SystemExit(f"could not fetch {SITEMAP}")

    urls: list[str] = []
    for sub in LOC.findall(index.text):
        if not sub.endswith(".xml"):
            urls.append(sub)
            continue
        page = get(session, sub)
        if page is not None:
            urls.extend(u for u in LOC.findall(page.text) if not u.endswith(".xml"))
        time.sleep(DELAY)

    return sorted(set(urls))


def extract(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return None

    text = " ".join(main.get_text(" ", strip=True).split())
    if len(text) < 250:                      # redirect stubs and empty shells
        return None

    heading = soup.find("h1") or soup.find("h2")
    title = heading.get_text(strip=True) if heading else ""
    if not title:                            # fall back to the URL slug
        title = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()

    return {"url": url.rstrip("/"), "title": title, "text": text}


def main(limit: int | None) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    if URLS.exists():
        urls = json.loads(URLS.read_text(encoding="utf-8"))
        print(f"{len(urls)} urls from cache")
    else:
        print("reading sitemap...")
        urls = collect_urls(session)
        URLS.write_text(json.dumps(urls, indent=1), encoding="utf-8")
        print(f"{len(urls)} urls")

    if limit:
        urls = urls[:limit]

    docs, failed = [], 0
    for i, url in enumerate(urls, 1):
        resp = get(session, url)
        if resp is None:
            failed += 1
        else:
            doc = extract(resp.text, url)
            if doc:
                docs.append(doc)
        if i % 25 == 0 or i == len(urls):
            print(f"  {i:>5}/{len(urls)}  kept {len(docs)}  failed {failed}")
        time.sleep(DELAY)

    OUT.write_text(json.dumps(docs, indent=1, ensure_ascii=False), encoding="utf-8")

    chars = sum(len(d["text"]) for d in docs)
    print(f"\n{len(docs)} pages kept, {failed} failed")
    print(f"~{chars:,} chars (~{chars // 4:,} tokens)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
