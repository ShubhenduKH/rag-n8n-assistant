"""Provider abstraction for embeddings and chat.

The pipeline does not care who serves the model. Set PROVIDER in .env and the
rest of the project is unchanged — which is also what makes the eval portable:
you can score the same gold set across providers and compare cost per point of
accuracy.

Providers are raw HTTP rather than vendor SDKs on purpose: one dependency
(`requests`), no SDK version churn, and every request is inspectable.

    PROVIDER=gemini    free tier, no card required
    PROVIDER=groq      free tier, fast, chat only
    PROVIDER=openai    paid

Discover which models your key can actually use:
    python -m api.providers --list
"""

import os
import time
from typing import Iterable

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

TIMEOUT = 60
RETRIES = 4


def _post(url: str, payload: dict, headers: dict) -> dict:
    """POST with backoff on rate limits and transient failures."""
    last = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f"    {resp.status_code}, retrying in {wait}s")
                time.sleep(wait)
                last = f"{resp.status_code}: {resp.text[:200]}"
                continue
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {RETRIES} attempts — {last}")


# ---------------------------------------------------------------- Gemini

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at https://aistudio.google.com/apikey — no card needed."
        )
    return key


def gemini_list_models() -> list[dict]:
    resp = requests.get(f"{GEMINI_BASE}/models",
                        headers={"x-goog-api-key": _gemini_key()}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("models", [])


def gemini_embed(texts: list[str], model: str, task: str) -> list[list[float]]:
    """Batch embed. task is RETRIEVAL_DOCUMENT for the corpus and
    RETRIEVAL_QUERY for a search — asymmetric embedding measurably improves
    retrieval over using the same task type for both."""
    key = _gemini_key()
    requests_payload = [
        {"model": f"models/{model}",
         "content": {"parts": [{"text": t}]},
         "taskType": task}
        for t in texts
    ]
    data = _post(
        f"{GEMINI_BASE}/models/{model}:batchEmbedContents",
        {"requests": requests_payload},
        {"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    return [e["values"] for e in data["embeddings"]]


def gemini_chat(system: str, user: str, model: str) -> str:
    data = _post(
        f"{GEMINI_BASE}/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0},
        },
        {"x-goog-api-key": _gemini_key(), "Content-Type": "application/json"},
    )
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        reason = data.get("candidates", [{}])[0].get("finishReason", "unknown")
        return f"[no answer returned — finishReason={reason}]"


# ---------------------------------------------------------------- Groq

def groq_chat(system: str, user: str, model: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set — free key at https://console.groq.com/keys")
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "temperature": 0,
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]},
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- OpenAI

def openai_embed(texts: list[str], model: str) -> list[list[float]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    data = _post("https://api.openai.com/v1/embeddings",
                 {"model": model, "input": texts},
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return [d["embedding"] for d in data["data"]]


def openai_chat(system: str, user: str, model: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    data = _post("https://api.openai.com/v1/chat/completions",
                 {"model": model, "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- dispatch

DEFAULTS = {
    "gemini": {"embed": "gemini-embedding-001", "chat": "gemini-2.5-flash"},
    "groq":   {"embed": None,                   "chat": "llama-3.3-70b-versatile"},
    "openai": {"embed": "text-embedding-3-small", "chat": "gpt-5.6-luna"},
}


def provider() -> str:
    return os.getenv("PROVIDER", "gemini").lower()


def embed_model() -> str:
    return os.getenv("EMBED_MODEL") or DEFAULTS[provider()]["embed"]


def chat_model() -> str:
    return os.getenv("CHAT_MODEL") or DEFAULTS[provider()]["chat"]


def embed(texts: Iterable[str], is_query: bool = False) -> np.ndarray:
    """Returns an L2-normalised float32 matrix, so cosine similarity is a dot product."""
    texts = list(texts)
    name = provider()
    model = embed_model()

    if name == "gemini":
        vectors = gemini_embed(
            texts, model,
            task="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
        )
    elif name == "openai":
        vectors = openai_embed(texts, model)
    else:
        raise SystemExit(
            f"PROVIDER={name} has no embedding endpoint. "
            "Use gemini or openai for embeddings (groq is chat-only)."
        )

    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def chat(system: str, user: str) -> str:
    name = provider()
    model = chat_model()
    if name == "gemini":
        return gemini_chat(system, user, model)
    if name == "groq":
        return groq_chat(system, user, model)
    if name == "openai":
        return openai_chat(system, user, model)
    raise SystemExit(f"unknown PROVIDER={name}")


if __name__ == "__main__":
    import sys

    if "--list" in sys.argv:
        print("models your GEMINI_API_KEY can use:\n")
        for m in gemini_list_models():
            methods = ",".join(m.get("supportedGenerationMethods", []))
            print(f"  {m['name'].removeprefix('models/'):<42} {methods}")
    else:
        print(f"PROVIDER={provider()}  embed={embed_model()}  chat={chat_model()}")
        vec = embed(["hello world"])
        print(f"embedding ok — shape {vec.shape}, norm {np.linalg.norm(vec[0]):.4f}")
        print("chat ok —", chat("Answer in exactly one word.", "Say: working")[:80])
