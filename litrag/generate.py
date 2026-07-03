"""Generation: retrieved chunks + LLM -> grounded answer with citations.

Design decisions that matter:
- The model may ONLY answer from the provided chunks, must cite [paper:idx]
  after each claim, and must say so when the context doesn't contain the answer.
  Retrieval adds knowledge; the refusal instruction is what keeps the model
  honest when retrieval comes back empty-handed.
- Plain HTTP to Groq's OpenAI-compatible endpoint (no SDK): one function,
  fully inspectable, provider swappable by changing one URL.
- The API key lives in .env (git-ignored) or the environment — never in code.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import requests

from .retrieve import Hit
from .search import Searcher

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GEN_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a careful research assistant answering questions about a corpus of "
    "computer-vision papers. Rules:\n"
    "1. Use ONLY the provided context chunks. Do not use outside knowledge.\n"
    "2. Cite the chunk label, e.g. [WinCLIP:12], immediately after each claim.\n"
    "3. If the context does not contain the answer, say exactly that — do not guess.\n"
    "4. Be concise: 2-6 sentences."
)


def load_api_key() -> str:
    """GROQ_API_KEY from the environment, else from .env in the repo root."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "No GROQ_API_KEY found. Put GROQ_API_KEY=... in .env (git-ignored) "
        "or export it as an environment variable."
    )


def build_context(hits: List[Hit]) -> str:
    return "\n\n".join(f"[{h.paper}:{h.idx}]\n{h.text}" for h in hits)


def call_llm(messages: List[dict], model: str = DEFAULT_GEN_MODEL,
             temperature: float = 0.0, max_tokens: int = 400,
             max_retries: int = 5) -> str:
    """One chat completion. temperature=0: factual QA wants determinism, not flair.

    Retries on 429/5xx with exponential backoff: the free tier allows ~6k
    tokens/min and each call with 5 context chunks is ~2k tokens, so any batch
    workload (like the eval harness) WILL hit the limit — pacing is part of the
    design, not an afterthought.
    """
    import time

    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {load_api_key()}"},
            json={"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = float(resp.headers.get("retry-after", delay))
            time.sleep(min(wait, 30.0))
            delay *= 2
            continue
        raise RuntimeError(f"LLM call failed ({resp.status_code}): {resp.text[:300]}")
    raise RuntimeError(f"LLM call failed after {max_retries} retries (rate limited).")


def answer(query: str, searcher: Optional[Searcher] = None, k: int = 5,
           model: str = DEFAULT_GEN_MODEL) -> dict:
    """Retrieve -> generate. Returns the answer plus the evidence it saw."""
    searcher = searcher or Searcher(mode="hybrid", rerank=False)
    hits = searcher.search(query, k=k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context chunks:\n\n{build_context(hits)}\n\n"
                                    f"Question: {query}"},
    ]
    return {"query": query, "answer": call_llm(messages, model=model),
            "hits": hits, "retrieval": searcher.tag(), "model": model}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Ask the corpus a question.")
    p.add_argument("--query", required=True)
    p.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--rerank", action="store_true")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--model", default=DEFAULT_GEN_MODEL)
    args = p.parse_args()

    result = answer(args.query, Searcher(mode=args.mode, rerank=args.rerank),
                    k=args.k, model=args.model)
    print(f"Q: {result['query']}")
    print(f"   (retrieval: {result['retrieval']}, model: {result['model']})\n")
    print(result["answer"])
    print("\nevidence:", " ".join(h.cite() for h in result["hits"]))


if __name__ == "__main__":
    main()
