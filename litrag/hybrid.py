"""Lexical BM25 retrieval + Reciprocal Rank Fusion (RRF) hybrid.

Dense and lexical retrieval fail differently: embeddings capture paraphrase but
can whiff on exact terms (acronyms, method names, Greek-letter hyperparameters);
BM25 nails exact terms but has no notion of meaning. The hybrid keeps both.

Fusion is by RANK, not score: a cosine in [-1, 1] and an unbounded BM25 score
share no scale, so mixing raw scores is brittle. RRF — score(d) = sum over
systems of 1 / (K + rank_d) — needs no normalization and is hard to break.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from .embed_index import load_chunks
from .retrieve import DenseRetriever, Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over the chunk corpus. Index builds in-memory at init
    (instant at this scale; nothing to persist)."""

    def __init__(self, chunks_path: str = "index/chunks.jsonl"):
        self.chunks = load_chunks(chunks_path)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, k: int = 5) -> List[Hit]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        top = np.argsort(-scores)[:k]
        return [Hit(float(scores[i]), self.chunks[i]["paper"],
                    self.chunks[i]["idx"], self.chunks[i]["text"]) for i in top]


def rrf_fuse(rankings: List[List[Hit]], K: int = 60, top_k: int = 5) -> List[Hit]:
    """Reciprocal Rank Fusion. K=60 is the standard from the original paper;
    it damps the influence of any single system's top ranks."""
    scores: dict = {}
    first_seen: dict = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.paper, hit.idx)
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank)
            first_seen.setdefault(key, hit)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [Hit(scores[key], first_seen[key].paper, first_seen[key].idx,
                first_seen[key].text) for key in ordered]


class HybridRetriever:
    """Dense + BM25 candidates fused with RRF."""

    def __init__(self, index_path: str, chunks_path: str = "index/chunks.jsonl",
                 candidates: int = 20):
        self.dense = DenseRetriever(index_path)
        self.bm25 = BM25Retriever(chunks_path)
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> List[Hit]:
        return rrf_fuse(
            [self.dense.search(query, self.candidates),
             self.bm25.search(query, self.candidates)],
            top_k=k,
        )
