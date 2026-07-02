"""Unified search entry point — the one function the demo and the eval call.

Every retrieval configuration in the ablation is expressible here:
    mode = dense | bm25 | hybrid     x     rerank = on | off
"""
from __future__ import annotations

from typing import List, Optional

from .retrieve import DenseRetriever, Hit
from .hybrid import BM25Retriever, HybridRetriever
from .rerank import Reranker

DEFAULT_INDEX = "index/dense_all-MiniLM-L6-v2.npz"
DEFAULT_CHUNKS = "index/chunks.jsonl"


class Searcher:
    def __init__(self, mode: str = "hybrid", rerank: bool = False,
                 index_path: str = DEFAULT_INDEX, chunks_path: str = DEFAULT_CHUNKS,
                 candidates: int = 20):
        if mode == "dense":
            self.retriever = DenseRetriever(index_path)
        elif mode == "bm25":
            self.retriever = BM25Retriever(chunks_path)
        elif mode == "hybrid":
            self.retriever = HybridRetriever(index_path, chunks_path, candidates)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.candidates = candidates
        self.reranker: Optional[Reranker] = Reranker() if rerank else None

    def search(self, query: str, k: int = 5) -> List[Hit]:
        if self.reranker is None:
            return self.retriever.search(query, k)
        candidates = self.retriever.search(query, self.candidates)
        return self.reranker.rerank(query, candidates, k)

    def tag(self) -> str:
        return self.mode + ("+rerank" if self.reranker else "")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Search the corpus.")
    p.add_argument("--query", required=True)
    p.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--rerank", action="store_true")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    s = Searcher(mode=args.mode, rerank=args.rerank)
    print(f'[{s.tag()}] "{args.query}"\n')
    for rank, hit in enumerate(s.search(args.query, args.k), 1):
        snippet = hit.text[:170].rsplit(" ", 1)[0]
        print(f"{rank}. {hit.cite():<18} score={hit.score:.3f}\n   {snippet}...\n")


if __name__ == "__main__":
    main()
