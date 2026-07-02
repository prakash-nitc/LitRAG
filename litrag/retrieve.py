"""Dense retrieval: query -> top-k chunks by cosine similarity.

Search is one matmul over the whole corpus (vectors are L2-normalized, so
dot product = cosine). At ~10^3 chunks this is sub-millisecond — exact search,
no approximate index, nothing to tune or explain away.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .embed_index import Embedder


@dataclass
class Hit:
    score: float
    paper: str
    idx: int
    text: str

    def cite(self) -> str:
        return f"[{self.paper}:{self.idx}]"


class DenseRetriever:
    def __init__(self, index_path: str):
        z = np.load(index_path, allow_pickle=True)
        self.vectors: np.ndarray = z["vectors"]          # (N, d), normalized
        self.papers = z["papers"]
        self.idxs = z["idxs"]
        self.texts = z["texts"]
        self.model_name = str(z["model"])
        self.embedder = Embedder(self.model_name)

    def search(self, query: str, k: int = 5) -> List[Hit]:
        q = self.embedder.encode_query(query)            # (d,), normalized
        sims = self.vectors @ q                          # cosine similarities
        top = np.argsort(-sims)[:k]
        return [Hit(float(sims[i]), str(self.papers[i]), int(self.idxs[i]),
                    str(self.texts[i])) for i in top]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Query the dense index.")
    p.add_argument("--index", default="index/dense_all-MiniLM-L6-v2.npz")
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    retriever = DenseRetriever(args.index)
    print(f'query: "{args.query}"  (model: {retriever.model_name})\n')
    for rank, hit in enumerate(retriever.search(args.query, args.k), 1):
        snippet = hit.text[:180].rsplit(" ", 1)[0]
        print(f"{rank}. {hit.cite():<18} score={hit.score:.3f}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
