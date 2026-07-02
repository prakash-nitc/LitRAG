"""Cross-encoder reranking — the second stage of two-stage retrieval.

A bi-encoder (our dense index) embeds query and passage SEPARATELY, then
compares vectors: fast, but the two texts never see each other. A cross-encoder
feeds query and passage TOGETHER through full attention and scores the actual
interaction — far more accurate, far too slow to run over the whole corpus.

Hence the classic pattern: a cheap first stage produces a short candidate list,
the cross-encoder reorders it. Accuracy of a slow model at the cost of a fast one.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .retrieve import Hit

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, hits: List[Hit], k: int = 5) -> List[Hit]:
        if not hits:
            return []
        self._ensure_loaded()
        scores = np.asarray(self._model.predict([(query, h.text) for h in hits]))
        order = np.argsort(-scores)[:k]
        return [Hit(float(scores[i]), hits[i].paper, hits[i].idx, hits[i].text)
                for i in order]
