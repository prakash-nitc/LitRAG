"""Embedding index: chunks.jsonl -> L2-normalized vectors in a single .npz.

Model choice is the second ablation axis, so models are a registry here.
Some retrieval models (E5, BGE) require instruction prefixes on queries and/or
passages — forgetting them silently degrades recall, so the prefixes live next
to the model name and nowhere else.

Normalized vectors mean cosine similarity == dot product, which keeps search
a single numpy matmul (no vector DB needed at this corpus scale).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

# model name -> instruction prefixes (query, passage)
MODELS: Dict[str, Dict[str, str]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {"query": "", "passage": ""},
    "BAAI/bge-small-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
    "intfloat/e5-small-v2": {"query": "query: ", "passage": "passage: "},
}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def model_slug(name: str) -> str:
    return name.split("/")[-1]


class Embedder:
    """Lazy wrapper around a sentence-transformers model with prefix handling."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model {model_name!r}; add it to MODELS with its prefixes.")
        self.model_name = model_name
        self.prefixes = MODELS[model_name]
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode_passages(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._ensure_loaded()
        prefixed = [self.prefixes["passage"] + t for t in texts]
        return self._model.encode(prefixed, batch_size=batch_size,
                                  normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 100).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode([self.prefixes["query"] + query],
                                  normalize_embeddings=True).astype(np.float32)[0]


def load_chunks(chunks_path: str) -> List[dict]:
    with open(chunks_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(chunks_path: str, model_name: str = DEFAULT_MODEL,
                out_dir: str = "index") -> str:
    """Embed all chunks and persist vectors + metadata in one npz."""
    chunks = load_chunks(chunks_path)
    embedder = Embedder(model_name)
    vectors = embedder.encode_passages([c["text"] for c in chunks])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dense_{model_slug(model_name)}.npz")
    np.savez(
        out_path,
        vectors=vectors,
        model=np.array(model_name),
        papers=np.array([c["paper"] for c in chunks], dtype=object),
        idxs=np.array([c["idx"] for c in chunks]),
        texts=np.array([c["text"] for c in chunks], dtype=object),
    )
    print(f"{len(chunks)} chunks embedded with {model_name} "
          f"(dim={vectors.shape[1]}) -> {out_path}")
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Build a dense embedding index from chunks.")
    p.add_argument("--chunks", default="index/chunks.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    p.add_argument("--out-dir", default="index")
    args = p.parse_args()
    build_index(args.chunks, args.model, args.out_dir)


if __name__ == "__main__":
    main()
