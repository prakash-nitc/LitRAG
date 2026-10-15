"""BM25, Reciprocal Rank Fusion, and the Searcher front door.

No embedding model is loaded here: these cover the lexical path and the
fusion arithmetic, which is where silent ranking bugs would hide.
"""
import json

import pytest

from litrag.hybrid import BM25Retriever, rrf_fuse, tokenize
from litrag.retrieve import Hit
from litrag.search import Searcher

DOCS = [
    ("PatchCore", 0, "we store a memory bank of nominal patch features for anomaly scoring"),
    ("CLIP", 0, "contrastive pretraining aligns image and text embeddings at scale"),
    ("WinCLIP", 0, "window based zero shot anomaly segmentation with clip"),
    ("LAVAD", 0, "language models reason over video captions to detect anomalies"),
]


@pytest.fixture
def chunks_path(tmp_path):
    path = tmp_path / "chunks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for paper, idx, text in DOCS:
            f.write(json.dumps({"paper": paper, "idx": idx, "text": text,
                                "n_words": len(text.split())}) + "\n")
    return str(path)


def hit(paper, idx=0):
    return Hit(0.0, paper, idx, f"{paper} text")


# ---------------- tokenize ----------------

def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("CLIP's ViT-B/16, alpha=0.5") == \
        ["clip", "s", "vit", "b", "16", "alpha", "0", "5"]


# ---------------- RRF ----------------

def test_rrf_rewards_agreement_between_systems():
    dense = [hit("x"), hit("y"), hit("z")]
    lexical = [hit("y"), hit("w"), hit("x")]
    fused = rrf_fuse([dense, lexical], K=60, top_k=4)
    # y: 1/62 + 1/61  >  x: 1/61 + 1/63  >  w: 1/62  >  z: 1/63
    assert [h.paper for h in fused] == ["y", "x", "w", "z"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_dedupes_by_paper_and_chunk_and_truncates():
    a = [hit("p", 0), hit("p", 1), hit("q", 0)]
    b = [hit("p", 1), hit("p", 0)]
    fused = rrf_fuse([a, b], top_k=2)
    assert len(fused) == 2
    assert {(h.paper, h.idx) for h in fused} == {("p", 0), ("p", 1)}


def test_rrf_of_nothing_is_empty():
    assert rrf_fuse([[], []]) == []


# ---------------- BM25 / Searcher ----------------

def test_bm25_finds_the_exact_phrase(chunks_path):
    # the phrase dense MiniLM missed in the sanity probes; lexical match must not
    hits = BM25Retriever(chunks_path).search("memory bank of nominal patch features", k=2)
    assert hits[0].paper == "PatchCore"
    assert hits[0].score > hits[1].score


def test_searcher_bm25_mode_and_tag(chunks_path):
    s = Searcher(mode="bm25", chunks_path=chunks_path)
    assert s.tag() == "bm25"
    assert [h.paper for h in s.search("contrastive image text", k=1)] == ["CLIP"]


def test_searcher_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Searcher(mode="sparse")
