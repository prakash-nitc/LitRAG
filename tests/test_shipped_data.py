"""Integrity of the artifacts the repo ships: the QA set and the demo index.

The README's numbers are only reproducible if these stay consistent — a
re-chunked chunks.jsonl with a stale .npz would silently misalign citations.
"""
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_PATH = os.path.join(ROOT, "eval", "qa_set.jsonl")
CHUNKS_PATH = os.path.join(ROOT, "index", "chunks.jsonl")
INDEX_PATH = os.path.join(ROOT, "index", "dense_bge-small-en-v1.5.npz")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def qa():
    return read_jsonl(QA_PATH)


@pytest.fixture(scope="module")
def chunks():
    return read_jsonl(CHUNKS_PATH)


# ---------------- QA set ----------------

def test_qa_set_size_matches_readme(qa):
    assert sum(1 for q in qa if q["paper"]) == 46
    assert sum(1 for q in qa if not q["paper"]) == 6


def test_qa_schema(qa):
    for item in qa:
        assert set(item) == {"q", "paper", "hint"}
        assert item["q"].strip()
        if item["paper"]:
            assert item["hint"], f"answerable question without a hint: {item['q']}"
        else:
            assert item["hint"] is None


def test_no_duplicate_questions(qa):
    questions = [item["q"].strip().lower() for item in qa]
    assert len(questions) == len(set(questions))


def test_every_ground_truth_paper_is_in_the_corpus(qa, chunks):
    corpus = {c["paper"] for c in chunks}
    assert {q["paper"] for q in qa if q["paper"]} <= corpus


# ---------------- shipped index ----------------

def test_chunks_are_well_formed(chunks):
    assert len(chunks) == 505
    for c in chunks:
        assert c["n_words"] == len(c["text"].split())
        assert 30 <= c["n_words"] <= 220


def test_index_aligns_with_chunks(chunks):
    z = np.load(INDEX_PATH, allow_pickle=True)
    assert str(z["model"]) == "BAAI/bge-small-en-v1.5"
    assert z["vectors"].shape == (len(chunks), 384)
    assert [(str(p), int(i)) for p, i in zip(z["papers"], z["idxs"])] == \
        [(c["paper"], c["idx"]) for c in chunks]


def test_index_vectors_are_unit_norm():
    # search is a dot product; it is only cosine if every row is normalized
    vectors = np.load(INDEX_PATH, allow_pickle=True)["vectors"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
