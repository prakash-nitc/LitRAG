"""Evaluation metrics — if these are wrong, every number in the README is wrong."""
import pytest

from litrag.eval import (answerable, citations_valid, eval_retrieval,
                         looks_like_refusal, unanswerable)
from litrag.retrieve import Hit


class FakeSearcher:
    """Returns a fixed paper ranking per question."""

    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, query, k=5):
        return [Hit(1.0, p, 0, "") for p in self.rankings[query][:k]]

    def tag(self):
        return "fake"


QA = [
    {"q": "q1", "paper": "A", "hint": "h"},   # correct at rank 1
    {"q": "q2", "paper": "B", "hint": "h"},   # correct at rank 3
    {"q": "q3", "paper": "C", "hint": "h"},   # never retrieved
    {"q": "q4", "paper": None, "hint": None},  # unanswerable: excluded from retrieval eval
]
RANKINGS = {
    "q1": ["A", "B", "C", "D", "E"],
    "q2": ["D", "E", "B", "A", "C"],
    "q3": ["A", "B", "D", "E", "F"],
}


def test_answerable_unanswerable_split():
    assert [q["q"] for q in answerable(QA)] == ["q1", "q2", "q3"]
    assert [q["q"] for q in unanswerable(QA)] == ["q4"]


def test_recall_and_mrr_by_hand():
    out = eval_retrieval(FakeSearcher(RANKINGS), QA, ks=(1, 3, 5))
    assert out["n"] == 3
    assert out["recall@1"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["recall@3"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["recall@5"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-3)
    assert out["config"] == "fake"


def test_paper_level_ground_truth_counts_any_chunk_of_the_paper():
    hits = {"q1": ["A", "A", "A", "A", "A"]}
    out = eval_retrieval(FakeSearcher(hits), [{"q": "q1", "paper": "A", "hint": ""}])
    assert out["recall@1"] == 1.0 and out["mrr"] == 1.0


# ---------------- citation validity ----------------

EVIDENCE = [Hit(1.0, "WinCLIP", 12, ""), Hit(0.9, "CLIP", 3, "")]


def test_citations_all_in_evidence():
    assert citations_valid("Windows [WinCLIP:12] and text [CLIP:3].", EVIDENCE) is True


def test_fabricated_citation_is_caught():
    assert citations_valid("Claim [WinCLIP:12] and [PatchCore:7].", EVIDENCE) is False


def test_right_paper_wrong_chunk_is_still_invalid():
    assert citations_valid("Claim [WinCLIP:13].", EVIDENCE) is False


def test_uncited_answer_is_none_not_false():
    assert citations_valid("An answer with no citations.", EVIDENCE) is None


def test_hyphenated_paper_names_parse():
    assert citations_valid("See [QuoVadis-AD:4].", [Hit(1.0, "QuoVadis-AD", 4, "")]) is True


# ---------------- refusal detection ----------------

@pytest.mark.parametrize("text", [
    "The provided context does not contain information about the weather.",
    "There is no information about stock picks in these papers.",
    "I cannot answer this from the given chunks.",
    "The answer is not mentioned in the context.",
])
def test_refusals_detected(text):
    assert looks_like_refusal(text)


def test_grounded_answer_is_not_a_refusal():
    assert not looks_like_refusal("CLIP is trained with a contrastive objective [CLIP:3].")
