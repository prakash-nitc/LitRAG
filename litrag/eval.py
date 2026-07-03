"""Evaluation harness — the point of the whole project.

Retrieval (local, free, runs over the FULL question set):
- recall@k  — is the ground-truth paper among the top-k results' papers?
  Paper-level ground truth is deliberate: several chunks of a paper can contain
  an answer, so chunk-level "the one right chunk" would be ambiguous and punish
  correct retrievals.
- MRR       — mean reciprocal rank of the first correct paper (1/rank, 0 if absent).

Generation (costs LLM calls, so it runs on a subset):
- citation validity — are all [paper:idx] labels cited in the answer actually
  among the retrieved evidence? (mechanical check, catches fabricated citations)
- hint hit          — does the answer contain the expected key phrase? (cheap proxy)
- judge correctness — an LLM judge marks the answer correct/incorrect given the
  hint. Caveat stated openly: the judge is the same model family as the answerer.
- refusal accuracy  — on unanswerable questions, did the system decline instead
  of hallucinating?
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from .search import Searcher


def load_qa(path: str = "eval/qa_set.jsonl") -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def answerable(qa: List[dict]) -> List[dict]:
    return [q for q in qa if q["paper"]]


def unanswerable(qa: List[dict]) -> List[dict]:
    return [q for q in qa if not q["paper"]]


# ---------------- retrieval ----------------

def eval_retrieval(searcher: Searcher, qa: List[dict], ks=(1, 3, 5)) -> dict:
    """Paper-level recall@k and MRR over all answerable questions."""
    qs = answerable(qa)
    hits_at = {k: 0 for k in ks}
    rr_sum = 0.0
    for item in qs:
        papers = [h.paper for h in searcher.search(item["q"], k=max(ks))]
        for k in ks:
            hits_at[k] += item["paper"] in papers[:k]
        if item["paper"] in papers:
            rr_sum += 1.0 / (papers.index(item["paper"]) + 1)
    n = len(qs)
    out = {f"recall@{k}": round(hits_at[k] / n, 3) for k in ks}
    out["mrr"] = round(rr_sum / n, 3)
    out["n"] = n
    out["config"] = searcher.tag()
    return out


# ---------------- generation ----------------

_CITE_RE = re.compile(r"\[([A-Za-z0-9\-]+):(\d+)\]")
_REFUSAL_RE = re.compile(
    r"do(es)? not (contain|include|provide|have)|no (information|relevant|mention)|"
    r"not (found|present|mentioned|available)|cannot (be )?(answer|find)|unable to",
    re.IGNORECASE,
)


def citations_valid(answer_text: str, hits) -> Optional[bool]:
    """True iff every cited label exists in the retrieved evidence.
    None if the answer cites nothing (counted separately)."""
    cited = set(_CITE_RE.findall(answer_text))
    if not cited:
        return None
    available = {(h.paper, str(h.idx)) for h in hits}
    return cited.issubset(available)


def looks_like_refusal(answer_text: str) -> bool:
    return bool(_REFUSAL_RE.search(answer_text))


def judge_correct(question: str, hint: str, answer_text: str, model: str) -> bool:
    from .generate import call_llm
    verdict = call_llm(
        [{"role": "user", "content":
          f"Question: {question}\nExpected key idea (reference): {hint}\n"
          f"Candidate answer: {answer_text}\n\n"
          "Does the candidate answer correctly address the question, consistent "
          "with the expected key idea? Reply with exactly YES or NO."}],
        model=model, max_tokens=4,
    )
    return verdict.strip().upper().startswith("YES")


def eval_generation(searcher: Searcher, qa: List[dict], n_answerable: int = 8,
                    k: int = 5, model: str = None) -> dict:
    """Answer quality on a subset (LLM calls are rate-limited on the free tier)."""
    from .generate import answer, DEFAULT_GEN_MODEL
    model = model or DEFAULT_GEN_MODEL

    # spread the subset across papers rather than taking the first N
    qs = answerable(qa)
    step = max(1, len(qs) // n_answerable)
    subset = qs[::step][:n_answerable]

    cite_ok = cite_none = hint_ok = judged_ok = 0
    for item in subset:
        r = answer(item["q"], searcher, k=k, model=model)
        v = citations_valid(r["answer"], r["hits"])
        if v is None:
            cite_none += 1
        elif v:
            cite_ok += 1
        hint_ok += item["hint"].lower() in r["answer"].lower()
        judged_ok += judge_correct(item["q"], item["hint"], r["answer"], model)

    refusals = 0
    un = unanswerable(qa)
    for item in un:
        r = answer(item["q"], searcher, k=k, model=model)
        refusals += looks_like_refusal(r["answer"])

    n = len(subset)
    return {
        "config": searcher.tag(), "model": model, "n_answerable": n,
        "citation_valid": round(cite_ok / max(n - cite_none, 1), 3),
        "uncited_answers": cite_none,
        "hint_hit": round(hint_ok / n, 3),
        "judge_correct": round(judged_ok / n, 3),
        "refusal_rate": round(refusals / len(un), 3) if un else None,
        "n_unanswerable": len(un),
    }
