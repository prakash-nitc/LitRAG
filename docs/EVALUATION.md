# Evaluation methodology

How the numbers in the README are produced, and what they can and cannot support.

## The QA set: `eval/qa_set.jsonl`

One JSON object per line:

```json
{"q": "How many image-text pairs was CLIP trained on?", "paper": "CLIP", "hint": "400 million"}
{"q": "What is the capital of France?", "paper": null, "hint": null}
```

- **46 answerable questions**, hand-written from the papers: 4 each for 11
  papers, 2 for the QuoVadis-AD survey. `paper` is the ground-truth source;
  `hint` is a short key phrase a correct answer should contain.
- **6 unanswerable questions** (`paper: null`): general-knowledge or
  off-topic questions (cooking, finance, algorithms) that the corpus cannot
  answer. These test refusal, not retrieval.

## Retrieval metrics (full set, local, deterministic)

For each answerable question, take the papers of the top-k retrieved chunks.

- **recall@k**: fraction of questions whose ground-truth paper appears in the top k.
- **MRR**: mean of `1 / rank` of the first chunk from the ground-truth paper (0 if absent from the top 5).

**Why paper-level ground truth.** Several chunks of a paper often contain the
answer (abstract, method, conclusion). Labelling one "correct chunk" would
penalize correct retrievals and depend on the chunking, which would break
comparisons across chunk sizes. The tradeoff is that this metric cannot tell
the most relevant chunk of a paper from a weaker one.

## Generation metrics (subset, LLM calls)

Run on 8 answerable questions (every 5th question, so the subset spans papers)
plus all 6 unanswerable ones. The free-tier rate limit is the constraint.

| metric | how it is computed |
|---|---|
| citation validity | every `[paper:idx]` in the answer must be among the retrieved hits. Purely mechanical; catches fabricated citations |
| uncited answers | answers with no citation at all, counted separately rather than as valid |
| key-phrase hit | case-insensitive substring match of `hint` in the answer |
| judge correctness | LLM judge sees question, hint and answer and replies YES/NO |
| refusal accuracy | on unanswerable questions, a regex for refusal phrasing ("does not contain", "no information", "cannot answer", ...) |

## Threats to validity

These limits are part of the result.

1. **Small n.** With 46 questions, one question is 2.2 points of recall@1.
   Many ablation gaps (e.g. 0.913 vs 0.957) are two questions. Read the grid
   for large, consistent patterns, not single cells.
2. **Vocabulary overlap.** The questions were written while reading the papers
   and reuse their terms. This favors BM25. Paraphrased user queries would
   likely shift the balance toward dense retrieval.
3. **Same-family judge.** The correctness judge is the same model family as the
   answerer and may share its blind spots. Treat judge correctness = 1.00 on 8
   questions as weak evidence.
4. **Heuristic refusal detection.** A refusal phrased unusually would be missed,
   and a hedged answer that happens to contain "not mentioned" would count as a
   refusal. The one leak reported in the README was checked by hand.
5. **Single run.** Retrieval is deterministic. Generation runs at temperature 0
   but is not guaranteed bit-identical across provider-side model updates.

## Reproducing

```bash
python scripts/download_corpus.py        # PDFs -> corpus/
python -m litrag.ingest                  # -> index/chunks.jsonl
python -m litrag.embed_index             # -> index/dense_all-MiniLM-L6-v2.npz
python scripts/run_eval.py               # -> results/retrieval_eval.csv
python scripts/run_eval.py --generation  # + results/generation_eval.json (needs GROQ_API_KEY)
python scripts/run_ablation.py           # -> results/ablation.csv (30 configs, resumable)
```

## Adding a question

Append a line to `eval/qa_set.jsonl` whose `paper` matches a corpus short name
(the PDF stem, e.g. `WinCLIP`), then run `python -m pytest tests/test_shipped_data.py`.
The schema and corpus-membership checks will flag typos. Update the expected
counts in that test, and the numbers in the README, when the set grows.
