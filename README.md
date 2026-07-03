# LitRAG — Retrieval-Augmented QA over research papers, measured properly

Ask questions across a corpus of computer-vision research papers and get answers
**with citations** — built the unglamorous way: **evaluation first**.

Most RAG demos are vibes. This one reports numbers: retrieval **recall@k** and answer
**faithfulness** over a curated question set, with an ablation of the choices that
actually matter — chunk size, embedding model, dense vs. hybrid retrieval, and reranking.

## Design principles

1. **Evaluation before demo.** The eval harness is built before the chat UI; every
   pipeline change is a measured diff, not a feeling.
2. **No framework magic.** Plain Python. At this corpus scale (~10–20 papers,
   a few thousand chunks) exact numpy cosine search beats a vector database on both
   speed and simplicity — dependencies are things I can explain line by line.
3. **Honest numbers.** Whatever the ablation says, ships. Negative results included.

## Pipeline

```
PDFs → text extraction → chunking (size/overlap configurable)
     → embedding index (sentence-transformers, numpy)
     → retrieval: dense cosine top-k  [+ optional BM25 hybrid, reranker]
     → LLM answer with inline citations [chunk N]
     → eval: recall@k / faithfulness over a curated QA set
```

## Status

- [x] Repo scaffold, corpus downloader
- [x] Ingestion: PDF → cleaned, chunked text (505 chunks over 12 papers)
- [x] Embedding index + dense retrieval (numpy exact cosine; 4/5 correct@1 on sanity probes)
- [x] BM25 + RRF hybrid + cross-encoder reranker — hybrid recovers the dense miss (5/5 correct@1):
      dense embeddings whiffed on the exact phrase "memory bank of nominal patch features";
      lexical BM25 nails it. Different failure modes are the whole point of hybrid.
- [x] Generation with citations — grounded-only prompting, [paper:chunk] citations,
      explicit refusal on out-of-corpus questions (verified); 429-aware backoff for the
      free-tier token budget (~3 calls/min at k=5)
- [ ] Eval harness (recall@k, faithfulness) + QA set
- [ ] Ablation grid + results
- [ ] Gradio demo

## Results — retrieval (46 curated questions, paper-level ground truth)

| config | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| dense (MiniLM) | 0.783 | 0.935 | 0.935 | 0.851 |
| BM25 | 0.870 | 0.978 | **1.000** | 0.915 |
| hybrid (RRF) | 0.804 | 0.935 | **1.000** | 0.879 |
| **hybrid + rerank** | **0.913** | **1.000** | **1.000** | **0.953** |

Reproduce: `python scripts/run_eval.py`

## Results — answer quality (hybrid retrieval, llama-3.1-8b-instant)

| metric | score | meaning |
|---|---|---|
| citation validity | **1.00** | every cited [paper:chunk] exists in the retrieved evidence — no fabricated citations (8/8) |
| judge correctness | **1.00** | LLM judge marks all subset answers correct (caveat: same-family judge) |
| key-phrase hit | 0.875 | 7/8 answers contain the expected key idea verbatim |
| refusal accuracy | **0.83** | 5/6 out-of-corpus questions correctly refused — **one leaked**: the model answered a general question from its own knowledge despite the grounded-only instruction. Grounding prompts reduce, not eliminate, parametric leakage. |

Small subset (8 answerable + 6 unanswerable) — the free-tier token budget paces LLM calls;
retrieval metrics above run on the full 46-question set. Reproduce:
`python scripts/run_eval.py --generation`

**Honest findings:**
- **BM25 beats dense embeddings here** — our hand-written questions share the papers'
  exact vocabulary ("memory bank", "harmonic averaging"), which favors lexical match.
  Paraphrased end-user queries would shift the balance toward dense; the eval measures
  what it measures.
- **Plain RRF hybrid does not beat BM25 at rank 1** — fusion averages in dense's weaker
  rankings. The win comes from the **cross-encoder reranker** on hybrid candidates:
  best rank-1, perfect recall@3.

## Corpus

Vision-language and anomaly-detection literature (CLIP, WinCLIP, LAVAD, VERA, …) —
papers I work with anyway, so I can judge answer quality myself.

```bash
python scripts/download_corpus.py   # fetches PDFs from arXiv into corpus/
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_corpus.py
# (further stages land as they are built — see Status)
```

## License

MIT
