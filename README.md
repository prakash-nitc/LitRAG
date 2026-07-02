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
- [ ] Generation with citations
- [ ] Eval harness (recall@k, faithfulness) + QA set
- [ ] Ablation grid + results
- [ ] Gradio demo

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
