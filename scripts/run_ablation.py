"""Ablation grid: chunk size x embedding model x retrieval mode.

Retrieval-only (local, free) so the full grid is affordable: BM25 depends only
on chunk size (3 rows); dense/hybrid/hybrid+rerank get the full cross-product
(3 sizes x 3 models x 3 modes = 27 rows). Generation is excluded on purpose —
30 configs x LLM calls would blow the free-tier budget without changing the
retrieval conclusions.

Resumable: chunk files and indexes are cached under index/ablation/ and reused
on rerun.
"""
import csv
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litrag.ingest import ingest_folder
from litrag.embed_index import MODELS, build_index, model_slug
from litrag.eval import eval_retrieval, load_qa
from litrag.search import Searcher

# size -> overlap (~18% of size)
CHUNK_SIZES = {120: 20, 220: 40, 400: 72}
ABL_DIR = "index/ablation"


def ensure_chunks(size: int, overlap: int) -> str:
    path = os.path.join(ABL_DIR, f"chunks_s{size}.jsonl")
    if os.path.isfile(path):
        return path
    os.makedirs(ABL_DIR, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for chunk in ingest_folder("corpus", size=size, overlap=overlap):
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            n += 1
    print(f"[ingest] size={size}: {n} chunks -> {path}", flush=True)
    return path


def ensure_index(chunks_path: str, model_name: str, size: int) -> str:
    path = os.path.join(ABL_DIR, f"dense_{model_slug(model_name)}_s{size}.npz")
    if os.path.isfile(path):
        return path
    print(f"[embed] {model_slug(model_name)} @ size={size} ...", flush=True)
    return build_index(chunks_path, model_name, out_path=path)


def main():
    qa = load_qa("eval/qa_set.jsonl")
    rows = []

    for size, overlap in CHUNK_SIZES.items():
        chunks_path = ensure_chunks(size, overlap)

        row = eval_retrieval(Searcher(mode="bm25", chunks_path=chunks_path), qa)
        row.update(chunk_size=size, model="-")
        rows.append(row)
        print(f"[eval] s{size:<4} bm25            R@1={row['recall@1']:.3f} mrr={row['mrr']:.3f}", flush=True)

        for model_name in MODELS:
            index_path = ensure_index(chunks_path, model_name, size)
            for mode, rerank in [("dense", False), ("hybrid", False), ("hybrid", True)]:
                s = Searcher(mode=mode, rerank=rerank,
                             index_path=index_path, chunks_path=chunks_path)
                row = eval_retrieval(s, qa)
                row.update(chunk_size=size, model=model_slug(model_name))
                rows.append(row)
                print(f"[eval] s{size:<4} {model_slug(model_name):<22} {s.tag():<14} "
                      f"R@1={row['recall@1']:.3f} mrr={row['mrr']:.3f}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = "results/ablation.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out}", flush=True)

    # compact pivot: recall@1 by (model, mode) x chunk size
    sizes = list(CHUNK_SIZES)
    print("\n=== recall@1 pivot ===")
    print(f"{'model':<22} {'mode':<14}" + "".join(f"  s{s:<5}" for s in sizes))
    keys = sorted({(r["model"], r["config"]) for r in rows})
    for model, config in keys:
        by_size = {r["chunk_size"]: r["recall@1"] for r in rows
                   if r["model"] == model and r["config"] == config}
        print(f"{model:<22} {config:<14}" +
              "".join(f"  {by_size.get(s, float('nan')):<6.3f}" for s in sizes))


if __name__ == "__main__":
    main()
