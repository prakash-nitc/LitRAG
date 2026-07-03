"""Run the evaluation across retrieval configurations.

Retrieval eval is local and free -> runs on the full question set for every
config. Generation eval costs rate-limited LLM calls -> runs on a subset,
only when --generation is passed.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litrag.eval import eval_retrieval, eval_generation, load_qa
from litrag.search import Searcher

CONFIGS = [("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]


def main():
    p = argparse.ArgumentParser(description="Evaluate retrieval (and optionally generation).")
    p.add_argument("--qa", default="eval/qa_set.jsonl")
    p.add_argument("--generation", action="store_true",
                   help="also run answer-quality eval (LLM calls; slower)")
    p.add_argument("--gen-config", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    qa = load_qa(args.qa)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for mode, rerank in CONFIGS:
        s = Searcher(mode=mode, rerank=rerank)
        row = eval_retrieval(s, qa)
        rows.append(row)
        print(f"{row['config']:<16} recall@1={row['recall@1']:.3f}  "
              f"recall@3={row['recall@3']:.3f}  recall@5={row['recall@5']:.3f}  "
              f"mrr={row['mrr']:.3f}   (n={row['n']})")

    path = os.path.join(args.out_dir, "retrieval_eval.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {path}")

    if args.generation:
        s = Searcher(mode=args.gen_config, rerank=False)
        g = eval_generation(s, qa)
        print(json.dumps(g, indent=2))
        with open(os.path.join(args.out_dir, "generation_eval.json"), "w") as f:
            json.dump(g, f, indent=2)


if __name__ == "__main__":
    main()
