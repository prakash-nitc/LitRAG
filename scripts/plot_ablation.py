"""Plot the ablation grid as a recall@1 heatmap: (model, mode) x chunk size.

Reads results/ablation.csv (written by scripts/run_ablation.py) and writes
docs/figures/ablation_recall1.png for the README. Plotting only; it never
re-runs retrieval, so the figure always shows exactly the CSV's numbers.
Needs matplotlib (not in requirements.txt: the demo doesn't use it).
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# one-hue sequential ramp, light -> dark (magnitude only; no rainbow)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SURFACE = "#1f1f1d", "#6b6a66", "#ffffff"

MODEL_ORDER = ["-", "all-MiniLM-L6-v2", "bge-small-en-v1.5", "e5-small-v2"]
MODEL_LABEL = {"-": "BM25 only", "all-MiniLM-L6-v2": "MiniLM-L6",
               "bge-small-en-v1.5": "bge-small", "e5-small-v2": "e5-small"}
MODE_ORDER = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    sizes = sorted({int(r["chunk_size"]) for r in rows})
    keys = sorted({(r["model"], r["config"]) for r in rows},
                  key=lambda k: (MODEL_ORDER.index(k[0]), MODE_ORDER.index(k[1])))
    grid = [[next(float(r["recall@1"]) for r in rows
                  if (r["model"], r["config"]) == key and int(r["chunk_size"]) == s)
             for s in sizes] for key in keys]
    return keys, sizes, grid, len(rows)


def plot(keys, sizes, grid, n_configs, out):
    flat = [v for row in grid for v in row]
    best = max(flat)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(grid, cmap=cmap, vmin=0.70, vmax=1.0, aspect="auto")

    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            dark = (v - 0.70) / 0.30 > 0.5
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color=SURFACE if dark else INK,
                    fontweight="bold" if v == best else "normal")

    # white gutters between cells, heavier between model groups
    ax.set_xticks([x + 0.5 for x in range(len(sizes) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(keys) - 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    for i in range(1, len(keys)):
        if keys[i][0] != keys[i - 1][0]:
            ax.axhline(i - 0.5, color=SURFACE, linewidth=6)
    ax.tick_params(which="both", length=0)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s} words" for s in sizes], color=INK)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{MODEL_LABEL[m]} · {mode}" if m != "-" else MODEL_LABEL[m]
                        for m, mode in keys], color=INK)
    ax.xaxis.set_ticks_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"Retrieval recall@1 across {n_configs} configurations (46 questions)",
                 loc="left", fontsize=11, color=INK, pad=28)
    ax.text(0, -0.035, f"Bold: best ({best:.3f}).  One question = {1 / 46:.3f} recall@1.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser(description="Plot the ablation recall@1 heatmap.")
    p.add_argument("--csv", default="results/ablation.csv")
    p.add_argument("--out", default="docs/figures/ablation_recall1.png")
    args = p.parse_args()
    plot(*load(args.csv), out=args.out)


if __name__ == "__main__":
    main()
