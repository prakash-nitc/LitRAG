"""Download the paper corpus from arXiv into corpus/.

The corpus is vision-language / anomaly-detection literature — papers I already
work with, so answer quality is easy to judge by eye. PDFs are gitignored;
this script is the reproducible way to rebuild the corpus.
"""
import os
import sys
import time

import requests

# (arxiv_id, short_name) — short_name becomes the filename and citation label
PAPERS = [
    ("2103.00020", "CLIP"),
    ("2304.08485", "LLaVA"),
    ("2106.08265", "PatchCore"),
    ("2011.08785", "PaDiM"),
    ("2303.14814", "WinCLIP"),
    ("2310.18961", "AnomalyCLIP"),
    ("2308.15366", "AnomalyGPT"),
    ("2311.07042", "OVVAD"),
    ("2404.01014", "LAVAD"),
    ("2412.01095", "VERA"),
    ("2505.19877", "VAD-R1"),
    ("2412.18298", "QuoVadis-AD"),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")


def download(arxiv_id: str, name: str) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    path = os.path.join(OUT_DIR, f"{name}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        print(f"  [skip] {name} (already present)")
        return True
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError("response is not a PDF")
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  [ok]   {name}  ({len(r.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {name} ({arxiv_id}): {e}")
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {len(PAPERS)} papers -> {OUT_DIR}")
    ok = 0
    for arxiv_id, name in PAPERS:
        ok += download(arxiv_id, name)
        time.sleep(1.0)  # be polite to arXiv
    print(f"\n{ok}/{len(PAPERS)} papers ready.")
    if ok < len(PAPERS):
        print("Some downloads failed — rerun, or fetch those PDFs manually into corpus/.")
        sys.exit(1)


if __name__ == "__main__":
    main()
