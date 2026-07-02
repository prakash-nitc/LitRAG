"""Ingestion: PDF -> cleaned text -> overlapping chunks.

Chunking is the first ablation axis (size x overlap), so it is configurable here
and nowhere else. Chunks carry their source paper and position so answers can
cite them as [paper:chunk].
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Iterator, List


@dataclass
class Chunk:
    paper: str        # short name, e.g. "WinCLIP"
    idx: int          # chunk index within the paper
    text: str
    n_words: int

    def cite(self) -> str:
        return f"[{self.paper}:{self.idx}]"


def extract_text(pdf_path: str) -> str:
    """Extract raw text from a PDF (concatenated pages)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


_REFERENCES_RE = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def clean_text(raw: str) -> str:
    """Light cleanup tuned for arXiv papers.

    - cut everything after the References heading (citations pollute retrieval:
      they contain every buzzword and answer nothing)
    - de-hyphenate line-break splits ("detec-\ntion" -> "detection")
    - collapse whitespace
    """
    m = _REFERENCES_RE.search(raw)
    if m and m.start() > len(raw) * 0.4:   # only cut if References is in the back half
        raw = raw[: m.start()]
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def chunk_text(text: str, paper: str, size: int = 220, overlap: int = 40) -> List[Chunk]:
    """Split into overlapping word-window chunks.

    Word windows (not sentences) keep chunk size uniform, which matters for a fair
    chunk-size ablation. Overlap prevents answers from being cut at boundaries.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
        window = words[start: start + size]
        if len(window) < 30:               # drop tail fragments too small to answer anything
            continue
        chunks.append(Chunk(paper=paper, idx=i, text=" ".join(window), n_words=len(window)))
    return chunks


def ingest_folder(corpus_dir: str, size: int = 220, overlap: int = 40) -> Iterator[Chunk]:
    """Ingest every PDF in a folder. Yields chunks; paper name = filename stem."""
    pdfs = sorted(f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {corpus_dir} — run scripts/download_corpus.py first.")
    for fname in pdfs:
        paper = os.path.splitext(fname)[0]
        text = clean_text(extract_text(os.path.join(corpus_dir, fname)))
        yield from chunk_text(text, paper, size=size, overlap=overlap)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Ingest corpus PDFs into a chunks JSONL.")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--out", default="index/chunks.jsonl")
    p.add_argument("--size", type=int, default=220, help="chunk size in words")
    p.add_argument("--overlap", type=int, default=40, help="overlap in words")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    per_paper: dict = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in ingest_folder(args.corpus, size=args.size, overlap=args.overlap):
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            per_paper[chunk.paper] = per_paper.get(chunk.paper, 0) + 1
            n += 1
    print(f"{n} chunks (size={args.size}, overlap={args.overlap}) -> {args.out}")
    for paper, count in sorted(per_paper.items()):
        print(f"  {paper:<14} {count} chunks")


if __name__ == "__main__":
    main()
