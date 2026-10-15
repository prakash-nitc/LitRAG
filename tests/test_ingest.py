"""Chunking and cleaning — the first stage, and the first ablation axis.

Chunk boundaries decide what retrieval can ever return, so the window
arithmetic is pinned down exactly here.
"""
import pytest

from litrag.ingest import Chunk, chunk_text, clean_text


def words(n):
    return " ".join(f"w{i}" for i in range(n))


# ---------------- chunk_text ----------------

def test_window_starts_and_sizes():
    # 500 words, size 220, overlap 40 -> step 180 -> starts at 0, 180, 360
    chunks = chunk_text(words(500), "P", size=220, overlap=40)
    assert [c.text.split()[0] for c in chunks] == ["w0", "w180", "w360"]
    assert [c.n_words for c in chunks] == [220, 220, 140]
    assert [c.idx for c in chunks] == [0, 1, 2]


def test_consecutive_chunks_share_exactly_the_overlap():
    chunks = chunk_text(words(1000), "P", size=120, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-30:] == b.text.split()[:30]


def test_every_word_is_covered():
    text = words(777)
    chunks = chunk_text(text, "P", size=220, overlap=40)
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(text.split())


def test_tiny_document_yields_nothing():
    # below the 30-word floor: too small to answer anything
    assert chunk_text(words(20), "P") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text(words(100), "P", size=50, overlap=50)


def test_chunk_carries_its_citation_label():
    c = Chunk(paper="WinCLIP", idx=12, text="...", n_words=1)
    assert c.cite() == "[WinCLIP:12]"


# ---------------- clean_text ----------------

def test_references_section_is_cut_when_in_back_half():
    body = " ".join(["body"] * 200)
    out = clean_text(body + "\nReferences\n[1] Foo et al. A paper. 2021.")
    assert "Foo" not in out
    assert out.endswith("body")


def test_early_references_heading_is_kept():
    # a "References" line near the start is not the bibliography — keep the text
    body = " ".join(["body"] * 200)
    out = clean_text("Intro\nReferences\n" + body)
    assert out.count("body") == 200


def test_line_break_hyphenation_is_joined_and_whitespace_collapsed():
    assert clean_text("anomaly detec-\ntion   with\n\n CLIP") == "anomaly detection with CLIP"
