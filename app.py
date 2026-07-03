"""LitRAG demo — ask questions over a corpus of CV research papers.

The default retrieval configuration is the one the ablation SELECTED:
bge-small-en-v1.5, dense-only, 220-word chunks (0.957 recall@1 / 0.978 MRR
on the 46-question eval). The demo literally runs the config the evaluation
chose — eval-first, end to end.

Without a GROQ_API_KEY the demo degrades gracefully to retrieval-only
(evidence chunks, no generated answer) — on HF Spaces, add the key as a
repository secret to enable answers.
"""
import gradio as gr

from litrag.search import Searcher

INDEX = "index/dense_bge-small-en-v1.5.npz"
CHUNKS = "index/chunks.jsonl"

_searchers: dict = {}


def get_searcher(mode: str, rerank: bool) -> Searcher:
    key = (mode, rerank)
    if key not in _searchers:
        _searchers[key] = Searcher(mode=mode, rerank=rerank,
                                   index_path=INDEX, chunks_path=CHUNKS)
    return _searchers[key]


def has_api_key() -> bool:
    try:
        from litrag.generate import load_api_key
        load_api_key()
        return True
    except Exception:
        return False


def respond(question: str, mode: str, rerank: bool, k: int):
    if not question or not question.strip():
        return "Ask a question about the papers.", ""
    searcher = get_searcher(mode, bool(rerank))

    if has_api_key():
        from litrag.generate import answer
        result = answer(question.strip(), searcher, k=int(k))
        answer_md = result["answer"]
        hits = result["hits"]
    else:
        hits = searcher.search(question.strip(), k=int(k))
        answer_md = ("*(Retrieval-only mode: no `GROQ_API_KEY` configured — showing the "
                     "evidence chunks. Add the key as a Space secret to enable answers.)*")

    evidence = "\n\n---\n\n".join(
        f"**{h.cite()}** · score {h.score:.3f}\n\n{h.text[:600]}…" for h in hits
    )
    return answer_md, evidence


with gr.Blocks(title="LitRAG — evaluated RAG over research papers") as demo:
    gr.Markdown(
        "## 📚 LitRAG — ask a corpus of vision/anomaly-detection papers\n"
        "Answers cite evidence as **[paper:chunk]** and refuse when the corpus lacks the "
        "answer. Default retrieval = **the ablation winner** (bge-small, dense-only: "
        "0.957 recall@1 over a 46-question eval — full tables in the "
        "[repo](https://github.com/prakash-nitc/LitRAG)). Corpus: CLIP, WinCLIP, PatchCore, "
        "PaDiM, AnomalyCLIP, AnomalyGPT, LLaVA, OVVAD, LAVAD, VERA, VAD-R1, and a survey."
    )
    with gr.Row():
        question = gr.Textbox(label="Question", scale=4,
                              placeholder="How does WinCLIP aggregate scores from overlapping windows?")
        ask = gr.Button("Ask", variant="primary", scale=1)
    with gr.Accordion("Retrieval settings (defaults = ablation winner)", open=False):
        mode = gr.Radio(["dense", "bm25", "hybrid"], value="dense", label="First stage")
        rerank = gr.Checkbox(value=False, label="Cross-encoder rerank (downloads a model on first use)")
        k = gr.Slider(3, 8, value=5, step=1, label="Evidence chunks (k)")
    answer_box = gr.Markdown("*Answer appears here.*")
    with gr.Accordion("Evidence (retrieved chunks)", open=True):
        evidence_box = gr.Markdown()

    gr.Examples(
        examples=[
            "How does WinCLIP aggregate scores from overlapping windows?",
            "What does PatchCore store in its memory bank?",
            "Does LAVAD require training, and what role does the LLM play?",
            "What does VERA learn instead of updating model weights?",
        ],
        inputs=question,
    )

    ask.click(respond, [question, mode, rerank, k], [answer_box, evidence_box])
    question.submit(respond, [question, mode, rerank, k], [answer_box, evidence_box])


if __name__ == "__main__":
    demo.launch()
