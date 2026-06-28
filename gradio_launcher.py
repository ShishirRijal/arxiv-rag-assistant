"""
Gradio launcher — interactive demo UI for the RAG system.

Standalone entry point (separate from FastAPI).
Run with: uv run python gradio_launcher.py
Open:     http://localhost:7861

Why Gradio?
  - Turns a Python function into a web UI in ~20 lines
  - Built-in streaming support: yield tokens from the predict function
  - Shareable via Gradio's public tunnel (gradio.live link)
  - Perfect for demos and development testing

Features:
  - Live streaming: watch the answer appear token by token
  - Source citations: clickable arXiv links below the answer
  - Category filter: restrict retrieval to specific arXiv categories
  - Search mode selector: hybrid vs BM25-only
  - Character counter: shows prompt size before and after optimisation
"""

import json
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import requests

# ── Config ────────────────────────────────────────────────────────────────────
# The Gradio UI talks to the FastAPI backend via HTTP.
# This means it works whether FastAPI is running locally or in Docker.
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
ASK_URL        = f"{API_BASE}/api/v1/ask"
ASK_STREAM_URL = f"{API_BASE}/api/v1/ask/stream"

CATEGORY_CHOICES = [
    "All categories",
    "cs.AI  — Artificial Intelligence",
    "cs.LG  — Machine Learning",
    "cs.CL  — Computation and Language",
    "cs.CV  — Computer Vision",
    "cs.IR  — Information Retrieval",
    "cs.NE  — Neural and Evolutionary Computing",
    "stat.ML — Statistics - Machine Learning",
]

CATEGORY_MAP = {
    "cs.AI  — Artificial Intelligence":   "cs.AI",
    "cs.LG  — Machine Learning":          "cs.LG",
    "cs.CL  — Computation and Language":  "cs.CL",
    "cs.CV  — Computer Vision":           "cs.CV",
    "cs.IR  — Information Retrieval":     "cs.IR",
    "cs.NE  — Neural and Evolutionary Computing": "cs.NE",
    "stat.ML — Statistics - Machine Learning":    "stat.ML",
}

EXAMPLE_QUESTIONS = [
    "What is the transformer architecture and how does self-attention work?",
    "How does RLHF improve language model alignment?",
    "What are the main approaches to retrieval-augmented generation?",
    "Explain the key ideas behind diffusion models for image generation.",
    "What is chain-of-thought prompting and why does it help LLMs?",
]


# ── Predict function ──────────────────────────────────────────────────────────

def predict_streaming(
    question:   str,
    use_hybrid: bool,
    category:   str,
) -> gr.update:
    """
    Gradio predict function — streams tokens from the FastAPI SSE endpoint.

    Gradio's streaming works by yielding partial results from a generator.
    Each yield replaces the previous content in the output textbox.

    We hit the FastAPI /ask/stream endpoint (SSE) and relay tokens to Gradio.
    This keeps the Gradio UI decoupled from the pipeline internals.
    """
    if not question.strip():
        yield "Please enter a question."
        return

    # Resolve category filter
    cats = CATEGORY_MAP.get(category, None)
    cat_param = cats if cats else None

    # Build the SSE URL
    url = f"{ASK_STREAM_URL}?question={requests.utils.quote(question.strip())}"
    url += f"&use_hybrid={'true' if use_hybrid else 'false'}"
    if cat_param:
        url += f"&categories={cat_param}"

    full_answer = ""
    sources: list[dict] = []

    try:
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                # SSE format: each line is "data: ..."
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()

                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                content    = event.get("content", "")

                if event_type == "token":
                    full_answer += content
                    yield full_answer  # Gradio updates the display

                elif event_type == "sources":
                    sources = content or []

                elif event_type == "error":
                    yield f"Error from server: {content}"
                    return

    except requests.exceptions.ConnectionError:
        yield "Could not connect to the API. Is Docker Compose running?\n`docker compose up -d`"
        return
    except Exception as exc:
        yield f"Request failed: {exc}"
        return

    # Append formatted sources after all tokens
    if sources:
        sources_md = "\n\n---\n**Sources:**\n"
        for s in sources:
            sources_md += f"[{s['index']}] [{s['title']}]({s['url']})\n"
        yield full_answer + sources_md
    else:
        yield full_answer


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
## arXiv RAG Assistant

Ask questions about AI research papers. The system searches your ingested arXiv corpus using
hybrid BM25 + semantic search, then generates a grounded answer using a local LLM (Ollama).

**Tips:**
- Ask specific questions: *"How does RLHF work?"* not *"tell me about AI"*
- Citations `[1]`, `[2]` etc. in the answer correspond to the sources listed below
- Toggle **Hybrid search** off for faster keyword-only retrieval
"""

with gr.Blocks(
    title="arXiv RAG Assistant",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your question",
                placeholder="What is the main idea behind attention mechanisms in transformers?",
                lines=3,
            )
            with gr.Row():
                use_hybrid_toggle = gr.Checkbox(
                    value=True,
                    label="Hybrid search (BM25 + semantic)",
                    info="Uncheck for BM25-only (faster, exact-term matching)",
                )
                category_dropdown = gr.Dropdown(
                    choices=CATEGORY_CHOICES,
                    value="All categories",
                    label="Filter by category",
                )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=5):
            answer_output = gr.Markdown(
                label="Answer",
                value="*Your answer will appear here...*",
            )

    gr.Examples(
        examples=[[q, True, "All categories"] for q in EXAMPLE_QUESTIONS],
        inputs=[question_input, use_hybrid_toggle, category_dropdown],
        label="Example questions",
    )

    # Wire up the streaming predict function
    submit_btn.click(
        fn=predict_streaming,
        inputs=[question_input, use_hybrid_toggle, category_dropdown],
        outputs=answer_output,
    )
    # Also trigger on Enter in the textbox
    question_input.submit(
        fn=predict_streaming,
        inputs=[question_input, use_hybrid_toggle, category_dropdown],
        outputs=answer_output,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Gradio interface...")
    print(f"API backend: {API_BASE}")
    print("Open: http://localhost:7861")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True,
        share=False,   # set True to get a public gradio.live link
    )
