"""
Context builder for the RAG pipeline.

Responsibility: transform raw search hits into a minimal, clean context
string that the LLM can efficiently process.

The 80% prompt reduction:
  Raw chunk object has: arxiv_id, chunk_id, chunk_text (with embedding header),
  section_name, chunk_index, title, abstract, authors, categories, published_at, score.

  After context building: arxiv_id + title + stripped chunk text.

  This is 80% fewer characters — faster generation, lower hallucination risk,
  more effective use of the context window.

The embedding header stripping:
  Every chunk was prefixed with:
    Title: {paper title}
    Abstract: {first 80 words}

    Section: {section_name}

  This header helped the embedding model understand context during indexing.
  The LLM doesn't need it — it has the title already and the repeated info
  just wastes tokens. We strip everything before and including the Section: line.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default context budget: 5 chunks × 800 chars ≈ 4000 chars of context
DEFAULT_MAX_CHUNKS          = 5
DEFAULT_MAX_CHARS_PER_CHUNK = 800

# System prompt — establishes the LLM's role and citation rules
SYSTEM_PROMPT = """\
You are an expert AI research assistant helping researchers understand academic papers.

Your task: Answer the user's question based ONLY on the provided context from research papers.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- Cite your sources using [1], [2], etc. corresponding to the numbered papers in the context.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and precise. Aim for 2–4 focused paragraphs.
- Do not reproduce large chunks of text verbatim — summarise and synthesise.
- Do not speculate beyond what the papers say.\
"""


@dataclass
class ContextSource:
    """Minimal representation of a source paper for the response sources list."""
    index:    int
    arxiv_id: str
    title:    str
    url:      str


class ContextBuilder:
    """
    Transforms raw search hits into an optimised LLM prompt.

    Two outputs per call:
    1. sources: list[ContextSource] — for including in the API response
    2. prompt:  str — the full prompt to send to Ollama
    """

    def __init__(
        self,
        max_chunks:          int = DEFAULT_MAX_CHUNKS,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    ):
        self._max_chunks = max_chunks
        self._max_chars  = max_chars_per_chunk

    def build(
        self,
        question: str,
        hits:     list[dict],    # dicts with at least arxiv_id, title, chunk_text
    ) -> tuple[str, list[ContextSource]]:
        """
        Build a full RAG prompt from a question and search hits.

        Returns:
            (prompt_string, sources_list)

        The prompt_string is ready to pass directly to OllamaService.generate().
        The sources_list is returned to the API caller for citation display.
        """
        selected = hits[:self._max_chunks]
        if not selected:
            # No context: prompt the LLM to say it couldn't find relevant papers
            no_context_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Context: No relevant papers were found for this query.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            return no_context_prompt, []

        context_parts: list[str] = []
        sources: list[ContextSource] = []
        seen_arxiv_ids: set[str] = set()

        for i, hit in enumerate(selected):
            raw_text = hit.get("chunk_text", "")
            text     = self._strip_embedding_header(raw_text)
            text     = self._truncate(text)

            context_parts.append(
                f"[{i+1}] Paper: \"{hit.get('title', 'Unknown')}\" "
                f"(arxiv:{hit.get('arxiv_id', '')})\n{text}"
            )
            arxiv_id = hit.get("arxiv_id", "")
            if arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.add(arxiv_id)
                sources.append(ContextSource(
                    index=len(sources) + 1,
                    arxiv_id=arxiv_id,
                    title=hit.get("title", "Unknown"),
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                ))

        context_str = "\n\n".join(context_parts)
        prompt      = self._assemble_prompt(question, context_str)

        logger.debug(
            "Built prompt: %d chunks, %d context chars, %d total chars",
            len(selected), len(context_str), len(prompt),
        )
        return prompt, sources

    # ── Private ───────────────────────────────────────────────────────────────

    def _strip_embedding_header(self, text: str) -> str:
        """
        Remove the 'Title: ... Abstract: ... Section: ...' header.

        The header was prepended during chunking to improve embedding quality.
        For LLM generation we don't need it — the title is already in the
        context part and the repeated abstract just wastes tokens.

        Splits on 'Section:' line — everything after it is the actual content.
        If no such header is found, returns the original text unchanged.
        """
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("Section:"):
                content_lines = lines[i + 1:]
                return "\n".join(content_lines).strip()
        return text.strip()

    def _truncate(self, text: str) -> str:
        """Truncate text to max_chars_per_chunk with an ellipsis marker."""
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars].rstrip() + "..."

    def _assemble_prompt(self, question: str, context_str: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context from retrieved papers:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
