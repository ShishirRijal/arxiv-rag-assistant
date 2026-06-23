"""
RAG pipeline orchestrator.

Wires together:
  1. HybridSearchService  — retrieves relevant chunks (BM25 + kNN or BM25 fallback)
  2. ContextBuilder       — strips noise, builds an optimised prompt
  3. OllamaService        — generates the answer (standard or streaming)

Two public methods:
  ask()        — waits for the complete answer, returns RAGResponse
  ask_stream() — yields (event_type, payload) tuples for SSE streaming

Design: the pipeline never raises user-visible errors for partial failures.
If retrieval returns no hits, it generates a graceful "no relevant papers found"
answer rather than a 500 error. If the LLM fails, it raises so the API can
return a proper 503.

Usage:
    from arxiv_rag_curator.services.rag.factory import make_rag_pipeline
    pipeline = make_rag_pipeline()

    # Complete response
    result = pipeline.ask("What is RLHF?")
    print(result.answer)

    # Streaming
    for event_type, payload in pipeline.ask_stream("What is RLHF?"):
        if event_type == "token":
            print(payload, end="", flush=True)
        elif event_type == "sources":
            print("\\nSources:", payload)
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Generator, Optional

from ..ollama.service import OllamaService
from ..opensearch.hybrid_service import HybridSearchService
from .context_builder import ContextBuilder, ContextSource

logger = logging.getLogger(__name__)

_REFUSAL_ANSWER = (
    "I couldn't find relevant evidence in the indexed papers to answer that "
    "question. Try asking about the papers currently in the corpus, or ingest "
    "more relevant papers first."
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "what", "when", "where", "which", "why", "with",
    "you", "your", "explain", "describe", "tell", "about", "system",
}


# ── Response schemas ──────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Complete response from a non-streaming RAG query."""
    question:    str
    answer:      str
    sources:     list[dict]    # serialisable dicts for API response
    search_mode: str           # 'hybrid' | 'bm25'
    n_chunks:    int
    took_ms:     int


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline: retrieve → build context → generate.

    Stateless after construction — safe to share across requests.
    """

    def __init__(
        self,
        search_svc:   HybridSearchService,
        ollama_svc:   OllamaService,
        context:      ContextBuilder,
    ):
        self._search  = search_svc
        self._ollama  = ollama_svc
        self._context = context

    # ── Standard (non-streaming) ──────────────────────────────────────────────

    def ask(
        self,
        question:    str,
        use_hybrid:  bool = True,
        categories:  Optional[list[str]] = None,
        date_from:   Optional[date] = None,
        date_to:     Optional[date] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline, waiting for the complete answer.

        Steps:
        1. Search for relevant chunks (hybrid or BM25)
        2. Build an optimised prompt from the top chunks
        3. Generate an answer with Ollama (non-streaming)
        4. Return answer + sources in a structured response
        """
        t0 = time.perf_counter()

        # 1. Retrieve
        search_result = self._search.search(
            query      = question,
            use_hybrid = use_hybrid,
            categories = categories,
            date_from  = date_from,
            date_to    = date_to,
            page_size  = self._context._max_chunks,
        )

        hits = self._hits_to_dicts(search_result.hits)
        if not self._has_sufficient_evidence(question, hits):
            took_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "RAG ask refused: query=%r mode=%s chunks=%d took=%dms",
                question, search_result.search_mode, len(hits), took_ms,
            )
            return RAGResponse(
                question=question,
                answer=_REFUSAL_ANSWER,
                sources=[],
                search_mode=search_result.search_mode,
                n_chunks=0,
                took_ms=took_ms,
            )

        # 2. Build prompt
        prompt, sources = self._context.build(question, hits)

        # 3. Generate
        answer = self._ollama.generate(prompt)

        took_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "RAG ask: query=%r mode=%s chunks=%d took=%dms",
            question, search_result.search_mode, len(hits), took_ms,
        )

        return RAGResponse(
            question    = question,
            answer      = answer,
            sources     = [self._source_to_dict(s) for s in sources],
            search_mode = search_result.search_mode,
            n_chunks    = len(hits),
            took_ms     = took_ms,
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def ask_stream(
        self,
        question:    str,
        use_hybrid:  bool = True,
        categories:  Optional[list[str]] = None,
        date_from:   Optional[date] = None,
        date_to:     Optional[date] = None,
    ) -> Generator[tuple[str, object], None, None]:
        """
        Streaming RAG pipeline — yields events for SSE delivery.

        Yields tuples:
          ("token",   str)       — one LLM token
          ("sources", list[dict]) — after all tokens, the source list
          ("done",    None)       — terminal signal

        The FastAPI route converts these into SSE events:
          data: {"type": "token",   "content": "The"}
          data: {"type": "sources", "content": [...]}
          data: [DONE]
        """
        # Retrieve + build prompt (fast — happens before first token is yielded)
        search_result = self._search.search(
            query      = question,
            use_hybrid = use_hybrid,
            categories = categories,
            date_from  = date_from,
            date_to    = date_to,
            page_size  = self._context._max_chunks,
        )

        hits            = self._hits_to_dicts(search_result.hits)
        if not self._has_sufficient_evidence(question, hits):
            yield ("token", _REFUSAL_ANSWER)
            yield ("sources", [])
            yield ("done", None)
            return

        prompt, sources = self._context.build(question, hits)

        # Stream tokens from Ollama
        for token in self._ollama.stream(prompt):
            yield ("token", token)

        # Emit sources after all tokens
        yield ("sources", [self._source_to_dict(s) for s in sources])
        yield ("done", None)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hits_to_dicts(hits) -> list[dict]:
        """Convert HybridSearchService hits to plain dicts."""
        return [
            {
                "arxiv_id":   h.arxiv_id,
                "title":      h.title,
                "abstract":   h.abstract,
                "chunk_text": h.chunk_text,
            }
            for h in hits
        ]

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in _STOPWORDS
        }

    @classmethod
    def _has_sufficient_evidence(cls, question: str, hits: list[dict]) -> bool:
        """
        Refuse clearly out-of-domain questions before calling the LLM.

        With a tiny corpus, OpenSearch will always return "best available"
        chunks, even when they are not truly relevant. We require meaningful
        lexical overlap between the user's question and the retrieved text.
        """
        if not hits:
            return False

        query_terms = cls._tokenise(question)
        if not query_terms:
            return False

        max_overlap = 0
        max_metadata_overlap = 0
        supporting_hits = 0
        metadata_supporting_hits = 0
        for hit in hits:
            doc_text = " ".join([
                hit.get("title", ""),
                hit.get("chunk_text", "")[:1200],
            ])
            metadata_text = " ".join([
                hit.get("title", ""),
                hit.get("abstract", "")[:800],
            ])
            overlap = len(query_terms & cls._tokenise(doc_text))
            metadata_overlap = len(query_terms & cls._tokenise(metadata_text))
            max_overlap = max(max_overlap, overlap)
            max_metadata_overlap = max(max_metadata_overlap, metadata_overlap)
            if overlap > 0:
                supporting_hits += 1
            if metadata_overlap > 0:
                metadata_supporting_hits += 1

        if len(query_terms) == 1:
            return metadata_supporting_hits >= 1

        return max_metadata_overlap >= 1 and max_overlap >= 2

    @staticmethod
    def _source_to_dict(s: ContextSource) -> dict:
        return {
            "index":    s.index,
            "arxiv_id": s.arxiv_id,
            "title":    s.title,
            "url":      s.url,
        }
