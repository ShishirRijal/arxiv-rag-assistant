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

from ..cache.key_builder import build_ask_cache_key
from ..cache.redis_service import RedisCacheService
from ..observability.langfuse_service import LangfuseService
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
    cached:      bool = False
    cache_key:   Optional[str] = None


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
        cache_svc:    Optional[RedisCacheService] = None,
        tracing_svc:  Optional[LangfuseService] = None,
        model_name:   str = "",
    ):
        self._search  = search_svc
        self._ollama  = ollama_svc
        self._context = context
        self._cache   = cache_svc
        self._tracing = tracing_svc
        self._model   = model_name

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
        cache_key = build_ask_cache_key(
            question=question,
            use_hybrid=use_hybrid,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            model=self._model,
            max_chunks=self._context._max_chunks,
        )

        with self._trace(
            name="rag.ask",
            input={
                "question": question,
                "use_hybrid": use_hybrid,
                "categories": categories,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
                "cache_key": cache_key,
            },
        ) as trace:
            cached = None
            with self._trace(name="cache.lookup", input={"cache_key": cache_key}) as span:
                if self._cache:
                    cached = self._cache.get_json(cache_key)
                span.update(output={"hit": cached is not None})

            if cached is not None:
                took_ms = int((time.perf_counter() - t0) * 1000)
                response = self._response_from_cache(cached, took_ms=took_ms, cache_key=cache_key)
                logger.info("RAG ask cache hit: query=%r took=%dms", question, took_ms)
                trace.update(output={
                    "cached": True,
                    "took_ms": response.took_ms,
                    "sources": response.sources,
                    "n_chunks": response.n_chunks,
                })
                self._flush_trace()
                return response

            # 1. Retrieve
            with self._trace(name="retrieval.hybrid_search", input={"query": question}) as span:
                search_result = self._search.search(
                    query      = question,
                    use_hybrid = use_hybrid,
                    categories = categories,
                    date_from  = date_from,
                    date_to    = date_to,
                    page_size  = self._context._max_chunks,
                )
                span.update(output={
                    "search_mode": search_result.search_mode,
                    "total": search_result.total,
                    "hits": len(search_result.hits),
                    "took_ms": search_result.took_ms,
                })

            hits = self._hits_to_dicts(search_result.hits)
            if not self._has_sufficient_evidence(question, hits):
                took_ms = int((time.perf_counter() - t0) * 1000)
                logger.info(
                    "RAG ask refused: query=%r mode=%s chunks=%d took=%dms",
                    question, search_result.search_mode, len(hits), took_ms,
                )
                response = RAGResponse(
                    question=question,
                    answer=_REFUSAL_ANSWER,
                    sources=[],
                    search_mode=search_result.search_mode,
                    n_chunks=0,
                    took_ms=took_ms,
                    cached=False,
                    cache_key=cache_key,
                )
                self._store_cache(cache_key, response)
                trace.update(output={
                    "cached": False,
                    "refused": True,
                    "took_ms": response.took_ms,
                })
                self._flush_trace()
                return response

            # 2. Build prompt
            with self._trace(name="context.build") as span:
                prompt, sources = self._context.build(question, hits)
                span.update(output={
                    "prompt_chars": len(prompt),
                    "sources": [self._source_to_dict(s) for s in sources],
                    "chunks": len(hits),
                })

            # 3. Generate
            with self._trace(
                name="ollama.generate",
                as_type="generation",
                input={"prompt_chars": len(prompt)},
                model=self._model,
            ) as span:
                answer = self._ollama.generate(prompt)
                span.update(output=answer)

            took_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "RAG ask: query=%r mode=%s chunks=%d took=%dms",
                question, search_result.search_mode, len(hits), took_ms,
            )

            response = RAGResponse(
                question    = question,
                answer      = answer,
                sources     = [self._source_to_dict(s) for s in sources],
                search_mode = search_result.search_mode,
                n_chunks    = len(hits),
                took_ms     = took_ms,
                cached      = False,
                cache_key   = cache_key,
            )

            self._store_cache(cache_key, response)
            trace.update(output={
                "cached": False,
                "took_ms": response.took_ms,
                "sources": response.sources,
                "n_chunks": response.n_chunks,
            })
            self._flush_trace()
            return response

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
        with self._trace(name="rag.ask_stream", input={"question": question, "use_hybrid": use_hybrid}):
            # Retrieve + build prompt (fast — happens before first token is yielded)
            with self._trace(name="retrieval.hybrid_search", input={"query": question}) as span:
                search_result = self._search.search(
                    query      = question,
                    use_hybrid = use_hybrid,
                    categories = categories,
                    date_from  = date_from,
                    date_to    = date_to,
                    page_size  = self._context._max_chunks,
                )
                span.update(output={
                    "search_mode": search_result.search_mode,
                    "total": search_result.total,
                    "hits": len(search_result.hits),
                    "took_ms": search_result.took_ms,
                })

            hits = self._hits_to_dicts(search_result.hits)
            if not self._has_sufficient_evidence(question, hits):
                yield ("token", _REFUSAL_ANSWER)
                yield ("sources", [])
                yield ("done", None)
                self._flush_trace()
                return

            with self._trace(name="context.build") as span:
                prompt, sources = self._context.build(question, hits)
                span.update(output={"prompt_chars": len(prompt), "chunks": len(hits)})

            generated_parts: list[str] = []
            with self._trace(
                name="ollama.stream",
                as_type="generation",
                input={"prompt_chars": len(prompt)},
                model=self._model,
            ) as span:
                for token in self._ollama.stream(prompt):
                    generated_parts.append(token)
                    yield ("token", token)
                span.update(output="".join(generated_parts))

            # Emit sources after all tokens
            yield ("sources", [self._source_to_dict(s) for s in sources])
            yield ("done", None)
            self._flush_trace()

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

    def _trace(self, **kwargs):
        if self._tracing:
            return self._tracing.observation(**kwargs)
        from contextlib import nullcontext

        class _Noop:
            def update(self, **kwargs) -> None:
                return None

        return nullcontext(_Noop())

    def _flush_trace(self) -> None:
        if self._tracing:
            self._tracing.flush()

    def _store_cache(self, cache_key: str, response: RAGResponse) -> bool:
        if not self._cache:
            return False
        with self._trace(name="cache.store", input={"cache_key": cache_key}) as span:
            stored = self._cache.set_json(cache_key, self._response_to_cache(response))
            span.update(output={"stored": stored, "ttl_seconds": self._cache.ttl_seconds})
            return stored

    @staticmethod
    def _response_to_cache(response: RAGResponse) -> dict:
        return {
            "question": response.question,
            "answer": response.answer,
            "sources": response.sources,
            "search_mode": response.search_mode,
            "n_chunks": response.n_chunks,
            "took_ms": response.took_ms,
        }

    @staticmethod
    def _response_from_cache(payload: dict, *, took_ms: int, cache_key: str) -> RAGResponse:
        return RAGResponse(
            question=payload.get("question", ""),
            answer=payload.get("answer", ""),
            sources=payload.get("sources", []),
            search_mode=payload.get("search_mode", "cache"),
            n_chunks=int(payload.get("n_chunks", 0)),
            took_ms=took_ms,
            cached=True,
            cache_key=cache_key,
        )
