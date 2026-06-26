"""
Ask router — the RAG question-answering API.

Two endpoints:
  POST /api/v1/ask          — complete response (waits for full answer)
  GET  /api/v1/ask/stream   — SSE streaming (tokens arrive in real time)

Why GET for streaming?
  The browser's native EventSource API only supports GET requests.
  Query parameters carry the question and options.
  POST-based SSE requires fetch() with ReadableStream — more complex client code.
  We use GET for the streaming endpoint to keep the browser client simple.

SSE event format:
  data: {"type": "token",   "content": "The transformer..."}

  data: {"type": "sources", "content": [{"index": 1, "arxiv_id": "...", ...}]}

  data: [DONE]

Two blank lines after each event is required by the SSE specification.
The EventSource client fires an 'message' event for each one.
"""

import json
import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ...services.rag.factory import make_rag_pipeline
from ...services.rag.pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ask", tags=["ask"])


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["What is the main contribution of the attention mechanism paper?"],
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid BM25 + semantic search. Falls back to BM25 if embedding unavailable.",
    )
    categories: Optional[list[str]] = Field(
        default=None,
        description="Restrict retrieval to these arXiv categories e.g. ['cs.AI']",
    )
    date_from: Optional[date] = Field(default=None)
    date_to:   Optional[date] = Field(default=None)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class SourceItem(BaseModel):
    index:    int
    arxiv_id: str
    title:    str
    url:      str


class AskResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list[SourceItem]
    search_mode: str
    n_chunks:    int
    took_ms:     int
    cached:      bool = False
    cache_key:   Optional[str] = None


# ── Dependency ────────────────────────────────────────────────────────────────

def get_rag_pipeline() -> RAGPipeline:
    """Provide a RAGPipeline with all dependencies wired."""
    return make_rag_pipeline()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=AskResponse,
    summary="Ask a question (complete response)",
    description=(
        "Submit a question and receive a complete answer grounded in retrieved arXiv papers. "
        "The response includes the answer text and a list of source papers with arXiv links. "
        "For real-time token streaming, use the /stream endpoint instead."
    ),
)
async def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> AskResponse:
    """
    Non-streaming RAG endpoint.

    Retrieves relevant chunks, builds an optimised prompt, generates
    a complete answer with Ollama, and returns everything as JSON.
    Typical latency: 15–40s depending on model and CPU.
    """
    logger.info("Ask: question=%r use_hybrid=%s", request.question, request.use_hybrid)

    try:
        result: RAGResponse = pipeline.ask(
            question   = request.question,
            use_hybrid = request.use_hybrid,
            categories = request.categories,
            date_from  = request.date_from,
            date_to    = request.date_to,
        )

        logger.info(
            "Ask complete: chunks=%d mode=%s cached=%s took=%dms",
            result.n_chunks, result.search_mode, result.cached, result.took_ms,
        )

        return AskResponse(
            question    = result.question,
            answer      = result.answer,
            sources     = [SourceItem(**s) for s in result.sources],
            search_mode = result.search_mode,
            n_chunks    = result.n_chunks,
            took_ms     = result.took_ms,
            cached      = result.cached,
            cache_key   = result.cache_key,
        )

    except RuntimeError as exc:
        # LLM generation failure — likely Ollama is down
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}")
    except Exception as exc:
        logger.error("Ask failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/stream",
    summary="Ask a question (SSE streaming)",
    description=(
        "Submit a question and receive the answer token-by-token via Server-Sent Events. "
        "Significantly better UX than the standard endpoint — first tokens arrive within seconds. "
        "Use the browser EventSource API or any SSE-compatible client. "
        "The final event before [DONE] includes the sources list."
    ),
    response_class=StreamingResponse,
)
async def ask_stream(
    question:    str    = Query(..., min_length=1, max_length=1000,
                                description="The question to ask"),
    use_hybrid:  bool   = Query(default=True,
                                description="Enable hybrid search"),
    categories:  Optional[str]  = Query(default=None,
                                        description="Comma-separated arXiv categories e.g. cs.AI,cs.LG"),
    pipeline:    RAGPipeline = Depends(get_rag_pipeline),
) -> StreamingResponse:
    """
    Streaming RAG endpoint using Server-Sent Events.

    JavaScript client example:
        const source = new EventSource(
            `/api/v1/ask/stream?question=${encodeURIComponent(q)}`
        );
        source.onmessage = (e) => {
            if (e.data === "[DONE]") { source.close(); return; }
            const event = JSON.parse(e.data);
            if (event.type === "token")   appendToken(event.content);
            if (event.type === "sources") displaySources(event.content);
        };
    """
    logger.info("Ask stream: question=%r use_hybrid=%s", question, use_hybrid)

    # Parse comma-separated categories from query param
    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else None

    async def event_generator():
        """Async generator that yields SSE-formatted events."""
        try:
            for event_type, payload in pipeline.ask_stream(
                question   = question.strip(),
                use_hybrid = use_hybrid,
                categories = cats,
            ):
                if event_type == "token":
                    data = json.dumps({"type": "token", "content": payload})
                    yield f"data: {data}\n\n"

                elif event_type == "sources":
                    data = json.dumps({"type": "sources", "content": payload})
                    yield f"data: {data}\n\n"

                elif event_type == "done":
                    yield "data: [DONE]\n\n"
                    break

        except RuntimeError as exc:
            error_data = json.dumps({"type": "error", "content": str(exc)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming ask failed: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Internal error'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx proxy buffering
        },
    )
