"""Agentic RAG question-answering API."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from ...services.agentic_rag.factory import make_agentic_rag_service
from ...services.agentic_rag.service import AgenticRAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ask-agentic", tags=["agentic-rag"])


class AgenticAskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["What is the transformer architecture?"],
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        return value.strip()


class AgenticSourceItem(BaseModel):
    index: int
    arxiv_id: str
    title: str
    url: str
    relevance_score: float = 0.0


class AgenticReasoningStep(BaseModel):
    step: str
    message: str
    metadata: dict = Field(default_factory=dict)


class AgenticAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[AgenticSourceItem]
    reasoning_steps: list[AgenticReasoningStep]
    retrieval_attempts: int
    rewritten_query: Optional[str] = None
    guardrail_score: Optional[int] = None
    search_mode: str
    took_ms: int
    trace_id: Optional[str] = None


def get_agentic_rag_service() -> AgenticRAGService:
    """Provide a fully wired agentic RAG service."""
    return make_agentic_rag_service()


@router.post(
    "/",
    response_model=AgenticAskResponse,
    summary="Ask a question with graph-controlled RAG",
    description=(
        "Runs an agentic RAG workflow with query guardrails, retrieval, "
        "document grading, optional query rewriting, and grounded answer generation."
    ),
)
async def ask_agentic(
    request: AgenticAskRequest,
    service: AgenticRAGService = Depends(get_agentic_rag_service),
) -> AgenticAskResponse:
    """Run the agentic RAG workflow for one question."""
    logger.info("Agentic ask: question=%r", request.question)

    try:
        result = service.ask(request.question)
        return AgenticAskResponse(
            question=result.question,
            answer=result.answer,
            sources=[AgenticSourceItem(**source.model_dump()) for source in result.sources],
            reasoning_steps=[
                AgenticReasoningStep(**step.model_dump())
                for step in result.reasoning_steps
            ],
            retrieval_attempts=result.retrieval_attempts,
            rewritten_query=result.rewritten_query,
            guardrail_score=result.guardrail_score,
            search_mode=result.search_mode,
            took_ms=result.took_ms,
            trace_id=result.trace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Agentic RAG LLM failure: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}")
    except Exception as exc:
        logger.error("Agentic ask failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
