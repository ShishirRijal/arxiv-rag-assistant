"""Structured models used by the agentic RAG workflow."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Structured result from query scope validation."""

    score: int = Field(ge=0, le=100, description="Domain relevance score")
    reason: str = Field(description="Short explanation for the score")


class DocumentGrade(BaseModel):
    """Structured relevance grade for one retrieved chunk."""

    chunk_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    is_relevant: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class SourceItem(BaseModel):
    """Source metadata exposed by the agentic response."""

    index: int
    arxiv_id: str
    title: str
    url: str
    relevance_score: float = 0.0


class ReasoningStep(BaseModel):
    """Human-readable workflow event for debugging and transparency."""

    step: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Decision that chooses the next graph node."""

    route: Literal[
        "retrieve",
        "out_of_scope",
        "grade_documents",
        "generate_answer",
        "rewrite_query",
        "insufficient_evidence",
    ]
    reason: str = ""


class AgenticRAGResult(BaseModel):
    """Internal result returned by the agentic workflow service."""

    question: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    retrieval_attempts: int = 0
    rewritten_query: Optional[str] = None
    guardrail_score: Optional[int] = None
    search_mode: str = "hybrid"
    took_ms: int = 0
    trace_id: Optional[str] = None
