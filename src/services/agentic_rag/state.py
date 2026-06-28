"""Shared state passed between agentic RAG graph nodes."""

from typing import Any, Optional, TypedDict

from .models import DocumentGrade, GuardrailResult, ReasoningStep, RoutingDecision, SourceItem


class AgenticRAGState(TypedDict, total=False):
    """
    Mutable workflow state for graph execution.

    LangGraph nodes will receive this dictionary, add or update fields, and
    route to the next node based on values such as guardrail score, document
    grades, and retrieval attempts.
    """

    question: str
    current_query: str
    rewritten_query: Optional[str]
    answer: Optional[str]

    retrieval_attempts: int
    guardrail_result: Optional[GuardrailResult]
    routing_decision: Optional[RoutingDecision]
    document_grades: list[DocumentGrade]

    retrieved_hits: list[dict[str, Any]]
    search_mode: str
    sources: list[SourceItem]
    reasoning_steps: list[ReasoningStep]
    metadata: dict[str, Any]
    trace_id: Optional[str]


def initial_state(question: str, *, metadata: Optional[dict[str, Any]] = None) -> AgenticRAGState:
    """Create the initial state for one agentic RAG request."""
    cleaned_question = question.strip()
    return {
        "question": cleaned_question,
        "current_query": cleaned_question,
        "rewritten_query": None,
        "answer": None,
        "retrieval_attempts": 0,
        "guardrail_result": None,
        "routing_decision": None,
        "document_grades": [],
        "retrieved_hits": [],
        "search_mode": "hybrid",
        "sources": [],
        "reasoning_steps": [],
        "metadata": metadata or {},
        "trace_id": None,
    }
