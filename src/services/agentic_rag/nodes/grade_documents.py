"""Document grading node for agentic RAG."""

import logging
from typing import Literal

from ...ollama.service import OllamaService
from ..config import AgenticRAGConfig
from ..models import DocumentGrade
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgenticRAGState
from .utils import add_reasoning_step, parse_structured_json

logger = logging.getLogger(__name__)


def run_grade_documents_node(
    state: AgenticRAGState,
    *,
    ollama: OllamaService,
) -> AgenticRAGState:
    """
    Grade whether retrieved chunks are useful enough for answer generation.

    The node writes:
    - document_grades
    - routing_decision
    - reasoning_steps
    """
    hits = state.get("retrieved_hits", [])
    if not hits:
        grade = DocumentGrade(is_relevant=False, score=0.0, reason="No chunks were retrieved.")
        state["document_grades"] = [grade]
        add_reasoning_step(
            state,
            step="grade_documents",
            message="No retrieved chunks were available to grade.",
            metadata={"route": "rewrite_query"},
        )
        return state

    question = state["question"]
    context = _format_context(hits)
    prompt = GRADE_DOCUMENTS_PROMPT.format(question=question, context=context)

    try:
        raw = ollama.generate(prompt, temperature=0.0, num_predict=180, json_mode=True)
        parsed = parse_structured_json(raw, DocumentGrade)
        grade = DocumentGrade(
            chunk_id="retrieved_context",
            arxiv_id=None,
            is_relevant=parsed.is_relevant,
            score=parsed.score,
            reason=parsed.reason,
        )
    except Exception as exc:
        logger.warning("Document grading LLM call failed; using conservative fallback: %s", exc)
        grade = DocumentGrade(
            chunk_id="retrieved_context",
            is_relevant=False,
            score=0.0,
            reason=f"LLM grading failed; relevance was not proven: {exc}",
        )

    state["document_grades"] = [grade]
    route = "generate_answer" if grade.is_relevant else "rewrite_query"
    add_reasoning_step(
        state,
        step="grade_documents",
        message=(
            "Retrieved chunks were graded as relevant."
            if grade.is_relevant
            else "Retrieved chunks were graded as not relevant enough."
        ),
        metadata={
            "score": grade.score,
            "reason": grade.reason,
            "route": route,
        },
    )
    return state


def route_after_grading(
    state: AgenticRAGState,
    *,
    config: AgenticRAGConfig,
) -> Literal["generate_answer", "rewrite_query", "insufficient_evidence"]:
    """Choose generation, rewrite, or refusal after document grading."""
    grades = state.get("document_grades", [])
    if grades and any(grade.is_relevant for grade in grades):
        return "generate_answer"

    if state.get("retrieval_attempts", 0) < config.max_retrieval_attempts:
        return "rewrite_query"

    return "insufficient_evidence"


def _format_context(hits: list[dict]) -> str:
    parts = []
    for index, hit in enumerate(hits, start=1):
        parts.append(
            f"[{index}] Title: {hit.get('title', 'Unknown')}\n"
            f"arXiv ID: {hit.get('arxiv_id', '')}\n"
            f"Section: {hit.get('section_name', '')}\n"
            f"Text: {hit.get('chunk_text', '')[:1200]}"
        )
    return "\n\n".join(parts)
