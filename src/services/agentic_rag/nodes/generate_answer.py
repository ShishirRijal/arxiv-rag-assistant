"""Answer generation node for agentic RAG."""

import logging

from ...ollama.service import OllamaService
from ...rag.context_builder import ContextBuilder, ContextSource
from ..config import AgenticRAGConfig
from ..models import SourceItem
from ..state import AgenticRAGState
from .utils import add_reasoning_step

logger = logging.getLogger(__name__)


def run_generate_answer_node(
    state: AgenticRAGState,
    *,
    config: AgenticRAGConfig,
    context_builder: ContextBuilder,
    ollama: OllamaService,
) -> AgenticRAGState:
    """
    Generate the final answer from graded-relevant retrieved chunks.

    The node writes:
    - answer
    - sources
    - reasoning_steps
    """
    hits = state.get("retrieved_hits", [])
    question = state["question"]

    prompt, context_sources = context_builder.build(question, hits)
    answer = ollama.generate(
        prompt,
        temperature=config.temperature,
    )

    state["answer"] = answer
    state["sources"] = [
        _source_to_item(source, hits)
        for source in context_sources
    ]
    add_reasoning_step(
        state,
        step="generate_answer",
        message=f"Generated answer using {len(hits)} retrieved chunks.",
        metadata={
            "sources": len(state["sources"]),
            "search_mode": state.get("search_mode", "hybrid"),
            "temperature": config.temperature,
        },
    )
    logger.info(
        "Agentic answer generated: chunks=%d sources=%d",
        len(hits),
        len(state["sources"]),
    )
    return state


def _source_to_item(source: ContextSource, hits: list[dict]) -> SourceItem:
    """Convert ContextSource to the agentic response source model."""
    relevance_score = 0.0
    for hit in hits:
        if hit.get("arxiv_id") == source.arxiv_id:
            relevance_score = max(relevance_score, float(hit.get("score") or 0.0))

    return SourceItem(
        index=source.index,
        arxiv_id=source.arxiv_id,
        title=source.title,
        url=source.url,
        relevance_score=relevance_score,
    )
