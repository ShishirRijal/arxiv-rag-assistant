"""Query rewriting node for agentic RAG."""

import logging

from ...ollama.service import OllamaService
from ..config import AgenticRAGConfig
from ..models import QueryRewriteResult
from ..prompts import REWRITE_QUERY_PROMPT
from ..state import AgenticRAGState
from .utils import add_reasoning_step, parse_structured_json

logger = logging.getLogger(__name__)


def run_rewrite_query_node(
    state: AgenticRAGState,
    *,
    config: AgenticRAGConfig,
    ollama: OllamaService,
) -> AgenticRAGState:
    """
    Rewrite the current query to improve the next retrieval attempt.

    The node writes:
    - current_query
    - rewritten_query
    - reasoning_steps
    """
    original_question = state["question"]
    current_query = state.get("current_query") or original_question
    prompt = REWRITE_QUERY_PROMPT.format(
        question=original_question,
        current_query=current_query,
    )

    try:
        raw = ollama.generate(prompt, temperature=0.2, num_predict=160, json_mode=True)
        result = parse_structured_json(raw, QueryRewriteResult)
        rewritten = result.rewritten_query.strip()
        reason = result.reason
        if not rewritten:
            raise ValueError("Rewritten query was empty")
    except Exception as exc:
        logger.warning("Query rewrite LLM call failed; using fallback rewrite: %s", exc)
        rewritten = f"{original_question} arxiv research paper machine learning neural network"
        reason = f"Fallback rewrite because LLM rewriting failed: {exc}"

    state["current_query"] = rewritten
    state["rewritten_query"] = rewritten
    add_reasoning_step(
        state,
        step="rewrite_query",
        message=f"Rewrote query for another retrieval attempt: {rewritten!r}.",
        metadata={
            "original_question": original_question,
            "previous_query": current_query,
            "rewritten_query": rewritten,
            "reason": reason,
            "attempts_so_far": state.get("retrieval_attempts", 0),
            "max_attempts": config.max_retrieval_attempts,
        },
    )
    return state
