"""Guardrail node for deciding whether a question belongs to the corpus domain."""

import logging
from typing import Literal

from ...ollama.service import OllamaService
from ..config import AgenticRAGConfig
from ..models import GuardrailResult
from ..prompts import GUARDRAIL_PROMPT
from ..state import AgenticRAGState
from .utils import add_reasoning_step, parse_structured_json

logger = logging.getLogger(__name__)


def run_guardrail_node(
    state: AgenticRAGState,
    *,
    config: AgenticRAGConfig,
    ollama: OllamaService,
) -> AgenticRAGState:
    """
    Score whether the user's question is in scope for research-paper RAG.

    The node writes:
    - guardrail_result
    - reasoning_steps
    """
    question = state["question"]
    prompt = GUARDRAIL_PROMPT.format(question=question)

    try:
        raw = ollama.generate(prompt, temperature=0.0, num_predict=160, json_mode=True)
        result = parse_structured_json(raw, GuardrailResult)
    except Exception as exc:
        logger.warning("Guardrail LLM call failed; using conservative fallback: %s", exc)
        result = GuardrailResult(
            score=40,
            reason=f"Guardrail validation failed, using conservative fallback: {exc}",
        )

    state["guardrail_result"] = result
    accepted = result.score >= config.guardrail_threshold
    add_reasoning_step(
        state,
        step="guardrail",
        message=(
            f"Accepted query with score {result.score}/100"
            if accepted
            else f"Rejected query with score {result.score}/100"
        ),
        metadata={
            "score": result.score,
            "threshold": config.guardrail_threshold,
            "reason": result.reason,
        },
    )
    return state


def route_after_guardrail(
    state: AgenticRAGState,
    *,
    config: AgenticRAGConfig,
) -> Literal["retrieve", "out_of_scope"]:
    """Route to retrieval only when the guardrail score passes threshold."""
    result = state.get("guardrail_result")
    if result is None:
        logger.warning("Missing guardrail_result; routing to out_of_scope")
        return "out_of_scope"

    return "retrieve" if result.score >= config.guardrail_threshold else "out_of_scope"
