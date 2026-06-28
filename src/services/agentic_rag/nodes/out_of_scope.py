"""Out-of-scope response node."""

from ..state import AgenticRAGState
from .utils import add_reasoning_step


def run_out_of_scope_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Produce a helpful refusal for questions outside the indexed corpus domain.

    This is different from an error. The system is working correctly when it
    refuses to answer questions that cannot be grounded in its paper corpus.
    """
    question = state["question"]
    state["answer"] = (
        "I can only answer questions grounded in the indexed arXiv papers about "
        "computer science, AI, machine learning, NLP, retrieval, and related "
        f"research topics. Your question was: {question!r}. "
        "Try asking about a model, paper, method, architecture, benchmark, or "
        "research concept represented in the corpus."
    )
    add_reasoning_step(
        state,
        step="out_of_scope",
        message="Stopped before retrieval because the question is outside the corpus domain.",
    )
    return state
