"""Terminal node for exhausted retrieval attempts without relevant evidence."""

from ..state import AgenticRAGState
from .utils import add_reasoning_step


def run_insufficient_evidence_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Return a grounded refusal when retrieval/rewrite attempts did not find evidence.

    This is the correct terminal path when the query is in-domain but the local
    corpus does not contain enough relevant information.
    """
    state["answer"] = (
        "I could not find enough relevant evidence in the indexed arXiv papers "
        "to answer this question reliably. Try rephrasing the question with more "
        "specific paper titles, model names, methods, or technical terms, or "
        "ingest more relevant papers first."
    )
    add_reasoning_step(
        state,
        step="insufficient_evidence",
        message="Stopped because retrieval attempts were exhausted without relevant evidence.",
        metadata={
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "rewritten_query": state.get("rewritten_query"),
        },
    )
    return state
