"""Retrieval node for agentic RAG."""

import logging

from ...opensearch.hybrid_service import ChunkHit, HybridSearchService
from ..config import AgenticRAGConfig
from ..state import AgenticRAGState
from .utils import add_reasoning_step

logger = logging.getLogger(__name__)


def run_retrieve_node(
    state: AgenticRAGState,
    *,
    config: AgenticRAGConfig,
    search: HybridSearchService,
) -> AgenticRAGState:
    """
    Retrieve candidate chunks for the current query.

    The node writes:
    - retrieval_attempts
    - retrieved_hits
    - search_mode
    - reasoning_steps
    """
    attempts = state.get("retrieval_attempts", 0)
    if attempts >= config.max_retrieval_attempts:
        state["retrieved_hits"] = []
        add_reasoning_step(
            state,
            step="retrieve",
            message=(
                "Skipped retrieval because the maximum retrieval attempts "
                f"({config.max_retrieval_attempts}) were already used."
            ),
            metadata={"attempts": attempts},
        )
        return state

    query = state.get("current_query") or state["question"]
    attempt_number = attempts + 1
    state["retrieval_attempts"] = attempt_number

    result = search.search(
        query=query,
        use_hybrid=config.use_hybrid,
        page_size=config.top_k,
    )
    hits = [_hit_to_dict(hit) for hit in result.hits]

    state["retrieved_hits"] = hits
    state["search_mode"] = result.search_mode
    add_reasoning_step(
        state,
        step="retrieve",
        message=(
            f"Retrieved {len(hits)} chunks using {result.search_mode} search "
            f"for query: {query!r}."
        ),
        metadata={
            "attempt": attempt_number,
            "max_attempts": config.max_retrieval_attempts,
            "total": result.total,
            "took_ms": result.took_ms,
            "search_mode": result.search_mode,
        },
    )
    logger.info(
        "Agentic retrieve: query=%r mode=%s hits=%d attempt=%d/%d",
        query,
        result.search_mode,
        len(hits),
        attempt_number,
        config.max_retrieval_attempts,
    )
    return state


def _hit_to_dict(hit: ChunkHit) -> dict:
    """Convert a ChunkHit dataclass to the dict shape used by context builders."""
    return {
        "arxiv_id": hit.arxiv_id,
        "chunk_id": hit.chunk_id,
        "chunk_text": hit.chunk_text,
        "section_name": hit.section_name,
        "chunk_index": hit.chunk_index,
        "title": hit.title,
        "abstract": hit.abstract,
        "authors": hit.authors,
        "categories": hit.categories,
        "published_at": hit.published_at,
        "score": hit.score,
    }
