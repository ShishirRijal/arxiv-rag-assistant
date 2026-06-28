"""Factory for the agentic RAG service."""

from ..embeddings.factory import make_embeddings_service
from ..ollama.factory import make_ollama_service
from ..opensearch.factory import make_opensearch_client
from ..opensearch.hybrid_service import HybridSearchService
from ..rag.context_builder import ContextBuilder
from .config import default_agentic_rag_config
from .service import AgenticRAGService


def make_agentic_rag_service() -> AgenticRAGService:
    """
    Create a fully wired AgenticRAGService.

    This mirrors the normal RAG factory while keeping the graph-controlled path
    separate enough to compare both implementations side by side.
    """
    config = default_agentic_rag_config()
    search_svc = HybridSearchService(
        os_client=make_opensearch_client(),
        embeddings_svc=make_embeddings_service(),
    )
    context_builder = ContextBuilder(
        max_chunks=config.top_k,
        max_chars_per_chunk=800,
    )
    return AgenticRAGService(
        search_svc=search_svc,
        ollama_svc=make_ollama_service(),
        context_builder=context_builder,
        config=config,
    )
