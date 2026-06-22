"""
Factory for the RAG pipeline.

Wires together HybridSearchService + OllamaService + ContextBuilder
into a ready-to-use RAGPipeline.

Usage:
    from arxiv_rag_curator.services.rag.factory import make_rag_pipeline
    pipeline = make_rag_pipeline()
    result   = pipeline.ask("What is RLHF?")
"""

from ...core.config import settings
from ..embeddings.factory import make_embeddings_service
from ..ollama.factory import make_ollama_service
from ..opensearch.factory import make_opensearch_client
from ..opensearch.hybrid_service import HybridSearchService
from .context_builder import ContextBuilder
from .pipeline import RAGPipeline


def make_rag_pipeline(
    max_chunks:          int = 5,
    max_chars_per_chunk: int = 800,
) -> RAGPipeline:
    """
    Create a fully wired RAGPipeline.

    Args:
        max_chunks:          how many retrieved chunks to include in context
        max_chars_per_chunk: character budget per chunk after stripping header
    """
    search_svc = HybridSearchService(
        os_client      = make_opensearch_client(),
        embeddings_svc = make_embeddings_service(),
    )
    ollama_svc = make_ollama_service()
    context    = ContextBuilder(
        max_chunks          = max_chunks,
        max_chars_per_chunk = max_chars_per_chunk,
    )
    return RAGPipeline(
        search_svc = search_svc,
        ollama_svc = ollama_svc,
        context    = context,
    )