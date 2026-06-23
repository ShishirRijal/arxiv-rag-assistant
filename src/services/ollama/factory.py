"""
Factory for the Ollama service.

Usage:
    from arxiv_rag_curator.services.ollama.factory import make_ollama_service
    svc = make_ollama_service()
"""

from ...core.config import settings
from .service import OllamaService


def make_ollama_service() -> OllamaService:
    """Create an OllamaService from application settings."""
    return OllamaService(
        url   = settings.ollama_url,
        model = settings.ollama_model,
    )