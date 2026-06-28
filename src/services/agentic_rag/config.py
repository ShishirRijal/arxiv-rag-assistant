"""Runtime configuration for the agentic RAG workflow."""

from typing import Any

from pydantic import BaseModel, Field

from ...core.config import settings


class AgenticRAGConfig(BaseModel):
    """
    Execution controls for graph-based RAG.

    This config is intentionally separate from global settings so each request
    can eventually override safe runtime parameters like top-k or retry count.
    """

    max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    guardrail_threshold: int = Field(default=60, ge=0, le=100)
    model: str = Field(default="llama3.2:1b")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = True
    enable_tracing: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


def default_agentic_rag_config() -> AgenticRAGConfig:
    """Build the default workflow config from application settings."""
    return AgenticRAGConfig(
        max_retrieval_attempts=settings.agentic_max_retrieval_attempts,
        guardrail_threshold=settings.agentic_guardrail_threshold,
        model=settings.ollama_model,
        temperature=settings.agentic_temperature,
        top_k=settings.agentic_top_k,
        use_hybrid=True,
        enable_tracing=settings.langfuse_enabled,
    )
