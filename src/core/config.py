"""
Central configuration — updated to include JINA_API_KEY for embeddings.

New field:
  jina_api_key: Optional — when absent, embedding service is disabled
                and hybrid search falls back to BM25. The application
                still starts and works; hybrid is just an enhancement.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    postgres_host:     str = Field(default="localhost")
    postgres_port:     int = Field(default=5432)
    postgres_db:       str = Field(default="ragdb")
    postgres_user:     str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")

    # ── OpenSearch ──────────────────────────────────────────────────────────
    opensearch_host: str = Field(default="localhost")
    opensearch_port: int = Field(default=9200)

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_host:  str = Field(default="localhost")
    ollama_port:  int = Field(default=11434)
    ollama_model: str = Field(default="llama3.2:1b")

    # ── Jina AI Embeddings ────────────────────────────────────────────────
    # Get a free key (no signup): https://jina.ai/embeddings/
    # When absent: hybrid search silently falls back to BM25.
    jina_api_key: str = Field(default="")

    # ── Redis Response Cache ──────────────────────────────────────────────
    redis_host:      str  = Field(default="localhost")
    redis_port:      int  = Field(default=6379)
    redis_db:        int  = Field(default=0)
    redis_ttl_hours: int  = Field(default=24)
    redis_enabled:   bool = Field(default=True)

    # ── Langfuse Observability ────────────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host:       str = Field(default="")
    langfuse_base_url:   str = Field(default="")
    langfuse_enabled:    bool = Field(default=True)

    # ── Agentic RAG ───────────────────────────────────────────────────────
    agentic_guardrail_threshold:    int   = Field(default=60)
    agentic_max_retrieval_attempts: int   = Field(default=2)
    agentic_top_k:                  int   = Field(default=5)
    agentic_temperature:            float = Field(default=0.0)

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str  = Field(default="0.0.0.0")
    api_port: int  = Field(default=8000)
    debug:    bool = Field(default=False)

    # ── Computed properties ─────────────────────────────────────────────────
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def opensearch_url(self) -> str:
        return f"http://{self.opensearch_host}:{self.opensearch_port}"

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def effective_langfuse_host(self) -> str:
        return self.langfuse_host or self.langfuse_base_url

    class Config:
        env_file         = ".env"
        env_file_encoding = "utf-8"
        extra            = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
