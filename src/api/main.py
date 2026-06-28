"""
arXiv RAG Curator — FastAPI entry point.

Updated to:
  - Register the ask router (/api/v1/ask and /api/v1/ask/stream)
  - Pull the Ollama model on startup if not already cached
  - Surface LLM status in the health endpoint
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.database import check_health as db_health
from ..core.database import close_connection_pool, init_connection_pool, init_schema
from ..core.search import check_health as os_health
from ..core.search import init_index
from .routers.search import router as search_router
from .routers.hybrid_search import router as hybrid_router
from .routers.ask import router as ask_router
from .routers.agentic_ask import router as agentic_ask_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting arXiv RAG Curator API...")

    # Database + BM25 search index
    init_connection_pool()
    init_schema()
    init_index()

    # Chunk index + RRF pipeline
    try:
        from ..services.opensearch.factory import make_opensearch_client
        from ..services.embeddings.factory import make_embeddings_service
        from ..services.opensearch.chunk_indexer import ChunkIndexer
        from ..core.database import get_db
        ChunkIndexer(make_opensearch_client(), make_embeddings_service(), get_db).setup()
    except Exception as exc:
        logger.warning("Chunk indexer setup failed: %s", exc)

    # Ollama model — pull if not cached (non-blocking: runs async)
    try:
        from ..services.ollama.factory import make_ollama_service
        ollama = make_ollama_service()
        ollama.ensure_model()
    except Exception as exc:
        logger.warning("Ollama model ensure failed: %s", exc)

    logger.info("Startup complete. API ready.")
    yield

    close_connection_pool()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="arXiv RAG Curator API",
    description=(
        "Production RAG system for arXiv papers. "
        "Hybrid BM25 + semantic search, local LLM generation via Ollama, "
        "streaming responses, and Gradio interface."
    ),
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(search_router)   # /api/v1/search
app.include_router(hybrid_router)   # /api/v1/hybrid-search
app.include_router(ask_router)      # /api/v1/ask  +  /api/v1/ask/stream
app.include_router(agentic_ask_router)  # /api/v1/ask-agentic


@app.get("/")
async def root():
    return {
        "service":        "arXiv RAG Curator",
        "version":        "0.4.0",
        "docs":           "/docs",
        "endpoints": {
            "bm25_search":    "/api/v1/search",
            "hybrid_search":  "/api/v1/hybrid-search",
            "ask":            "/api/v1/ask",
            "ask_stream":     "/api/v1/ask/stream",
            "ask_agentic":     "/api/v1/ask-agentic",
            "health":         "/health",
        },
        "hybrid_enabled": bool(settings.jina_api_key),
    }


@app.get("/health")
async def health_check():
    postgres    = db_health()
    opensearch_ = os_health()

    # Ollama health
    try:
        from ..services.ollama.factory import make_ollama_service
        ollama = make_ollama_service().check_health()
    except Exception as exc:
        ollama = {"status": "unhealthy", "error": str(exc)}

    # Redis cache is non-critical: failures should make requests slower, not down.
    try:
        from ..services.cache.factory import make_cache_service
        cache = make_cache_service().check_health()
    except Exception as exc:
        cache = {"status": "unhealthy", "error": str(exc)}

    # Langfuse is non-critical: missing keys or network errors should not block RAG.
    try:
        from ..services.observability.factory import make_langfuse_service
        observability = make_langfuse_service().check_health()
    except Exception as exc:
        observability = {"status": "unhealthy", "error": str(exc)}

    embedding = {
        "status":  "enabled" if settings.jina_api_key else "disabled",
        "reason":  None if settings.jina_api_key else "JINA_API_KEY not set",
        "fallback": "bm25",
    }

    all_critical = all(
        s["status"] == "healthy"
        for s in [postgres, opensearch_]
    )

    return {
        "status":  "healthy" if all_critical else "degraded",
        "version": "0.4.0",
        "services": {
            "postgresql": postgres,
            "opensearch": opensearch_,
            "ollama":     ollama,
            "redis":      cache,
            "langfuse":   observability,
            "embeddings": embedding,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "arxiv_rag_curator.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
