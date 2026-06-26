"""Factory for observability services."""

from ...core.config import settings
from .langfuse_service import LangfuseService


def make_langfuse_service() -> LangfuseService:
    return LangfuseService(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.effective_langfuse_host,
        enabled=settings.langfuse_enabled,
    )
