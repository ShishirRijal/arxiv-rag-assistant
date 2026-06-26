"""Factory for response cache services."""

from ...core.config import settings
from .redis_service import RedisCacheService


def make_cache_service() -> RedisCacheService:
    return RedisCacheService(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        ttl_seconds=settings.redis_ttl_hours * 60 * 60,
        enabled=settings.redis_enabled,
    )
