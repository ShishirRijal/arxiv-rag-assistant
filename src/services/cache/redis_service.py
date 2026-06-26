"""Redis-backed JSON response cache with graceful degradation."""

import json
import logging
from typing import Any, Optional

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisCacheService:
    """
    Thin Redis wrapper for JSON values.

    The service never raises for cache misses or Redis outages on request paths.
    Cache failures should make the app slower, not unavailable.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        db: int,
        ttl_seconds: int,
        enabled: bool = True,
    ):
        self._ttl_seconds = ttl_seconds
        self._enabled = enabled
        self._client: Optional[Redis] = None

        if not enabled:
            logger.info("Redis cache disabled by configuration")
            return

        self._client = Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def get_json(self, key: str) -> Optional[dict[str, Any]]:
        """Return a cached JSON object or None on miss/error."""
        if not self.is_enabled:
            return None

        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except (RedisError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Redis cache get failed for %s: %s", key, exc)
            return None

    def set_json(self, key: str, value: dict[str, Any]) -> bool:
        """Store a JSON object with the configured TTL."""
        if not self.is_enabled:
            return False

        try:
            raw = json.dumps(value, sort_keys=True)
            return bool(self._client.setex(key, self._ttl_seconds, raw))
        except (RedisError, TypeError) as exc:
            logger.warning("Redis cache set failed for %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        if not self.is_enabled:
            return False

        try:
            return bool(self._client.delete(key))
        except RedisError as exc:
            logger.warning("Redis cache delete failed for %s: %s", key, exc)
            return False

    def check_health(self) -> dict:
        """Return Redis cache health for API health output."""
        if not self._enabled:
            return {
                "status": "disabled",
                "ttl_seconds": self._ttl_seconds,
            }
        if self._client is None:
            return {
                "status": "unhealthy",
                "error": "client_not_initialised",
                "ttl_seconds": self._ttl_seconds,
            }

        try:
            self._client.ping()
            return {
                "status": "healthy",
                "ttl_seconds": self._ttl_seconds,
            }
        except RedisError as exc:
            return {
                "status": "unhealthy",
                "error": str(exc),
                "ttl_seconds": self._ttl_seconds,
            }
