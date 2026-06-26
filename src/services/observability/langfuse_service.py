"""Langfuse tracing wrapper with no-op fallback."""

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **kwargs) -> None:
        return None


class LangfuseService:
    """
    Small wrapper around the Langfuse SDK.

    The rest of the app should not need to know whether Langfuse is configured.
    Missing credentials or SDK errors degrade to no-op tracing.
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        enabled: bool = True,
    ):
        self._enabled = enabled and bool(public_key and secret_key)
        self._client = None
        self._host = host

        if not self._enabled:
            logger.info("Langfuse tracing disabled or missing credentials")
            return

        try:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
            if host:
                os.environ.setdefault("LANGFUSE_HOST", host)

            from langfuse import get_client

            self._client = get_client()
        except Exception as exc:
            self._enabled = False
            logger.warning("Langfuse client initialisation failed: %s", exc)

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Optional[dict[str, Any]] = None,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Iterator[Any]:
        """
        Start a Langfuse observation or yield a no-op object.

        Nested calls automatically become child observations through the
        Langfuse/OpenTelemetry context.
        """
        if not self.is_enabled:
            noop = _NoopObservation()
            if output is not None:
                noop.update(output=output)
            yield noop
            return

        kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
        if model:
            kwargs["model"] = model

        try:
            observation_context = self._client.start_as_current_observation(**kwargs)
        except Exception as exc:
            logger.warning("Langfuse observation start failed for %s: %s", name, exc)
            yield _NoopObservation()
            return

        with observation_context as obs:
            update: dict[str, Any] = {}
            if input is not None:
                update["input"] = input
            if metadata is not None:
                update["metadata"] = metadata
            if update:
                try:
                    obs.update(**update)
                except Exception as exc:
                    logger.warning("Langfuse observation update failed for %s: %s", name, exc)

            yield obs

            if output is not None:
                try:
                    obs.update(output=output)
                except Exception as exc:
                    logger.warning("Langfuse observation output update failed for %s: %s", name, exc)

    def flush(self) -> None:
        if not self.is_enabled:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)

    def check_health(self) -> dict:
        if not self._enabled:
            return {
                "status": "disabled",
                "reason": "missing_credentials_or_disabled",
            }
        if self._client is None:
            return {"status": "unhealthy", "error": "client_not_initialised"}

        try:
            ok = bool(self._client.auth_check())
            return {
                "status": "healthy" if ok else "unhealthy",
                "host": self._host or os.environ.get("LANGFUSE_HOST"),
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "error": str(exc),
                "host": self._host or os.environ.get("LANGFUSE_HOST"),
            }
