"""HTTP client used by the Telegram bot to call the RAG API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class RAGClientError(RuntimeError):
    """Raised when the bot cannot get a usable answer from the RAG API."""


@dataclass(frozen=True)
class RAGClientConfig:
    """Runtime configuration for calling the local FastAPI service."""

    base_url: str
    timeout_seconds: int = 60
    use_agentic_rag: bool = True

    @property
    def ask_path(self) -> str:
        return "/api/v1/ask-agentic/" if self.use_agentic_rag else "/api/v1/ask/"


class RAGClient:
    """Small API client that keeps Telegram handlers independent of RAG internals."""

    def __init__(self, config: RAGClientConfig) -> None:
        self._config = config

    async def ask(self, question: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        if not self._config.use_agentic_rag:
            payload["use_hybrid"] = True

        try:
            async with httpx.AsyncClient(
                base_url=self._config.base_url.rstrip("/"),
                timeout=self._config.timeout_seconds,
            ) as client:
                response = await client.post(self._config.ask_path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise RAGClientError(f"RAG API returned {exc.response.status_code}: {detail}") from exc
        except httpx.TimeoutException as exc:
            raise RAGClientError("RAG API timed out before returning an answer.") from exc
        except httpx.HTTPError as exc:
            raise RAGClientError(f"Could not reach RAG API: {exc}") from exc

    async def health(self) -> dict[str, Any]:
        """Return the FastAPI health response."""
        try:
            async with httpx.AsyncClient(
                base_url=self._config.base_url.rstrip("/"),
                timeout=min(self._config.timeout_seconds, 10),
            ) as client:
                response = await client.get("/health")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise RAGClientError(f"Health check returned {exc.response.status_code}: {detail}") from exc
        except httpx.TimeoutException as exc:
            raise RAGClientError("Health check timed out.") from exc
        except httpx.HTTPError as exc:
            raise RAGClientError(f"Could not reach RAG API health endpoint: {exc}") from exc

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:300] or "empty response"
        if isinstance(body, dict):
            return str(body.get("detail") or body)
        return str(body)
