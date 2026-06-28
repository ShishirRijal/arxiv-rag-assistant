"""
Ollama service — local LLM inference.

Ollama runs LLMs locally via a REST API. No data leaves your machine,
no API costs, no rate limits. Perfect for a privacy-sensitive research tool.

Key endpoints we use:
  GET  /api/tags       — list downloaded models
  POST /api/pull       — download a model
  POST /api/generate   — single-turn text generation (standard + streaming)
  GET  /api/           — health check (returns 200 if running)

Design decisions:
  - Non-streaming (ask): waits for the complete response, returns a string.
    Used by the standard /ask API endpoint.
  - Streaming (ask_stream): yields tokens as a Python generator.
    Used by the SSE /ask/stream endpoint.
  - Auto-pull: if the configured model isn't downloaded, pull it on first use.
    Avoids a hard startup failure when the model isn't cached.
  - Low temperature (0.1): keeps answers factual and grounded in context.
  - num_predict cap (512): prevents runaway generation.

Usage:
    from arxiv_rag_curator.services.ollama.service import OllamaService
    svc = OllamaService(url="http://localhost:11434", model="llama3.2:1b")
    answer = svc.generate(prompt)
    for token in svc.stream(prompt):
        print(token, end="", flush=True)
"""

import json
import logging
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)

# Generation defaults — tuned for factual RAG answers
DEFAULT_TEMPERATURE    = 0.1    # low = deterministic, factual
DEFAULT_NUM_PREDICT    = 512    # max tokens per response (~400 words)
DEFAULT_TOP_P          = 0.9
DEFAULT_REPEAT_PENALTY = 1.1    # discourage verbatim repetition

# HTTP timeouts
HEALTH_TIMEOUT   = 5
GENERATE_TIMEOUT = 180   # generation can take a while on CPU
PULL_TIMEOUT     = 600   # model download: ~2GB for llama3.2:1b


class OllamaService:
    """
    Client for Ollama's local LLM inference API.

    Thread-safe: uses a shared requests.Session with connection pooling.
    One instance per application is appropriate.
    """

    def __init__(
        self,
        url:   str = "http://localhost:11434",
        model: str = "llama3.2:1b",
    ):
        self._url   = url.rstrip("/")
        self._model = model
        self._session = requests.Session()

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt:      str,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int   = DEFAULT_NUM_PREDICT,
        json_mode:   bool  = False,
    ) -> str:
        """
        Non-streaming generation — waits for the complete response.

        Used by the standard /ask endpoint where the caller wants a
        complete JSON response at once.
        """
        payload = self._build_payload(prompt, stream=False,
                                       temperature=temperature,
                                       num_predict=num_predict,
                                       json_mode=json_mode)
        try:
            resp = self._session.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=GENERATE_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama generate failed: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    def stream(
        self,
        prompt:      str,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int   = DEFAULT_NUM_PREDICT,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields one token at a time.

        Ollama returns NDJSON with stream=True:
          {\"model\": \"llama3.2:1b\", \"response\": \"The\", \"done\": false}
          {\"model\": \"llama3.2:1b\", \"response\": \" transformer\", \"done\": false}
          ...
          {\"model\": \"llama3.2:1b\", \"response\": \"\", \"done\": true}

        This generator reads each line and yields the \"response\" field.
        The FastAPI route wraps these yields into SSE events.
        """
        payload = self._build_payload(prompt, stream=True,
                                       temperature=temperature,
                                       num_predict=num_predict,
                                       json_mode=False)
        try:
            resp = self._session.post(
                f"{self._url}/api/generate",
                json=payload,
                stream=True,
                timeout=GENERATE_TIMEOUT,
            )
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data  = json.loads(raw_line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed NDJSON line: %s", exc)
                    continue

        except requests.exceptions.RequestException as exc:
            logger.error("Ollama stream failed: %s", exc)
            raise RuntimeError(f"LLM streaming failed: {exc}") from exc

    # ── Model management ──────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        """Return list of downloaded model names."""
        try:
            resp = self._session.get(f"{self._url}/api/tags", timeout=HEALTH_TIMEOUT)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def ensure_model(self) -> None:
        """
        Pull the configured model if it isn't already downloaded.

        Logs progress — pulling llama3.2:1b downloads ~2GB.
        Called once on startup so the first real query doesn't stall.
        """
        available = self.list_models()
        if any(self._model in m for m in available):
            logger.info("Ollama model '%s' already available", self._model)
            return

        logger.info("Pulling Ollama model '%s' (this downloads ~2GB)...", self._model)
        try:
            resp = self._session.post(
                f"{self._url}/api/pull",
                json={"name": self._model, "stream": True},
                stream=True,
                timeout=PULL_TIMEOUT,
            )
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data   = json.loads(raw_line)
                status = data.get("status", "")
                if data.get("status") == "success":
                    logger.info("Model '%s' pulled successfully", self._model)
                    break
                elif "pulling" in status or "downloading" in status:
                    completed = data.get("completed", 0)
                    total     = data.get("total", 1) or 1
                    pct = int(completed / total * 100)
                    logger.debug("Pull progress: %d%%", pct)
        except Exception as exc:
            logger.error("Failed to pull model '%s': %s", self._model, exc)
            raise RuntimeError(f"Model pull failed: {exc}") from exc

    # ── Health check ──────────────────────────────────────────────────────────

    def check_health(self) -> dict:
        """Return health status of the Ollama service."""
        try:
            resp = self._session.get(f"{self._url}/api/tags", timeout=HEALTH_TIMEOUT)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_ready = any(self._model in m for m in models)
            return {
                "status":      "healthy",
                "model":       self._model,
                "model_ready": model_ready,
                "available_models": models,
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        prompt:      str,
        stream:      bool,
        temperature: float,
        num_predict: int,
        json_mode:   bool,
    ) -> dict:
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature":    temperature,
                "num_predict":    num_predict,
                "top_p":          DEFAULT_TOP_P,
                "repeat_penalty": DEFAULT_REPEAT_PENALTY,
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload
