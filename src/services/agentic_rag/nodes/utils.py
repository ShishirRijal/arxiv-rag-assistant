"""Shared helpers for agentic RAG graph nodes."""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..models import ReasoningStep

ModelT = TypeVar("ModelT", bound=BaseModel)


def add_reasoning_step(
    state: dict[str, Any],
    *,
    step: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a human-readable workflow event to state."""
    state.setdefault("reasoning_steps", []).append(
        ReasoningStep(step=step, message=message, metadata=metadata or {})
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.

    Local models sometimes wrap JSON in prose or Markdown fences. The workflow
    should still validate the extracted object before trusting it.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON is not an object")
    return parsed


def parse_structured_json(text: str, model: type[ModelT]) -> ModelT:
    """Parse and validate an LLM JSON response as a Pydantic model."""
    try:
        return model.model_validate(extract_json_object(text))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse {model.__name__}: {exc}") from exc
