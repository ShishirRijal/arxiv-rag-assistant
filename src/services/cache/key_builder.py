"""
Stable cache-key helpers for RAG responses.

The key must include every request parameter that can change the answer.
Changing prompt or retrieval behavior should bump the version constants.
"""

import hashlib
import json
from datetime import date
from typing import Optional

PROMPT_VERSION = "rag-v1"
RETRIEVAL_VERSION = "hybrid-v1"


def build_ask_cache_key(
    *,
    question: str,
    use_hybrid: bool,
    categories: Optional[list[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    model: str = "",
    max_chunks: int = 5,
) -> str:
    """Build a deterministic cache key for the non-streaming ask endpoint."""
    payload = {
        "question": question.strip(),
        "use_hybrid": bool(use_hybrid),
        "categories": sorted(categories or []),
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
        "model": model,
        "max_chunks": max_chunks,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"rag:ask:{digest}"
