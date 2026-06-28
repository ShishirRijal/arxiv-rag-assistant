"""Formatting helpers for Telegram messages."""

from __future__ import annotations

from html import escape
from typing import Any

TELEGRAM_MESSAGE_LIMIT = 4096
SAFE_MESSAGE_LIMIT = 3800


def format_rag_answer(result: dict[str, Any]) -> str:
    """Format a RAG API response as Telegram-safe HTML."""
    answer = str(result.get("answer") or "No answer returned.").strip()
    sources = _format_sources(result.get("sources") or [])

    parts = [escape(answer)]
    if sources:
        parts.append(f"<b>Sources</b>\n{sources}")

    metadata = _format_metadata(result)
    if metadata:
        parts.append(metadata)

    return "\n\n".join(parts)


def split_telegram_message(text: str, limit: int = SAFE_MESSAGE_LIMIT) -> list[str]:
    """Split long Telegram messages without cutting in the middle of a paragraph."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _format_sources(sources: list[dict[str, Any]]) -> str:
    seen: set[str] = set()
    lines: list[str] = []

    for source in sources:
        arxiv_id = str(source.get("arxiv_id") or "").strip()
        title = str(source.get("title") or "Untitled source").strip()
        url = str(source.get("url") or "").strip()
        key = arxiv_id or url or title

        if key in seen:
            continue
        seen.add(key)

        label = escape(title)
        if url:
            lines.append(f"- <a href=\"{escape(url)}\">{label}</a>")
        else:
            lines.append(f"- {label}")

    return "\n".join(lines)


def _format_metadata(result: dict[str, Any]) -> str:
    fields: list[str] = []

    if result.get("search_mode"):
        fields.append(f"mode={escape(str(result['search_mode']))}")
    if result.get("cached") is True:
        fields.append("cached=true")
    if result.get("retrieval_attempts") is not None:
        fields.append(f"attempts={escape(str(result['retrieval_attempts']))}")
    if result.get("took_ms") is not None:
        fields.append(f"took={escape(str(result['took_ms']))}ms")

    if not fields:
        return ""
    return f"<i>{' | '.join(fields)}</i>"
