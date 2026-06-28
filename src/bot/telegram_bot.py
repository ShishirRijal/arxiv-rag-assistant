"""Telegram polling bot for the arXiv RAG assistant."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..core.config import settings
from .formatters import format_rag_answer, split_telegram_message
from .rag_client import RAGClient, RAGClientConfig, RAGClientError

logger = logging.getLogger(__name__)


def build_application() -> Application:
    """Build the Telegram application and register handlers."""
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required to start the Telegram bot.")

    rag_client = RAGClient(
        RAGClientConfig(
            base_url=settings.telegram_api_base_url,
            timeout_seconds=settings.telegram_request_timeout,
            use_agentic_rag=settings.telegram_use_agentic_rag,
        )
    )

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()
    application.bot_data["rag_client"] = rag_client
    application.bot_data["allowed_user_ids"] = settings.parsed_telegram_allowed_user_ids

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_question))

    return application


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Introduce the bot."""
    if not await _ensure_authorized(update, context):
        return

    await update.effective_message.reply_text(
        "Ask me questions about the arXiv papers indexed in your local RAG system.\n\n"
        "Commands:\n"
        "/help - show usage\n"
        "/health - check backend status",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show usage instructions."""
    if not await _ensure_authorized(update, context):
        return

    await update.effective_message.reply_text(
        "Send a normal message with your research question.\n\n"
        "Examples:\n"
        "- What is the transformer architecture?\n"
        "- How transparent is DiffusionGemma?\n\n"
        "The bot calls your local FastAPI RAG endpoint and returns grounded answers with sources."
    )


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the FastAPI backend health."""
    if not await _ensure_authorized(update, context):
        return

    rag_client: RAGClient = context.application.bot_data["rag_client"]

    try:
        result = await rag_client.health()
    except RAGClientError as exc:
        await update.effective_message.reply_text(f"Backend health check failed: {exc}")
        return

    services = result.get("services") or {}
    service_lines = [
        f"- {name}: {details.get('status', 'unknown')}"
        for name, details in services.items()
        if isinstance(details, dict)
    ]
    message = "\n".join([
        f"API status: {result.get('status', 'unknown')}",
        f"version: {result.get('version', 'unknown')}",
        "",
        *service_lines,
    ]).strip()
    await update.effective_message.reply_text(message)


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a user question through the RAG API."""
    if not await _ensure_authorized(update, context):
        return

    question = (update.effective_message.text or "").strip()
    if not question:
        await update.effective_message.reply_text("Send a question as text.")
        return

    rag_client: RAGClient = context.application.bot_data["rag_client"]

    await update.effective_chat.send_action(ChatAction.TYPING)

    try:
        result = await rag_client.ask(question)
    except RAGClientError as exc:
        logger.warning("RAG request failed: %s", exc)
        await update.effective_message.reply_text(
            "I could not get an answer from the RAG backend.\n\n"
            f"Reason: {exc}"
        )
        return
    except Exception as exc:
        logger.exception("Unexpected Telegram bot failure")
        await update.effective_message.reply_text(
            "Unexpected bot error while processing this question."
        )
        return

    formatted = format_rag_answer(result)
    for chunk in split_telegram_message(formatted):
        await update.effective_message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
        )


async def _ensure_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    allowed_user_ids: set[int] = context.application.bot_data.get("allowed_user_ids", set())
    user = update.effective_user

    if not allowed_user_ids or user is None or user.id in allowed_user_ids:
        return True

    logger.warning("Rejected Telegram user id=%s username=%s", user.id, user.username)
    await update.effective_message.reply_text("This bot is private.")
    return False


def run_polling() -> None:
    """Start the Telegram bot with long polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    application = build_application()
    logger.info("Starting Telegram bot polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_polling()
