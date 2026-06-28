# Telegram Bot Interface

This project can expose the existing RAG API through a private Telegram bot.
The bot is only a delivery channel: it does not implement retrieval, ranking, generation, caching, or observability itself.
It calls the FastAPI service and returns the answer plus sources.

## Runtime Flow

```text
Telegram user
  -> Telegram bot process
  -> FastAPI /api/v1/ask-agentic/
  -> Agentic RAG workflow
  -> OpenSearch, Redis, Ollama, Langfuse
```

By default, the bot uses the agentic RAG endpoint because that path includes guardrails, document grading, query rewriting, and insufficient-evidence handling.

## Environment

Create a bot with BotFather and add the token to `.env`:

```bash
TELEGRAM_BOT_TOKEN=1234567890:your_bot_token
TELEGRAM_ALLOWED_USER_IDS=123456789
TELEGRAM_API_BASE_URL=http://localhost:8000
TELEGRAM_REQUEST_TIMEOUT=60
TELEGRAM_USE_AGENTIC_RAG=true
```

`TELEGRAM_ALLOWED_USER_IDS` is optional but recommended. If it is empty, any Telegram user who can reach the bot can use it.

To find your numeric Telegram user ID, message `@userinfobot` or similar Telegram ID helper bot.

## Run Locally

Start the backend first:

```bash
docker compose up -d postgres opensearch redis ollama api
```

Check the API:

```bash
curl http://localhost:8000/health
```

Run the bot from your local virtual environment:

```bash
.venv/bin/python -m src.bot
```

Then open Telegram and test:

```text
/start
/health
What is the transformer architecture?
```

## Run With Docker Compose

The bot can also run as a separate Docker Compose service:

```bash
docker compose up -d telegram-bot
```

The Compose service sets:

```bash
TELEGRAM_API_BASE_URL=http://api:8000
```

That is the correct URL inside the Docker network. Do not use `localhost` from inside the bot container, because that would point to the bot container itself, not the API container.

View logs:

```bash
docker compose logs -f telegram-bot
```

Stop the bot:

```bash
docker compose stop telegram-bot
```

## Supported Commands

- `/start`: show a short intro.
- `/help`: show usage examples.
- `/health`: call the FastAPI health endpoint.
- Any normal text message: treated as a RAG question.

## Notes

- Long answers are split before Telegram's message limit.
- Answers are sent as Telegram HTML, so output is escaped before sending.
- Sources are deduplicated before rendering.
- The bot uses polling for local-first development. Webhooks are better for hosted deployments but require public HTTPS.
