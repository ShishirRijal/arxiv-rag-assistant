# arXiv RAG Curator

A local-first research assistant for collecting, indexing, searching, and asking questions over arXiv papers.

The project combines arXiv ingestion, PDF parsing, keyword search, semantic search, hybrid retrieval, local LLM answering, response caching, tracing, a graph-based RAG workflow, a Gradio UI, and a Telegram bot interface.

## What It Does

- Fetches arXiv papers and metadata for selected research categories.
- Downloads and parses PDFs into searchable text chunks.
- Stores paper metadata in PostgreSQL.
- Indexes papers and chunks in OpenSearch.
- Supports BM25 keyword search, vector search, and hybrid search.
- Generates grounded answers with a local Ollama model.
- Provides citations back to the source arXiv papers.
- Caches repeated answers with Redis.
- Sends traces and spans to Langfuse when configured.
- Exposes the system through FastAPI, Gradio, and Telegram.

## Current Interfaces

- FastAPI docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Gradio UI: `gradio_launcher.py`
- Telegram bot: `python -m src.bot`
- Airflow UI: `http://localhost:8080`
- OpenSearch: `http://localhost:9200`

## Architecture

```text
arXiv API
  -> ingestion pipeline
  -> PDF download + Docling parsing
  -> PostgreSQL metadata store
  -> OpenSearch paper + chunk indexes
  -> BM25 / semantic / hybrid retrieval
  -> RAG answer generation with Ollama
  -> FastAPI / Gradio / Telegram
```

Supporting services:

```text
PostgreSQL   paper metadata and ingestion state
OpenSearch   keyword, vector, and hybrid retrieval
Ollama       local LLM generation
Redis        response cache
Langfuse     optional tracing and observability
Airflow      scheduled ingestion DAG
```

## Tech Stack

- Python 3.11
- FastAPI
- PostgreSQL
- OpenSearch
- Redis
- Apache Airflow
- Ollama
- Docling
- Jina embeddings
- LangGraph
- Langfuse
- Gradio
- python-telegram-bot
- Docker Compose

## Project Structure

```text
.
├── airflow/
│   └── dags/
│       └── arxiv_ingestion.py
├── docs/
│   └── telegram_bot.md
├── notebooks/
│   ├── infra-setup.ipynb
│   ├── data_pipeline.ipynb
│   ├── keyword-searchi.ipynb
│   ├── hybrid_search.ipynb
│   ├── complete_rag.ipynb
│   ├── observability_caching.ipynb
│   └── agentic_rag.ipynb
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── routers/
│   ├── bot/
│   ├── core/
│   └── services/
│       ├── agentic_rag/
│       ├── arxiv/
│       ├── cache/
│       ├── embeddings/
│       ├── indexing/
│       ├── observability/
│       ├── ollama/
│       ├── opensearch/
│       ├── pdf_parser/
│       └── rag/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── gradio_launcher.py
└── requirements.txt
```

## Prerequisites

- Docker Desktop
- Python 3.11
- An Ollama-compatible machine with enough memory for the selected model
- Optional: Jina API key for semantic embeddings
- Optional: Langfuse account for tracing
- Optional: Telegram bot token from BotFather

## Environment Setup

Create a local `.env` file:

```bash
cp .env.example .env
```

Important values:

```bash
OLLAMA_MODEL=llama3.2:1b

JINA_API_KEY=

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL_HOURS=24
REDIS_ENABLED=true

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_ENABLED=true

AGENTIC_GUARDRAIL_THRESHOLD=60
AGENTIC_MAX_RETRIEVAL_ATTEMPTS=2
AGENTIC_TOP_K=5
AGENTIC_TEMPERATURE=0.0

TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USER_IDS=
TELEGRAM_API_BASE_URL=http://localhost:8000
TELEGRAM_REQUEST_TIMEOUT=60
TELEGRAM_USE_AGENTIC_RAG=true
```

If `JINA_API_KEY` is empty, hybrid search falls back to BM25 where needed.

## Run With Docker Compose

Start the core services and API:

```bash
docker compose up -d postgres opensearch redis ollama api
```

Check health:

```bash
curl http://localhost:8000/health
```

Start everything, including Airflow and Telegram bot:

```bash
docker compose up -d
```

Follow logs:

```bash
docker compose logs -f api
docker compose logs -f telegram-bot
```

Stop services:

```bash
docker compose down
```

## Pull the Ollama Model

The API attempts to ensure the configured model is available. You can also pull it manually:

```bash
docker exec -it rag-ollama ollama pull llama3.2:1b
```

List local Ollama models:

```bash
docker exec -it rag-ollama ollama list
```

## API Endpoints

### Health

```bash
curl http://localhost:8000/health
```

Returns service status for PostgreSQL, OpenSearch, Ollama, Redis, Langfuse, and embeddings.

### BM25 Search

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query":"transformer architecture"}'
```

### Hybrid Search

```bash
curl -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query":"transformer architecture","use_hybrid":true}'
```

### Standard RAG Answer

```bash
curl -X POST http://localhost:8000/api/v1/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?","use_hybrid":true}'
```

### Agentic RAG Answer

```bash
curl -X POST http://localhost:8000/api/v1/ask-agentic/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?"}'
```

The agentic endpoint adds a graph-controlled workflow around retrieval and generation:

- question guardrail
- retrieval
- document grading
- optional query rewriting
- insufficient-evidence response
- grounded answer generation

## Ingesting Papers

The project includes an Airflow DAG at:

```text
airflow/dags/arxiv_ingestion.py
```

It is designed to:

- verify required services
- fetch arXiv papers for configured categories
- download PDFs
- parse PDFs with Docling
- store metadata in PostgreSQL
- index documents into OpenSearch
- retry failed PDF parsing
- generate a run report

Airflow UI:

```text
http://localhost:8080
```

Default login from the Compose setup:

```text
username: admin
password: admin
```

## Gradio UI

Start the API first, then run:

```bash
.venv/bin/python gradio_launcher.py
```

Open:

```text
http://localhost:7861
```

The Gradio UI supports:

- streamed answers
- source links
- category filtering
- hybrid search toggle

## Telegram Bot

The Telegram bot calls the FastAPI backend and returns answers directly in Telegram.

Required `.env` value:

```bash
TELEGRAM_BOT_TOKEN=your_botfather_token
```

Recommended:

```bash
TELEGRAM_ALLOWED_USER_IDS=your_numeric_telegram_user_id
```

Run locally:

```bash
.venv/bin/python -m src.bot
```

Run with Docker Compose:

```bash
docker compose up -d telegram-bot
docker compose logs -f telegram-bot
```

Supported commands:

- `/start`
- `/help`
- `/health`
- any normal text message as a research question

More details: [docs/telegram_bot.md](docs/telegram_bot.md)

## Caching and Tracing

Redis is used to cache repeated RAG responses. Cached responses return quickly and include cache metadata in the API response.

Langfuse tracing is enabled when these values are configured:

```bash
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_ENABLED=true
```

Tracing is optional. If Langfuse is not configured, the rest of the application still runs.

## Tests

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Current tests cover:

- agentic RAG service behavior
- out-of-scope question handling
- document grading and retry flow
- Telegram response formatting
- Telegram RAG client endpoint selection
- API error handling in the Telegram client

## Development Notes

Install dependencies locally:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Run the FastAPI app locally:

```bash
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Run the Telegram bot locally:

```bash
.venv/bin/python -m src.bot
```

Run tests before pushing:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Troubleshooting

### API is healthy but answers are slow

Local LLM generation can take several seconds, especially with longer context or agentic retrieval. Repeated questions should be faster if Redis caching is enabled.

### Hybrid search returns BM25

Check whether `JINA_API_KEY` is set and whether the chunk index contains embeddings.

### Telegram commands work but questions take time

Question answering calls the RAG endpoint and waits for retrieval plus local LLM generation. `/start`, `/help`, and `/health` are much faster because they do not invoke the LLM.

### Telegram bot cannot reply

Check:

```bash
docker compose logs -f telegram-bot
```

Also verify that the bot token is valid and that the container can reach Telegram's API.

### Docker images are large

The application image includes PDF parsing and ML-related dependencies. This is expected for the current single-image setup.

## Security Notes

- Do not commit `.env`.
- Rotate a Telegram bot token if it appears in logs or terminal output.
- Use `TELEGRAM_ALLOWED_USER_IDS` for private bot access.
- Keep external API keys in local environment variables or deployment secrets.

## License

No license has been specified yet.
