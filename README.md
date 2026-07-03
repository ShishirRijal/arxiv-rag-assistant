# arXiv RAG Assistant

**A local-first, production-style Retrieval-Augmented Generation system for searching and asking questions over arXiv papers** — hybrid search, an agentic reasoning workflow, response caching, full observability, and a Telegram bot, all running on a local Ollama model.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic%20workflow-1C3C3C)
![OpenSearch](https://img.shields.io/badge/OpenSearch-hybrid%20search-005EB8?logo=opensearch&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-cache-DC382D?logo=redis&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-daily%20ingestion-017CEE?logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## What it does

Ask a research question in plain English → get a grounded answer with cited arXiv sources, generated entirely from a locally-hosted LLM.

```
"What are the main approaches to reducing hallucination in RAG systems?"
        │
        ▼
┌───────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Hybrid Search    │ →   │  Evidence Gate    │ →   │  Ollama (local)  │
│  BM25 + kNN + RRF │     │  reject off-topic │     │  cited answer    │
└───────────────────┘     └───────────────────┘     └──────────────────┘
```

New papers from configured categories (`cs.AI`, `cs.LG`, `cs.CL`) are ingested daily via an Airflow pipeline, parsed, chunked, and indexed automatically — the assistant stays current without manual intervention.

![Demo](assets/demo.gif)

---

## Highlights

- 🔎 **Hybrid retrieval** — BM25 keyword search fused with dense vector search (Jina embeddings v3, 1024-dim) via Reciprocal Rank Fusion in OpenSearch, with automatic fallback to BM25-only if the embedding API is unavailable.
- 🧠 **Agentic RAG workflow** (LangGraph, 7 nodes) — guardrails out-of-scope questions before they hit the LLM, grades retrieved documents for relevance, rewrites the query and retries when evidence is weak, and terminates gracefully rather than hallucinating.
- 🛡️ **Evidence-sufficiency gate** — even the standard (non-agentic) pipeline checks lexical overlap between the query and retrieved chunks before calling the LLM, avoiding wasted generation on clearly out-of-domain questions.
- ⚡ **Response caching** — Redis-backed, keyed on the full retrieval configuration (question + filters + model), with silent degradation if Redis is unavailable.
- 📊 **Full observability** — every request is traced end-to-end in Langfuse (cache lookup → retrieval → context build → generation) as opt-in, zero-config-required instrumentation.
- 🤖 **Telegram bot** — the same RAG pipeline, accessible from chat, with an allowlist and HTML-formatted, deduplicated source citations.
- 🔄 **Automated ingestion** — a 5-task Airflow DAG fetches, parses (via Docling), retries failures, and syncs new papers daily on a schedule.
- 🧩 **Section-aware chunking** — chunks carry paper title + abstract as context headers, so retrieved fragments stay interpretable in isolation — a common failure point in naive RAG chunking.

---

## Architecture

```
                              ┌──────────────┐
                    ┌────────▶│   Airflow    │  daily ingestion (arXiv → Postgres → OpenSearch)
                    │         └──────────────┘
                    │
   ┌────────────┐   │   ┌───────────────┐    ┌──────────────┐
   │  FastAPI   │ ──┼──▶│  OpenSearch   │───▶│  Jina Embed. │  hybrid search (BM25 + kNN + RRF)
   │ (4 routers)│   │   │  (BM25+kNN)   │    └──────────────┘
   └────────────┘   │   └───────────────┘
        │           │
        │           │   ┌──────────────┐
        │           └──▶│  PostgreSQL  │  paper metadata + parsed text
        │               └──────────────┘
        ▼
   ┌────────────┐    ┌──────────────┐    ┌──────────────┐
   │ LangGraph  │───▶│    Ollama    │    │    Redis     │  response cache
   │ agentic RAG│    │ (local LLM)  │    │   Langfuse   │  tracing
   └────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
   Telegram Bot · Browser Chat UI
```

**Supporting services:**

| Service | Role |
|---|---|
| PostgreSQL | Paper metadata and ingestion state |
| OpenSearch | Keyword, vector, and hybrid retrieval |
| Ollama | Local LLM generation |
| Redis | Response cache |
| Langfuse | Optional tracing and observability |
| Airflow | Scheduled ingestion DAG |

---

## Tech stack

Python 3.11 · FastAPI · PostgreSQL · OpenSearch · Redis · Apache Airflow · Ollama · Docling · Jina Embeddings · LangGraph · Langfuse · python-telegram-bot · Docker Compose

---

## Engineering decisions worth noting

- **Graceful degradation everywhere** — every optional dependency (Jina, Redis, Langfuse) fails safe: the core search-and-answer path keeps working with only PostgreSQL, OpenSearch, and Ollama running.
- **Idempotent ingestion** — re-running the ingestion pipeline never overwrites successfully parsed data with a failed re-parse.
- **Retry-bounded agentic loop** — the query-rewrite loop has a hard retry limit with an explicit terminal "insufficient evidence" state, preventing infinite loops on ambiguous queries.
- **Structured output from a small local model** — guardrail scoring, document grading, and query rewriting all coerce structured JSON from a 1B-parameter local model via Ollama's JSON mode, with conservative fallbacks if parsing fails.

<p float="left">
  <img src="assets/chat-ui.png" width="49%" alt="Browser chat UI" />
  <img src="assets/agentic-reasoning.png" width="49%" alt="Agentic reasoning steps shown in response" />
</p>

---

## Project structure

```
.
├── airflow/
│   └── dags/
│       └── arxiv_ingestion.py
├── docs/
│   └── telegram_bot.md
├── notebooks/                  # week-by-week build notebooks
│   ├── infra-setup.ipynb
│   ├── data_pipeline.ipynb
│   ├── keyword_search.ipynb
│   ├── hybrid_search.ipynb
│   ├── complete_rag.ipynb
│   ├── observability_caching.ipynb
│   └── agentic_rag.ipynb
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── static/              # browser chat UI
│   │   └── routers/
│   ├── bot/                     # Telegram bot
│   ├── core/                    # config, DB, search client init
│   └── services/
│       ├── agentic_rag/         # LangGraph workflow
│       ├── arxiv/                # arXiv API client
│       ├── cache/
│       ├── embeddings/
│       ├── indexing/             # chunking
│       ├── observability/        # Langfuse
│       ├── ollama/
│       ├── opensearch/
│       ├── pdf_parser/           # Docling
│       └── rag/                  # standard pipeline
├── tests/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Getting started

### Prerequisites
- Docker Desktop
- Python 3.11
- An Ollama-compatible machine with enough memory for the selected model
- Optional: Jina API key (semantic embeddings), Langfuse account (tracing), Telegram bot token (bot interface)

### Environment setup

```bash
cp .env.example .env
```

Key values:

```env
OLLAMA_MODEL=llama3.2:1b

JINA_API_KEY=                     # optional — falls back to BM25-only if empty

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL_HOURS=24
REDIS_ENABLED=true

LANGFUSE_PUBLIC_KEY=              # optional — tracing disabled if empty
LANGFUSE_SECRET_KEY=
LANGFUSE_ENABLED=true

AGENTIC_GUARDRAIL_THRESHOLD=60
AGENTIC_MAX_RETRIEVAL_ATTEMPTS=2
AGENTIC_TOP_K=5

TELEGRAM_BOT_TOKEN=                # optional — bot disabled if empty
TELEGRAM_ALLOWED_USER_IDS=
TELEGRAM_USE_AGENTIC_RAG=true
```

### Run with Docker Compose

```bash
# Core services + API only
docker compose up -d postgres opensearch redis ollama api

# Check health
curl http://localhost:8000/health

# Everything, including Airflow and the Telegram bot
docker compose up -d

# Follow logs
docker compose logs -f api
```

### Pull the Ollama model

```bash
docker exec -it rag-ollama ollama pull llama3.2:1b
docker exec -it rag-ollama ollama list
```

---

## Interfaces

| Interface | URL / Command |
|---|---|
| API docs (Swagger) | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/health` |
| Browser chat UI | `http://localhost:8000/chat` |
| Telegram bot | `python -m src.bot` |
| Airflow UI | `http://localhost:8080` (default login: `admin` / `admin`) |
| OpenSearch | `http://localhost:9200` |

The chat UI supports standard and agentic RAG modes, source cards with arXiv links, retrieval metadata, agentic reasoning steps, live service health, category filtering, and a hybrid-search toggle.

---

## API examples

```bash
# BM25 keyword search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query":"transformer architecture"}'

# Hybrid search (BM25 + kNN)
curl -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query":"transformer architecture","use_hybrid":true}'

# Standard RAG answer
curl -X POST http://localhost:8000/api/v1/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?","use_hybrid":true}'

# Agentic RAG answer
curl -X POST http://localhost:8000/api/v1/ask-agentic/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?"}'
```

The agentic endpoint runs a graph-controlled workflow: guardrail → retrieval → document grading → (optional query rewrite + retry) → grounded answer generation, or an explicit insufficient-evidence response.

---

## Ingestion

The Airflow DAG at `airflow/dags/arxiv_ingestion.py`:
1. Verifies required services are healthy
2. Fetches arXiv papers for configured categories
3. Downloads and parses PDFs with Docling
4. Stores metadata in PostgreSQL
5. Indexes documents into OpenSearch, retrying failed PDF parses
6. Generates a run report

Monitor at `http://localhost:8080`.

---

## Telegram bot

```env
TELEGRAM_BOT_TOKEN=your_botfather_token
TELEGRAM_ALLOWED_USER_IDS=your_numeric_telegram_user_id   # recommended
```

```bash
.venv/bin/python -m src.bot          # local
docker compose up -d telegram-bot     # or via Docker
```

Supported commands: `/start`, `/help`, `/health`, or any plain-text research question. See `docs/telegram_bot.md` for details.

---

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Covers: agentic RAG service behavior, out-of-scope handling, document grading and retry flow, Telegram formatting, RAG client endpoint selection, and API error handling.

---

## Development

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
.venv/bin/python -m src.bot
```

---

## Troubleshooting

**API is healthy but answers are slow** — local LLM generation can take several seconds, especially with agentic retrieval. Repeated questions are faster once Redis caching kicks in.

**Hybrid search returns BM25 only** — check whether `JINA_API_KEY` is set and whether the chunk index actually contains embeddings.

**Telegram bot doesn't reply** — check `docker compose logs -f telegram-bot` and verify the bot token is valid and the container can reach Telegram's API.

---

## Status & limitations

This is an actively-developed learning/portfolio project, not a production deployment:
- Dev-mode configuration (open CORS, disabled OpenSearch security) — not hardened for production
- No CI/CD pipeline; tests run manually
- No benchmarked retrieval/answer-quality metrics yet
  
---

## Acknowledgements

Built while working through the [Mother of AI Project](https://github.com/jamwithai/production-agentic-rag-course) curriculum on production agentic RAG systems, adapted and extended for the arXiv domain.
