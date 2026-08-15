# Agentic Research Assistant

A research assistant that imports arXiv papers and answers questions using
retrieved passages from those papers.

## What it does

- Downloads paper metadata and PDFs from arXiv
- Extracts PDF content with Docling and stores it in PostgreSQL
- Splits papers into searchable chunks and creates embeddings with Jina AI
- Supports BM25, vector, and hybrid search with OpenSearch
- Generates grounded answers with Ollama and streams them through the API or Gradio
- Caches repeated questions in Redis
- Traces the RAG pipeline with Langfuse

## Architecture

```mermaid
flowchart LR
    User --> API[FastAPI]
    User --> Gradio
    Gradio --> API

    API --> Ingestion
    Ingestion --> arXiv
    Ingestion --> Docling
    Ingestion --> PostgreSQL[(PostgreSQL)]

    PostgreSQL --> Indexing
    Indexing --> Jina[Jina embeddings]
    Indexing --> OpenSearch[(OpenSearch)]

    API --> RAG
    RAG --> Redis[(Redis cache)]
    RAG --> Jina
    RAG --> OpenSearch
    RAG --> Ollama
    RAG -. traces .-> Langfuse
```

## Stack

- Python 3.12 and FastAPI
- PostgreSQL, SQLAlchemy, and Alembic
- Docling
- OpenSearch and Jina AI embeddings
- Ollama
- Redis and Langfuse
- Gradio and Docker Compose
- pytest and Ruff

## Setup

Add the database connection and Jina API key to `.env`:

```env
DATABASE_URL=postgresql+psycopg://research_user:research_password@localhost:5433/research_assistant
JINA_API_KEY=your_jina_api_key
```

Add the Langfuse credentials to the same file:

```env
LANGFUSE__PUBLIC_KEY=your_public_key
LANGFUSE__SECRET_KEY=your_secret_key
LANGFUSE__HOST=https://cloud.langfuse.com
LANGFUSE__ENABLED=true
```

Start the project:

```bash
uv sync
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:1b
uv run alembic upgrade head
```

- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Gradio interface: [http://localhost:7861](http://localhost:7861)

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Check the API |
| `POST` | `/api/v1/papers/ingest` | Import papers from arXiv |
| `POST` | `/api/v1/papers/index` | Index complete papers |
| `POST` | `/api/v1/papers/index/chunks` | Create and index paper chunks |
| `GET` | `/api/v1/papers/search` | Search complete papers |
| `GET` | `/api/v1/papers/search/vector` | Run vector search over chunks |
| `GET` | `/api/v1/papers/search/hybrid` | Run hybrid search over chunks |
| `POST` | `/api/v1/ask` | Ask a RAG question |
| `POST` | `/api/v1/stream` | Stream a RAG answer |

The interactive API documentation contains all query parameters and response
schemas.

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
