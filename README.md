# Agentic Research Assistant

A research assistant for finding academic papers and answering questions using
retrieved sources.

The project currently supports arXiv ingestion, PDF processing, OpenSearch
indexing, and BM25 paper search. Question answering will be added next.

## What works

- Fetch paper metadata from arXiv
- Skip papers that have already been stored
- Download and parse PDFs with Docling
- Store paper metadata and extracted content in PostgreSQL
- Index processed papers in OpenSearch
- Search papers with pagination, filters, highlighting, and optional typo tolerance
- Run the API and supporting services with Docker Compose

## Architecture

```mermaid
flowchart LR
    Client --> FastAPI
    FastAPI --> Ingestion[Ingestion service]
    Ingestion --> arXiv
    Ingestion --> PDF[PDF downloader]
    PDF --> Docling
    Docling --> Ingestion
    Ingestion --> PostgreSQL[(PostgreSQL)]
    FastAPI --> Indexing[Indexing service]
    PostgreSQL --> Indexing
    Indexing --> OpenSearch[(OpenSearch)]
    FastAPI --> Search[Search service]
    Search --> OpenSearch
```

## Stack

- Python 3.12 and FastAPI
- PostgreSQL, SQLAlchemy, and Alembic
- Docling
- OpenSearch
- Ollama
- Docker Compose
- pytest and Ruff

## Running the project

```bash
uv sync
docker compose up -d --build
uv run alembic upgrade head
```

API documentation is available at <http://localhost:8000/docs>.

To ingest one recent AI paper:

```bash
curl -X POST \
  "http://localhost:8000/api/v1/papers/ingest?query=cat%3Acs.AI&max_results=1"
```

To index processed papers in OpenSearch:

```bash
curl -X POST "http://localhost:8000/api/v1/papers/index?limit=100"
```

To search indexed papers:

```bash
curl "http://localhost:8000/api/v1/papers/search?query=language%20models&size=10&offset=0"
```

Run the checks with:

```bash
uv run ruff check .
uv run pytest
```
