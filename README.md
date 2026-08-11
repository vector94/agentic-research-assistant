# Agentic Research Assistant

A research assistant for finding academic papers and answering questions using
retrieved sources.

The project supports arXiv ingestion, PDF processing, hybrid retrieval, and
question answering with a local language model.

## What works

- Fetch paper metadata from arXiv
- Skip papers that have already been stored
- Download and parse PDFs with Docling
- Store paper metadata and extracted content in PostgreSQL
- Index processed papers in OpenSearch
- Search papers with pagination, filters, highlighting, and optional typo tolerance
- Split papers into overlapping, section-aware chunks
- Generate passage and query embeddings with Jina AI
- Search chunks with vector similarity or BM25
- Combine keyword and semantic rankings with Reciprocal Rank Fusion
- Generate grounded answers with Ollama and retrieved paper chunks
- Stream answers through the API and Gradio interface
- Run the API and supporting services with Docker Compose

## Architecture

```mermaid
flowchart LR
    Client --> FastAPI
    Gradio --> FastAPI
    FastAPI --> Ingestion[Ingestion service]
    Ingestion --> arXiv
    Ingestion --> PDF[PDF downloader]
    PDF --> Docling
    Docling --> Ingestion
    Ingestion --> PostgreSQL[(PostgreSQL)]
    FastAPI --> Indexing[Indexing service]
    PostgreSQL --> Indexing
    Indexing --> Chunking[Text chunker]
    Chunking --> Embeddings[Jina embeddings]
    Embeddings --> OpenSearch[(OpenSearch)]
    FastAPI --> Search[Search service]
    Search --> OpenSearch
    Search --> Embeddings
    FastAPI --> RAG[RAG service]
    RAG --> Search
    RAG --> Ollama
```

## Stack

- Python 3.12 and FastAPI
- PostgreSQL, SQLAlchemy, and Alembic
- Docling
- OpenSearch
- Jina AI embeddings
- Ollama
- Gradio
- Docker Compose
- pytest and Ruff

## Running the project

```bash
uv sync
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:1b
uv run alembic upgrade head
```

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).
The Gradio interface is available at [http://localhost:7861](http://localhost:7861).

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

To create and index searchable paper chunks:

```bash
curl -X POST "http://localhost:8000/api/v1/papers/index/chunks?limit=100"
```

To run hybrid chunk search:

```bash
curl "http://localhost:8000/api/v1/papers/search/hybrid?query=3D%20spatial%20reasoning&size=5"
```

To ask a question using RAG:

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"How do vision-language models understand 3D geometry?","top_k":3}'
```

To stream an answer:

```bash
curl --no-buffer -X POST "http://localhost:8000/api/v1/stream" \
  -H "Content-Type: application/json" \
  -d '{"query":"How do vision-language models understand 3D geometry?","top_k":3}'
```

Run the checks with:

```bash
uv run ruff check .
uv run pytest
```
