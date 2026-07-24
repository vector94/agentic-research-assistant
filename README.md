# Agentic Research Assistant

An AI system for discovering research papers and answering questions using
retrieved academic sources.

The system will progressively support research-paper ingestion, hybrid
retrieval, grounded question answering, observability, and agentic workflows.

## Current progress

- [x] Python 3.14 project managed with `uv`
- [x] FastAPI application and health endpoint
- [x] Automated testing with pytest
- [x] Linting and formatting with Ruff
- [x] Typed application configuration
- [x] PostgreSQL 18 with Docker Compose
- [x] Async SQLAlchemy connection
- [x] Initial `Paper` model
- [x] Database migrations with Alembic
- [ ] arXiv paper ingestion
- [ ] PDF parsing
- [ ] OpenSearch retrieval
- [ ] Embeddings and hybrid search
- [ ] RAG question answering
- [ ] Monitoring and caching
- [ ] Agentic research workflows

## Technology stack

- Python 3.14
- FastAPI
- PostgreSQL 18
- SQLAlchemy and Psycopg 3
- Alembic
- Docker Compose
- pytest
- Ruff
- uv
