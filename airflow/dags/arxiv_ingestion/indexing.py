import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from src.database import session_factory
from src.models import Paper
from src.repositories.paper import PaperRepository
from src.services.hybrid_indexing_factory import make_hybrid_indexing_service

logger = logging.getLogger(__name__)


def _empty_indexing_statistics() -> dict[str, int]:
    return {
        "papers_processed": 0,
        "total_chunks_created": 0,
        "total_chunks_indexed": 0,
        "total_embeddings_generated": 0,
        "total_errors": 0,
    }


async def _load_papers(fetch_results: dict[str, Any]) -> list[Paper]:
    async with session_factory() as session:
        repository = PaperRepository(session)
        stored_count = int(fetch_results.get("papers_stored", 0))

        if stored_count > 0:
            return await repository.list_recent(limit=stored_count)

        cutoff = datetime.now(UTC) - timedelta(days=1)
        return await repository.list_created_since(cutoff)


async def _index_papers_with_chunks(papers: list[Paper]) -> dict[str, int]:
    """Async helper to index papers with chunking and embeddings."""
    indexing_service = make_hybrid_indexing_service()

    try:
        statistics = await indexing_service.index_papers(
            papers,
            replace_existing=True,
        )
    finally:
        await indexing_service.close()

    return {
        "papers_processed": statistics["papers_found"],
        "total_chunks_created": statistics["chunks_created"],
        "total_chunks_indexed": statistics["chunks_indexed"],
        "total_embeddings_generated": statistics["embeddings_generated"],
        "total_errors": statistics["papers_failed"] + statistics["chunks_failed"],
    }


def index_papers_hybrid(**context):
    """Index papers with chunking and vector embeddings for hybrid search.

    This task:
    1. Fetches recently processed papers from PostgreSQL
    2. Chunks them into overlapping segments
    3. Generates embeddings using Jina AI
    4. Indexes chunks with embeddings into OpenSearch
    """
    try:
        ti = context.get("ti")

        fetch_results = None
        if ti:
            fetch_results = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        papers = asyncio.run(_load_papers(fetch_results or {}))

        if not papers:
            logger.info("No processed papers are available for hybrid indexing")
            stats = _empty_indexing_statistics()
        else:
            logger.info("Indexing %s papers for hybrid search", len(papers))

            stats = asyncio.run(_index_papers_with_chunks(papers))

            logger.info(
                "Hybrid indexing complete: %s papers, %s chunks created, %s chunks indexed",
                stats["papers_processed"],
                stats["total_chunks_created"],
                stats["total_chunks_indexed"],
            )

        if ti:
            ti.xcom_push(key="hybrid_index_stats", value=stats)

        return stats

    except Exception as e:
        logger.error(f"Failed to index papers for hybrid search: {e}")
        raise
