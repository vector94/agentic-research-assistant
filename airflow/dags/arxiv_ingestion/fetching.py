import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from src.database import session_factory
from src.repositories.paper import PaperRepository
from src.services.ingestion import PaperIngestionService

from .common import get_cached_services

logger = logging.getLogger(__name__)


async def run_paper_ingestion_pipeline(
    target_date: str,
    max_results: int,
) -> dict[str, Any]:
    """Async wrapper for the paper ingestion pipeline.

    :param target_date: Date to fetch papers for (YYYYMMDD format)
    :param max_results: Maximum number of papers to fetch
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, pdf_parser, pdf_downloader, _opensearch_client = get_cached_services()

    query = f"({os.getenv('ARXIV_QUERY', 'cat:cs.AI')}) AND submittedDate:[{target_date}0000 TO {target_date}2359]"

    async with session_factory() as session:
        repository = PaperRepository(session)
        service = PaperIngestionService(
            arxiv_client=arxiv_client,
            paper_repository=repository,
            pdf_downloader=pdf_downloader,
            pdf_parser=pdf_parser,
        )
        result = await service.ingest(query=query, max_results=max_results)

    return {
        "date": target_date,
        "papers_fetched": result["fetched"],
        "papers_stored": result["stored"],
        "papers_skipped": result["skipped"],
        "papers_processed": result["processed"],
        "papers_failed": result["failed"],
    }


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL.

    This task:
    1. Determines the target date (defaults to yesterday)
    2. Fetches papers from arXiv API
    3. Downloads and processes PDFs using Docling
    4. Stores metadata and parsed content in PostgreSQL
    """
    logger.info("Starting daily paper fetching task")

    logical_date = context.get("logical_date")
    if isinstance(logical_date, datetime):
        target_date = (logical_date - timedelta(days=1)).strftime("%Y%m%d")
    else:
        target_date = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y%m%d")

    max_results = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
    logger.info("Fetching up to %s papers for %s", max_results, target_date)

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            max_results=max_results,
        )
    )

    logger.info(
        "Daily fetch complete: %s fetched, %s stored, %s processed",
        results["papers_fetched"],
        results["papers_stored"],
        results["papers_processed"],
    )

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
