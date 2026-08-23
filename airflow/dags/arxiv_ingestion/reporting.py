import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.database import session_factory
from src.repositories.paper import PaperRepository

from .common import get_cached_services

logger = logging.getLogger(__name__)


async def _count_papers() -> int:
    async with session_factory() as session:
        repository = PaperRepository(session)
        return await repository.count()


def _get_opensearch_statistics() -> dict[str, Any]:
    _arxiv_client, _pdf_parser, _pdf_downloader, opensearch_client = get_cached_services()

    if not opensearch_client.health_check():
        return {
            "index_name": opensearch_client.index_name,
            "error": "OpenSearch is unavailable",
        }

    stats_response = opensearch_client.client.indices.stats(index=opensearch_client.index_name)
    count_response = opensearch_client.client.count(index=opensearch_client.index_name)
    index_stats = stats_response["indices"][opensearch_client.index_name]["total"]

    return {
        "index_name": opensearch_client.index_name,
        "document_count": count_response["count"],
        "index_size_mb": round(index_stats["store"]["size_in_bytes"] / (1024 * 1024), 2),
    }


def generate_daily_report(**context):
    """Generate a daily report of the ingestion pipeline results.

    Collects statistics from all previous tasks and generates a summary report.
    """
    logger.info("Generating daily ingestion report")

    ti = context.get("ti")
    if not ti:
        logger.warning("No task instance available, generating basic report")
        return {"status": "basic_report", "message": "No task instance for XCom data"}

    fetch_results = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_results") or {}
    hybrid_stats = ti.xcom_pull(task_ids="index_papers_hybrid", key="hybrid_index_stats") or {}

    logical_date = context.get("logical_date") or datetime.now(UTC)

    report: dict[str, Any] = {
        "execution_date": logical_date.isoformat(),
        "fetch_statistics": {
            "papers_fetched": fetch_results.get("papers_fetched", 0),
            "papers_stored": fetch_results.get("papers_stored", 0),
            "target_date": fetch_results.get("date", "unknown"),
        },
        "indexing_statistics": {
            "papers_processed": hybrid_stats.get("papers_processed", 0),
            "chunks_created": hybrid_stats.get("total_chunks_created", 0),
            "chunks_indexed": hybrid_stats.get("total_chunks_indexed", 0),
            "embeddings_generated": hybrid_stats.get("total_embeddings_generated", 0),
        },
        "pipeline_status": "success" if fetch_results and hybrid_stats else "partial",
    }

    try:
        report["database_statistics"] = {
            "total_papers": asyncio.run(_count_papers()),
        }
        report["opensearch_statistics"] = _get_opensearch_statistics()
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        report["error"] = str(e)
        report["pipeline_status"] = "partial"

    logger.info("Daily Ingestion Report:")
    logger.info(json.dumps(report, indent=2))

    ti.xcom_push(key="daily_report", value=report)

    return report
