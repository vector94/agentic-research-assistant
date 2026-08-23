import asyncio
import logging

from sqlalchemy import text

from src.config import get_settings
from src.database import session_factory
from src.services.opensearch.hybrid_index_config import make_chunk_index_mapping

from .common import get_cached_services

logger = logging.getLogger(__name__)


async def _verify_database() -> None:
    async with session_factory() as session:
        await session.execute(text("SELECT 1"))


def setup_environment(**context):
    """Verify dependencies and create missing search indexes."""
    logger.info("Setting up the arXiv paper ingestion environment")

    try:
        asyncio.run(_verify_database())
        logger.info("Database connection verified")

        _arxiv_client, _pdf_parser, _pdf_downloader, chunk_client = get_cached_services()

        if not chunk_client.health_check():
            raise RuntimeError("OpenSearch connection failed")

        settings = get_settings()
        index_created = chunk_client.create_index(
            make_chunk_index_mapping(settings.embedding_dimensions),
        )

        logger.info("OpenSearch connection verified")
        logger.info("Hybrid index created: %s", index_created)
        logger.info("arXiv client and PDF parser are ready")

        result = {
            "status": "success",
            "message": "Environment setup completed",
            "hybrid_index_created": index_created,
        }

        ti = context.get("ti")
        if ti:
            ti.xcom_push(key="setup_results", value=result)

        return result

    except Exception as e:
        error_msg = f"Environment setup failed: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
