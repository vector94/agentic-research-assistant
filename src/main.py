import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text

from src.config import get_settings
from src.database import session_factory
from src.routers.papers import router as papers_router
from src.routers.rag import router as rag_router
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_chunk_opensearch_client
from src.services.opensearch.setup import setup_opensearch

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    setup_result = setup_opensearch()

    if setup_result["available"]:
        logger.info(
            "OpenSearch indexes ready: papers_created=%s, chunks_created=%s",
            setup_result["paper_index_created"],
            setup_result["chunk_index_created"],
        )
    else:
        logger.warning("OpenSearch is unavailable; search features will be limited")

    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.app_debug,
    lifespan=lifespan,
)

app.include_router(papers_router)
app.include_router(rag_router)


async def _check_database() -> bool:
    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database health check failed")
        return False


async def _check_opensearch() -> bool:
    try:
        return await asyncio.to_thread(make_chunk_opensearch_client().health_check)
    except Exception:
        logger.exception("OpenSearch health check failed")
        return False


async def _check_ollama() -> bool:
    ollama_client = make_ollama_client()
    try:
        return await ollama_client.health_check()
    except Exception:
        logger.exception("Ollama health check failed")
        return False
    finally:
        await ollama_client.close()


@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    database_healthy, opensearch_healthy, ollama_healthy = await asyncio.gather(
        _check_database(),
        _check_opensearch(),
        _check_ollama(),
    )

    services = {
        "database": "healthy" if database_healthy else "unhealthy",
        "opensearch": "healthy" if opensearch_healthy else "unhealthy",
        "ollama": "healthy" if ollama_healthy else "unhealthy",
    }

    return {
        "status": "healthy" if all(v == "healthy" for v in services.values()) else "degraded",
        "environment": settings.environment,
        "version": settings.app_version,
        "services": services,
    }
