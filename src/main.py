import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_settings
from src.routers.papers import router as papers_router
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
    title="Agentic Research Assistant API",
    version="0.1.0",
    debug=settings.app_debug,
    lifespan=lifespan,
)

app.include_router(papers_router)


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": settings.app_version,
    }
