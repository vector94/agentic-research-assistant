from fastapi import FastAPI

from src.config import get_settings
from src.routers.papers import router as papers_router

settings = get_settings()

app = FastAPI(
    title="Agentic Research Assistant API",
    version="0.1.0",
    debug=settings.app_debug,
)

app.include_router(papers_router)


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": settings.app_version,
    }
