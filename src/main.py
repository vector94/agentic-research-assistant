from fastapi import FastAPI

from src.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Agentic Research Assistant API",
    version="0.1.0",
    debug=settings.debug,
)


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": settings.app_version,
    }
