from src.config import Settings, get_settings
from src.services.ollama.client import OllamaClient


def make_ollama_client(
    settings: Settings | None = None,
) -> OllamaClient:
    if settings is None:
        settings = get_settings()

    return OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )
