from src.config import Settings, get_settings
from src.services.embeddings.client import EmbeddingClient
from src.services.embeddings.jina import JinaEmbeddingClient


def make_embedding_client(
    settings: Settings | None = None,
) -> EmbeddingClient:
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider != "jina":
        raise NotImplementedError(f"Embedding provider '{settings.embedding_provider}' is not implemented")

    return JinaEmbeddingClient(
        api_key=settings.jina_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        base_url=settings.jina_base_url,
    )
