from src.config import get_settings
from src.services.embeddings.factory import make_embedding_client
from src.services.hybrid_search import HybridSearchService
from src.services.opensearch.factory import make_chunk_opensearch_client


def make_hybrid_search_service() -> HybridSearchService:
    settings = get_settings()

    return HybridSearchService(
        embedding_client=make_embedding_client(settings),
        opensearch_client=make_chunk_opensearch_client(),
    )
