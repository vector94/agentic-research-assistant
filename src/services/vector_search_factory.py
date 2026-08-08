from src.config import get_settings
from src.services.embeddings.factory import make_embedding_client
from src.services.opensearch.factory import make_chunk_opensearch_client
from src.services.vector_search import VectorSearchService


def make_vector_search_service() -> VectorSearchService:
    settings = get_settings()

    return VectorSearchService(
        embedding_client=make_embedding_client(settings),
        opensearch_client=make_chunk_opensearch_client(),
    )
