from src.config import get_settings
from src.services.embeddings.factory import make_embedding_client
from src.services.hybrid_indexing import HybridIndexingService
from src.services.opensearch.factory import make_chunk_opensearch_client
from src.services.text_chunker import TextChunker


def make_hybrid_indexing_service() -> HybridIndexingService:
    settings = get_settings()

    return HybridIndexingService(
        chunker=TextChunker(),
        embedding_client=make_embedding_client(settings),
        opensearch_client=make_chunk_opensearch_client(),
        embedding_model=settings.embedding_model,
    )
