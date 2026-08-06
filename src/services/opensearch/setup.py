from src.config import get_settings
from src.services.opensearch.factory import (
    make_chunk_opensearch_client,
    make_opensearch_client,
)
from src.services.opensearch.hybrid_index_config import make_chunk_index_mapping
from src.services.opensearch.index_config import PAPER_INDEX_MAPPING


def setup_opensearch() -> dict[str, bool]:
    paper_client = make_opensearch_client()

    if not paper_client.health_check():
        return {
            "available": False,
            "paper_index_created": False,
            "chunk_index_created": False,
        }

    settings = get_settings()
    chunk_client = make_chunk_opensearch_client()

    return {
        "available": True,
        "paper_index_created": paper_client.create_index(PAPER_INDEX_MAPPING),
        "chunk_index_created": chunk_client.create_index(make_chunk_index_mapping(settings.embedding_dimensions)),
    }
