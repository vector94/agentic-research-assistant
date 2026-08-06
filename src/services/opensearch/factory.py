from functools import lru_cache

from src.config import get_settings
from src.services.opensearch.client import OpenSearchClient


@lru_cache
def make_opensearch_client() -> OpenSearchClient:
    settings = get_settings()
    return OpenSearchClient(host=settings.opensearch_url, index_name=settings.opensearch_index_name)


@lru_cache
def make_chunk_opensearch_client() -> OpenSearchClient:
    settings = get_settings()
    return OpenSearchClient(
        host=settings.opensearch_url,
        index_name=settings.opensearch_chunk_index_name,
    )
