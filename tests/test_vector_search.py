import asyncio
from unittest.mock import AsyncMock, Mock

from src.services.vector_search import VectorSearchService


def test_search_passes_categories_to_opensearch() -> None:
    opensearch_client = Mock()
    opensearch_client.search_chunks_vector.return_value = {
        "hits": {"total": {"value": 0}, "hits": []},
    }

    embedding_client = Mock()
    embedding_client.embed_query = AsyncMock(return_value=[0.1, 0.2])

    service = VectorSearchService(
        embedding_client=embedding_client,
        opensearch_client=opensearch_client,
    )

    asyncio.run(service.search(query="attention mechanisms", size=5, categories=["cs.AI"]))

    opensearch_client.search_chunks_vector.assert_called_once_with(
        [0.1, 0.2],
        5,
        ["cs.AI"],
    )
