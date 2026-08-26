import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from src.services.hybrid_search import HybridSearchService


def make_hit(chunk_id: str) -> dict[str, object]:
    return {
        "_source": {
            "chunk_id": chunk_id,
            "arxiv_id": "1234.5678v1",
            "paper_id": "paper-1",
            "chunk_index": 0,
            "chunk_text": f"Content for {chunk_id}",
            "chunk_word_count": 3,
            "section_title": "Introduction",
            "title": "Test paper",
            "abstract": "Test abstract",
            "authors": ["Ada Lovelace"],
            "categories": ["cs.AI"],
            "published_at": datetime(2026, 1, 1, tzinfo=UTC),
            "pdf_url": "https://arxiv.org/pdf/1234.5678",
        }
    }


def test_rrf_rewards_chunks_ranked_highly_by_both_searches() -> None:
    service = HybridSearchService(
        embedding_client=Mock(),
        opensearch_client=Mock(),
    )
    bm25_response = {
        "hits": {
            "hits": [
                make_hit("chunk-a"),
                make_hit("chunk-b"),
            ]
        }
    }
    vector_response = {
        "hits": {
            "hits": [
                make_hit("chunk-b"),
                make_hit("chunk-c"),
                make_hit("chunk-a"),
            ]
        }
    }

    results = service._fuse_results(
        bm25_response=bm25_response,
        vector_response=vector_response,
        size=2,
    )

    assert [result.chunk_id for result in results] == [
        "chunk-b",
        "chunk-a",
    ]
    assert results[0].score > results[1].score


def test_search_passes_categories_to_both_underlying_searches() -> None:
    opensearch_client = Mock()
    opensearch_client.search_chunks_bm25.return_value = {"hits": {"hits": []}}
    opensearch_client.search_chunks_vector.return_value = {"hits": {"hits": []}}

    embedding_client = Mock()
    embedding_client.embed_query = AsyncMock(return_value=[0.1, 0.2])

    service = HybridSearchService(
        embedding_client=embedding_client,
        opensearch_client=opensearch_client,
    )

    asyncio.run(service.search(query="attention mechanisms", size=5, categories=["cs.AI"]))

    opensearch_client.search_chunks_bm25.assert_called_once_with(
        "attention mechanisms",
        10,
        ["cs.AI"],
    )
    opensearch_client.search_chunks_vector.assert_called_once_with(
        [0.1, 0.2],
        10,
        ["cs.AI"],
    )
