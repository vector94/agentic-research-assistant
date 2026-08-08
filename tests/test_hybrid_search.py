from datetime import UTC, datetime
from unittest.mock import Mock

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
