import asyncio
from unittest.mock import Mock

from src.services.search import PaperSearchService


def test_search_converts_opensearch_response() -> None:
    opensearch_client = Mock()
    opensearch_client.search_papers.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {
                    "_score": 2.5,
                    "highlight": {
                        "title": ["<mark>Test</mark> paper"],
                        "raw_text": ["A matching <mark>test</mark> fragment"],
                    },
                    "_source": {
                        "arxiv_id": "1234.5678v1",
                        "title": "Test paper",
                        "abstract": "Test abstract",
                        "authors": ["Ada Lovelace"],
                        "categories": ["cs.AI"],
                        "published_at": "2026-01-01T00:00:00+00:00",
                        "pdf_url": "https://arxiv.org/pdf/1234.5678",
                    },
                }
            ],
        }
    }

    service = PaperSearchService(opensearch_client)

    result = asyncio.run(
        service.search(
            query="test",
            size=5,
            from_=10,
            categories=["cs.AI"],
            latest=True,
            fuzzy=True,
        )
    )

    assert result.total == 1
    assert len(result.results) == 1
    assert result.results[0].arxiv_id == "1234.5678v1"
    assert result.results[0].score == 2.5
    assert result.results[0].highlights == {
        "title": ["<mark>Test</mark> paper"],
        "raw_text": ["A matching <mark>test</mark> fragment"],
    }

    opensearch_client.search_papers.assert_called_once_with(
        query="test",
        size=5,
        from_=10,
        categories=["cs.AI"],
        latest=True,
        fuzzy=True,
    )
