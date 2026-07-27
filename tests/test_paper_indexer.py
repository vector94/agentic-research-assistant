from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

from src.services.paper_indexer import PaperIndexer


def test_index_converts_paper_to_opensearch_document() -> None:
    paper = SimpleNamespace(
        arxiv_id="1234.5678v1",
        title="Test paper",
        abstract="Test abstract",
        raw_text="Extracted PDF content",
        authors=["Ada Lovelace"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
    )

    opensearch_client = Mock()
    opensearch_client.index_paper.return_value = True

    indexer = PaperIndexer(opensearch_client)

    result = indexer.index_paper(paper)

    assert result is True
    opensearch_client.index_paper.assert_called_once_with(
        {
            "arxiv_id": "1234.5678v1",
            "title": "Test paper",
            "abstract": "Test abstract",
            "raw_text": "Extracted PDF content",
            "authors": ["Ada Lovelace"],
            "categories": ["cs.AI"],
            "published_at": "2026-01-01T00:00:00+00:00",
            "pdf_url": "https://arxiv.org/pdf/1234.5678",
        }
    )
