import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from src.services.indexing import PaperIndexingService


def test_index_processed_papers_counts_successes_and_failures() -> None:
    papers = [
        SimpleNamespace(arxiv_id="1234.0001v1"),
        SimpleNamespace(arxiv_id="1234.0002v1"),
    ]

    paper_repository = Mock()
    paper_repository.list_processed = AsyncMock(return_value=papers)

    paper_indexer = Mock()
    paper_indexer.index_paper.side_effect = [
        True,
        RuntimeError("OpenSearch unavailable"),
    ]

    service = PaperIndexingService(
        paper_repository=paper_repository,
        paper_indexer=paper_indexer,
    )

    result = asyncio.run(service.index_processed_papers(limit=2))

    assert result == {
        "found": 2,
        "indexed": 1,
        "failed": 1,
    }
    paper_repository.list_processed.assert_awaited_once_with(limit=2)
