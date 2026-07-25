import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from src.schemas.paper import ArxivPaper
from src.services.ingestion import PaperIngestionService


def test_ingest_stores_new_and_skips_existing_papers() -> None:
    new_paper = ArxivPaper(
        arxiv_id="1234.0001v1",
        title="New paper",
        abstract="Abstract",
        authors=["Author"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.0001",
    )
    existing_paper = new_paper.model_copy(update={"arxiv_id": "1234.0002v1"})

    arxiv_client = Mock()
    arxiv_client.search.return_value = [new_paper, existing_paper]

    paper_repository = Mock()
    paper_repository.get_by_arxiv_id = AsyncMock(side_effect=[None, object()])
    paper_repository.add = AsyncMock()

    service = PaperIngestionService(arxiv_client, paper_repository)

    result = asyncio.run(service.ingest("cat:cs.AI", max_results=2))

    assert result == {
        "fetched": 2,
        "stored": 1,
        "skipped": 1,
    }
    paper_repository.add.assert_awaited_once_with(new_paper)
