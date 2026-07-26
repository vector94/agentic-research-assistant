import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from src.schemas.paper import ArxivPaper
from src.schemas.pdf import ParserType, PdfContent
from src.services.ingestion import PaperIngestionService


def test_ingest_stores_and_processes_new_paper() -> None:
    paper = ArxivPaper(
        arxiv_id="1234.0001v1",
        title="New paper",
        abstract="Abstract",
        authors=["Author"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.0001",
    )

    database_paper = SimpleNamespace(pdf_processed=False)
    pdf_content = PdfContent(
        raw_text="Parsed paper content",
        parser_used=ParserType.DOCLING,
    )

    arxiv_client = Mock()
    arxiv_client.search.return_value = [paper]

    paper_repository = Mock()
    paper_repository.get_by_arxiv_id = AsyncMock(return_value=None)
    paper_repository.add = AsyncMock(return_value=database_paper)
    paper_repository.update_pdf_content = AsyncMock()

    pdf_downloader = Mock()
    pdf_downloader.download_pdf = AsyncMock(return_value=Path("/tmp/1234.0001v1.pdf"))

    pdf_parser = Mock()
    pdf_parser.parse = AsyncMock(return_value=pdf_content)

    service = PaperIngestionService(
        arxiv_client=arxiv_client,
        paper_repository=paper_repository,
        pdf_downloader=pdf_downloader,
        pdf_parser=pdf_parser,
    )

    result = asyncio.run(service.ingest("cat:cs.AI", max_results=1))

    assert result == {
        "fetched": 1,
        "stored": 1,
        "skipped": 0,
        "processed": 1,
        "failed": 0,
    }

    paper_repository.update_pdf_content.assert_awaited_once_with(
        database_paper,
        pdf_content,
    )
