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


def test_ingest_bounds_pdf_work_and_serializes_database_updates() -> None:
    papers = [
        ArxivPaper(
            arxiv_id=f"1234.000{index}v1",
            title=f"Paper {index}",
            abstract="Abstract",
            authors=["Author"],
            categories=["cs.AI"],
            published_at=datetime(2026, 1, index, tzinfo=UTC),
            pdf_url=f"https://arxiv.org/pdf/1234.000{index}",
        )
        for index in range(1, 4)
    ]
    database_papers = [SimpleNamespace(pdf_processed=False) for _paper in papers]
    pdf_content = PdfContent(
        raw_text="Parsed paper content",
        parser_used=ParserType.DOCLING,
    )

    active_downloads = 0
    peak_downloads = 0
    active_parses = 0
    peak_parses = 0
    active_updates = 0
    peak_updates = 0

    async def download_pdf(arxiv_id: str, _pdf_url: str) -> Path:
        nonlocal active_downloads, peak_downloads
        active_downloads += 1
        peak_downloads = max(peak_downloads, active_downloads)
        await asyncio.sleep(0)
        active_downloads -= 1
        return Path(f"/tmp/{arxiv_id}.pdf")

    async def parse_pdf(_pdf_path: Path) -> PdfContent:
        nonlocal active_parses, peak_parses
        active_parses += 1
        peak_parses = max(peak_parses, active_parses)
        await asyncio.sleep(0)
        active_parses -= 1
        return pdf_content

    async def update_pdf_content(_database_paper: object, _content: PdfContent) -> None:
        nonlocal active_updates, peak_updates
        active_updates += 1
        peak_updates = max(peak_updates, active_updates)
        await asyncio.sleep(0)
        active_updates -= 1

    arxiv_client = Mock()
    arxiv_client.search.return_value = papers

    paper_repository = Mock()
    paper_repository.get_by_arxiv_id = AsyncMock(return_value=None)
    paper_repository.add = AsyncMock(side_effect=database_papers)
    paper_repository.update_pdf_content = AsyncMock(side_effect=update_pdf_content)

    pdf_downloader = Mock()
    pdf_downloader.download_pdf = AsyncMock(side_effect=download_pdf)

    pdf_parser = Mock()
    pdf_parser.parse = AsyncMock(side_effect=parse_pdf)

    service = PaperIngestionService(
        arxiv_client=arxiv_client,
        paper_repository=paper_repository,
        pdf_downloader=pdf_downloader,
        pdf_parser=pdf_parser,
        max_concurrent_downloads=2,
        max_concurrent_parsing=1,
    )

    result = asyncio.run(service.ingest("cat:cs.AI", max_results=3))

    assert result == {
        "fetched": 3,
        "stored": 3,
        "skipped": 0,
        "processed": 3,
        "failed": 0,
    }
    assert peak_downloads == 2
    assert peak_parses == 1
    assert peak_updates == 1
