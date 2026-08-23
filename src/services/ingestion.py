import asyncio
import logging

from src.models import Paper
from src.repositories.paper import PaperRepository
from src.schemas.paper import ArxivPaper
from src.schemas.pdf import PdfContent
from src.services.arxiv import ArxivClient
from src.services.pdf_downloader import PdfDownloader
from src.services.pdf_parser.parser import PdfParserService

logger = logging.getLogger(__name__)


class PaperIngestionService:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        paper_repository: PaperRepository,
        pdf_downloader: PdfDownloader,
        pdf_parser: PdfParserService,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 1,
    ) -> None:
        if max_concurrent_downloads < 1:
            raise ValueError("max_concurrent_downloads must be at least 1")
        if max_concurrent_parsing < 1:
            raise ValueError("max_concurrent_parsing must be at least 1")

        self.arxiv_client = arxiv_client
        self.paper_repository = paper_repository
        self.pdf_downloader = pdf_downloader
        self.pdf_parser = pdf_parser
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing

    async def ingest(self, query: str, max_results: int = 5) -> dict[str, int]:
        papers = await asyncio.to_thread(self.arxiv_client.search, query=query, max_results=max_results)

        stored = 0
        skipped = 0
        processed = 0
        failed = 0
        pending_papers: list[tuple[ArxivPaper, Paper]] = []

        for paper in papers:
            database_paper = await self.paper_repository.get_by_arxiv_id(paper.arxiv_id)
            if database_paper is None:
                database_paper = await self.paper_repository.add(paper)
                stored += 1
            else:
                skipped += 1

            if database_paper.pdf_processed:
                continue

            pending_papers.append((paper, database_paper))

        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)
        pdf_tasks = [
            self._download_and_parse(paper, download_semaphore, parse_semaphore)
            for paper, _database_paper in pending_papers
        ]
        pdf_results = await asyncio.gather(*pdf_tasks, return_exceptions=True)

        for (paper, database_paper), result in zip(pending_papers, pdf_results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                logger.error(
                    "Failed to process PDF for paper %s",
                    paper.arxiv_id,
                    exc_info=result,
                )
                failed += 1
                continue

            try:
                await self.paper_repository.update_pdf_content(database_paper, result)
                processed += 1
            except Exception:
                logger.exception(
                    "Failed to store PDF content for paper %s",
                    paper.arxiv_id,
                )
                failed += 1

        return {"fetched": len(papers), "stored": stored, "skipped": skipped, "processed": processed, "failed": failed}

    async def _download_and_parse(
        self,
        paper: ArxivPaper,
        download_semaphore: asyncio.Semaphore,
        parse_semaphore: asyncio.Semaphore,
    ) -> PdfContent:
        async with download_semaphore:
            pdf_path = await self.pdf_downloader.download_pdf(paper.arxiv_id, paper.pdf_url)

        async with parse_semaphore:
            return await self.pdf_parser.parse(pdf_path)
